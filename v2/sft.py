"""Supervised fine-tuning on smol-smoltalk (train.md sec 9).

Reads the pre-training checkpoint READ-ONLY and writes to its own --out
directory. A failure here cannot damage the base model.

Differences from train.py:
  - loss is masked to assistant turns only (see prepare_sft.py)
  - much lower peak LR, since we are adapting a trained model, not training one
  - resumes the same way, on ckpt_last.pt in its own directory
"""

import argparse
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from config import Config, build_model
from model import MoE, moe_load_stats, qb_step
from optim import build_optimizers
from train import Jsonl, keep_awake, lr_at, qb_reset_all, save_ckpt, set_lr


class MaskedLoader:
    """Like PackedLoader but serves (tokens, loss_mask) pairs."""

    def __init__(self, data_dir, seq_len, seed=0, pos=0, holdout=False):
        root = Path(data_dir)
        m = json.loads((root / "manifest.json").read_text())
        hold = set(m["holdout"])
        use = [s for s in m["shards"] if (s["file"] in hold) == holdout]
        if not use:
            raise ValueError(f"no shards for holdout={holdout}")
        self.paths = [str(root / s["file"]) for s in use]
        self.mpaths = [str(root / s["mask"]) for s in use]
        self.tok = [np.memmap(p, dtype=np.uint16, mode="r") for p in self.paths]
        self.msk = [np.memmap(p, dtype=np.uint8, mode="r") for p in self.mpaths]
        self.seq_len, self.seed, self.manifest = seq_len, seed, m

        counts = [max(0, (len(a) - 1) // seq_len) for a in self.tok]
        self.starts = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        self.total = int(self.starts[-1])
        if self.total == 0:
            raise ValueError("no full sequences")
        self.epoch, self.pos = -1, pos
        self._ensure_perm()

    def _ensure_perm(self):
        e = self.pos // self.total
        if e != self.epoch:
            self.epoch = e
            self.perm = np.random.default_rng(self.seed + e).permutation(self.total)

    def _locate(self, g):
        s = int(np.searchsorted(self.starts, g, side="right") - 1)
        return s, int(g - self.starts[s]) * self.seq_len

    def batch(self, n, device="cuda"):
        x = np.empty((n, self.seq_len + 1), dtype=np.int64)
        w = np.empty((n, self.seq_len + 1), dtype=np.int64)
        for j in range(n):
            self._ensure_perm()
            s, o = self._locate(self.perm[self.pos % self.total])
            x[j] = self.tok[s][o : o + self.seq_len + 1]
            w[j] = self.msk[s][o : o + self.seq_len + 1]
            self.pos += 1
        t = torch.from_numpy(x).to(device, non_blocking=True)
        mm = torch.from_numpy(w).to(device, non_blocking=True)
        # predict token i+1 from i, so the mask that matters is the target's
        return t[:, :-1].contiguous(), t[:, 1:].contiguous(), mm[:, 1:].contiguous()

    def warm(self):
        for a in self.tok:
            for i in range(0, len(a), 1 << 22):
                _ = a[i]

    def state_dict(self):
        return {"pos": self.pos, "seed": self.seed, "seq_len": self.seq_len, "paths": self.paths}

    def load_state_dict(self, sd):
        if sd["paths"] != self.paths or sd["seq_len"] != self.seq_len:
            raise ValueError("loader state does not match this dataset")
        self.seed, self.pos = sd["seed"], sd["pos"]
        self._ensure_perm()


def masked_ce(logits, y, mask, vocab, fp32):
    if fp32:
        logits = logits.float()
    ll = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none")
    m = mask.reshape(-1).to(ll.dtype)
    denom = m.sum().clamp(min=1.0)
    return (ll * m).sum() / denom


@torch.no_grad()
def evaluate(model, val, cfg, vocab, device, n):
    model.eval()
    val.pos = 0
    val._ensure_perm()
    tot = 0.0
    for _ in range(n):
        x, y, mk = val.batch(cfg.micro_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        tot += masked_ce(logits, y, mk, vocab, cfg.fp32_loss).item()
    model.train()
    return tot / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=r"D:\ml\runs\main\ckpt_last.pt")
    ap.add_argument("--data", default=r"D:\ml\data-sft")
    ap.add_argument("--out", default=r"D:\ml\runs\sft")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--total-steps", type=int, default=0, help="0 = derive from --epochs")
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--lr-muon", type=float, default=2e-3)
    ap.add_argument("--lr-adamw", type=float, default=3e-4)
    ap.add_argument("--warmup-frac", type=float, default=0.03)
    ap.add_argument("--ckpt-minutes", type=float, default=15.0)
    ap.add_argument("--require-complete", action="store_true",
                    help="refuse to start unless the base run reached its total_steps")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip torch.compile (smoke tests; avoids a cold compile per shape)")
    args = ap.parse_args()

    keep_awake()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(args.base, map_location="cpu", weights_only=False)
    base_cfg = Config(**ck["config"])
    if args.require_complete and ck["step"] < base_cfg.total_steps:
        raise SystemExit(f"base run incomplete: {ck['step']:,}/{base_cfg.total_steps:,}")
    print(f"base: step {ck['step']:,}  tokens {ck['tokens_seen']/1e9:.3f}B")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = replace(base_cfg, name="sft", data=args.data, out=args.out,
                  lr_muon=args.lr_muon, lr_adamw=args.lr_adamw,
                  warmup_frac=args.warmup_frac, ckpt_minutes=args.ckpt_minutes,
                  micro_batch=args.micro_batch or base_cfg.micro_batch,
                  compile="" if args.no_compile else base_cfg.compile)

    train = MaskedLoader(cfg.data, cfg.seq_len, cfg.seed, holdout=False)
    try:
        val = MaskedLoader(cfg.data, cfg.seq_len, cfg.seed + 1, holdout=True)
    except ValueError:
        val = None
    vocab = train.manifest["vocab_size"]
    if vocab != ck["vocab_size"]:
        raise SystemExit(f"vocab mismatch: sft data {vocab} vs base {ck['vocab_size']}")

    steps_per_epoch = train.total // (cfg.micro_batch * cfg.accum)
    total_steps = args.total_steps or max(1, int(args.epochs * steps_per_epoch))
    cfg = replace(cfg, total_steps=total_steps)
    print(f"data: {train.total:,} seqs  {steps_per_epoch:,} steps/epoch  "
          f"-> total_steps {total_steps:,}")

    model = build_model(cfg, vocab).to(device)
    model.load_state_dict(ck["model"])
    qb_reset_all(model)                     # sec 7, same reason as a resume
    opt_m, opt_a, _ = build_optimizers(model, cfg)

    net = model
    if cfg.compile == "chunk":
        import model as _m
        print("compiling KdaAttention._chunk")
        _m.KdaAttention._chunk = torch.compile(_m.KdaAttention._chunk, dynamic=False)

    step = 0
    last = out / "ckpt_last.pt"
    if last.exists():
        r = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(r["model"])
        opt_m.load_state_dict(r["muon"]); opt_a.load_state_dict(r["adamw"])
        train.load_state_dict(r["train_loader"])
        step = r["step"]
        qb_reset_all(model)
        print(f"resumed sft at step {step:,}")

    log = Jsonl(out / "log.jsonl")
    (out / "config.json").write_text(json.dumps(cfg.dict(), indent=2, default=str))
    warmup = max(1, int(cfg.warmup_frac * total_steps))
    model.train()
    train.warm()

    t_ckpt = t_win = time.time()
    tok_win = 0
    while step < total_steps:
        lm = lr_at(step, cfg.lr_muon, total_steps, warmup, cfg.min_lr_frac)
        la = lr_at(step, cfg.lr_adamw, total_steps, warmup, cfg.min_lr_frac)
        set_lr(opt_m, lm); set_lr(opt_a, la)

        loss_sum = 0.0
        for _ in range(cfg.accum):
            x, y, mk = train.batch(cfg.micro_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            loss = masked_ce(logits, y, mk, vocab, cfg.fp32_loss) / cfg.accum
            loss.backward()
            loss_sum += loss.item()

        gn = clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt_m.step(); opt_a.step()
        opt_m.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
        qb_step(model)
        step += 1
        tok_win += cfg.tokens_per_step()

        if step % cfg.log_every == 0:
            now = time.time()
            ts = tok_win / (now - t_win); t_win, tok_win = now, 0
            log.write(kind="light", step=step, loss=loss_sum, lr_muon=lm,
                      grad_norm=gn, tok_s=ts, mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            print(f"{step:>7}/{total_steps}  loss {loss_sum:6.3f}  gn {gn:5.2f}  {ts/1e3:6.1f}k tok/s")

        if cfg.heavy_every and step % cfg.heavy_every == 0:
            stats = moe_load_stats(model)
            vl = evaluate(model, val, cfg, vocab, device, cfg.eval_batches) if val else None
            log.write(kind="heavy", step=step, loss=loss_sum, val_loss=vl, moe=stats)
            print(f"        val {vl if vl is None else round(vl, 4)}")

        now = time.time()
        if now - t_ckpt > cfg.ckpt_minutes * 60:
            t_ckpt = now
            save_ckpt(last, dict(step=step, tokens_seen=step * cfg.tokens_per_step(),
                                 model=model.state_dict(), muon=opt_m.state_dict(),
                                 adamw=opt_a.state_dict(), train_loader=train.state_dict(),
                                 config=cfg.dict(), vocab_size=vocab))

    save_ckpt(last, dict(step=step, tokens_seen=step * cfg.tokens_per_step(),
                         model=model.state_dict(), muon=opt_m.state_dict(),
                         adamw=opt_a.state_dict(), train_loader=train.state_dict(),
                         config=cfg.dict(), vocab_size=vocab))
    print(f"sft done: {step:,} steps")


if __name__ == "__main__":
    main()
