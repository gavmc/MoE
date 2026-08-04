"""Mini-k3 pre-training loop (train.md §7).

Auto-resume is the default, not a flag: point --out at a run directory and it
picks up ckpt_last.pt if one is there.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from dataclasses import replace

from config import PRESETS, build_model
from loader import make_loaders
from model import AttnRes, KdaAttention, MoE, moe_load_stats, qb_step
from optim import build_optimizers, print_table


# ---------------------------------------------------------------- schedule


def lr_at(step, peak, total, warmup, min_frac):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(1.0, t)
    return peak * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


def set_lr(opt, lr):
    for g in opt.param_groups:
        g["lr"] = lr


def keep_awake():
    """Stop Windows suspending the machine mid-run.

    The first main run died 49 h in: a user-mode process called SetSuspendState at
    00:39 and the system slept, taking the supervisor with it. Both powercfg idle
    timeouts were already 0 -- they only govern *idle* sleep and cannot block an
    explicit request. ES_AWAYMODE_REQUIRED turns such a request into away-mode,
    where the display sleeps but the CPU keeps running.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        CONT, SYSTEM, AWAY = 0x80000000, 0x00000001, 0x00000040
        f = ctypes.windll.kernel32.SetThreadExecutionState
        r = f(CONT | SYSTEM | AWAY)
        if r == 0:                       # away-mode not supported here
            r = f(CONT | SYSTEM)
        print(f"keep-awake: {'on' if r else 'FAILED - machine may sleep'}")
    except Exception as e:               # never let this stop a run
        print(f"keep-awake unavailable: {e}")


def ce_loss(logits, y, vocab, fp32):
    """Cross-entropy over the vocab.

    fp32=True follows train.md §3, but `logits.float()` is a real allocation: at
    micro_batch 8 / vocab 32000 / seq 2048 it is 1.95 GiB, plus its gradient, and it
    is the exact allocation that OOMs micro_batch 8. With fp32=False the log-softmax
    reduction still accumulates in fp32 (acc_type); only the saved tensor is bf16.
    """
    if fp32:
        logits = logits.float()
    return F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))


# ---------------------------------------------------------------- probes


class Probes:
    """AttnRes entropy and KDA alpha range (§6). Off unless explicitly enabled."""

    def __init__(self, model):
        self.on = False
        self.ent, self.a_lo, self.a_hi = [], [], []
        for m in model.modules():
            if isinstance(m, AttnRes):
                m.register_forward_hook(self._res_hook)
            elif isinstance(m, KdaAttention):
                m.a_proj.register_forward_hook(self._alpha_hook(m))

    def _res_hook(self, mod, inp, out):
        if not self.on:
            return
        a = out[1].detach().float()
        self.ent.append((-(a.clamp_min(1e-9).log() * a).sum(-1)).mean().item())

    def _alpha_hook(self, kda):
        def hook(mod, inp, out):
            if not self.on:
                return
            B, T, _ = out.shape
            z = out.detach().float().reshape(B, T, kda.n_heads, kda.d_head)
            g = kda.g_min * torch.sigmoid(kda.A_log.detach().float().exp().view(1, 1, -1, 1) * z)
            a = g.exp()
            self.a_lo.append(a.min().item())
            self.a_hi.append(a.max().item())
        return hook

    def reset(self):
        self.ent.clear()
        self.a_lo.clear()
        self.a_hi.clear()

    def summary(self):
        return {
            "attnres_entropy": self.ent[:],
            "alpha_min": min(self.a_lo) if self.a_lo else None,
            "alpha_max": max(self.a_hi) if self.a_hi else None,
        }


# ---------------------------------------------------------------- io


def save_ckpt(path, obj):
    """Atomic: .partial then replace. A half-written ckpt is worse than none (§7)."""
    path = Path(path)
    tmp = path.with_suffix(".partial")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def qb_reset_all(model):
    """§7: derive the histogram range from the restored bias after loading."""
    for m in model.modules():
        if isinstance(m, MoE):
            m.qb_reset()


class Jsonl:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1, encoding="utf-8")

    def write(self, **rec):
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


# ---------------------------------------------------------------- eval


@torch.no_grad()
def evaluate(model, net, val, cfg, vocab, device):
    if val is None:
        return None
    model.eval()
    val.pos = 0            # §6: identical sequences every eval
    val._ensure_perm()
    tot = 0.0
    for _ in range(cfg.eval_batches):
        x, y = val.batch(cfg.micro_batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = net(x)
        tot += ce_loss(logits, y, vocab, cfg.fp32_loss).item()
    model.train()
    return tot / cfg.eval_batches


# ---------------------------------------------------------------- pre-flight


def preflight(cfg, model, net, vocab, table, muon, adamw, opts, train, device, steps=200):
    """§8 items 4-8 in one command."""
    print("\n=== 4. step-0 loss vs ln(vocab) ===")
    x, y = train.batch(cfg.micro_batch, device)
    # no_grad matters: holding a live autograd graph here would still be resident
    # when item 7 measures peak memory, and would roughly double the reading
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = net(x)
        loss0 = ce_loss(logits, y, vocab, cfg.fp32_loss).item()
    del logits, x, y
    torch.cuda.empty_cache()
    print(f"  step-0 loss {loss0:.3f}   ln({vocab}) = {math.log(vocab):.3f}   "
          f"ratio {loss0/math.log(vocab):.3f}")
    if loss0 > 2 * math.log(vocab):
        print("  ** FAIL - init regressed (see §11 tied-embedding init)")

    print("\n=== 5. parameter groups ===")
    print_table(table, muon, adamw)
    print("  assertions passed (coverage, disjoint, ndim>=2, buffers excluded)")

    print("\n=== 6. bf16 vs fp32 in _chunk ===")
    chunk_precision_check(model, device)
    torch.cuda.empty_cache()

    print("\n=== 7. memory and throughput ===")
    tok_s, peak = throughput(model, net, cfg, vocab, opts, train, device, steps)
    print(f"  micro_batch {cfg.micro_batch} x accum {cfg.accum}  "
          f"= {cfg.tokens_per_step():,} tokens/step")
    print(f"  peak memory {peak:.2f} GB")
    print(f"  {tok_s:,.0f} tok/s")
    active = 63.3e6 if cfg.arch == "minik3" and cfg.d_model == 512 else None
    if active:
        print(f"  effective {6*active*tok_s/1e12:.2f} TFLOP/s")

    print("\n=== 8. budget ===")
    for hours in (48, 55, 60):
        toks = tok_s * hours * 3600
        print(f"  {hours}h -> {toks/1e9:6.2f}B tokens -> "
              f"total_steps = {int(toks/cfg.tokens_per_step()):,}")
    print("  pick the 55h row, round it, write it into config.py, do not reopen it (sec 3)")


@torch.no_grad()
def chunk_precision_check(model, device):
    kda = next(m for m in model.modules() if isinstance(m, KdaAttention))
    B, T, H, D = 2, 256, kda.n_heads, kda.d_head
    gen = torch.Generator(device=device).manual_seed(0)
    q = F.normalize(torch.randn(B, T, H, D, device=device, generator=gen), dim=-1)
    k = F.normalize(torch.randn(B, T, H, D, device=device, generator=gen), dim=-1)
    v = torch.randn(B, T, H, D, device=device, generator=gen)
    g = kda.g_min * torch.sigmoid(torch.randn(B, T, H, D, device=device, generator=gen))
    beta = torch.rand(B, T, H, device=device, generator=gen)
    s0 = torch.randn(B, H, D, D, device=device, generator=gen) * 0.1

    ref, _ = kda._recurrent(*(t.double() for t in (q, k, v)), g.double().exp(),
                            beta.double(), s0.double())

    o32, _ = kda._chunk(q, k, v, g, beta, s0.clone())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        o16, _ = kda._chunk(q, k, v, g, beta, s0.clone())

    scale = ref.abs().max()
    e32 = ((o32.double() - ref).abs().max() / scale).item()
    e16 = ((o16.double() - ref).abs().max() / scale).item()
    print(f"  fp32 rel err {e32:.3e}")
    print(f"  bf16 rel err {e16:.3e}")
    if e16 > 1e-2:
        print("  ** NOT CLEAN - wrap the decay-and-solve section in autocast(enabled=False) (sec 3)")
    else:
        print("  clean - leave _chunk under autocast")


def throughput(model, net, cfg, vocab, opts, train, device, steps):
    opt_m, opt_a = opts
    torch.cuda.reset_peak_memory_stats()
    model.train()

    t0 = None
    for i in range(steps + 10):
        if i == 10:                      # discard warmup
            torch.cuda.synchronize()
            t0 = time.time()
        for _ in range(cfg.accum):
            x, y = train.batch(cfg.micro_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            loss = ce_loss(logits, y, vocab, cfg.fp32_loss) / cfg.accum
            loss.backward()
        clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt_m.step(); opt_a.step()
        opt_m.zero_grad(set_to_none=True); opt_a.zero_grad(set_to_none=True)
        qb_step(model)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return steps * cfg.tokens_per_step() / dt, torch.cuda.max_memory_allocated() / 1e9


# ---------------------------------------------------------------- main


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="main", choices=list(PRESETS))
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--accum", type=int, default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--lr-muon", type=float, default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--fp32-loss", dest="fp32_loss", action="store_true", default=None)
    ap.add_argument("--no-fp32-loss", dest="fp32_loss", action="store_false", default=None)
    ap.add_argument("--ckpt-minutes", type=float, default=None)
    ap.add_argument("--grad-checkpoint", dest="grad_checkpoint",
                    action="store_true", default=None)
    ap.add_argument("--compile", default=None,
                    help='"", cudagraphs, inductor, aot_eager')
    ap.add_argument("--dry-run", action="store_true", help="preflight items 4-8, then exit")
    ap.add_argument("--dry-run-steps", type=int, default=200)
    args = ap.parse_args()

    cfg = PRESETS[args.config]
    overrides = {k: getattr(args, k) for k in
                 ("out", "data", "micro_batch", "accum", "total_steps", "lr_muon",
                  "max_hours", "fp32_loss", "compile", "grad_checkpoint",
                  "ckpt_minutes")
                 if getattr(args, k) is not None}
    if overrides:
        cfg = replace(cfg, **overrides)

    keep_awake()
    device = "cuda"
    torch.manual_seed(cfg.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)

    train, val, manifest = make_loaders(cfg.data, cfg.seq_len, cfg.seed)
    vocab = manifest["vocab_size"]          # §2: never the MiniK3 default
    hold = f"{val.total*cfg.seq_len/1e9:.3f}B" if val else "none"
    print(f"data    {cfg.data}  vocab {vocab}  "
          f"train {train.total*cfg.seq_len/1e9:.3f}B  holdout {hold}")

    model = build_model(cfg, vocab).to(device)
    opt_m, opt_a, table = build_optimizers(model, cfg)
    probes = Probes(model)          # hooks must be registered before compiling
    # `model` stays the eager module for state_dict / modules() / parameters();
    # `net` is what we call forward on. Keeping them separate avoids the
    # "_orig_mod." key prefix that would break checkpoint compatibility (§7).
    net = model
    if cfg.compile == "chunk":
        # Compiling the whole model is pointless here: dynamo breaks the graph at
        # qb_accumulate's buffer mutation and at _checkpoint_block's variadic call,
        # so inductor never sees the KDA chunk loop where the time actually goes.
        # _chunk touches no self attributes and no buffers, so it compiles as a pure
        # function -- patched on the class so all 9 KDA layers share one compilation.
        import model as _m
        print("compiling KdaAttention._chunk with inductor (first step will be slow)")
        _m.KdaAttention._chunk = torch.compile(_m.KdaAttention._chunk, dynamic=False)
    elif cfg.compile:
        print(f"compiling with backend={cfg.compile!r} (first step will be slow)")
        net = torch.compile(model, backend=cfg.compile)

    if args.dry_run:
        muon = [p for g in opt_m.param_groups for p in g["params"]]
        adamw = [p for g in opt_a.param_groups for p in g["params"]]
        preflight(cfg, model, net, vocab, table, muon, adamw, (opt_m, opt_a),
                  train, device, args.dry_run_steps)
        return

    # ---- resume (§7): auto, not a flag
    ckpt_last = out / "ckpt_last.pt"
    step, tokens_seen = 0, 0
    if ckpt_last.exists():
        ck = torch.load(ckpt_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt_m.load_state_dict(ck["muon"])
        opt_a.load_state_dict(ck["adamw"])
        train.load_state_dict(ck["train_loader"])
        step, tokens_seen = ck["step"], ck["tokens_seen"]
        qb_reset_all(model)                 # §7 — one line, removes a resume/continuous difference
        print(f"resumed step {step:,}  tokens {tokens_seen/1e9:.3f}B")
    else:
        print("fresh run")

    log = Jsonl(out / "log.jsonl")
    (out / "config.json").write_text(json.dumps(cfg.dict(), indent=2, default=str))

    warmup = max(1, int(cfg.warmup_frac * cfg.total_steps))
    model.train()
    train.warm()

    t_start = time.time()
    t_ckpt = time.time()
    t_keep = time.time()
    t_win = time.time()
    tok_win = 0

    while step < cfg.total_steps:
        lr_m = lr_at(step, cfg.lr_muon, cfg.total_steps, warmup, cfg.min_lr_frac)
        lr_a = lr_at(step, cfg.lr_adamw, cfg.total_steps, warmup, cfg.min_lr_frac)
        set_lr(opt_m, lr_m)
        set_lr(opt_a, lr_a)

        heavy = cfg.heavy_every and step % cfg.heavy_every == 0
        probes.reset()

        loss_sum = 0.0
        for micro in range(cfg.accum):
            probes.on = heavy and micro == 0     # one micro-batch is enough
            x, y = train.batch(cfg.micro_batch, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = net(x)
            # fp32 loss (§3); scale by accum so effective LR is right (§11)
            loss = ce_loss(logits, y, vocab, cfg.fp32_loss) / cfg.accum
            loss.backward()
            loss_sum += loss.item()
        probes.on = False

        gnorm = clip_grad_norm_(model.parameters(), cfg.grad_clip).item()
        opt_m.step()
        opt_a.step()
        opt_m.zero_grad(set_to_none=True)
        opt_a.zero_grad(set_to_none=True)
        qb_step(model)                      # §5 — takes effect next step

        step += 1
        tokens_seen += cfg.tokens_per_step()
        tok_win += cfg.tokens_per_step()

        if step % cfg.log_every == 0:
            now = time.time()
            tok_s = tok_win / (now - t_win)
            t_win, tok_win = now, 0
            log.write(kind="light", step=step, tokens=tokens_seen, loss=loss_sum,
                      lr_muon=lr_m, lr_adamw=lr_a, grad_norm=gnorm, tok_s=tok_s,
                      mem_gb=torch.cuda.max_memory_allocated() / 1e9)
            print(f"{step:>7} {tokens_seen/1e9:7.3f}B  loss {loss_sum:6.3f}  "
                  f"gn {gnorm:6.2f}  {tok_s/1e3:6.1f}k tok/s")

        if heavy:
            stats = moe_load_stats(model)   # §6 — BEFORE the eval pass
            vloss = evaluate(model, net, val, cfg, vocab, device)
            log.write(kind="heavy", step=step, tokens=tokens_seen, loss=loss_sum,
                      val_loss=vloss, moe=stats, **probes.summary())
            if stats:
                md = max(s["max_dev"] for s in stats)
                dr = max(s["dropped"] for s in stats)
                print(f"        val {vloss:.4f}  max_dev {md:.3f}  dropped {dr:.4f}")

        now = time.time()
        if now - t_ckpt > cfg.ckpt_minutes * 60:
            t_ckpt = now
            payload = dict(step=step, tokens_seen=tokens_seen, model=model.state_dict(),
                           muon=opt_m.state_dict(), adamw=opt_a.state_dict(),
                           train_loader=train.state_dict(), config=cfg.dict(),
                           vocab_size=vocab)
            save_ckpt(ckpt_last, payload)
            if now - t_keep > cfg.keep_hours * 3600:
                t_keep = now
                save_ckpt(out / f"ckpt_{step:07d}.pt", payload)

        if cfg.max_hours and (now - t_start) > cfg.max_hours * 3600:
            print("max_hours reached")
            break

    save_ckpt(ckpt_last, dict(step=step, tokens_seen=tokens_seen, model=model.state_dict(),
                              muon=opt_m.state_dict(), adamw=opt_a.state_dict(),
                              train_loader=train.state_dict(), config=cfg.dict(),
                              vocab_size=vocab))
    print(f"done: step {step:,}  tokens {tokens_seen/1e9:.3f}B")


if __name__ == "__main__":
    main()
