"""Run configuration. Dumped verbatim into every checkpoint (train.md §7)."""

from dataclasses import dataclass, asdict, replace, field


@dataclass
class Config:
    name: str = "main"
    arch: str = "minik3"

    # data
    data: str = r"D:\ml\data"
    out: str = r"D:\ml\runs\main"
    seq_len: int = 2048
    seed: int = 0

    # model — MiniK3 defaults are the main config (125.9M total / 63.3M active)
    d_model: int = 512
    n_heads: int = 8
    d_head: int = 64
    d_kv: int = 128
    d_latent: int = 256
    d_ff_e: int = 224
    d_ff_s: int = 448
    n_shared: int = 2
    n_routed: int = 32
    top_k: int = 4
    n_blocks: int = 3
    n_kda: int = 3
    capacity_factor: float = 1.25
    qb_ema: float = 0.0

    # batching — tokens/step = micro_batch * accum * seq_len
    micro_batch: int = 16
    accum: int = 1

    # optimizer (§3)
    total_steps: int = 91_500
    warmup_frac: float = 0.01
    min_lr_frac: float = 0.10
    lr_muon: float = 0.02
    lr_adamw: float = 3e-3
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    adamw_betas: tuple = (0.9, 0.95)
    router_on_muon: bool = True

    # fp32 loss follows train.md §3, but the logits.float() copy is 1.95 GiB at
    # micro_batch 8 and is what OOMs it. See ce_loss() in train.py.
    fp32_loss: bool = True
    # "", "cudagraphs", "inductor", "aot_eager" -- §7 says the run must not depend on it
    compile: str = ""
    # legal since Block.forward stopped mutating values/keys in place
    grad_checkpoint: bool = False

    # run
    log_every: int = 20
    heavy_every: int = 500
    eval_batches: int = 20
    ckpt_minutes: float = 20.0
    keep_hours: float = 3.0
    max_hours: float = 0.0        # 0 = run to total_steps

    def tokens_per_step(self):
        return self.micro_batch * self.accum * self.seq_len

    def dict(self):
        return asdict(self)


# Main run. Measured on the 3090 as configured (curve undervolt, 350 W board limit)
# against the real corpus:
#   4 x 4, no checkpointing    16.72 GB    3,119 tok/s   1.18 TFLOP/s
#   16 x 1, checkpointed       20.54 GB    5,400 tok/s   2.05 TFLOP/s
#   16 x 1, ckpt + bf16 loss   20.54 GB    6,050 tok/s   2.30 TFLOP/s
#   16 x 1, + compile=chunk    18.87 GB   10,492 tok/s   3.99 TFLOP/s   <- this config
# micro_batch 8 and 16 both OOM without checkpointing (~32 GB of activations).
# compile="chunk" costs ~7 min cold / ~1 min warm; inductor caches to disk, so
# run.ps1 relaunches are cheap. Compiling the whole model gains nothing (+0.9%):
# dynamo breaks the graph at qb_accumulate and at _checkpoint_block's *args splat.
MAIN = replace(
    Config(),
    micro_batch=16,
    accum=1,
    grad_checkpoint=True,
    fp32_loss=False,
    compile="chunk",
    # Thu 23:00 launch -> Sun 18:00 deadline = 67 h. The 300-step drill sustained
    # 10,000-11,600 tok/s (mean ~11,000) including checkpoint writes. 73,000 steps
    # lands Sun 11:24 at 11.0k, Sun 14:16 at 10.5k, Sun 17:26 at 10.0k -- i.e. it
    # still finishes if throughput degrades ~10%. Overrunning is the bad outcome:
    # the cosine would not have decayed, so the schedule would mean nothing (sec 3).
    total_steps=73_000,        # 2.39B tokens
)

# Memory fallback: same 32K tokens/step, ~12% slower, lower peak. Untested for speed.
MAIN_8x2 = replace(MAIN, name="main-8x2", micro_batch=8, accum=2)

# Debug — structure.md §5: d=256, 3 KDA + 2 MLA, 8 experts top-2, 1 shared, seq 512.
# n_blocks=1/n_kda=3 gives exactly 3 KDA + 2 MLA once the final n_kda=0 block is counted.
DEBUG = replace(
    MAIN,
    name="debug",
    data=r"D:\ml\data-debug",
    out=r"D:\ml\runs\debug",
    seq_len=512,
    d_model=256,
    d_latent=128,
    d_ff_e=112,
    d_ff_s=224,
    n_shared=1,
    n_routed=8,
    top_k=2,
    n_blocks=1,
    micro_batch=8,
    accum=1,
    total_steps=2_000,
    log_every=10,
    heavy_every=100,
    eval_batches=10,
    ckpt_minutes=5.0,
)

# Dense control (§10) — same active params, tokenizer, data and budget.
CONTROL = replace(MAIN, name="control", arch="dense", out=r"D:\ml\runs\control")

PRESETS = {"main": MAIN, "main-8x2": MAIN_8x2, "debug": DEBUG, "control": CONTROL}


def build_model(cfg, vocab_size):
    """Model factory. train.py takes this, not MiniK3 (§7)."""
    if cfg.arch == "minik3":
        from model import MiniK3, MoE

        m = MiniK3(
            vocab_size=vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            d_head=cfg.d_head,
            d_kv=cfg.d_kv,
            d_latent=cfg.d_latent,
            d_ff_e=cfg.d_ff_e,
            d_ff_s=cfg.d_ff_s,
            n_shared=cfg.n_shared,
            n_routed=cfg.n_routed,
            top_k=cfg.top_k,
            n_blocks=cfg.n_blocks,
            n_kda=cfg.n_kda,
        )
        m.grad_checkpoint = cfg.grad_checkpoint
        for mod in m.modules():
            if isinstance(mod, MoE):
                mod.capacity_factor = cfg.capacity_factor
                mod.qb_ema = cfg.qb_ema
        return m

    if cfg.arch == "dense":
        raise NotImplementedError(
            "dense control model not written yet — §12 puts it the week after the main run. "
            "Add it here; train.py needs no changes."
        )

    raise ValueError(f"unknown arch {cfg.arch!r}")
