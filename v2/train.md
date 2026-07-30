# Mini-k3 — training and post-training

Companion to `structure.md`, which specifies the architecture. This document covers the run:
budget, data, optimizer, what has to exist before you start, and what to do with the model
afterwards.

Numbers assume the main config — 125.9M total / 63.3M active, verified against the assembled
model, not estimated.

---

## 1. The compute reality

**Your binding constraint is FLOPs, not VRAM.** A 3090 could hold a 500M-parameter version of
this model comfortably; what it cannot do is feed one enough tokens. This is worth internalizing
because it inverts the usual instinct to shrink the model when things get tight — shrinking below
~63M active buys you almost nothing, and costs capability.

| | |
|---|---|
| Training FLOPs | `6 × 63.3M × tokens` |
| At 3B tokens | `1.14e18` FLOPs |
| 3090 dense BF16 peak | ~71 TFLOP/s |
| Realistic effective | 10–15 TFLOP/s (see below) |
| Pure compute, 3B tokens | ~21–32 hours |
| **Wall clock to budget** | **2–3 days** |

MFU will be poor and that is expected, not a bug to chase. `d_ff_e = 224` and `ℓ = 256` mean the
expert GEMMs are small; small GEMMs do not saturate tensor cores. 15–20% MFU is a realistic target.
Budget for restarts.

Two things that were going to dominate that budget are now fixed (§5): KDA runs `T/16` sequential
steps instead of `T`, and MoE dispatch runs one batched GEMM per layer instead of 32 plus a host
sync — measured 5.2× on the dispatch alone. Re-measure tokens/sec before assuming the §1 wall-clock
estimate still holds; it was written against the old code and is now pessimistic.

**Token budget.** Chinchilla-optimal for 63M active is ~1.2B tokens, but MoEs want more tokens per
active parameter — K3 retuned its tokens-per-parameter ratio for exactly this reason. Aim for
**2–4B**, and if the run is stable at 3B and you can spare another day, keep going. FineWeb-Edu's
10B sample will not run out.

Architecture fidelity is what makes this project interesting; **token count is what will make the
model good.** SmolLM-135M is a strong model at this size because it saw 600B tokens. You are not
doing that, but the direction of the tradeoff is worth respecting: if you have a spare day, spend
it on tokens rather than on another architectural refinement.

---

## 2. Data

| | Choice | Why |
|---|---|---|
| Tokenizer | 32K BPE, trained on the corpus | Tied embeddings are 16.4M params — 13% of the model. Going smaller starves capacity; larger wastes it. |
| Main corpus | FineWeb-Edu, 10B sample | Highest quality-per-token available without a curation pipeline. Path of least resistance. |
| Debug corpus | TinyStories | Overfits in ~90 seconds at the debug config. Every phase gate in `structure.md §5` runs against this. |
| Seq length | 2048 | |

Pack documents to fixed length with an EOS separator; don't pad. At 2048 tokens and a 32K vocab,
one packed sequence is a reasonable unit for the loader to hand out.

Hold out a fixed FineWeb-Edu slice — a few million tokens, never trained on — from the very first
run. You need it for the before/after comparison in §8, and retrofitting a clean holdout after the
fact is not possible.

---

## 3. Optimizer and schedule

```
precision       bf16 autocast, fp32 master weights
optimizer       Muon (matrices) + AdamW (everything else)
lr schedule     cosine decay to ~10% of peak, 1% linear warmup
weight decay    0.1
grad clip       1.0
seq len         2048
tokens/step     ~32K  (batch 16 × 2048, via gradient accumulation)
checkpointing   on
embeddings      tied
```

The report ran a dedicated scaling-law search per schedule and found **cosine beats WSD** once each
gets its own hyperparameters — the two have substantially different optimal peak LR and batch size,
so a shared sweep unfairly favours one. Use cosine; you cannot afford the sweep that would justify
anything else.

### Parameter groups

Muon operates on 2D matrices via Newton–Schulz orthogonalization. Anything that isn't a matrix goes
to AdamW. Getting this split wrong is quiet — Muon on a 1D tensor either errors or silently does
something meaningless.

| Parameter | Optimizer | Note |
|---|---|---|
| `wq`, `wk`, `wv`, `wo`, `g_proj` (KDA, MLA) | Muon | |
| `k_up`, `v_up`, `kv_down` (MLA) | Muon | |
| `down_proj`, `up_proj` (MoE) | Muon | |
| `w_gate`, `w_up`, `w_down` (routed experts) | Muon | 3D stacked — orthogonalize per expert slice, not across the stack |
| Shared expert matrices | Muon | |
| `embedding.weight` (tied with `lm_head`) | AdamW | |
| All `RMSNorm` weights | AdamW | 1D |
| `AttnRes.w` pseudo-queries | AdamW | 1D vectors, 27 of them |
| `A_log` (KDA per-head log-scale) | AdamW | 8 scalars per layer |
| `a_proj[1].bias`, `b_proj` | AdamW | |
| `router.weight` | **judgment call** | It's a matrix, but routers are sometimes kept on AdamW for stability. Start on Muon; if load balance oscillates for reasons QB can't explain, move it. |

The 3D expert stacks are the one real trap: `w_gate` is `(32, 256, 224)`. Orthogonalizing that as a
single 2D reshape couples all 32 experts into one update direction, which is precisely the failure
mode Per-Head Muon exists to avoid. Loop over the expert dimension, or reshape so each expert's
block is handled separately.

**Per-Head Muon is not worth implementing.** K3 partitions attention momentum along the head
dimension so heads with large gradient scales don't dominate. At 8 heads you will not measure the
difference. Plain Muon on the full projection matrices is fine.

### Learning rate

Neither the report nor `structure.md` pins this down, and K3's values came from a scaling-law search
you cannot afford. Starting points, to be swept at the debug config before the main run:

```
Muon  peak lr   0.02        (matrices)
AdamW peak lr   3e-3        (embeddings, norms, vectors)
warmup          1% of total steps, linear
min lr          10% of peak
```

Sweep the Muon LR over roughly `{0.01, 0.02, 0.04}` on a few hundred debug-config steps and take the
best final loss. That is two hours of work and it is the highest-leverage tuning available.

---

## 4. Memory

Rough accounting at batch 8 × 2048, to show where the headroom is:

| | |
|---|---|
| fp32 master weights | ~504 MB |
| Gradients (fp32) | ~504 MB |
| Muon momentum | ~440 MB (one state per matrix — cheaper than AdamW's two) |
| AdamW states (embeddings etc.) | ~131 MB |
| AttnRes source list | ~450 MB (27 tensors × B × T × 512, bf16) |
| Module activations | remainder |

Comfortable on 24 GB. Two notes:

The **AttnRes source list is the one activation cost unique to this architecture** — the `O(Ld)`
memory the report names as the real overhead of full AttnRes. It scales linearly with batch, so it
is the first thing to feel if you push batch size up. If it bites, gradient-checkpoint the blocks
before you consider Block AttnRes; blocking is a multi-device optimization and buys nothing here.

Your 64 GB of system RAM is useful for the data pipeline — tokenize and pack to memory-mapped
`.bin` shards ahead of time so the loader never competes with training for CPU.

---

## 5. Prerequisites — status

All three are implemented in `model.py`. Each fast path kept its reference implementation and was
validated against it to machine precision, so a future kernel swap has something to adjudicate
against.

**1. KDA sequential loop — done.** `KdaAttention._chunk` implements the chunkwise parallel form
(report §2.1.1, Eq. 3–4) via the UT/WY transform: keys are rescaled by the cumulative decay, the
intra-chunk write vectors come from one batched triangular solve, and only the chunk-to-chunk state
carry stays sequential. Sequential steps drop from `T` to `T/16` — 2048 to 128 at seq 2048.
`_recurrent` is retained as the reference and still serves decode, where `forward` routes `T < 32`
to it so `generate()` skips the padding and solve entirely.

Verified exact against the loop: **2e-16** in float64 across ragged lengths and non-zero entering
states, and **6e-14** for prefill vs 40-step incremental decode through the full model.

Chunk size is capped at 16 by `g_min = -5`. The reciprocal cumulative decay `1/γ` reaches `e^{5C}`,
so `C=16` gives `e^80` — inside fp32/bf16 range — and `C=32` overflows. **Do not raise `C` without
two-level tiling.** The state carry uses `k·γ_C/γ ≤ 1` instead, so the value persisted across chunks
and into the cache never touches the unbounded quantity.

FLA's Triton path is still an option if you want more speed later; the chunk form here is pure
PyTorch and adds no dependency. FlashKDA itself is CUTLASS and likely won't build for sm_86.

**2. QB bias update — done.** `MoE.route` now does Top-(k+1), taking the cutoff `α_i` straight from
routing rather than a separate token-side quantile pass. `qb_accumulate` bins the required bias
`r_ij = α_i − s_ij` into a per-expert histogram every forward — no communication, no host sync,
accumulating across micro-batches. `qb_update` reads the `(k/n)`-quantile off the pooled histogram
and mean-centres it (report §2.3.3 Eq. 14, histogram estimator §D).

The one-step lag is structural, not a convention you have to remember: accumulation happens in
`forward`, the update only when the training loop calls `qb_step`. A batch cannot be routed with a
bias derived from itself. Accumulation is gated on `self.training` and `router_bias` is a persistent
buffer, so the bias is frozen and checkpointed for inference.

**3. Per-expert load logging — done.** `moe_load_stats(model)` returns per-layer max deviation from
`mk/n`, dead-expert count, capacity drop fraction, and router-bias spread. One sync; call it every
N steps, not every step.

**New — capacity-based dispatch.** `MoE.dispatch` packs tokens into a static
`(n_routed, capacity, d_latent)` buffer and runs every expert in one batched GEMM, replacing 32
per-expert GEMMs and the `.tolist()` host sync. Measured **5.2× faster** (fwd+bwd, batch 8×2048),
exact against `_dispatch_loop` (forward 2.6e-18, gradients ~5e-19) whenever nothing is dropped.

This introduces one behavioural change: **assignments past `capacity` are dropped.** That is the
tradeoff QB pays for — perfect balance is what makes static shapes affordable (report §5.2.1).
`capacity_factor` defaults to 1.25; the padding is wasted FLOPs, so drop it to 1.1 once the load
logging shows QB converged. With an untrained router, 1.25 drops 0.44% and 1.0 drops 6.5%.

### What the training loop must call

```python
loss.backward()
clip_grad_norm_(model.parameters(), 1.0)
opt.step(); opt.zero_grad()
qb_step(model)                        # QB update; takes effect next step
if step % log_every == 0:
    stats = moe_load_stats(model)     # one sync
```

Under DDP, `qb_step` all-reduces the histogram and token count internally — the quantile must be
over the pooled global batch, and counts being additive is what makes one all-reduce sufficient.
**Do not rely on `broadcast_buffers` to paper over this**: it would silently balance on rank 0's
shard alone.

---

## 6. What to log

Beyond loss and LR. The first four come straight out of `moe_load_stats(model)`.

- **Per-expert token counts**, per layer (`MoE.load`). The headline MoE health metric.
- **Max deviation from target load** `mk/n` (`max_dev`) — one scalar summarizing the above, easy to
  alarm on. See the expected floor below before setting a threshold.
- **Capacity drop fraction** (`dropped`). New with the batched dispatch, and a *distinct* failure
  signature from imbalance: a rising drop rate with flat `max_dev` means `capacity_factor` is too
  tight, not that QB is failing. Should sit at ~0 once QB has converged.
- **Router bias spread** (`bias_spread`, `max − min`). QB's correction effort. Growing without a
  corresponding fall in `max_dev` means the router has saturated — see below.
- **AttnRes weight entropy, per sublayer.** If it collapses to one-hot on the immediately preceding
  source, you have reinvented the residual stream and gained nothing for the `O(Ld)` memory.
- **α range per KDA layer.** Assert the `e⁻⁵` floor holds. If it drops below, you are on the old
  unbounded Softplus mapping.
- **Activation norm per sublayer output.** AttnRes is a convex combination so magnitudes shouldn't
  drift, but this is how you'd find out if something in the routed path is exploding.
- **Grad norm, pre-clip.** Spikes here precede loss spikes.
- **Tokens/sec and effective TFLOP/s.** So you know whether the kernel work actually helped.

### Expected `max_dev` — do not chase it to zero

QB plateaus at a non-zero deviation and this is statistics, not a bug. The bias is fitted on step
*t*'s batch and applied to step *t+1*'s different batch, so the residual is sampling noise on a
target load of `mk/n`. Measured, against a `2.5/√(target load)` prediction:

| tokens/step | target load | measured | predicted |
|---|---|---|---|
| 2,048 | 256 | 17.7% | 15.6% |
| 8,192 | 1,024 | 8.1% | 7.8% |
| 32,768 | 4,096 | 4.2% | 3.9% |

At the config in §3 (~32K tokens/step, accumulated across micro-batches) **expect ~4%, and alarm
above roughly 10%.** More tokens per step is the only thing that lowers this floor; `qb_ema` (report
§D's refinement, default off) buys ~1–2 points at small batch and hurts above 0.9.

### If QB genuinely cannot balance

Check whether the router's sigmoid has saturated before suspecting the update. QB derives biases
from score *quantiles* and assumes no ties; if pre-activations are large enough that scores pile up
at 0 and 1, the quantiles become degenerate and **no bias can balance the load** — verified by
computing the exact quantile and watching it fail too. The signature is `bias_spread` growing while
`max_dev` stays flat. The fix is at the router (init scale, or LR — see the §3 judgment call about
moving `router.weight` to AdamW), not in QB.

---

## 7. Post-training

Be realistic about what 63M active parameters can do. It will not be a general assistant, it cannot
do multi-step reasoning, and chat-style RLHF at this scale mostly teaches fluent hallucination. What
works is a narrow, well-matched instruct model.

### SFT

Use **`HuggingFaceTB/smol-smoltalk`**. It was curated by the SmolLM team specifically for their
135M/360M models — short contexts, simple instructions, rewriting, summarization, simple QA. That is
your regime exactly. Standard Alpaca/UltraChat-style data is written for 7B+ models, and small
models imitate its surface register without the substance behind it, which reads worse than a model
that stays in its lane.

Tasks where a model this size can look genuinely good:

- **Summarization and rewriting** — a constrained mapping, and small models handle it well.
- **Constrained generation** — short stories, product descriptions, simple formatted output.
  Fluency is the one thing this scale has in abundance.
- **Simple instruction following** — tone changes, list formatting, "write X in the style of Y".

### Effort-conditioned SFT — recommended

K3's post-training centrepiece is **effort conditioning**: reasoning effort is exposed as an
in-context option message stating the requested level in natural language, and the model is trained
across `{low, high, max}` budgets (report §4.1.2, §F). A miniature of the *mechanism* is affordable
here: tag SFT examples by response length band, condition on the tag, and show at inference that
the model actually modulates output length.

This is cheap, it is faithful to the architecture you reimplemented, and it demonstrates something
about the training recipe rather than just "I ran SFT". Measure it: mean output length per effort
tag, with the tag as the only thing that varies.

### DPO — optional

A single pass on `HuggingFaceH4/ultrafeedback_binarized` is roughly an hour of training at this
scale. Worth doing if you want the preference-optimization stage in the pipeline; do not expect it
to make the model smarter. Run it after SFT and keep the SFT checkpoint — DPO at small scale can
degrade fluency, and you want to be able to compare.

---

## 8. Evaluation

Measure enough that claims about the model are defensible rather than descriptive.

**During pre-training:** perplexity on the held-out FineWeb-Edu slice from §2, at fixed intervals.
This is the only number that tracks whether the run is working.

**Architecture validation:** the phase gates in `structure.md §5`. These are correctness tests, not
quality metrics, but a run whose gates never passed is not evidence of anything.

**Before/after post-training:** run `tinyBenchmarks` (100-example subsets of standard suites — cheap
enough for this scale) plus a small held-out instruction set scored by hand or by a larger model.
Report SFT deltas against the base checkpoint on the same tasks.

**Effort conditioning, if implemented:** mean output tokens per effort tag on a fixed prompt set.

**A GPT-2-scale control is worth the day it costs.** Train a plain dense transformer with the same
active parameter count, tokenizer, data, and token budget. It is the only way to say anything
grounded about what the K3 architecture bought you — otherwise you have a number with nothing to
compare it to. This is also the most interesting result the project can produce, and it is the one
result nobody can get from reading the paper.

---

## 9. Training-specific failure modes

None of these crash. All of them waste the run.

Now structurally prevented in `model.py` — listed so you know not to re-introduce them:

- [x] ~~KDA sequential loop in the training path~~ — `forward` routes `T ≥ 32` to `_chunk`
- [x] ~~QB bias never updated~~ — `qb_step(model)`, one line in the loop
- [x] ~~QB bias derived from the batch it routes~~ — accumulation is in `forward`, update only in
      `qb_step`, so the lag cannot be skipped
- [x] ~~Router bias not frozen at inference~~ — accumulation gated on `self.training`; verified
      bit-identical across `generate()`
- [x] ~~Tied `lm_head` inheriting `nn.Embedding`'s N(0,1) init~~ — logits of std ~12 and an initial
      loss ~16× `ln(vocab)`, with "copy the current token" as the nearest escape. Now `std=0.02`,
      which puts step 0 at a uniform predictor

Still live:

- [ ] `capacity_factor` too tight for an unconverged router — silent token dropping. Watch
      `dropped`; keep 1.25 until QB settles, then 1.1
- [ ] Under DDP, QB histogram not all-reduced — `qb_step` handles it, but only if you call it on
      every rank. Do not let `broadcast_buffers` mask a missing call
- [ ] Muon applied to 1D parameters, or to the expert stack as one coupled matrix
- [ ] Muon applied to the QB buffers — `router_bias`, `qb_hist` etc. are buffers, not parameters, so
      they are invisible to `model.parameters()`. Confirm your param groups still cover everything
- [ ] `values` list truncated because a block's return value was dropped — trains, sees fewer sources
- [ ] `(y, cache)` tuple appended to `values` as a source — fails at `stack`, but only after a wasted run
- [ ] No held-out split reserved before the first run — no defensible before/after later
- [ ] Holdout accidentally inside the training shards — perplexity looks great, means nothing
- [ ] Grad accumulation without scaling the loss by accumulation steps — effective LR is wrong by that factor
- [ ] Initial loss not checked against `ln(vocab)` on step 0 — the cheapest possible sanity check and
      it catches init bugs before they cost you a day

---

## 10. Order of operations

1. Phase gates 0–4 pass at the debug config (`structure.md §5`)
2. ~~Chunkwise KDA verified against the reference loop~~ — done, 2e-16
3. ~~QB update wired~~ — done; confirm load balance settles near the §6 floor on real data
4. LR sweep at debug config, a few hundred steps per point
5. Tokenizer trained, corpus packed to mmap shards, holdout split reserved
6. Dense control run — same active params, same data, same budget
7. Main run, 2–4B tokens
8. SFT on smol-smoltalk, with effort tags
9. Optional DPO
10. Evaluation table: base vs. SFT vs. control

Steps 2 and 3 were verified on synthetic data. Re-check both once real data is flowing — QB's
behaviour depends on the router's score distribution, which random inputs do not represent.
