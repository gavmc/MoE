# Mini-k3

A from-scratch, single-GPU implementation of the Kimi K3 architecture, scaled to ~120M parameters
so it trains on one RTX 3090 in a couple of days.

This is an educational reimplementation. The goal is to get every architectural component
*correct*, not to produce a competitive model. Everything here follows the Kimi K3 technical
report (Moonshot AI, July 2026); equation numbers in this document refer to that report.

---

## 1. Target configuration

| | Kimi K3 | Mini-k3 | why |
|---|---|---|---|
| Hidden dim `d` | 7,168 | **512** | fits 24 GB with room for activations |
| Latent MoE dim `ℓ` | 3,584 (0.5×) | **256** (0.5×) | same ratio as K3 |
| Expert hidden `d_ff_e` | 3,072 | **224** | K3 ratio is 0.857× of `ℓ`; rounded to a multiple of 32 |
| Shared expert hidden | — | **448** | same 0.857× ratio, against `d` |
| Layers | 93 | **13** | 3 blocks of (3 KDA + 1 MLA), plus a trailing MLA |
| Attention composition | 69 KDA + 24 MLA | **9 KDA + 4 MLA** | preserves the 3:1 ratio |
| Routed experts | 896 | **32** | see note below |
| Active per token | 16 | **4** | sparsity 8, vs K3's 56 |
| Shared experts | 2 | **2** | unchanged |
| Heads | 96 | **8 × 64** | `n_heads × d_head == d`, unlike a naive first pass |
| Vocab | 160K | **32K** | tied embeddings |
| Context (train) | 1M | **2048** | |
| **Total / active** | 2.78T / 104.2B | **~120M / ~63M** | |

**Why so few experts.** MoE sparsity buys memory-bandwidth and all-to-all communication wins
that only exist across many devices. On a single GPU you pay full VRAM for every expert while
only getting FLOPs from `k` of them, so a large pool is pure cost. 32 experts is enough for
routing collapse and load imbalance to actually show up in the logs, which is the point.

Note the honest consequence: at this scale total/active sparsity is only ~1.9×, because
attention, embeddings and the shared experts dominate. K3's 27× ratio is a property of its
width, not of the architecture. Don't try to recover it by adding experts.

**Optional:** K3 makes its first layer a dense FFN instead of a MoE. Harmless to copy, harmless
to skip.

---

## 2. Architecture in one paragraph

K3 scales information flow along three independent axes, and each gets its own mechanism.
**Sequence:** hybrid attention — KDA (linear, fixed-size recurrent state) for cheap long-range
mixing, with periodic Gated MLA layers for exact global retrieval. **Depth:** Attention Residuals,
which replace the residual stream with attention over earlier layer outputs. **Width:** Stable
LatentMoE, sparse channel mixing with experts operating in a compressed latent space. The three
compose freely; you can implement and validate them one at a time.

---

## 3. Component specifications

### 3.1 KDA (Kimi Delta Attention)

The recurrence (Eq. 1), for one head, with state `S ∈ R^(d_k × d_v)`:

```
S_t = (I − β_t k_t k_tᵀ) Diag(α_t) S_{t−1} + β_t k_t v_tᵀ
õ_t = S_tᵀ q_t
```

Read it as: decay the state channel-wise, read what's currently stored at `k_t`, and write back
only the correction toward `v_t`, scaled by `β_t`. That's one step of online gradient descent on
a regression loss — an associative memory that overwrites rather than accumulates.

Parameterization (Eq. 2):

```
q, k = L2Norm(Swish(ShortConv(W_{q/k} x)))
v    = Swish(ShortConv(W_v x))
β    = Sigmoid(W_β x)                    ∈ (0, 1)     per head
z    = W↑_α W↓_α x + b_α                              low-rank, per-head bias
```

**Decay mapping (Eq. 5) — changed in K3.** Kimi Linear used an unbounded negative-Softplus
mapping. K3 replaces it with a scaled sigmoid bounded from below:

```
g = g_min · Sigmoid(exp(A_h) · z)        ∈ (g_min, 0),   g_min = −5 fixed
α = exp(g)                               ∈ (e^−5, 1)
```

`A_h` is a **learnable per-head log-scale initialized to 0** — one scalar per head, not per
channel. The bound matters for the chunkwise kernel: it keeps the cumulative log-decay over a
16-token tile inside (−80, 0), so the reciprocal rescaling factor stays in BF16 range and every
causal tile can use dense Tensor Core matmuls instead of an explicit position-pair path.
Numerically irrelevant to a naive sequential loop, but it is a different model.

**Output gate (Eq. 6) — also changed in K3.** Full-rank projection, and note the order:

```
y = W_o[ Sigmoid(W_g x) ⊙ RMSNorm(õ) ]
```

RMSNorm applies to the attention output **first**, then the gate multiplies. The gate
nonlinearity is **Sigmoid**, not SiLU. Getting either wrong trains fine and scores worse.

### 3.2 Gated MLA

Multi-head Latent Attention from DeepSeek-V2: compress KV into a latent `c_t = W_c x_t`, cache
`c_t`, reconstruct keys and values through learned up-projections at attention time.

Two K3-specific details:

- **NoPE on all MLA layers.** No RoPE, no positional encoding at all. The KDA layers supply
  position-sensitivity through their decay; the MLA layers provide unrestricted global content
  interaction. This is also why K3 extrapolates to 1M without RoPE rescaling or YaRN.
- **Full-rank output gate (Eq. 7):** `y = W_o[Sigmoid(W_g x) ⊙ õ]`. Same gate as KDA but
  **without** the RMSNorm.

An extra MLA layer sits at the end of the backbone, so the final layer always does global
attention. That's why 13 layers is 9 KDA + 4 MLA rather than a clean 3:1.

### 3.3 Attention Residuals (AttnRes)

Standard residuals compress all prior information into a single state `h_l` over depth — a
bottleneck the report compares to an RNN over time. AttnRes applies the transformer's own fix to
the depth axis: each layer selectively retrieves from all preceding layers instead of
accumulating them uniformly.

Full form (Eq. 8–9). Per-layer learnable pseudo-query `q_l = w_l ∈ R^d`; keys and values are the
earlier layer outputs, with the token embedding as source 0:

```
k_i = v_i = h_1              if i = 0        (token embedding)
k_i = v_i = f_i(h_i)         if 1 ≤ i ≤ l−1  (output of layer i)

φ(q, k)  = exp( qᵀ RMSNorm(k) )
α_{i→l}  = φ(q_l, k_i) / Σ_j φ(q_l, k_j)
h_l      = Σ_i α_{i→l} · v_i
```

The RMSNorm on keys stops layers with large-magnitude outputs from dominating the weights.

**Critical:** `h_l` is the weighted sum and nothing else. This *replaces* the residual
connection; it is not added on top of one. There is no `x + f(x)` anywhere in the backbone.

**Use the full form, not Block AttnRes.** The report notes the `O(L²d)` cost of the full form is
affordable at modest depth (`L < 100`), and that the real overhead is the `O(Ld)` memory and
cross-stage communication under pipeline parallelism. At `L = 13` on one GPU with no pipeline
parallelism, both costs are nothing. Block AttnRes (K3 uses 8 blocks of 12 layers) is purely a
systems optimization — implementing it here buys you complexity and zero benefit.

Practical consequence for your code: you cannot express the backbone as a `nn.Sequential`.
Carry a list of prior layer outputs and pass it into every sublayer.

### 3.4 Stable LatentMoE

Layer definition (Eq. 11), for `x ∈ R^d`:

```
u = Σ_{i ∈ T_k(x)}  p_i · E_i^routed( W↓ x )              # routed path, in latent space
y = Σ_{j=1..N_s}  E_j^shared(x)  +  W↑ RMSNorm(u)         # shared path at full width
```

`E^shared : R^d → R^d`, `E^routed : R^ℓ → R^ℓ`, `N_s = 2`. Three things to get right:

- **The router runs on `x` at full width**, not on the latent vector. Only expert *computation*
  is compressed.
- **Shared experts bypass the latent path entirely.** They see `x` directly and their output is
  added at full width.
- **RMSNorm sits between expert aggregation and the up-projection.** This is the "Normalized
  LatentMoE" contribution. Motivation: the routed path chains `W↓`, a gated multi-branch FFN, and
  `W↑` into nearly four consecutive matmuls, and that ill-conditioned structure produces
  exploding activations at scale. The report also reports it improves validation loss
  independently of stability.

**SiTU-GLU (Eq. 12)** — the expert activation, a soft-capped SwiGLU:

```
SiTU-GLU(x) = [ β₁ tanh(W_g x / β₁) ⊙ Sigmoid(W_g x) ] ⊙ [ β₂ tanh(W_u x / β₂) ]

β₁ = 4   (gate branch)
β₂ = 25  (up branch)
```

Both of SwiGLU's multiplicative factors are unbounded, so coincident large coordinates produce
activation outliers. SiTU caps the *linear* factor of Swish while keeping the sigmoid, preserving
Swish's shape near the origin (it matches SwiGLU to first order) with output bounded by
`β₁β₂ = 100`. Note `W_g x` appears in both the tanh and the sigmoid of the gate branch. Three
matrices per expert: `W_g`, `W_u`, and the down-projection back to `ℓ`.

**Quantile Balancing (Eq. 13–14)** — auxiliary-loss-free load balancing:

```
s_i    = Sigmoid(W_r x_i)                       # scores, all experts
T_i    = argtopk(s_i + b)                       # selection uses the bias
p_i,j  = s_i,j / Σ_{r ∈ T_i} s_i,r              # weights do NOT use the bias
```

Omitting `b` from `p` is deliberate: it regulates dispatch without altering the mixture weights
or the router's gradients.

The bias update. Target load `q = mk/n`. Route with **Top-(k+1)** instead of Top-k: the first `k`
entries are the routes actually taken, and the `(k+1)`-th is the cutoff `α_i` an expert must
exceed to enter token `i`'s Top-k. Then:

```
b̂_j = − quantile_{1−k/n}( s_{:,j} − α )
b   = b̂ − mean(b̂)
```

Applied on the **next** step (a batch is never routed with a bias derived from itself), and
frozen at inference.

Two simplifications for your scale:

1. **Skip the histogram estimator.** K3 needs it because the margins number in the millions and
   are sharded across ranks and gradient-accumulation steps. With 32 experts on one device,
   `torch.quantile` on the exact margins is a one-liner and is *more* accurate.
2. **Start with the DeepSeek-V3 sign update** (`b_j += γ · sign(target − load_j)`) if you want
   something working today. Appendix C shows QB and the sign update are SignSGD versus exact
   coordinate minimization on the *same* LP dual objective — the sign rule is a crude
   approximation of QB, which is why QB needs no learning-rate hyperparameter and equilibrates in
   a few steps even at ~10³ experts.

### 3.5 Optimizer

Muon for matrix parameters (embeddings, norms, and biases stay on AdamW). K3 uses Per-Head Muon —
partitioning momentum along the head dimension and orthogonalizing each head's block separately,
so heads with large gradient scales don't dominate the shared update direction. At 8 heads you
will not measure the difference; plain Muon is fine.

---

## 4. What to skip, and why

| Skip | Reason |
|---|---|
| Block AttnRes | Systems optimization for 93 layers across PP stages. Use the full form. |
| Histogram quantile estimator | Needed only for sharded, million-token batches. |
| MXFP4/MXFP8 QAT | Deployment. Also: the 3090 has no FP8 or FP4 support. |
| MTP layer, EAGLE-3 draft model | Inference acceleration. |
| Vision (MoonViT-V2, projector) | Doubles the project. Add it later if you want. |
| MoonEP, KCP, pipeline/expert parallelism | Multi-device only. |
| Prefix caching, paged KV | Serving infrastructure. |
| Per-Head Muon | Plain Muon is indistinguishable at 8 heads. |

---

## 5. Build order

Each phase has a validation gate. **Do not proceed past a failing gate.** Every bug in this
architecture is silent — the model still trains, just worse — so the gates are the only cheap way
to catch them.

Use the debug config for all of phases 0–4: `d=256`, 3 KDA + 2 MLA, 8 experts top-2, 1 shared,
vocab 8K, seq 512. That's ~6M parameters and overfits TinyStories in about ninety seconds.

### Phase 0 — Harness

Tokenizer, data loader, training loop, logging, checkpointing. Build a plain pre-norm transformer
with MHA and SwiGLU first. This is your control: you need a known-good loss curve to compare
everything else against.

*Gate:* overfits a few thousand tokens to near-zero loss.

### Phase 1 — KDA

Replace MHA in the KDA slots. Write the naive sequential loop first — it is your reference
implementation, not your training implementation.

*Gates:*
1. **State carry test.** Running a sequence in two halves with state passed between them must
   equal running it whole. `torch.allclose(atol=1e-5)`. This catches ShortConv cache bugs and
   off-by-one decay errors.
2. **Kernel equivalence.** Swap in `flash-linear-attention`'s chunkwise KDA (the report footnotes
   FLA PR #691) and assert it matches your loop on identical weights. FlashKDA itself is CUTLASS
   and likely won't build for sm_86 — you'll get FLA's Triton path, which is the right target.
3. **Decay range.** Assert `α ∈ (e^−5, 1)` everywhere. If you see values below `e^−5`, you're
   still on the old negative-Softplus mapping.

Then delete the loop from the training path. At seq 2048 it is 2048 sequential kernel launches
per layer per step and will dominate every other cost by orders of magnitude.

### Phase 2 — Stable LatentMoE

Replace the FFN. Router, down/up projections, SiTU-GLU experts, the aggregate RMSNorm, shared
experts.

*Gates:*
1. **Dense equivalence.** With `top_k == n_routed`, output must equal computing every expert and
   summing with the router weights.
2. **SiTU bound.** Assert `|output| ≤ β₁β₂ = 100` elementwise.
3. **Utilization histogram.** Log per-expert token counts every step from now on. You want to
   watch collapse happen before you fix it.

Benchmark the sparse gather against dense-compute-and-mask. At 32 experts, dense is often faster
on GPU — full-size GEMMs and no gather/scatter overhead beat 2× the FLOPs.

### Phase 3 — AttnRes

Rip out the residual connections. Carry a list of prior outputs; add the per-layer pseudo-queries.

*Gates:*
1. **Uniform-weight test.** Initialize all pseudo-queries to zero. Then `φ(0, k) = 1` for every
   source, so `h_l` must equal the exact mean of all prior sources. Compare against a manual
   mean. This is the single best test in this document — it validates the softmax, the source
   list, and the indexing all at once.
2. **Source count.** Layer `l` must see exactly `l` sources (layers `0..l−1`, where 0 is the
   embedding). Assert it.
3. **Weight entropy.** Log the AttnRes attention distribution per layer. If it collapses to
   one-hot on the immediately preceding layer, you've reinvented the residual stream and gained
   nothing.

### Phase 4 — Quantile Balancing

Add the bias. Sign update first if you like, then QB.

*Gates:*
1. **Causality.** Assert the bias used at step `t` was computed at step `t−1`.
2. **Mean-zero.** Assert `mean(b) ≈ 0` after every update.
3. **Convergence.** Per-expert load within a few percent of `mk/n` after a few hundred steps.
   Compare the sign update and QB on the same seed — QB should equilibrate visibly faster.

### Phase 5 — Scale up

Move to the main config. Change nothing architectural.

---

## 6. Known bugs to fix in the current `KdaAttention`

If you're starting from the draft in `model.py`:

| Line | Bug | Fix |
|---|---|---|
| `proj_width = n_heads*d_head` with `d_head=64, n_heads=8, d_model=256` | internal width is 2× the residual stream, doubling every projection | make `n_heads × d_head == d_model` |
| `b_proj(x).sigmoid() * 2` | β range is (0,2); Eq. 2 says (0,1) | drop the `* 2` |
| `exp(-exp(A_log) * softplus(f))` | Kimi Linear's unbounded decay | use Eq. 5's bounded sigmoid, `g_min = −5` |
| `A_log` shaped `proj_width`, init `log(U(1,16))` | `A_h` is per-head, init 0 | shape `n_heads`, `zeros_` |
| `o_norm(out * F.silu(g))` | norm after the multiply, and wrong nonlinearity | `torch.sigmoid(g) * o_norm(out)` |
| `torch.stack(shared, dim=1)` in `MoE` | shared experts run in parallel and are summed | `sum(...)` |
| `F.softmax` over top-k logits | K3 uses sigmoid over all experts, renormalized over the selected | see §3.4 |
| `ReLU` experts | should be SiTU-GLU | see §3.4 |
| `shared_norm` (dimensioned `d_latent`) | correct tensor, misleading name — it's the norm on `u` | rename `expert_norm` |

What the draft already gets right: the recurrence ordering and the channel-wise decay broadcast
over the key dimension, the low-rank `z` projection with per-head bias, the ShortConv cache, and
a full-rank output gate (`g_proj`), which happens to match K3's change from Kimi Linear.

---

## 7. Training setup

```
precision       bf16 autocast, fp32 master weights
optimizer       Muon (matrices) + AdamW (embeddings, norms, biases)
lr schedule     cosine decay, 1% linear warmup
weight decay    0.1
seq len         2048
batch           ~16 with gradient accumulation
checkpointing   on
embeddings      tied
```

The report ran a dedicated scaling-law search per schedule and found cosine decay beats WSD once
each gets its own hyperparameters — the two have substantially different optimal peak LR and
batch size, so a shared sweep unfairly favors one. Use cosine.

**Token budget.** Chinchilla-optimal for 63M active is ~1.2B tokens, but MoEs want more tokens per
active parameter and K3 retuned its tokens-per-parameter ratio for exactly this reason. Aim for
2–4B. FineWeb-Edu's 10B sample plus a 32K BPE tokenizer is the path of least resistance;
TinyStories for the debug runs.

**Wall clock.** ~8–12 hours at a realistic 15 TFLOP/s effective on a 3090. Budget two to three
days accounting for the Triton path being slower than a tuned kernel.

---

## 8. Silent failure checklist

None of these will crash. All of them cost you loss.

- [ ] `α` bounded below by `e^−5`, not unbounded
- [ ] `A_h` per-head, initialized to 0
- [ ] `β ∈ (0,1)`
- [ ] RMSNorm **before** the KDA output gate
- [ ] Gate nonlinearity is Sigmoid, not SiLU
- [ ] KDA gate has RMSNorm; MLA gate does not
- [ ] No positional encoding anywhere (NoPE)
- [ ] No `x + f(x)` in the backbone — AttnRes replaces the residual
- [ ] AttnRes keys pass through RMSNorm inside `φ`
- [ ] Embedding is source 0 for every layer's AttnRes
- [ ] Router reads full-width `x`, not the latent
- [ ] Router bias affects selection but not the mixture weights
- [ ] RMSNorm between expert aggregate and up-projection
- [ ] Shared experts bypass the latent path
- [ ] QB bias lags one step and is mean-centered

---

## 9. References

- Kimi K3: Open Frontier Intelligence — Moonshot AI technical report, July 2026. Equation
  numbers throughout this document refer to it. Weights: `huggingface.co/moonshotai/Kimi-K3`
- Kimi Linear: An Expressive, Efficient Attention Architecture — arXiv 2510.26692. Origin of KDA
  and the chunkwise form; K3's §2.1.1 defers to it for the UT transform derivation.
- Gated Delta Networks — Yang, Kautz, Hatamizadeh, ICLR 2025. The delta rule plus gating that KDA
  extends with channel-wise decay.
- LatentMoE — arXiv 2601.18089. The latent-space expert factorization.
- DeepSeekMoE — arXiv 2401.06066. The shared/routed expert split.
- DeepSeek-V3 Technical Report — arXiv 2412.19437. Auxiliary-loss-free balancing; the sign update
  QB generalizes.
- flash-linear-attention — `github.com/fla-org/flash-linear-attention`. KDA kernels; PR #691.
- Muon — `kellerjordan.github.io/posts/muon/`