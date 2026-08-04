"""Parameter groups + Muon (train.md §3).

The one change from the stock Muon implementation is the normalization inside
Newton-Schulz: `X.norm(dim=(-2, -1), keepdim=True)` instead of a global `X.norm()`.
That is what makes the 3D expert stacks orthogonalize per expert instead of
coupling all 32 into a single update direction.
"""

import torch
from torch import nn


def newtonschulz(G, steps=5, eps=1e-7):
    """Orthogonalize the last two dims of G. Broadcasts over any leading dims."""
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    # per-slice, not global — the whole point (§3)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True,
                 ns_steps=5, weight_decay=0.1):
        super().__init__(list(params), dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                            ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, mom, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(g)
                buf = st["m"]
                buf.lerp_(g, 1.0 - mom)
                u = g.lerp(buf, mom) if group["nesterov"] else buf

                u = newtonschulz(u, group["ns_steps"]).to(p.dtype)
                # shape-corrected step size; matches the stock implementation
                scale = max(1.0, p.size(-2) / p.size(-1)) ** 0.5

                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * scale)


def _owners(model):
    """param id -> (owning module, dotted name).

    First writer wins, and nn.Embedding always wins outright: `lm_head.weight` IS
    `embedding.weight` (one object, tied), and nn.Linear comes later in module order.
    Letting it overwrite would classify the tied embedding as a plain matrix and send
    16.4M parameters to Muon instead of AdamW.
    """
    out = {}
    for mname, m in model.named_modules():
        for pname, p in m.named_parameters(recurse=False):
            key, val = id(p), (m, f"{mname}.{pname}" if mname else pname)
            if key not in out or isinstance(m, nn.Embedding):
                out[key] = val
    return out


def split_params(model, router_on_muon=True):
    """Muon gets 2D+ matrices; everything else goes to AdamW (§3)."""
    owners = _owners(model)
    muon, adamw = [], []
    table = []

    for p in model.parameters():
        if not p.requires_grad:
            continue
        mod, name = owners[id(p)]

        if p.ndim < 2:
            why = "1D"
        elif isinstance(mod, nn.Embedding):
            why = "embedding (tied)"
        elif isinstance(mod, nn.Conv1d):
            # depthwise: weight is (channels, 1, kernel) — 512 independent length-4
            # filters, not a matrix. ndim >= 2 so the §3 assertion would not catch it.
            why = "depthwise conv"
        elif name.endswith("router.weight") and not router_on_muon:
            why = "router (judgment call)"
        else:
            why = "matrix"

        (muon if why == "matrix" else adamw).append(p)
        table.append((name, tuple(p.shape), "muon" if why == "matrix" else "adamw", why))

    return muon, adamw, table


def assert_groups(model, muon, adamw):
    """The three startup assertions from §3. Compare by id, never by value."""
    all_ids = set(map(id, model.parameters()))
    m_ids, a_ids = set(map(id, muon)), set(map(id, adamw))

    missing = all_ids - (m_ids | a_ids)
    assert not missing, f"{len(missing)} parameters in no optimizer group"
    assert m_ids | a_ids == all_ids, "optimizer groups cover params outside the model"
    assert not (m_ids & a_ids), f"{len(m_ids & a_ids)} parameters in both groups"
    assert all(p.ndim >= 2 for p in muon), "Muon got a 1D parameter"

    # buffers must be invisible to both (§3, §11)
    buf_ids = {id(b) for b in model.buffers()}
    assert not (buf_ids & (m_ids | a_ids)), "a buffer landed in an optimizer group"


def print_table(table, muon, adamw):
    print(f"{'parameter':<52} {'shape':>22}  {'opt':<6} why")
    print("-" * 104)
    for name, shape, opt, why in table:
        print(f"{name:<52} {str(shape):>22}  {opt:<6} {why}")
    n_m = sum(p.numel() for p in muon)
    n_a = sum(p.numel() for p in adamw)
    print("-" * 104)
    print(f"muon   {len(muon):>4} tensors  {n_m/1e6:8.2f}M params")
    print(f"adamw  {len(adamw):>4} tensors  {n_a/1e6:8.2f}M params")
    print(f"total  {len(muon)+len(adamw):>4} tensors  {(n_m+n_a)/1e6:8.2f}M params")


def build_optimizers(model, cfg):
    muon, adamw, table = split_params(model, cfg.router_on_muon)
    assert_groups(model, muon, adamw)
    opt_m = Muon(muon, lr=cfg.lr_muon, momentum=cfg.muon_momentum,
                 ns_steps=cfg.muon_ns_steps, weight_decay=cfg.weight_decay)
    opt_a = torch.optim.AdamW(adamw, lr=cfg.lr_adamw, betas=tuple(cfg.adamw_betas),
                              weight_decay=0.0)  # §3: wd 0 on 1D / embeddings
    return opt_m, opt_a, table
