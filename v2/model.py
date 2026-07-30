import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
import math


def emit(out, values, keys):
    values.append(out)
    keys.append(F.rms_norm(out, out.shape[-1:]))

def situ_glu(g, u, b1, b2):
    gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
    return gate * (b2 * torch.tanh(u / b2))

def qb_step(model):
    for m in model.modules():
        if isinstance(m, MoE):
            m.qb_update()

@torch.no_grad()
def moe_load_stats(model):
    stats = []
    for m in model.modules():
        if isinstance(m, MoE) and m.load is not None:
            load = m.load.float()
            target = load.sum() / m.n_routed
            stats.append(dict(
                max_dev=((load - target).abs().max() / target).item(),
                dead=int((load == 0).sum()),
                dropped=(m.n_dropped / load.sum()).item() if m.n_dropped is not None else 0.0,
                bias_spread=(m.router_bias.max() - m.router_bias.min()).item(),
            ))
    return stats


class KdaAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head, conv_size=4, g_min=-5.0):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head
        self.conv_size = conv_size
        self.g_min = g_min

        proj_width = n_heads*d_head 

        self.wq = nn.Linear(d_model, proj_width, bias=False)
        self.wk = nn.Linear(d_model, proj_width, bias=False)
        self.wv = nn.Linear(d_model, proj_width, bias=False)

        self.conv_q = nn.Conv1d(proj_width, proj_width, kernel_size=conv_size, groups=proj_width, bias=False)
        self.conv_k = nn.Conv1d(proj_width, proj_width, kernel_size=conv_size, groups=proj_width, bias=False)
        self.conv_v = nn.Conv1d(proj_width, proj_width, kernel_size=conv_size, groups=proj_width, bias=False)

        self.a_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, proj_width)
        )

        dt = torch.exp(torch.rand(proj_width) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3))
        self.a_proj[1].bias.data.copy_(dt + torch.log(-torch.expm1(-dt)))
        self.A_log = nn.Parameter(torch.zeros(n_heads))

        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        self.g_proj = nn.Linear(d_model, proj_width, bias=False) 
        self.o_norm = nn.RMSNorm(d_head)
        self.wo = nn.Linear(proj_width, d_model, bias=False)
        
    def _causal_conv(self, proj, conv, cache=None):
        pad = self.conv_size - 1
        if cache is None:
            padded = F.pad(proj, (pad, 0))
        else:
            padded = torch.cat([cache, proj], dim=-1)
        return F.silu(conv(padded)), padded[..., -pad:]

    def _recurrent(self, q, k, v, alpha, beta, s):
        outs = []
        for t in range(q.shape[1]):
            s = alpha[:, t].unsqueeze(-1) * s
            v_hat = torch.einsum("bhk,bhkv->bhv", k[:, t], s)
            err = (v[:, t] - v_hat).unsqueeze(-2)              
            s = s + beta[:, t, :, None, None] * (k[:, t].unsqueeze(-1) @ err)
            outs.append(torch.einsum("bhk,bhkv->bhv", q[:, t], s))
        return torch.stack(outs, 1), s

    def _chunk(self, q, k, v, g, beta, s, C=16):
        B, T, H, D = q.shape
        pad = (-T) % C
        if pad:
            q, k, v = (F.pad(z, (0, 0, 0, 0, 0, pad)) for z in (q, k, v))
            g = F.pad(g, (0, 0, 0, 0, 0, pad))      
            beta = F.pad(beta, (0, 0, 0, pad))
        N = (T + pad) // C

        fold = lambda z: z.reshape(B, N, C, H, -1).permute(0, 3, 1, 2, 4) 
        q, k, v, g = fold(q), fold(k), fold(v), fold(g)
        beta = beta.reshape(B, N, C, H).permute(0, 3, 1, 2).unsqueeze(-1)  

        g = g.cumsum(-2)                   
        k_hat  = k * (-g).exp()        
        k_star = k * g.exp()           
        q_star = q * g.exp()
        g_last = g[..., -1:, :]
        gamma_C = g_last.exp()          
        k_tilde = k * (g_last - g).exp()      

        A = beta * (k_star @ k_hat.transpose(-1, -2)).tril(-1)
        I = torch.eye(C, device=q.device, dtype=q.dtype)
        U = torch.linalg.solve_triangular(I + A, beta * v,      upper=False, unitriangular=True)
        W = torch.linalg.solve_triangular(I + A, beta * k_star, upper=False, unitriangular=True)

        attn = (q_star @ k_hat.transpose(-1, -2)).tril()  

        outs = []
        for n in range(N):                          
            z = U[:, :, n] - W[:, :, n] @ s                           
            outs.append(q_star[:, :, n] @ s + attn[:, :, n] @ z)     
            s = gamma_C[:, :, n].transpose(-1, -2) * s \
                + k_tilde[:, :, n].transpose(-1, -2) @ z
        o = torch.stack(outs, 2).reshape(B, H, N * C, D).transpose(1, 2)
        return o[:, :T], s



    def forward(self, x, state=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head
        s, q_cache, k_cache, v_cache = state if state is not None else (None,)*4

        q = self.wq(x).transpose(1, 2)
        k = self.wk(x).transpose(1, 2)
        v = self.wv(x).transpose(1, 2)

        q, q_cache = self._causal_conv(q, self.conv_q, q_cache)
        k, k_cache = self._causal_conv(k, self.conv_k, k_cache)
        v, v_cache = self._causal_conv(v, self.conv_v, v_cache)

        q = q.transpose(1, 2).reshape(B, T, H, D)
        k = k.transpose(1, 2).reshape(B, T, H, D)
        v = v.transpose(1, 2).reshape(B, T, H, D)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        z = self.a_proj(x).reshape(B, T, H, D)

        g = self.g_min * torch.sigmoid(self.A_log.exp().view(1, 1, H, 1) * z)
        beta = self.b_proj(x).sigmoid()
        if s is None:
            s = x.new_zeros(B, H, D, D)
        if T < 32:
            o, s = self._recurrent(q, k, v, g.exp(), beta, s)
        else:
            o, s = self._chunk(q, k, v, g, beta, s)

        gate = self.g_proj(x).reshape(B, T, H, D)
        o = self.o_norm(o) * torch.sigmoid(gate)

        return self.wo(o.reshape(B, T, -1)), (s, q_cache, k_cache, v_cache)

class GatedMla(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_kv=128):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        proj_width = n_heads * d_head

        self.wq = nn.Linear(d_model, proj_width, bias=False)

        self.kv_down = nn.Linear(d_model, d_kv, bias=False)
        self.kv_norm = nn.RMSNorm(d_kv)
        self.k_up = nn.Linear(d_kv, proj_width, bias=False)
        self.v_up = nn.Linear(d_kv, proj_width, bias=False)

        self.g_proj = nn.Linear(d_model, proj_width, bias=False)
        self.wo = nn.Linear(proj_width, d_model, bias=False)

    def forward(self, x, cache=None):
        B, T, _ = x.shape
        H, D = self.n_heads, self.d_head

        c = self.kv_norm(self.kv_down(x))
        if cache is not None:
            c = torch.cat([cache, c], dim=1)
        S = c.shape[1]

        q = self.wq(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_up(c).reshape(B, S, H, D).transpose(1, 2)
        v = self.v_up(c).reshape(B, S, H, D).transpose(1, 2)

        if cache is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pos = torch.arange(S - T, S, device=x.device).unsqueeze(-1)
            mask = pos >= torch.arange(S, device=x.device)
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        o = o.transpose(1, 2).reshape(B, T, H * D)

        return self.wo(torch.sigmoid(self.g_proj(x)) * o), c

class Expert(nn.Module):
    def __init__(self, d_in, d_ff, b1=4.0, b2=25.0):
        super().__init__()
        self.w_gate = nn.Linear(d_in, d_ff, bias=False)
        self.w_up = nn.Linear(d_in, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_in, bias=False)
        self.b1, self.b2 = b1, b2
 
    def forward(self, x):
        return self.w_down(situ_glu(self.w_gate(x), self.w_up(x), self.b1, self.b2))

class MoE(nn.Module):
    def __init__(self, d_model, d_latent, d_ff_e, d_ff_s, n_shared, n_routed, top_k,
                 b1=4.0, b2=25.0, capacity_factor=1.25, qb_bins=1000, qb_ema=0.0):
        super().__init__()

        self.n_routed, self.top_k = n_routed, top_k
        self.b1, self.b2 = b1, b2
        self.capacity_factor = capacity_factor
        self.n_dropped = None
        self.load = None
        self.qb_bins, self.qb_enabled, self._qb_seen = qb_bins, True, False
        self.qb_ema = qb_ema

        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("router_bias", torch.zeros(n_routed))

        self.register_buffer("qb_hist", torch.zeros(n_routed, qb_bins), persistent=False)
        self.register_buffer("qb_tokens", torch.zeros(()), persistent=False)
        self.register_buffer("qb_lo", torch.tensor(-1.0), persistent=False)
        self.register_buffer("qb_w", torch.tensor(2.0 / qb_bins), persistent=False)

        self.down_proj = nn.Linear(d_model, d_latent, bias=False)
        self.up_proj = nn.Linear(d_latent, d_model, bias=False)
        self.expert_norm = nn.RMSNorm(d_latent)

        self.w_gate = nn.Parameter(torch.empty(n_routed, d_latent, d_ff_e))
        self.w_up = nn.Parameter(torch.empty(n_routed, d_latent, d_ff_e))
        self.w_down = nn.Parameter(torch.empty(n_routed, d_ff_e, d_latent))

        nn.init.normal_(self.w_gate, std=0.02)
        nn.init.normal_(self.w_up, std=0.02)
        nn.init.normal_(self.w_down, std=0.02)

        self.shared = nn.ModuleList(Expert(d_model, d_ff_s, b1, b2) for _ in range(n_shared))        


    def _dispatch_loop(self, latent, idx, w):
        N, _ = latent.shape
        out = torch.zeros_like(latent)

        flat_e = idx.reshape(-1)
        flat_w = w.reshape(-1)
        flat_t = torch.arange(N, device=latent.device).repeat_interleave(self.top_k)

        order = flat_e.argsort()
        counts = torch.bincount(flat_e, minlength=self.n_routed)
        bounds = counts.cumsum(0).tolist()

        start = 0
        for e, end in enumerate(bounds):
            if end == start:
                continue
            sel = order[start:end]
            start = end
            tok = flat_t[sel]
            z = latent[tok]
            h = situ_glu(z @ self.w_gate[e], z @ self.w_up[e], self.b1, self.b2)
            y = (h @ self.w_down[e]) * flat_w[sel].unsqueeze(-1)
            out.index_add_(0, tok, y)
        return out

    def dispatch(self, latent, idx, w):
        N, d_latent = latent.shape
        E, K = self.n_routed, self.top_k
        cap = math.ceil(self.capacity_factor * N * K / E)

        flat_e = idx.reshape(-1)
        flat_w = w.reshape(-1)
        flat_t = torch.arange(N, device=latent.device).repeat_interleave(K)

        order = flat_e.argsort(stable=True)
        sorted_e = flat_e[order]
        counts = torch.bincount(flat_e, minlength=E)
        offsets = counts.cumsum(0) - counts                    
        rank = torch.arange(flat_e.numel(), device=latent.device) - offsets[sorted_e]

        keep = rank < cap
        slot = (sorted_e * cap + rank)[keep]
        src = flat_t[order][keep]
        self.n_dropped = (~keep).sum().detach()
        self.load = counts.detach()

        buf = latent.new_zeros(E * cap, d_latent)
        buf[slot] = latent[src]
        buf = buf.view(E, cap, d_latent)

        h = situ_glu(buf @ self.w_gate, buf @ self.w_up, self.b1, self.b2)
        y = (h @ self.w_down).reshape(E * cap, d_latent)

        out = torch.zeros_like(latent)
        out.index_add_(0, src, y[slot] * flat_w[order][keep].unsqueeze(-1))
        return out

    def route(self, xf):
        scores = self.router(xf).sigmoid()
        top = torch.topk(scores + self.router_bias, self.top_k + 1, dim=-1)
        idx, cutoff = top.indices[:, :self.top_k], top.values[:, self.top_k]
        w = scores.gather(-1, idx)
        return scores, idx, w / w.sum(-1, keepdim=True), cutoff

    @torch.no_grad()
    def qb_reset(self):
        lo, hi = self.router_bias.min(), self.router_bias.max()
        self.qb_lo.copy_(lo - 1.0)
        self.qb_w.copy_((hi - lo + 2.0) / self.qb_bins)
        self.qb_hist.zero_()
        self.qb_tokens.zero_()
        self._qb_seen = False

    @torch.no_grad()
    def qb_accumulate(self, scores, cutoff):
        r = cutoff.unsqueeze(-1) - scores
        b = ((r - self.qb_lo) / self.qb_w).long().clamp_(0, self.qb_bins - 1)
        flat = b + torch.arange(self.n_routed, device=b.device) * self.qb_bins
        self.qb_hist.view(-1).index_add_(
            0, flat.reshape(-1), torch.ones(flat.numel(), device=b.device, dtype=self.qb_hist.dtype))
        self.qb_tokens += scores.shape[0]
        self._qb_seen = True

    @torch.no_grad()
    def qb_update(self):
        if not (self.qb_enabled and self._qb_seen):
            return
        hist, m = self.qb_hist, self.qb_tokens
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(hist)
            dist.all_reduce(m)

        q = m * self.top_k / self.n_routed
        cum = hist.cumsum(-1)
        i = (cum < q.ceil()).sum(-1, keepdim=True).clamp_(max=self.qb_bins - 1)
        c = cum.gather(-1, i) - hist.gather(-1, i)
        h = hist.gather(-1, i).clamp_(min=1.0)
        frac = ((q - c) / h).clamp_(0.0, 1.0)
        b_hat = self.qb_lo + (i.squeeze(-1) + frac.squeeze(-1)) * self.qb_w

        b_new = b_hat - b_hat.mean()
        if self.qb_ema > 0.0:
            b_new = self.qb_ema * self.router_bias + (1 - self.qb_ema) * b_new
        self.router_bias.copy_(b_new)
        self.qb_reset()

    def forward(self, x, return_counts=False):
        B, T, D = x.shape
        xf = x.reshape(B * T, D)

        shared_out = sum(e(xf) for e in self.shared)
        scores, idx, w, cutoff = self.route(xf)
        if self.training and self.qb_enabled:
            self.qb_accumulate(scores.detach(), cutoff.detach())
        latent = self.down_proj(xf)

        u = self.dispatch(latent, idx, w)

        y = shared_out + self.up_proj(self.expert_norm(u))
        y = y.reshape(B, T, D)

        if return_counts:
            return y, self.load
        return y

class AttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()

        self.w = nn.Parameter(torch.zeros(d_model))

    def forward(self, values, keys):
        v = torch.stack(values, dim=2)         
        k = torch.stack(keys, dim=2)             
        a = (k @ self.w).softmax(-1)                 
        return torch.einsum("btl,btld->btd", a, v), a

class Block(nn.Module):
    def __init__(self, d_model, n_heads, d_head, d_kv, moe_kw, n_kda=3):
        super().__init__()
        self.attn = nn.ModuleList(
            [KdaAttention(d_model, n_heads, d_head) for _ in range(n_kda)]
            + [GatedMla(d_model, n_heads, d_head, d_kv)]
        )
        self.moe = nn.ModuleList(MoE(d_model, **moe_kw) for _ in range(n_kda + 1))
        self.res_attn = nn.ModuleList(AttnRes(d_model) for _ in range(n_kda + 1))
        self.res_moe = nn.ModuleList(AttnRes(d_model) for _ in range(n_kda + 1))
        self.key_norm = nn.RMSNorm(d_model, elementwise_affine=False)

    def forward(self, values, keys, state=None):
        if state is None:
            state = [None] * len(self.attn)
        new_state = []

        for i in range(len(self.attn)):
            h, _ = self.res_attn[i](values, keys)
            out, st = self.attn[i](h, state[i])
            new_state.append(st)
            emit(out, values, keys)

            h, _ = self.res_moe[i](values, keys)
            out = self.moe[i](h)
            emit(out, values, keys)
        return new_state

class MiniK3(nn.Module):
    def __init__(self, vocab_size=32000, d_model=512, n_heads=8, d_head=64, d_kv=128,
                 d_latent=256, d_ff_e=224, d_ff_s=448,
                 n_shared=2, n_routed=32, top_k=4, n_blocks=3, n_kda=3):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

        moe_kw = dict(d_latent=d_latent, d_ff_e=d_ff_e, d_ff_s=d_ff_s,
                      n_shared=n_shared, n_routed=n_routed, top_k=top_k)

        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, d_head, d_kv, moe_kw, n_kda) for _ in range(n_blocks)]
            + [Block(d_model, n_heads, d_head, d_kv, moe_kw, n_kda=0)]  
        )

        self.res_out = AttnRes(d_model)
        self.out_norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight        

    def forward(self, ids, state=None, use_cache=False):
        h = self.embedding(ids)
        values, keys = [h], [F.rms_norm(h, h.shape[-1:])]

        if state is None:
            state = [None] * len(self.blocks)
        new_state = []

        for block, st in zip(self.blocks, state):
            new_state.append(block(values, keys, st))

        h, _ = self.res_out(values, keys)
        logits = self.lm_head(self.out_norm(h))

        return (logits, new_state) if use_cache else logits

    @torch.no_grad()
    def generate(self, ids, max_new_tokens, temperature=1.0):
        logits, state = self(ids, use_cache=True)

        for _ in range(max_new_tokens):
            probs = (logits[:, -1] / temperature).softmax(-1)
            nxt = torch.multinomial(probs, 1)
            ids = torch.cat([ids, nxt], dim=1)
            logits, state = self(nxt, state, use_cache=True)

        return ids


if __name__ == "__main__":
    model = MiniK3()

    num = 0

    for p in model.parameters():
        num += p.numel() if p.requires_grad else 0


    print(num)

    tmp = torch.randint(0, 32000, (1, 64))
    model(tmp)
