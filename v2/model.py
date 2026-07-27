import torch
from torch import nn
import torch.nn.functional as F
import math


class KdaAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_head):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = d_head

        proj_width = n_heads*d_head 

        self.wq = nn.Linear(d_model, proj_width, bias=False)
        self.wk = nn.Linear(d_model, proj_width, bias=False)
        self.wv = nn.Linear(d_model, proj_width, bias=False)

        self.conv_q = nn.Conv1d(proj_width, proj_width, kernel_size=4, groups=proj_width, bias=False)
        self.conv_k = nn.Conv1d(proj_width, proj_width, kernel_size=4, groups=proj_width, bias=False)
        self.conv_v = nn.Conv1d(proj_width, proj_width, kernel_size=4, groups=proj_width, bias=False)

        self.f_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, proj_width)
        )

        dt = torch.exp(torch.rand(proj_width) * (math.log(1e-1) - math.log(1e-3)) + math.log(1e-3))
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        self.f_proj[1].bias.data.copy_(inv_dt)
        self.A_log = nn.Parameter(torch.log(torch.empty(proj_width).uniform_(1, 16)))

        self.b_proj = nn.Linear(d_model, n_heads, bias=False)

        self.g_proj = nn.Linear(d_model, proj_width, bias=False)
        self.o_norm = nn.RMSNorm(d_head)

        self.wo = nn.Linear(proj_width, d_model, bias=False)
        
    def _causal_conv(self, proj, conv, cache=None):
        if cache is None:
            padded = F.pad(proj, (3, 0))
        else:
            padded = torch.cat([cache, proj], dim=-1)
        new_cache = padded[..., -3:].detach()
        return F.silu(conv(padded)), new_cache

    def forward(self, x, state=None):
        B, T, _ = x.shape
        s_t, q_cache, k_cache, v_cache = state if state is not None else (None,)*4

        q = self.wq(x).transpose(1, 2)
        k = self.wk(x).transpose(1, 2)
        v = self.wv(x).transpose(1, 2)

        q, q_cache = self._causal_conv(q, self.conv_q, q_cache)
        k, k_cache = self._causal_conv(k, self.conv_k, k_cache)
        v, v_cache = self._causal_conv(v, self.conv_v, v_cache)

        q = q.transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        k = k.transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        v = v.transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        f = self.f_proj(x)                              
        a = torch.exp(-torch.exp(self.A_log) * F.softplus(f))
        a = a.reshape(B, T, self.n_heads, self.d_head)
        b = self.b_proj(x).sigmoid() * 2

        if s_t is None:
            s_t = x.new_zeros(B, self.n_heads, self.d_head, self.d_head)

        outs = []

        for t in range(T):
            s_t = torch.mul(a[:, t].reshape(B, self.n_heads,1, self.d_head), s_t)
            v_t = (s_t @ k[:, t].reshape(B, self.n_heads, self.d_head, 1)).reshape(B, self.n_heads, self.d_head)

            err = (v[:, t] - v_t).reshape(B, self.n_heads, self.d_head, 1)

            corr = err @ k[:, t].reshape(B, self.n_heads, 1, self.d_head)
            corr = torch.mul(b[:, t].reshape(B, self.n_heads, 1, 1), corr)

            s_t = s_t + corr

            outs.append((s_t @ q[:, t].reshape(B, self.n_heads, self.d_head, 1)).reshape(B, self.n_heads, self.d_head))

        out = torch.stack(outs, dim=1)
        g = self.g_proj(x).reshape(B, T, self.n_heads, self.d_head)
        out = self.o_norm(out * F.silu(g))
        out = self.wo(out.reshape(B, T, -1))

        return out, (s_t, q_cache, k_cache, v_cache)





attn = KdaAttention(256, 8, 64)

inp = torch.rand(1, 10, 256)

full, _ = attn(inp)                        # T=10 in one shot
state, chunks = None, []
for t in range(10):
    o, state = attn(inp[:, t:t+1], state)  # one token at a time
    chunks.append(o)
step = torch.cat(chunks, dim=1)
print((full - step).abs().max())           # want < 1e-5

