import torch
from torch import nn
import torch.nn.functional as F
import math


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
        alpha = (self.g_min * torch.sigmoid(self.A_log.exp().view(1, 1, H, 1) * z)).exp()

        beta = self.b_proj(x).sigmoid()

        if s is None:
            s = x.new_zeros(B, H, D, D)

        outs = []
        for t in range(T):
            k_t = k[:, t].reshape(B, H, D, 1)
            s = s * alpha[:, t].reshape(B, H, 1, D)

            v_hat = (s @ k_t).reshape(B, H, D)

            err = (v[:, t] - v_hat).reshape(B, H, D, 1, ) 
            s = s + beta[:, t].reshape(B, H, 1, 1) * (err @ k_t.transpose(-1, -2))

            outs.append((s @ q[:, t].reshape(B, H, D, 1)).reshape(B, H, D))

        o = torch.stack(outs, dim=1)
        gate = self.g_proj(x).reshape(B, T, H, D)
        o = self.o_norm(o) * torch.sigmoid(gate)

        return self.wo(o.reshape(B, T, -1)), (s, q_cache, k_cache, v_cache)

def situ_glu(g, u, b1, b2):
    gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
    return gate * (b2 * torch.tanh(u / b2))

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
    def __init__(self, d_model, d_latent, d_ff_e, d_ff_s, n_shared, n_routed, top_k, b1=4.0, b2=25.0):
        super().__init__()

        self.n_routed, self.top_k = n_routed, top_k
        self.b1, self.b2 = b1, b2

        self.router = nn.Linear(d_model, n_routed, bias=False)
        self.register_buffer("router_bias", torch.zeros(n_routed))

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


    def dispatch(self, latent, idx, w):
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

    def route(self, xf):
        scores = self.router(xf).sigmoid()
        _, idx = torch.topk(scores + self.router_bias, self.top_k, dim=-1)
        w = scores.gather(-1, idx)
        return scores, idx, w / w.sum(-1, keepdim=True)

    def forward(self, x, return_counts=False):
        B, T, D = x.shape
        xf = x.reshape(B * T, D)
 
        shared_out = sum(e(xf) for e in self.shared)        
        _, idx, w = self.route(xf)                    
        latent = self.down_proj(xf)                          
 
        u = self.dispatch(latent, idx, w)
 
        y = shared_out + self.up_proj(self.expert_norm(u))  
        y = y.reshape(B, T, D)
 
        if return_counts:
            counts = torch.bincount(idx.reshape(-1), minlength=self.n_routed)
            return y, counts
        return y
        

        

 
