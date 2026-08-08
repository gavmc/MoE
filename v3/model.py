import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from torch.utils.checkpoint import checkpoint

from fla.ops.kda import chunk_kda
from fla.modules import FusedLinearCrossEntropyLoss
from fla.modules.conv import ShortConvolution


torch._dynamo.config.recompile_limit = 64


def situ_glu(x, w_gate, w_up, b1=4.0, b2=25.0):
    g = w_gate(x)
    gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
    up = b2 * torch.tanh(w_up(x) / b2)
    return gate * up

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps)
        return (x * self.weight).to(dt)

class KDA(nn.Module):
    def __init__(self, d_model, n_head, d_k, d_v, g_min=-5, chunk=16, a_rank=None):
        super().__init__()
 
        self.d_model = d_model
        self.n_head = n_head
        self.d_k, self.d_v = d_k, d_v
        self.k_width = n_head * d_k
        self.v_width = n_head * d_v
 
        self.g_min = g_min
        self.chunk = chunk
 
 
        self.q_proj = nn.Linear(d_model, self.k_width, bias=False)
        self.k_proj = nn.Linear(d_model, self.k_width, bias=False)
        self.v_proj = nn.Linear(d_model, self.v_width, bias=False)
 
        #self.q_conv = nn.Conv1d(self.k_width, self.k_width, kernel_size=4, stride=1, groups=self.k_width)
        #self.k_conv = nn.Conv1d(self.k_width, self.k_width, kernel_size=4, stride=1, groups=self.k_width)
        #self.v_conv = nn.Conv1d(self.v_width, self.v_width, kernel_size=4, stride=1, groups=self.v_width)

        self.q_conv = ShortConvolution(self.k_width, kernel_size=4)
        self.k_conv = ShortConvolution(self.k_width, kernel_size=4)
        self.v_conv = ShortConvolution(self.v_width, kernel_size=4)

        self.b_proj = nn.Linear(d_model, n_head, bias=False)
        a_rank = a_rank or d_k
        self.a_proj = nn.Sequential(
            nn.Linear(d_model, a_rank, bias=False),
            nn.Linear(a_rank, self.k_width)      
        )
 
        self.a_log = nn.Parameter(torch.zeros(n_head))
 
        self.o_norm = RMSNorm(d_v)
        self.g_proj = nn.Linear(d_model, self.v_width, bias=False)
        self.o_proj = nn.Linear(self.v_width, d_model, bias=False)
 
    def _chunk(self, q, k, v, beta, g):
        B, T, H, Dk = q.shape
        Dv = v.shape[-1]

        C = self.chunk
        n_chunk = T // C
 
        q = q.reshape(B, n_chunk, C, H, Dk)
        k = k.reshape(B, n_chunk, C, H, Dk)
        v = v.reshape(B, n_chunk, C, H, Dv)
        g = g.reshape(B, n_chunk, C, H, Dk)
        beta = beta.reshape(B, n_chunk, C, H)
 
        G = g.cumsum(dim=2)
 
        i = torch.arange(C, device=q.device)
        causal = (i[:, None] >= i[None, :]).view(1, C, C, 1, 1)
        strict = (i[:, None] >  i[None, :]).view(1, C, C, 1)
        eye = torch.eye(C, device=q.device, dtype=q.dtype)
 
        s = q.new_zeros(B, H, Dk, Dv)
        out = []
 
        for c in range(n_chunk):
            q_c, k_c, v_c = q[:, c], k[:, c], v[:, c]
            b_c, G_c = beta[:, c], G[:, c]
 
            q_decay = q_c * G_c.exp()
            k_read  = k_c * G_c.exp()
 
            decay = (G_c[:, :, None] - G_c[:, None, :])
            decay = decay.masked_fill(~causal, float('-inf')).exp()
 
            kk = torch.einsum('bihd,bjhd,bijhd->bhij', k_c, k_c, decay)
            qk = torch.einsum('bihd,bjhd,bijhd->bhij', q_c, k_c, decay)
 
            b = b_c.permute(0, 2, 1).unsqueeze(-1)
            A = eye + b * kk * strict.permute(0, 3, 1, 2)
 
            rhs = torch.cat([b * v_c.permute(0, 2, 1, 3), b * k_read.permute(0, 2, 1, 3)], dim=-1)
            UW = torch.linalg.solve_triangular(A, rhs, upper=False, unitriangular=True)
            U, W = UW[..., :Dv], UW[..., Dv:]
 
            v_pseudo = U - W @ s
 
            o_c = q_decay.permute(0, 2, 1, 3) @ s + (qk * causal.squeeze(-1).permute(0, 3, 1, 2)) @ v_pseudo
            out.append(o_c.permute(0, 2, 1, 3))
 
            to_end = (G_c[:, -1:] - G_c).exp()
            s = G_c[:, -1].exp().unsqueeze(-1) * s + torch.einsum('bchk,bhcd->bhkd', k_c * to_end, v_pseudo)
 
        return torch.stack(out, dim=1).reshape(B, T, H, Dv)
 
    def forward(self, x):
        B, T, _ = x.shape
        H, Dk, Dv = self.n_head, self.d_k, self.d_v
 
        q, _ = self.q_conv(self.q_proj(x))
        k, _ = self.k_conv(self.k_proj(x))
        v, _ = self.v_conv(self.v_proj(x))

        q = q.reshape(B, T, H, Dk)
        k = k.reshape(B, T, H, Dk)
        v = v.reshape(B, T, H, Dv)
 
        beta = self.b_proj(x).sigmoid()
        z = self.a_proj(x).reshape(B, T, H, Dk)
 
        g = self.g_min * F.sigmoid(self.a_log.exp().view(1, 1, -1, 1) * z)

        #o = self._chunk(q, k, v, beta, g)
        o, _ = chunk_kda(q, k, v, g, beta, scale=1.0, use_qk_l2norm_in_kernel=True)
        o = self.o_norm(o).reshape(B, T, self.v_width)
        return self.o_proj(self.g_proj(x).sigmoid() * o)

class MLA(nn.Module):
    def __init__(self, d_model, d_c, n_head, d_head):
        super().__init__()


        self.d_model = d_model
        self.d_c = d_c
        self.d_head = d_head
        self.n_head = n_head
        self.width = n_head * d_head

        self.q_proj = nn.Linear(d_model, self.width, bias=False)
        self.c_proj = nn.Linear(d_model, d_c, bias=False)
        self.c_norm = RMSNorm(d_c)

        self.k_proj = nn.Linear(d_c, self.width, bias=False)
        self.v_proj = nn.Linear(d_c, self.width, bias=False)

        self.g_proj = nn.Linear(d_model, self.width, bias=False)
        self.o_proj = nn.Linear(self.width, d_model, bias=False)

    def forward(self, x):
        B, T, _ = x.shape
        H, D = self.n_head, self.d_head

        c = self.c_norm(self.c_proj(x))

        q = self.q_proj(x).reshape(B, T, H, D).transpose(1, 2)
        k = self.k_proj(c).reshape(B, T, H, D).transpose(1, 2)
        v = self.v_proj(c).reshape(B, T, H, D).transpose(1, 2)

        #scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)
        #mask = torch.ones(T, T, dtype=torch.bool, device=x.device).triu(1)
        #scores = scores.masked_fill(mask, float('-inf'))
        #attn = F.softmax(scores, dim=-1) @ v
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(B, T, self.width)

        return self.o_proj(self.g_proj(x).sigmoid() * attn)

class FFN(nn.Module):
    def __init__(self, d_model, d_ff, b1=4.0, b2=25.0):
        super().__init__()
        self.d_model, self.d_ff = d_model, d_ff
        self.b1, self.b2 = b1, b2

        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up   = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(situ_glu(x, self.w_gate, self.w_up, self.b1, self.b2))

class MoE(nn.Module):
    def __init__(self, d_model, d_latent, d_expert, d_shared,
                 n_experts=16, top_k=2, bias_lr=1e-3, cap_factor=1.5):
        super().__init__()
        self.n_experts, self.top_k, self.bias_lr = n_experts, top_k, bias_lr
        self.cap_factor = cap_factor

        self.shared = FFN(d_model, d_shared)              
        self.w_down = nn.Linear(d_model, d_latent, bias=False)
        self.w_up   = nn.Linear(d_latent, d_model, bias=False)
        self.u_norm = RMSNorm(d_latent) 

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.register_buffer('r_bias', torch.zeros(n_experts))

        self.e_gate = nn.Parameter(torch.empty(n_experts, d_latent, d_expert))
        self.e_up   = nn.Parameter(torch.empty(n_experts, d_latent, d_expert))
        self.e_down = nn.Parameter(torch.empty(n_experts, d_expert, d_latent))

        for w in (self.e_gate, self.e_up, self.e_down):
            nn.init.normal_(w, std=0.02)

    def _route_experts(self, z, idx, p):
        N, l = z.shape
        E, k = self.n_experts, self.top_k
        C = max(1, int(N * k / E * self.cap_factor))

        flat_e = idx.reshape(-1)                 
        order = flat_e.argsort(stable=True)
        tok = order // k                                    
        sorted_e = flat_e[order]
        counts = torch.bincount(flat_e, minlength=E)
        offset = counts.cumsum(0) - counts              
        rank = torch.arange(N * k, device=z.device) - offset[sorted_e]

        in_cap = rank < C
        dest = sorted_e * (C + 1) + rank.clamp(max=C)        

        buf = z.new_zeros(E * (C + 1), l)
        buf.index_copy_(0, dest, z[tok])
        buf = buf.view(E, C + 1, l)

        g = torch.bmm(buf, self.e_gate)
        h = (4.0 * torch.tanh(g / 4.0) * g.sigmoid()) * \
            (25.0 * torch.tanh(torch.bmm(buf, self.e_up) / 25.0))
        out = torch.bmm(h, self.e_down).reshape(E * (C + 1), l)

        w = p.reshape(-1)[order] * in_cap            
        u = z.new_zeros(N, l)
        u.index_add_(0, tok, (out[dest] * w[:, None]).to(u.dtype))

        self.last_counts = counts.detach()
        return u, counts

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)

        s = torch.sigmoid(self.router(flat))     
        _, idx = torch.topk(s + self.r_bias, self.top_k)  
        p = s.gather(-1, idx)
        p = p / p.sum(-1, keepdim=True) 

        u, counts = self._route_experts(self.w_down(flat), idx, p)

        y = self.shared(x) + self.w_up(self.u_norm(u)).reshape(B, T, D)
        return y

    @torch.no_grad()  # <--------------------------- NEED TO REMEMBER TO CALL IN OPTIMIZER STEP
    def update_bias(self):
        c = self.last_counts.float()
        err = c / c.sum() - 1.0 / self.n_experts
        self.r_bias -= self.bias_lr * err.sign()

class AttnRes(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, outputs):
        scores = torch.stack([
            (v @ self.w) * torch.rsqrt(v.square().mean(-1) + self.eps)
            for v in outputs], dim=-1)
        alpha = scores.softmax(dim=-1)
        out = outputs[0] * alpha[..., 0:1]
        for i in range(1, len(outputs)):
            out = out + outputs[i] * alpha[..., i:i + 1]
        return out


class MiniK3(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_k, d_v, d_c, d_latent, d_expert, d_shared, mix_ratio=4, layers=16, vocab_size=32_000, tie_embeddings=True, use_ckpt=False, attnres_block=6):
        super().__init__()

        self.layers = layers
        self.use_ckpt = use_ckpt
        self.attnres_block = attnres_block  
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.mixers = nn.ModuleList(
            MLA(d_model, d_c, n_head, d_head) if (i + 1) % mix_ratio == 0
            else KDA(d_model, n_head, d_k, d_v)
            for i in range(layers)
        )
        self.moe = nn.ModuleList(MoE(d_model, d_latent, d_expert, d_shared) if i != 0 else FFN(d_model, d_shared) for i in range(layers))
        self.attn_res = nn.ModuleList(AttnRes(d_model) for _ in range(2 * layers + 1))

        self.norm_mix = nn.ModuleList(RMSNorm(d_model) for _ in range(layers))
        self.norm_ff = nn.ModuleList(RMSNorm(d_model) for _ in range(layers))

        self.norm_out = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.embedding.weight

        self.loss = FusedLinearCrossEntropyLoss(num_chunks=8)

        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, tgt=None):
        ckpt = (lambda f, t: checkpoint(f, t, use_reentrant=False)) if self.use_ckpt else (lambda f, t: f(t))

        S = self.attnres_block
        blocks = [self.embedding(idx)]
        partial, filled = None, 0

        for l in range(self.layers):
            for j, f in ((2*l,     lambda t, m=self.mixers[l], n=self.norm_mix[l]: m(n(t))),
                         (2*l + 1, lambda t, m=self.moe[l],    n=self.norm_ff[l]:  m(n(t)))):
                srcs = blocks if partial is None else blocks + [partial]
                h = self.attn_res[j](srcs)
                out = ckpt(f, h)
                partial = out if partial is None else partial + out
                filled += 1
                if filled == S:
                    blocks.append(partial)
                    partial, filled = None, 0

        x = self.attn_res[-1](blocks if partial is None else blocks + [partial])
        x = self.norm_out(x)

        if tgt is None:
            return self.head(x), None

        loss = self.loss(
            x[:, :-1].contiguous(), 
            tgt[:, 1:].contiguous(),
            self.head.weight,
        )

        return None, loss

        

