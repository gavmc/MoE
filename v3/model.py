import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from fla.ops.kda import chunk_kda


def swish(x):
    return x * torch.sigmoid(x)

def situ_glu(x, w_gate, w_up, b1=4.0, b2=25.0):
    g = w_gate(x)
    gate = b1 * torch.tanh(g / b1) * torch.sigmoid(g)
    up = b2 * torch.tanh(w_up(x) / b2)
    return gate * up

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
 
        self.q_conv = nn.Conv1d(self.k_width, self.k_width, kernel_size=4, stride=1, groups=self.k_width)
        self.k_conv = nn.Conv1d(self.k_width, self.k_width, kernel_size=4, stride=1, groups=self.k_width)
        self.v_conv = nn.Conv1d(self.v_width, self.v_width, kernel_size=4, stride=1, groups=self.v_width)
 
        self.b_proj = nn.Linear(d_model, n_head, bias=False)
        a_rank = a_rank or d_k
        self.a_proj = nn.Sequential(
            nn.Linear(d_model, a_rank, bias=False),
            nn.Linear(a_rank, self.k_width)      
        )
 
        self.a_log = nn.Parameter(torch.zeros(n_head))
 
        self.o_norm = nn.RMSNorm(d_v)
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
 
        q = self.q_proj(x).permute(0, 2, 1)
        k = self.k_proj(x).permute(0, 2, 1)
        v = self.v_proj(x).permute(0, 2, 1)
 
        q = self.q_conv(F.pad(q, (3, 0))).permute(0, 2, 1).reshape(B, T, H, Dk)
        k = self.k_conv(F.pad(k, (3, 0))).permute(0, 2, 1).reshape(B, T, H, Dk)
        v = self.v_conv(F.pad(v, (3, 0))).permute(0, 2, 1).reshape(B, T, H, Dv)
 
        q = F.normalize(swish(q), p=2, dim=-1)
        k = F.normalize(swish(k), p=2, dim=-1)
        v = swish(v)
 
        beta = self.b_proj(x).sigmoid()
        z = self.a_proj(x).reshape(B, T, H, Dk)
 
        g = self.g_min * F.sigmoid(self.a_log.exp().view(1, 1, -1, 1) * z)

        #o = self._chunk(q, k, v, beta, g)
        o, _ = chunk_kda(q, k, v, g, beta, scale=1.0)
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
        self.c_norm = nn.RMSNorm(d_c)

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

class AttnRes(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(d_model))
        self.k_norm = nn.RMSNorm(d_model, elementwise_affine=False)

    def forward(self, outputs):             
        V = torch.stack(outputs, dim=2)    
        K = self.k_norm(V)                   
        scores = K @ self.w                 
        alpha = scores.softmax(dim=-1)       
        return (alpha.unsqueeze(-1) * V).sum(dim=2) 


class MiniK3(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_k, d_v, d_c, d_ff, mix_ratio=4, layers=16, vocab_size=32_000, tie_embeddings=True):
        super().__init__()

        self.layers = layers
        self.embedding = nn.Embedding(vocab_size, d_model)

        self.mixers = nn.ModuleList(
            MLA(d_model, d_c, n_head, d_head) if (i + 1) % 4 == 0
            else KDA(d_model, n_head, d_k, d_v)
            for i in range(layers)
        )
        self.ffn = nn.ModuleList(FFN(d_model, d_ff) for _ in range(layers))
        self.attn_res = nn.ModuleList(AttnRes(d_model) for _ in range(2 * layers + 1))

        self.norm_mix = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(layers))
        self.norm_ff = nn.ModuleList(nn.RMSNorm(d_model) for _ in range(layers))

        self.norm_out = nn.RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.embedding.weight

            self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, tgt=None):
        pool = [self.embedding(idx)]

        for l in range(self.layers):
            h = self.attn_res[2*l](pool)
            pool.append(self.mixers[l](self.norm_mix[l](h)))

            h = self.attn_res[2*l + 1](pool)
            pool.append(self.ffn[l](self.norm_mix[l](h)))

        x = self.attn_res[-1](pool)
        logits = self.head(self.norm_out(x))

        if tgt is None:
            return logits, None

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)).float(),
            tgt[:, 1:].reshape(-1),
        )
        return logits, loss

        




device = 'cuda'



torch.set_float32_matmul_precision('high')

model=MiniK3(
    d_model=1024,
    n_head=32,
    d_head=64,
    d_k=64,
    d_v=64,
    d_c=64,
    d_ff=2084,
).to(device)


r = torch.arange(0, 2048).reshape(1, 2048).to(device)

import time

print("-")
start = time.perf_counter()
log, _ = model(r)
print(time.perf_counter()-start)

num = 0
for p in model.parameters():
    num += p.numel() if p.requires_grad else 0

print(num)