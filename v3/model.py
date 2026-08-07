import torch
import torch.nn as nn
import torch.nn.functional as F

from fla.ops.kda import chunk_kda


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class KDA(nn.Module):
    def __init__(self, d_model, n_head, d_head, g_min=-5, chunk=16):
        super().__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.proj_width = n_head * d_head

        self.g_min = g_min
        self.chunk = chunk

        self.swish = Swish()

        self.q_proj = nn.Linear(d_model, self.proj_width, bias=False)
        self.k_proj = nn.Linear(d_model, self.proj_width, bias=False)
        self.v_proj = nn.Linear(d_model, self.proj_width, bias=False)

        self.q_conv = nn.Conv1d(self.proj_width, self.proj_width, kernel_size=4, stride=1, groups=self.proj_width)
        self.k_conv = nn.Conv1d(self.proj_width, self.proj_width, kernel_size=4, stride=1, groups=self.proj_width)
        self.v_conv = nn.Conv1d(self.proj_width, self.proj_width, kernel_size=4, stride=1, groups=self.proj_width)

        self.b_proj = nn.Linear(d_model, n_head, bias=False)
        self.a_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, self.proj_width)
        )

        self.a_log = nn.Parameter(torch.zeros(n_head))

        self.o_norm = nn.RMSNorm(d_head)
        self.g_proj = nn.Linear(d_model, self.proj_width, bias=False)
        self.o_proj = nn.Linear(self.proj_width, d_model, bias=False)


    def _chunk(self, q, k, v, beta, g):
        B, T, H, D = q.shape
        C = self.chunk
        n_chunk = T // C

        q = q.reshape(B, n_chunk, C, H, D)
        k = k.reshape(B, n_chunk, C, H, D)
        v = v.reshape(B, n_chunk, C, H, D)
        g = g.reshape(B, n_chunk, C, H, D)
        beta = beta.reshape(B, n_chunk, C, H)

        G = g.cumsum(dim=2)

        i = torch.arange(C, device=q.device)

        causal = (i[:, None] >= i[None, :]).view(1, C, C, 1, 1)
        strict = (i[:, None] >  i[None, :]).view(1, C, C, 1)
        eye = torch.eye(C, device=q.device, dtype=q.dtype)

        s = q.new_zeros(B, H, D, D)
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
            U, W = UW[..., :D], UW[..., D:]

            v_pseudo = U - W @ s 

            o_c = q_decay.permute(0, 2, 1, 3) @ s + (qk * causal.squeeze(-1).permute(0, 3, 1, 2)) @ v_pseudo
            out.append(o_c.permute(0, 2, 1, 3)) 

            to_end = (G_c[:, -1:] - G_c).exp()                        
            s = G_c[:, -1].exp().unsqueeze(-1) * s + torch.einsum('bchk,bhcd->bhkd', k_c * to_end, v_pseudo)
 
        return torch.stack(out, dim=1).reshape(B, T, H, D)

    def forward(self, x):
        B, T, _ = x.shape
        H, D, P = self.n_head, self.d_head, self.proj_width

        q = self.q_proj(x).permute(0, 2, 1)
        k = self.k_proj(x).permute(0, 2, 1)
        v = self.v_proj(x).permute(0, 2, 1)

        q = self.q_conv(F.pad(q, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)
        k = self.k_conv(F.pad(k, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)
        v = self.v_conv(F.pad(v, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)

        q = F.normalize(self.swish(q), p=2, dim=-1)
        k = F.normalize(self.swish(k), p=2, dim=-1)
        v = self.swish(v)

        beta = self.b_proj(x).sigmoid()
        z = self.a_proj(x).reshape(B, T, H, D)

        g = self.g_min*F.sigmoid(self.a_log.exp().view(1, 1, -1, 1) * z)

        #return self._chunk(q, k, v, beta, g)

        o, _ = chunk_kda(q, k, v, g, beta, scale=1.0)
        o = self.o_norm(o).reshape(B, T, P)
        return self.o_proj(self.g_proj(x).sigmoid() * o)




device = 'cuda'

model = KDA(
    d_model=256,
    n_head=8,
    d_head=64,
    g_min=-5
).to(device)


r = torch.rand(1, 32, 256, device=device)

print(model(r).shape)