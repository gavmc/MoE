import torch
from torch import nn
import torch.nn.functional as F


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

        self.silu = nn.SiLU()

        self.f_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, proj_width)
        )

        self.b_proj = nn.Linear(d_model, n_heads, bias=False)

        self.wo = nn.Linear(proj_width, d_model, bias=True)
        

    def forward(self, x):
        B, T, _ = x.shape

        q = self.wq(x).transpose(1, 2)
        k = self.wk(x).transpose(1, 2)
        v = self.wv(x).transpose(1, 2)

        q = self.silu(self.conv_q(F.pad(q, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        k = self.silu(self.conv_k(F.pad(k, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        v = self.silu(self.conv_v(F.pad(v, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        f = self.f_proj(x).reshape(B, T, self.n_heads, self.d_head)
        a = torch.exp(-F.softplus(f))
        b = self.b_proj(x).sigmoid()

        s_t = torch.zeros(B, self.n_heads, self.d_head, self.d_head)

        out = torch.zeros(B, T, self.n_heads, self.d_head)

        for t in range(T):
            s_t = torch.mul(a[:, t].reshape(B, self.n_heads,1, self.d_head), s_t)
            v_t = (s_t @ k[:, t].reshape(B, self.n_heads, self.d_head, 1)).reshape(B, self.n_heads, self.d_head)

            err = (v[:, t] - v_t).reshape(B, self.n_heads, self.d_head, 1)

            corr = err @ k[:, t].reshape(B, self.n_heads, 1, self.d_head)
            corr = torch.mul(b[:, t].reshape(B, self.n_heads, 1, 1), corr)

            s_t += corr

            out[:, t] = (s_t @ q[:, t].reshape(B, self.n_heads, self.d_head, 1)).reshape(B, self.n_heads, self.d_head)


        out = self.wo(out.reshape(B, T, self.n_heads*self.d_head))
        print(out.shape)
        return out





attn = KdaAttention(256, 8, 64)

inp = torch.rand(1, 10, 256)

attn(inp)