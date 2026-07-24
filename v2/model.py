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

        self.norm_q = nn.LayerNorm(d_head)
        self.norm_k = nn.LayerNorm(d_head)

        self.f_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, proj_width)
        )

        self.b_proj = nn.Linear(d_model, n_heads, bias=False)
        

    def forward(self, x):
        B, T, _ = x.shape

        q = self.wq(x).transpose(1, 2)
        k = self.wk(x).transpose(1, 2)
        v = self.wv(x).transpose(1, 2)

        q = self.silu(self.conv_q(F.pad(q, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        k = self.silu(self.conv_k(F.pad(k, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)
        v = self.silu(self.conv_v(F.pad(v, (3, 0)))).transpose(1, 2).reshape(B, T, self.n_heads, self.d_head)


        q = self.norm_q(q)
        k = self.norm_k(k)

        print(q.shape)
        print(k.shape)
        print(v.shape)

        f = self.f_proj(x).reshape(B, T, self.n_heads, self.d_head)
        b = self.b_proj(x).sigmoid()

        s_t = torch.zeros(B, self.n_heads, self.d_head, self.d_head)

        print()
        print(s_t.shape)

        v_t = (s_t @ k[:, 0].reshape(B, self.n_heads, self.d_head, 1)).reshape(B, self.n_heads, self.d_head)

        err = (v[:, 0] - v_t).reshape(B, self.n_heads, self.d_head, 1)
        corr = err @ k[:, 0].reshape(B, self.n_heads, 1, self.d_head)


        print()


        corr = torch.mul(b[:, 0].reshape(B, self.n_heads, 1, 1), corr)

        print(f.shape)

        decay = torch.mul(f[:, 0].reshape(B, self.n_heads,1, self.d_head), s_t)

        s_t = corr + decay

        out = s_t @ q[:, 0].reshape(B, self.n_heads, self.d_head, 1)





attn = KdaAttention(256, 8, 64)

a = torch.rand(1, 10, 256)

attn(a)