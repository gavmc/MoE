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

        self.beta = nn.Linear(d_model, n_heads, bias=False)
        self.alpha = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, proj_width, bias=False)
        )

        self.A_log = nn.Parameter(torch.rand(1, 1, n_heads, 1)*16, requires_grad=True)
        self.dt_bias = nn.Parameter(torch.rand(proj_width), requires_grad=True)

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

        b = self.beta(x).sigmoid()
        f = self.alpha(x)
        g = self.alpha(x)


        print(b.shape)
        print(f.shape)
        print(g.shape)






attn = KdaAttention(256, 8, 32)

a = torch.rand(1, 10, 256)

attn(a)