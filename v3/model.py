import torch
import torch.nn as nn
import torch.nn.functional as F



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
        self.q_conv = nn.Conv1d(self.proj_width, self.proj_width, kernel_size=4, stride=1, groups=self.proj_width)

        self.b_proj = nn.Linear(d_model, n_head, bias=False)
        self.a_proj = nn.Sequential(
            nn.Linear(d_model, d_head, bias=False),
            nn.Linear(d_head, self.proj_width)
        )

        self.a_log = nn.Parameter(torch.zeros(n_head))


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




        

    def forward(self, x):
        B, T, _ = x.shape
        H, D, P = self.n_head, self.d_head, self.proj_width

        q = self.q_proj(x).permute(0, 2, 1)
        k = self.k_proj(x).permute(0, 2, 1)
        v = self.v_proj(x).permute(0, 2, 1)

        q = self.q_conv(F.pad(q, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)
        k = self.q_conv(F.pad(k, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)
        v = self.q_conv(F.pad(v, (3, 0))).permute(0, 2, 1).reshape(B, T, H, D)

        q = F.normalize(self.swish(q), p=2, dim=-1)
        k = F.normalize(self.swish(k), p=2, dim=-1)

        beta = F.softmax(self.b_proj(x), dim=-1)
        z = self.a_proj(x).reshape(B, T, H, D)

        g = self.g_min*F.sigmoid(self.a_log.exp().view(1, 1, -1, 1) * z)

        self._chunk(q, k, v, beta, g)




model = KDA(
    d_model=256,
    n_head=8,
    d_head=64,
    g_min=-5
)


r = torch.rand(1, 5, 256)

model(r)