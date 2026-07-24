import torch
import torch.nn as nn





class KdaAttention(nn.Module):
    def __init__(self, d_model, ctx_len, n_heads, d_head):
        super().__init__()

        proj_width = n_heads*d_head

        self.norm = nn.RMSNorm(d_model)

        self.wq = nn.Linear(d_model, proj_width, bias=False)
        self.wk = nn.Linear(d_model, proj_width, bias=False)
        self.wv = nn.Linear(d_model, proj_width, bias=False)

        self.conv_q = nn.Conv1d(proj_width, proj_width, kernel_size=4, stride=2, padding=2)
        self.conv_k = nn.Conv1d(proj_width, proj_width, kernel_size=4, stride=2, padding=2)
        self.conv_v = nn.Conv1d(proj_width, proj_width, kernel_size=4, stride=2, padding=2)

        self.silu = nn.SiLU()

    def forward(self, x):

        x = self.norm(x)

        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        print(q.shape)
        print(k.shape)
        print(v.shape)

        q = self.conv_q(q)

        print(q.shape)


class MoE(nn.Module):
    def __init__(self, vocab_size, ctx_len, d_model, n_layers, n_blocks):
        super().__init__()

        self.residual_weights = nn.Parameter(1, n_blocks, requires_grad=True)

        self.embeddings = nn.Embedding(vocab_size, d_model)








attn = KdaAttention(256, 1024, 8, 32)

a = torch.rand(1, 10, 256)

attn(a)