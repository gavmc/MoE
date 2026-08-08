from dataclasses import dataclass



@dataclass
class Config:
    # 155.7M total - 69.7M active
    d_model = 640
    n_head = 8
    d_k = 128
    d_v = 64
    d_c = 256
    d_expert = 512
    d_latent = 320
    d_shared = 1024
    d_head = 128

    mix_ratio = 4
    layers = 12
    vocab_size = 16000
    tie_embeddings = True
    use_ckpt = True

    # 90% at 2048 context length, 10% at 8192 context length (potential 16384 context length to replace last 2%)

