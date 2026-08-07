import time
import torch
from model import MiniK3

def benchmark(model, B, T, vocab, steps=10, warmup=3, bf16=True, compile_model=True,
              peak_tflops=71.0):
    """peak_tflops: 3090 bf16 tensor-core peak with fp32 accumulate is ~71 TFLOPS."""
    dev = next(model.parameters()).device
    cuda = dev.type == 'cuda'
    if compile_model:
        model = torch.compile(model, dynamic=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=cuda)
    idx = torch.randint(0, vocab, (B, T), device=dev)
    

    def step():
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
            _, loss = model(idx, tgt=idx)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    '''
    torch.cuda.memory._record_memory_history(max_entries=100_000)
    step()
    torch.cuda.memory._dump_snapshot('snap.pickle')
    '''
    
    for _ in range(warmup):
        step()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    if cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / steps

    n_params = sum(p.numel() for p in model.parameters())
    n_embed = sum(p.numel() for n, p in model.named_parameters() if 'embedding' in n)
    n_train = n_params - n_embed

    # active (per-token) params: routed experts only count at top_k / n_experts
    n_routed = n_routed_active = 0
    for mod in model.modules():
        if hasattr(mod, 'e_gate'):  # MoE layer
            r = mod.e_gate.numel() + mod.e_up.numel() + mod.e_down.numel()
            n_routed += r
            n_routed_active += round(r * mod.top_k / mod.n_experts)
    n_active = n_train - n_routed + n_routed_active

    # tied LM head is excluded from n_train but still does per-token compute
    n_lm_head = model.head.weight.numel() if hasattr(model, 'head') else 0
    flops = 6 * (n_active + n_lm_head) * B * T       # fwd+bwd, active compute
    tok_s = B * T / dt

    print(f'params      {n_params/1e6:.1f}M total, {n_train/1e6:.1f}M non-embedding, {n_active/1e6:.1f}M active')
    print(f'step        {dt*1e3:.1f} ms   ({tok_s:,.0f} tok/s)')
    print(f'model FLOPs {flops/dt/1e12:.1f} TFLOP/s   MFU {flops/dt/1e12/peak_tflops*100:.1f}%')
    if cuda:
        print(f'peak mem    {torch.cuda.max_memory_allocated()/2**30:.2f} GiB')
    return dict(dt=dt, tok_s=tok_s, n_train=n_train, n_active=n_active)


def project(tok_s, n_active, budgets=(5e9, 10e9, 20e9, 30e9)):
    print(f'\nchinchilla-optimal (20 tok/active param): {20 * n_active / 1e9:.1f}B tokens')
    for b in budgets:
        hours = b / tok_s / 3600
        print(f'  {b/1e9:>4.0f}B tokens  ->  {hours:6.1f} h  ({hours/24:.1f} days)')


if __name__ == '__main__':
    import torch.nn as nn
    import os

    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    m = MiniK3(d_model=640, n_head=8, d_k=128, d_v=64, d_c=256, d_expert=512, d_latent=320, d_shared=1024,
           d_head=128, layers=12, vocab_size=16000).cuda()

    r = benchmark(m, B=22, T=1024, vocab=16000, compile_model=True)
    project(r['tok_s'], r['n_active'])