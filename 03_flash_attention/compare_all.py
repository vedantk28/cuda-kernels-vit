import torch
import torch.nn.functional as F
import time

print("PyTorch SDPA benchmark — RTX 3050")
print(f"PyTorch: {torch.__version__}  CUDA: {torch.version.cuda}\n")
print(f"{'N':>6} | {'PyTorch SDPA':>20} | {'Memory saved vs naive':>22}")
print("-" * 60)

D = 64
for N in [128, 256, 512, 1024, 2048, 4096]:
    Q = torch.randn(1, 1, N, D, device='cuda', dtype=torch.float32)
    K = torch.randn(1, 1, N, D, device='cuda', dtype=torch.float32)
    V = torch.randn(1, 1, N, D, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(10):
        _ = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()

    # Timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(100):
        out = F.scaled_dot_product_attention(Q, K, V)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    ms = (t1 - t0) * 1000 / 100
    naive_mem_mb = N * N * 4 / 1e6
    flash_mem_mb = N * D * 4 * 4 / 1e6
    print(f"N={N:4d} | {ms:8.3f} ms              | "
          f"naive={naive_mem_mb:.1f}MB flash={flash_mem_mb:.2f}MB")

