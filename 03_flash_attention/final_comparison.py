# Numbers from our CUDA benchmarks + PyTorch SDPA above
# Fill in the PyTorch numbers after running compare_all.py

data = {
    "N":          [128,   256,   512,   1024,  2048,  4096],
    "Naive (ms)": [0.020, 0.058, 0.214, 0.937, 3.248, 12.837],
    "Flash v1":   [0.091, 0.177, 0.419, 1.254, 4.678, 19.701],
    "Flash v2":   [0.055, 0.111, 0.378, 1.317, 5.248, 21.253],
}

print(f"\n{'N':>6} | {'Naive':>10} | {'Flash v1':>10} | {'Flash v2':>10} | {'v2 vs naive':>12}")
print("-" * 65)
for i, n in enumerate(data["N"]):
    naive = data["Naive (ms)"][i]
    v1    = data["Flash v1"][i]
    v2    = data["Flash v2"][i]
    ratio = naive / v2
    print(f"N={n:4d} | {naive:8.3f}ms | {v1:8.3f}ms | {v2:8.3f}ms | {ratio:8.2f}x {'faster' if ratio>1 else 'slower'}")

print("""
Key insight: our flash kernels are still slower than naive.
Why? The real FlashAttention gains come from:
1. Larger tile sizes (Br=64-128, not 16-32)
2. Tensor core wmma instructions for the inner matmul
3. Vectorized float4 global memory loads
4. Register tiling — keeping O in registers across ALL tiles

Our implementation correctly demonstrates the algorithm and O(N)
memory property. The performance gap vs PyTorch SDPA is expected —
PyTorch uses cuDNN's FlashAttention which has all of the above.
This gap is honestly documented in the blog post.
""")
