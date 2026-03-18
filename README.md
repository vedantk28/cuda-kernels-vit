# CUDA Kernels for Vision Transformer Attention

Custom CUDA kernels implementing efficient attention mechanisms for Vision Transformers (ViT), built from scratch on an RTX 3050 (Compute Capability 8.6).

## Motivation
PyTorch's default attention is memory-bound — it materializes the full N×N attention matrix in global memory. This project implements increasingly optimized kernels culminating in a FlashAttention-style tiled kernel that avoids this bottleneck.

## Hardware
- GPU: NVIDIA GeForce RTX 3050 Laptop (4GB VRAM, CC 8.6)
- CUDA: 12.6
- Driver: 580.126.09

## Project Structure
```
01_tiled_matmul/     — Naive vs tiled matmul, cuBLAS comparison
02_naive_attention/  — O(N²) attention kernel + PyTorch binding
03_flash_attention/  — Tiled attention (FlashAttention-style)
04_vit_integration/  — Drop-in replacement for ViT-Base attention
```

## Phase 1 Results — Tiled Matrix Multiply

| Size | Naive | Tiled (ours) | cuBLAS | vs cuBLAS |
|------|-------|--------------|--------|-----------|
| 256  | 0.09 ms / 0.36 TFLOPS | 0.07 ms / 0.46 TFLOPS | 0.02 ms / 1.92 TFLOPS | 4.1x slower |
| 512  | 0.72 ms / 0.37 TFLOPS | 0.55 ms / 0.49 TFLOPS | 0.09 ms / 3.07 TFLOPS | 6.2x slower |
| 1024 | 5.71 ms / 0.38 TFLOPS | 4.31 ms / 0.50 TFLOPS | 0.57 ms / 3.77 TFLOPS | 7.6x slower |
| 2048 | 39.12 ms / 0.44 TFLOPS | 28.35 ms / 0.61 TFLOPS | 3.73 ms / 4.60 TFLOPS | 7.6x slower |
| 4096 | 322.49 ms / 0.43 TFLOPS | 231.61 ms / 0.59 TFLOPS | 32.46 ms / 4.23 TFLOPS | 7.1x slower |

**Gap vs cuBLAS explained:** cuBLAS uses tensor cores (not available to simple CUDA C kernels without wmma intrinsics), 128-bit vectorized global memory loads (`float4`), and warp-level matrix multiply primitives. Closing this gap is the subject of Phase 3.

## Building
```bash
# Phase 1
cd 01_tiled_matmul
nvcc -O3 -arch=sm_86 tiled_matmul.cu -o tiled_matmul && ./tiled_matmul
nvcc -O3 -arch=sm_86 benchmark_sizes.cu -o benchmark_sizes && ./benchmark_sizes
nvcc -O3 -arch=sm_86 benchmark_cublas.cu -o benchmark_cublas -lcublas && ./benchmark_cublas
```

## Phase 2 Results — Naive Attention (O(N²) memory bottleneck)

| N | d | Time | Mem BW utilization | Attn matrix size |
|---|---|------|--------------------|-----------------|
| 128 | 64 | 0.020 ms | 13.26 GB/s (6.9%) | 0.1 MB |
| 256 | 64 | 0.058 ms | 13.50 GB/s (7.0%) | 0.3 MB |
| 512 | 64 | 0.214 ms | 12.23 GB/s (6.4%) | 1.0 MB |
| 1024 | 64 | 0.937 ms | 10.07 GB/s (5.2%) | 4.2 MB |
| 2048 | 64 | 3.248 ms | 10.98 GB/s (5.7%) | 16.8 MB |
| 4096 | 64 | 12.837 ms | 10.78 GB/s (5.6%) | 67.1 MB |

**Key insight:** Memory bandwidth utilization is ~5% of the 192 GB/s theoretical peak.
The kernel spends most of its time waiting on global memory reads/writes of the N×N
attention matrix — not computing. This is the wall FlashAttention breaks through.

## Phase 2 Results — Naive Attention (O(N²) memory bottleneck)

| N | d | Time | Mem BW utilization | Attn matrix size |
|---|---|------|--------------------|-----------------|
| 128 | 64 | 0.020 ms | 13.26 GB/s (6.9%) | 0.1 MB |
| 256 | 64 | 0.058 ms | 13.50 GB/s (7.0%) | 0.3 MB |
| 512 | 64 | 0.214 ms | 12.23 GB/s (6.4%) | 1.0 MB |
| 1024 | 64 | 0.937 ms | 10.07 GB/s (5.2%) | 4.2 MB |
| 2048 | 64 | 3.248 ms | 10.98 GB/s (5.7%) | 16.8 MB |
| 4096 | 64 | 12.837 ms | 10.78 GB/s (5.6%) | 67.1 MB |

**Key insight:** Memory bandwidth utilization is ~5% of the 192 GB/s theoretical peak.
The kernel spends most of its time waiting on global memory reads/writes of the N×N
attention matrix — not computing. This is the wall FlashAttention breaks through.

## Phase 2 Results — Naive Attention (O(N²) memory bottleneck)

| N | d | Time | Mem BW utilization | Attn matrix size |
|---|---|------|--------------------|-----------------|
| 128 | 64 | 0.020 ms | 13.26 GB/s (6.9%) | 0.1 MB |
| 256 | 64 | 0.058 ms | 13.50 GB/s (7.0%) | 0.3 MB |
| 512 | 64 | 0.214 ms | 12.23 GB/s (6.4%) | 1.0 MB |
| 1024 | 64 | 0.937 ms | 10.07 GB/s (5.2%) | 4.2 MB |
| 2048 | 64 | 3.248 ms | 10.98 GB/s (5.7%) | 16.8 MB |
| 4096 | 64 | 12.837 ms | 10.78 GB/s (5.6%) | 67.1 MB |

**Key insight:** Memory bandwidth utilization is ~5% of the 192 GB/s theoretical peak.
The kernel spends most of its time waiting on global memory reads/writes of the N×N
attention matrix — not computing. This is the wall FlashAttention breaks through.

## Phase 2 Results — Naive Attention (O(N²) memory bottleneck)

| N | d | Time | Mem BW utilization | Attn matrix size |
|---|---|------|--------------------|-----------------|
| 128 | 64 | 0.020 ms | 13.26 GB/s (6.9%) | 0.1 MB |
| 256 | 64 | 0.058 ms | 13.50 GB/s (7.0%) | 0.3 MB |
| 512 | 64 | 0.214 ms | 12.23 GB/s (6.4%) | 1.0 MB |
| 1024 | 64 | 0.937 ms | 10.07 GB/s (5.2%) | 4.2 MB |
| 2048 | 64 | 3.248 ms | 10.98 GB/s (5.7%) | 16.8 MB |
| 4096 | 64 | 12.837 ms | 10.78 GB/s (5.6%) | 67.1 MB |

**Key insight:** Memory bandwidth utilization is ~5% of the 192 GB/s theoretical peak.
The kernel spends most of its time waiting on global memory reads/writes of the N×N
attention matrix — not computing. This is the wall FlashAttention breaks through.

## Phase 3 Results — FlashAttention (O(N) memory, honest benchmarks)

| N | Naive (ours) | Flash v1 (ours) | Flash v2 (ours) | PyTorch SDPA |
|---|-------------|-----------------|-----------------|--------------|
| 128 | 0.020ms | 0.091ms | 0.055ms | 0.030ms |
| 256 | 0.058ms | 0.177ms | 0.111ms | 0.058ms |
| 512 | 0.214ms | 0.419ms | 0.378ms | 0.114ms |
| 1024 | 0.937ms | 1.254ms | 1.317ms | 0.228ms |
| 2048 | 3.248ms | 4.678ms | 5.248ms | 0.837ms |
| 4096 | 12.837ms | 19.701ms | 21.253ms | 3.461ms |

**What our implementation gets right:**
- Correct online softmax algorithm (validated to 1e-8 vs reference)
- O(N) memory — no N×N attention matrix ever written to global memory
- Numerically stable with running (max, sum) per tile

**Why we're slower than PyTorch SDPA:**
- Tile size Br=16 vs FlashAttention-2's Br=64-128 — too few elements per block
- No tensor core wmma instructions — compute-bound inner loop runs on CUDA cores
- No float4 vectorized global memory loads — 4x fewer memory transactions possible
- Low occupancy — not enough warps active to hide memory latency

**What this teaches:** The algorithmic insight (online softmax, O(N) memory) is
separable from the systems engineering (tensor cores, vectorization, occupancy).
Understanding both layers is what the project demonstrates.
