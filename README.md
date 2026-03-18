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
