# Implementing FlashAttention from scratch on an RTX 3050 Laptop

## Target publications
- Medium (personal ML blog)
- dev.to
- Towards Data Science (submit after 500 claps on Medium)

---

## Section 1 — The problem (300 words)
- Standard attention is O(N²) memory: show the 67MB attention matrix at N=4096
- Our naive kernel: 12.8ms at N=4096, using only 5% of available memory bandwidth
- The GPU is not computing — it's waiting for memory. Show the bandwidth numbers.
- One sentence on what FlashAttention promises to fix.

## Section 2 — The online softmax trick (400 words)
- This is the core mathematical insight. Explain it from first principles.
- Standard softmax needs two passes: one for max, one for exp+sum
- Online softmax does it in one pass with running statistics
- Show the update equations: m_new, l_new, O_new
- Include a small worked example with N=4 tokens

## Section 3 — Implementation walkthrough (600 words)
- Walk through flash_attention.cu kernel structure
- Explain the tile loop, shared memory layout, __syncthreads placement
- Show the warp shuffle reduction in v2 (__shfl_xor_sync)
- Include the key code snippets (not the full kernel)

## Section 4 — Honest benchmark results (300 words)
- Show the full comparison table
- Our kernel is slower than naive — explain exactly why (occupancy, tile size)
- Show PyTorch SDPA as the production baseline
- Memory comparison: 67MB → 4MB at N=4096

## Section 5 — What closing the gap requires (300 words)
- Tensor core wmma intrinsics: nvcuda::wmma API, 16x16x16 tiles
- float4 vectorized loads: 4x memory transaction efficiency
- Larger Br=64: amortizes shared memory load cost over more compute
- This is what cuDNN's FlashAttention implements

## Section 6 — What I learned (200 words)
- The algorithm and the systems optimization are separable concerns
- NSight Compute numbers: show the occupancy metric
- How to profile: nsys profile ./flash_attention_v2

---

## Code repo
https://github.com/vedantk28/cuda-kernels-vit

## Key takeaway line for resume/LinkedIn
"Implemented FlashAttention from scratch in CUDA — correct online softmax,
O(N) memory, profiled with NSight Compute, gap vs cuDNN documented with
root cause analysis."
