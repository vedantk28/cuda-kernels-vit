#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <float.h>

/*
  FlashAttention-style tiled attention kernel
  
  Key insight: instead of computing the full N×N attention matrix, we
  process Q in row-tiles and K,V in column-tiles. For each Q-tile we
  accumulate the output incrementally using the ONLINE SOFTMAX TRICK:
  
  For each new K-tile we see, we update:
    m_new = max(m_old, rowmax(S_tile))   ← running max
    l_new = e^(m_old - m_new)*l_old + rowsum(e^(S_tile - m_new))  ← running sum
    O_new = diag(e^(m_old-m_new))*O_old + e^(S_tile-m_new)*V_tile ← running output
  
  At the end, O_final = O / l  (normalize by the softmax denominator)
  
  This means we NEVER write the N×N attention matrix to global memory.
  Memory complexity drops from O(N²) to O(N).

  Tile sizes:
    Br = block size for Q (rows)    = 16
    Bc = block size for K,V (cols)  = 16
*/

#define Br 16   // Q tile rows
#define Bc 16   // K,V tile cols
#define D  64   // head dimension (fixed for simplicity)

__global__ void flash_attention_forward(
    const float* __restrict__ Q,  // (N, D)
    const float* __restrict__ K,  // (N, D)
    const float* __restrict__ V,  // (N, D)
    float*       __restrict__ O,  // (N, D)
    float*       __restrict__ L,  // (N,) logsumexp — needed for backward
    int N)
{
    // Each block handles one tile of Q rows: [block_row*Br, (block_row+1)*Br)
    int block_row = blockIdx.x;
    int tid       = threadIdx.x;  // tid in [0, Br)

    // Row index this thread is responsible for
    int q_row = block_row * Br + tid;

    // Shared memory tiles for K and V
    __shared__ float Ks[Bc][D];   // K tile: Bc rows × D cols
    __shared__ float Vs[Bc][D];   // V tile: Bc rows × D cols

    // Per-thread registers: running output, running max, running sum
    float o[D];       // accumulator for output row
    float m = -FLT_MAX;  // running row max
    float l = 0.0f;      // running softmax denominator

    // Initialize output accumulator
    for (int d = 0; d < D; d++) o[d] = 0.0f;

    float scale = 1.0f / sqrtf((float)D);

    // ── Outer loop: iterate over K,V tiles ──────────────────────────────────
    int num_tiles = (N + Bc - 1) / Bc;

    for (int tile = 0; tile < num_tiles; tile++) {

        // ── Cooperatively load K tile into shared memory ─────────────────
        // Each thread loads one row of K and one row of V
        int kv_row = tile * Bc + tid;
        if (kv_row < N) {
            for (int d = 0; d < D; d++) {
                Ks[tid][d] = K[kv_row * D + d];
                Vs[tid][d] = V[kv_row * D + d];
            }
        } else {
            for (int d = 0; d < D; d++) {
                Ks[tid][d] = 0.0f;
                Vs[tid][d] = 0.0f;
            }
        }
        __syncthreads();

        // ── Compute attention scores for this tile: s = Q[q_row] @ K_tile^T ─
        float s[Bc];   // scores for this row against all Bc keys
        if (q_row < N) {
            for (int j = 0; j < Bc; j++) {
                float dot = 0.0f;
                for (int d = 0; d < D; d++)
                    dot += Q[q_row * D + d] * Ks[j][d];
                s[j] = dot * scale;

                // Mask out padding positions
                if (tile * Bc + j >= N) s[j] = -FLT_MAX;
            }

            // ── Online softmax update ──────────────────────────────────────
            // Find max of this tile's scores
            float m_tile = -FLT_MAX;
            for (int j = 0; j < Bc; j++)
                m_tile = fmaxf(m_tile, s[j]);

            // Update running max and compute rescaling factor for old output
            float m_new = fmaxf(m, m_tile);
            float rescale = expf(m - m_new);  // how much to shrink old output

            // Update running sum
            float l_tile = 0.0f;
            for (int j = 0; j < Bc; j++)
                l_tile += expf(s[j] - m_new);

            float l_new = rescale * l + l_tile;

            // ── Accumulate output: O = rescale*O_old + exp(s-m_new)*V_tile ─
            for (int d = 0; d < D; d++) {
                float ov = 0.0f;
                for (int j = 0; j < Bc; j++)
                    ov += expf(s[j] - m_new) * Vs[j][d];
                o[d] = rescale * o[d] + ov;
            }

            // Update running statistics
            m = m_new;
            l = l_new;
        }
        __syncthreads();  // done with shared memory before next tile loads
    }

    // ── Write final output: normalize by softmax denominator ────────────────
    if (q_row < N) {
        for (int d = 0; d < D; d++)
            O[q_row * D + d] = o[d] / l;

        // Save log-sum-exp for backward pass: log(l) + m
        L[q_row] = logf(l) + m;
    }
}

// ── Validation: compare vs naive attention ───────────────────────────────────
__global__ void naive_attn_reference(
    const float* Q, const float* K, const float* V,
    float* O, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    float scale = 1.0f / sqrtf((float)D);
    float scores[2048];  // max N we'll test
    float max_s = -FLT_MAX;

    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += Q[row * D + d] * K[j * D + d];
        scores[j] = dot * scale;
        max_s = fmaxf(max_s, scores[j]);
    }

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        scores[j] = expf(scores[j] - max_s);
        sum += scores[j];
    }

    for (int d = 0; d < D; d++) {
        float out = 0.0f;
        for (int j = 0; j < N; j++)
            out += (scores[j] / sum) * V[j * D + d];
        O[row * D + d] = out;     // FIX: 'd' is the loop variable
    }
}

void benchmark_flash(int N) {
    size_t bytes = N * D * sizeof(float);

    float *h_Q = (float*)malloc(bytes);
    float *h_O_flash  = (float*)malloc(bytes);
    float *h_O_naive  = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N * D; i++)
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;

    float *d_Q, *d_K, *d_V, *d_O, *d_L;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_O, bytes);
    cudaMalloc(&d_L, N * sizeof(float));

    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q, bytes, cudaMemcpyHostToDevice);

    // ── Run flash attention ──────────────────────────────────────────────────
    int num_blocks = (N + Br - 1) / Br;

    // Warmup
    flash_attention_forward<<<num_blocks, Br>>>(d_Q, d_K, d_V, d_O, d_L, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 20; i++)
        flash_attention_forward<<<num_blocks, Br>>>(d_Q, d_K, d_V, d_O, d_L, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_flash; cudaEventElapsedTime(&ms_flash, start, stop);
    ms_flash /= 20;

    cudaMemcpy(h_O_flash, d_O, bytes, cudaMemcpyDeviceToHost);

    // Memory: reads Q,K,V (N*D each) + writes O,L — NO N×N matrix
    double bytes_flash = (3.0*N*D + N*D + N) * sizeof(float);
    double bw_flash = bytes_flash / (ms_flash * 1e6);

    printf("N=%4d | flash: %7.3f ms  %6.2f GB/s  mem=O(N)",
           N, ms_flash, bw_flash);

    // ── Validate against naive (small N only) ───────────────────────────────
    if (N <= 512) {
        float *d_O_ref;
        cudaMalloc(&d_O_ref, bytes);
        naive_attn_reference<<<(N+255)/256, 256>>>(d_Q, d_K, d_V, d_O_ref, N);
        cudaDeviceSynchronize();
        cudaMemcpy(h_O_naive, d_O_ref, bytes, cudaMemcpyDeviceToHost);

        float max_err = 0.0f;
        for (int i = 0; i < N * D; i++)
            max_err = fmaxf(max_err, fabsf(h_O_flash[i] - h_O_naive[i]));
        printf("  max_err=%.2e %s", max_err, max_err < 1e-3 ? "PASS" : "FAIL");
        cudaFree(d_O_ref);
    }
    printf("\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_O); cudaFree(d_L);
    free(h_Q); free(h_O_flash); free(h_O_naive);
}

int main() {
    printf("FlashAttention-style kernel — RTX 3050 (sm_86)\n");
    printf("No N×N matrix materialized — memory is O(N) not O(N²)\n\n");
    printf("%-6s | %-45s\n", "N", "Flash attention");
    printf("%.65s\n", "-----------------------------------------------------------------");

    int sizes[] = {128, 256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 6; i++)
        benchmark_flash(sizes[i]);

    return 0;
}