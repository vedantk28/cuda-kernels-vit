#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>

/*
  Scaled Dot-Product Attention — naive CUDA implementation
  
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  Shapes:
    Q, K, V : (N, d)   — N tokens, d head dimension
    output  : (N, d)

  This kernel materializes the full (N, N) attention matrix in global memory.
  That is the fundamental bottleneck FlashAttention eliminates in Phase 3.

  We split into 3 kernels:
    1. attn_scores   — compute S = Q @ K^T / sqrt(d),  shape (N,N)
    2. softmax_rows  — softmax over each row of S
    3. attn_output   — compute O = S @ V,              shape (N,d)
*/

#define TILE_SIZE 16

// ─── Kernel 1: S = Q @ Kᵀ / sqrt(d) ────────────────────────────────────────
__global__ void attn_scores(
    const float* Q,   // (N, d)
    const float* K,   // (N, d)
    float*       S,   // (N, N)  output
    int N, int d)
{
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;  // query token index
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;  // key   token index

    __shared__ float tileQ[TILE_SIZE][TILE_SIZE];
    __shared__ float tileK[TILE_SIZE][TILE_SIZE];

    float acc = 0.0f;

    for (int t = 0; t < (d + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int d_col = t * TILE_SIZE + threadIdx.x;
        int d_row = t * TILE_SIZE + threadIdx.y;

        tileQ[threadIdx.y][threadIdx.x] = (row < N && d_col < d) ? Q[row * d + d_col] : 0.0f;
        tileK[threadIdx.y][threadIdx.x] = (col < N && d_row < d) ? K[col * d + d_row] : 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            acc += tileQ[threadIdx.y][k] * tileK[k][threadIdx.x];

        __syncthreads();
    }

    if (row < N && col < N)
        S[row * N + col] = acc / sqrtf((float)d);
}

// ─── Kernel 2: softmax over each row of S (in-place) ────────────────────────
// One block per row, one thread per element (works for N <= 1024)
__global__ void softmax_rows(float* S, int N) {
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // Step 1: find row max for numerical stability
    __shared__ float smax;
    float local_max = -1e38f;
    for (int i = tid; i < N; i += blockDim.x)
        local_max = fmaxf(local_max, S[row * N + i]);

    // Reduce max across threads in block
    __shared__ float reduce_buf[1024];
    reduce_buf[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) smax = reduce_buf[0];
    __syncthreads();

    // Step 2: exp(x - max) and sum
    __shared__ float ssum;
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        float val = expf(S[row * N + i] - smax);
        S[row * N + i] = val;
        local_sum += val;
    }

    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            reduce_buf[tid] += reduce_buf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) ssum = reduce_buf[0];
    __syncthreads();

    // Step 3: normalize
    for (int i = tid; i < N; i += blockDim.x)
        S[row * N + i] /= ssum;
}

// ─── Kernel 3: O = S @ V ────────────────────────────────────────────────────
__global__ void attn_output(
    const float* S,   // (N, N)
    const float* V,   // (N, d)
    float*       O,   // (N, d)
    int N, int d)
{
    __shared__ float tileS[TILE_SIZE][TILE_SIZE];
    __shared__ float tileV[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int s_col = t * TILE_SIZE + threadIdx.x;
        int v_row = t * TILE_SIZE + threadIdx.y;

        tileS[threadIdx.y][threadIdx.x] = (row < N && s_col < N) ? S[row * N + s_col] : 0.0f;
        tileV[threadIdx.y][threadIdx.x] = (v_row < N && col < d) ? V[v_row * d + col] : 0.0f;

        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++)
            acc += tileS[threadIdx.y][k] * tileV[k][threadIdx.x];
        __syncthreads();
    }

    if (row < N && col < d)
        O[row * d + col] = acc;
}

// ─── Host launcher ───────────────────────────────────────────────────────────
void run_attention(int N, int d, bool print_result) {
    size_t bytes_qkv = N * d * sizeof(float);
    size_t bytes_s   = N * N * sizeof(float);

    float *h_Q = (float*)malloc(bytes_qkv);
    float *h_O = (float*)malloc(bytes_qkv);

    // Init Q,K,V with small random values
    for (int i = 0; i < N * d; i++) {
        h_Q[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }

    float *d_Q, *d_K, *d_V, *d_S, *d_O;
    cudaMalloc(&d_Q, bytes_qkv);
    cudaMalloc(&d_K, bytes_qkv);
    cudaMalloc(&d_V, bytes_qkv);
    cudaMalloc(&d_S, bytes_s);
    cudaMalloc(&d_O, bytes_qkv);

    cudaMemcpy(d_Q, h_Q, bytes_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_Q, bytes_qkv, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_Q, bytes_qkv, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks_s((N+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);
    dim3 blocks_o((d+TILE_SIZE-1)/TILE_SIZE, (N+TILE_SIZE-1)/TILE_SIZE);

    // Warmup
    attn_scores<<<blocks_s, threads>>>(d_Q, d_K, d_S, N, d);
    softmax_rows<<<N, min(N, 1024)>>>(d_S, N);
    attn_output<<<blocks_o, threads>>>(d_S, d_V, d_O, N, d);
    cudaDeviceSynchronize();

    // Timed run
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++) {
        attn_scores<<<blocks_s, threads>>>(d_Q, d_K, d_S, N, d);
        softmax_rows<<<N, min(N, 1024)>>>(d_S, N);
        attn_output<<<blocks_o, threads>>>(d_S, d_V, d_O, N, d);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop);
    ms /= 10;

    // Memory: read Q,K (N*d each) + write S (N*N) + read S,V + write O
    double bytes_moved = (4.0*N*d + 2.0*N*N) * sizeof(float);
    double bandwidth_gb = bytes_moved / (ms * 1e6);

    printf("N=%4d d=%3d | time: %7.3f ms | mem BW: %6.2f GB/s | "
           "attn matrix: %5.1f MB\n",
           N, d, ms, bandwidth_gb, bytes_s / 1e6);

    // Check for NaN
    cudaMemcpy(h_O, d_O, bytes_qkv, cudaMemcpyDeviceToHost);
    bool has_nan = false;
    for (int i = 0; i < N * d; i++)
        if (isnan(h_O[i])) { has_nan = true; break; }
    if (has_nan) printf("  WARNING: NaN detected in output!\n");

    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V);
    cudaFree(d_S); cudaFree(d_O);
    free(h_Q); free(h_O);
}

int main() {
    printf("Naive attention benchmark — RTX 3050 (sm_86)\n");
    printf("Note: attn matrix is materialized in global memory (the bottleneck)\n\n");
    printf("%-35s | %s\n",
           "Config", "Results");
    printf("%.75s\n",
           "---------------------------------------------------------------------------");

    // d=64 is standard ViT head dimension
    int configs[][2] = {{128,64},{256,64},{512,64},{1024,64},{2048,64}};
    for (int i = 0; i < 5; i++)
        run_attention(configs[i][0], configs[i][1], false);

    printf("\n--- Memory pressure: watch attn matrix size grow as O(N²) ---\n\n");

    // Show how memory explodes with sequence length
    int long_seqs[] = {1024, 2048, 4096};
    for (int i = 0; i < 3; i++)
        run_attention(long_seqs[i], 64, false);

    return 0;
}