#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

/*
  Naive matmul — each thread loads from global memory every time.
  This is the baseline we'll beat with tiling.
  C = A @ B  where A is (M,K), B is (K,N), C is (M,N)
*/
__global__ void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

/*
  Tiled matmul — threads cooperatively load tiles into shared memory.
  Each element of A and B is loaded from global memory only once per tile.
  Shared memory acts as a manually managed L1 cache.

  Key insight: for a TILE_SIZE=16 tile, we do 16*16=256 multiplications
  using only 16+16=32 global memory loads (shared across the tile).
  That's 8x fewer global memory accesses vs naive.
*/
__global__ void matmul_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float acc = 0.0f;

    // Loop over tiles along the K dimension
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // Each thread loads one element into shared memory
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K)
            ? A[row * K + tiledCol] : 0.0f;

        tileB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N)
            ? B[tiledRow * N + col] : 0.0f;

        // Wait for ALL threads in the block to finish loading
        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Wait before loading the next tile (avoid race condition)
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// Simple host-side launcher — we'll replace this with pybind11 next
int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C = (float*)malloc(bytes_C);

    // Fill with simple values for validation
    for (int i = 0; i < M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE - 1) / TILE_SIZE,
                (M + TILE_SIZE - 1) / TILE_SIZE);

    // --- Time naive kernel ---
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_naive;
    cudaEventElapsedTime(&ms_naive, start, stop);

    // --- Time tiled kernel ---
    cudaEventRecord(start);
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms_tiled;
    cudaEventElapsedTime(&ms_tiled, start, stop);

    printf("Matrix size: %dx%d\n", M, N);
    printf("Naive kernel:  %.3f ms\n", ms_naive);
    printf("Tiled kernel:  %.3f ms\n", ms_tiled);
    printf("Speedup:       %.2fx\n", ms_naive / ms_tiled);

    // Validate — every element should be K (= 1024.0)
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    bool correct = true;
    for (int i = 0; i < M * N; i++) {
        if (fabs(h_C[i] - K) > 1e-1) { correct = false; break; }
    }
    printf("Correctness:   %s\n", correct ? "PASS" : "FAIL");

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}