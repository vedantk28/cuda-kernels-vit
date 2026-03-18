#include <cuda_runtime.h>
#include <stdio.h>

#define TILE_SIZE 16

__global__ void matmul_naive(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) acc += A[row * K + k] * B[k * N + col];
        C[row * N + col] = acc;
    }
}

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float acc = 0.0f;
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tiledCol = t * TILE_SIZE + threadIdx.x;
        int tiledRow = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && tiledCol < K) ? A[row * K + tiledCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (tiledRow < K && col < N) ? B[tiledRow * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE_SIZE; k++) acc += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = acc;
}

void benchmark(int M, int N, int K) {
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks((N + TILE_SIZE-1)/TILE_SIZE, (M + TILE_SIZE-1)/TILE_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    // Warmup
    matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_naive<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_naive; cudaEventElapsedTime(&ms_naive, start, stop);

    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
    cudaEventRecord(stop); cudaEventSynchronize(stop);
    float ms_tiled; cudaEventElapsedTime(&ms_tiled, start, stop);

    // Compute TFLOPS
    double flops = 2.0 * M * N * K * 10;
    double tflops_naive = flops / (ms_naive * 1e9);
    double tflops_tiled = flops / (ms_tiled * 1e9);

    printf("N=%4d | naive: %7.2f ms  %5.2f TFLOPS | tiled: %7.2f ms  %5.2f TFLOPS | speedup: %.2fx\n",
           M, ms_naive/10, tflops_naive, ms_tiled/10, tflops_tiled, ms_naive/ms_tiled);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
}

int main() {
    printf("%-6s | %-35s | %-35s | %s\n", "Size", "Naive", "Tiled", "Speedup");
    printf("%s\n", "-----------------------------------------------------------------------");
    int sizes[] = {256, 512, 1024, 2048, 4096};
    for (int i = 0; i < 5; i++)
        benchmark(sizes[i], sizes[i], sizes[i]);
    return 0;
}
