#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define TILE_SIZE 16

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

int main() {
    int sizes[] = {256, 512, 1024, 2048, 4096};
    cublasHandle_t handle;
    cublasCreate(&handle);

    printf("%-6s | %-20s | %-20s | %-20s\n", "Size", "Tiled (ours)", "cuBLAS", "cuBLAS speedup");
    printf("%.75s\n", "---------------------------------------------------------------------------------");

    for (int s = 0; s < 5; s++) {
        int M = sizes[s], N = sizes[s], K = sizes[s];
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));
        cudaMalloc(&d_C, M * N * sizeof(float));

        dim3 threads(TILE_SIZE, TILE_SIZE);
        dim3 blocks((N+TILE_SIZE-1)/TILE_SIZE, (M+TILE_SIZE-1)/TILE_SIZE);

        // Warmup
        matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaDeviceSynchronize();

        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);

        // Time tiled
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++)
            matmul_tiled<<<blocks, threads>>>(d_A, d_B, d_C, M, N, K);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_tiled; cudaEventElapsedTime(&ms_tiled, start, stop);
        ms_tiled /= 20;

        // Time cuBLAS
        cudaEventRecord(start);
        for (int i = 0; i < 20; i++)
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms_cublas; cudaEventElapsedTime(&ms_cublas, start, stop);
        ms_cublas /= 20;

        double flops = 2.0 * M * N * K;
        double tflops_tiled  = flops / (ms_tiled  * 1e9);
        double tflops_cublas = flops / (ms_cublas * 1e9);

        printf("N=%4d | %5.2f ms  %4.2f TFLOPS | %5.2f ms  %4.2f TFLOPS | %.1fx\n",
               M, ms_tiled, tflops_tiled, ms_cublas, tflops_cublas, ms_tiled/ms_cublas);

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    }
    cublasDestroy(handle);
    return 0;
}
