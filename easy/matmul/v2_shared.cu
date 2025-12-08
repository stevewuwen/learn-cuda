#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 32

__global__ void matrix_multiplication_shared_mem(const float* __restrict__ A, const float* __restrict__ B, float* C, int M, int N, int K) {

    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float acc = 0.0f;

    // 循环遍历所有的 Tile
    for (int t = 0; t < (N + TILE_WIDTH - 1) / TILE_WIDTH; ++t) {

        if (row < M && t * TILE_WIDTH + tx < N) {
            As[ty][tx] = A[row * N + t * TILE_WIDTH + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        if (col < K && t * TILE_WIDTH + ty < N) {
            // 注意这里 B 的索引：行是 t*TILE_WIDTH + ty, 列是 col
            Bs[ty][tx] = B[(t * TILE_WIDTH + ty) * K + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; ++i) {
            acc += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }

    if (row < M && col < K) {
        C[row * K + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_shared_mem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    cudaDeviceSynchronize();
}
