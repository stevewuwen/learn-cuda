#include <stdio.h>
#include <cuda_runtime.h>
#define TILE_WIDTH 32 // 假设 BlockDim 为 32x32

__global__ void matrix_multiplication_shared_mem(const float *__restrict__ A, const float *__restrict__ B, float *C, int M, int N, int K)
{
    // 申请共享内存
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int ty = threadIdx.y;
    int tx = threadIdx.x;

    int row = blockIdx.y * TILE_WIDTH + ty;
    int col = blockIdx.x * TILE_WIDTH + tx;

    float acc = 0.0f;

    for (int i = 0; i < (N + TILE_WIDTH - 1) / TILE_WIDTH; i++)
    {
        // 读取数据
        if (row < M && TILE_WIDTH * i + tx < N)
        {
            As[ty][tx] = A[row * N + TILE_WIDTH * i + tx];
        }
        else
        {
            As[ty][tx] = 0.0f;
        }
        if (col < K && TILE_WIDTH * i + ty < N)
        {
            Bs[tx][ty] = B[(TILE_WIDTH * i + ty) * K + col];
        }
        else
        {
            Bs[tx][ty] = 0.0f;
        }
        __syncthreads(); // 等待线程块里面的线程都搬运完成

        for (int j = 0; j < TILE_WIDTH; j++)
        {
            acc += As[ty][j] * Bs[tx][j]; //访问线程块里面的共享内存时，没有合并访问，只在乎bank config
        }

        __syncthreads();
    }
    if (row<M && col<K)
    {
        C[row*K+col] = acc;
    }
}

// 宿主端 wrapper 函数
extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K)
{
    dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
    // 注意：grid 的计算需要向上取整，你的代码已经包含了这个逻辑，但建议加上括号保证运算顺序
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_shared_mem<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    // 检查是否有错误发生（调试用）
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }

    // 等待 GPU 完成
    cudaDeviceSynchronize();
}
