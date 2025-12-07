#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row =  blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(col >= K || row >= M){
        return;
    }

    float acc = 0.0f;
    for(int i = 0; i < N; i++){
        acc += A[row * N + i] * B[i * K + col];
    }
    C[row * K + col] = acc; 
}

// 宿主端 wrapper 函数
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((K + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (M + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrix_multiplication_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, M, N, K);

    // 检查是否有错误发生
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}
