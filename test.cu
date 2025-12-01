#include <iostream>
#include <cmath>  
#include <algorithm> 
#include <cuda_runtime.h> // 通常包含这个以明确使用CUDA Runtime API

// 定义一个简单的错误检查宏（可选，但强烈建议）
#define CHECK_CUDA(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        std::cerr << "Error: " << __FILE__ << ":" << __LINE__ << ", " \
                << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
}

__global__ void matrix_multiplication_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    int row =  blockDim.y*blockIdx.y + threadIdx.y;
    if(row>=M){
        return;
    }
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if(col>=K){
        return;
    }
    int index = row*M+col;
    C[index] = 0;
    for(int i=0;i<N;i+=1){
        C[index] += A[row*M+i]*B[i*N+col]; 
    }
}

    int main(){
        int N = 3;
        float* a = new float[N];
        float* b = new float[N];
        float* c = new float[N];
        a[0] = 1.0;
        a[1] = 2.0;
        a[2] = 3.0;
        b[0] = 4.0;
        b[1] = 5.0;
        b[2] = 6.0;
        float* p1;
        float* p2;
        float* p3;
        cudaMalloc(&p1, sizeof(int)*N);
        cudaMalloc(&p2, sizeof(int)*N);
        cudaMalloc(&p3, sizeof(int)*N);
        int blockSize = 256;
        int blockNum = (N+blockSize-1)/blockSize;
        cudaMemcpy(p1, a, N*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(p2, b, N*sizeof(int), cudaMemcpyHostToDevice);
        matrix_multiplication_kernel<<<blockNum , blockSize>>>(p1,p2,p3,N, 1, N);

        CHECK_CUDA(cudaGetLastError());
        // 同步
        CHECK_CUDA(cudaDeviceSynchronize());
        cudaDeviceSynchronize();
        cudaMemcpy(c, p3, N*sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        float maxError = 0;
        for(int i=0;i<N;i+=1){
            float diff = std::abs(c[i] - 6); 
            if(diff>maxError){
                std::cout<<diff<<std::endl;
                maxError = diff;
            }
        }
        std::cout<<"max error: "<<maxError<<std::endl;
        cudaFree(p1);
        cudaFree(p2);
        cudaFree(p3);
        delete[] a;
        delete[] b;
        delete[] c;
        return 0;
    }