#include <iostream>
#include <cmath>      // For fabs
#include <algorithm>  // For std::max

__global__ void add(int *a, int *b, int *c, int N){
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    if(index < N){
        c[index] = a[index] + b[index];
    }
    __syncthreads(); // 等待线程块里面的所有线程执行完毕，进行同步
}

int main(){
    int N = 2000;
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];
    for (int i = 0; i < N; i++)
    {
        a[i] = 5;
        b[i] = 1;
        c[i] = 0;
    }
    int *p1;
    int *p2;
    int *p3;
    cudaMalloc(&p1, sizeof(int)*N);
    cudaMalloc(&p2, sizeof(int)*N);
    cudaMalloc(&p3, sizeof(int)*N);
    int blockSize = 256;
    int blockNum = (N+blockSize-1)/blockSize;
    cudaMemcpy(p1, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p2, b, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(p3, c, N*sizeof(int), cudaMemcpyHostToDevice);
    add<<<blockNum , blockSize>>>(p1,p2,p3,N);
    cudaDeviceSynchronize();
    cudaMemcpy(c, p3, N*sizeof(int), cudaMemcpyDeviceToHost);
    float maxError = 0;
    for(int i=0;i<N;i+=1){
        if(fabs(c[i]-6)>maxError){
            std::cout<<fabs(c[i]-6)<<std::endl;
        }
        maxError = max(fabs(c[i]-6), maxError);
    }
    std::cout<<"max error: "<<maxError<<std::endl;
    cudaFree(&p1);
    cudaFree(&p2);
    cudaFree(&p3);
}