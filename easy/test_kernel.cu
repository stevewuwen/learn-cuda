#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Kernel 函数
template <typename scalar_t>
__global__ void square_cuda_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    int size) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx] * input[idx];
    }
}

// C++ 调用 CUDA 的辅助函数
void square_cuda_launcher(torch::Tensor input, torch::Tensor output) {
    const int size = input.numel();
    const int threads = 256;
    const int blocks = (size + threads - 1) / threads;

    // AT_DISPATCH_FLOATING_TYPES 是 PyTorch 提供的宏，用于自动处理 float/double 类型分发
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "square_cuda", ([&] {
        square_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));
}