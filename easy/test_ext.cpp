#include <torch/extension.h>

// 声明在 .cu 文件中定义的函数
void square_cuda_launcher(torch::Tensor input, torch::Tensor output);

// C++ 接口函数（Python 将调用这个）
torch::Tensor square_forward(torch::Tensor input) {
    // 1. 检查输入是否在 GPU 上
    TORCH_CHECK(input.is_cuda(), "Input tensor must be a CUDA tensor");
    // 2. 检查输入是否连续（非常重要，否则指针运算会出错）
    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");

    auto output = torch::empty_like(input);
    
    // 调用 CUDA 启动器
    square_cuda_launcher(input, output);
    
    return output;
}

// PyBind11 模块定义
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("square", &square_forward, "A custom square operator (CUDA)");
}