**Role:** 你是一位精通 CUDA 性能优化的资深工程师。
**Task:**
我正在学习 CUDA C++，目标是在 NVIDIA T4 (Compute Capability 7.5, CUDA 12.8) 硬件上实现一个高性能的矩阵乘法 (GEMM) 内核，并尝试接近或超越 `torch.mm` 的性能。
**Requirements:**
1.  **算法优化**：请不要写朴素的矩阵乘法。请实现基于 **Shared Memory Tiling (分块)** 的算法。如果可能，请展示如何利用 T4 的 **Tensor Cores (WMMA API)** 进行 FP16 运算，或者使用高度优化的 FP32 算法（包含 Double Buffering 或 向量化内存访问）。
2.  **接口规范**：输出一个完整的 `.cu` 文件。
    - 必须包含 `extern "C"` 接口，以便 Python `ctypes` 调用。
    - 接口应接收：指向 A, B, C 矩阵的指针 (float* 或 half*)，以及 M, N, K 维度。
3.  **健壮性**：代码中需要包含必要的 CUDA Error Check 宏。
4.  **测试脚本**：请额外提供一个 Python 脚本。
    - 使用 `nvcc` 动态编译 `.cu` 文件为 `.so`。
    - 使用 `ctypes` 加载并调用该内核。
    - 生成随机 Tensor，对比你的 Kernel 与 `torch.mm` 的计算结果（验证正确性）。
    - 进行 Benchmark，计算两者的执行时间（ms）和 TFLOPS，看我是否能在 T4 上逼近 cuBLAS 的性能。
**Context:**
- 硬件：Tesla T4
- 环境：CUDA 12.8
- 目标矩阵大小：假设 M=4096, N=4096, K=4096 (或适合 T4 负载的大小)。



**Role:** 你是一位精通 NVIDIA GPU 体系结构（特别是 Turing 架构）和 CUDA 汇编级优化的资深 HPC 工程师。你非常熟悉 cuBLAS 的底层实现原理。

**Objective:** 在 NVIDIA Tesla T4 (Compute Capability 7.5) 上实现一个高性能 GEMM Kernel，目标是尽可能逼近 `torch.mm` (cuBLAS) 的性能。

**Hardware Context (Deep Search Requirement):**
请首先针对 **Tesla T4** 进行技术调研/思考：
1.  T4 的 SM 数量、L1/Shared Memory 容量限制、Register File 大小。
2.  Turing 架构下 Tensor Core (WMMA) 的最佳指令配置（例如 `wmma::m16n16k16`）。
3.  Turing 架构不支持 `cp.async` (Async Copy)，需要使用什么样的 **Software Pipelining (软件流水线)** 策略来掩盖全局内存延迟？

**Task Requirements:**

1.  **Kernel Implementation (CUDA C++)**:
    * **核心算法**：实现基于 **WMMA (Warp Matrix Multiply Accumulate)** API 的 FP16 矩阵乘法。
    * **内存优化**：
        * **Global Memory**: 必须使用 `float4` (或 `int4` 用于 half 对) 向量化加载 (Vectorized Load, 128-bit) 以满足内存合并访问 (Coalescing)。
        * **Shared Memory**: 实现 **Double Buffering (双缓冲)** 或多级流水线以掩盖加载延迟。
        * **Bank Conflict**: 必须实现 **Shared Memory Swizzling (XOR 映射)** 或 Padding 来彻底消除 Bank Conflicts。
    * **并行策略**：精细设计的 Block Size 和 Warp Tiling 策略（例如 128x128 的 Block Tile，配合 Warp 级别的分块）。
    * **接口**：`extern "C"` 接口，接收 `half* A`, `half* B`, `half* C` (或 float C), 以及 M, N, K。

2.  **Python Inference & Benchmark Script**:
    * 使用 `nvcc` 启用架构标志 `-arch=sm_75` 编译为 `.so`。
    * 使用 `ctypes` 调用。
    * **正确性验证**：允许 1e-2 级别的误差（FP16 精度限制）。
    * **性能测试**：
        * 使用 `torch.cuda.Event` 进行精确计时（warm-up 10次，run 100次）。
        * 计算 TFLOPS，并输出 **Performance relative to torch.mm (%)**。

3.  **Analysis Report**:
    * 解释你选择的 Block Size 和 Thread 布局的数学依据（Occupancy vs Register Pressure）。
    * 解释如何在没有 `cp.async` 的 T4 上实现延迟隐藏（Prefetching 策略）。

**Constraints:**
* Target: CUDA 12.8 / Tesla T4 (sm_75).
* Matrix Size: M=4096, N=4096, K=4096.
* Input/Output: FP16 输入，FP16 或 FP32 累加 (Accumulator)。