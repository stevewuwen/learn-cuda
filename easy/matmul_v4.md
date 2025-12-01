这是一个非常好的学习案例。你提供的代码是一个标准的“共享内存分块（Shared Memory Tiling）”实现，这通常是学习 CUDA 优化的第一步。它的性能比纯 Global Memory 版本好很多，但离硬件极限还有很大差距。

我们需要解决该代码中的两个主要瓶颈：
1.  **访存带宽未跑满**：一次只加载一个 `float`，利用率低。
2.  **计算密度（Arithmetic Intensity）低**：每个线程只计算 C 中 **1个** 元素。这意味着每做几次乘加运算，就要去共享内存甚至全局内存读数据。内存读取的速度远慢于计算速度，导致计算单元（CUDA Core）大部分时间在等待数据。

---

### 优化思路：增加“每个线程的工作量” (Thread Coarsening)

我们要让每个线程不再只计算 1 个点，而是计算 **一个 4x4 的小块**（或者 8x8）。
*   **原理**：如果一个线程计算 C 中的 4x4 个元素，它只需要加载 A 的 4 个值和 B 的 4 个值到寄存器中，就可以进行 $4 \times 4 = 16$ 次乘加运算。
*   **好处**：大幅减少了从 Shared Memory 读取数据的次数，利用**寄存器（Register）**极高的速度来复用数据。
*   **辅助优化**：使用 `float4` 进行向量化加载，一次读取 128 位数据，提高全局内存带宽利用率。

---

### 优化后的代码

这里我们采用 **Register Tiling（寄存器分块）** 技术。
为了方便理解，我保留了你的变量命名习惯（虽然数学上通常称 Inner Dimension 为 K，但你的代码里是 N，我沿用你的定义：$A[M][N] \times B[N][K] = C[M][K]$）。

**假设条件**：为了代码简洁，假设 M, N, K 均为 64 的倍数（实际工程中需要处理边界 Padding）。

```cpp
#include <stdio.h>
#include <cuda_runtime.h>

// 宏定义块大小
// TS (Tile Size): 每个 Block 计算 64x64 的 C
// WPT (Work Per Thread): 每个线程计算 4x4 的 C
// TS_K: K 维度(你的代码里是 N 维度)的分块大小，设为 8 或 16
#define TS 64
#define WPT 4
#define TS_K 16 

// 优化后的 Kernel
__global__ void matrix_multiplication_optimized(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // 每个 Block 处理 C 中 TS x TS (64x64) 的区域
    // 线程块维度: dim3(TS/WPT, TS/WPT) -> (16, 16) -> 256 个线程
    
    // 1. 声明共享内存
    // As: 存储 A 的切片 [TS][TS_K] -> [64][16]
    // Bs: 存储 B 的切片 [TS_K][TS] -> [16][64]
    __shared__ float As[TS][TS_K];
    __shared__ float Bs[TS_K][TS];

    // 2. 声明寄存器
    // accum: 累加器，每个线程负责计算 4x4 = 16 个元素
    float accum[WPT][WPT] = {0.0f};
    
    // reg_A, reg_B: 用于在内循环中缓存从 SMEM 读取的值
    float reg_A[WPT];
    float reg_B[WPT];

    // 线程 ID 和 Block ID
    int tx = threadIdx.x; // range 0-15
    int ty = threadIdx.y; // range 0-15
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 当前线程负责的 C 矩阵起始坐标 (C 的分块左上角 + 线程偏移)
    // 每个线程覆盖 WPT(4) 个像素宽/高
    int row_c = by * TS + ty * WPT; 
    int col_c = bx * TS + tx * WPT;

    // 3. 循环遍历 N 维度 (步长 TS_K = 16)
    for (int t = 0; t < N; t += TS_K) {
        
        // --- 加载数据到 Shared Memory (协作加载) ---
        // 我们有 256 个线程。
        // 需要加载 A 的 Tile: 64行 * 16列 = 1024 元素。每个线程加载 1024/256 = 4 个元素。
        // 需要加载 B 的 Tile: 16行 * 64列 = 1024 元素。每个线程加载 4 个元素。
        
        // 加载 As (A 的子块): 
        // 这里的逻辑是将 256 个线程映射到 64x16 的区域
        // 我们使用 float4 向量化加载来极致优化带宽
        
        // 计算当前线程加载 As 的位置
        // 将 16x16 的线程块视为 256 个线性线程
        int tid = ty * (TS / WPT) + tx; // 0 ~ 255
        
        // 映射到 As[64][16]: 每一行 16 个元素，如果是 float4 就是 4 个 float4
        // 256 个线程，每个加载 1 个 float4 (4个float)，正好 1024 个 float
        // As 的行索引
        int load_a_row = tid / (TS_K / 4); 
        int load_a_col = (tid % (TS_K / 4)) * 4;
        
        // 从全局内存 A 加载到 As
        // 全局索引: A[(by * TS + load_a_row) * N + (t + load_a_col)]
        // 注意边界检查省略了，假设维度对其
        if (by * TS + load_a_row < M && t + load_a_col < N) {
             // 使用 float4 指针强转进行向量加载
             float4 tmp = reinterpret_cast<const float4*>(&A[(by * TS + load_a_row) * N + (t + load_a_col)])[0];
             As[load_a_row][load_a_col + 0] = tmp.x;
             As[load_a_row][load_a_col + 1] = tmp.y;
             As[load_a_row][load_a_col + 2] = tmp.z;
             As[load_a_row][load_a_col + 3] = tmp.w;
        }

        // 加载 Bs (B 的子块): [16][64]
        // 同样用 tid 映射。每行 64 个元素 = 16 个 float4。
        // 总共 16 行。总 float4 数 = 16 * 16 = 256。正好每个线程取 1 个 float4。
        int load_b_row = tid / (TS / 4);
        int load_b_col = (tid % (TS / 4)) * 4;

        if (t + load_b_row < N && bx * TS + load_b_col < K) {
             float4 tmp = reinterpret_cast<const float4*>(&B[(t + load_b_row) * K + (bx * TS + load_b_col)])[0];
             Bs[load_b_row][load_b_col + 0] = tmp.x;
             Bs[load_b_row][load_b_col + 1] = tmp.y;
             Bs[load_b_row][load_b_col + 2] = tmp.z;
             Bs[load_b_row][load_b_col + 3] = tmp.w;
        }

        __syncthreads(); // 等待数据加载完成

        // --- 在寄存器上进行计算 ---
        // 遍历 Shared Memory 中的 TS_K (16) 维度
        #pragma unroll
        for (int k = 0; k < TS_K; ++k) {
            
            // 1. 将所需的 As 和 Bs 数据预加载到寄存器
            // 每个线程计算 4x4，需要 As 的一列 4 个值，Bs 的一行 4 个值
            for (int i = 0; i < WPT; ++i) {
                reg_A[i] = As[ty * WPT + i][k];
                reg_B[i] = Bs[k][tx * WPT + i];
            }

            // 2. 外积计算 (Outer Product)
            // 计算 4x4 的结果，复用 reg_A 和 reg_B
            for (int row = 0; row < WPT; ++row) {
                for (int col = 0; col < WPT; ++col) {
                    accum[row][col] += reg_A[row] * reg_B[col];
                }
            }
        }
        
        __syncthreads(); // 等待计算完成，准备加载下一块
    }

    // 4. 写回结果到全局内存
    // 每个线程写回 4x4 个点
    for (int row = 0; row < WPT; ++row) {
        for (int col = 0; col < WPT; ++col) {
            int global_row = row_c + row;
            int global_col = col_c + col;
            
            if (global_row < M && global_col < K) {
                C[global_row * K + global_col] = accum[row][col];
            }
        }
    }
}

// Host 端调用示例
void run_kernel(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // 线程块大小: 16x16 = 256 线程
    dim3 threadsPerBlock(TS / WPT, TS / WPT); 
    
    // Grid 大小: 因为每个 Block 处理 64x64，所以除以 TS(64)
    dim3 numBlocks((K + TS - 1) / TS, (M + TS - 1) / TS);

    matrix_multiplication_optimized<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
```

### 为什么这个版本更快？

1.  **寄存器级复用 (最重要的优化)**
    *   **原代码**：计算 C 中 1 个点，需要读 1 次 A，读 1 次 B。
    *   **新代码**：计算 C 中 $4 \times 4 = 16$ 个点。
        *   内层循环 `k` 的每一步：我们从 Shared Memory 读取 **4个** A值和 **4个** B值（共8个读操作）。
        *   然后我们在寄存器中进行了 $4 \times 4 = 16$ 次乘法累加。
        *   **比率提升**：以前是 2次读取:1次计算。现在是 8次读取:16次计算。**计算密度提升了2倍，访存压力减半。**

2.  **向量化加载 (`float4`)**
    *   GPU 的显存总线非常宽。如果你一次只读一个 `float` (4 bytes)，就像开着大卡车只拉一箱快递。
    *   使用 `reinterpret_cast<float4*>` 一次性读取 16 bytes，相当于让指令流水线更饱满，大幅提升 Global Memory 到 Shared Memory 的传输带宽。

3.  **减少了 Block 数量**
    *   原来的 `TILE_WIDTH=32`，每个 Block 处理 $32 \times 32$ 个元素。
    *   现在 `TS=64`，每个 Block 处理 $64 \times 64$ 个元素。
    *   更大的分块通常意味着更好的 Shared Memory 数据复用率。

### 学习建议

这段代码比原来的版本复杂了不少，建议按以下步骤消化：
1.  **画图理解 WPT (Work Per Thread)**：画一个 64x64 的方块，里面切分成 16x16 个小格子（代表线程），每个小格子里面再画 4x4 的点（代表该线程负责计算的 C 元素）。
2.  **理解加载逻辑**：注意看 `tid` 是如何映射到 `As` 和 `Bs` 的坐标的。这里的核心思想是：加载数据时的线程布局（为了连续读取）和计算时的线程布局（为了计算 4x4 块）是可以不一样的，只要最后存到 Shared Memory 的位置对就行。
3.  **注意对齐**：为了使用 `float4`，矩阵的列数（stride）通常要求是 4 的倍数。如果不是，需要回退到普通的 float 读取，或者在 Host 端对矩阵进行 Padding（补零）。