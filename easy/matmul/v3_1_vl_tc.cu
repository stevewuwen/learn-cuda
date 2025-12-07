#include <stdio.h>
#include <cuda_runtime.h>

// 宏定义块大小
// TS (Tile Size): 每个 Block 计算 128x128 的 C
// WPT (Work Per Thread): 每个线程计算 8x8 的 C
// TS_K: 共享内存中的分块大小
#define TS 128
#define WPT 8
#define TS_K 16 

__global__ void matrix_multiplication_optimized(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    __shared__ float As[TS][TS_K];
    __shared__ float Bs[TS_K][TS];

    float accum[WPT][WPT] = {0.0f};

    float reg_A[WPT];
    float reg_B[WPT];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 当前线程负责的 C 矩阵起始坐标 (C 的分块左上角 + 线程偏移)
    int row_c = by * TS + ty * WPT; 
    int col_c = bx * TS + tx * WPT;

    for (int t = 0; t < N; t += TS_K) {

        int tid = ty * (TS / WPT) + tx;

        int load_a_row = tid / (TS_K / WPT); 
        int load_a_col = (tid % (TS_K / WPT)) * WPT;

        if (by * TS + load_a_row < M && t + load_a_col < N) {
             // 使用 float4 指针强转进行向量加载
             // 1. 计算内存中的基础偏移量 (base index)
            int base_offset = (by * TS + load_a_row) * N + (t + load_a_col);

            // 2. 将指针强转为 float4* 并读取两个连续的 float4
            // A[base_offset] 是起点
            const float4* A_ptr = reinterpret_cast<const float4*>(&A[base_offset]);

            float4 tmp1 = A_ptr[0]; // 加载第 0-15 字节 (float 0-3)
            float4 tmp2 = A_ptr[1]; // 加载第 16-31 字节 (float 4-7)

            // 3. 将数据写入共享内存 As
            // 优化：同样使用 float4 向量化写入共享内存，比逐个 float 赋值更快
            float4* As_ptr = reinterpret_cast<float4*>(&As[load_a_row][load_a_col]);

            As_ptr[0] = tmp1; // 写入 As[row][col + 0~3]
            As_ptr[1] = tmp2; // 写入 As[row][col + 4~7]
        }

        // 加载 Bs (B 的子块): [16][64]
        // 同样用 tid 映射。每行 64 个元素 = 16 个 float4。
        // 总共 16 行。总 float4 数 = 16 * 16 = 256。正好每个线程取 1 个 float4。
        int load_b_row = tid / (TS / WPT);
        int load_b_col = (tid % (TS / WPT)) * WPT;

        if (t + load_b_row < N && bx * TS + load_b_col < K) {
            // 使用 float4 指针强转进行向量加载
            // 1. 计算内存中的基础偏移量 (base index)
            int base_offset = (t + load_b_row) * K + (bx * TS + load_b_col);

            // 2. 将指针强转为 float4* 并读取两个连续的 float4
            // A[base_offset] 是起点
            const float4* B_ptr = reinterpret_cast<const float4*>(&B[base_offset]);

            float4 tmp1 = B_ptr[0]; // 加载第 0-15 字节 (float 0-3)
            float4 tmp2 = B_ptr[1]; // 加载第 16-31 字节 (float 4-7)

            // 3. 将数据写入共享内存 As
            // 优化：同样使用 float4 向量化写入共享内存，比逐个 float 赋值更快
            float4* Bs_ptr = reinterpret_cast<float4*>(&Bs[load_b_row][load_b_col]);

            Bs_ptr[0] = tmp1; // 写入 As[row][col + 0~3]
            Bs_ptr[1] = tmp2; // 写入 As[row][col + 4~7]
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
extern "C" void solve(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // 线程块大小: 16x16 = 256 线程
    dim3 threadsPerBlock(TS / WPT, TS / WPT); 

    // Grid 大小: 因为每个 Block 处理 64x64，所以除以 TS(64)
    dim3 numBlocks((K + TS - 1) / TS, (M + TS - 1) / TS);

    matrix_multiplication_optimized<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
