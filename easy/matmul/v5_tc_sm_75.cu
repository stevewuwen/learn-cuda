#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// --- 配置参数 ---
// 整个 Block 处理的矩阵大小 (M, N)
#define BM 128
#define BN 128
// K 维度的分块大小 (每次循环加载的深度)
// T4 Tensor Core 基础粒度是 16，这里取 32 以平衡 Shared Memory 大小和加载效率
#define BK 32 

// 每个 Warp 处理的区域大小
#define WM 64
#define WN 32

// Warp 的布局: 256 线程 = 8 Warps
// 我们将 8 个 Warps 排列成 2 行 4 列 (2 * 64 = 128, 4 * 32 = 128)
#define WARPS_M 2
#define WARPS_N 4

// Shared Memory Padding (避免 Bank Conflict)
// half 类型占 2 字节，32个 half 是 64 字节。+8 偏移量可以错开 Bank。
#define PAD 8

__global__ void matrix_multiplication_tensor_core(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // --- 1. 声明 Tensor Core 片段 ---
    // WMMA 形状: 16x16x16
    // Frag_A matrix_a, Frag_B matrix_b, Accumulator accumulator
    // 输入必须是 half (fp16)，累加器是 float (fp32)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag[WM/16][BK/16];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[BK/16][WN/16];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag[WM/16][WN/16];

    // 初始化累加器为 0
    for (int i = 0; i < WM/16; i++) {
        for (int j = 0; j < WN/16; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // --- 2. 声明 Shared Memory ---
    // A: BM x BK, B: BK x BN
    // 使用 half 存储以供 Tensor Core 使用
    __shared__ half As[BM][BK + PAD];
    __shared__ half Bs[BK][BN + PAD];

    // --- 3. 线程索引计算 ---
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int warpId = tx / 32;

    // 当前 Warp 在 Block 内负责的区域索引 (grid of warps)
    int warp_row = warpId / WARPS_N; // 0 or 1
    int warp_col = warpId % WARPS_N; // 0, 1, 2, 3

    // --- 4. 主循环 (K 维度分块) ---
    // 每次推进 BK 步
    for (int k_step = 0; k_step < K; k_step += BK) {

        // === 加载数据 Global -> Shared 并转换 Float -> Half ===
        // 每个线程负责加载一部分 A 和 B
        // Block 大小 256 线程。
        // A Tile: 128 * 32 = 4096 元素。 4096 / 256 = 16 元素/线程 (4个 float4)
        // B Tile: 32 * 128 = 4096 元素。 16 元素/线程

        // 加载 A (BM x BK)
        // 映射线程到 As 的坐标
        // 我们视 As 为 (128, 32)。线程线性展开。
        // 每个线程加载 4 个 float4 (16 floats)
        int tid = tx;
        int load_a_row = tid / (BK / 4); // BK=32, BK/4=8. row = tid / 8 (0~31) ?? 
        // 256 threads loading 128 rows is tricky with simple divide.
        // Let's use a stride loop for safety and flexibility

        // 加载 A: 每个人加载 4 个 float, 循环 4 次 -> 16 floats? No.
        // As size: 128*32 = 4096. Threads: 256. Elements per thread: 16.
        // 使用 float4 加载，每个线程需要执行 4 次 float4 加载

        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            // 我们把 128x32 看作线性的一维数组进行索引，然后映射回二维
            // 总共 128 行，每行 32 元素。
            // 每次 float4 加载 4 个元素。总共需要 128 * (32/4) = 1024 次 float4 加载。
            // 256 线程，每个线程负责 4 次。

            int linear_idx = i * 256 + tid; // 0 ~ 1023

            // 转换 linear_idx 到 (row, col_chunk)
            // 一行有 32/4 = 8 个 float4 块
            int row = linear_idx / 8;
            int col = (linear_idx % 8) * 4;

            // 边界检查
            if (row < BM && col < BK) {
                int global_row = by * BM + row;
                int global_col = k_step + col;

                float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < M && global_col < K) {
                    // 注意：这里假设 K 是 4 的倍数，否则边缘需要特殊处理
                    // 为简化代码，这里使用 reinterpret_cast
                    tmp = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
                }

                // 转换 float4 -> half (手动逐个转换或使用 intrinsics)
                As[row][col + 0] = __float2half(tmp.x);
                As[row][col + 1] = __float2half(tmp.y);
                As[row][col + 2] = __float2half(tmp.z);
                As[row][col + 3] = __float2half(tmp.w);
            }
        }

        // 加载 B (BK x BN) -> (32 x 128)
        // 同样 4096 元素，每线程 16 元素
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx = i * 256 + tid;
            // 通过 linear_idx 计算在Bs里面的索引
            // 一行有 128/4 = 32 个 float4 块
            int row = linear_idx / 32;
            int col = (linear_idx % 32) * 4;

            if (row < BK && col < BN) {
                int global_row = k_step + row;
                int global_col = bx * BN + col;

                float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < K && global_col < N) { // 注意 B 是 N x K 还是 K x N? 题目中 B 是 [K][N] 布局 (row-major)
                    // 题目原代码 B_ptr 移动逻辑: B + load_b_row * K + col. 
                    // 似乎题目原代码 B 是 [N][K] 但被视作 [N_dim][K_dim] 进行 GEMM? 
                    // 标准 GEMM: C = A(MxK) * B(KxN)。B 假如是 RowMajor，则 B[row][col] = B[k][n]。
                    // 让我们按标准 RowMajor B(K, N) 处理。
                    tmp = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                }

                Bs[row][col + 0] = __float2half(tmp.x);
                Bs[row][col + 1] = __float2half(tmp.y);
                Bs[row][col + 2] = __float2half(tmp.z);
                Bs[row][col + 3] = __float2half(tmp.w);
            }
        }

        __syncthreads();

        // === Tensor Core 计算 ===

        // 每个 Warp 计算输出矩阵 C 的一块 (WM x WN) = (64 x 32)
        // 需要在 K 维度上进一步切分 (每次切 16)

        // 内部 K 循环 (BK / 16 = 2 次)
        for (int ki = 0; ki < BK / 16; ++ki) {

            // 1. 加载 A 的片段 (Warp 负责的行)
            // Warp 负责的行偏移: warp_row * WM
            // A fragment 布局: 64行，按 16行一块加载 -> 4块
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                // Shared Memory 地址: As[base_row + i*16][base_k]
                int row_idx = warp_row * WM + i * 16;
                int col_idx = ki * 16;
                // wmma::load_matrix_sync(dst, src_ptr, stride_in_elements)
                // Stride 是 Shared Memory 的列宽 (BK + PAD)
                wmma::load_matrix_sync(a_frag[i][ki], &As[row_idx][col_idx], BK + PAD);
            }

            // 2. 加载 B 的片段 (Warp 负责的列)
            // Warp 负责的列偏移: warp_col * WN
            // B fragment 布局: 32列，按 16列一块加载 -> 2块
            #pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                int row_idx = ki * 16;
                int col_idx = warp_col * WN + j * 16;
                wmma::load_matrix_sync(b_frag[ki][j], &Bs[row_idx][col_idx], BN + PAD);
            }

            // 3. 矩阵乘法 (Outer Product)
            // Accumulator += A_frag * B_frag
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                #pragma unroll
                for (int j = 0; j < WN / 16; ++j) {
                    // 对应位置相乘
                    wmma::mma_sync(c_frag[i][j], a_frag[i][ki], b_frag[ki][j], c_frag[i][j]);
                }
            }
        }

        __syncthreads();
    }

    // --- 5. 写回结果 ---
    // Tensor Core 的结果现在在寄存器 (c_frag) 中。
    // 需要先存回 Shared Memory (或者直接用 store_matrix_sync 存到 Global, 但通常这里需要处理 Layout)
    // 为简单起见，利用 Shared Memory 做中转，按 float 写回 Global

    // 这里我们复用 As 或 Bs 的空间，或者直接计算 Global 指针写回。
    // WMMA store 要求 stride。
    // 让我们直接计算 Global Memory 地址并 Store (需要仔细处理 Stride)
    // C 是 Row Major (M x N)。Stride = N。

    // 计算当前 Warp 在 Global C 中的起始位置
    int c_base_row = by * BM + warp_row * WM;
    int c_base_col = bx * BN + warp_col * WN;

    #pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) {
            int row = c_base_row + i * 16;
            int col = c_base_col + j * 16;

            // 边界检查 (以 16x16 块为单位)
            if (row < M && col < N) {
                // 注意：store_matrix_sync 直接写回 Global Memory
                // 如果 C 的内存没有对齐到 16 bytes 或者 N 不是 8/16 的倍数，可能需要注意。
                // 这里的 layout 是 mem_row_major
                wmma::store_matrix_sync(&C[row * N + col], c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    }
}

extern "C" void solve(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // Block Size: 256 threads (8 warps)
    dim3 threadsPerBlock(256);

    // Grid Size
    dim3 numBlocks((N + BN - 1) / BN, (M + BM - 1) / BM);

    matrix_multiplication_tensor_core<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
