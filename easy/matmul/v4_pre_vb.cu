#include <stdio.h>
#include <cuda_runtime.h>

// 宏定义块大小
#define TS 128      // Tile Size M, N
#define TS_K 16     // Tile Size K
#define WPT 8       // Work Per Thread

__global__ void matrix_multiplication_double_buffer(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    // Double Buffering: 申请 2 倍的 Shared Memory
    // [2] 代表两个缓冲区：buffer_curr 和 buffer_next
    __shared__ float As[2][TS][TS_K];
    __shared__ float Bs[2][TS_K][TS];

    // 寄存器累加器
    float accum[WPT][WPT] = {0.0f};

    // 用于计算的寄存器片段
    float reg_A[WPT];
    float reg_B[WPT];

    // 用于预取 Global Memory 数据的寄存器
    // 每个线程负责加载 8 个 float (2个 float4)
    float ldg_a_reg[2][4]; // [2个float4][每个包含4个float]
    float ldg_b_reg[2][4];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 在线程块中取出这个线程的索引， TS/WPT 表示一行有多少个线程（写回的字节块宽度/每一个线程处理的字节数）
    int tid = ty * (TS / WPT) + tx; // 0 ~ 255

    // --- 1. 预计算加载索引 ---
    // 线程负责加载 A 的位置: A_tile 是 128x16
    // 每个线程加载 8 个元素。总共 256 线程 * 8 = 2048 元素 = 128*16。
    // A 的行索引 (0-127) 和 列索引 (0, 8)
    int load_a_row = tid >> 1; // tid / 2
    int load_a_col = (tid & 1) << 3; // (tid % 2) * 8

    // 线程负责加载 B 的位置: B_tile 是 16x128
    // B 的行索引 (0-15) 和 列索引 (0-120，步长8)
    int load_b_row = tid >> 4; // tid / 16
    int load_b_col = (tid & 15) << 3; // (tid % 16) * 8

    // A 和 B 在 Global Memory 中的基础指针（调整到当前 Block 的行/列起点）
    // 注意：这里假设 A是(M, N/K_dim)，B是(N/K_dim, K/N_width)。
    // 根据用户代码逻辑：
    // A: [M][N_arg], B: [N_arg][K_arg] (这里 N_arg 对应 K 维度)
    // 使用题目中的变量名：
    // A 的行偏移由 by 决定，B 的列偏移由 bx 决定
    const float* A_ptr_base = A + (by * TS + load_a_row) * N; 
    const float* B_ptr_base = B + load_b_row * K + (bx * TS + load_b_col);

    // --- 2. Prologue: 加载第一个 Tile 到寄存器，然后放入 SMEM ---

    // 边界检查并加载 A
    {
        // 这里的 t=0，对应 A 的列偏移 load_a_col
        bool row_valid = (by * TS + load_a_row < M);
        // 使用 float4 加载
        const float4* A_vec_ptr = reinterpret_cast<const float4*>(A_ptr_base + load_a_col);
        // 只能在内存安全时读取，否则填 0
        float4 tmp1 = (row_valid && (load_a_col < N)) ? A_vec_ptr[0] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 tmp2 = (row_valid && (load_a_col + 4 < N)) ? A_vec_ptr[1] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

        // 写入 SMEM buffer 0
        // 手动展开 float4 赋值以匹配 float 数组结构，或者转换指针
        // 这里为了 Padding 对齐安全，建议逐个或 reinterpret_cast 写入
        // As[0][load_a_row][load_a_col + 0..3]
        reinterpret_cast<float4*>(&As[0][load_a_row][load_a_col])[0] = tmp1;
        reinterpret_cast<float4*>(&As[0][load_a_row][load_a_col + 4])[0] = tmp2;
    }

    // 边界检查并加载 B
    {
        int col_B = bx * TS + load_b_col;
        // t=0
        const float4* B_vec_ptr = reinterpret_cast<const float4*>(B_ptr_base); 
        // 注意：B 是 Row-Major，B_ptr_base 是起点。
        // 但这里 B 的 Tile 是随着 K 维度(用户变量 N) 移动的。
        // 实际上 B 的指针移动是：ptr + t * K_width。
        // 原代码逻辑：(t + load_b_row) * K

        // 修正指针计算：每次循环 B 向下移，A 向右移
        const float* B_ptr_curr = B + load_b_row * K + col_B;

        float4 tmp1 = (load_b_row < N && col_B<K) ? 
                      reinterpret_cast<const float4*>(B_ptr_curr)[0] : make_float4(0.0f,0.0f,0.0f,0.0f);
        float4 tmp2 = (load_b_row < N && (col_B + 4 < K)) ? 
                      reinterpret_cast<const float4*>(B_ptr_curr + 4)[0] : make_float4(0.0f,0.0f,0.0f,0.0f);

        reinterpret_cast<float4*>(&Bs[0][load_b_row][load_b_col])[0] = tmp1;
        reinterpret_cast<float4*>(&Bs[0][load_b_row][load_b_col + 4])[0] = tmp2;
    }

    __syncthreads();

    // --- 3. Main Loop ---
    int write_stage_idx = 1; // 下一次写入的 buffer
    int load_stage_idx = 0;  // 当前计算使用的 buffer

    // t 从 0 开始，每次步进 TS_K
    // 注意：Prologue 已经加载了 t=0 的数据。
    // 循环内主要做：计算(t)，预加载(t+TS_K)
    for (int t = 0; t < N; t += TS_K) {

        // === Step 3.1: 预加载 下一个 Tile (t + TS_K) 到 寄存器 ===
        // 这样不会覆盖当前正在被 Compute 读取的 SMEM
        int next_t = t + TS_K;

        if (next_t < N) {
            // Load A
            bool row_valid = (by * TS + load_a_row < M);
            // 指针偏移：A 向右移 TS_K
            int load_next_a_col = next_t + load_a_col;
            const float4* A_vec_ptr = reinterpret_cast<const float4*>(A_ptr_base + load_next_a_col);

            float4 t1 = (row_valid && (load_next_a_col < N)) ? A_vec_ptr[0] : make_float4(0.0f,0.0f,0.0f,0.0f);
            float4 t2 = (row_valid && (load_next_a_col + 4 < N)) ? A_vec_ptr[1] : make_float4(0.0f,0.0f,0.0f,0.0f);

            // 暂存到寄存器
            reinterpret_cast<float4*>(ldg_a_reg)[0] = t1;
            reinterpret_cast<float4*>(ldg_a_reg)[1] = t2;

            // Load B
            int load_next_b_col = bx * TS + load_b_col;
            int load_next_b_row = next_t + load_b_row;
            // 指针偏移：B 向下移 TS_K
            const float* B_ptr_next = B + load_next_b_row * K + load_next_b_col;

            float4 t3 = (load_next_b_row < N && load_next_b_col<K) ? 
                        reinterpret_cast<const float4*>(B_ptr_next)[0] : make_float4(0.0f,0.0f,0.0f,0.0f);
            float4 t4 = (load_next_b_row < N && (load_next_b_col + 4 < K)) ? 
                        reinterpret_cast<const float4*>(B_ptr_next + 4)[0] : make_float4(0.0f,0.0f,0.0f,0.0f);

            reinterpret_cast<float4*>(ldg_b_reg)[0] = t3;
            reinterpret_cast<float4*>(ldg_b_reg)[1] = t4;
        }

        // === Step 3.2: 计算 当前 Tile (SMEM[load_stage_idx]) ===
        #pragma unroll
        for (int k = 0; k < TS_K; ++k) {
            // 加载 A 的一列 (由当前线程负责的 WPT 行)
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                // As 有 Padding，访问安全
                reg_A[i] = As[load_stage_idx][ty * WPT + i][k];
            }
            // 加载 B 的一行 (由当前线程负责的 WPT 列)
            #pragma unroll
            for (int i = 0; i < WPT; ++i) {
                reg_B[i] = Bs[load_stage_idx][k][tx * WPT + i];
            }
            // 外积计算
            #pragma unroll
            for (int row = 0; row < WPT; ++row) {
                #pragma unroll
                for (int col = 0; col < WPT; ++col) {
                    accum[row][col] += reg_A[row] * reg_B[col];
                }
            }
        }

        // === Step 3.3: Store 寄存器 -> SMEM ===
        // 必须先 Sync，确保所有线程都完成了 Compute (读 SMEM)，才能写入下一轮的数据
        __syncthreads();

        if (next_t < N) {
            // 将预加载到寄存器的数据写入 Shared Memory 的另一半
            reinterpret_cast<float4*>(&As[write_stage_idx][load_a_row][load_a_col])[0] = reinterpret_cast<float4*>(ldg_a_reg)[0];
            reinterpret_cast<float4*>(&As[write_stage_idx][load_a_row][load_a_col + 4])[0] = reinterpret_cast<float4*>(ldg_a_reg)[1];

            reinterpret_cast<float4*>(&Bs[write_stage_idx][load_b_row][load_b_col])[0] = reinterpret_cast<float4*>(ldg_b_reg)[0];
            reinterpret_cast<float4*>(&Bs[write_stage_idx][load_b_row][load_b_col + 4])[0] = reinterpret_cast<float4*>(ldg_b_reg)[1];
        }

        // 交换 buffer索引
        load_stage_idx = 1 - load_stage_idx;
        write_stage_idx = 1 - write_stage_idx;

        // 必须再次 Sync，确保 SMEM 写入完成，下一轮 Compute 可以读取
        __syncthreads();
    }

    // --- 4. 写回结果到全局内存 (向量化) ---
    // 每个线程负责 8x8 的块
    // 我们可以每行写两个 float4
    int row_c = by * TS + ty * WPT; 
    int col_c = bx * TS + tx * WPT;

    #pragma unroll
    for (int row = 0; row < WPT; ++row) {
        int global_row = row_c + row;
        if (global_row < M) {
            #pragma unroll
            for (int col_offset = 0; col_offset < WPT; col_offset += 4) {
                int global_col = col_c + col_offset;
                if (global_col < K) { // 简单边界检查，假设 K 是 4 的倍数
                     // 构造 float4
                     float4 res;
                     res.x = accum[row][col_offset + 0];
                     res.y = accum[row][col_offset + 1];
                     res.z = accum[row][col_offset + 2];
                     res.w = accum[row][col_offset + 3];

                     // 写入
                     if (global_col + 4 <= K) {
                        reinterpret_cast<float4*>(&C[global_row * K + global_col])[0] = res;
                     } else {
                        // 边缘处理 (略，假设对其)
                        for(int i=0; i<4; i++) {
                            if(global_col + i < K) C[global_row * K + global_col+i] = ((float*)&res)[i];
                        }
                     }
                }
            }
        }
    }
}

extern "C" void solve(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 threadsPerBlock(TS / WPT, TS / WPT); 
    dim3 numBlocks((K + TS - 1) / TS, (M + TS - 1) / TS);
    matrix_multiplication_double_buffer<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);
}
