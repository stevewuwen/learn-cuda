// 第7个版本
#include <cuda_runtime.h>
#include <stdio.h>

// ==========================================
// T4 优化参数 + 双重缓冲策略
// ==========================================
const int BM = 128;
const int BN = 128;
const int BK = 8;
const int TM = 8;
const int TN = 8;

__global__ __launch_bounds__(256)
void sgemm_double_buffer_t4(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    int tid = threadIdx.x;
    int ty = tid / 16;
    int tx = tid % 16;
    int by = blockIdx.y;
    int bx = blockIdx.x;

    // =========================================================
    // 关键改变 1: 双重缓冲 Shared Memory
    // 使用 [2] 个 buffer，write_stage 用于写，read_stage 用于算
    // =========================================================
    __shared__ float As[2][BK][BM]; 
    __shared__ float Bs[2][BK][BN];

    float accum[TM][TN] = {0.0f};

    // 寄存器缓存，用于计算
    float rag[TM];
    float rbg[TN];

    // 寄存器缓存，用于 Global Memory 预取 (Prefetch)
    // 每个线程搬运 1 个 float4 的 A 和 1 个 float4 的 B
    float4 load_a_reg; 
    float4 load_b_reg;

    const float* A_ptr = A + by * BM * K;
    const float* B_ptr = B + bx * BN;

    // 加载索引计算
    int load_a_row = tid / 2; 
    int load_a_col = (tid % 2) * 4;
    int load_b_row = tid / 32;
    int load_b_col = (tid % 32) * 4;

    // =========================================================
    // Prologue (序幕): 加载第一个 Tile 到 Buffer 0
    // =========================================================
    {
        int k_start = 0;
        // Load A
        if (by * BM + load_a_row < M && k_start + load_a_col < K) {
            load_a_reg = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + k_start + load_a_col])[0];
        } else {
            load_a_reg = {0.0f, 0.0f, 0.0f, 0.0f};
        }
        // Load B
        if (k_start + load_b_row < K && bx * BN + load_b_col < N) {
            load_b_reg = reinterpret_cast<const float4*>(&B_ptr[(k_start + load_b_row) * N + load_b_col])[0];
        } else {
            load_b_reg = {0.0f, 0.0f, 0.0f, 0.0f};
        }

        // 写入 SMEM Buffer 0
        // A 转置写入
        As[0][load_a_col + 0][load_a_row] = load_a_reg.x;
        As[0][load_a_col + 1][load_a_row] = load_a_reg.y;
        As[0][load_a_col + 2][load_a_row] = load_a_reg.z;
        As[0][load_a_col + 3][load_a_row] = load_a_reg.w;

        // B 直接写入
        reinterpret_cast<float4*>(&Bs[0][load_b_row][load_b_col])[0] = load_b_reg;
    }

    __syncthreads();

    // =========================================================
    // Main Loop
    // =========================================================
    int write_stage_idx = 1; // 下一轮写入的位置
    int read_stage_idx = 0;  // 当前计算读取的位置

    // 注意：循环从 k=0 开始算，但在 k 时我们要预加载 k+BK 的数据
    for (int k = 0; k < K; k += BK) {

        // -----------------------------------------------------
        // 1. Prefetch Next Tile to Registers (Global -> Register)
        // 这里的关键是：当我们发起 Global Load 指令后，GPU 不会阻塞，
        // 而是会继续向下执行计算指令 (Math)，从而隐藏内存延迟。
        // -----------------------------------------------------
        int next_k = k + BK;
        if (next_k < K) {
            // Load A to Reg
            if (by * BM + load_a_row < M && next_k + load_a_col < K) {
                load_a_reg = reinterpret_cast<const float4*>(&A_ptr[load_a_row * K + next_k + load_a_col])[0];
            } else {
                load_a_reg = {0.0f, 0.0f, 0.0f, 0.0f};
            }
            // Load B to Reg
            if (next_k + load_b_row < K && bx * BN + load_b_col < N) {
                load_b_reg = reinterpret_cast<const float4*>(&B_ptr[(next_k + load_b_row) * N + load_b_col])[0];
            } else {
                load_b_reg = {0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        // -----------------------------------------------------
        // 2. Compute Current Tile (Register <-> SMEM)
        // 使用 read_stage_idx
        // -----------------------------------------------------
        #pragma unroll
        for (int i = 0; i < BK; ++i) {
            // Load A from SMEM to Reg
            float4 tmpA0 = reinterpret_cast<const float4*>(&As[read_stage_idx][i][ty * TM])[0];
            float4 tmpA1 = reinterpret_cast<const float4*>(&As[read_stage_idx][i][ty * TM + 4])[0];
            rag[0] = tmpA0.x; rag[1] = tmpA0.y; rag[2] = tmpA0.z; rag[3] = tmpA0.w;
            rag[4] = tmpA1.x; rag[5] = tmpA1.y; rag[6] = tmpA1.z; rag[7] = tmpA1.w;

            // Load B from SMEM to Reg
            float4 tmpB0 = reinterpret_cast<const float4*>(&Bs[read_stage_idx][i][tx * TN])[0];
            float4 tmpB1 = reinterpret_cast<const float4*>(&Bs[read_stage_idx][i][tx * TN + 4])[0];
            rbg[0] = tmpB0.x; rbg[1] = tmpB0.y; rbg[2] = tmpB0.z; rbg[3] = tmpB0.w;
            rbg[4] = tmpB1.x; rbg[5] = tmpB1.y; rbg[6] = tmpB1.z; rbg[7] = tmpB1.w;

            // Compute
            #pragma unroll
            for (int r = 0; r < TM; ++r) {
                #pragma unroll
                for (int c = 0; c < TN; ++c) {
                    accum[r][c] += rag[r] * rbg[c];
                }
            }
        }

        // -----------------------------------------------------
        // 3. Store Prefetched Data to SMEM (Register -> SMEM)
        // 此时计算已经完成，我们等待所有线程都算完了当前块
        // -----------------------------------------------------
        __syncthreads(); // 确保 read_stage 的数据大家都不用了

        if (next_k < K) {
            // 将寄存器里的下一块数据写入 write_stage 的 SMEM
            // A Transposed
            As[write_stage_idx][load_a_col + 0][load_a_row] = load_a_reg.x;
            As[write_stage_idx][load_a_col + 1][load_a_row] = load_a_reg.y;
            As[write_stage_idx][load_a_col + 2][load_a_row] = load_a_reg.z;
            As[write_stage_idx][load_a_col + 3][load_a_row] = load_a_reg.w;

            // B Direct
            reinterpret_cast<float4*>(&Bs[write_stage_idx][load_b_row][load_b_col])[0] = load_b_reg;
        }

        // 翻转 buffer 索引
        // write_stage: 1 -> 0 -> 1
        // read_stage:  0 -> 1 -> 0
        write_stage_idx ^= 1;
        read_stage_idx ^= 1;

        __syncthreads(); // 确保 write_stage 的数据已经写好，可以作为下一轮的 read_stage
    }

    // =========================================================
    // Write Back
    // =========================================================
    int global_row_start = by * BM + ty * TM;
    int global_col_start = bx * BN + tx * TN;

    #pragma unroll
    for (int r = 0; r < TM; ++r) {
        int global_r = global_row_start + r;
        if (global_r < M) {
            int global_c = global_col_start;
            if (global_c + 7 < N) {
                float4 tmp0, tmp1;
                tmp0.x = accum[r][0]; tmp0.y = accum[r][1]; tmp0.z = accum[r][2]; tmp0.w = accum[r][3];
                tmp1.x = accum[r][4]; tmp1.y = accum[r][5]; tmp1.z = accum[r][6]; tmp1.w = accum[r][7];
                reinterpret_cast<float4*>(&C[global_r * N + global_c])[0] = tmp0;
                reinterpret_cast<float4*>(&C[global_r * N + global_c + 4])[0] = tmp1;
            } else {
                for (int c = 0; c < TN; ++c) {
                    if (global_c + c < N) {
                        C[global_r * N + global_c + c] = accum[r][c];
                    }
                }
            }
        }
    }
}

// 宿主调用
extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 block(256);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_double_buffer_t4<<<grid, block>>>(A, B, C, M, N, K);
}
