#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

// --- 配置参数 ---
#define BM 128
#define BN 128
#define BK 32 

#define WM 64
#define WN 32
#define WARPS_M 2
#define WARPS_N 4

#define PAD 4

__global__ void matrix_multiplication_tensor_core(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K) 
{
    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag[WM/16][BK/8];
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag[BK/8][WN/16];
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> c_frag[WM/16][WN/16];
    for (int i = 0; i < WM/16; i++) {
        for (int j = 0; j < WN/16; j++) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    __shared__ float As[BM][BK + PAD];
    __shared__ float Bs[BK][BN + PAD];

    // 方便计算线程在global memory和shared momery对应的索引
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // 计算warp索引，方便计算warp在block中对应的索引
    int warpId = tx / 32;
    int warp_row = warpId / WARPS_N;
    int warp_col = warpId % WARPS_N;

    for (int k_step = 0; k_step < K; k_step += BK) {

        int tid = tx;
        int load_a_row = tid / (BK / 4);

        // =====搬数据： global memory -> shared memory====
        // 每一个block256个线程，需要搬运的数据为128*32，一个线程需要搬16个字节，一次搬4个，一共搬四次
        #pragma unroll
        for (int i = 0; i < 4; ++i) {

            // shared memory index
            int linear_idx = i * 256 + tid;
            int row = linear_idx / 8;
            int col = (linear_idx % 8) * 4;

            if (row < BM && col < BK) {
                int global_row = by * BM + row;
                int global_col = k_step + col;

                float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < M && global_col < K) {
                    tmp = *reinterpret_cast<const float4*>(&A[global_row * K + global_col]);
                }
                As[row][col + 0] = tmp.x;
                As[row][col + 1] = tmp.y;
                As[row][col + 2] = tmp.z;
                As[row][col + 3] = tmp.w;
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int linear_idx = i * 256 + tid;
            int row = linear_idx / 32;
            int col = (linear_idx % 32) * 4;

            if (row < BK && col < BN) {
                int global_row = k_step + row;
                int global_col = bx * BN + col;

                float4 tmp = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                if (global_row < K && global_col < N) {
                    tmp = *reinterpret_cast<const float4*>(&B[global_row * N + global_col]);
                }

                Bs[row][col + 0] = tmp.x;
                Bs[row][col + 1] = tmp.y;
                Bs[row][col + 2] = tmp.z;
                Bs[row][col + 3] = tmp.w;
            }
        }

        __syncthreads();

        // === Tensor Core 计算 ===
        // 每个 Warp 计算输出矩阵 C 的一块 (WM x WN) = (64 x 32)， 一个tensor core只能处理16*8@8*16的矩阵，需要切开每一个warp负责的区域
        for (int ki = 0; ki < BK / 8; ++ki) {

            // 1. 加载 A 的片段
            #pragma unroll
            for (int i = 0; i < WM / 16; ++i) {
                int row_idx = warp_row * WM + i * 16;
                int col_idx = ki * 8;
                wmma::load_matrix_sync(a_frag[i][ki], &As[row_idx][col_idx], BK + PAD);
            }

            // 2. 加载 B 的片段
            #pragma unroll
            for (int j = 0; j < WN / 16; ++j) {
                int row_idx = ki * 8;
                int col_idx = warp_col * WN + j * 16;
                wmma::load_matrix_sync(b_frag[ki][j], &Bs[row_idx][col_idx], BN + PAD);
            }
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
    int c_base_row = by * BM + warp_row * WM;
    int c_base_col = bx * BN + warp_col * WN;

    #pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < WN / 16; ++j) {
            int row = c_base_row + i * 16;
            int col = c_base_col + j * 16;
            if (row < M && col < N) {
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
