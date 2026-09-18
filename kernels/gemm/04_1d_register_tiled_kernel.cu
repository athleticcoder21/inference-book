// One block owns 64 x 32 outputs. Each thread owns 8 rows x 1 column.

#include <cuda_runtime.h>


template <int BM, int BN, int BK, int TM>
__global__ void register_tiled_1d_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float A_tile[BM * BK];
    __shared__ float B_tile[BK * BN];

    constexpr int threads = (BM / TM) * BN;
    int thread = threadIdx.x;
    int thread_column = thread % BN;
    int thread_row = thread / BN;
    int block_row = blockIdx.y * BM;
    int block_column = blockIdx.x * BN;

    float results[TM];
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        results[i] = 0.0f;
    }

    for (int tile_start = 0; tile_start < K; tile_start += BK) {
        for (int index = thread; index < BM * BK; index += threads) {
            int local_row = index / BK;
            int local_column = index % BK;
            int global_row = block_row + local_row;
            int global_column = tile_start + local_column;

            A_tile[index] =
                global_row < M && global_column < K
                    ? A[global_row * K + global_column]
                    : 0.0f;
        }

        for (int index = thread; index < BK * BN; index += threads) {
            int local_row = index / BN;
            int local_column = index % BN;
            int global_row = tile_start + local_row;
            int global_column = block_column + local_column;

            B_tile[index] =
                global_row < K && global_column < N
                    ? B[global_row * N + global_column]
                    : 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < BK; ++inner) {
            float B_value = B_tile[inner * BN + thread_column];

            #pragma unroll
            for (int result_row = 0; result_row < TM; ++result_row) {
                int A_row = thread_row * TM + result_row;
                results[result_row] +=
                    A_tile[A_row * BK + inner] * B_value;
            }
        }

        __syncthreads();
    }

    int column = block_column + thread_column;

    #pragma unroll
    for (int result_row = 0; result_row < TM; ++result_row) {
        int row = block_row + thread_row * TM + result_row;
        if (row < M && column < N) {
            C[row * N + column] = results[result_row];
        }
    }
}


void launch_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    constexpr int BM = 64;
    constexpr int BN = 32;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int threads = (BM / TM) * BN;

    dim3 block_size(threads);
    dim3 grid_size(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    register_tiled_1d_gemm_kernel<BM, BN, BK, TM>
        <<<grid_size, block_size>>>(A, B, C, M, N, K);
}
