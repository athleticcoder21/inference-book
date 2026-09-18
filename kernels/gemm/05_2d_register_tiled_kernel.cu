// One block owns 128 x 128 outputs. Each thread owns an 8 x 8 micro-tile.

#include <cuda_runtime.h>


template <int BM, int BN, int BK, int TM, int TN>
__global__ void register_tiled_2d_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float A_tile[BM * BK];
    __shared__ float B_tile[BK * BN];

    constexpr int thread_columns = BN / TN;
    constexpr int threads = (BM / TM) * thread_columns;

    int thread = threadIdx.x;
    int thread_column = thread % thread_columns;
    int thread_row = thread / thread_columns;
    int block_row = blockIdx.y * BM;
    int block_column = blockIdx.x * BN;

    float results[TM * TN];
    #pragma unroll
    for (int i = 0; i < TM * TN; ++i) {
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
            float A_values[TM];
            float B_values[TN];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                int local_row = thread_row * TM + i;
                A_values[i] = A_tile[local_row * BK + inner];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                int local_column = thread_column * TN + j;
                B_values[j] = B_tile[inner * BN + local_column];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    results[i * TN + j] += A_values[i] * B_values[j];
                }
            }
        }

        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        int row = block_row + thread_row * TM + i;

        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int column = block_column + thread_column * TN + j;
            if (row < M && column < N) {
                C[row * N + column] = results[i * TN + j];
            }
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
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int threads = (BM / TM) * (BN / TN);

    dim3 block_size(threads);
    dim3 grid_size(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    register_tiled_2d_gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid_size, block_size>>>(A, B, C, M, N, K);
}
