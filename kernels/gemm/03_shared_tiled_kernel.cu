// One 16 x 16 output tile per block, with one output per thread.

#include <cuda_runtime.h>


template <int tile_size>
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float A_tile[tile_size][tile_size];
    __shared__ float B_tile[tile_size][tile_size];

    int local_column = threadIdx.x;
    int local_row = threadIdx.y;
    int column = blockIdx.x * tile_size + local_column;
    int row = blockIdx.y * tile_size + local_row;

    float sum = 0.0f;

    for (int tile_start = 0;
         tile_start < K;
         tile_start += tile_size) {
        int A_column = tile_start + local_column;
        int B_row = tile_start + local_row;

        A_tile[local_row][local_column] =
            row < M && A_column < K
                ? A[row * K + A_column]
                : 0.0f;

        B_tile[local_row][local_column] =
            B_row < K && column < N
                ? B[B_row * N + column]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < tile_size; ++inner) {
            sum += A_tile[local_row][inner]
                 * B_tile[inner][local_column];
        }

        __syncthreads();
    }

    if (row < M && column < N) {
        C[row * N + column] = sum;
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
    constexpr int tile_size = 16;
    dim3 block_size(tile_size, tile_size);
    dim3 grid_size(
        (N + tile_size - 1) / tile_size,
        (M + tile_size - 1) / tile_size
    );

    tiled_gemm_kernel<tile_size><<<grid_size, block_size>>>(
        A, B, C, M, N, K
    );
}
