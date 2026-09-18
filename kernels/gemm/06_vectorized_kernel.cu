// 2D register tiling with aligned float4 tile loads and output stores.

#include <cuda_runtime.h>
#include <stdint.h>


__device__ __forceinline__ bool is_aligned_16(const float* pointer) {
    return (reinterpret_cast<uintptr_t>(pointer) & 0xfu) == 0;
}


template <int BM, int BN, int BK, int TM, int TN>
__global__ void vectorized_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    static_assert(BK % 4 == 0, "BK must contain complete float4 vectors");
    static_assert(BN % 4 == 0, "BN must contain complete float4 vectors");
    static_assert(TN == 8, "The vectorized output path expects TN=8");

    __shared__ float A_tile[BM * BK];
    __shared__ float B_tile[BK * BN];

    constexpr int thread_columns = BN / TN;
    constexpr int threads = (BM / TM) * thread_columns;
    static_assert(BM * BK / 4 == threads,
                  "Each thread loads one float4 from A");
    static_assert(BK * BN / 4 == threads,
                  "Each thread loads one float4 from B");

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
        int A_vector_start = thread * 4;
        int A_local_row = A_vector_start / BK;
        int A_local_column = A_vector_start % BK;
        int A_global_row = block_row + A_local_row;
        int A_global_column = tile_start + A_local_column;
        bool A_in_bounds =
            A_global_row < M && A_global_column + 3 < K;
        const float* A_source = A_in_bounds
            ? A + A_global_row * K + A_global_column
            : A;

        if (A_in_bounds && is_aligned_16(A_source)) {
            float4 values =
                *reinterpret_cast<const float4*>(A_source);
            A_tile[A_vector_start + 0] = values.x;
            A_tile[A_vector_start + 1] = values.y;
            A_tile[A_vector_start + 2] = values.z;
            A_tile[A_vector_start + 3] = values.w;
        } else {
            #pragma unroll
            for (int value = 0; value < 4; ++value) {
                int column = A_global_column + value;
                A_tile[A_vector_start + value] =
                    A_global_row < M && column < K
                        ? A[A_global_row * K + column]
                        : 0.0f;
            }
        }

        int B_vector_start = thread * 4;
        int B_local_row = B_vector_start / BN;
        int B_local_column = B_vector_start % BN;
        int B_global_row = tile_start + B_local_row;
        int B_global_column = block_column + B_local_column;
        bool B_in_bounds =
            B_global_row < K && B_global_column + 3 < N;
        const float* B_source = B_in_bounds
            ? B + B_global_row * N + B_global_column
            : B;

        if (B_in_bounds && is_aligned_16(B_source)) {
            float4 values =
                *reinterpret_cast<const float4*>(B_source);
            B_tile[B_vector_start + 0] = values.x;
            B_tile[B_vector_start + 1] = values.y;
            B_tile[B_vector_start + 2] = values.z;
            B_tile[B_vector_start + 3] = values.w;
        } else {
            #pragma unroll
            for (int value = 0; value < 4; ++value) {
                int column = B_global_column + value;
                B_tile[B_vector_start + value] =
                    B_global_row < K && column < N
                        ? B[B_global_row * N + column]
                        : 0.0f;
            }
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
        int column = block_column + thread_column * TN;

        if (row >= M || column >= N) {
            continue;
        }

        float* destination = C + row * N + column;

        if (column + TN - 1 < N && is_aligned_16(destination)) {
            float4 first = make_float4(
                results[i * TN + 0],
                results[i * TN + 1],
                results[i * TN + 2],
                results[i * TN + 3]
            );
            float4 second = make_float4(
                results[i * TN + 4],
                results[i * TN + 5],
                results[i * TN + 6],
                results[i * TN + 7]
            );
            *reinterpret_cast<float4*>(destination) = first;
            *reinterpret_cast<float4*>(destination + 4) = second;
        } else {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                if (column + j < N) {
                    destination[j] = results[i * TN + j];
                }
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

    vectorized_gemm_kernel<BM, BN, BK, TM, TN>
        <<<grid_size, block_size>>>(A, B, C, M, N, K);
}
