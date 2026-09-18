// Two-pass online Softmax with one warp per row.

#include <cuda_runtime.h>
#include <math_constants.h>


__device__ __forceinline__ float warpReduceMax(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, offset)
        );
    }
    return value;
}


__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}


__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // One block contains one warp, and one warp owns one row.
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    float local_max = -CUDART_INF_F;
    float local_denominator = 0.0f;

    // First coalesced pass: compute online statistics for this lane.
    for (int column = lane; column < N; column += warpSize) {
        float current = input_row[column];

        if (current > local_max) {
            local_denominator *= expf(local_max - current);
            local_max = current;
        }

        local_denominator += expf(current - local_max);
    }

    // Reduce the local maxima. The complete result lands in lane 0.
    float row_max = warpReduceMax(local_max);

    // Broadcast the row maximum from lane 0 to every lane.
    row_max = __shfl_sync(0xffffffffu, row_max, 0);

    // Put every local denominator on the same scale.
    float corrected_denominator =
        local_denominator * expf(local_max - row_max);

    // Reduce the corrected denominators and broadcast their sum.
    float row_denominator = warpReduceSum(corrected_denominator);
    row_denominator =
        __shfl_sync(0xffffffffu, row_denominator, 0);

    // Second coalesced pass: normalize and write the row.
    for (int column = lane; column < N; column += warpSize) {
        output_row[column] =
            expf(input_row[column] - row_max) / row_denominator;
    }
}


void launch_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    constexpr int threads_per_block = 32;

    dim3 block_size(threads_per_block);
    dim3 grid_size(M);

    softmax_warp_kernel<<<grid_size, block_size>>>(
        input,
        output,
        M,
        N
    );
}
