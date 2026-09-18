// Naive three-pass row-wise Softmax.

#include <cuda_runtime.h>
#include <math_constants.h>


int ceil_division(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}


__global__ void naive_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    float row_max = -CUDART_INF_F;

    // Pass 1: find the maximum value in the row.
    for (int column = 0; column < N; ++column) {
        row_max = fmaxf(row_max, input_row[column]);
    }

    float row_denominator = 0.0f;

    // Pass 2: calculate the denominator relative to the row maximum.
    for (int column = 0; column < N; ++column) {
        row_denominator += expf(input_row[column] - row_max);
    }

    // Pass 3: normalize and write the row.
    for (int column = 0; column < N; ++column) {
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
    constexpr int threads_per_block = 256;

    dim3 block_size(threads_per_block);
    dim3 grid_size(ceil_division(M, threads_per_block));

    naive_softmax_kernel<<<grid_size, block_size>>>(
        input,
        output,
        M,
        N
    );
}
