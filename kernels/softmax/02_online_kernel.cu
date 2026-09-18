// Two-pass online Softmax with one thread per row.

#include <cuda_runtime.h>
#include <math_constants.h>


int ceil_division(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}


__global__ void online_softmax_kernel(
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
    float row_denominator = 0.0f;

    // Pass 1: update the maximum and denominator together.
    for (int column = 0; column < N; ++column) {
        float current = input_row[column];

        if (current > row_max) {
            row_denominator *= expf(row_max - current);
            row_max = current;
        }

        row_denominator += expf(current - row_max);
    }

    // Pass 2: normalize and write the row.
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

    online_softmax_kernel<<<grid_size, block_size>>>(
        input,
        output,
        M,
        N
    );
}
