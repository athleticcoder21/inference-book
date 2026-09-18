// Naive three-pass row-wise LayerNorm.

#include <cuda_runtime.h>


int ceil_division(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}


__global__ void naive_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M,
    int N,
    float epsilon
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    float mean = 0.0f;
    for (int column = 0; column < N; ++column) {
        mean += input_row[column];
    }
    mean /= static_cast<float>(N);

    float variance = 0.0f;
    for (int column = 0; column < N; ++column) {
        float difference = input_row[column] - mean;
        variance += difference * difference;
    }
    variance /= static_cast<float>(N);

    float reciprocal_std = rsqrtf(variance + epsilon);

    for (int column = 0; column < N; ++column) {
        float normalized =
            (input_row[column] - mean) * reciprocal_std;
        output_row[column] =
            normalized * weight[column] + bias[column];
    }
}


void launch_layernorm(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M,
    int N,
    float epsilon
) {
    constexpr int threads_per_block = 256;

    dim3 block_size(threads_per_block);
    dim3 grid_size(ceil_division(M, threads_per_block));

    naive_layernorm_kernel<<<grid_size, block_size>>>(
        input,
        weight,
        bias,
        output,
        M,
        N,
        epsilon
    );
}
