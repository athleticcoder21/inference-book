// Two-pass row-wise LayerNorm using Welford statistics.

#include <cuda_runtime.h>


int ceil_division(int numerator, int denominator) {
    return (numerator + denominator - 1) / denominator;
}


struct WelfordState {
    float mean;
    float m2;
    int count;
};


__device__ __forceinline__ WelfordState welford_update(
    WelfordState state,
    float value
) {
    state.count += 1;

    float difference = value - state.mean;
    state.mean += difference / static_cast<float>(state.count);
    float difference_from_new_mean = value - state.mean;
    state.m2 += difference * difference_from_new_mean;

    return state;
}


__global__ void welford_layernorm_kernel(
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

    WelfordState state = {0.0f, 0.0f, 0};

    // Pass 1: update the mean and M2 together.
    for (int column = 0; column < N; ++column) {
        state = welford_update(state, input_row[column]);
    }

    float variance = state.m2 / static_cast<float>(state.count);
    float reciprocal_std = rsqrtf(variance + epsilon);

    // Pass 2: normalize, apply the affine transform, and write.
    for (int column = 0; column < N; ++column) {
        float normalized =
            (input_row[column] - state.mean) * reciprocal_std;
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

    welford_layernorm_kernel<<<grid_size, block_size>>>(
        input,
        weight,
        bias,
        output,
        M,
        N,
        epsilon
    );
}
