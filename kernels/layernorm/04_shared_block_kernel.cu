// One-global-input-read LayerNorm with one block per row.

#include <cuda_runtime.h>


__device__ __forceinline__ float block_reduce_sum(
    float value,
    float* reduction
) {
    int thread = threadIdx.x;
    reduction[thread] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (thread < stride) {
            reduction[thread] += reduction[thread + stride];
        }
        __syncthreads();
    }

    return reduction[0];
}


__global__ void shared_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M,
    int N,
    float epsilon
) {
    extern __shared__ float shared[];

    float* row_cache = shared;
    float* reduction = shared + N;

    int row = blockIdx.x;
    int thread = threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    // The only global-memory read of the input row.
    for (int column = thread; column < N; column += blockDim.x) {
        row_cache[column] = input_row[column];
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int column = thread; column < N; column += blockDim.x) {
        local_sum += row_cache[column];
    }

    float sum = block_reduce_sum(local_sum, reduction);
    float mean = sum / static_cast<float>(N);

    float local_squared_difference = 0.0f;
    for (int column = thread; column < N; column += blockDim.x) {
        float difference = row_cache[column] - mean;
        local_squared_difference += difference * difference;
    }

    float squared_difference =
        block_reduce_sum(local_squared_difference, reduction);
    float variance = squared_difference / static_cast<float>(N);
    float reciprocal_std = rsqrtf(variance + epsilon);

    for (int column = thread; column < N; column += blockDim.x) {
        float normalized =
            (row_cache[column] - mean) * reciprocal_std;
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
    size_t shared_bytes =
        static_cast<size_t>(N + threads_per_block) * sizeof(float);

    dim3 block_size(threads_per_block);
    dim3 grid_size(M);

    shared_layernorm_kernel<<<grid_size, block_size, shared_bytes>>>(
        input,
        weight,
        bias,
        output,
        M,
        N,
        epsilon
    );
}
