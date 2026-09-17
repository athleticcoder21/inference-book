// Two-pass LayerNorm with one warp per row and Welford statistics.

#include <cuda_runtime.h>


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


__device__ __forceinline__ WelfordState welford_combine(
    WelfordState left,
    WelfordState right
) {
    if (right.count == 0) {
        return left;
    }

    if (left.count == 0) {
        return right;
    }

    int count = left.count + right.count;
    float difference = right.mean - left.mean;
    float right_fraction =
        static_cast<float>(right.count) / static_cast<float>(count);
    float cross_weight =
        static_cast<float>(left.count) * right_fraction;

    WelfordState combined;
    combined.mean = left.mean + difference * right_fraction;
    combined.m2 = left.m2 + right.m2
        + difference * difference * cross_weight;
    combined.count = count;

    return combined;
}


__device__ __forceinline__ WelfordState warp_reduce_welford(
    WelfordState state
) {
    unsigned int mask = 0xffffffffu;
    int lane = threadIdx.x & (warpSize - 1);

    for (int offset = 16; offset > 0; offset /= 2) {
        WelfordState other;
        other.mean = __shfl_down_sync(mask, state.mean, offset);
        other.m2 = __shfl_down_sync(mask, state.m2, offset);
        other.count = __shfl_down_sync(mask, state.count, offset);

        if (lane + offset < warpSize) {
            state = welford_combine(state, other);
        }
    }

    return state;
}


__global__ void warp_layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int M,
    int N,
    float epsilon
) {
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    WelfordState local = {0.0f, 0.0f, 0};

    // First coalesced pass: every lane builds statistics for its columns.
    for (int column = lane; column < N; column += warpSize) {
        local = welford_update(local, input_row[column]);
    }

    WelfordState row_state = warp_reduce_welford(local);

    float mean = __shfl_sync(0xffffffffu, row_state.mean, 0);
    float m2 = __shfl_sync(0xffffffffu, row_state.m2, 0);
    int count = __shfl_sync(0xffffffffu, row_state.count, 0);
    float variance = m2 / static_cast<float>(count);
    float reciprocal_std = rsqrtf(variance + epsilon);

    // Second coalesced pass: normalize, apply the affine transform, and write.
    for (int column = lane; column < N; column += warpSize) {
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
    constexpr int threads_per_block = 32;

    dim3 block_size(threads_per_block);
    dim3 grid_size(M);

    warp_layernorm_kernel<<<grid_size, block_size>>>(
        input,
        weight,
        bias,
        output,
        M,
        N,
        epsilon
    );
}
