// GEMV Coalesced Warp Implementation

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__global__ void performant_gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N    
) {
    // Ensure block size equals warp size for optimal performance
    assert(blockDim.x == warpSize);

    int block_id = blockIdx.x;
    
    if (block_id >= M)
        return;

    int thread_id = threadIdx.x;

    float partial_sum = 0.f;

    for (int column = thread_id; column < N; column += warpSize) {
        partial_sum += A[block_id * N + column] * x[column];
    }

    float sum = warpReduceSum(partial_sum);

    if (thread_id == 0){
        y[block_id] = sum;
    }
}


void launch_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N    
) {
    int num_threads = 32;

    dim3 block_size(num_threads); 
    dim3 grid_size(M);

    performant_gemv_kernel<<<grid_size, block_size>>>(A, x, y, M, N);
}
