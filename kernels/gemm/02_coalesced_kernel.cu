// Direct GEMM with threadIdx.x mapped to consecutive output columns.

#include <cuda_runtime.h>


__global__ void coalesced_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= M || column >= N) {
        return;
    }

    float sum = 0.0f;
    for (int inner = 0; inner < K; ++inner) {
        sum += A[row * K + inner] * B[inner * N + column];
    }

    C[row * N + column] = sum;
}


void launch_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    dim3 block_size(32, 8);
    dim3 grid_size(
        (N + block_size.x - 1) / block_size.x,
        (M + block_size.y - 1) / block_size.y
    );

    coalesced_gemm_kernel<<<grid_size, block_size>>>(
        A, B, C, M, N, K
    );
}
