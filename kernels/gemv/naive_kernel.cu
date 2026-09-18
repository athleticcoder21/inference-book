// GEMV Naive Implementation

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>


int ceil_division(int M, int N){
    return (M + N - 1) / N;
}


__global__ void naive_gemv_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N
) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < M) {
        float sum = 0.f;

        for (int column = 0; column < N; ++column) {
            sum += A[row * N + column] * x[column];
        }

        y[row] = sum;
    }
}


void launch_kernel(
    const float* __restrict__ A,
    const float* __restrict__ x,
    float* __restrict__ y,
    int M,
    int N    
) {
    dim3 block_size(1024); 
    dim3 grid_size(ceil_division(M, block_size.x));

    naive_gemv_kernel<<<grid_size, block_size>>>(A, x, y, M, N);
}
