# GEMM kernels

These files follow the six implementations developed in the GEMM worklog:

- `01_row_first_kernel.cu`: one output per thread, with `threadIdx.x` mapped
  to output rows to make the initial strided access pattern explicit.
- `02_coalesced_kernel.cu`: one output per thread, with `threadIdx.x` remapped
  to consecutive output columns.
- `03_shared_tiled_kernel.cu`: one 16 x 16 output tile per block, with input
  tiles reused from shared memory.
- `04_1d_register_tiled_kernel.cu`: one 64 x 32 output tile per block, with
  every thread accumulating eight output rows for one column.
- `05_2d_register_tiled_kernel.cu`: one 128 x 128 output tile per block, with
  every thread accumulating an 8 x 8 register micro-tile.
- `06_vectorized_kernel.cu`: the 2D register-tiled kernel with guarded
  `float4` input loads and output stores.

Every file defines the same `launch_gemm` entry point and is intended to be
compiled separately. The matrices are row-major FP32 arrays with shapes
`A=(M,K)`, `B=(K,N)`, and `C=(M,N)`. Edge tiles are zero-filled, so the
dimensions do not need to be multiples of the tile sizes.

The final two kernels intentionally use 64 FP32 accumulators per thread. They
demonstrate register reuse clearly, but they are teaching kernels rather than
universal launch configurations. Register pressure, occupancy, shared-memory
bank conflicts, and the best tile dimensions vary by GPU and matrix shape.
