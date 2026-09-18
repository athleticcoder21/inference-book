# LayerNorm kernels

These files follow the four implementations developed in the LayerNorm
worklog:

- `01_naive_kernel.cu`: three input passes, with one thread processing one row.
- `02_welford_kernel.cu`: two input passes using Welford statistics, still with
  one thread processing one row.
- `03_warp_welford_kernel.cu`: two coalesced input passes, with one warp
  processing one row and Welford states merged through warp shuffles.
- `04_shared_block_kernel.cu`: one global input read, with one block processing
  one row cached in shared memory.

Each file defines the same `launch_layernorm` entry point and is intended to be
compiled separately. All kernels operate on an FP32 matrix of shape `(M, N)`
and normalize across the last dimension. They assume `M > 0` and `N > 0`.

The shared-memory implementation also assumes a power-of-two block size and
requires `(N + block_size) * sizeof(float)` bytes of dynamic shared memory per
block. The launcher uses 256 threads, so very wide rows may exceed the GPU's
per-block shared-memory limit. In that case, use the warp implementation or a
tiled block implementation.
