# Softmax kernels

These files follow the three implementations developed in the Softmax
worklog:

- `01_naive_kernel.cu`: three passes, with one thread processing one row.
- `02_online_kernel.cu`: two passes using online Softmax, still with one thread
  processing one row.
- `03_warp_kernel.cu`: two coalesced passes, with one warp processing one row
  and warp shuffles combining the partial statistics.

Each file defines the same `launch_softmax` entry point and is intended to be
compiled separately. All kernels operate on an FP32 matrix of shape `(M, N)`
and apply Softmax across its last dimension. They assume `M > 0` and `N > 0`.
