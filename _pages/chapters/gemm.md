---
layout: distill
permalink: /gemm/
title: "GEMM"
description: "A worklog on optimizing matrix multiplication in CUDA by fixing thread ownership, reusing global-memory tiles in shared memory, and accumulating register micro-tiles."
date: 2026-09-18
section_number: 4
previous_section_url: ../layernorm
previous_section_name: "LayerNorm"
next_section_url: ../index
next_section_name: "Introduction"
authors:
  - name: Anshuman Mishra
    url: https://heyyanshuman.com
    affiliations:
      name: Independent researcher
toc:
  - name: How much work does GEMM perform?
    subsections:
      - name: The best-case traffic
  - name: One thread per output
    subsections:
      - name: A bad ownership choice
      - name: Let x mean column
      - name: Cost of the direct kernel
  - name: One block per output tile
    subsections:
      - name: A four-by-four example
      - name: Where the reuse comes from
      - name: Turning tiling into code
  - name: More than one output per thread
    subsections:
      - name: One-dimensional register tiling
      - name: Two-dimensional register tiling
      - name: The complete 2D register-tiled schedule
  - name: Vectorizing the tile loads
  - name: What each optimization changed
  - name: What we should benchmark
  - name: Main takeaways
---

The code for this chapter lives in [`kernels/gemm/`](https://github.com/athleticcoder21/inference-book/tree/main/kernels/gemm).

GEMM means **GEneral Matrix-Matrix multiplication**. Given

$$
\mathbf{A}\in\mathbb{R}^{M\times K},
\qquad
\mathbf{B}\in\mathbb{R}^{K\times N},
$$

we want to calculate

$$
\mathbf{C}_{M\times N}
  =\mathbf{A}_{M\times K}\mathbf{B}_{K\times N}.
$$

Every output element is the dot product of one row of $\mathbf{A}$ and one
column of $\mathbf{B}$:

$$
C_{i,j}=\sum_{k=0}^{K-1}A_{i,k}B_{k,j}.
$$

{% include figure.liquid path="assets/gemm/gemm-overview.svg" class="img-fluid" alt="An M by K matrix A multiplied by a K by N matrix B to produce an M by N matrix C. One row of A and one column of B are highlighted because their dot product produces one highlighted output element." %}

This equation looks like GEMV repeated for every column of $\mathbf{B}$, but
that extra matrix changes the optimization problem. In GEMV, an element of the
matrix is useful for one output dot product. In GEMM, one value from
$\mathbf{A}$ can contribute to many columns of $\mathbf{C}$, while one value
from $\mathbf{B}$ can contribute to many rows. If we keep those values close to
the processors, the same load can feed many multiply-adds.

That reuse is why GEMM powers the large linear layers used during prompt
processing and training. It is also why this chapter will feel different from
GEMV, Softmax, and LayerNorm. Those kernels made progress mainly by moving
fewer bytes. GEMM must also arrange enough independent arithmetic to keep the
floating-point pipelines busy.

<aside class="callout"><strong>Why this chapter uses FP32</strong>

The kernels below use FP32 so that the change from one version to the next is
about work ownership and data movement, not a simultaneous change of datatype
or execution unit. FP16 and BF16 reduce operand traffic, and tensor cores
change how the multiply-accumulate itself is issued. They build on the same
tiling hierarchy, but deserve a separate treatment.

</aside>

Before touching CUDA, let us write the complete calculation:

```text
for row = 0 to M - 1:
    for column = 0 to N - 1:
        sum = 0

        for inner = 0 to K - 1:
            sum += A[row, inner] * B[inner, column]

        C[row, column] = sum
```

There are $MN$ independent output elements. The first kernel will expose all
of that parallelism by assigning one output element to one thread.

## How much work does GEMM perform?

Each output dot product contains $K$ multiplications and $K-1$ additions.
Across all $MN$ outputs, the exact ordinary floating-point work is

$$
MN\left(K+(K-1)\right)
  =2MNK-MN \quad \text{FLOPs}.
$$

On a GPU, the multiply and addition are normally emitted as a fused
multiply-add. One FMA is one instruction, but it still represents two
floating-point operations. For large $K$, we therefore use the familiar
approximation

$$
\text{work}\approx2MNK \quad \text{FLOPs}.
$$

### The best-case traffic

Assume FP32, so every element occupies 4 bytes. If each input matrix could be
read exactly once and the result written exactly once, the traffic would be

| Operation | Elements transferred | Bytes transferred |
|---|---:|---:|
| Read $\mathbf{A}$ | $MK$ | $4MK$ |
| Read $\mathbf{B}$ | $KN$ | $4KN$ |
| Write $\mathbf{C}$ | $MN$ | $4MN$ |
| **Total** | **$MK+KN+MN$** | **$4(MK+KN+MN)$** |

The corresponding arithmetic intensity is

$$
AI_{\text{ideal}}(M,N,K)
  =\frac{2MNK-MN}{4(MK+KN+MN)}
  \quad \text{FLOPs/byte}.
$$

For a square multiplication with $M=N=K=L$,

$$
AI_{\text{ideal}}(L)
  =\frac{2L^3-L^2}{12L^2}
  =\frac{2L-1}{12}
  \approx\frac{L}{6}
  \quad \text{FLOPs/byte}.
$$

Unlike GEMV's intensity of roughly $0.5$ FLOPs/byte, this value grows with the
matrix size. A $4096\times4096$ square GEMM has an ideal FP32 intensity of
roughly $683$ FLOPs/byte.

There is one important word in that calculation: **ideal**. The equation only
counts each matrix element once. A direct kernel asks for the same values many
times. Caches may satisfy some of those requests, but the kernel itself has not
created a place where a block of threads can deliberately reuse the data.

We will keep two traffic models separate throughout this chapter:

- **Algorithmic traffic** counts the minimum useful data: one read of each
  input and one write of the output.

- **Kernel-issued traffic** counts the loads and stores requested by a
  particular implementation before making assumptions about cache hits.

The first number explains GEMM's potential. The second explains why a naive
kernel can be slow despite that potential.

## One thread per output

The simplest parallel mapping gives one element of $\mathbf{C}$ to one CUDA
thread. That thread walks across one row of $\mathbf{A}$ and down one column of
$\mathbf{B}$.

The two output coordinates are independent, so a 2D thread block feels
natural. But there is a trap: CUDA forms warps by linearizing the block with
`threadIdx.x` changing fastest. The meaning we attach to the x-coordinate
therefore changes which output elements neighboring lanes own.

### A bad ownership choice

Suppose `threadIdx.x` selects the output row and `threadIdx.y` selects the
output column:

```cpp
int row = blockIdx.y * blockDim.x + threadIdx.x;
int column = blockIdx.x * blockDim.y + threadIdx.y;
```

For a block whose x-dimension is 32, the first warp has a fixed
`threadIdx.y = 0` and `threadIdx.x = 0,1,...,31`. At one inner-loop iteration,
its addresses are

```text
lane 0  -> A[(row + 0)  * K + inner]   B[inner * N + column]
lane 1  -> A[(row + 1)  * K + inner]   B[inner * N + column]
lane 2  -> A[(row + 2)  * K + inner]   B[inner * N + column]
...
lane 31 -> A[(row + 31) * K + inner]   B[inner * N + column]
```

All lanes request the same value from $\mathbf{B}$, which can be broadcast.
But their $\mathbf{A}$ values are $K$ floats apart, and their final
$\mathbf{C}$ stores are $N$ floats apart. For large matrices, one warp scatters
both its matrix loads and its output stores across memory.

```cpp
__global__ void row_first_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    int row = blockIdx.y * blockDim.x + threadIdx.x;
    int column = blockIdx.x * blockDim.y + threadIdx.y;

    if (row >= M || column >= N) {
        return;
    }

    float sum = 0.0f;
    for (int inner = 0; inner < K; ++inner) {
        sum += A[row * K + inner] * B[inner * N + column];
    }

    C[row * N + column] = sum;
}
```

The arithmetic is correct. The ownership is not friendly to the way a warp
moves data.

### Let x mean column

Now swap the meaning of the thread coordinates:

```cpp
int column = blockIdx.x * blockDim.x + threadIdx.x;
int row = blockIdx.y * blockDim.y + threadIdx.y;
```

With `blockDim.x = 32`, one warp owns 32 consecutive columns from the same
output row. At inner-loop position $k$, lane $l$ reads

$$
A_{i,k}
\qquad\text{and}\qquad
B_{k,j+l}.
$$

The $\mathbf{A}$ address is identical across the warp, while the
$\mathbf{B}$ addresses are consecutive. The output stores are consecutive as
well:

```text
lane 0  -> A[row, inner]  B[inner, column + 0]  C[row, column + 0]
lane 1  -> A[row, inner]  B[inner, column + 1]  C[row, column + 1]
lane 2  -> A[row, inner]  B[inner, column + 2]  C[row, column + 2]
...
lane 31 -> A[row, inner]  B[inner, column + 31] C[row, column + 31]
```

One change in ownership turns the $\mathbf{B}$ loads and $\mathbf{C}$ stores
into coalesced accesses.

```cpp
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
```

<aside class="callout"><strong>Coalesced does not mean reused</strong>

The remapped kernel makes each warp's requests efficient, but every output
thread still issues $K$ loads from $\mathbf{A}$ and $K$ loads from
$\mathbf{B}$. Coalescing changes how those requests are served. It does not
remove the repeated requests.

</aside>

### Cost of the direct kernel

There are $MN$ output threads. Each one requests $K$ values from
$\mathbf{A}$, $K$ values from $\mathbf{B}$, and writes one result:

| Operation | Elements requested | Bytes requested |
|---|---:|---:|
| Read $\mathbf{A}$ | $MNK$ | $4MNK$ |
| Read $\mathbf{B}$ | $MNK$ | $4MNK$ |
| Write $\mathbf{C}$ | $MN$ | $4MN$ |
| **Total** | **$2MNK+MN$** | **$4(2MNK+MN)$** |

Ignoring cache reuse, its arithmetic intensity is

$$
\begin{aligned}
AI_{\text{direct}}
  &=\frac{2MNK-MN}{4(2MNK+MN)}\\
  &=\frac{2K-1}{4(2K+1)}
  \approx\frac{1}{4}
  \quad \text{FLOPs/byte}.
\end{aligned}
$$

So the same mathematical GEMM that had an ideal intensity growing with matrix
size now looks like a low-intensity kernel. The difference is reuse. The
algorithm says that $A_{i,k}$ is useful for every output column, but the kernel
asks for it again from every output thread.

The cache can rescue nearby threads that request the same line. We should be
glad when it does. We should not make the whole kernel depend on that rescue.

## One block per output tile

The direct kernel finishes one output at a time. A tiled kernel changes the
unit of ownership: one thread block owns a rectangular tile of
$\mathbf{C}$.

Assume for now that the output tile is $T\times T$. The block cannot calculate
that tile in one step because each dot product has length $K$. It therefore
walks across the inner dimension in chunks of width $T$:

```text
for each T-wide slice along K:
    load one T x T tile from A into shared memory
    load one T x T tile from B into shared memory
    synchronize

    multiply the two shared-memory tiles
    add the partial result to the accumulator
    synchronize

write the finished C tile
```

{% include figure.liquid path="assets/gemm/gemm-tiling.svg" class="img-fluid" alt="A block computes one tile of C by repeatedly loading a horizontal tile from A and a vertical tile from B along the K dimension. Each pair of input tiles contributes a partial product to the same output tile." %}

The first barrier ensures that the complete input tiles are available before
any thread reads them. The second ensures that every thread has finished using
the current tiles before the block overwrites shared memory with the next
pair.

### A four-by-four example

Use $T=2$ for a small matrix:

$$
\mathbf{A}=
\begin{bmatrix}
a_{00}&a_{01}&a_{02}&a_{03}\\
a_{10}&a_{11}&a_{12}&a_{13}\\
a_{20}&a_{21}&a_{22}&a_{23}\\
a_{30}&a_{31}&a_{32}&a_{33}
\end{bmatrix},
\qquad
\mathbf{B}=
\begin{bmatrix}
b_{00}&b_{01}&b_{02}&b_{03}\\
b_{10}&b_{11}&b_{12}&b_{13}\\
b_{20}&b_{21}&b_{22}&b_{23}\\
b_{30}&b_{31}&b_{32}&b_{33}
\end{bmatrix}.
$$

To produce the top-left $2\times2$ tile of $\mathbf{C}$, the block first loads

$$
\mathbf{A}^{(0)}=
\begin{bmatrix}
a_{00}&a_{01}\\
a_{10}&a_{11}
\end{bmatrix},
\qquad
\mathbf{B}^{(0)}=
\begin{bmatrix}
b_{00}&b_{01}\\
b_{10}&b_{11}
\end{bmatrix}.
$$

Their product is only the first contribution:

$$
\mathbf{C}_{0:2,0:2}
  \mathrel{+}=\mathbf{A}^{(0)}\mathbf{B}^{(0)}.
$$

The block then advances two positions along $K$ and loads

$$
\mathbf{A}^{(1)}=
\begin{bmatrix}
a_{02}&a_{03}\\
a_{12}&a_{13}
\end{bmatrix},
\qquad
\mathbf{B}^{(1)}=
\begin{bmatrix}
b_{20}&b_{21}\\
b_{30}&b_{31}
\end{bmatrix}.
$$

After

$$
\mathbf{C}_{0:2,0:2}
  \mathrel{+}=\mathbf{A}^{(1)}\mathbf{B}^{(1)},
$$

the four accumulators contain the finished output tile. Nothing is written to
global memory until all $K/T$ partial products have been accumulated.

### Where the reuse comes from

During one tile iteration, the block loads

$$
T^2+T^2=2T^2
$$

values from global memory. It then performs $T^3$ FMAs, because there are
$T^2$ output elements and each receives $T$ multiply-adds.

Each $\mathbf{A}$ value is used by $T$ output columns. Each $\mathbf{B}$ value
is used by $T$ output rows. The input tiles cross global memory once, then
shared memory distributes them to all the threads that need them.

For FP32, the arithmetic intensity of one tile multiplication, ignoring the
eventual output store, is

$$
\begin{aligned}
AI_{\text{tile}}
  &=\frac{2T^3}{4(2T^2)}\\
  &=\frac{T}{4}
  \quad \text{FLOPs/byte}.
\end{aligned}
$$

The direct kernel approached $0.25$ FLOPs/byte before cache effects. A tile of
width 16 gives $4$ FLOPs/byte at the global-memory boundary, while a tile of
width 32 gives $8$ FLOPs/byte.

For rectangular block tiles of shape $B_M\times B_N$ and an inner tile width
$B_K$, the same calculation becomes

$$
AI_{\text{tile}}
  =\frac{2B_MB_NB_K}
         {4(B_MB_K+B_KB_N)}
  =\frac{B_MB_N}{2(B_M+B_N)}.
$$

Notice that $B_K$ cancels. Increasing $B_K$ changes loop overhead, shared
memory use, and synchronization frequency, but it does not change the reuse
factor by itself.

### Turning tiling into code

For the first tiled kernel, use a $16\times16$ block. Every thread loads one
value into each shared-memory tile and computes one output element:

```cpp
template <int tile_size>
__global__ void tiled_gemm_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K
) {
    __shared__ float A_tile[tile_size][tile_size];
    __shared__ float B_tile[tile_size][tile_size];

    int local_column = threadIdx.x;
    int local_row = threadIdx.y;
    int column = blockIdx.x * tile_size + local_column;
    int row = blockIdx.y * tile_size + local_row;

    float sum = 0.0f;

    for (int tile_start = 0;
         tile_start < K;
         tile_start += tile_size) {
        int A_column = tile_start + local_column;
        int B_row = tile_start + local_row;

        A_tile[local_row][local_column] =
            row < M && A_column < K
                ? A[row * K + A_column]
                : 0.0f;

        B_tile[local_row][local_column] =
            B_row < K && column < N
                ? B[B_row * N + column]
                : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < tile_size; ++inner) {
            sum += A_tile[local_row][inner]
                 * B_tile[inner][local_column];
        }

        __syncthreads();
    }

    if (row < M && column < N) {
        C[row * N + column] = sum;
    }
}
```

The zero fill handles an incomplete tile at the right or bottom edge. It also
handles the final inner tile when $K$ is not a multiple of 16. Every thread
must still reach both barriers, so the bounds check surrounds the memory
operation instead of returning early from the kernel.

The global loads are coalesced:

- neighboring x-lanes load neighboring columns from a row of $\mathbf{A}$;
- those same lanes load neighboring columns from a row of $\mathbf{B}$; and
- neighboring x-lanes eventually write neighboring elements of $\mathbf{C}$.

We have now created reuse at the global-memory boundary. But inside the tile,
each thread still computes only one output.

## More than one output per thread

Look at the inner loop of the basic tiled kernel:

```text
load one A value from shared memory
load one B value from shared memory
perform one FMA
```

Shared memory is much faster than global memory, but two shared-memory loads
for every FMA can still leave the arithmetic pipelines waiting. We can reuse
the values one more time by giving each thread several accumulators.

This creates a hierarchy of ownership:

```text
grid
  -> one block owns a BM x BN output tile
       -> one thread owns a TM x TN output micro-tile
            -> one register owns one output accumulator
```

Shared memory reuses data across threads in the block. Registers reuse data
across the outputs owned by one thread.

### One-dimensional register tiling

First let one thread compute $T_M$ output rows and one output column. At a
fixed inner position, the thread needs

- $T_M$ values from the shared $\mathbf{A}$ tile;
- one value from the shared $\mathbf{B}$ tile; and
- $T_M$ accumulators in registers.

The single $\mathbf{B}$ value is reused across all $T_M$ FMAs:

```cpp
float B_value = B_tile[inner][thread_column];

#pragma unroll
for (int result_row = 0; result_row < TM; ++result_row) {
    results[result_row] +=
        A_tile[thread_row * TM + result_row][inner]
        * B_value;
}
```

For $T_M=8$, nine shared-memory loads enable eight FMAs. The basic tiled
kernel needed sixteen loads for those same eight FMAs. This is useful reuse,
but it is one-sided: values from $\mathbf{B}$ are reused while every
$\mathbf{A}$ value still feeds one accumulator in that thread.

### Two-dimensional register tiling

Now give one thread a $T_M\times T_N$ rectangle of outputs. For one inner
position, the thread loads

$$
T_M \quad \text{values from }\mathbf{A}
$$

and

$$
T_N \quad \text{values from }\mathbf{B}.
$$

It then forms their outer product:

$$
\text{results}_{i,j}
  \mathrel{+}=A_iB_j,
\qquad
0\le i<T_M,
\quad
0\le j<T_N.
$$

```cpp
float A_values[TM];
float B_values[TN];

for (int inner = 0; inner < BK; ++inner) {
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        A_values[i] = A_tile[thread_row * TM + i][inner];
    }

    #pragma unroll
    for (int j = 0; j < TN; ++j) {
        B_values[j] = B_tile[inner][thread_column * TN + j];
    }

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            results[i][j] += A_values[i] * B_values[j];
        }
    }
}
```

For $T_M=T_N=8$, sixteen shared-memory loads enable 64 FMAs. Each loaded
$\mathbf{A}$ value is used across eight output columns, and each loaded
$\mathbf{B}$ value is used across eight output rows.

The shared-memory arithmetic intensity of this inner step is

$$
AI_{\text{register tile}}
  =\frac{2T_MT_N}{4(T_M+T_N)}
  =\frac{T_MT_N}{2(T_M+T_N)}
  \quad \text{FLOPs/byte}.
$$

For $T_M=T_N=8$, that is

$$
\frac{8\cdot8}{2(8+8)}=2
\quad \text{FLOPs per shared-memory byte}.
$$

The basic one-output thread achieved

$$
\frac{2}{4(1+1)}=0.25
\quad \text{FLOPs per shared-memory byte}.
$$

So the register micro-tile increases reuse at the shared-memory boundary by
8x.

### The complete 2D register-tiled schedule

Use

$$
B_M=B_N=128,
\qquad B_K=8,
\qquad T_M=T_N=8.
$$

The block owns $128\times128$ outputs. Since every thread owns $8\times8$ of
them, the block needs

$$
\frac{128}{8}\frac{128}{8}
  =16\cdot16
  =256 \quad \text{threads}.
$$

For every 8-wide step along $K$, those 256 threads cooperatively load

$$
128\cdot8+8\cdot128=2048
$$

FP32 values, or 8192 bytes, into shared memory. They then perform

$$
128\cdot128\cdot8=131{,}072 \quad \text{FMAs}
$$

before the next pair of global-memory tiles is needed.

The complete implementation in `05_2d_register_tiled_kernel.cu` uses a
strided loading loop because 2048 tile elements must be loaded by only 256
threads. Each thread loads several values, the block synchronizes, and then
the register outer products begin.

<aside class="callout"><strong>Why not make the micro-tile enormous?</strong>

A larger micro-tile creates more reuse, but every output needs an accumulator.
An $8\times8$ tile already asks for 64 accumulator registers per thread before
counting addresses and temporary values. More registers can reduce the number
of resident warps, and enough pressure can make the compiler spill values into
local memory. Register tiling trades occupancy for reuse; it does not remove a
cost for free.

</aside>

## Vectorizing the tile loads

The register-tiled kernel has reduced how often input tiles cross global
memory. We can now reduce the number of instructions used to move each tile.

A scalar copy moves one FP32 value per load instruction:

```cpp
shared[destination + 0] = global[source + 0];
shared[destination + 1] = global[source + 1];
shared[destination + 2] = global[source + 2];
shared[destination + 3] = global[source + 3];
```

When the address is 16-byte aligned, a `float4` copy expresses the same 16
bytes as one vector load:

```cpp
float4 values =
    *reinterpret_cast<const float4*>(&global[source]);
*reinterpret_cast<float4*>(&shared[destination]) = values;
```

Vectorization does **not** make those 16 bytes disappear. Coalescing already
combines neighboring lanes' requests into memory transactions. The benefit is
that each thread issues fewer load instructions and gives the compiler an
explicit alignment guarantee.

That guarantee is also the danger. A `float4` access requires suitable
alignment and four valid consecutive elements. The vectorized kernel checks
the leading dimensions and tile edges, then falls back to scalar copies when
those conditions are not satisfied. Casting an arbitrary pointer and hoping
it is aligned is undefined behaviour, not an optimization.

The same idea applies to the output. Every row of an $8\times8$ thread
micro-tile contains eight consecutive floats, so an aligned interior tile can
be written with two `float4` stores per row.

## What each optimization changed

We can now separate the optimization ladder by memory boundary:

| Kernel | Outputs per thread | Global-memory change | Shared-memory change | Main cost introduced |
|---|---:|---|---|---|
| Row-first direct | 1 | Strided $\mathbf{A}$ and $\mathbf{C}$ access | None | Repeated global loads |
| Coalesced direct | 1 | Coalesced $\mathbf{B}$ and $\mathbf{C}$ access | None | Repeated global loads remain |
| Shared-memory tiled | 1 | Each input tile is loaded once per block | 2 loads per FMA per thread | Tile storage and barriers |
| 1D register tiled | $T_M$ | Same block-level reuse | Reuses one $\mathbf{B}$ value | More accumulators |
| 2D register tiled | $T_MT_N$ | Same block-level reuse | Reuses both $\mathbf{A}$ and $\mathbf{B}$ values | Register pressure |
| Vectorized 2D tile | $T_MT_N$ | Fewer wide load/store instructions | Same register reuse | Alignment and edge paths |

The changes are cumulative. Register tiling does not replace shared-memory
tiling; it adds another level of reuse below it. Vectorization does not replace
coalescing; it reduces instruction count after the access pattern is already
contiguous.

## What we should benchmark

A cost model tells us what changed. It does not tell us which tile sizes win on
every GPU or matrix shape. A useful benchmark should report at least

- GPU model and CUDA version;
- $M$, $N$, and $K$ separately, not only square matrices;
- warm-up count and timing method;
- achieved TFLOPs;
- global-load efficiency and DRAM throughput;
- shared-memory throughput and bank conflicts;
- registers per thread, occupancy, and any local-memory spills; and
- numerical error against a trusted reference such as cuBLAS.

The most revealing experiments vary one dimension at a time. A square
$4096\times4096$ multiplication rewards large tiles and reuse. A tall, narrow
projection can expose wasted work at tile edges. Small matrices may be limited
by launch overhead and may fit in cache well enough to hide differences that
are obvious at realistic model sizes.

We should also compare against cuBLAS without pretending that a CUDA-core FP32
teaching kernel has the same goal as a production library. cuBLAS selects
different algorithms by shape, datatype, layout, hardware, and workspace. Its
value as a baseline is precisely that it shows how much room remains.

## Main takeaways

GEMM optimization is a story about preserving the same value across several
levels of the memory hierarchy long enough to use it more than once.

- **GEMM has high potential arithmetic intensity.** Its ideal intensity grows
  with matrix size because every input value can contribute to many outputs.

- **Potential reuse is not automatic reuse.** A one-thread-per-output kernel
  requests $2MNK$ input elements even though the two input matrices contain
  only $MK+KN$ elements.

- **Thread ownership determines coalescing.** Since `threadIdx.x` changes
  fastest inside a warp, mapping x to output columns gives consecutive lanes
  consecutive $\mathbf{B}$ loads and $\mathbf{C}$ stores.

- **A block tile creates deliberate global-memory reuse.** The block loads one
  tile from each input into shared memory and uses them to update many output
  elements before advancing along $K$.

- **The $K$ dimension is accumulated, not tiled independently.** Every pair of
  input tiles produces only a partial output tile. The registers hold those
  partial sums until the complete inner dimension has been visited.

- **Register tiling creates a second level of reuse.** A thread loads short
  vectors from the shared tiles and forms an outer product, updating a grid of
  register accumulators.

- **More reuse consumes more resources.** Larger block tiles need more shared
  memory. Larger thread tiles need more registers. Both can reduce occupancy
  or make edge handling more wasteful.

- **Vectorization reduces instructions, not bytes.** It is useful only when
  alignment and bounds make the wider access valid.

- **The optimization target moves inward.** First global memory is the
  bottleneck. Shared-memory tiling moves the pressure to shared memory.
  Register tiling moves more of the work into FMAs. A good GEMM kernel keeps
  repeating this process until the arithmetic pipelines, rather than data
  delivery, set the pace.

The equation $\mathbf{C}=\mathbf{A}\mathbf{B}$ never changed. What changed was
the unit of ownership: first one thread owned one output, then one block owned
an output tile, and finally each thread owned a register micro-tile inside that
block. Every step made reuse explicit at one more level of the machine.
