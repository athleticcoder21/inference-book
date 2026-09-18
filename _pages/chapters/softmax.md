---
layout: distill
permalink: /softmax/
title: "Softmax"
description: "A worklog on optimizing Softmax kernel in CUDA"
date: 2026-09-11
section_number: 2
previous_section_url: ../gemv
previous_section_name: "GEMV"
next_section_url: ../layernorm
next_section_name: "LayerNorm"
authors:
  - name: Anshuman Mishra
    url: https://heyyanshuman.com
    affiliations:
      name: Independent researcher
toc:
  - name: One thread per row
  - subsections:
    - name: Walking over one row
    - name: Cost of three passes
  - name: Removing one pass over the row
  - subsections:
    - name: Updating statistics online
    - name: Cost of two passes
  - name: One warp per row
  - subsections:
    - name: Splitting a row across the warp
    - name: Cost after coalescing
    - name: Turning the complete algorithm into code
  - name: Main takeaways
---

The code for this chapter lives in [`kernels/softmax/`](https://github.com/athleticcoder21/inference-book/tree/main/kernels/softmax).

Softmax is a very important part of attention mechanism in transformers. It converts raw attention scores to a probability distribution, where each element in the output is between 0 and 1, and sum of all elements is 1.

Formally, for an input vector $\mathbf{x}$ containing $N$ elements, Softmax is defined as:

$$
\operatorname{softmax}(\mathbf{x})_i
  = \frac{e^{x_i}}
         {\displaystyle\sum_{j=0}^{N-1} e^{x_j}},
  \qquad i \in \{0, \ldots, N-1\}.
$$

{% include figure.liquid path="assets/softmax/softmax-header.png" class="img-fluid" alt="A hand-drawn row of four input scores passing through Softmax to become four nonnegative probabilities whose sum is one." %}

Here, $x_i$ is the $i$-th element of the input and the denominator is the sum of the exponentials of all elements in the vector.

There is one problem with implementing this equation directly. The exponential function grows very quickly, so a large value of $x_i$ can make $e^{x_i}$ overflow. To make the computation numerically stable, we first find the maximum value in the input vector:

$$
m = \max_{0 \leq j < N} x_j.
$$

We then subtract $m$ from every element before taking its exponential:

$$
\operatorname{softmax}(\mathbf{x})_i
  = \frac{e^{x_i-m}}
         {\displaystyle\sum_{j=0}^{N-1} e^{x_j-m}},
  \qquad i \in \{0, \ldots, N-1\}.
$$

If you notice, subtracting the same value from every element does not change the final probability distribution. But now the largest exponent is $e^0=1$, which prevents the exponential from overflowing and gives us a numerically stable implementation. 

## One thread per row

### Walking over one row

Assume our input is a matrix $\mathbf{X[M,N]}$, let's talk about how we write this kernel. Let's start with the simplest approach.  The first thing that might come to mind is to assign one thread to each row.
and have it apply Softmax independently
to every row i.e. we are really calculating Softmax for $M$ different vectors,
each containing $N$ elements.

That thread first walks over its row to find the maximum value, walks over
it again to calculate the denominator, and then makes one final pass to write
the output.

```text
parallel for row = 0 to M - 1:
    maximum = -infinity

    for column = 0 to N - 1:
        maximum = max(maximum, X[row, column])

    denominator = 0

    for column = 0 to N - 1:
        denominator += exp(X[row, column] - maximum)

    for column = 0 to N - 1:
        Y[row, column] = exp(X[row, column] - maximum) / denominator
```

Pretty straightforward. Different rows are processed in parallel, but inside
each row a single thread still does all the work sequentially. Before we try to
improve that, let us understand how much work this algorithm actually performs.

### Cost of three passes

Again, assume that the input and output use FP32, so every element occupies 4
bytes. We will also assume that values are read from global memory on every
pass.

#### Computation

For one row, finding the maximum requires $N-1$ comparisons. Comparisons are
not floating-point operations, so we keep them separate from the FLOP count.

The second pass, which calculates the denominator, performs $N$ subtractions
and $N-1$ additions. 

The third pass performs another $N$ subtractions and $N$
divisions. Therefore, excluding the exponential function for a moment, the
ordinary floating-point work for one row is

$$
N+(N-1)+N+N = 4N-1 \quad \text{FLOPs}.
$$

Across all $M$ rows, this becomes

$$
M(4N-1) = 4MN-M \quad \text{FLOPs}.
$$

We also evaluate the
exponential twice for every element, giving us $2MN$ exponential evaluations.
An exponential is not a single FLOP. It is a special function whose cost and
throughput depend on the hardware, so folding it into the FLOP count would make
the number look more precise than it really is.

#### Memory access

The naive algorithm makes three complete passes over $\mathbf{X}$ and one pass
to write $\mathbf{Y}$:

| Operation | Elements transferred | Bytes transferred |
|---|---:|---:|
| Read $\mathbf{X}$ to find the maximum | $MN$ | $4MN$ |
| Read $\mathbf{X}$ to calculate the denominator | $MN$ | $4MN$ |
| Read $\mathbf{X}$ to calculate the output | $MN$ | $4MN$ |
| Write $\mathbf{Y}$ | $MN$ | $4MN$ |
| **Total** | **$4MN$** | **$16MN$** |

Using only the ordinary floating-point operations, the arithmetic intensity is

$$
\begin{aligned}
AI(M,N)
  &= \frac{M(4N-1)}{16MN} \\
  &= \frac{4N-1}{16N}
     \quad \text{FLOPs/byte}.
\end{aligned}
$$

For a large $N$,

$$
AI(M,N) \approx \frac{4N}{16N}
        = \frac{1}{4}
        = 0.25 \quad \text{FLOPs/byte}.
$$

So even before looking at how much parallelism we use, this is already a
low-arithmetic-intensity algorithm. It performs roughly one ordinary
floating-point operation for every 4 bytes transferred. The exponential
evaluations certainly add computation, but they do not remove the three full
reads of the input.

<aside class="callout"><strong>Why are exponentials not included in the FLOP count?</strong>
FLOPs traditionally count basic floating-point operations such as addition,
subtraction, multiplication, and division. Functions such as `exp` are
implemented using multiple instructions or specialized hardware and can have
very different throughput from those basic operations.

If we use a simplified operation count and pretend that each exponential is one
operation, the algorithm performs $6MN-M$ operations and has an operation
intensity approaching $0.375$ operations/byte. That number can be useful for a
rough comparison between our own kernels, but it should not be confused with a
hardware-independent FLOP count.

</aside>
There are two problems hiding inside this simple implementation:

- **Unnecessary repeated iterations.** We iterate over the same row three
  times: once to find the maximum, once to calculate the denominator, and once
  to write the output. Every input element is fetched from global memory three
  times. We also calculate its exponential in the second pass and then
  calculate the same exponential again in the third pass.

- **Uncoalesced memory access.** One thread owns one row, so neighboring
  threads in a warp read from different rows. We already studied this access
  pattern in the GEMV chapter, so we will not go into it again here. We will
  come back and fix it later.

## Removing one pass over the row

### Updating statistics online

Let's double-click on the first problem and see if it is possible to find the
maximum and calculate the denominator in the same pass.

Our denominator is defined relative to
the maximum value of the complete row, but while reading that row from left to
right, we do not know the final maximum yet.

Turns out, we do not need to know it in advance. Bear with me for a moment.

Assume we have processed the row up to index $i$. The largest value we have
seen so far is $m_1$. Since every exponential must be calculated relative to
the maximum, our running denominator is

$$
L_i
  = e^{x_0-m_1}
  + e^{x_1-m_1}
  + \cdots
  + e^{x_i-m_1}.
$$

Now we move to index $i+1$ and discover that $x_{i+1}$ is larger than $m_1$.
Let us call this new maximum $m_2$:

$$
m_2 = x_{i+1}, \qquad m_2 > m_1.
$$

We have a problem. All the terms already stored in $L_i$ were calculated
relative to $m_1$, but they now need to be relative to $m_2$. The corrected
sum of the old elements should be

$$
e^{x_0-m_2}
+ e^{x_1-m_2}
+ \cdots
+ e^{x_i-m_2}.
$$

However it is solvable. We can use the multiplication property of exponentials:

$$
e^a e^b = e^{a+b}.
$$

Multiply the old running sum by $e^{m_1-m_2}$:

$$
\begin{aligned}
L_i e^{m_1-m_2}
  &= \left(
       e^{x_0-m_1}
       + e^{x_1-m_1}
       + \cdots
       + e^{x_i-m_1}
     \right)e^{m_1-m_2} \\
  &= e^{x_0-m_1}e^{m_1-m_2}
     + e^{x_1-m_1}e^{m_1-m_2}
     + \cdots
     + e^{x_i-m_1}e^{m_1-m_2} \\
  &= e^{x_0-m_2}
     + e^{x_1-m_2}
     + \cdots
     + e^{x_i-m_2}.
\end{aligned}
$$

**Boom!**

With one multiplication, we corrected the contribution of every element we
had already processed. All that remains is to add the contribution of the new
maximum itself:

$$
e^{m_2-m_2}=e^0=1.
$$

Therefore, when we encounter a new maximum, the update is simply

$$
L_{i+1} = L_i e^{m_1-m_2}+1.
$$

And if $x_{i+1}$ is not a new maximum, there is nothing to correct. We keep
$m_1$ and update the sum normally:

$$
L_{i+1}=L_i+e^{x_{i+1}-m_1}.
$$

That is the entire trick behind **online Softmax**. Instead of waiting until the
end to learn the maximum, we update the maximum as we go and rescale the work
we have already done whenever that maximum changes.

That gives us the online algorithm:

```text
parallel for row = 0 to M - 1:
    maximum = -infinity
    denominator = 0

    for column = 0 to N - 1:
        current = X[row, column]

        if current > maximum:
            denominator = denominator * exp(maximum - current) + 1
            maximum = current
        else:
            denominator += exp(current - maximum)

    for column = 0 to N - 1:
        Y[row, column] = exp(X[row, column] - maximum) / denominator
```

We still need the final normalization pass, but the separate maximum and
denominator passes have now become one. Our three-pass algorithm is down to two
passes.

### Cost of two passes

Let's now analyze what this trick actually does to our algorithm.

#### Computation

Let $R$ be the total number of times an element becomes a new running maximum
across all $M$ rows. We process $MN$ elements in total. Now let us account for
the work one operation at a time.

- We compare every element with the running maximum. That gives us $MN$
  comparisons. We keep these separate because comparisons are not FLOPs.

- Whether the element becomes a new maximum or not, we subtract one value from
  another to form the exponent. That gives us $MN$ subtractions, or $MN$
  FLOPs.

- We evaluate one exponential for every element during the first pass. That
  gives us $MN$ exponential evaluations. As before, we track these separately
  from ordinary FLOPs.

- We add one contribution to the running denominator for every element. That
  gives us $MN$ additions, or another $MN$ FLOPs.

- Whenever an element becomes the new maximum, we multiply the old denominator
  by a correction factor. This happens $R$ times, giving us $R$
  multiplications, or $R$ FLOPs.

Now we can add the ordinary floating-point work from the first pass:

$$
\underbrace{MN}_{\text{subtractions}}
+ \underbrace{MN}_{\text{additions}}
+ \underbrace{R}_{\text{rescaling multiplications}}
= 2MN+R \quad \text{FLOPs}.
$$

We then make the normalization pass over all $MN$ elements:

- Subtracting the final maximum contributes $MN$ FLOPs.
- Evaluating the exponential contributes another $MN$ exponential
  evaluations.
- Dividing by the denominator contributes $MN$ FLOPs.

So the normalization pass contributes

$$
\underbrace{MN}_{\text{subtractions}}
+ \underbrace{MN}_{\text{divisions}}
= 2MN \quad \text{FLOPs}.
$$

Finally, adding both passes together gives us

$$
\begin{aligned}
\text{FLOPs}_{\text{online}}
  &= (2MN+R)+2MN \\
  &= 4MN+R.
\end{aligned}
$$

We also perform $MN+MN=2MN$ exponential evaluations in total. So this
optimization does not reduce the computational work. It adds $R$ rescaling
multiplications in exchange for reading the input fewer times.

#### Memory access

| Operation | Elements transferred | Bytes transferred |
|---|---:|---:|
| Read $\mathbf{X}$ to update the maximum and denominator | $MN$ | $4MN$ |
| Read $\mathbf{X}$ to calculate the output | $MN$ | $4MN$ |
| Write $\mathbf{Y}$ | $MN$ | $4MN$ |
| **Total** | **$3MN$** | **$12MN$** |

We reduced the input reads from $3MN$ elements to $2MN$ elements. Including the
output write, total memory traffic falls from $16MN$ bytes to $12MN$ bytes

The arithmetic intensity is now approximately

$$
\begin{aligned}
AI_{\text{online}}(M,N)
  &\approx \frac{4MN+R}{12MN} \\
  &= \frac{1}{3}+\frac{R}{12MN}
     \quad \text{FLOPs/byte}.
\end{aligned}
$$

Usually, only a small fraction of the elements become new running maxima, so
$R \ll MN$. In that case,

$$
AI_{\text{online}}(M,N) \approx \frac{1}{3}
  \approx 0.333 \quad \text{FLOPs/byte}.
$$

We moved from $0.25$ to approximately $0.333$ FLOPs/byte. **Note that, this bump did not come
from doing fewer FLOPs, we actually do a little more. It came from eliminating
one complete read of the input matrix.**


<aside class="callout"><strong>Why this helps</strong>

Note that this bump did not come
from doing fewer FLOPs, we actually do a little more. It came from eliminating
one complete read of the input matrix.

</aside>
## One warp per row

### Splitting a row across the warp

Alright, that's one problem down. Let's see if we can further optimize this kernel by leveraging **coalesced memory access pattern.**

As of now our kernel still assigns one entire row to one thread which is bad  because neighboring
threads are reading the same column from different rows, so their addresses are
$N$ elements apart. ( I recommend referring to GEMV kernel optimization chapter to further understand why)

We have to somehow reimplement this algorithm so that instead of giving
one row to one thread, let us give one row to an entire thread block:

```text
block 0 -> row 0
block 1 -> row 1
block 2 -> row 2
...
```

Let's assume that one block owns one row. But we still need to decide how the columns of
that row are divided among the threads.

Assume that our block has $T$ threads. The most natural mapping is:

```text
thread 0 -> columns 0, T, 2T, ...
thread 1 -> columns 1, T + 1, 2T + 1, ...
thread 2 -> columns 2, T + 2, 2T + 2, ...
...
```

Or, written as one indexing rule,

$$
\text{column} = \text{threadIdx.x} + kT,
\qquad k=0,1,2,\ldots
$$

Let us use a tiny example before jumping to a real warp. Say our row has *eight*
elements and the block has *four* threads.

During the first loop iteration, the four threads read

```text
thread 0 -> x[0]
thread 1 -> x[1]
thread 2 -> x[2]
thread 3 -> x[3]
```

During the next iteration, every thread moves forward by four:

```text
thread 0 -> x[4]
thread 1 -> x[5]
thread 2 -> x[6]
thread 3 -> x[7]
```

{% include figure.liquid path="assets/softmax/coalesced_row_access.svg" class="img-fluid" alt="A four-thread teaching block reading an eight-element Softmax row. Threads zero through three first read adjacent columns zero through three, then advance by four to read adjacent columns four through seven." %}

If you notice, one thread jumps by $T$ elements over time, but neighboring
threads read neighboring elements at the same instant.

For a real warp, $T=32$. The first loop iteration reads columns 0 through 31,
the second reads columns 32 through 63, and so on. Every iteration therefore
produces one coalesced chunk of 32 FP32 values.

Great, this is the access pattern we were looking for. Except we also broke something.  Now every thread sees only a slice of
the row.

Let us use this row:

$$
\mathbf{x}=[3,1,2,0,6,4,5,1].
$$

With four threads, the work is divided like this:

```text
thread 0 -> [x0, x4] -> [3, 6]
thread 1 -> [x1, x5] -> [1, 4]
thread 2 -> [x2, x6] -> [2, 5]
thread 3 -> [x3, x7] -> [0, 1]
```

Each thread can run online Softmax on its two values. But it only produces a
*local* maximum and a *local* denominator:

{% include figure.liquid path="assets/softmax/local_online_statistics.svg" class="img-fluid" alt="Four threads process different slices of one row and produce four local maxima and four local denominators." %}

So now we have

```text
local maxima       = [6, 4, 5, 1]
local denominators = [1.0498, 1.0498, 1.0498, 1.3679]
```

#### Reducing the maximum

What we actually need is one maximum and one denominator for the complete row.
For that we need to turn

$$
[6,4,5,1]
$$

into a single value. We could ask thread 0 to read the other three values and
find the maximum sequentially. But then we are back to making one thread do all
the work.

Instead, let the threads compare pairs of values.

In the first round, we compare values that are two positions apart:

```text
thread 0 -> max(6, 5) = 6
thread 1 -> max(4, 1) = 4
```

Four candidates have now become two. In the second round, we compare the two
remaining values:

```text
thread 0 -> max(6, 4) = 6
```

And we are done. Thread 0 now holds the maximum of the complete row.

This pattern is called a **reduction**. At every round, half of the candidates
disappear. Our four-value example takes two rounds. A warp has 32 values, so it
takes five:

```text
offset 16 -> 32 values become 16
offset  8 -> 16 values become 8
offset  4 ->  8 values become 4
offset  2 ->  4 values become 2
offset  1 ->  2 values become 1
```

Notice that we still perform 31 comparisons to reduce 32 values. The number of
comparisons did not magically shrink. The difference is that those comparisons
happen in five parallel rounds instead of one chain of 31 comparisons.


#### Correcting the local denominators

With the row maximum in hand, the next step is to add the local denominators.
But before we do that, notice that thread 0 calculated its denominator relative
to 6, thread 1 used 4, thread 2 used 5, and thread 3 used 1:

```text
thread 0 -> 1.0498 is relative to 6
thread 1 -> 1.0498 is relative to 4
thread 2 -> 1.0498 is relative to 5
thread 3 -> 1.3679 is relative to 1
```

Adding these numbers directly would be wrong because their exponentials use
different maxima. Before adding them, we need to express all four denominators
relative to the row maximum, which is 6.

We already learned how to do that in online Softmax. If a thread has local
maximum $m_t$ and local denominator $L_t$, we correct it using

$$
L_t^{\text{corrected}}=L_t e^{m_t-m},
$$

where $m$ is the maximum of the complete row.

Let us apply this to every thread:

$$
\begin{aligned}
\text{thread 0}:&\quad 1.0498e^{6-6} \approx 1.0498,\\
\text{thread 1}:&\quad 1.0498e^{4-6} \approx 0.1421,\\
\text{thread 2}:&\quad 1.0498e^{5-6} \approx 0.3862,\\
\text{thread 3}:&\quad 1.3679e^{1-6} \approx 0.0092.
\end{aligned}
$$

{% include figure.liquid path="assets/softmax/max_and_sum_reductions.svg" class="img-fluid" alt="Four local maxima are reduced to row maximum six. Each local denominator is then corrected so that all four use the same row maximum." %}

After this correction, all four denominators are relative to the same maximum,
so the next step is to add them.

#### Reducing the sum

The sum reduction follows the exact same pattern. We just replace `max` with
addition. Our corrected values are

$$
[1.0498,0.1421,0.3862,0.0092].
$$

First round:

```text
thread 0 -> 1.0498 + 0.3862 = 1.4360
thread 1 -> 0.1421 + 0.0092 = 0.1513
```

Second round:

```text
thread 0 -> 1.4360 + 0.1513 = 1.5873
```

{% include figure.liquid path="assets/softmax/sum_reduction_steps.svg" class="img-fluid" alt="A four-lane sum reduction. Two pairs are added in the first round, and the remaining pair is added in the second round to produce 1.5873." %}

So the denominator for the complete row is

$$
L=1.5873.
$$

We finally have everything required to calculate Softmax: the row maximum is 6
and the row denominator is 1.5873. Every thread can now revisit its columns and
write the corresponding outputs.

#### Exchanging values between threads

We kept saying that thread 0 reads a value from thread 2, then thread 1 reads a
value from thread 3. But all these partial values live in registers owned by
different threads. How does one thread read another thread's register?

Remember `__shfl_down_sync`? which we used for exact such usage in GEMV kernel worklog. It lets a thread read a
register value held by another thread in the same warp.

```cpp
__device__ __forceinline__ float warpReduceMax(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = max(
            value,
            __shfl_down_sync(0xffffffff, value, offset)
        );
    }
    return value;
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}
```

<aside class="callout" markdown="1"><strong>What does __forceinline__ mean?</strong>

Normally, calling a function means jumping to that function and returning when
it is done. When a function is **inlined**, the compiler instead places the
function's instructions directly where it is called.

`__forceinline__` tells the CUDA compiler to inline the function instead of
deciding for itself. That makes sense for helpers such as `warpReduceMax` and
`warpReduceSum`: they are small, called frequently, and we do not want a
function call around a five-step reduction.

Inlining does not change what the function computes or how threads cooperate.
It only changes how the compiler places its instructions in the generated GPU
code. The trade-off is that forcing a large function to be inlined can make the
compiled code much bigger, so we mainly use it for small helpers like these.

</aside>

<aside class="callout" markdown="1"><strong>What does __shfl_down_sync actually do?</strong>
Let us unpack this line:

```cpp
__shfl_down_sync(0xffffffff, value, offset)
```

- `0xffffffff` is a 32-bit mask with every bit set to 1. Each bit represents
  one lane, so this says that all 32 lanes in the warp are participating in the
  shuffle. This is valid here because our complete warp executes the reduction.
  If only some lanes participated, the mask would need to describe that subset.

- `value` is the number held by the current lane. Since every lane calls this
  function, every lane passes its own number. The function then gives each lane
  the `value` from the lane `offset` positions ahead of it. For example, when
  `offset` is 2, lane 0 gets lane 2's `value`.

- `offset` tells a lane how far down the warp it should read. Lane $l$ receives
  the value held by lane $l+\text{offset}$.

For our four-value example, start with

```text
lane           0  1  2  3
value          6  4  5  1
```

At `offset = 2`, we can think of the call as

```cpp
float other_value = __shfl_down_sync(0xffffffff, value, 2);
```

Lane 0 gets lane 2's copy of `value`, which is 5. Lane 1 gets lane 3's copy,
which is 1. The shuffle only fetches those partner values. The `max` outside it
performs the actual update:

```text
lane 0: value = max(6, 5) = 6
lane 1: value = max(4, 1) = 4
```

Our remaining values are now `[6, 4]`. At `offset = 1`, lane 0 receives lane
1's updated value, which is 4:

```text
lane 0: value = max(6, 4) = 6
```

That is all the intrinsic is doing. `__shfl_down_sync` moves the partner value;
the surrounding `max` or addition decides how that value updates the running
result. In a real 32-lane warp, only the lower half contains useful combined
results after each round, but all lanes named by the mask still execute the
shuffle.

</aside>
For an offset of 16, lane 0 reads lane 16, lane 1 reads lane 17, and so on. The
offset is then halved until lane 0 contains the final result. We do not need to
move these values through global memory or even shared memory. They travel
directly between registers inside the warp.

Only lane 0 owns the final reduced value. Since every lane needs the row maximum
and denominator during normalization, lane 0 broadcasts them back to the warp.

Alright, we can now write the complete algorithm:

```text
one warp processes one row:
    every lane runs online Softmax over columns
        lane_id, lane_id + 32, lane_id + 64, ...

    row_max = reduce_max(local_max)
    broadcast row_max from lane 0

    corrected_denominator =
        local_denominator * exp(local_max - row_max)

    row_denominator = reduce_sum(corrected_denominator)
    broadcast row_denominator from lane 0

    every lane revisits its columns and writes:
        exp(X[row, column] - row_max) / row_denominator
```

For now, we are deliberately using one warp per row. If we later use multiple
warps for one row, a warp shuffle alone will not be enough because shuffles do
not cross warp boundaries. We would then need one additional reduction through
shared memory. Let us not add that machinery until we actually need it.

### Cost after coalescing

#### Memory access

Let us start with memory. The online algorithm still reads every input twice
and writes every output once:

| Operation | Elements transferred | Bytes transferred |
|---|---:|---:|
| First read of $\mathbf{X}$ | $MN$ | $4MN$ |
| Second read of $\mathbf{X}$ | $MN$ | $4MN$ |
| Write $\mathbf{Y}$ | $MN$ | $4MN$ |
| **Total useful traffic** | **$3MN$** | **$12MN$** |

So the number of useful bytes did not change. What changed is how efficiently
those bytes travel through the memory system.

Recall the transaction efficiency from the GEMV chapter. With one thread per
row, a warp could use only 128 bytes out of 1024 bytes requested from the memory
system:

$$
\eta_{\text{one thread per row}}
  =\frac{128}{1024}
  =\frac{1}{8}
  =12.5\%.
$$

With one warp reading across one row, the same 128 useful bytes occupy four
32-byte sectors:

$$
\eta_{\text{one warp per row}}
  =\frac{128}{128}
  =1.
$$

Under this simplified transaction model, the effective traffic changes from

$$
\frac{12MN}{1/8}=96MN \quad \text{bytes}
$$

to

$$
\frac{12MN}{1}=12MN \quad \text{bytes}.
$$

That is up to an $8\times$ improvement in memory-transaction efficiency. It
does not mean that the kernel must become exactly $8\times$ faster. Cache hits,
alignment, row size, exponentials, and the reduction work all affect the final
runtime. But we have stopped throwing away seven out of every eight bytes moved
by our simplified model.

Ignoring the small reduction overhead for a moment, the effective arithmetic
intensity moves from

$$
AI_{\text{uncoalesced}}
  \approx\frac{4MN}{96MN}
  =\frac{1}{24}
  \approx0.0417\ \text{FLOPs/byte}
$$

to

$$
AI_{\text{coalesced}}
  \approx\frac{4MN}{12MN}
  =\frac{1}{3}
  \approx0.333\ \text{FLOPs/byte}.
$$

**Boom!**

We did not change Softmax's theoretical arithmetic intensity here. We simply
stopped the bad access pattern from reducing its *effective* arithmetic
intensity.

#### Computation and communication

Of course, the reductions are not free. Let the warp size be $W=32$:

- Finding the row maximum requires $W-1=31$ comparisons.
- Correcting the local denominators requires $W$ subtractions, $W$
  exponentials, and $W$ multiplications.
- Reducing the corrected denominators requires $W-1=31$ additions.

The ordinary floating-point overhead per row is therefore

$$
\underbrace{W}_{\text{correction subtractions}}
+\underbrace{W}_{\text{correction multiplications}}
+\underbrace{W-1}_{\text{reduction additions}}
=3W-1 \quad \text{FLOPs}.
$$

Across all $M$ rows, that is

$$
M(3W-1) \quad \text{additional FLOPs},
$$

plus $MW$ additional exponential evaluations. For rows containing hundreds or
thousands of elements, this fixed amount of work is small compared with the
work across all $MN$ elements.

<aside class="callout" markdown="1"><strong>Communication cost of the warp algorithm</strong>

Let us first count the communication in our four-thread example.

The maximum reduction took two rounds. In the first round, threads 0 and 1 read
from threads 2 and 3. In the second round, thread 0 read the remaining value
from thread 1. Once thread 0 had the maximum, it broadcast that value to the
other threads. So finding and sharing the maximum required

$$
\underbrace{2}_{\text{reduce maximum}}
+\underbrace{1}_{\text{broadcast maximum}}
=3
$$

communication steps.

We then repeated the same pattern for the denominator: two rounds to add the
four partial denominators and one broadcast to share the result. The complete
Softmax calculation therefore required

$$
\underbrace{2}_{\text{reduce maximum}}
+\underbrace{1}_{\text{broadcast maximum}}
+\underbrace{2}_{\text{reduce denominator}}
+\underbrace{1}_{\text{broadcast denominator}}
=6
$$

warp-level communication steps for four threads.

Now we can generalize. Every reduction round cuts the number of remaining
values in half. Reducing $W$ values therefore takes
$\lceil\log_2 W\rceil$ rounds. Softmax performs two reductions and two
broadcasts, giving

$$
C_{\text{comm}}(W)
=2\left\lceil\log_2 W\right\rceil+2.
$$

For a full warp of 32 threads,

$$
C_{\text{comm}}(32)=2(5)+2=12.
$$

Those 12 steps are five shuffles to find the maximum, one broadcast of the
maximum, five shuffles to add the denominators, and one broadcast of the final
denominator.

All 32 lanes perform each step together. We are not sending 32 messages one
after another. The values also move directly between registers, so none of
this communication is added to the $12MN$ bytes moving through global memory.

If one shuffle takes $t_{\text{shuffle}}$, the communication time for one row
is approximately

$$
T_{\text{comm,row}}
\approx
\left(2\left\lceil\log_2 W\right\rceil+2\right)t_{\text{shuffle}}.
$$

</aside>

Previously, one thread handled all $N$ elements in each pass. Now each lane
handles only about $\lceil N/W\rceil$ elements per pass, followed by the fixed
communication cost above.

We added a little communication inside the warp, but in return we got coalesced
memory access and 32 threads working on the row together. That is a pretty good
deal for a memory-bound kernel.


### Turning the complete algorithm into code

Alright, we now have every piece of the final kernel. One block contains exactly
one warp, and one warp owns one row. Lane 0 handles columns
$0,32,64,\ldots$, lane 1 handles columns $1,33,65,\ldots$, and so on.

During the first pass, every lane keeps its own maximum and denominator in
registers. The warp combines those partial results, broadcasts the two row-wide
values, and makes one final coalesced pass to write the output.

```cpp
#include <cuda_runtime.h>
#include <math_constants.h>

__device__ __forceinline__ float warpReduceMax(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value = fmaxf(
            value,
            __shfl_down_sync(0xffffffffu, value, offset)
        );
    }
    return value;
}

__device__ __forceinline__ float warpReduceSum(float value) {
    for (int offset = 16; offset > 0; offset /= 2) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void softmax_warp_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    // One block contains one warp, and one warp owns one row.
    int row = blockIdx.x;
    int lane = threadIdx.x;

    if (row >= M) {
        return;
    }

    const float* input_row = input + row * N;
    float* output_row = output + row * N;

    float local_max = -CUDART_INF_F;
    float local_denominator = 0.0f;

    // First coalesced pass: compute online statistics for this lane.
    for (int column = lane; column < N; column += warpSize) {
        float current = input_row[column];

        if (current > local_max) {
            local_denominator *= expf(local_max - current);
            local_max = current;
        }

        local_denominator += expf(current - local_max);
    }

    // Reduce the local maxima. The complete result lands in lane 0.
    float row_max = warpReduceMax(local_max);

    // Broadcast the row maximum from lane 0 to every lane.
    row_max = __shfl_sync(0xffffffffu, row_max, 0);

    // Put every local denominator on the same scale.
    float corrected_denominator =
        local_denominator * expf(local_max - row_max);

    // Reduce the corrected denominators and broadcast their sum.
    float row_denominator = warpReduceSum(corrected_denominator);
    row_denominator =
        __shfl_sync(0xffffffffu, row_denominator, 0);

    // Second coalesced pass: normalize and write the row.
    for (int column = lane; column < N; column += warpSize) {
        output_row[column] =
            expf(input_row[column] - row_max) / row_denominator;
    }
}

void launch_softmax(
    const float* __restrict__ input,
    float* __restrict__ output,
    int M,
    int N
) {
    constexpr int threads_per_block = 32;

    dim3 block_size(threads_per_block);
    dim3 grid_size(M);

    softmax_warp_kernel<<<grid_size, block_size>>>(
        input,
        output,
        M,
        N
    );
}
```

The launcher uses exactly 32 threads because the reduction helpers communicate
inside one warp. The kernel assumes that every row contains at least one
element. A row can be wider than 32 elements; the loop simply gives each lane
more columns separated by `warpSize`.

Notice that the code follows the same order as our worked example:

```text
compute one (local_max, local_denominator) pair per lane
-> reduce and broadcast row_max
-> correct every local_denominator
-> reduce and broadcast row_denominator
-> normalize the row
```

## Main takeaways

Alright, that was a lot of work for an operation that fits in one line of
mathematics. We started with the stable Softmax equation, wrote the most direct
CUDA kernel, and then kept changing the algorithm until its execution matched
the way a GPU wants to read memory and share work.

If I had to compress the entire worklog into a few points, these would be the
ones:

- **Numerical stability comes first.** Subtracting the row maximum keeps the
  largest exponential equal to 1 and prevents large inputs from overflowing.
  A fast Softmax that is numerically unstable is not useful.

- **The obvious parallelization still left each row sequential.** Giving one
  row to one thread parallelized the $M$ rows, but one thread still performed
  every operation within its row.

- **Online Softmax trades a little arithmetic for one fewer memory pass.** By
  correcting the running denominator whenever the maximum changes, we fused
  the maximum and denominator passes. Global-memory traffic fell from $16MN$
  bytes to $12MN$ bytes, even though we added some rescaling work.

- **Coalescing is decided by a warp, not by one thread.** With one thread per
  row, neighboring lanes accessed different rows. With one warp per row,
  neighboring lanes accessed neighboring columns.

- **Splitting a row means no lane has the complete answer.** Each lane produces
  a local maximum and a local denominator. Warp reductions turn those 32
  partial answers into one row-wide answer.

- **Partial denominators must use the same maximum before we add them.** The
  factor $e^{m_{\text{local}}-m_{\text{row}}}$ puts every local denominator on
  the same scale before the sum reduction.

- **A little on-chip communication buys a lot of useful parallelism.** A
  32-lane warp pays 12 communication steps per row. In return, 32 lanes work on
  that row together and its global-memory accesses become coalesced.

And that is really the lesson here. We did not change the mathematical
definition of Softmax. We changed the order in which its statistics are
computed and which threads own the work. That was enough to remove a complete
read of the input, recover efficient memory transactions, and expose
parallelism inside every row.
