# How to make your models fast

A slowly growing collection of CUDA kernel worklogs for understanding inference
performance from first principles. The focus is currently narrow: write one
kernel, count its work and memory traffic, inspect what the GPU actually does,
and improve one bottleneck at a time.

## Read locally

```bash
bundle exec jekyll serve --baseurl ""
```

Jekyll renders the complete book into `_site/` using the Distill book theme.

## Current chapters

- [GEMV](_pages/chapters/gemv.md)
- [Softmax](_pages/chapters/softmax.md)
- [LayerNorm](_pages/chapters/layernorm.md)
- [GEMM](_pages/chapters/gemm.md)

Jekyll builds these Distill pages into `_site/`.

## CUDA implementations

- [GEMV kernels](kernels/gemv/)
- [Softmax kernels](kernels/softmax/)
- [LayerNorm kernels](kernels/layernorm/)
- [GEMM kernels](kernels/gemm/)
