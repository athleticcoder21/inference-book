---
layout: distill
title: "How to make your models fast"
subtitle: "CUDA kernel worklogs for understanding inference performance"
description: "A slowly growing book about reasoning through CUDA kernels from first principles."
date: 2026-09-10
section_number: 0
previous_section_url: ./
previous_section_name: ""
next_section_url: gemv
next_section_name: "GEMV"
authors:
  - name: Anshuman Mishra
    url: https://heyyanshuman.com
    affiliations:
      name: Independent researcher
toc:
  - name: How each worklog proceeds
  - name: Current chapters
---

I am learning how to make model inference faster by starting at the lowest level
I can reasonably reach: individual CUDA kernels.

This book is a slowly growing compilation of those worklogs. For now, its scope
is deliberately narrow. We write one kernel, count its computation and memory
traffic, inspect how its threads use the GPU, and then improve it one bottleneck
at a time.

This is not meant to be a broad guide to every layer of model scaling. There is
no large roadmap that I am promising to fill in. I will add a chapter when I
have learned enough to explain the problem honestly and work through an
implementation.

## How each worklog proceeds

Every chapter follows roughly the same loop:

1. Write the most straightforward implementation.
2. Count its FLOPs, memory traffic, and arithmetic intensity.
3. Look at what a warp actually computes and reads.
4. Change one part of the algorithm or its mapping to the GPU.
5. Recalculate the cost and write the improved kernel.

The intermediate versions stay in the chapter. The point is not only to arrive
at fast code, but to make the reasoning that led there reusable.

## Current chapters

{% include chapter-card.html %}
