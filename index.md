---
layout: distill
title: "How to make your models fast"
subtitle: "A systems view of LLM inference on GPUs"
description: "A work-in-progress book for learning inference performance from first principles: kernels, memory traffic, transformer execution, and serving systems."
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
  - name: The question
  - name: How the book grows
---

## The question

Inference optimization is not magic. Every performance result is a consequence
of computation, memory traffic, communication, and how an algorithm maps onto
the hardware underneath it.

This book develops a practical way to answer one question:

> What is stopping this workload from going faster—compute, memory,
> communication, or the way it is mapped onto the GPU?

Each chapter starts with a model of the work, follows it with a runnable
implementation, and then uses measurements to challenge that model. The code
lives beside the writing so the reasoning can be inspected and reproduced.

## How the book grows

The structure moves from a single GPU outward:

1. **Kernels** — memory layout, warp execution, tiling, and reductions.
2. **Transformer inference** — attention, MLPs, prefill, decode, and KV caches.
3. **Serving systems** — batching, scheduling, quantization, and latency versus throughput.
4. **Scaling out** — parallelism, communication, and fleet-level efficiency.

Only finished chapters appear in the navigation. Start with
[GEMV](gemv).
