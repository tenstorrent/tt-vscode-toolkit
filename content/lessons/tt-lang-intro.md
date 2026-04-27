---
id: tt-lang-intro
title: "Introduction to tt-lang"
description: >-
  Write your first tt-lang kernel: a concurrent compute + data-movement program
  that runs on the Tensix grid. Try it live in the browser via ttlang-sim-lite.
category: compilers
tags:
  - tt-lang
  - sim
  - kernels
  - tensix
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
  - sim
status: draft
estimatedMinutes: 20
playground: ttlang-sim
---

# Introduction to tt-lang

You're already running on Tensix — the playground above is a live kernel
sandbox, no hardware or install required. Pick any kernel from the dropdown
and hit **Run** to see it execute on a simulated Tensix grid.

| Kernel | What it teaches |
|--------|----------------|
| **Element-wise Add** | Minimal DFB producer/consumer loop — the hello-world of tt-lang |
| **Fused Multiply-Add** | Three DFBs, zero intermediate DRAM writes — what fusion looks like |
| **Matmul + Bias + ReLU** | K-reduction accumulator ping-pong; the core matmul pattern on Tensix |
| **Row-partitioned Matmul** | `ttl.node()` work partitioning across a multi-core grid |

To understand what just ran, read on.

## The DRAM Wall

TTNN dispatches each op as a separate kernel. Between every op, tensor data
lands in DRAM. For a single transformer layer, that's roughly five DRAM
write/read round-trips — RMSNorm → DRAM → projection → DRAM → attention →
DRAM → projection → DRAM → FFN → DRAM. At model scale, this is the
bottleneck, not compute.

**tt-lang breaks that pattern.** You write the full fused operation as one
kernel. Input tiles stream in from DRAM once via Data Movement threads, flow
through L1 using Dataflow Buffers (DFBs), compute happens in registers, and
results drain to DRAM once. One read. One write. Everything in between stays
in L1.

Real measured improvements from production projects:

| Project | What was fused | Improvement |
|---------|---------------|-------------|
| [SkyReels-1.3B transformer block](https://github.com/zoecarver/tt-lang-models) | 5 ops → 1 kernel | 3–5× vs TTNN |
| [DFlash speculative decoder](https://github.com/zoecarver/dflash) | RoPE, RMSNorm, SiLU, residuals | 5–6× decode speedup |
| [DeepSeek Engram module](https://github.com/zoecarver/Engram) | gating + depthwise conv | 2.2× all kernels; 3.4× gating alone |
| [nanochat fused MLP](https://github.com/zoecarver/nanochat/commit/f849d3f) | 7 dispatches → 1 | +21% tok/s (13.13 → 15.89) |
| [Qwen-Image generation](https://github.com/zoecarver/tt-lang-models) | attention + norms | 4–8× vs XLA at 512×512 |

## What People Have Built

## Getting tt-lang

## The Tensix Thread Model

## Kernel Patterns

## Claude Code Slash Commands

## What's Next
