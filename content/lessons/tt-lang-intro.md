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

## What People Have Built

## Getting tt-lang

## The Tensix Thread Model

## Kernel Patterns

## Claude Code Slash Commands

## What's Next
