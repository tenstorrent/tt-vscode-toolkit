---
id: cs-fundamentals-08-matmul-labs
title: "Module 8: Matrix Math and Matmul Labs"
description: >-
  Learn matrix math on Tenstorrent by walking Lab 1/2/3 matmul progression:
  single-core tiles, multi-core work split, and multicast data reuse.
category: cs-fundamentals
tags:
  - matmul
  - matrix-math
  - metalium
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
  - simulator
status: draft
estimatedMinutes: 45
---

# Module 8: Matrix Math and Matmul Labs

## Introduction: The Kernel Pattern Behind Modern AI

`C = A × B` looks simple. In practice, matrix multiply is the dominant kernel in transformer inference and training.

This module ports the full Lab 1 → Lab 2 → Lab 3 arc from TT-Metalium<sup>™</sup> docs into the interactive lesson style used by the rest of this collection.

You will move from:

1. **One core + tiled matmul** (correctness and dataflow)
2. **Many cores + split work** (parallel decomposition)
3. **Multicast + reuse** (network-aware optimization)

By the end, you should be able to explain not just *what* command to run, but *why* each optimization exists.

### What You'll Learn

- ✅ Why 32×32 tiling is the practical unit for matrix kernels
- ✅ How reader/compute/writer kernels compose a full matmul pipeline
- ✅ How `split_work_to_cores(...)` maps output tiles across a core grid
- ✅ Why multicast + semaphores reduce DRAM pressure in multi-core kernels
- ✅ How to run and study the same progression on hardware or simulator

---

## Upstream Labs (Source of Truth)

Use these as the canonical references while you work through this module:

- Lab 1: Single Core Matrix Multiplication  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst
- Lab 2: Multi Core Matrix Multiplication  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab2/lab2.rst
- Lab 3: Multicast for Improved Data Reuse  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst

---

## Part 1: Concept Primer (Before You Run Anything)

### Matmul refresher

For `A[M×K]`, `B[K×N]`, output `C[M×N]`:

```text
C[i, j] = Σ (A[i, k] * B[k, j]) for k in [0, K)
```

A naive kernel streams every operand from DRAM repeatedly. The labs show why that is too expensive and how to move reuse closer to compute.

### Why tiles matter

In these labs, matmul is expressed as tile operations (commonly 32×32):

- Tiles fit naturally into local scratchpad workflows
- Reader kernels stage tile blocks for compute
- Compute kernels execute matrix ops on staged tiles
- Writer kernels flush output tiles back to DRAM

Think of each output tile as an independent unit of work that can be assigned to one core (Lab 1) or many cores (Lab 2/3).

### Dataflow mental model

```text
DRAM A/B tiles -> Reader kernel -> Circular buffers in L1
                          -> Compute kernel -> Accumulator tiles
                          -> Writer kernel -> DRAM C tiles
```

In Lab 3, the first arrow is optimized further with NoC multicast so multiple cores can consume one producer's data movement.

---

## Part 2: Environment Setup (Hardware or Simulator)

If `~/tt-metal` is missing (common on TT-QuietBox<sup>®</sup> 2 preconfigured images), start here first:

[🛠️ Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"])

Build programming examples used by this module:

```bash
cd ~/tt-metal
./build_metal.sh --build-programming-examples
```

[🔨 Build Programming Examples](command:tenstorrent.buildProgrammingExamples)

### Optional simulator path (hardware-free)

[🧪 Set Up ttsim Simulator](command:tenstorrent.setupTtsim)

```bash
# Wormhole simulator
export TT_METAL_SIMULATOR=$HOME/sim/libttsim_wh.so

# Blackhole simulator (P-series)
# export TT_METAL_SIMULATOR=$HOME/sim/libttsim_bh.so

# Required for many simulator workflows
export TT_METAL_SLOW_DISPATCH_MODE=1

cd ~/tt-metal
```

More simulator workflows: [Twenty-and-Ten Things You Can Do with ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])

---

## Part 3: Lab 1 Deep Dive — Single-Core Tiled Matmul

### Goal of Lab 1

Build intuition for tiled data movement and kernel decomposition on a **single core**.

### Run it

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_single_core
```

### What success looks like

```text
Metalium vs Golden -- PCC = ...
Test Passed
```

### What to inspect while reading Lab 1 code

- Tile dimensions and how matrix dimensions map to tile counts
- Reader runtime args (which tile block is loaded)
- Compute loop structure over `K` tiles
- Writer mapping from tile index -> output tensor location

### Why this matters

Lab 1 is the correctness baseline. If you cannot reason about tile lifecycle here, Lab 2 and Lab 3 optimizations become opaque.

### Checkpoint questions

- If `K` doubles, which loop grows and why?
- Why is output accumulation tied to the `K` sweep?
- What changes if matrices are not multiples of tile size?

---

## Part 4: Lab 2 Deep Dive — Multi-Core Work Distribution

### Goal of Lab 2

Scale the same tiled algorithm across many cores by partitioning output tiles.

### Run baseline multi-core

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_multi_core
```

### Run reuse-oriented variant

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_multicore_reuse
```

### Core idea

`split_work_to_cores(...)` (or equivalent decomposition logic) partitions output tiles so each core receives a contiguous share of the output space.

### What to inspect in Lab 2

- How total output tiles are divided by core count
- How per-core runtime args encode tile ranges
- Whether compute is balanced (similar work per core)
- Where synchronization/barrier points appear between stages

### Performance intuition

Lab 2 often improves throughput because:

1. More cores compute output tiles concurrently
2. Per-core work can stay local in L1 buffers
3. The kernel pipeline overlaps data movement and compute better

But scaling is never free: load imbalance and DRAM contention can flatten speedup.

### Checkpoint questions

- If one core gets extra tiles, what happens to total runtime?
- Why can more cores still bottleneck on the same DRAM channels?
- Which dimensions (`M`, `N`, `K`) most influence partition quality?

---

## Part 5: Lab 3 Deep Dive — Multicast Data Reuse

### Goal of Lab 3

Reduce redundant memory traffic by sharing operand tiles across cores through NoC multicast rather than issuing repeated DRAM reads.

### Conceptual flow

```text
Producer core reads source tile once
-> multicasts tile to receiver core set
-> semaphore/handshake guarantees readiness
-> receivers consume tile for compute
```

### Toolkit runnable multicast check

```bash
cd ~/tt-metal
if [ -x ./build/programming_examples/contributed/multicast ]; then
  ./build/programming_examples/contributed/multicast
else
  echo "multicast example not found yet. Rebuild with ./build_metal.sh --build-programming-examples and follow Lab 3 source walkthrough."
fi
```

### What to inspect in Lab 3

- Sender/receiver role assignment
- NoC destination group construction
- Semaphore increment/wait sequence correctness
- How many DRAM reads are eliminated versus unicast pattern

### Why this matters for transformers

Attention and MLP blocks repeatedly reuse shared weights/activations. Multicast patterns directly target those reuse opportunities and reduce bandwidth pressure.

### Checkpoint questions

- Why is synchronization mandatory before receivers consume multicast data?
- What failure mode appears if a receiver reads before producer completion?
- Which workloads benefit most from multicast (high fanout vs low fanout)?

---

## Part 6: Guided RST-to-Interactive Study Plan

Use this sequence to get the "full thrust" of the original labs while staying hands-on:

1. **Read Lab 1 objective + run single-core binary**
2. **Trace one output tile end-to-end (reader -> compute -> writer)**
3. **Read Lab 2 partition section + run multi-core binaries**
4. **Map output tile ranges to cores and confirm balance assumptions**
5. **Read Lab 3 multicast section + run multicast example / source walk-through**
6. **Sketch traffic difference: repeated DRAM reads vs one read + multicast fanout**

If you're teaching this module, require students to answer all checkpoint questions before advancing.

---

## Part 7: Troubleshooting and Validation Signals

### Build or binary missing

```bash
cd ~/tt-metal
./build_metal.sh --build-programming-examples
```

### Simulator runs are very slow

That is expected with `TT_METAL_SLOW_DISPATCH_MODE=1`. Use it for understanding correctness and dataflow, not throughput benchmarking.

### Numerical mismatch or failed validation

- Re-run on small dimensions first
- Confirm environment variables are sane
- Verify build is fresh after any source edits

### Repro checklist

- `TT_METAL_SIMULATOR` set correctly (or unset on hardware)
- `TT_METAL_SLOW_DISPATCH_MODE=1` for simulator
- Programming examples freshly built
- Expected `Test Passed` / positive PCC signal observed

---

## Part 8: Mastery Checklist

Before moving on, confirm you can explain:

- ✅ Why tiled matmul beats naive scalar formulations on this architecture
- ✅ How Lab 2 partitions output work across cores
- ✅ Why balancing core work matters for wall-clock runtime
- ✅ How Lab 3 multicast reduces redundant DRAM reads
- ✅ Where semaphore synchronization protects correctness in multicast flows
- ✅ How these patterns transfer directly to transformer kernels

---

## What's Next?

- Revisit simulator experiments in [ttsim Twenty-and-Ten](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"]) and repeat the flow with controlled dimensions.
- Continue with [Explore Metalium](command:tenstorrent.showLesson?["explore-metalium"]) to inspect lower-level kernel/runtime details.

[→ Continue to Bounty Program: Model Bring-Up](command:tenstorrent.showLesson?["bounty-program"])
