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

This module brings the official TT-Metalium<sup>™</sup> matmul labs directly into the CS Fundamentals track.

## Upstream Labs (source of truth)

- Lab 1: Single Core Matrix Multiplication  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst
- Lab 2: Multi Core Matrix Multiplication  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab2/lab2.rst
- Lab 3: Multicast for Improved Data Reuse  
  https://github.com/tenstorrent/tt-metal/blob/main/docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst

---

## Part 1: Environment Setup (hardware or simulator)

If `~/tt-metal` is missing (common on TT-QuietBox<sup>®</sup> 2 preconfigured images), build from source first:

[🛠️ Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"])

Build programming examples:

```bash
cd ~/tt-metal
./build_metal.sh --build-programming-examples
```

[🔨 Build Programming Examples](command:tenstorrent.buildProgrammingExamples)

### Optional: run this module on ttsim (no hardware required)

[🧪 Set Up ttsim Simulator](command:tenstorrent.setupTtsim)

```bash
# Wormhole simulator
export TT_METAL_SIMULATOR=$HOME/sim/libttsim_wh.so
# Blackhole simulator (P-series):
# export TT_METAL_SIMULATOR=$HOME/sim/libttsim_bh.so

export TT_METAL_SLOW_DISPATCH_MODE=1
cd ~/tt-metal
```

More simulator workflows: [Twenty-and-Ten Things You Can Do with ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])

---

## Part 2: Lab 1 — Single-Core Matmul

**What to focus on in Lab 1:**
- Row-major vs tiled memory layout
- Why 32×32 tiles are the basic compute unit
- Reader/compute/writer kernel roles on one core

Run the single-core reference example:

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_single_core
```

Expected signal:

```text
Metalium vs Golden -- PCC = ...
Test Passed
```

---

## Part 3: Lab 2 — Multi-Core Work Distribution

**What to focus on in Lab 2:**
- `split_work_to_cores(...)` for balanced SPMD partitioning
- Per-core runtime args for reader/compute/writer kernels
- Tradeoff between more cores vs better tile reuse per core

Run multi-core baseline:

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_multi_core
```

Then run the data-reuse variant:

```bash
cd ~/tt-metal
./build/programming_examples/metal_example_matmul_multicore_reuse
```

---

## Part 4: Lab 3 — Multicast Reuse Across Cores

**What to focus on in Lab 3:**
- Why DRAM re-reads dominate at scale
- NoC multicast sender/receiver flow
- Semaphore protocol for correctness

Run the multicast primitive example used in existing toolkit lessons:

```bash
cd ~/tt-metal
if [ -x ./build/programming_examples/contributed/multicast ]; then
  ./build/programming_examples/contributed/multicast
else
  echo "multicast example not found yet. Rebuild with ./build_metal.sh --build-programming-examples and follow Lab 3 source walkthrough."
fi
```

---

## Part 5: Practical Matmul Checklist

Before moving on, confirm you can explain:

- ✅ Why tiled layout improves matrix engine efficiency
- ✅ How output tiles are split across cores in Lab 2
- ✅ Why Lab 3 multicast reduces DRAM pressure
- ✅ Why these ideas directly map to transformer inference

---

## What's Next?

- Revisit simulator performance paths in [ttsim Twenty-and-Ten](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"]) for repeatable experiments.
- Apply these same optimization patterns when exploring [TT-Metalium internals](command:tenstorrent.showLesson?["explore-metalium"]).

[→ Continue to Bounty Program: Model Bring-Up](command:tenstorrent.showLesson?["bounty-program"])
