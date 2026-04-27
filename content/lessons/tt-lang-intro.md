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

These are real projects built with tt-lang — each started as a "what if" and
ended with custom Tensix kernels running in production or close to it.

**[SkyReels-1.3B](https://github.com/zoecarver/tt-lang-models)** — The full
WAN transformer block fused into a single kernel on QB2 (4-chip Blackhole).
Five ops collapsed into one: input tiles stream in once, compute flows through
L1, results drain to DRAM once. 3–5× throughput improvement over op-by-op
TTNN dispatch at production model dimensions.

**[WAN Animate 14B](https://github.com/tenstorrent/tt-lang)** — A 40-layer,
5120-hidden diffusion transformer brought up on a 4-chip QB2 (2×2 mesh).
TT-Lang kernels cover 3D RoPE, AdaLN modulation, and attention softcap.
The bring-up involved debugging seven integration bugs across the pipeline in
a single session — possible because the functional simulator catches DFB
deadlocks before touching hardware.

**[Freeciv game AI](https://github.com/tenstorrent/tt-lang)** — tt-lang
kernels accelerating Freeciv's map generation (Perlin noise terrain) and
pathfinding, developed and validated entirely in the functional simulator. A
game AI that started as a "what if we ran this on Tensix?" and ended with
working Tensix kernels — no hardware required to get there.

**[DFlash speculative decoder](https://github.com/zoecarver/dflash)** — A
draft model that proposes 16 tokens in parallel, verified by Qwen3-30B as the
target. Draft kernels (RoPE, RMSNorm, SiLU, residuals) run entirely on-device
via TT-Lang. Result: 5–6× decode speedup end-to-end, 93ms draft forward pass
with caching (vs 887ms without), acceptance rate matching the PyTorch
reference.

**[Oasis — real-time Minecraft](https://github.com/zoecarver/open-oasis)** —
A 500M diffusion transformer generating Minecraft frames on a single Blackhole
card, running at 8 FPS in a single captured trace with 4-way tensor
parallelism. Interactive browser play across 26 Atari games. Everything the
DiT needs — denoising, VAE decode, video output — runs in one end-to-end
trace.

## Getting tt-lang

### Browser (already done)

The playground above is the browser path. No install, no hardware. Use it to
prototype and explore before setting up a local environment.

### Local — ttsim (no Tenstorrent hardware required)

ttsim is a full-system simulator for Wormhole and Blackhole. It runs any
tt-metal/tt-lang workload on Linux/x86_64 — including Windows via WSL2 —
without a Tenstorrent card. Results are bit-exact with silicon for all
documented code paths.

**Prerequisites:** tt-metal built and `TT_METAL_HOME` set.
See the [Build tt-metal lesson](command:tenstorrent.showLesson?["build-tt-metal"]) if you haven't done this yet.

```bash
# Download the simulator binary — choose Wormhole or Blackhole
mkdir -p ~/sim && cd ~/sim

# Wormhole (N150, N300, T3K, Galaxy)
wget https://github.com/tenstorrent/ttsim/releases/download/v1.5.4/libttsim_wh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# OR: Blackhole (P100, P150, P300c, QB2)
wget https://github.com/tenstorrent/ttsim/releases/download/v1.5.4/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
```

Set the simulator env vars and run any tt-metal example:

```bash
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so   # or libttsim_bh.so for Blackhole
export TT_METAL_SLOW_DISPATCH_MODE=1              # required — fast dispatch is in progress

cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

Check [ttsim releases](https://github.com/tenstorrent/ttsim/releases/latest) for newer versions — the download URL format is stable, just replace the version number.

### Local — build tt-lang

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
# Follow docs/sphinx/build.md for CMake options and build modes
source build/env/activate   # required before running any tt-lang command
python examples/eltwise_add.py
```

`source build/env/activate` must be run every new shell session before using tt-lang. The [tt-lang docs](https://github.com/tenstorrent/tt-lang) cover build options including simulator-only mode (no hardware required).

### Real hardware

If you have a Tenstorrent card, skip the `TT_METAL_SIMULATOR` and
`TT_METAL_SLOW_DISPATCH_MODE` variables. Everything else is identical.

## The Tensix Thread Model

## Kernel Patterns

## Claude Code Slash Commands

## What's Next
