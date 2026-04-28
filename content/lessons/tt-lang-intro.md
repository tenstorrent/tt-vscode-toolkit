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
estimatedMinutes: 25
playground: ttlang-sim
---

# Introduction to tt-lang

If you use Claude Code, Copilot, or any AI coding agent in your daily work,
you've probably already had a version of this moment: you're moving fast,
shipping Python, and an AI is helping you every step of the way. Then someone
says *"we need to optimize this inference pipeline"* or *"we need a custom kernel
for this attention pattern"* — and suddenly you're in territory where AI
assistance gets murky. The abstraction gap between "Python-level thinking" and
"hardware-level implementation" is just too wide for the tools to bridge cleanly.

tt-lang was built specifically to close that gap. And it was designed — from
day one, not as an afterthought — to be used alongside AI coding agents. The
playground above is running a live kernel in your browser right now, no install
required. Hit **Run** and watch it go. Then read on.

---

## The Hardware Behind the Language

Tenstorrent builds AI accelerators with a fundamentally different architecture
than GPUs. To understand why tt-lang exists, you first need a picture of what
it's running on.

Most AI accelerators — NVIDIA GPUs, Google TPUs — are built around a central
compute cluster with a shared memory pool. They're extraordinarily fast at the
math, but the memory hierarchy creates a bottleneck that's hard to escape: every
time a tensor operation finishes, its result writes out to DRAM, and the next
operation reads it back in. The chips can calculate faster than they can move
data.

Tenstorrent hardware is built around **Tensix cores** — a 2D grid of small,
independent processor tiles. Each tile has its own fast SRAM (called L1), its
own compute unit, and its own programmable data movement engines. Instead of
one giant shared memory pool, you have a fabric of processors that each own
their local memory and can stream data between themselves and off-chip DRAM with
precise, explicit control.

The key insight: on Tensix, you decide when data lives in L1 and when it moves
to DRAM. You decide how compute and data movement overlap. That control — used
well — is where the performance comes from. And giving you clean Python
abstractions over that control is exactly what tt-lang does.

---

## Where tt-lang Fits

The Tenstorrent software stack has three layers, and tt-lang occupies the
middle — the same position that Triton occupies in the NVIDIA ecosystem:

| Layer | What it is | Analogous to |
|-------|-----------|--------------|
| **TTNN** | High-level tensor ops — `ttnn.matmul`, `ttnn.softmax`, ready to use | PyTorch / cuDNN |
| **tt-lang** | Python DSL for custom fused kernels with explicit data movement | OpenAI Triton |
| **TT-Metalium** | Full hardware control — every register, every DMA, every semaphore | CUDA C / PTX |

If you've used PyTorch on a GPU, you've lived at the top layer — calling
library functions and letting cuDNN figure out the rest. If you've written a
Triton kernel to fuse ops for performance, you've been in the middle. If you've
written CUDA C to squeeze every cycle from GPU hardware, you've been at the
bottom. tt-lang is Tenstorrent's equivalent of that middle layer.

The rule of thumb: **start with TTNN**. When TTNN can't express what you need,
or when you need more performance than pre-built ops can deliver, drop into
tt-lang. When even tt-lang isn't enough (rare), go to TT-Metalium.

---

## Why This Language Exists

The tt-lang README describes the problem that drove its creation:

> *"Tenstorrent developers today face a choice between TT-NN — which provides
> high-level operations that are straightforward to use but lack the expressivity
> needed for custom kernels — and TT-Metalium — which provides full hardware
> control through explicit low-level management of memory and compute. The problem
> is that there is no middle ground where the compiler handles what it does best
> — resource management, validation, optimization — while maintaining high
> expressivity for application-level concerns."*

TT-Metalium is powerful but takes weeks to learn and demands hardware debugging
expertise. TTNN is approachable but can't express fused custom ops. Engineers
porting models kept hitting the same wall: they'd need to fuse a sequence of
TTNN ops for performance, and the only path forward was a full rewrite in
TT-Metalium. That's a multi-week detour just to get one optimization landed.

tt-lang bridges that gap through **progressive disclosure**: simple kernels
require minimal specification — the compiler infers NOC addressing, register
allocation, and memory layout from high-level Python syntax. Complex kernels let
you open the hood and control every aspect of pipelining and synchronization.
You write Python. The compiler generates the metal.

## Designed for AI-Assisted Development

Here's what makes tt-lang different from every other hardware DSL: it was
explicitly designed to be used with AI coding agents, and the design choices
reflect that at every level.

The README states it plainly:

> *"Python as the host language enables AI tools to translate GPU DSL kernels
> (Triton, CUDA, cuTile, TileLang) to Tenstorrent hardware more reliably than
> direct TT-Metalium translation, while tight integration with functional
> simulation will allow AI agents to propose kernel implementations, validate
> correctness, and iterate on configurations autonomously. Developers should be
> able to catch errors and performance issues in their IDE rather than on
> hardware."*

This matters in practice. When you ask an AI agent to write a CUDA kernel or a
Triton kernel, it can draw on years of training data and produce something
reasonable. When you ask it to write TT-Metalium C++, the abstraction gap is so
large that the results are usually wrong in subtle ways — wrong memory addresses,
wrong synchronization primitives, hardware assumptions baked in incorrectly.
tt-lang narrows that gap dramatically: the concepts map cleanly from Triton or
CUDA, the functional simulator catches mistakes instantly, and the tooling is
designed to support an iterative agent-driven workflow.

The TT Developer Toolkit ships `/ttl-*` Claude Code slash commands that embody
this workflow: `/ttl-import` translates an existing kernel from CUDA, Triton,
or PyTorch; `/ttl-simulate` validates it in the simulator; `/ttl-profile` shows
where cycles are going; `/ttl-optimize` applies targeted improvements. You can
go from a PyTorch attention function to a validated, profiled Tensix kernel in
a single Claude Code session — hardware not required.

More on those commands later. First, the performance story that explains why
any of this effort is worthwhile.

---

## The DRAM Wall: Why Fusion Matters

This is the concrete performance story. It's worth understanding before you
write a single line of tt-lang.

When TTNN executes a model, it dispatches each operation as a separate kernel.
Between every op, tensor data writes out to DRAM and reads back in. For a
single transformer layer, that's a chain of round-trips that looks like this:

```
Input → DRAM → RMSNorm → DRAM → Projection → DRAM
      → Attention → DRAM → Projection → DRAM → FFN → DRAM
```

At model scale, **this memory traffic is the bottleneck, not compute**. The
chips can calculate faster than they can move data. Every unnecessary DRAM write
is wasted cycles.

**tt-lang breaks that pattern.** You write the full fused operation as one
kernel. Input tiles stream in from DRAM once. They flow through L1 using
Dataflow Buffers (typed ring buffers shared between threads). Compute happens in
registers. Results drain to DRAM once. **One read. One write. Everything in
between stays in L1.**

Here's what that looks like in practice — real measured improvements from
production projects:

| Project | What was fused | Improvement |
|---------|---------------|-------------|
| [SkyReels-1.3B transformer block](https://github.com/zoecarver/tt-lang-models) | 5 ops → 1 kernel | 3–5× vs TTNN |
| [DFlash speculative decoder](https://github.com/zoecarver/dflash) | RoPE, RMSNorm, SiLU, residuals | 5–6× decode speedup |
| [DeepSeek Engram module](https://github.com/zoecarver/Engram) | gating + depthwise conv | 2.2× all kernels; 3.4× gating alone |
| [nanochat fused MLP](https://github.com/zoecarver/nanochat/commit/f849d3f) | 7 dispatches → 1 | +21% tok/s (13.13 → 15.89) |
| [Qwen-Image generation](https://github.com/zoecarver/tt-lang-models) | attention + norms | 4–8× vs XLA at 512×512 |

These aren't synthetic benchmarks — they're from real projects by Tenstorrent
engineers and contributors pushing production models.

---

## What People Have Built

These are real projects built with tt-lang. Each one started as a "what if we
ran this on Tensix?" and ended with working kernels.

**[SkyReels-1.3B](https://github.com/zoecarver/tt-lang-models)** — The full WAN
video transformer block fused into a single kernel on QB2 (4-chip Blackhole).
Five ops collapsed into one: input tiles stream in once, compute flows through
L1, results drain to DRAM once. 3–5× throughput improvement over op-by-op TTNN
dispatch at production model dimensions.

**[WAN Animate 14B](https://github.com/tenstorrent/tt-lang)** — A 40-layer,
5120-hidden diffusion transformer brought up on a 4-chip QB2 (2×2 mesh).
TT-Lang kernels cover 3D RoPE, AdaLN modulation, and attention softcap.
The bring-up involved debugging seven integration bugs in a single session —
possible because the functional simulator catches DFB deadlocks before touching
hardware.

**[DFlash speculative decoder](https://github.com/zoecarver/dflash)** — A draft
model that proposes 16 tokens in parallel, verified by Qwen3-30B as the target.
Draft kernels (RoPE, RMSNorm, SiLU, residuals) run entirely on-device via
tt-lang. Result: 5–6× decode speedup end-to-end, 93ms draft forward pass with
caching (vs 887ms without).

**[Freeciv game AI](https://github.com/tenstorrent/tt-lang)** — tt-lang kernels
accelerating Freeciv's map generation (Perlin noise terrain) and pathfinding,
developed and validated entirely in the functional simulator. A game AI that
went from idea to working Tensix kernels without any hardware.

**[Oasis — real-time Minecraft](https://github.com/zoecarver/open-oasis)** — A
500M diffusion transformer generating Minecraft frames on a single Blackhole
card at 8 FPS, with 4-way tensor parallelism. Everything the DiT needs —
denoising, VAE decode, video output — runs in one end-to-end trace.

---

## How a Tensix Core Works

Now let's build the mental model you need to write kernels.

Think of a Tensix core as a kitchen. There's one cook (the **Compute thread**)
who does the actual math. There's a prep worker (**Data Movement 0**) who fetches
ingredients (input tensors) from the pantry (DRAM) and sets them on the
counter. There's a server (**Data Movement 1**) who takes finished dishes off the
counter and delivers them back to the pantry.

The three threads run **simultaneously**. While the cook is preparing one tile,
the prep worker is already fetching the next one. While the server is writing
the previous result, the cook is finishing the current one. This overlap — hiding
DMA latency behind compute — is where the throughput comes from.

| Thread | Role |
|--------|------|
| **Compute** | Math — matrix multiply, activations, reductions |
| **Data Movement 0 (DM0)** | Reads input tiles from DRAM into L1 |
| **Data Movement 1 (DM1)** | Writes output tiles from L1 to DRAM |

The threads coordinate through **Dataflow Buffers (DFBs)** — typed ring buffers
in L1 with a fixed number of slots. The DFB is the counter between the prep
worker and the cook: the prep worker fills a slot and rings a bell
(`reserve()`); the cook waits for the bell, processes the tile, and clears the
slot (`wait()`).

```
DRAM ──[DM0 reads]──► DFB_in ──[Compute]──► DFB_out ──[DM1 writes]──► DRAM
```

Two DFB primitives drive all synchronization:

- **`wait()`** — consumer role: blocks until a filled slot is ready to read
- **`reserve()`** — producer role: blocks until an empty slot is available to write

That's it. Two primitives. Everything in tt-lang flows from them.

---

## Reading the Playground Kernels

Go back to the playground at the top of this page. Each kernel in the dropdown
demonstrates a different concept:

| Kernel | What it teaches |
|--------|----------------|
| **Element-wise Add** | Minimal DFB producer/consumer loop — the hello-world of tt-lang |
| **Fused Multiply-Add** | Three DFBs, zero intermediate DRAM writes — what fusion looks like |
| **Matmul + Bias + ReLU** | K-reduction accumulator ping-pong; the core matmul pattern on Tensix |
| **Row-partitioned Matmul** | `ttl.node()` work partitioning across a multi-core grid |

Run "Element-wise Add" first. You'll see three function definitions inside the
`@ttl.operation()` decorator: one marked `@ttl.compute()` and two marked
`@ttl.datamovement()`. Those are the three Tensix threads.

---

## Kernel Patterns

### Element-wise addition

> **Try it now:** select "Element-wise Add" in the playground above.

The simplest possible kernel: add two tensors element-by-element. In TTNN you'd
call `ttnn.add(a, b)`. In tt-lang, you express the same operation with explicit
control over when data moves and where it lives.

```python
import numpy as np
import ttl
import ttnn

TILE_SIZE = 32

@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE
    col_tiles = a_in.shape[1] // TILE_SIZE

    # Typed ring buffers — one slot per tile, depth 2 (double-buffer)
    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(1, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk, out_dfb.reserve() as o_blk:
                    o_blk.store(a_blk + b_blk)  # element-wise add in L1

    @ttl.datamovement()
    def read():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                    ttl.copy(a_in[row:row+1, col:col+1], a_blk).wait()
                    ttl.copy(b_in[row:row+1, col:col+1], b_blk).wait()

    @ttl.datamovement()
    def write():
        for row in range(row_tiles):
            for col in range(col_tiles):
                with out_dfb.wait() as o_blk:
                    ttl.copy(o_blk, out[row:row+1, col:col+1]).wait()
```

Notice the roles:

- **`read()`** uses `reserve()` on the input DFBs — it's the *producer*, filling
  slots from DRAM and handing them to compute.
- **`compute()`** uses `wait()` on the input DFBs — it's the *consumer*, blocking
  until a filled slot is ready. It uses `reserve()` on the output DFB, filling it
  for the write thread.
- **`write()`** uses `wait()` on the output DFB — it's the *consumer* of compute's
  results, draining them to DRAM.

Every tile makes **one DRAM read (DM0) and one DRAM write (DM1)**. The `+`
operation happens entirely in L1.

### Fused operations: three inputs, one DMA trip out

> **Try it now:** select "Fused Multiply-Add" in the playground above.

Here's where fusion pays off. Computing `y = a * b + c` naively requires three
separate TTNN ops — three DRAM round-trips. In tt-lang, you wire three input
DFBs and fuse all the math in a single L1 pass:

```python
@ttl.operation(grid=(1, 1))
def fused_mma(a, b, c, y):
    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1,1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1,1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (a_dfb.wait() as ab, b_dfb.wait() as bb,
                      c_dfb.wait() as cb, y_dfb.reserve() as yb):
                    yb.store(ab * bb + cb)   # fused: three reads, one write, all in L1

    @ttl.datamovement()
    def read():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.reserve() as ab, b_dfb.reserve() as bb, c_dfb.reserve() as cb:
                    ttl.copy(a[r, c], ab).wait()
                    ttl.copy(b[r, c], bb).wait()
                    ttl.copy(c[r, c], cb).wait()

    @ttl.datamovement()
    def write():
        for r in range(rows):
            for c in range(cols):
                with y_dfb.wait() as yb:
                    ttl.copy(yb, y[r, c]).wait()
```

Three inputs stream in. One output streams out. The `ab * bb + cb` expression
runs entirely in L1. This is the pattern that gives you the 5–6× speedups from
the benchmark table above — instead of five ops each touching DRAM, you have one
pass through L1.

### Matrix multiply: the K-reduction accumulator

> **Try it now:** select "Matmul + Bias + ReLU" in the playground above.

Matrix multiply is the canonical heavy workload. The inner product over K
requires accumulating partial tile products, and where those partials live
matters enormously. Naive DRAM accumulation writes each partial product out and
reads it back — prohibitively slow. tt-lang keeps the running sum in L1 using
a DFB **ping-pong** pattern:

```python
@ttl.operation(grid=(1, 1))
def matmul_relu(a, b, bias, y):
    M, K, N = a.shape[0]//TILE_SIZE, a.shape[1]//TILE_SIZE, b.shape[1]//TILE_SIZE
    a_dfb    = ttl.make_dataflow_buffer_like(a,    shape=(1,1), block_count=2)
    b_dfb    = ttl.make_dataflow_buffer_like(b,    shape=(1,1), block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1,1), block_count=2)
    acc_dfb  = ttl.make_dataflow_buffer_like(y,    shape=(1,1), block_count=2)  # ping-pong
    y_dfb    = ttl.make_dataflow_buffer_like(y,    shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(M):
            for _ in range(N):
                with acc_dfb.reserve() as acc:       # initialize accumulator to zero
                    acc.store(ttl.math.fill(acc, 0))

                for _ in range(K):
                    with (a_dfb.wait() as ab, b_dfb.wait() as bb,
                          acc_dfb.wait() as prev):    # reads current partial sum
                        with acc_dfb.reserve() as acc: # writes updated partial sum
                            acc.store(prev + ab @ bb)

                with bias_dfb.wait() as bib, acc_dfb.wait() as acc:
                    with y_dfb.reserve() as yb:
                        yb.store(ttl.math.relu(acc + bib))  # fused bias + ReLU
    ...
```

The two `acc_dfb` slots alternate roles each k-step. The updated partial sum
is pushed into slot 1 before slot 0 is released — there is always one valid
partial sum in L1. No DRAM writes occur until the K-loop finishes. The final
step fuses the bias addition and ReLU into the same tile write to DRAM.

This ping-pong pattern is the foundation of every matmul in tt-lang. It scales
directly to multi-node grids by wrapping the outer loops with `ttl.node()` work
partitioning.

---

## Getting tt-lang

### Browser (you're already here)

The playground at the top of this page runs kernels in your browser using
ttlang-sim-lite — a lightweight Python interpreter built on Pyodide. No
install, no hardware. Use it to prototype ideas and explore the language before
setting up a local environment.

### Local — ttsim (no hardware required)

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

```bash
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so   # or libttsim_bh.so for Blackhole
export TT_METAL_SLOW_DISPATCH_MODE=1              # required — fast dispatch is in progress

cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

Check [ttsim releases](https://github.com/tenstorrent/ttsim/releases/latest) for newer versions.

### Local — build tt-lang

tt-lang ships two Docker images. The **dist** image has everything pre-built
and is the fastest path to running kernels:

```bash
docker run -d --name $USER-dist \
  ghcr.io/tenstorrent/tt-lang/tt-lang-dist-ubuntu-22-04:latest \
  sleep infinity
docker exec -it $USER-dist /bin/bash
# Environment activates automatically on login
python /opt/ttlang-toolchain/examples/elementwise-tutorial/step_4_multinode_grid_auto.py
```

To build from source (needed for modifying tt-lang itself), clone and build
against the **ird** (Internal Reference Dev) image:

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
cmake -G Ninja -B build -DTTLANG_SIM_ONLY=ON   # simulator only, no hardware needed
source build/env/activate                        # required every new shell session
python examples/eltwise_add.py
```

The [tt-lang docs](https://github.com/tenstorrent/tt-lang/blob/main/docs/sphinx/build.md)
cover all CMake options, including full hardware builds.

### Real hardware

If you have a Tenstorrent card, skip the `TT_METAL_SIMULATOR` and
`TT_METAL_SLOW_DISPATCH_MODE` variables. Everything else is identical. The
same kernel source runs bit-exact on simulation and silicon.

---

## Claude Code Slash Commands

The TT Developer Toolkit installs `/ttl-*` slash commands for Claude Code that
take you from an idea — or an existing kernel in another language — to a
validated, profiled tt-lang kernel in one session.

**Example workflow:** you have a PyTorch multi-head attention function and want
a fused Tensix kernel.

```bash
/ttl-import attention.py
```

Translates CUDA, Triton, PyTorch, or TTNN code to a tt-lang DFB pattern.
Handles the mechanical mapping: ops become compute thread logic, tensor loads
become DM0 reads, tensor stores become DM1 writes. Output is a runnable `.py`
file ready for simulation.

```bash
/ttl-simulate attention_ttl.py
```

Runs the kernel in the functional simulator. Catches DFB deadlocks, shape
mismatches, and thread synchronization bugs before you touch hardware. Iterate
here — simulation is fast and catches most correctness issues.

```bash
/ttl-test attention_ttl.py
```

Generates a correctness test suite comparing tt-lang output against a NumPy or
PyTorch reference. Covers edge cases: small matrices, non-tile-aligned shapes,
zero inputs, and the shapes your model actually runs at.

```bash
/ttl-profile attention_ttl.py
```

Returns per-line cycle counts. Shows which DFB `wait()` calls are blocking
compute and where the throughput bottleneck is — typically L1 buffer depth too
small, or the DM thread stalling on DRAM latency.

```bash
/ttl-optimize attention_ttl.py
```

Applies targeted optimizations based on profile output: increase DFB depth for
double-buffering, adjust tile sizes to fit L1, reorder loops to hide DMA
latency. Returns an improved kernel file.

```bash
/ttl-export attention_ttl.py
```

Compiles through LLVM → tt-mlir → tt-metal and produces production C++ for use
in a tt-metal project. Also emits intermediate MLIR at each compiler pass for
debugging.

```bash
/ttl-bug "description"
```

Files a structured bug report with a minimal reproducer when the compiler or
simulator behaves unexpectedly.

| Command | When to reach for it |
|---------|---------------------|
| `/ttl-import <file>` | You have an existing kernel in CUDA, Triton, PyTorch, or TTNN |
| `/ttl-simulate <file>` | After any change — validate before profiling or hardware |
| `/ttl-test <file>` | Simulation passes — build a regression suite |
| `/ttl-profile <file>` | Kernel is correct, want to find the bottleneck |
| `/ttl-optimize <file>` | Profile shows where to improve |
| `/ttl-export <file>` | Ready for production — generate C++ for tt-metal |
| `/ttl-bug <desc>` | Compiler or simulator behaves unexpectedly |

---

## What's Next

- **[tt-lang on GitHub](https://github.com/tenstorrent/tt-lang)** — source,
  examples, the full programming guide, and the language specification
- **[zoecarver/tt-lang-models](https://github.com/zoecarver/tt-lang-models)**
  — reference model implementations: DFlash, Engram, Oasis, nanochat, Gemma4,
  Qwen-Image, and more
- **[zoecarver/tt-lang-kernels](https://github.com/zoecarver/tt-lang-kernels)**
  — standalone kernel library, originally ported from LeetGPU challenges
- **[ttsim releases](https://github.com/tenstorrent/ttsim/releases/latest)**
  — latest simulator binaries for Wormhole and Blackhole
- **[tt-mlir](https://github.com/tenstorrent/tt-mlir)** — the MLIR-based
  compiler stack that tt-lang targets; useful when debugging compiler output
  or writing custom compiler passes
