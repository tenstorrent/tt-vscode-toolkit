# TT-Lang Lesson Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `content/lessons/tt-lang-intro.md` into a comprehensive lesson that hooks with the browser playground, motivates tt-lang with real project numbers, covers ttsim setup, documents the Claude Code `/ttl-*` slash commands, and showcases real projects (SkyReels, WAN Animate, Freeciv, zoecarver's work).

**Architecture:** Single file replacement — `content/lessons/tt-lang-intro.md` is fully rewritten. Front matter is unchanged (same id, title, hardware support, `playground: ttlang-sim`). No registry changes, no new commands, no package.json changes.

**Tech Stack:** Markdown (VSCode walkthrough rendering), `npm test` for validation, `npm run validate:lessons` for registry sync check.

---

## File Structure

**Modified:**
- `content/lessons/tt-lang-intro.md` — full rewrite, same front matter

**Unchanged:**
- `content/lesson-registry.json` — no changes needed
- `package.json` — version bumped only in final task
- All other files

---

## Context for implementer

This lesson lives in a VSCode extension. Markdown is rendered natively by VSCode in a walkthrough panel. The `playground: ttlang-sim` front matter field causes the browser-based kernel sandbox to render alongside the lesson.

**Markdown rendering rules in this codebase:**
- Checklist items (`✅ text`) MUST have `- ` prefix: `- ✅ text` — otherwise consecutive items collapse to one line
- Code blocks MUST have matched opening/closing fences (` ``` `)
- No raw HTML unless the lesson already uses it (this one doesn't)
- External links use standard markdown `[text](url)` syntax

**Existing lesson validation:**
- `npm run validate:lessons` — checks front matter fields match `content/lesson-registry.json`
- `npm test` — runs link validator + markdown fence matching; must pass before commit

**Front matter to preserve verbatim (lines 1–25 of current file):**
```yaml
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
```

**ttsim release info (verified at plan-writing time, v1.5.4):**
- Wormhole: `https://github.com/tenstorrent/ttsim/releases/download/v1.5.4/libttsim_wh.so`
- Blackhole: `https://github.com/tenstorrent/ttsim/releases/download/v1.5.4/libttsim_bh.so`
- Use `https://github.com/tenstorrent/ttsim/releases/latest` as the releases page link so it stays current.

**Key external URLs:**
- tt-lang: `https://github.com/tenstorrent/tt-lang`
- ttsim releases: `https://github.com/tenstorrent/ttsim/releases/latest`
- tt-mlir: `https://github.com/tenstorrent/tt-mlir`
- zoecarver/tt-lang-models: `https://github.com/zoecarver/tt-lang-models`
- zoecarver/tt-lang-kernels: `https://github.com/zoecarver/tt-lang-kernels`
- zoecarver/dflash: `https://github.com/zoecarver/dflash`
- zoecarver/open-oasis: `https://github.com/zoecarver/open-oasis`
- zoecarver/Engram: `https://github.com/zoecarver/Engram`
- zoecarver/nanochat (commit): `https://github.com/zoecarver/nanochat/commit/f849d3f`

---

## Task 1: Scaffold the new file with front matter and section headings

**Files:**
- Modify: `content/lessons/tt-lang-intro.md` (full replacement)

- [ ] **Step 1: Write the scaffolded file**

Replace the entire file contents with the front matter (verbatim from above) followed by section heading stubs. This gives us a valid file to build on incrementally.

```markdown
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

## The DRAM Wall

## What People Have Built

## Getting tt-lang

## The Tensix Thread Model

## Kernel Patterns

## Claude Code Slash Commands

## What's Next
```

- [ ] **Step 2: Run validation to confirm scaffold is clean**

```bash
cd /home/ttuser/code/tt-vscode-toolkit
npm run validate:lessons 2>&1 | tail -5
```

Expected: no errors (front matter unchanged, registry stays in sync).

- [ ] **Step 3: Commit the scaffold**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "chore: scaffold tt-lang lesson rewrite sections"
```

---

## Task 2: Write the opening and playground section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Replace the `# Introduction to tt-lang` heading and add opening content**

Replace everything between `# Introduction to tt-lang` and `## The DRAM Wall` with:

```markdown
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

```

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing, no failures mentioning `tt-lang-intro.md`.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add opening playground section"
```

---

## Task 3: Write the DRAM wall section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## The DRAM Wall` section**

Replace the `## The DRAM Wall` stub with:

```markdown
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

```

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add DRAM wall motivation section with benchmark table"
```

---

## Task 4: Write the "What people have built" section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## What People Have Built` section**

Replace the stub with:

```markdown
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

```

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add real project showcase section"
```

---

## Task 5: Write the Getting tt-lang section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## Getting tt-lang` section**

Replace the stub with:

```markdown
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

```

- [ ] **Step 2: Verify the command link in the lesson is valid**

The `command:tt-vscode-toolkit.openLesson?%22build-tt-metal%22` link references the build-tt-metal lesson. Confirm it exists:

```bash
grep -r "build-tt-metal" /home/ttuser/code/tt-vscode-toolkit/content/lesson-registry.json | head -3
```

Expected: at least one match showing the lesson is registered.

- [ ] **Step 3: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing.

- [ ] **Step 4: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add ttsim setup and build-from-source instructions"
```

---

## Task 6: Write the Tensix thread model section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## The Tensix Thread Model` section**

Replace the stub with the following (the core diagram and table are from the original lesson, with two new explanation paragraphs added):

```markdown
## The Tensix Thread Model

Every Tensix core runs three threads simultaneously:

| Thread | Role |
|--------|------|
| **Compute** | Math — matrix ops, activations, reductions |
| **Data Movement 0** | Load input tiles from DRAM → local L1 buffer |
| **Data Movement 1** | Store output tiles from L1 buffer → DRAM |

The threads communicate through a **Dataflow Buffer (DFB)** — a typed ring
buffer that provides zero-copy handoffs and automatic backpressure between
threads:

```
DRAM ──[DM0 reads]──► DFB_in ──[Compute]──► DFB_out ──[DM1 writes]──► DRAM
```

A DFB has a fixed number of slots (`block_count`). The producer thread calls
`reserve()` to claim an empty slot and fill it. The consumer thread calls
`wait()` to block until a filled slot is available. When the consumer is done,
the slot is released back to the producer. This gives you double-buffering and
pipeline overlap with no manual synchronization.

**`wait()` vs `reserve()`** — the two roles:
- `wait()` — consumer: blocks until a filled slot is ready to read
- `reserve()` — producer: blocks until an empty slot is available to write

The DFB scheduler handles all synchronization. You express the data dependency;
the runtime manages the pipeline.

```

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add Tensix thread model section with DFB explanation"
```

---

## Task 7: Write the kernel patterns section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## Kernel Patterns` section**

Replace the stub with the three kernels from the original lesson, each with a playground callout added before the code block. The code is identical to the original — copy it exactly.

The section must contain these three subsections in order, each with the playground callout line immediately before the code block:

**Subsection: Element-Wise Addition**
- Callout: `> **Try it now:** select "Element-wise Add" in the playground above.`
- Then the `eltwise_add` kernel code block (copy verbatim from original lesson lines 55–93)
- Then the "Key concepts" explanation (copy verbatim from original lines 96–107)

**Subsection: Fused Operations: Three Inputs, One DMA Trip Out**
- Callout: `> **Try it now:** select "Fused Multiply-Add" in the playground above.`
- Then the `fused_mma` kernel code block (copy verbatim from original lines 117–150)
- Then the explanation paragraph (copy verbatim from original lines 151–153)

**Subsection: Matrix Multiply: The K-Reduction Accumulator**
- Callout: `> **Try it now:** select "Matmul + Bias + ReLU" in the playground above.`
- Then the motivation paragraphs (copy verbatim from original lines 155–165)
- Then the `matmul_relu` kernel code block (copy verbatim from original lines 168–196)
- Then the explanation paragraphs (copy verbatim from original lines 198–205)

The full `## Kernel Patterns` section content, written out completely:

````markdown
## Kernel Patterns

### Element-Wise Addition

> **Try it now:** select "Element-wise Add" in the playground above.

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
                    o_blk.store(a_blk + b_blk)  # element-wise add

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

**`@ttl.compute()`** — defines the math thread. It _waits_ on filled input buffers and
_reserves_ output buffer slots. The `with` statements are blocking: compute pauses until
the data movement thread fills the slot.

**`@ttl.datamovement()`** — defines a DMA thread. It _reserves_ input buffer slots,
fills them from DRAM, then releases them to compute. The second DM thread
drains the output buffer.

**`a_dfb.wait()` vs `a_dfb.reserve()`** — `wait()` blocks until a filled slot is
available (consumer role); `reserve()` blocks until an empty slot is available (producer
role). The DFB scheduler handles all synchronization.

### Fused Operations: Three Inputs, One DMA Trip Out

> **Try it now:** select "Fused Multiply-Add" in the playground above.

The `eltwise_add` kernel shows the basic pattern but only uses two inputs. A more
representative real-world case is a **fused multiply-add**: `y = a * b + c`.

With a naive implementation you'd write three separate kernels, each making a DRAM round-trip.
With tt-lang you wire three DFBs and fuse everything in a single L1 pass:

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
                    yb.store(ab * bb + cb)   # fused in L1 — one DMA trip out

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

The compute thread issues one `store` that computes `a*b+c` entirely in L1 registers.
Only the final result travels to DRAM — three reads in, one write out per tile.

### Matrix Multiply: The K-Reduction Accumulator

> **Try it now:** select "Matmul + Bias + ReLU" in the playground above.

Matrix multiply is the canonical heavy workload. The inner product loop over K requires
accumulating partial tile products, and where those partials live matters enormously:

- **DRAM accumulation** — write each partial product out and read it back. Prohibitively
  slow; L1 bandwidth to DRAM is the bottleneck.
- **L1 accumulator via DFB ping-pong** — keep the running sum in L1 by making `acc_dfb`
  both a producer and consumer within the compute thread. This is what tt-lang does.

The pattern looks like this:

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
                    # consume previous partial sum, add a@b tile, push updated sum
                    with (a_dfb.wait() as ab, b_dfb.wait() as bb,
                          acc_dfb.wait() as prev):    # ← reads slot 0
                        with acc_dfb.reserve() as acc: # ← writes slot 1
                            acc.store(prev + ab @ bb)
                        # push() (slot 1 visible) happens before pop() (slot 0 freed)
                        # next iteration: reads slot 1, writes slot 0 — true ping-pong

                with bias_dfb.wait() as bib, acc_dfb.wait() as acc:
                    with y_dfb.reserve() as yb:
                        yb.store(ttl.math.relu(acc + bib))  # fused bias + ReLU
    ...
```

The two `acc_dfb` slots alternate roles each k-step. `push()` on the new slot runs
before `pop()` on the old one, so there is always one valid partial sum in L1. No
DRAM writes occur until the k-loop finishes. The final step fuses the bias addition
and ReLU activation into the same tile write.

This is the foundational matmul pattern in tt-lang — the same structure scales to
multi-node grids simply by adding `ttl.node()` work partitioning around it.
````

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing, no fence mismatches.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add kernel patterns section with playground callouts"
```

---

## Task 8: Write the Claude Code slash commands section

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## Claude Code Slash Commands` section**

Replace the stub with:

```markdown
## Claude Code Slash Commands

The TT Developer Toolkit installs a set of `/ttl-*` slash commands for Claude
Code that take you from an idea — or an existing kernel in another language —
to a validated, profiled tt-lang kernel in one session.

**Example workflow:** you have a PyTorch multi-head attention function and want
a Tensix kernel for it.

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

### Command reference

| Command | When to reach for it |
|---------|---------------------|
| `/ttl-import <file>` | You have an existing kernel in CUDA, Triton, PyTorch, or TTNN |
| `/ttl-simulate <file>` | After any change — validate before profiling or hardware |
| `/ttl-test <file>` | Simulation passes — build a regression suite |
| `/ttl-profile <file>` | Kernel is correct, want to find the bottleneck |
| `/ttl-optimize <file>` | Profile shows where to improve |
| `/ttl-export <file>` | Ready for production — generate C++ for tt-metal |
| `/ttl-bug <desc>` | Compiler or simulator behaves unexpectedly |

```

- [ ] **Step 2: Run tests**

```bash
npm test 2>&1 | grep -E "passing|failing|tt-lang"
```

Expected: all passing.

- [ ] **Step 3: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add Claude Code slash commands workflow section"
```

---

## Task 9: Write the What's Next section and finalize

**Files:**
- Modify: `content/lessons/tt-lang-intro.md`

- [ ] **Step 1: Fill in the `## What's Next` section**

Replace the stub with:

```markdown
## What's Next

- **[tt-lang on GitHub](https://github.com/tenstorrent/tt-lang)** — source
  code, examples directory, build docs, and the full programming guide
- **[zoecarver/tt-lang-models](https://github.com/zoecarver/tt-lang-models)**
  — reference model implementations: DFlash, Engram, Oasis, nanochat, Gemma4,
  Qwen-Image, LingBot-World, and more
- **[zoecarver/tt-lang-kernels](https://github.com/zoecarver/tt-lang-kernels)**
  — standalone kernel library, originally imported from LeetGPU challenges
- **[ttsim releases](https://github.com/tenstorrent/ttsim/releases/latest)**
  — latest simulator binaries for Wormhole and Blackhole
- **[tt-mlir](https://github.com/tenstorrent/tt-mlir)** — the MLIR-based
  compiler stack that tt-lang targets; useful when debugging compiler output
  or writing custom compiler passes

```

- [ ] **Step 2: Run full test suite and lesson validation**

```bash
cd /home/ttuser/code/tt-vscode-toolkit
npm run validate:lessons
npm test
```

Expected: `validate:lessons` exits 0, all test suites pass.

- [ ] **Step 3: Confirm file length is reasonable**

```bash
wc -l content/lessons/tt-lang-intro.md
```

Expected: roughly 350–500 lines. If over 600, review for accidental duplication.

- [ ] **Step 4: Commit**

```bash
git add content/lessons/tt-lang-intro.md
git commit -m "docs(tt-lang): add What's Next section, lesson complete"
```

---

## Task 10: Version bump, rebuild, and package

**Files:**
- Modify: `package.json` (version field only)
- Modify: `CHANGELOG.md` (add new entry)

- [ ] **Step 1: Bump version in package.json**

Read current version:
```bash
grep '"version"' /home/ttuser/code/tt-vscode-toolkit/package.json | head -1
```

Increment the PATCH number (e.g. `0.0.410` → `0.0.411`). Edit only the `"version"` line.

- [ ] **Step 2: Add CHANGELOG entry**

Add a new entry at the top of the changelog (after the `---` separator following the header), using today's date `2026-04-25`:

```markdown
## [0.0.411] - 2026-04-25

### Changed

- **tt-lang lesson rewrite** — expanded `tt-lang-intro` from a technical reference into a full getting-started guide: browser playground opening, DRAM wall motivation with real benchmark numbers, project showcase (SkyReels, WAN Animate, Freeciv, DFlash, Oasis), ttsim setup instructions, Tensix thread model with DFB explanation, kernel deep-dives with playground callouts, and Claude Code `/ttl-*` slash commands workflow

---
```

- [ ] **Step 3: Build and package**

```bash
npm run package 2>&1 | tail -5
```

Expected: `✅ Package created: tt-vscode-toolkit-0.0.411-dev.vsix`

- [ ] **Step 4: Commit and push**

```bash
git add package.json package-lock.json CHANGELOG.md
git commit -m "docs(tt-lang): comprehensive lesson rewrite — ttsim, projects, slash commands (v0.0.411)"
git push origin HEAD
```
