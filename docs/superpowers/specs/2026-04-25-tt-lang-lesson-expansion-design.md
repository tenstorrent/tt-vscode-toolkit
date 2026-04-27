# TT-Lang Lesson Expansion — Design Spec

**Goal:** Rewrite `content/lessons/tt-lang-intro.md` into a single comprehensive lesson that hooks readers with the browser playground immediately, motivates tt-lang with real project numbers, teaches installation (ttsim + hardware), explains the kernel programming model, and walks through the full Claude Code slash command workflow from idea to validated kernel.

**Architecture:** One expanded markdown lesson replacing the existing `tt-lang-intro.md`. No new files, no registry changes, no new commands. The lesson uses the existing `playground: ttlang-sim` front matter so the browser sandbox renders automatically.

**Target reader:** A developer who has heard of tt-lang and wants to know why it exists, whether it's worth their time, and how to actually start using it — with or without Tenstorrent hardware.

---

## Section 1 — Browser playground (opening)

The playground panel renders before any prose (controlled by VSCode walkthrough step order and the `playground: ttlang-sim` field). The opening paragraph is a single sentence acknowledging that the reader is already running code. A table follows listing all four embedded kernels with one-line descriptions of what each teaches:

| Kernel | What it teaches |
|--------|----------------|
| Element-wise Add | Minimal DFB producer/consumer loop — the hello-world of tt-lang |
| Fused Multiply-Add | Three DFBs, zero intermediate DRAM writes — what fusion looks like |
| Matmul + Bias + ReLU | K-reduction accumulator ping-pong; core matmul pattern on Tensix |
| Row-partitioned Matmul | `ttl.node()` work partitioning across a multi-core grid |

Transition sentence leads into the "why" section: "To understand what just ran, read on."

No installation steps here — the playground is the install.

---

## Section 2 — The DRAM wall (why tt-lang exists)

Three short paragraphs plus one table.

**Para 1:** TTNN dispatches each op as a separate kernel. Data written to DRAM after every op. For a transformer forward pass, that's roughly one DRAM round-trip per layer component (norm → proj → attn → proj → FFN = 5+ DRAM writes per layer).

**Para 2:** TT-Lang breaks that pattern. Tile data streams in from DRAM once, flows through L1 using Dataflow Buffers (DFBs), compute happens entirely in registers, and results drain to DRAM once. The whole fused block is one kernel dispatch.

**Para 3:** Real measured improvements:

| Project | What was fused | Improvement |
|---------|---------------|-------------|
| SkyReels-1.3B transformer block | 5 ops → 1 kernel | 3–5× vs TTNN |
| DFlash speculative decoder (draft forward) | RoPE, RMSNorm, SiLU, residuals | 5–6× decode speedup |
| DeepSeek Engram module | gating + depthwise conv | 2.2× all kernels; 3.4× gating alone |
| nanochat fused MLP projection | 7 dispatches → 1 | +21% tok/s (13.13 → 15.89) |
| Qwen-Image generation kernels | attention + norms | 4–8× vs XLA at 512×512 |

The table includes links to the relevant GitHub repos.

---

## Section 3 — What people have built

Five punchy paragraphs, each 2–3 sentences, each with a concrete result and a GitHub link. No filler. Chronological roughly by ambition level.

1. **SkyReels-1.3B** (`zoecarver/tt-lang-models`): Full WAN transformer block kernel on QB2. Fused to one DRAM read + one write per layer. 3–5× throughput improvement over op-by-op TTNN dispatch at production batch dimensions.

2. **WAN Animate 14B** (`~/code/wan-animate-ttlang`): 40-layer, 5120-hidden DiT brought up on a 4-chip QB2 (2×2 mesh). TT-Lang kernels cover 3D RoPE, AdaLN modulation, and softcap. The bring-up involved debugging seven integration bugs across the pipeline, traced in a single session log.

3. **Freeciv game AI** (`~/tt-lang-freeciv`): tt-lang kernels for Freeciv's map generation (Perlin noise terrain) and pathfinding, running in the functional simulator. A game AI that started as a "what if" conversation and ended with Tensix doing the world-building.

4. **DFlash speculative decoder** (`zoecarver/dflash`): Draft model proposes 16 tokens in parallel, verified by Qwen3-30B. Draft forward pass at 93ms with caching (vs 887ms without). 5–6× decode speedup end-to-end. Acceptance rate matches the PyTorch reference.

5. **Oasis real-time Minecraft** (`zoecarver/open-oasis`): 500M diffusion transformer generating Minecraft frames on a single Blackhole card. Runs at 8 FPS in a single captured trace with 4-way tensor parallelism. Interactive browser play across 26 Atari games.

---

## Section 4 — Getting tt-lang

Three subsections with explicit commands for each path.

### Browser (already done)
One sentence: the playground above is the browser path, no install required.

### Local — ttsim (no hardware needed)

```bash
# Prerequisites: tt-metal built, TT_METAL_HOME set
mkdir -p ~/sim && cd ~/sim

# Wormhole simulator
wget https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_wh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# OR Blackhole simulator
wget https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_bh.so
cp $TT_METAL_HOME/tt_metal/soc_descriptors/blackhole_140_arch.yaml ~/sim/soc_descriptor.yaml
```

Set simulator env var and run:

```bash
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so   # or libttsim_bh.so
export TT_METAL_SLOW_DISPATCH_MODE=1              # required: fast dispatch not yet supported

cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

Note the known limitation: `TT_METAL_SLOW_DISPATCH_MODE=1` is required; fast dispatch is in progress.

### Local — build tt-lang

```bash
git clone https://github.com/tenstorrent/tt-lang.git
cd tt-lang
# Follow docs/sphinx/build.md for cmake options
source build/env/activate
python examples/eltwise_add.py
```

Explicit note: the `source build/env/activate` step is required before running any tt-lang commands. Link to `docs/sphinx/build.md`.

### Real hardware

For users with a Tenstorrent device: skip `TT_METAL_SIMULATOR` and `TT_METAL_SLOW_DISPATCH_MODE`. Everything else is identical.

---

## Section 5 — The Tensix thread model

Keep the existing diagram and table from the current lesson, tightened:

```
DRAM ──[DM0 reads]──► DFB_in ──[Compute]──► DFB_out ──[DM1 writes]──► DRAM
```

| Thread | Role |
|--------|------|
| Compute | Math — matrix ops, activations, reductions |
| Data Movement 0 | Load input tiles from DRAM → local L1 buffer |
| Data Movement 1 | Store output tiles from L1 buffer → DRAM |

Two additional paragraphs: what a DFB is (typed ring buffer, zero-copy, automatic backpressure), and the `wait()` vs `reserve()` distinction (consumer blocks until filled; producer blocks until empty). This is the minimum needed to read kernel code.

---

## Section 6 — Kernel deep-dives

Keep the three existing kernel examples verbatim with their explanatory text. Each kernel gets a callout: "You can run this kernel right now in the playground above." The matmul_relu section keeps its ping-pong DFB diagram — it's the most conceptually dense and benefits from the current explanation.

No new kernels added here. The existing three (eltwise_add, fused_mma, matmul_relu) cover the full conceptual surface: single-input DFB, multi-input fusion, and accumulator ping-pong.

---

## Section 7 — The Claude Code slash commands

This section documents the `/ttl-*` skills installed in the TT Developer Toolkit. Available when working in Claude Code with tt-lang.

### The workflow

Walk through a complete example: starting from a PyTorch multi-head attention function and ending with a validated, profiled TT-Lang kernel.

```
/ttl-import attention.py
```
Translates CUDA, Triton, PyTorch, or TTNN code to tt-lang DFB pattern. Handles the mechanical mapping: ops → compute thread, tensor loads → DM0, tensor stores → DM1. Output is a runnable `.py` file.

```
/ttl-simulate attention_ttl.py
```
Runs the kernel in the functional simulator (ttlang-sim). Catches DFB deadlocks, shape mismatches, and incorrect synchronization before touching hardware. Iterate here — it's fast.

```
/ttl-test attention_ttl.py
```
Generates a correctness test suite comparing tt-lang output against a NumPy or PyTorch reference. Covers edge cases: small matrices, non-tile-aligned shapes, zero inputs.

```
/ttl-profile attention_ttl.py
```
Runs the profiler and returns per-line cycle counts. Identifies which DFB waits are blocking compute and where the bottleneck is (typically: L1 buffer depth too small, or DM thread stalling on DRAM latency).

```
/ttl-optimize attention_ttl.py
```
Applies optimizations based on profile output: increase DFB depth for double-buffering, adjust tile sizes, suggest loop reordering to hide DMA latency. Returns an optimized kernel file.

```
/ttl-export attention_ttl.py
```
Compiles the kernel through LLVM → tt-mlir → tt-metal and produces production C++ code. Also outputs the MLIR at each pass stage for debugging.

```
/ttl-bug "description"
```
Files a structured bug report with a minimal reproducer when the compiler or simulator behaves unexpectedly.

### Reference table

| Command | When to use |
|---------|-------------|
| `/ttl-import <file>` | You have an existing kernel in another language |
| `/ttl-simulate <file>` | After any change — validate before profiling |
| `/ttl-test <file>` | After simulation passes — build a regression suite |
| `/ttl-profile <file>` | Kernel is correct, want to find the bottleneck |
| `/ttl-optimize <file>` | Profile shows where to improve |
| `/ttl-export <file>` | Ready for production — generate C++ |
| `/ttl-bug <desc>` | Compiler/simulator behaves unexpectedly |

---

## Section 8 — What's next

Links section:

- [tt-lang on GitHub](https://github.com/tenstorrent/tt-lang) — source, examples, docs
- [zoecarver/tt-lang-models](https://github.com/zoecarver/tt-lang-models) — reference models (DFlash, Engram, Oasis, nanochat, Gemma4)
- [zoecarver/tt-lang-kernels](https://github.com/zoecarver/tt-lang-kernels) — standalone kernel library
- [ttsim releases](https://github.com/tenstorrent/ttsim/releases/latest) — hardware-free simulator binaries
- [tt-mlir](https://github.com/tenstorrent/tt-mlir) — the compiler stack under tt-lang

---

## Implementation Notes

- **File:** `content/lessons/tt-lang-intro.md` — full replacement, not a patch
- **Front matter:** unchanged (id, title, hardware support list, `playground: ttlang-sim`)
- **Status:** keep as `draft` until hardware-validated; ttsim path is fully testable
- **Length:** target ~400–500 lines of markdown — long but structured with clear `##` anchors
- **No new commands, no registry changes, no package.json changes**
- **Version bump:** PATCH after implementation
