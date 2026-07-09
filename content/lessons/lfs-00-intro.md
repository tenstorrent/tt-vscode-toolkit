---
id: lfs-00-intro
title: "Build an LLM from Scratch — Pick Your Altitude"
description: >-
  Start an LLM from scratch, TT-native from line one. Meet the 32×32 tile, the
  reader→compute→writer pipeline, and the "pick your altitude" ladder — grounded
  for CUDA programmers.
category: llm-from-scratch
tags:
  - llm
  - from-scratch
  - tt-lang
  - cuda
  - transformers
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
estimatedMinutes: 20
---

# Build an LLM from Scratch — Pick Your Altitude

## Inspiration

This arc was sparked by a post on **r/LocalLLaMA** and the repo behind it,
**[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM)** — an
~80M-parameter model built from scratch, in the open, at a size one person
can reason about end to end.

We follow its modern recipe: **RoPE** instead of learned positional
embeddings, **RMSNorm**, **SwiGLU**, **grouped-query attention (GQA)**, and a
**BPE** tokenizer. Every architectural choice here matches it, re-expressed
TT-native.

When you finish this arc, go build your own and share it back.

---

## What we're building

You'll build a small **Llama-style** language model **from scratch**,
expressed **TT-native from the first line of code**. Not a PyTorch model you
later port — a model whose forward pass, attention, and normalization are
written directly against `ttnn.Tensor`, with the hot inner loops hand-authored
in TT-Lang.

Five labs follow this one:

1. **[Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"])** — A BPE tokenizer from scratch, encode/decode,
   batching — the raw material every model needs.
2. **[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])** — Token embeddings and
   **RoPE** (rotary positional embeddings — no learned position table), plus
   your first TT-native kernel: elementwise add.
3. **[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])** — Multi-head self-attention extended to
   **grouped-query attention (GQA)**, written out fully — Q·Kᵀ, scale, mask,
   softmax, ·V, KV-head sharing — and re-expressed as a TT-Lang kernel.
4. **[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])** — **SwiGLU** MLP, RMSNorm, residuals, and
   the full transformer block stacked into a model you can actually run.
5. **[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])** — A from-scratch training loop —
   cross-entropy, AdamW, backprop — that we ran for real on Blackhole<sup>®</sup>
   hardware.

**The promise is two-tier, and both tiers are honest.**

Every lab builds and runs a **nano Llama** configuration live: 6 transformer
blocks, 6 attention heads sharing down to **3 KV groups (grouped-query
attention)**, 384-dim embeddings, RoPE with θ=500000, and a small
character-level vocabulary. That's `ttml`'s own `nanollama3` config
(`num_heads=6, num_groups=3, embedding_dim=384, num_blocks=6, theta=500000,
vocab≈96, max_sequence_length=256`, ~9.8M parameters).

It's small enough to train on a laptop CPU in the reference implementation,
and to run through `ttnn`/TT-Lang op-by-op without waiting around. It
converges — you'll watch the loss drop.

The **~80M-parameter** scale — the project that inspired this arc — is the
framed *target*. It's the same code and the same architecture with the config
knobs turned up: more layers, wider embeddings, a real tokenizer-scale
vocabulary. Each lab also gives you the params/DRAM math for exactly what
changes when you do that.

We don't run an 80M training job inside a lesson. That's a
multi-hour-to-multi-day job depending on hardware and data. But by the time you
finish **The Transformer Block & the Model** you'll know precisely how to get
there from the nano baseline.

---

## Set up your workspace

The reference code for this whole arc lives in
`~/tt-scratchpad/llm-from-scratch/` — the from-scratch tokenizer, the
`reference_gpt.py` model, and the hand-authored TT-Lang kernels. Every command
in the labs ahead runs out of that folder. Scaffold it once now and follow
along in your own copy:

[🧪 Create the LLM-from-Scratch Project](command:tenstorrent.llmFromScratch.createProject)

The button copies the arc's reference files into that folder and opens it. If
you land in a later lab first, the same button appears there — run it whenever
you're ready to start typing along.

---

## Pick your altitude

If you've written CUDA, the first question you ask about any new stack is
"where's my `model.cuda()`?" The honest answer on CUDA is that there are three
entry points, not one. You just rarely think about them as a stack, because
the tooling blurs the seams.

Tenstorrent's stack has the same three tiers, made explicit:

| You want to… | On CUDA you'd use… | On Tensix, write at… |
|---|---|---|
| Just run my PyTorch/JAX model | `model.cuda()` + `torch.compile` | **TT-Forge<sup>™</sup> / TT-XLA** — traces the graph, lowers it automatically |
| Call optimized library ops | cuBLAS / cuDNN / CUTLASS | **TT-NN<sup>™</sup>** — `ttnn.matmul`, `ttnn.softmax`, fused attention |
| Write a custom kernel, but in Python | a hand-rolled CUDA C kernel | **TT-Lang** — a Python DSL; explicit reader/compute/writer |
| Drop all the way to the metal | raw CUDA C + PTX tuning | **TT-Metalium<sup>™</sup>** — RISC-V kernels, hand-routed NoC moves |

This arc lives on the middle two rungs and deliberately never touches the
bottom one. **We build at TT-NN altitude and descend to TT-Lang for the hot
kernels** — attention, RMSNorm, the residual add.

That descent is the whole point of the arc, and it's a specific kind:
**inception, not conversion**. There's no existing CUDA or Triton kernel being
translated. You (or an agent, working from a plain-English
reader/compute/writer spec) write the TT-Lang kernel directly, as the original
expression of the idea — TT-native from line one, exactly as the title
promises.

---

## The 32×32 tile

Everything downstream of this lesson depends on one fact. **Tensix hardware
doesn't move or compute on scalars, rows, or arbitrary blocks. It moves and
computes on 32×32 tiles.** A tile is 1,024 elements, laid out so the matrix
engine and the data-movement engines can address it as a single unit.

This shows up directly in `ttnn`. When you convert a PyTorch tensor with
`ttnn.from_torch(...)`, you choose a **layout**:

- `ttnn.ROW_MAJOR_LAYOUT` — the tensor is stored the way PyTorch already
  stores it, row by row. Easy to reason about, but most compute ops can't
  consume it directly.
- `ttnn.TILE_LAYOUT` — the tensor is reshuffled into 32×32 tiles internally.
  This is the layout matmul, attention, and virtually every fused op expect.

You'll see `ttnn.Tensor` objects and `TILE_LAYOUT` conversions constantly,
starting in [Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"]).
Every embedding table, every attention score matrix, every
hidden-state activation in this arc is, underneath, a grid of 32×32 tiles
moving between DRAM and each Tensix core's local memory.

Pad your sequence length and embedding width to multiples of 32 and the
hardware is happy. Don't, and you pay a padding/reshuffle cost you can usually
see in the profile.

---

## reader → compute → writer: the Tensix execution model

On a GPU, a warp scheduler hides memory latency for you. When one warp stalls
on a global-memory read, the SM instantly swaps in another resident warp, and
you mostly never think about the gap. That trick is *why* CUDA code can look
like straight-line math, with the data movement implicit.

**Tensix has no warp scheduler, and there's no implicit version of this.**
Instead, every Tensix core runs an explicit three-part pipeline. The three
parts run as **concurrent threads**:

1. **A reader thread** streams input tiles from DRAM into the core's local L1
   memory.
2. **A compute thread** does the actual math on tiles that have arrived in L1.
3. **A writer thread** streams finished tiles from L1 back out to DRAM.

They overlap **by design**, not by luck. While compute works on tile *N*, the
reader is already fetching tile *N+1* and the writer is already draining tile
*N-1*.

At the TT-NN level, this pipeline is built into every op for you — you never
see it. But it's why **TT-Lang exists**. TT-Lang lets you write the pipeline
out explicitly: one `@ttl.compute` function and two `@ttl.datamovement`
functions inside a single `@ttl.operation`, coordinated through typed L1 ring
buffers (Dataflow Buffers). Two primitives drive it — `reserve()` claims a
slot to fill, `wait()` blocks until a slot is ready to consume.

That's the shape of every hand-authored kernel in this arc, starting with the
elementwise-add kernel in [Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"]).
Hold onto this mental picture — reader,
compute, writer, three threads, explicit handoff — because from here on,
"writing a TT-Lang kernel" just means writing those three functions.

---

## Coming from CUDA: the concept map

A quick-reference table for translating CUDA vocabulary into Tensix
vocabulary. You'll see pieces of this called out again, scoped to whatever
each lab is building.

| CUDA concept | Tensix / TT-NN equivalent |
|---|---|
| Streaming Multiprocessor (SM) | Tensix core |
| Thread block | Tile computation on one Tensix core |
| Shared memory | L1 SRAM, local to each Tensix core |
| `cudaMemcpy` host→device | `ttnn.from_torch(tensor, device=device)` |
| `cudaMemcpy` device→host | `ttnn.to_torch(tt_tensor)` |
| Kernel launch `<<<grid, block>>>` | Automatic TT-NN op dispatch — the op decides how work is spread across the Tensix grid |
| Warp scheduler hiding memory latency | Explicit reader → compute → writer pipeline (see above) — the overlap is designed in, not scheduled around |

**On the L1/shared-memory row, the precise number matters, so here it is
verified rather than rounded.** Each Tensix core's L1 SRAM is **1,464 KB**, on
both Wormhole<sup>™</sup> and Blackhole. That's per the TT-Lang specification's
platform-limits table (`TTLangSpecification.md`, Appendix E).

You'll sometimes see this rounded to "1.5 MB" elsewhere — close enough for
intuition. But 1,464 KB is what the compiler and simulator actually enforce,
so it's the one to use for tile-budget math.

The same appendix gives the maximum single-chip Tensix grid size (unharvested):
**13×10 on Blackhole** (130 cores) and **8×9 on Wormhole** (72 cores). Real
chips can ship with fewer cores enabled than that maximum. Harvesting and SKU
differences mean other Tenstorrent materials quote enabled-core counts like
120 or 140 for a full Blackhole chip, depending on the product and firmware.

So treat the 13×10 figure as the *architectural* ceiling from the TT-Lang
spec, not a product's enabled-core count. It's "how big the grid can be
addressed," not "how many cores your card has."

---

## The honest runtime matrix

Not every lab runs the same way, and pretending otherwise would set the
wrong expectations. Here's exactly what runs where:

| Lab | What you build | Runs on |
|---|---|---|
| **[Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"])** | BPE tokenizer, batching | Pure Python, anywhere — no device, no simulator |
| **[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])** | PyTorch **RoPE** reference + first TT-Lang inception kernel (elementwise add) | Functional simulator + the browser-based TT-Lang playground |
| **[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])** | PyTorch reference (**GQA** — KV-head sharing) + TT-Lang attention/softmax kernel | Kernel is **authored, faithful to the vendor kernel, and torch-verified via oracle** (correlation gate wired, pending first execution) — not sim-run yet, the functional simulator can't execute attention's softmax reduction |
| **[The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])** | PyTorch reference (**SwiGLU**) + TT-Lang RMSNorm/matmul kernels, wired in as drop-in `ttnn.Tensor` ops | Matmul kernel is sim-validated; RMSNorm hits the same sim gap as attention — authored and torch-verified |
| **[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])** | A real from-scratch training loop for the nano Llama model (RoPE, RMSNorm, SwiGLU, GQA) — cross-entropy, AdamW, backprop | **Real hardware.** `ttml` built from source, `train_nanogpt.py` (llama path) run on a Blackhole p300c — loss dropped monotonically from **4.69 to 3.23** over 20 steps, ~65 ms/step, 16.5 TFLOPS, on-device, exit 0 |

The gap for **Attention from Scratch** and **The Transformer Block & the Model**
has one honest cause: **the functional simulator runs
ahead of the hardware compiler.** Some kernel patterns — attention's softmax
reduction, RMSNorm's normalization — are expressible and torch-verified in
TT-Lang today. But the simulator build this arc is pinned to doesn't yet
accept the broadcast pattern they need.

That's a simulator limitation, not a correctness problem with the kernel. Each
lab says so plainly rather than quietly running something narrower than
advertised.

**Train It & Run for Real** deserves one more honest note. **Upstream tt-metal does not run
continuous-integration training tests for Blackhole.** The softmax,
cross-entropy, RMSNorm, and scaled-dot-product-attention training-op tests are
skipped on P100/P150 in CI. So there's no upstream guarantee that from-scratch
training works on Blackhole.

We built `ttml` from source against `~/tt-metal` **v0.73** and ran the training
loop on a p300c. It worked. If you want the same result, **pin your tt-metal
version to v0.73** and follow the build recipe in **Train It & Run for Real**. A different version
may hit different edges of a fast-moving stack.

---

## Next

You've got the map: the altitude ladder, the tile, the reader→compute→writer
pipeline, and an honest picture of what runs where. Time to build the first
real piece — the tokenizer.

[→ Continue to Lab 1: Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"])
