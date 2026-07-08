---
id: lfs-02-embeddings
title: "Embeddings & the Residual Stream"
description: >-
  Build token and positional embeddings, meet the residual stream, and write
  your first TT-Lang inception kernel live in the browser playground.
category: llm-from-scratch
tags: [embeddings, positional, residual, tt-lang, playground]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 30
playground: ttlang-sim
---

# Embeddings & the Residual Stream

Lab 1 ended at a single explicit line: `ttnn.from_torch(x, device=device,
layout=ttnn.TILE_LAYOUT)`, a `[batch, seq]` tensor of raw token IDs, tiled and
sitting in device DRAM. IDs alone don't mean anything to a model yet — this
lab turns each ID into a vector, adds in *where* that token sits in the
sequence, and sums both into the **residual stream**: the running tensor
every block in the model reads from and writes back into. Then, for the
first time in this arc, you'll write that summation as a hand-authored
TT-Lang kernel — not port one, author it — and run it live, in your browser,
before you finish reading.

## Coming from CUDA: shared memory becomes L1, and there's no warp scheduler

CUDA gives every thread block a pool of fast `__shared__` memory, scoped to
the block and the one Streaming Multiprocessor it runs on. Tensix has a
direct analog: **L1 SRAM**, scoped to one Tensix core — 1,464 KB per core on
both Wormhole<sup>™</sup> and Blackhole<sup>®</sup>, the exact figure lfs-00
verified against the TT-Lang specification. Same idea, fast and local and
explicitly managed, just owned by a core instead of an SM.

The bigger difference is what happens *around* that memory. On CUDA, a warp
scheduler hides memory latency for you automatically: while one warp stalls
on a global-memory load, the SM silently swaps in another resident warp, and
your kernel body reads like straight-line math with the data movement
implicit. **Tensix has no warp scheduler, and TT-Lang has no implicit
version of that trick.** Every Tensix core runs three concurrent threads,
and you author them as three separate Python functions inside one
`@ttl.operation`:

| Thread | Role | TT-Lang decorator |
|---|---|---|
| Reader | Streams input tiles from DRAM into L1 | `@ttl.datamovement()` |
| Compute | Does the math on tiles already in L1 | `@ttl.compute()` |
| Writer | Streams finished tiles from L1 back to DRAM | `@ttl.datamovement()` |

lfs-00 introduced this as a *diagram* — reader → compute → writer, three
threads, explicit handoff through Dataflow Buffers (DFBs) with `reserve()`
and `wait()`. This lab is where that diagram stops being a diagram. The
kernel you're about to read is the first place in the arc where you write
those three functions out yourself.

## Token + positional embeddings, from scratch

A token ID by itself carries no information a matrix multiply can use — it's
just an index. The standard fix, unchanged since the original Transformer
paper, is a **lookup table**: one learned vector per token ID, plus a second
learned vector per sequence position, summed together. Here's the exact
code, quoted verbatim from `content/templates/llm-from-scratch/reference_gpt.py`'s
`NanoGPT` module so the prose and the code never drift:

```python
class NanoGPT(nn.Module):
    """The whole model: embeddings -> blocks -> norm -> LM head."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # residual stream
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss
```

Two embedding tables, both just `nn.Embedding` — a lookup, nothing more:

- **`tok_emb`** — shape `[vocab_size, n_embd]`. Row `i` is the learned vector
  for token ID `i`. At the nano config from lfs-00 (`vocab_size=96,
  n_embd=384`), that's a `[96, 384]` table.
- **`pos_emb`** — shape `[block_size, n_embd]`. Row `t` is the learned vector
  for "this token is at position `t`." At nano scale (`block_size=256`),
  that's `[256, 384]`.

`tok_emb(idx)` gathers one row per token ID in the batch, giving
`[B, T, n_embd]`. `pos_emb(pos)` gathers one row per position (`pos =
torch.arange(T)`, the same range for every sequence in the batch), giving
`[T, n_embd]`, which broadcasts across the batch dimension. **The line that
matters most in this whole file is the one already commented
`# residual stream`:**

```python
x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  # residual stream
```

That `+` is the entire subject of this lab.

**A tile-alignment note, since lfs-01 flagged it and this is where it shows
up:** `vocab_size=96` in `GPTConfig` isn't the organic number of characters
in the training corpus — it's padded. The real from-scratch training run
(`train_nano_from_scratch.py`, verified in Task 2 on a Blackhole p300c)
prints `vocab=68 (padded 96)`: 68 distinct characters in the actual
Shakespeare excerpt, rounded up to 96 by
`vocab_size = round_up_to_tile(tokenizer.vocab_size, 32)` before the model is
built. 96 is the next multiple of 32 above 68, so `tok_emb`'s `[vocab_size,
n_embd]` table tiles cleanly under `ttnn.TILE_LAYOUT` instead of leaving a
partial, wasted tile at the edge. `n_embd=384` (12×32) and `block_size=256`
(8×32) are already clean multiples, so the padding only bites on the
vocabulary axis at nano scale — but it's the same rule from lfs-00 either
way: Tensix moves and computes in whole 32×32 tiles, so shapes get rounded up
to fit.

## The residual stream: the model's spine

`x = tok_emb(idx) + pos_emb(pos)` is the **first write** to a tensor that
the rest of the model calls the residual stream — `x` in the code above, and
in every block from here through lfs-04. Every transformer block reads the
current `x`, computes something (attention in lfs-03, an MLP in lfs-04), and
**adds its output back into `x`** rather than replacing it:

```python
x = x + attention(x)   # lfs-03
x = x + mlp(x)          # lfs-04
```

That's the whole idea: `x` is a running sum, never overwritten, only added
to. Information from every earlier block stays directly reachable in later
blocks because it was never erased — just accumulated. The name "residual"
comes from ResNets, where the same additive shortcut solved vanishing
gradients in very deep CNNs; transformers inherited the trick wholesale.

Here's the payoff for this lab specifically: **the very first residual-stream
write is itself nothing but an elementwise add of two same-shaped tensors** —
`tok_emb(idx) + pos_emb(pos)`, both `[T, n_embd]` (broadcast to `[B, T,
n_embd]`). Elementwise add of two tiled tensors is also the simplest
operation TT-Lang can express. That's not a coincidence this lab is built
around — it's why the residual stream's first write is exactly where a
from-scratch TT-Lang arc should hand you your first kernel.

## First inception kernel: adding the position embedding to the token embedding

Recall from lfs-00: this arc descends from TT-NN<sup>™</sup> altitude to
TT-Lang for the hot kernels, and that descent is **inception, not
conversion** — there's no existing CUDA or Triton `add` kernel being ported
here. You're writing the TT-Lang expression of `tok_emb(idx) + pos_emb(pos)`
directly, as its original TT-native form.

---

The playground above defaults to exactly this kernel — **Element-wise Add**
is pre-selected, no dropdown to touch. Hit **Run** now, then come back and
walk through the fuller version below.

The browser's built-in copy is deliberately the simplest possible cut of
this kernel — single-tile granularity, no grid partitioning — so it fits in
a small editable panel. What follows is the LFS-specific version,
`content/templates/llm-from-scratch/kernels/eltwise_add.py`, verified
sim-runnable in Task 2 (`max abs error vs torch: 0.000000`, `PASSED`). It's
the same reader → compute → writer / DFB pattern the browser just ran, with
two additions: `GRANULARITY = 2` batches two row-tiles per reserve/wait
cycle, and `ttl.node(dims=2)` partitions the work across a multi-core grid
instead of one core. Read it as `eltwise_add(a_in, b_in, out)` where `a_in`
is `tok_emb(idx)`, `b_in` is `pos_emb(pos)`, and `out` is the residual
stream `x`:

```python
TILE_SIZE = 32
GRANULARITY = 2  # tiles processed per (row) step — a small blocking factor


@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """Y = A + B, tiled across the core grid.

    ``grid="auto"`` lets TT-Lang pick a core grid; each core (node) handles a
    slice of the row/column tiles. This is the reader -> compute -> writer
    pattern that every other kernel in this arc builds on.
    """
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)  # ceil-div
    cols_per_node = -(-col_tiles // grid_cols)

    # Typed L1 ring buffers. block_count=2 = double-buffering, so the reader can
    # be filling block N+1 while compute drains block N.
    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with (
                            a_dfb.wait() as a_blk,
                            b_dfb.wait() as b_blk,
                            out_dfb.reserve() as out_blk,
                        ):
                            out_blk.store(a_blk + b_blk)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                            tx_a = ttl.copy(a_in[r0:r1, col : col + 1], a_blk)
                            tx_b = ttl.copy(b_in[r0:r1, col : col + 1], b_blk)
                            tx_a.wait()
                            tx_b.wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_row in range(rows_per_node):
            row = node_row * rows_per_node + local_row
            if row < row_tiles:
                r0, r1 = row * GRANULARITY, (row + 1) * GRANULARITY
                for local_col in range(cols_per_node):
                    col = node_col * cols_per_node + local_col
                    if col < col_tiles:
                        with out_dfb.wait() as out_blk:
                            tx = ttl.copy(out_blk, out[r0:r1, col : col + 1])
                            tx.wait()
```

Three functions, three threads, matching the table from the CUDA callout
above:

- **`read()`** is the reader thread. It calls `.reserve()` on `a_dfb` and
  `b_dfb` — the *producer* role, claiming an empty L1 slot — then
  `ttl.copy(...).wait()` to pull one tile of `tok_emb(idx)` and one tile of
  `pos_emb(pos)` in from DRAM.
- **`compute()`** is the compute thread. It calls `.wait()` on `a_dfb` and
  `b_dfb` — the *consumer* role, blocking until the reader has filled a slot —
  adds the two tiles with a plain `+`, and `.store()`s the sum into a slot it
  `.reserve()`d on `out_dfb`.
- **`write()`** is the writer thread. It `.wait()`s on `out_dfb` — consuming
  compute's result — and `ttl.copy(...).wait()`s it back out to DRAM. That
  DRAM write *is* the residual stream's first value landing in memory.

**One shape nuance worth naming, not glossing over:** in PyTorch, `+`
broadcasts `pos_emb(pos)`'s `[T, n_embd]` across the batch dimension
automatically, so `tok_emb(idx)` (`[B, T, n_embd]`) and `pos_emb(pos)`
(`[T, n_embd]`) just work. `eltwise_add` above expects `a_in` and `b_in` to
already be the *same* shape — no implicit broadcast. On device, that means
broadcasting `pos_emb(pos)` out to `[B, T, n_embd]` first (or, since `B` is
tiny at nano scale, invoking the kernel once per batch element against a
`[T, n_embd]` slice), then handing this kernel two identically-shaped
tensors. That's a real implementation detail, not a simplification hidden
from you — and it doesn't change the reader → compute → writer / DFB shape
that's the actual point of this section.

Every tile makes exactly one DRAM read per input and one DRAM write for the
output; the addition itself never leaves L1. `GRANULARITY = 2` just means
each reserve/wait cycle moves two row-tiles at once instead of one — a
blocking-factor knob, not a change to the reader → compute → writer shape.

---

## Run it

Two ways to see this kernel run, and both are honest — no fallback, no
mocked output:

**In the browser:** the playground above, no install. Hit **Run** and the
Pyodide-hosted `ttlang-sim-lite` simulator executes the same reader →
compute → writer DFB choreography you just read, entirely client-side.

**Locally, in the standalone functional simulator:**

```bash
python content/templates/llm-from-scratch/kernels/eltwise_add.py
```

Expected output (the actual Task 2 verification run):

```
max abs error vs torch: 0.000000
PASSED
```

`main()` builds two random `[256, 256]` bf16 tensors, runs `eltwise_add`
through the DFB pipeline above, converts the result back with
`ttnn.to_torch`, and diffs it against plain `a + b` in PyTorch. Zero error,
exit 0 — the kernel you just read is not a diagram, it's a working add.

---

## Graduate box

You just ran this kernel two ways — the browser (`ttlang-sim-lite`, Pyodide,
no install) and, if you ran the command above, the standalone functional
simulator (same DFB engine, from a terminal). **Not one line of the kernel
changes between those two environments, and TT-Lang's own design goes
further still:** as tt-lang-intro puts it, "if you installed via pip and
have a Tenstorrent card, skip the `TT_METAL_SIMULATOR` and
`TT_METAL_SLOW_DISPATCH_MODE` variables — everything else is identical. The
same kernel source runs bit-exact on simulation and silicon." That's the
general guarantee TT-Lang is built around, and this elementwise-add kernel —
the arc's very first hand-authored one — is the first place you're seeing it
apply.

One honest scope note: this lab doesn't re-run `eltwise_add.py` on a
physical chip to independently confirm that guarantee for this specific
file — lfs-05 is where you'll watch a real from-scratch training loop
execute on a Blackhole p300c and read actual loss numbers coming
off hardware. What you've verified here is the sim side, twice over, with a
kernel you can already reason about completely.

---

## Next

Embeddings are built, the residual stream has its first value, and you've
written your first TT-Lang kernel end to end. Next: the mechanism that lets
every position in the residual stream look at every other position —
attention.

[→ Continue to Lab 3: Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
