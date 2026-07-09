---
id: lfs-02-embeddings
title: "Embeddings & the Residual Stream"
description: >-
  Build token embeddings and RoPE (rotary position embeddings), meet the residual stream, and write
  your first TT-Lang inception kernel live in the browser playground.
category: llm-from-scratch
tags: [embeddings, rope, residual, tt-lang, playground]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 30
playground: ttlang-sim
---

# Embeddings & the Residual Stream

Lab 1 ended at a single explicit line — `ttnn.from_torch(x, device=device,
layout=ttnn.TILE_LAYOUT)` — a `[batch, seq]` tensor of raw token IDs, tiled
and sitting in device DRAM. IDs alone don't mean anything to a model yet. This
lab turns each ID into a vector and writes it into the **residual stream**:
the running tensor every block in the model reads from and writes back into.

Position gets a different treatment entirely. A modern Llama-3-style model —
the `nanollama3` config this arc trains — never looks up and adds a positional
vector. It carries position as a **rotation**, applied later to Q and K inside
attention (Lab 3). That's **RoPE**.

This lab builds both halves: the token embedding that seeds the residual
stream, and the RoPE math that will rotate Q/K in the next lab. And for the
first time in this arc, you'll write a summation as a hand-authored TT-Lang
kernel — not a positional add, but the **residual add** every transformer
sub-layer performs — and run it live in your browser before you finish
reading.

## Coming from CUDA: shared memory becomes L1, and there's no warp scheduler

CUDA gives every thread block a pool of fast `__shared__` memory, scoped to
the block and the one Streaming Multiprocessor it runs on. Tensix has a direct
analog: **L1 SRAM**, scoped to one Tensix core. That's 1,464 KB per core on
both Wormhole<sup>™</sup> and Blackhole<sup>®</sup>, the exact figure
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"])
verified against the TT-Lang specification. Same idea — fast, local,
explicitly managed — just owned by a core instead of an SM.

The bigger difference is what happens *around* that memory. On CUDA, a warp
scheduler hides memory latency for you automatically: while one warp stalls on
a global-memory load, the SM silently swaps in another resident warp, and your
kernel body reads like straight-line math with the data movement implicit.

**Tensix has no warp scheduler, and TT-Lang has no implicit version of that
trick.** Every Tensix core runs three concurrent threads, and you author them
as three separate Python functions inside one `@ttl.operation`:

| Thread | Role | TT-Lang decorator |
|---|---|---|
| Reader | Streams input tiles from DRAM into L1 | `@ttl.datamovement()` |
| Compute | Does the math on tiles already in L1 | `@ttl.compute()` |
| Writer | Streams finished tiles from L1 back to DRAM | `@ttl.datamovement()` |

**Pick Your Altitude** introduced this as a *diagram* — reader → compute → writer, three
threads, explicit handoff through Dataflow Buffers (DFBs) with `reserve()`
and `wait()`. This lab is where that diagram stops being a diagram. The
kernel you're about to read is the first place in the arc where you write
those three functions out yourself.

One more contrast before the code: whatever kernel handles *position* in a
modern model carries no learned weights at all. A GPT-2-style positional
embedding table is a `[block_size, n_embd]` matrix of trainable parameters —
the optimizer updates it every step, exactly like `tok_emb`. **RoPE, which
this lab uses instead, is pure math**: cosines and sines computed once from
position and a fixed constant, with nothing to train. You'll see exactly why
later in this lab.

## Token embeddings, from scratch

A token ID by itself carries no information a matrix multiply can use — it's
just an index. The fix is the same one from the original Transformer paper: a
**lookup table**, one learned vector per token ID. What's different in a modern
Llama-3-style model is what's *missing*: there's no second lookup table for
position.

Here's the exact code, quoted verbatim from
`~/tt-scratchpad/llm-from-scratch/reference_gpt.py`'s `NanoLlama` module so the
prose and the code never drift:

```python
class NanoLlama(nn.Module):
    """The whole model: token embeddings -> Llama blocks -> RMSNorm -> LM head.

    No learned positional embedding table — position enters only through RoPE
    inside attention. cos/sin are precomputed once and threaded to every block.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # RoPE tables, sized to the max sequence length. head_dim = n_embd/n_head.
        head_dim = cfg.n_embd // cfg.n_head
        cos, sin = precompute_rope_cos_sin(head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, "sequence longer than block_size"
        x = self.drop(self.tok_emb(idx))          # residual stream (no pos-emb add)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)
        x = self.ln_f(x)
        logits = self.head(x)                      # [B, T, vocab]
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        return logits, loss

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
```

One embedding table, `tok_emb` — a lookup, nothing more. Shape
`[vocab_size, n_embd]`; row `i` is the learned vector for token ID `i`. At the
nano config (`vocab_size=96, n_embd=384`), that's a `[96, 384]` table.
`tok_emb(idx)` gathers one row per token ID in the batch, giving
`[B, T, n_embd]` directly — no second gather, no second table, nothing to add
it to.

**The line that matters most in this whole file is the one already commented
`# residual stream (no pos-emb add)`:**

```python
x = self.drop(self.tok_emb(idx))          # residual stream (no pos-emb add)
```

That comment is the whole pivot this lab teaches, written directly into the
model.

One more thing to notice in this excerpt: `self.rope_cos` and `self.rope_sin`
are registered with `register_buffer(..., persistent=False)`, **not**
`nn.Parameter` like `tok_emb.weight` is. Buffers don't receive gradients and
never appear in an optimizer's parameter list. That's the tell that RoPE,
precomputed once in `__init__`, isn't something the model ever learns. More on
what it computes later in this lab.

**A tile-alignment note, since [Tokenizer & Data](command:tenstorrent.showLesson?["lfs-01-tokenizer"]) flagged it and this is where it shows
up.** `LlamaConfig.vocab_size` defaults to 96 — not because Shakespeare's raw
character set lands there, but because 96 is a clean multiple of 32 (3×32
tiles). **Tokenizer & Data**'s `CharTokenizer` counts the actual distinct characters in
whatever text it's given, comfortably fewer than 96 for a Shakespeare excerpt.
`LlamaConfig` doesn't grow the embedding table to match that organic count; it
holds `vocab_size` at a tile-friendly constant instead.

`n_embd=384` (12×32) and `block_size=256` (8×32) are already clean multiples
too. Vocab is the one axis at nano scale where tile alignment is a deliberate
config choice. Same rule as [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) either way: Tensix moves and computes in
whole 32×32 tiles, so shapes get chosen (or rounded up) to fit.

## The residual stream: the model's spine

`x = self.tok_emb(idx)` is the residual stream's first value — plain, no
addition needed, because there's no second table to sum in. But starting from
the very first transformer block, `x` stops being *written* and starts being
*added to*.

Here's the exact pattern, quoted verbatim from `reference_gpt.py`'s `Block`
class:

```python
class Block(nn.Module):
    """A pre-norm Llama transformer block: RMSNorm -> GQA -> +residual,
    RMSNorm -> SwiGLU -> +residual (mirrors ttml's LlamaBlock).
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        self.attention_norm = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.attn = GroupedQueryAttention(cfg)
        self.mlp_norm = make_rmsnorm(cfg.n_embd, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attention_norm(x), cos, sin)  # residual around attention
        x = x + self.mlp(self.mlp_norm(x))                   # residual around MLP
        return x
```

That's the whole idea: `x` is a running sum, never overwritten, only added to.
Every block reads the current `x`, computes something (grouped-query attention
in [Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"]), a SwiGLU MLP in [The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"])), and **adds its output back into `x`**
rather than replacing it. Information from every earlier block stays directly
reachable in later blocks because it was never erased, only accumulated.

The name "residual" comes from ResNets, where the same additive shortcut
solved vanishing gradients in very deep CNNs. Transformers inherited the trick
wholesale.

Here's the payoff for this lab specifically. **Every one of those additions is
nothing but an elementwise add of two same-shaped tensors** — `x` and
`self.attn(...)`, or `x` and `self.mlp(...)`, both `[B, T, n_embd]`.
Elementwise add of two tiled tensors is also the simplest operation TT-Lang
can express.

That's not a coincidence this lab is built around. The residual stream's
defining operation, repeated at every block for the entire depth of the model,
is exactly where a from-scratch TT-Lang arc should hand you your first kernel.

## First inception kernel: the residual add

Recall from [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]): this arc descends from TT-NN<sup>™</sup> altitude to
TT-Lang for the hot kernels, and that descent is **inception, not
conversion**. There's no existing CUDA or Triton `add` kernel being ported
here. You're writing the TT-Lang expression of `x + sublayer(x)` directly, as
its original TT-native form — the exact add that fires at every block, for
every sub-layer, all the way through [The Transformer Block & the Model](command:tenstorrent.showLesson?["lfs-04-block-and-model"]).

---

The playground above defaults to exactly this kernel — **Element-wise Add**
is pre-selected, no dropdown to touch. Hit **Run** now, then come back and
walk through the fuller version below.

The browser's built-in copy is deliberately the simplest possible cut of this
kernel — single-tile granularity, no grid partitioning — so it fits in a small
editable panel.

What follows is the LFS-specific version,
`~/tt-scratchpad/llm-from-scratch/kernels/eltwise_add.py`, verified
sim-runnable (`max abs error vs torch: 0.000000`, `PASSED`). It's the same
reader → compute → writer / DFB pattern the browser just ran, with two
additions: `GRANULARITY = 2` batches two row-tiles per reserve/wait cycle, and
`ttl.node(dims=2)` partitions the work across a multi-core grid instead of one
core.

Read it as `eltwise_add(a_in, b_in, out)`, where `a_in` is `x` (the residual
stream entering the block), `b_in` is a sub-layer's output (`self.attn(...)`
in [Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"]), or `self.mlp(...)` in **The Transformer Block & the Model**), and `out` is the updated residual
stream:

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
  `ttl.copy(...).wait()` to pull one tile of `x` and one tile of the
  sub-layer's output in from DRAM.
- **`compute()`** is the compute thread. It calls `.wait()` on `a_dfb` and
  `b_dfb` — the *consumer* role, blocking until the reader has filled a slot —
  adds the two tiles with a plain `+`, and `.store()`s the sum into a slot it
  `.reserve()`d on `out_dfb`.
- **`write()`** is the writer thread. It `.wait()`s on `out_dfb` — consuming
  compute's result — and `ttl.copy(...).wait()`s it back out to DRAM. That
  DRAM write *is* one residual-stream update landing in memory — the same
  shape as every `x = x + sublayer(x)` line in the model, all the way
  through **The Transformer Block & the Model**.

**One shape nuance worth naming — and it's a cleaner story than the arc's
first draft had.** Framing this kernel around the *residual* add, instead of
the old GPT-2-style token+position add, sidesteps a broadcast problem
entirely. `x` and `self.attn(...)` (or `self.mlp(...)`) are *always* exactly
the same `[B, T, n_embd]` shape, at every block, for the entire depth of the
model. There's no batch-dimension broadcasting to reason about, ever.

`eltwise_add`'s contract — two identically-shaped tensors in, one
identically-shaped tensor out — is exactly what a residual add needs, with
nothing to paper over.

Every tile makes exactly one DRAM read per input and one DRAM write for the
output. The addition itself never leaves L1. `GRANULARITY = 2` just means each
reserve/wait cycle moves two row-tiles at once instead of one — a
blocking-factor knob, not a change to the reader → compute → writer shape.

---

## Positional information: RoPE, not a learned table

GPT-2's recipe added a second learned lookup table — `pos_emb =
nn.Embedding(block_size, n_embd)` — right alongside `tok_emb`, and summed it
into the residual stream before the first block. Modern Llama-3-style models,
including the `nanollama3` config this arc trains, drop that table entirely.
Position isn't looked up and added. It's applied later, inside attention
([Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])), as a **rotation**: **RoPE, Rotary Position Embeddings**.

The idea: take the Q and K vectors inside each attention layer (never the
residual stream `x` itself, and never V) and rotate each adjacent pair of
features by an angle. That angle grows with the token's position and shrinks
with the feature's index inside the head.

Rotate Q and K by amounts that depend on position, and the dot product `Q·Kᵀ`
— the thing softmax attention runs on — ends up depending on the *relative*
distance between two positions rather than their absolute values. No addition,
no lookup table, nothing to train. The rotation angles fall out of `theta` (a
single fixed constant — 500000 for this arc's nano config) and the position
index, both known in closed form before the model ever sees a training
example.

Here's the exact math, quoted verbatim from `reference_gpt.py`:

```python
def precompute_rope_cos_sin(head_dim: int, max_seq: int, theta: float):
    """Precompute the cos/sin tables RoPE rotates Q and K with.

    RoPE encodes position by *rotating* pairs of features by an angle that
    grows with position and shrinks with feature index — so relative position
    falls out of the Q·K dot product, with no learned positional table.

    inv_freq[i] = theta^(-2i/head_dim) for i in [0, head_dim/2).
    angle(pos, i) = pos * inv_freq[i]; we duplicate the half-width table so it
    lines up with rotate_half's [-x2, x1] convention (the Llama/HF layout).

    Returns (cos, sin), each shaped [max_seq, head_dim].
    """
    assert head_dim % 2 == 0, "RoPE needs an even head_dim"
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq).float()
    freqs = torch.outer(positions, inv_freq)      # [max_seq, head_dim/2]
    emb = torch.cat([freqs, freqs], dim=-1)       # [max_seq, head_dim]
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(-x2, x1): rotate the two halves of the last dim by 90 degrees."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x shaped [B, n_head, T, head_dim].

    x_rot = x * cos + rotate_half(x) * sin — the standard rotary formula. cos
    and sin are [T, head_dim] and broadcast over batch and head dims.
    """
    T = x.shape[-2]
    cos = cos[:T].to(x.dtype)          # [T, head_dim] -> broadcasts to [1,1,T,hd]
    sin = sin[:T].to(x.dtype)
    return x * cos + rotate_half(x) * sin
```

`precompute_rope_cos_sin` builds the `cos`/`sin` tables once, sized to
`[max_seq, head_dim]`. You already saw `NanoLlama.__init__` call this and stash
the result in non-persistent buffers — no gradients, nothing learned.
`apply_rope` is what actually rotates a tensor: `x * cos + rotate_half(x) *
sin`, applied to Q and K separately, once per attention layer, in **Attention from Scratch**.

### The TT-Lang expression: `kernels/rope.py`

`~/tt-scratchpad/llm-from-scratch/kernels/rope.py` is this arc's second
hand-authored TT-Lang kernel. Like `eltwise_add.py`, it's **verified
sim-runnable** in the standalone functional simulator (max abs error vs a
torch reference ~0.0005, well inside bf16 tolerance, `PASSED`). But read the
honest annotation at the top of that file before trusting it as "RoPE":

```python
SEQ_TILES = 1   # 32 tokens
EMBD_TILES = 1  # 32 head-dim (single head)


@ttl.operation(grid=(1, 1))
def rotary_qk_kernel(q_in, k_in, cos, q_out, k_out):
    """Apply the SIMPLIFIED rotary embedding to Q and K: q_rot = q * cos,
    k_rot = k * cos.

    (Real RoPE also splits the head dim and adds a rotate_half(x) * sin term;
    see reference_gpt.py. This kernel keeps the elementwise-multiply shape that
    the reader/compute/writer pipeline makes concrete.)
    """
    q_dfb = ttl.make_dataflow_buffer_like(q_in, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k_in, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    qo_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    ko_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)

    @ttl.compute()
    def compute():
        # Keep cos in scope while rotating both Q and K by it.
        with cos_dfb.wait() as cv:
            with q_dfb.wait() as qv, qo_dfb.reserve() as qo:
                qo.store(qv * cv)
            with k_dfb.wait() as kv, ko_dfb.reserve() as ko:
                ko.store(kv * cv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(k_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with cos_dfb.reserve() as blk:
            tx = ttl.copy(cos[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with qo_dfb.wait() as blk:
            tx = ttl.copy(blk, q_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
        with ko_dfb.wait() as blk:
            tx = ttl.copy(blk, k_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```

Same reader → compute → writer shape as `eltwise_add`, just a multiply instead
of an add, and two outputs (`q_out`, `k_out`) instead of one. The compute
thread waits on `cos_dfb` once and rotates both Q and K by it before the writer
drains either buffer.

**Read the docstring closely: this kernel multiplies by `cos` only.** It does
not split the head dimension into pairs, call `rotate_half`, or add the `sin`
term. That's the simplified, elementwise-multiply cut of RoPE this sim build
can execute today. The full math — `x * cos + rotate_half(x) * sin` — is the
`apply_rope` function you just read in `reference_gpt.py`, and it's the version
the arc's hero training run ([Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])) uses on hardware via `ttml.ops.rope`.

This kernel exists to make the *data-movement* shape of a RoPE-style op
concrete — DRAM → L1 → rotate → L1 → DRAM — not to be a drop-in replacement for
the real thing.

Where GPT-2's `pos_emb` was a `[block_size, n_embd]` parameter matrix the
optimizer updates every training step, RoPE's `cos`/`sin` tables are computed
once from `theta` and never touched by an optimizer again.

That's the trade this lab walks through. The same *kind* of hardware operation
— an elementwise map over 32×32 tiles, the same reader → compute → writer shape
as `eltwise_add` — stands in for a fundamentally different mechanism: rotation
of Q/K instead of lookup-and-sum into the residual stream. And a fundamentally
different parameter count: zero, versus `block_size × n_embd`.

---

## Run it

Three ways to see this lab's kernels run, and all are honest — no fallback,
no mocked output:

**In the browser:** the playground above, no install. Hit **Run** and the
Pyodide-hosted `ttlang-sim-lite` simulator executes the residual-add kernel's
reader → compute → writer DFB choreography entirely client-side.

**Locally, in the standalone functional simulator — the residual add:**

```bash
python ~/tt-scratchpad/llm-from-scratch/kernels/eltwise_add.py
```

Expected output:

```
max abs error vs torch: 0.000000
PASSED
```

`main()` builds two random `[256, 256]` bf16 tensors, runs `eltwise_add`
through the DFB pipeline above, converts the result back with
`ttnn.to_torch`, and diffs it against plain `a + b` in PyTorch. Zero error,
exit 0 — the same add that fires at every `x = x + sublayer(x)` line in the
model, verified.

**Locally, in the standalone functional simulator — the RoPE kernel:**

```bash
python ~/tt-scratchpad/llm-from-scratch/kernels/rope.py
```

Expected output (an actual verification run):

```
torch reference q_rot[0,:6]: [0.146484375, 0.029052734375, 0.09619140625, 0.1572265625, 0.04931640625, -0.1513671875]
TT-Lang kernel ran in sim. max abs error  Q: 0.000481  K: 0.000580
PASSED
```

`main()` builds random Q/K tensors and a non-trivial per-position `cos`
table, runs `rotary_qk_kernel` through the same DFB pipeline, and diffs the
result against `q * cos` / `k * cos` in PyTorch — the simplified rotary this
kernel actually implements, not the full `apply_rope` formula. Small,
nonzero error (bf16 rounding through the tile pipeline), well inside
tolerance, exit 0.

---

## Graduate box

You just ran the residual-add kernel two ways: the browser (`ttlang-sim-lite`,
Pyodide, no install) and — if you ran the command above — the standalone
functional simulator (same DFB engine, from a terminal). Not one line of the
kernel changes between those two environments.

**TT-Lang's own design goes further still.** As [Introduction to TT-Lang](command:tenstorrent.showLesson?["tt-lang-intro"]) puts it: "if you
installed via pip and have a Tenstorrent card, skip the `TT_METAL_SIMULATOR`
and `TT_METAL_SLOW_DISPATCH_MODE` variables — everything else is identical. The
same kernel source runs bit-exact on simulation and silicon." That's the
general guarantee TT-Lang is built around, and this residual-add kernel — the
arc's very first hand-authored one — is the first place you're seeing it apply.

The RoPE kernel earns a narrower claim. It's sim-runnable and verified against
a torch reference today. But — as its own file says outright — it's the
simplified cos-only cut of rotary embeddings, not the full `x * cos +
rotate_half(x) * sin` math your model actually needs. That full math runs on
Blackhole<sup>®</sup> today, just not via this hand-authored kernel: it's
`ttml.ops.rope`, exercised for real in [Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])'s hero training run. Two
different levels of "verified," named honestly rather than blurred together.

One honest scope note for both. This lab doesn't re-run either kernel on a
physical chip to independently confirm TT-Lang's sim-to-silicon guarantee for
these specific files. **Train It & Run for Real** is where you'll watch a real from-scratch training
loop execute on a Blackhole p300c and read actual loss numbers coming off
hardware. What you've verified here is the sim side, twice over, with kernels
you can already reason about completely.

---

## Next

Token embeddings are built, the residual stream has its first value, RoPE's
math is on the page, and you've written your first TT-Lang kernel end to
end. Next: the mechanism that lets every position in the residual stream
look at every other position — attention, where the RoPE rotation you just
met gets applied to Q and K for real, inside grouped-query attention.

[→ Continue to Lab 3: Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
