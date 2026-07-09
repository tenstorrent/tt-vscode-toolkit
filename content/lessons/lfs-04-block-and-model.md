---
id: lfs-04-block-and-model
title: "The Transformer Block & the Model"
description: >-
  Assemble SwiGLU, RMSNorm, and residuals into a full Llama-style transformer block, stack it
  into a nano model, and see TT-Lang kernels drop in as ttnn ops. Scale to 80M.
category: llm-from-scratch
tags: [transformer, swiglu, rmsnorm, matmul, tt-lang]
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy, simulator]
status: draft
estimatedMinutes: 40
---

# The Transformer Block & the Model

> **Following along?** The code in this lab lives in your `~/tt-scratchpad/llm-from-scratch/` workspace. If you haven't created it yet, use the **Create the LLM-from-Scratch Project** button in [Lab 0](command:tenstorrent.showLesson?["lfs-00-intro"]).


[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
ended with a working attention output — GQA'd, RoPE'd, causal-masked, projected
back through `o_proj`. But attention is only half of what a transformer block
does to `x`. The other half is a feed-forward transformation applied
independently to every position, and in a modern Llama-3-style model that is
not a "GELU MLP" — it's **SwiGLU**.

This lab finishes the block. Three pieces: SwiGLU, **RMSNorm** (the pre-norm
every earlier lab has quoted inside `Block` without explaining), and the
residual wiring that turns `x = x + attn(...)` and `x = x + mlp(...)` into a
repeatable unit you stack six times. That's the exact `nanollama3` shape this
arc has been building toward since
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]). By the
end, you will have read, line by line, every component of the model
[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])
trains for real on Blackhole<sup>®</sup>.

## Coming from CUDA: kernel fusion, one more time

[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
built the case that TT-Lang's reader → compute → writer pipeline *is*
FlashAttention's memory-traffic argument, made textual: every intermediate
stays in L1, nothing round-trips DRAM between QKᵀ and ·V. That same argument
shows up twice more in this lab, at two different scales.

**RMSNorm is the small-scale case.** `RMSNorm(x) = x / sqrt(mean(x²) + eps) *
weight` looks like five separate operations — square, reduce-sum, rsqrt,
broadcast, multiply — and an unfused implementation would run each as its own
kernel, bouncing the `[T, n_embd]` activation tensor through DRAM between every
step. The TT-Lang `rmsnorm_kernel` you'll read below fuses all five into one
`@ttl.compute` body: `x` is read from DRAM into L1 exactly once, every
intermediate lives in an L1 dataflow buffer, and the normalized result is
written back exactly once. Same fusion story as attention's 17-buffer
pipeline, just five buffers instead of seventeen.

**SwiGLU is the larger-scale case, and it's fused a level up from tile math.**
`ttml.ops.swiglu` — the training-path op this lab's PyTorch reference mirrors —
is a single fused device op that runs the three projections (gate, up, down)
and the `silu(gate) * up` gate together, without round-tripping the
intermediate `silu(gate(x))` and `up(x)` tensors through DRAM the way three
separate `ttnn` calls would have to. TT-NN<sup>™</sup>'s fused ops are exactly
this trick — FlashAttention-style "keep it in fast memory," authored once
inside the op library instead of once per hand-written kernel. It's the same
payoff as a TT-Lang inception kernel, delivered at TT-NN altitude instead of
TT-Lang altitude — which is exactly why this lab teaches SwiGLU through
`ttml.ops.swiglu` rather than inventing a from-scratch TT-Lang kernel for it
(more on that boundary below).

## SwiGLU: the modern MLP

Every GPT-2-era MLP does the same two-step dance: expand `n_embd` to `4 *
n_embd`, apply GELU, project back down. Llama-3 — and the `nanollama3` config
this arc trains — replaces that with **SwiGLU**: a *gated* feed-forward that
adds a third projection and swaps GELU for SiLU. Here's the exact module,
quoted verbatim from `~/tt-scratchpad/llm-from-scratch/reference_gpt.py` so
the prose and the code can't drift apart:

```python
def compute_swiglu_intermediate_size(hidden_size: int, multiple_of: int = 256) -> int:
    """Llama SwiGLU intermediate size (mirrors ttml's helper of the same name).

    SwiGLU has three matrices (gate/up/down) vs two in a standard MLP, so the
    intermediate is scaled to 2/3 of 4*hidden = 8/3*hidden and rounded up to
    `multiple_of` for tile alignment. For hidden=384 this yields 1024.
    """
    unrounded = (4 * hidden_size * 2) // 3
    return ((unrounded + multiple_of - 1) // multiple_of) * multiple_of
```

```python
class SwiGLU(nn.Module):
    """Llama SwiGLU feed-forward: down(silu(gate(x)) * up(x)) — Lab 4.

    Three projections (no bias): `gate` (w1) and `up` (w3) both map n_embd ->
    intermediate; `down` (w2) maps back. SiLU-gates the up-projection. This
    replaces GPT-2's expand-4x + GELU MLP.
    """

    def __init__(self, cfg: LlamaConfig):
        super().__init__()
        inter = compute_swiglu_intermediate_size(cfg.n_embd)
        self.gate = nn.Linear(cfg.n_embd, inter, bias=False)  # w1
        self.up = nn.Linear(cfg.n_embd, inter, bias=False)    # w3
        self.down = nn.Linear(inter, cfg.n_embd, bias=False)  # w2
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))
```

Three linears, no bias, and one line of math: `down(silu(gate(x)) * up(x))`.
`gate` and `up` both project `n_embd` (384) up to an `intermediate` width
(1024, from the helper above).

`silu(gate(x))` acts as a learned, per-feature **gate** on `up(x)`. The gating
function `silu(z) = z * sigmoid(z)` is near-zero (and mildly negative, down to
about -0.28) for negative `z`, and tracks `z` itself for positive `z`. So each
element of `up(x)` gets smoothly let through, damped, or flipped slightly
negative, depending on what `gate(x)` computed for that same position — before
`down` projects back to `n_embd`.

That gate is the whole idea. A standard GELU MLP applies the *same* nonlinearity
everywhere; SwiGLU lets the network learn *which* features of `up(x)` to pass
through and by how much, on a per-token, per-feature basis. Empirically — this
is the finding Llama-3, and the
[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM) build this arc
credits, both act on — a gated feed-forward reaches lower loss per parameter and
per FLOP than the ungated GELU MLP it replaces. That's why every modern open
LLM you'll read the source of, this arc's own `nanollama3` included, uses some
SwiGLU variant instead.

**Why three projections don't cost three times as much.** A GELU MLP has two
matrices at width `4 * n_embd` (1536 for nano); SwiGLU has three matrices, but
`compute_swiglu_intermediate_size` scales the intermediate down to `8/3 *
n_embd` (rounded to a tile-friendly multiple of 256 — 1024 for nano) precisely
so the *total* parameter and FLOP count of `gate + up + down` lands close to
the two-matrix GELU MLP's `fc + proj`, rather than 1.5× it. You get the gating
mechanism essentially for free, budget-wise — the intermediate width shrinks
to pay for the extra matrix.

**The TT expression, stated honestly.** There is **no from-scratch TT-Lang
SwiGLU kernel in this arc** — that's a deliberate scope decision, not an
oversight. SwiGLU is taught two ways instead: the PyTorch module above (to
understand the math), and, on the real training path, the single fused device
op `ttml.ops.swiglu`, which composes `ttml.ops.unary.silu` and
`ttml.ops.binary.mul` around the three projections — i.e. exactly `silu(gate)
· up`, expressed as ops you already know the *shape* of from the elementwise add
in [Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
(`binary.mul` is the multiply cousin of the add you already wrote) rather than
as a new hand-authored reader/compute/writer pipeline. The
matmuls underneath `gate`, `up`, and `down` are ordinary matmuls — the same
`matmul` this lab shows you a few sections down.

## RMSNorm: normalizing before every sub-layer

Every `Block` you'll read in a moment starts both of its sub-layers —
attention and the MLP — by normalizing `x` first. That's **pre-norm**:
`RMSNorm(x)` runs *before* attention and *before* SwiGLU, not after, which is
what keeps a 6-block (or 6-, 12-, or deeper) stack numerically stable to
train. RMSNorm itself is a simplification of LayerNorm: no mean-subtraction,
no learned bias, just a rescale by the root-mean-square of the features:

```
RMSNorm(x) = x / sqrt(mean(x², dim=-1) + eps) * weight
```

Cheaper than LayerNorm (one reduction instead of two — no mean to subtract
first) and, like SwiGLU over GELU, it's simply what the modern recipe uses:
Llama-3, Mini-LLM, and this arc's own `nanollama3` all use RMSNorm in place of
LayerNorm throughout.

Here is the from-scratch TT-Lang kernel, quoted verbatim from
`~/tt-scratchpad/llm-from-scratch/kernels/rmsnorm.py` — this arc's third
hand-authored inception kernel, faithful to the RMSNorm compute block inside
`norm_qkv_kernel` / `norm_mlp_residual_kernel` in the vendor's
`test_transformer_block.py`:

```python
@ttl.operation(grid=(1, 1))
def rmsnorm_kernel(x, scaler, out):
    """out = RMSNorm(x).

    ``scaler`` is a tile pre-filled with 1/n_embd so the row-sum of squares
    becomes the mean of squares (the reduce op multiplies by it).
    """
    x_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)

    # RMSNorm intermediates
    sq_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    bcast_dfb = ttl.make_dataflow_buffer_like(x, shape=(SEQ_TILES, EMBD_TILES), block_count=2)

    @ttl.compute()
    def compute():
        with x_dfb.wait() as xv, scaler_dfb.wait() as sc:
            # 1. square
            with sq_dfb.reserve() as sq:
                sq.store(xv * xv)
            # 2. row-wise sum, scaled by 1/n -> mean of squares
            with sq_dfb.wait() as sqv, sum_dfb.reserve() as sm:
                sm.store(ttl.math.reduce_sum(sqv, sc, sm, dims=[0]))
            # 3. reciprocal sqrt -> 1/rms
            with sum_dfb.wait() as smv, sum_dfb.reserve() as rsq:
                rsq.store(ttl.math.rsqrt(smv))
            # 4. broadcast 1/rms across the feature dim
            with sum_dfb.wait() as rsqv, bcast_dfb.reserve() as bc:
                bc.store(ttl.math.broadcast(rsqv, bc, dims=[1]))
            # 5. normalize
            with bcast_dfb.wait() as bcv, out_dfb.reserve() as o:
                o.store(xv * bcv)

    @ttl.datamovement()
    def dm_read():
        with x_dfb.reserve() as blk:
            tx = ttl.copy(x[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
```

Read `compute()` as the five numbered stanzas the comments already name:
square the input into `sq_dfb`; `ttl.math.reduce_sum` collapses the row down
to a single scalar per row, pre-scaled by `scaler` (a tile filled with
`1/n_embd`) so the sum becomes a mean in one op; `ttl.math.rsqrt` turns that
mean-of-squares into `1/rms` directly — note the kernel takes `rsqrt(mean)`
with no separate `eps` tile added first, a pedagogical simplification of the
`+ eps` in the formula above (the PyTorch reference `main()` runs against
keeps its `eps=1e-6` for numerical safety on real, non-adversarial
activations); `ttl.math.broadcast` fans that one scalar back out across the
full feature-dim tile; and the final `xv * bcv` is the actual rescale. Five
tile ops, three intermediate buffers, one input read, one output write — the
fusion this lab's CUDA callout promised.

### Honest flag: the same simulator edge as attention

**This kernel does not run in the bundled functional simulator, for the exact
same reason attention's softmax didn't in
[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"]).**
RMSNorm's `reduce_sum`
collapses the feature dimension to a per-row scalar, then `ttl.math.broadcast`
has to fan that scalar back out across a full 32×32 tile — and the pinned
simulator's `broadcast` requires its *source* block to already have a genuine
size-1 sub-tile dimension, which a fresh reduction output doesn't have. The
file's own header states this outright:

> **Verification status: COMPILER / HARDWARE PATH ONLY** — *not* runnable in
> the bundled functional simulator at this tt-lang pin (`a19aaa8`).

Run it and you'll see the same honest fallback the attention kernel in
**Attention from Scratch** prints:

```bash
python ~/tt-scratchpad/llm-from-scratch/kernels/rmsnorm.py
```

```
PyTorch reference RMSNorm(x)[0,:6]: [...]
TT-Lang kernel: SIM-UNSUPPORTED at this pin (compiler/hardware path only): ...
```

`main()` always runs the exact PyTorch reference first — `xf /
sqrt((xf**2).mean(dim=-1, keepdim=True) + eps)` — so the file exits 0 with a
real numerical result no matter what the sim does, then attempts the TT-Lang
kernel and reports the honest status either way, exactly like `attention.py`.
The vendor's own `test_transformer_block.py` carries the same
`TTLANG_HARDWARE_CI: skip-compiler` marker and runs this exact compute block
on a real device — this is a simulator-ahead-of-compiler gap, not a
correctness problem with the tile math. On the real training path, this
normalization is `ttml.ops.rmsnorm` — the op `ttml.models.llama` calls before
every attention and MLP sub-layer, and the op this lab's hero training run in
[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])
actually exercises on Blackhole.

## Assembling the block

With SwiGLU and RMSNorm both on the page, the block itself is almost no code
at all — which is the point of pre-norm design. Here is `Block`, quoted
verbatim from `reference_gpt.py`:

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

Two lines, two identical shapes: `x = x + sublayer(norm(x))`. The first line is
the residual stream from
[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
and the `GroupedQueryAttention` from
[Attention from Scratch](command:tenstorrent.showLesson?["lfs-03-attention"])
meeting for the first time — `attention_norm` (RMSNorm) runs on `x`, `attn`
(GQA, RoPE'd Q/K, causal softmax) turns the normalized `x` into an attention
output, and the residual add — the exact `eltwise_add` kernel from
**Embeddings & the Residual Stream** — puts it back into `x`. The second line
is the same shape with this lab's two new pieces: `mlp_norm` (RMSNorm again, a
separate learned weight from `attention_norm`) normalizes `x`, `mlp` (SwiGLU)
transforms it, and another residual add puts the result back. `x` is read twice
and written twice per block, and every one of those four operations is something
you've now read the exact TT-native expression of: RMSNorm's kernel above, GQA
from **Attention from Scratch**, SwiGLU via `ttml.ops.swiglu`, and the residual
add from **Embeddings & the Residual Stream**.

## Stacking blocks into the model

`NanoLlama` is the same class
[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
quoted when it introduced the residual stream — but back then it asked you to
take `Block`'s internals on faith. Now that you've read every line inside
`Block`, here's the whole model again, verbatim from `reference_gpt.py`, with
nothing left unexplained:

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

Read `forward` top to bottom and it's four moves: embed (`tok_emb`, no
positional add — **Embeddings & the Residual Stream**), loop `n_layer` (6) times
through `Block` (this lab),
normalize once more (`ln_f`, one final RMSNorm after the last block, before
anything reads `x`), then project to vocabulary logits (`head`, a plain
linear — no bias, and **not** weight-tied to `tok_emb` in this reference,
unlike some GPT-2 implementations that share the embedding and output
matrices). `for block in self.blocks: x = block(x, ...)` is the entire "stack
N blocks" idea — six calls to the exact two-line function you just read,
each with its own independently-learned `attention_norm`, `attn`, `mlp_norm`,
and `mlp` weights, all reading and writing the same running `x`.

## Kernels as drop-in `ttnn` ops

Step back from the PyTorch reference for a moment and ask what "TT-native"
actually means for this model on device. The boundary is `ttnn.Tensor` in
`ttnn.TILE_LAYOUT`, sitting in DRAM or L1 — the 32×32 tile from
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]), still
doing all the work. Anything that consumes and produces that type can sit inside an
otherwise-`ttnn` forward pass, whether it's a library op (`ttnn.matmul`,
`ttnn.softmax`) or a hand-authored TT-Lang kernel like the two this lab has
shown you. **A TT-Lang kernel doesn't replace the forward pass — it drops
into one, op-by-op, exactly where a `ttnn` call would otherwise go.**

The clearest example is matmul, because it's everywhere in this lab's model:
`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate`, `up`, `down`, and `head` are
all, underneath, `Y = A @ B`. Here is `matmul`, quoted verbatim from
`~/tt-scratchpad/llm-from-scratch/kernels/matmul.py` — **sim-validated**,
unlike RMSNorm and attention, because a plain tiled matmul never needs the
broadcast pattern the pinned simulator rejects:

```python
@ttl.operation(grid=(1, 1))
def matmul(A: ttnn.Tensor, B: ttnn.Tensor, Y: ttnn.Tensor) -> None:
    """Y = A @ B.

    Shapes (torch)      Shapes (tiles)
    A : M, K            MT, KT
    B : K, N            KT, NT
    Y : M, N            MT, NT
    """
    M = A.shape[0]
    K = A.shape[1]
    N = B.shape[1]
    MT = M // TILE_SIZE
    NT = N // TILE_SIZE
    KT = K // TILE_SIZE

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1))
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1))
    y_dfb = ttl.make_dataflow_buffer_like(Y, shape=(1, 1))

    @ttl.datamovement()
    def matmul_read():
        for mt in range(MT):
            for nt in range(NT):
                for kt in range(KT):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        a_xf = ttl.copy(A[mt, kt], a_blk)
                        b_xf = ttl.copy(B[kt, nt], b_blk)
                        a_xf.wait()
                        b_xf.wait()

    @ttl.compute()
    def matmul_compute():
        for _ in range(MT):
            for _ in range(NT):
                with y_dfb.reserve() as y_blk:
                    y = ttl.math.fill(y_blk, 0)
                    for _ in range(KT):
                        with a_dfb.wait() as a_blk, b_dfb.wait() as b_blk:
                            # Accumulate one K-tile of the dot product.
                            y += a_blk @ b_blk
                    y_blk.store(y)

    @ttl.datamovement()
    def matmul_write():
        for mt in range(MT):
            for nt in range(NT):
                with y_dfb.wait() as y_blk:
                    y_xf = ttl.copy(y_blk, Y[mt, nt])
                    y_xf.wait()
```

This is the from-scratch shape of every linear projection in the model:
`matmul_read` streams one `(M-tile, K-tile)` pair of `A` and `B` tiles into L1
at a time; `matmul_compute` accumulates the dot product over all `KT` K-tiles
into a compute-local register (`y += a_blk @ b_blk`, the tiled-matmul inner
reduction loop you'd hand-write with shared memory and `__syncthreads()` on
CUDA — here the "sync" is just the `.reserve()`/`.wait()` handshake) before
storing the finished output tile; `matmul_write` drains it back to DRAM. Run
it (`python ~/tt-scratchpad/llm-from-scratch/kernels/matmul.py`) and it
prints `max abs error vs torch: ...` followed by `PASSED` — a real, clean sim
run, no honest-flag caveat needed.

**Note what the vendor original leaves out, and why.** The vendor's
`matmul.py` computes `Y = A @ B + C` — a fused bias add — but its bias term
uses a broadcast the functional simulator doesn't support yet (vendor marks
it `xfail`). This lab's kernel drops the bias and teaches the pure
accumulation loop, which is the pedagogical heart of matmul anyway; a bias
add, when you need one, is the same `eltwise_add` kernel from
[Embeddings & the Residual Stream](command:tenstorrent.showLesson?["lfs-02-embeddings"])
applied afterward.

Put the two kernels side by side and you get the honest picture the runtime
matrix in **Pick Your Altitude** promised: `matmul` sim-validated,
`rmsnorm_kernel`
compiler/hardware-path only — and both are, from the model's point of view,
interchangeable with a `ttnn` call. Sketching (not a shipped file — just the
shape) what one block's forward pass looks like with both dropped in:

```python
# Illustrative sketch, not a shipped file: a block forward at ttnn altitude,
# with TT-Lang kernels dropped in exactly where a ttnn library call would go.
def block_forward(x, w, rope_cos, rope_sin):
    normed = rmsnorm_kernel(x, w.attn_norm_scaler, out=...)          # <- TT-Lang
    q = matmul(normed, w.q_proj, out=...)                     # <- TT-Lang
    k = matmul(normed, w.k_proj, out=...)                     # <- TT-Lang
    v = matmul(normed, w.v_proj, out=...)                     # <- TT-Lang
    attn_out = ...                                                   # RoPE + GQA share + softmax·V (ttnn/ttml, lfs-03)
    x = ttnn.add(x, matmul(attn_out, w.o_proj, out=...))      # <- TT-Lang + ttnn
    normed2 = rmsnorm_kernel(x, w.mlp_norm_scaler, out=...)          # <- TT-Lang
    x = ttnn.add(x, ttml.ops.swiglu(normed2, w.gate, w.up, w.down))  # <- fused ttml op
    return x
```

Nothing here cares whether `rmsnorm_kernel` or `matmul` is the
hand-authored TT-Lang version from this lab or a future `ttnn.rms_norm` /
`ttnn.matmul` call — the contract is the same `ttnn.Tensor` in, `ttnn.Tensor`
out, at `TILE_LAYOUT`. That's the whole payoff of building at TT-NN altitude
and descending to TT-Lang only for the pieces you need to author yourself
(the ladder from **Pick Your Altitude**, now concrete): you get to choose, op
by op, which lines are
library calls and which are your own kernel — and swap one for the other
later without touching anything else in the forward pass.

## Scaling nano to ~80M

Every number in this arc so far has been the **nano** `nanollama3_char`
config — the one that actually trains in
[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"]).
The **~80M** number is the scale of
[Mini-LLM by Ashx098](https://github.com/Ashx098/Mini-LLM), the project this arc
credits in [Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]):
80M parameters, 361M training tokens,
~5 hours on one A100, final loss ~3.25. Same code, same architecture — every
class in this lab — with the config knobs turned up:

| Knob | Nano value | ~80M "hero" scaling |
|---|---|---|
| `embedding_dim` (`n_embd`) | 384 | larger (e.g. 768) |
| `num_heads` (`n_head`) | 6 | larger (e.g. 12) |
| `num_groups` (`n_kv_groups`, GQA) | 3 | larger, stays `< num_heads` |
| `num_blocks` (`n_layer`) | 6 | larger (e.g. 12) |
| `max_sequence_length` (`block_size`) | 256 | larger |
| RoPE `theta` | 500000 | 500000 (unchanged) |
| `vocab_size` | 96 (char-level) | 32000 (SentencePiece BPE, Mini-LLM's choice) |
| dropout | 0.0 | 0.0 |

Working through `NanoLlama.num_params()` by hand for the nano config —
tallying `tok_emb` (96×384), six blocks of `q_proj`/`k_proj`/`v_proj`/`o_proj`
plus SwiGLU's `gate`/`up`/`down` plus two RMSNorm weight vectors each, the
final `ln_f`, and `head` — lands at **9,810,816 parameters**, matching
`reference_gpt.py --smoke`'s reported **9.81M** exactly. That's the whole
"nano" promise made concrete: every parameter in a real, working Llama-3-shape
model, small enough to add up on paper.

**The params/DRAM reasoning, done the same way this project's own hardware
notes do it (`ct7-architecture-basics.md`'s training-memory breakdown):**
bf16 weights cost 2 bytes/parameter, bf16 gradients another 2 bytes/parameter,
and AdamW's two fp32 moment buffers cost roughly 8 bytes/parameter more — call
it ~12 bytes/parameter for weights + gradients + optimizer state, before
activations (which scale with batch size and sequence length, not just
parameter count).

- **Nano (9.81M params):** ≈ 118 MB.
- **~80M params (Mini-LLM's actual headline number):** ≈ 960 MB, call it ~1 GB.

Both totals are tiny next to even the most DRAM-constrained device this arc
supports: this repo's own lessons document **n150 at 12 GB DRAM** (see
`ct4-finetuning-basics.md`, `ct7-architecture-basics.md`) as the tight case —
tight enough that a real Llama-3.1-8B-Instruct model (8B parameters, ~100×
this lab's 80M target) reliably exhausts it, per this project's own hardware
notes. An 80M-parameter model is nowhere near that ceiling on any single
chip this arc targets. That's the honest reason nano *and* the ~80M hero
config both run as a **single-chip** job: `mesh_shape [1,1]` on one p300c or
p150 (p300c behaves as a single Blackhole chip, per this project's own
hardware-compatibility notes), or a single p100, n150, or n300 — no chip-to-
chip partitioning required. (This repo doesn't publish an exact DRAM-per-SKU
figure for p100/p150/p300c the way it does for n150 — verify against
Tenstorrent's published Blackhole<sup>®</sup> specs before treating any
specific GB number as load-bearing; the *comparison*, not a precise total, is
what this section's argument rests on.)

**When you'd actually reach for a multi-chip mesh** (T3000, Galaxy) is a
different regime entirely: once weights + gradients + optimizer state alone
start approaching or exceeding a single chip's DRAM — the billions-of-
parameters range, not the tens-of-millions range this arc trains at. At that
point the answer isn't "run a bigger kernel on one core," it's tensor or
pipeline parallelism across a mesh of chips, splitting the model itself
across device boundaries. Nothing in this lab's model needs that, and neither
does the ~80M hero target — which is exactly why this arc never asks you to
provision more than one chip.

## Run it

There's no new playground for this lab — the payoff is that the *whole
model* now runs, forward-pass only, in plain PyTorch, on CPU, anywhere:

```bash
python ~/tt-scratchpad/llm-from-scratch/reference_gpt.py --smoke
```

Expected output (an actual verification run):

```
model: nano Llama-3 (RoPE + RMSNorm + GQA + SwiGLU), mirrors ttml nanollama3
config: n_layer=6 n_head=6 n_kv_groups=3 (GQA) n_embd=384 vocab=96 rope_theta=500000
swiglu intermediate: 1024
parameters: 9,810,816
logits shape: [2, 32, 96]  (expected [2, 32, 96])
initial loss: 4.8314  (~ln(vocab) = 4.5643)
PASSED
```

`smoke()` builds the exact `LlamaConfig` this lab has walked through,
constructs `NanoLlama`, runs one forward pass on random token IDs, and checks
the logits come out `[batch, seq, vocab]`-shaped with an initial loss near
`ln(vocab_size)` — the value you'd expect from a freshly-initialized model
that hasn't learned anything yet (a uniform distribution over 96 characters
has exactly that cross-entropy). That "near `ln(vocab)`" check is the
forward-pass equivalent of `eltwise_add`'s `max abs error: 0.000000` — a real,
falsifiable signal that the whole assembled model — embeddings, six blocks of
RMSNorm/GQA/RMSNorm/SwiGLU, final norm, LM head — is wired together
correctly, computed entirely on CPU, no device or simulator required.

**What this lab does not do is train the model.** A forward pass and an
initial loss near `ln(vocab)` tell you the wiring is correct; watching that
loss actually drop needs a backward pass, an optimizer, and real data — the
from-scratch training loop (cross-entropy, AdamW, backprop) that is the entire
subject of
[Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"]),
run for real on Blackhole hardware.

## Graduate box

You've now read every component of a modern Llama-3-shape model from scratch:
SwiGLU's gated feed-forward (PyTorch reference, `ttml.ops.swiglu` on the
training path, honestly no from-scratch TT-Lang kernel), RMSNorm's fused
five-op reduction (a from-scratch TT-Lang kernel, compiler/hardware-path only
— the same simulator edge as attention), the two-line pre-norm `Block` that
wires both sub-layers around residual adds, and `NanoLlama`, which stacks six
of those blocks into a **9,810,816-parameter** model whose forward pass you
just ran and verified. You've also seen the boundary that makes any of this
TT-native in the first place: a `ttnn.Tensor` at `TILE_LAYOUT` doesn't care
whether the op that touches it is a `ttnn` library call or a hand-authored
TT-Lang kernel like `matmul` (sim-validated) or `rmsnorm_kernel`
(compiler/hardware-path only) — both drop in exactly the same way.

And you know precisely what changes, and what doesn't, on the way to the
~80M "hero" scale this arc has pointed at since
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]): the same
classes, bigger config knobs, a real BPE vocabulary instead of char-level —
and, worked through above, nowhere near enough parameters to need more than one
chip.

## Next

The model is fully assembled and its forward pass is verified. What's left
is the part that turns a freshly-initialized 9.81M-parameter model into one
that actually predicts text: a from-scratch training loop — cross-entropy,
AdamW, backprop — run for real on Blackhole<sup>®</sup> hardware, watching the
loss drop the way the honest runtime matrix in
[Pick Your Altitude](command:tenstorrent.showLesson?["lfs-00-intro"]) already
previewed (4.69 → 3.23 over 20 steps, on a p300c, exit 0).

[→ Continue to Lab 5: Train It & Run for Real](command:tenstorrent.showLesson?["lfs-05-train-and-run"])
