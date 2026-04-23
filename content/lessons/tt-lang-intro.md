---
id: tt-lang-intro
title: "Introduction to tt-lang"
description: >-
  Write your first tt-lang kernel: a concurrent compute + data-movement program
  that runs on the Tensix grid. Try it live in the browser via ttlang-sim-lite.
category: compilers
tags:
  - tt-lang
  - simulator
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
  - simulator
status: draft
estimatedMinutes: 20
playground: ttlang-sim
---

# Introduction to tt-lang

**tt-lang** is Tenstorrent's kernel programming model for the Tensix architecture.
Instead of writing GPU shaders or CPU threads, you write _concurrent programs_ that
run across hundreds of Tensix cores — each with dedicated compute and data-movement threads.

## The Tensix Thread Model

Every Tensix core runs three threads simultaneously:

| Thread | Role |
|--------|------|
| **Compute** | Math — matrix ops, activations, reductions |
| **Data Movement 0** | Load input tiles from DRAM → local buffer |
| **Data Movement 1** | Store output tiles from local buffer → DRAM |

The threads communicate through a **Dataflow Buffer (DFB)** — a typed ring buffer
that provides zero-copy handoffs and automatic backpressure.

```
DRAM ──[DM0 reads]──► DFB_in ──[Compute]──► DFB_out ──[DM1 writes]──► DRAM
```

## Your First Kernel: Element-Wise Addition

Below is a minimal `eltwise_add` kernel. The `@ttl.operation(grid="auto")` decorator
maps it across an automatically-selected Tensix grid.

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

### Key concepts

**`@ttl.compute()`** — defines the math thread. It _waits_ on filled input buffers and
_reserves_ output buffer slots. The `with` statements are blocking: compute pauses until
the data movement thread fills the slot.

**`@ttl.datamovement()`** — defines a DMA thread. It _reserves_ input buffer slots,
fills them from DRAM, then releases them to compute. The second DM thread
drains the output buffer.

**`a_dfb.wait()` vs `a_dfb.reserve()`** — `wait()` blocks until a filled slot is
available (consumer role); `reserve()` blocks until an empty slot is available (producer
role). The DFB scheduler handles all synchronization.

## Fused Operations: Three Inputs, One DMA Trip Out

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

## Matrix Multiply: The K-Reduction Accumulator

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
    M, K, N = a.shape[0]//TILE, a.shape[1]//TILE, b.shape[1]//TILE
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

## Running on Hardware

To run the `eltwise_add` kernel on a Tenstorrent device:

```bash
python3 -c "
import numpy as np, ttnn
device = ttnn.open_device(device_id=0)
a = ttnn.from_torch(np.random.rand(256,256).astype('float32'),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(np.random.rand(256,256).astype('float32'),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.add(a, b)
print('Result shape:', out.shape)
ttnn.close_device(device)
"
```

## Playground Kernels

The browser playground above includes several kernels you can run right now:

| Kernel | What it shows |
|--------|--------------|
| **Element-wise Add** | Minimal DFB producer/consumer loop — the hello-world of tt-lang |
| **Fused Multiply-Add** | `y = a * b + c` fused in L1; three DFBs, single-tile blocks — direct from the official tutorial |
| **Matmul + Bias + ReLU** | `y = relu(a @ b + bias)` with the k-reduction accumulator DFB ping-pong; the core matmul pattern on Tensix |
| **Row-partitioned Matmul** | `C = A @ B` distributed across a multi-core grid using `ttl.node()` |

## What's Next

- **Reductions** — sum, max, and softmax across tile grids
- **Multi-core dispatch** — `ttl.grid_size()` and `ttl.node()` for work partitioning
- **Game of Life** — cellular automaton as a tt-lang kernel
