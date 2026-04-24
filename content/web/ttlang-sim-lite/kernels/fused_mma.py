# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Fused Multiply-Add: y = a * b + c

Direct adaptation of the official TT-Lang elementwise tutorial (Step 1).
Demonstrates the three-thread DFB producer/consumer model running on a
single Tensix core:

    DRAM ──[DM read]──► a_dfb, b_dfb, c_dfb ──[Compute]──► y_dfb ──[DM write]──► DRAM

Key concepts illustrated:
  - @ttl.operation    — declare a kernel and the grid it runs on
  - @ttl.compute      — the math thread; waits on filled tiles, pushes results
  - @ttl.datamovement — DMA threads; reserves slots, fills them, releases to compute
  - DataflowBuffer    — typed L1 ring buffer; block_count=2 enables double-buffering
  - ttl.copy / tx.wait — initiate and await a tile transfer

The fused a*b+c operation shows how all three tensors are streamed tile by tile
through L1 without any intermediate DRAM writes.

Run with:
    python fused_mma.py
"""

import numpy as np
import ttl
import ttnn

TILE_SIZE = 32


# @ttl.operation runs the kernel on a 1×1 grid (one Tensix core).
# All four tensors live in DRAM; the function body specifies the DFB wiring
# and the three threads that execute concurrently on the core.

@ttl.operation(grid=(1, 1))
def fused_mma(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    c: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:

    # Convert from element count to tile coordinates.
    rows = a.shape[0] // TILE_SIZE
    cols = a.shape[1] // TILE_SIZE

    # DataflowBuffers are L1 ring buffers shared between threads.
    # shape=(1, 1) → each slot holds one 32×32 tile.
    # block_count=2 → two slots; one being filled while the other is processed
    # (double-buffering hides DMA latency in the real hardware).
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1, 1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1, 1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    # ── Compute thread ────────────────────────────────────────────────────────
    # wait()    — block until the DM reader has pushed a filled tile
    # reserve() — block until the DM writer has freed a slot for the result
    # The 'with' statement calls pop() / push() automatically on exit.

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                with (
                    a_dfb.wait() as a_blk,
                    b_dfb.wait() as b_blk,
                    c_dfb.wait() as c_blk,
                    y_dfb.reserve() as y_blk,
                ):
                    # Fused element-wise operation — runs in L1, no DRAM traffic.
                    y_blk.store(a_blk * b_blk + c_blk)

    # ── DM read thread ────────────────────────────────────────────────────────
    # Streams all three input tensors tile by tile from DRAM into their DFBs.
    # reserve() claims an empty L1 slot; ttl.copy() initiates the DMA transfer;
    # tx.wait() waits for it to complete; the 'with' exit calls push() to signal
    # the compute thread that the tile is ready.

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with (
                    a_dfb.reserve() as a_blk,
                    b_dfb.reserve() as b_blk,
                    c_dfb.reserve() as c_blk,
                ):
                    # Tile-coordinate indexing: a[row, col] selects the 32×32 tile
                    # at (row, col) — not element (row*32, col*32).
                    ttl.copy(a[row, col], a_blk).wait()
                    ttl.copy(b[row, col], b_blk).wait()
                    ttl.copy(c[row, col], c_blk).wait()

    # ── DM write thread ───────────────────────────────────────────────────────
    # Drains y_dfb: waits for a filled output tile, copies it to DRAM, then
    # pops the slot to free it for the compute thread.

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as y_blk:
                    ttl.copy(y_blk, y[row, col]).wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        np.random.seed(42)
        dim = 64  # 2×2 tile grid — completes in seconds in the browser

        a_np = np.random.rand(dim, dim).astype(np.float32)
        b_np = np.random.rand(dim, dim).astype(np.float32)
        c_np = np.random.rand(dim, dim).astype(np.float32)

        def to_tt(arr):
            return ttnn.from_torch(
                arr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        a = to_tt(a_np)
        b = to_tt(b_np)
        c = to_tt(c_np)
        y = to_tt(np.zeros((dim, dim), dtype=np.float32))

        fused_mma(a, b, c, y)

        result   = ttnn.to_torch(y)
        expected = a_np * b_np + c_np
        max_err  = np.max(np.abs(result - expected))

        # bfloat16 has ~0.4% precision; 1% tolerance is conservative.
        assert np.allclose(result, expected, rtol=0.01, atol=0.01), (
            f"Mismatch!  max_err={max_err:.4f}"
        )
        print("PASSED!  Fused multiply-add (a * b + c) verified.")
        print(f"  Shape: {a_np.shape}   Max abs error: {max_err:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
