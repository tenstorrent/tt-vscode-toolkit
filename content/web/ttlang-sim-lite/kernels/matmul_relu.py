# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Fused Matrix Multiply + Bias + ReLU: y = relu(a @ b + bias)

Direct adaptation of the official TT-Lang matmul tutorial (Step 1).
Showcases the k-reduction accumulator DFB ping-pong — the architectural
pattern at the heart of high-throughput matmul on Tenstorrent Tensix cores.

Key concepts illustrated:
  - K-reduction accumulator: acc_dfb is both consumed and produced by the
    compute thread in a ping-pong pattern — each k-step reads the previous
    partial sum (wait), adds the a@b tile product, and writes the updated
    partial sum (reserve). Two acc_dfb slots alternate like a double buffer.
  - Fused bias + ReLU: after the k-loop the bias tile is added and relu is
    applied in the same compute step, before the result hits DRAM.
  - Zero DRAM traffic during k-reduction: all partial sums stay in L1.

DFB data flow (single Tensix core):

    DRAM ──[DM read]──► a_dfb, b_dfb, bias_dfb
                              │
                         acc_dfb ⟳ (k ping-pong, compute-internal)
                              │
    DRAM ◄──[DM write]── y_dfb ◄── relu(acc + bias)

Run with:
    python matmul_relu.py
"""

import numpy as np
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid=(1, 1))
def matmul_relu(
    a: ttnn.Tensor,
    b: ttnn.Tensor,
    bias: ttnn.Tensor,
    y: ttnn.Tensor,
) -> None:

    # Tile-coordinate dimensions.
    m_tiles = a.shape[0] // TILE_SIZE
    n_tiles = b.shape[1] // TILE_SIZE
    k_tiles = a.shape[1] // TILE_SIZE

    # Input DFBs — filled by the DM reader, consumed by compute.
    a_dfb    = ttl.make_dataflow_buffer_like(a,    shape=(1, 1), block_count=2)
    b_dfb    = ttl.make_dataflow_buffer_like(b,    shape=(1, 1), block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1, 1), block_count=2)

    # Accumulator DFB — internal to compute.  Two slots enable the ping-pong:
    # slot 0 holds the previous partial sum; slot 1 receives the next one;
    # their roles swap on every k-step.
    acc_dfb  = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    # Output DFB — filled by compute, drained by the DM writer.
    y_dfb    = ttl.make_dataflow_buffer_like(y, shape=(1, 1), block_count=2)

    # ── DM read thread ────────────────────────────────────────────────────────
    # For each (m, n) output tile: first push the bias tile, then stream all
    # k tiles of a and b.  The bias is read first so it is ready when compute
    # finishes the k-reduction.

    @ttl.datamovement()
    def read():
        for m in range(m_tiles):
            for n in range(n_tiles):
                with bias_dfb.reserve() as bias_blk:
                    ttl.copy(bias[m, n], bias_blk).wait()

                for k in range(k_tiles):
                    with a_dfb.reserve() as a_blk, b_dfb.reserve() as b_blk:
                        ttl.copy(a[m, k], a_blk).wait()
                        ttl.copy(b[k, n], b_blk).wait()

    # ── Compute thread ────────────────────────────────────────────────────────
    # For each output tile:
    #   1. Reserve an acc_dfb slot and zero it (initialize accumulator).
    #   2. For each k: consume the previous partial sum + a/b tiles, compute
    #      a new partial sum, push it back into acc_dfb (ping-pong).
    #   3. Fuse the bias add and ReLU into the final write to y_dfb.

    @ttl.compute()
    def compute():
        for _ in range(m_tiles):
            for _ in range(n_tiles):

                # Initialize the accumulator to zero.
                # ttl.math.fill returns a zero-valued block expression;
                # store() materializes it into acc_blk; the 'with' exit
                # calls push(), making slot 0 visible.
                with acc_dfb.reserve() as acc_blk:
                    acc_blk.store(ttl.math.fill(acc_blk, 0))

                # K-reduction: acc_dfb ping-pong.
                # Each iteration consumes the previous partial sum (wait) and
                # writes the updated one (reserve), alternating the two slots.
                # Both wait and reserve on acc_dfb are nested in the same scope;
                # push() runs before pop() so the new sum is visible before the
                # old slot is freed.
                for _ in range(k_tiles):
                    with (
                        a_dfb.wait()   as a_blk,
                        b_dfb.wait()   as b_blk,
                        acc_dfb.wait() as pre_acc,
                    ):
                        with acc_dfb.reserve() as acc_blk:
                            acc_blk.store(pre_acc + a_blk @ b_blk)

                # Fuse bias add + ReLU into the final tile write.
                # relu() is applied element-wise over the 32×32 tile in L1.
                with bias_dfb.wait() as bias_blk, acc_dfb.wait() as acc_blk:
                    with y_dfb.reserve() as y_blk:
                        y_blk.store(ttl.math.relu(acc_blk + bias_blk))

    # ── DM write thread ───────────────────────────────────────────────────────

    @ttl.datamovement()
    def write():
        for m in range(m_tiles):
            for n in range(n_tiles):
                with y_dfb.wait() as y_blk:
                    ttl.copy(y_blk, y[m, n]).wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        np.random.seed(42)
        M, K, N = 64, 64, 64   # 2×2×2 tiles — runs in seconds in the browser

        a_np    = np.random.randn(M, K).astype(np.float32)
        b_np    = np.random.randn(K, N).astype(np.float32)
        bias_np = np.random.randn(M, N).astype(np.float32)

        def to_tt(arr):
            return ttnn.from_torch(
                arr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
            )

        a_t    = to_tt(a_np)
        b_t    = to_tt(b_np)
        bias_t = to_tt(bias_np)
        y_t    = to_tt(np.zeros((M, N), dtype=np.float32))

        matmul_relu(a_t, b_t, bias_t, y_t)

        result   = ttnn.to_torch(y_t)
        expected = np.maximum(a_np @ b_np + bias_np, 0.0)  # numpy reference

        max_err = np.max(np.abs(result - expected))
        # bfloat16 k-reductions accumulate ~2% relative error on 2 tiles.
        assert np.allclose(result, expected, rtol=0.05, atol=0.05), (
            f"Mismatch!  max_err={max_err:.4f}"
        )
        print("PASSED!  Fused matmul + bias + ReLU verified.")
        print(f"  Shape: ({M},{K}) @ ({K},{N}) + bias → relu → ({M},{N})")
        print(f"  Max abs error: {max_err:.6f}  (bfloat16 k-reduction)")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
