# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Element-wise tensor addition on the Tenstorrent Tensix architecture.

This kernel demonstrates the core tt-lang programming model:
  - A compute thread and two data-movement threads run concurrently per core.
  - Data flows through a typed dataflow buffer (DFB) that coordinates
    producer/consumer synchronization automatically.
  - The @ttl.operation decorator maps the kernel across an auto-selected grid.

Run with:
    python eltwise_add.py
"""

import numpy as np
import ttl
import ttnn

TILE_SIZE = 32
GRANULARITY = 2


@ttl.operation(grid="auto")
def eltwise_add(a_in: ttnn.Tensor, b_in: ttnn.Tensor, out: ttnn.Tensor) -> None:
    row_tiles = a_in.shape[0] // TILE_SIZE // GRANULARITY
    col_tiles = a_in.shape[1] // TILE_SIZE

    grid_rows, grid_cols = ttl.grid_size(dims=2)
    rows_per_node = -(-row_tiles // grid_rows)
    cols_per_node = -(-col_tiles // grid_cols)

    a_dfb = ttl.make_dataflow_buffer_like(a_in, shape=(GRANULARITY, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b_in, shape=(GRANULARITY, 1), block_count=2)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(GRANULARITY, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_row, node_col = ttl.node(dims=2)
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
        node_row, node_col = ttl.node(dims=2)
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
        node_row, node_col = ttl.node(dims=2)
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


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        dim = 64  # Use 64 instead of 256 for faster browser execution
        a_np = np.random.rand(dim, dim).astype(np.float32)
        b_np = np.random.rand(dim, dim).astype(np.float32)

        a = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        b = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        out = ttnn.from_torch(
            np.zeros_like(a_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
        )

        eltwise_add(a, b, out)

        result = ttnn.to_torch(out)
        expected = a_np + b_np
        assert np.allclose(result, expected, rtol=1e-2, atol=1e-2), (
            f"Mismatch! max_diff={np.max(np.abs(result - expected)):.4f}"
        )
        print("PASSED! Element-wise addition verified.")
        print(f"  Input shape: {a_np.shape}, dtype: {a_np.dtype}")
        print(f"  Max abs difference from expected: {np.max(np.abs(result - expected)):.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
