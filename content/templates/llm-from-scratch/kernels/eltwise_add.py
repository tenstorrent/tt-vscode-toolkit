# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from vendor/tt-lang/examples/eltwise_add.py @ a19aaa8.
# Drift-check before publishing (see README.md "Drift sources").
#
# Verification status: SIM-VALIDATED (functional simulator) AND
# compiler-supported upstream. In tt-lang's own sim test suite this kernel
# passes with both the "greedy" and "fair" schedulers.
#
# ----------------------------------------------------------------------------
# Lab 2 — Embeddings & the Residual Stream: the FIRST inception kernel.
#
# The residual stream is just repeated elementwise addition: every sub-layer
# adds its output back into the running activations. So the simplest possible
# TT-native kernel — add two tiled tensors — is also the first real building
# block of the model.
#
# Coming from CUDA: in CUDA you would write one __global__ function and let the
# warp scheduler interleave loads, math, and stores. In TT-Lang you write that
# pipeline out explicitly as THREE concurrent threads that hand tiles to each
# other through typed L1 ring buffers (Dataflow Buffers):
#
#     read (datamovement) --> compute --> write (datamovement)
#
# reader  : DRAM -> L1  (ttl.copy into a buffer you .reserve())
# compute : L1  -> L1   (a + b on 32x32 tiles held in the buffers)
# writer  : L1  -> DRAM (ttl.copy a buffer you .wait() for, back out)
# ----------------------------------------------------------------------------
"""Elementwise add (Y = A + B) as a from-scratch TT-Lang kernel.

Run it (functional simulator, no device needed):

    python content/templates/llm-from-scratch/kernels/eltwise_add.py

Expected: "PASSED" with max-abs-error vs a torch reference ~0 (bf16 tolerance).
"""

from _ttlang_sim import add_ttlang_sim_to_path

add_ttlang_sim_to_path()

import torch  # noqa: E402
from sim import ttl, ttnn  # noqa: E402  (simulator ttl / ttnn)

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


def main() -> int:
    dim = 256
    a_torch = torch.rand((dim, dim), dtype=torch.bfloat16)
    b_torch = torch.rand((dim, dim), dtype=torch.bfloat16)

    a = ttnn.from_torch(a_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    b = ttnn.from_torch(b_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    out = ttnn.from_torch(
        torch.zeros_like(a_torch), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT
    )

    eltwise_add(a, b, out)

    result = ttnn.to_torch(out)
    expected = a_torch + b_torch
    max_err = (result.float() - expected.float()).abs().max().item()
    print(f"max abs error vs torch: {max_err:.6f}")
    if torch.allclose(result, expected, rtol=1e-2, atol=1e-2):
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
