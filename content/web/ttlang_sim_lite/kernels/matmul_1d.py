# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Row-partitioned matrix multiply: C = A @ B.

Each Tensix core handles a contiguous row-slice of the A matrix.
The compute thread accumulates partial products over K tile-columns before
flushing a result tile.

Run with:
    python matmul_1d.py
"""

import numpy as np
import ttl
import ttnn

TILE_SIZE = 32


@ttl.operation(grid="auto")
def matmul_1d(A: ttnn.Tensor, B: ttnn.Tensor, C: ttnn.Tensor) -> None:
    M_tiles = A.shape[0] // TILE_SIZE
    K_tiles = A.shape[1] // TILE_SIZE
    N_tiles = B.shape[1] // TILE_SIZE

    grid_rows, grid_cols = ttl.grid_size(dims=2)
    rows_per_node = -(-M_tiles // grid_rows)

    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1, 1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1, 1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1, 1), block_count=2)

    @ttl.compute()
    def compute():
        node_row, node_col = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for _n in range(N_tiles):
                acc = None
                for _k in range(K_tiles):
                    with a_dfb.wait() as ab, b_dfb.wait() as bb:
                        tile = ab @ bb
                        acc = tile if acc is None else acc + tile
                with c_dfb.reserve() as cb:
                    cb.store(acc)

    @ttl.datamovement()
    def read():
        node_row, node_col = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for n in range(N_tiles):
                for k in range(K_tiles):
                    with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                        ttl.copy(A[m:m+1, k:k+1], ab).wait()
                        ttl.copy(B[k:k+1, n:n+1], bb).wait()

    @ttl.datamovement()
    def write():
        node_row, node_col = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for n in range(N_tiles):
                with c_dfb.wait() as cb:
                    ttl.copy(cb, C[m:m+1, n:n+1]).wait()


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        M, K, N = 64, 64, 64
        a_np = np.random.rand(M, K).astype(np.float32)
        b_np = np.random.rand(K, N).astype(np.float32)

        A = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        B = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        C = ttnn.from_torch(
            np.zeros((M, N), dtype=np.float32),
            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
        )

        matmul_1d(A, B, C)

        result   = ttnn.to_torch(C)
        expected = a_np @ b_np
        max_err  = np.max(np.abs(result - expected))
        # bfloat16 truncation gives ~1% relative error
        assert np.allclose(result, expected, rtol=0.02, atol=0.02), (
            f"Mismatch! max_err={max_err:.4f}"
        )
        print("PASSED! Matrix multiply verified.")
        print(f"  Shape: ({M},{K}) @ ({K},{N}) → ({M},{N})")
        print(f"  Max abs error: {max_err:.6f}")
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
