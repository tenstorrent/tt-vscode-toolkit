# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from vendor/tt-lang/examples/matmul.py @ a19aaa8.
# Drift-check before publishing (see README.md "Drift sources").
#
# Verification status: SIM-VALIDATED (functional simulator).
#   NOTE: the vendor matmul.py computes Y = A @ B + C. The bias term uses a
#   tile broadcast (ttl.math.broadcast(..., dims=[0])) that the current
#   functional simulator does NOT support — vendor marks matmul.py xfail in its
#   sim suite ("Required broadcast not yet supported"). We therefore teach the
#   pure matmul Y = A @ B here (the accumulation loop, which is the pedagogical
#   heart), and add the bias later as a separate elementwise op. This keeps the
#   lesson runnable today; re-check the broadcast support on drift.
#
# ----------------------------------------------------------------------------
# Lab 4 — The Block & the Model: matmul is the workhorse.
#
# Every linear projection (QKV, output, MLP fc/proj, the LM head) is a matmul.
# The from-scratch pattern accumulates over the K (contraction) dimension one
# tile at a time in a compute-local register, then writes the finished output
# tile out. This is the reader -> compute -> writer pipeline again, now with an
# inner reduction loop.
#
# Coming from CUDA: this is the tiled-matmul you would hand-write with shared
# memory + __syncthreads(). Here the "shared memory" is L1, and the sync is the
# .reserve()/.wait() handshake on the dataflow buffers — no explicit barrier.
# ----------------------------------------------------------------------------
"""Tiled matmul (Y = A @ B) as a from-scratch TT-Lang kernel.

Run it (functional simulator, no device needed):

    python content/templates/llm-from-scratch/kernels/matmul.py

Expected: "PASSED" with max-abs-error vs a torch reference small (fp32 sim).
"""

from _ttlang_sim import add_ttlang_sim_to_path

add_ttlang_sim_to_path()

import torch  # noqa: E402
from sim import ttl, ttnn  # noqa: E402

TILE_SIZE = 32


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


def main() -> int:
    M, K, N = 64, 96, 128
    A_torch = torch.rand((M, K), dtype=torch.float32)
    B_torch = torch.rand((K, N), dtype=torch.float32)

    A = ttnn.from_torch(A_torch)
    B = ttnn.from_torch(B_torch)
    Y = ttnn.empty((M, N), dtype=torch.float32)

    matmul(A, B, Y)

    result = ttnn.to_torch(Y)
    expected = A_torch @ B_torch
    max_err = (result.float() - expected.float()).abs().max().item()
    print(f"max abs error vs torch: {max_err:.6f}")
    if torch.allclose(result, expected, atol=1e-3, rtol=1e-3):
        print("PASSED")
        return 0
    print("FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
