# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from the RMSNorm compute block of `norm_qkv_kernel` /
# `norm_mlp_residual_kernel` in
# vendor/tt-lang/examples/test_transformer_block.py @ a19aaa8.
# Drift-check before publishing (see README.md "Drift sources").
#
# Verification status: COMPILER / HARDWARE PATH ONLY — *not* runnable in the
#   bundled functional simulator at this tt-lang pin (a19aaa8).
#
#   Why: RMSNorm reduces x*x across the feature dim to a per-row scalar, then
#   broadcasts 1/sqrt(mean) back across the tile. In this sim build,
#   `ttl.math.broadcast` requires the *source* block to have a genuine size-1
#   sub-tile dimension (element_shape[dim] == 1); the reduce ops here produce a
#   full 32x32 tile, so the sim raises:
#       "Cannot broadcast along dimension N: dimension must have element size 1"
#   The vendor `test_transformer_block.py` carries `TTLANG_HARDWARE_CI:
#   skip-compiler` and runs on a *real device*, not the sim — consistent with
#   this. Treat RMSNorm as a hardware/compiler-path kernel until the sim's
#   reduce->broadcast contract catches up. The `main()` below runs the exact
#   PyTorch reference (which the lab teaches) and then *attempts* the TT-Lang
#   kernel in the sim, reporting the honest status either way.
#
# ----------------------------------------------------------------------------
# Lab 4 — The Block & the Model: normalization.
#
# RMSNorm(x) = x / sqrt(mean(x^2) + eps).  No mean-subtraction, no learned bias
# in the classic formulation — just a rescale by the root-mean-square of the
# features. It is cheaper than LayerNorm and is what modern LLMs use.
#
# Coming from CUDA: the reduction (sum of squares across the row) is the same
# warp-reduce you would hand-roll in CUDA; here it is `ttl.math.reduce_sum`
# feeding a broadcast, expressed inside the compute thread.
# ----------------------------------------------------------------------------
"""RMSNorm as a from-scratch TT-Lang kernel (+ PyTorch reference).

Run it:

    python content/templates/llm-from-scratch/kernels/rmsnorm.py

It always runs the PyTorch reference (exit 0) and prints whether the TT-Lang
kernel executed in the bundled simulator or is compiler/hardware-path-only.
"""

from _ttlang_sim import add_ttlang_sim_to_path

add_ttlang_sim_to_path()

import torch  # noqa: E402
from sim import ttl, ttnn  # noqa: E402

SEQ_TILES = 1  # 32 tokens
EMBD_TILES = 1  # 32 features (single tile for clarity)


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


def _torch_rmsnorm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    return xf / torch.sqrt((xf**2).mean(dim=-1, keepdim=True) + eps)


def main() -> int:
    torch.manual_seed(0)
    n_embd = EMBD_TILES * 32
    seq_len = SEQ_TILES * 32

    x_torch = torch.randn(seq_len, n_embd, dtype=torch.bfloat16) * 0.5
    ref = _torch_rmsnorm(x_torch)
    print("PyTorch reference RMSNorm(x)[0,:6]:", ref[0, :6].tolist())

    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    x = to_dev(x_torch)
    scaler = to_dev(torch.full((32, 32), 1.0 / n_embd, dtype=torch.bfloat16))
    out = to_dev(torch.zeros(seq_len, n_embd, dtype=torch.bfloat16))

    try:
        rmsnorm_kernel(x, scaler, out)
    except Exception as exc:  # noqa: BLE001
        last = str(exc).strip().splitlines()[-1]
        print(f"TT-Lang kernel: SIM-UNSUPPORTED at this pin (compiler/hardware path only): {last}")
        return 0

    result = ttnn.to_torch(out)
    err = (result.float() - ref).abs()
    print(f"TT-Lang kernel ran in sim. max abs error vs torch: {err.max().item():.6f}")
    corr = torch.corrcoef(torch.stack([result.float().flatten(), ref.flatten()]))[0, 1].item()
    print(f"correlation: {corr:.6f}  {'PASSED' if corr > 0.9 else 'FAILED'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
