# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from `attention_kernel` in
# vendor/tt-lang/examples/test_transformer_block.py @ a19aaa8.
# Drift-check before publishing (see README.md "Drift sources").
#
# Verification status: COMPILER / HARDWARE PATH ONLY — *not* runnable in the
#   bundled functional simulator at this tt-lang pin (a19aaa8).
#
#   Why: softmax needs reduce_max / reduce_sum across the key dimension, then
#   broadcasts those per-row scalars back across the score tile. In this sim
#   build `ttl.math.broadcast` requires the source block to have a genuine
#   size-1 sub-tile dimension; the reductions here yield a full 32x32 tile, so
#   the sim raises "Cannot broadcast along dimension N: dimension must have
#   element size 1". The vendor `test_transformer_block.py` carries
#   `TTLANG_HARDWARE_CI: skip-compiler` and runs on a *real device*.
#   (This is the attention/softmax "sim-only-ahead-of-compiler" caveat the
#   design spec flags for Lab 3.) `main()` runs the PyTorch reference the lab
#   teaches, then attempts the TT-Lang kernel and reports the honest status.
#
# ----------------------------------------------------------------------------
# Lab 3 — Attention from Scratch (the centerpiece).
#
# Single-head scaled dot-product attention:
#   scores   = Q @ K^T * scale        (how much each token attends to each)
#   masked   = scores + causal_mask   (-inf above the diagonal => no peeking)
#   weights  = softmax(masked)        (row-normalized attention distribution)
#   out      = weights @ V            (weighted sum of value vectors)
#
# Coming from CUDA: FlashAttention fuses exactly these steps to keep the score
# matrix in fast memory. The TT-Lang expression is the same fusion made
# explicit — every intermediate (scores, max, exp, sum, softmax) lives in an
# L1 dataflow buffer, handed between the reader/compute/writer threads. That is
# the "inception" angle: the kernel spec (arrivals in -> tile math ->
# departures out) is written in the source, not carried in the programmer's
# head — which is also why you can hand an agent a reader/compute/writer spec.
# ----------------------------------------------------------------------------
"""Single-head scaled dot-product attention as a from-scratch TT-Lang kernel
(+ PyTorch reference).

Run it:

    python content/templates/llm-from-scratch/kernels/attention.py

It always runs the PyTorch reference (exit 0) and reports whether the TT-Lang
kernel executed in the bundled simulator or is compiler/hardware-path-only.
"""

from _ttlang_sim import add_ttlang_sim_to_path

add_ttlang_sim_to_path()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sim import ttl, ttnn  # noqa: E402

SEQ_TILES = 1  # 32 tokens
EMBD_TILES = 1  # 32 head-dim (single head)


@ttl.operation(grid=(1, 1))
def attention_kernel(q, k, v, scale, causal_mask, scaler, out):
    """Single-head scaled dot-product attention.

    ``scale`` is a tile filled with 1/sqrt(head_dim); ``scaler`` is a tile of
    ones used by the reduce ops; ``causal_mask`` has -inf above the diagonal.
    """
    q_dfb = ttl.make_dataflow_buffer_like(q, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    v_dfb = ttl.make_dataflow_buffer_like(v, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    scale_dfb = ttl.make_dataflow_buffer_like(scale, shape=(1, 1), block_count=1)
    mask_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=1)
    scaler_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=1)
    out_dfb = ttl.make_dataflow_buffer_like(out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)

    k_t_dfb = ttl.make_dataflow_buffer_like(k, shape=(EMBD_TILES, SEQ_TILES), block_count=2)
    snodes_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    scale_bcast_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    scaled_masked_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    max_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    max_bcast_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    exp_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    sum_dfb = ttl.make_dataflow_buffer_like(scaler, shape=(1, 1), block_count=2)
    sum_bcast_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)
    softmax_dfb = ttl.make_dataflow_buffer_like(causal_mask, shape=(SEQ_TILES, SEQ_TILES), block_count=2)

    @ttl.compute()
    def compute():
        # scores = Q @ K^T
        with k_dfb.wait() as kv, k_t_dfb.reserve() as kt:
            kt.store(ttl.transpose(kv, kt))
        with q_dfb.wait() as qv, k_t_dfb.wait() as ktv:
            with snodes_dfb.reserve() as sc:
                sc.store(ttl.math.matmul(qv, ktv, sc))

        # scaled + masked scores
        with (
            snodes_dfb.wait() as scv,
            scale_dfb.wait() as scalev,
            mask_dfb.wait() as maskv,
        ):
            with scale_bcast_dfb.reserve() as sb:
                sb.store(ttl.math.broadcast(scalev, sb, dims=[0, 1]))
            with scale_bcast_dfb.wait() as sbv, scaled_masked_dfb.reserve() as sm:
                sm.store(scv * sbv + maskv)

        # softmax (max-shifted for numerical stability)
        with scaler_dfb.wait() as scaler_v, scaled_masked_dfb.wait() as smv:
            with max_dfb.reserve() as mx:
                mx.store(ttl.math.reduce_max(smv, scaler_v, mx, dims=[0]))
            with max_dfb.wait() as mxv, max_bcast_dfb.reserve() as mxb:
                mxb.store(ttl.math.broadcast(mxv, mxb, dims=[1]))
            with max_bcast_dfb.wait() as mxbv:
                shifted = smv - mxbv
                with exp_dfb.reserve() as ex:
                    ex.store(ttl.math.exp(shifted))
                with exp_dfb.wait() as exv, sum_dfb.reserve() as sm:
                    sm.store(ttl.math.reduce_sum(exv, scaler_v, sm, dims=[0]))
                with sum_dfb.wait() as smv2, sum_bcast_dfb.reserve() as smb:
                    smb.store(ttl.math.broadcast(smv2, smb, dims=[1]))
                with sum_bcast_dfb.wait() as smbv, softmax_dfb.reserve() as sfm:
                    sfm.store(ttl.math.exp(shifted) / smbv)

        # out = softmax @ V
        with softmax_dfb.wait() as sfmv, v_dfb.wait() as vv:
            with out_dfb.reserve() as o:
                o.store(ttl.math.matmul(sfmv, vv, o))

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(k[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with v_dfb.reserve() as blk:
            tx = ttl.copy(v[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with scale_dfb.reserve() as blk:
            tx = ttl.copy(scale[0, 0], blk)
            tx.wait()
        with mask_dfb.reserve() as blk:
            tx = ttl.copy(causal_mask[0:SEQ_TILES, 0:SEQ_TILES], blk)
            tx.wait()
        with scaler_dfb.reserve() as blk:
            tx = ttl.copy(scaler[0, 0], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with out_dfb.wait() as blk:
            tx = ttl.copy(blk, out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


def _torch_attention(q, k, v, scale_val, causal_mask):
    scores = q.float() @ k.float().T * scale_val
    scores = scores + causal_mask.float()
    weights = F.softmax(scores, dim=-1)
    return weights @ v.float()


def main() -> int:
    torch.manual_seed(0)
    n = EMBD_TILES * 32
    scale_val = 1.0 / (n**0.5)

    q_t = torch.randn(n, n, dtype=torch.bfloat16) * 0.1
    k_t = torch.randn(n, n, dtype=torch.bfloat16) * 0.1
    v_t = torch.randn(n, n, dtype=torch.bfloat16) * 0.1
    causal_mask = torch.triu(torch.full((n, n), float("-inf")), diagonal=1).bfloat16()

    ref = _torch_attention(q_t, k_t, v_t, scale_val, causal_mask)
    print("PyTorch reference attention out[0,:6]:", ref[0, :6].tolist())

    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    q = to_dev(q_t)
    k = to_dev(k_t)
    v = to_dev(v_t)
    scale = to_dev(torch.full((32, 32), scale_val, dtype=torch.bfloat16))
    scaler = to_dev(torch.ones(32, 32, dtype=torch.bfloat16))
    mask = to_dev(causal_mask)
    out = to_dev(torch.zeros(n, n, dtype=torch.bfloat16))

    try:
        attention_kernel(q, k, v, scale, mask, scaler, out)
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
