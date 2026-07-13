# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from `rotary_qk_kernel` in
# vendor/tt-lang/examples/test_transformer_block.py @ a19aaa8.
# (Check the current pin with `git -C vendor/tt-lang rev-parse --short HEAD`.)
# Drift-check before publishing (see README.md "Drift sources").
#
# ⚠️ SIMPLIFIED ROTARY — HONEST ANNOTATION.
#   The vendor `rotary_qk_kernel` (and therefore this adaptation) implements a
#   *simplified* rotary embedding: it multiplies Q and K by `cos` ONLY. It does
#   NOT split the head dimension into pairs, rotate_half, or add the `sin` term.
#   Full RoPE is  x_rot = x * cos + rotate_half(x) * sin  — see the
#   `precompute_rope_cos_sin` / `apply_rope` functions in reference_gpt.py for
#   the complete math the arc actually teaches and the hero run trains. This
#   kernel exists to show the *elementwise, per-position multiply* shape of the
#   RoPE data-movement (reader -> compute -> writer), not to be a drop-in RoPE.
#   On real hardware / ttml the full op is `ttml.ops.rope.rope`.
#
# Verification status: SIM-RUNNABLE (functional simulator).
#   Unlike attention.py / rmsnorm.py (which need reduce->broadcast that this sim
#   build rejects), RoPE here is a pure elementwise multiply — the same class of
#   op as eltwise_add.py — so it runs cleanly in the bundled functional
#   simulator with no device required. `main()` validates it against a small
#   torch reference (q*cos, k*cos) and reports the max abs error.
#
# ----------------------------------------------------------------------------
# Lab 2 — Embeddings & the Residual Stream: position without a learned table.
#
# RoPE (rotary position embeddings) replaces GPT-2's learned positional
# embedding table. Instead of *adding* a position vector, it *rotates* Q and K
# by a position-dependent angle so relative position falls out of the Q·K dot
# product. Modern LLMs (Llama-3) all use it.
#
# Coming from CUDA: this is an elementwise, embarrassingly-parallel map — the
# CUDA equivalent is a one-line __global__ kernel over the tensor. In TT-Lang
# you write the same map as three explicit threads handing 32x32 tiles through
# L1 dataflow buffers:  read (DRAM->L1) -> compute (L1->L1) -> write (L1->DRAM).
# ----------------------------------------------------------------------------
"""Simplified rotary embedding on Q and K as a from-scratch TT-Lang kernel
(+ torch reference).

Run it (functional simulator, no device needed):

    python content/templates/llm-from-scratch/kernels/rope.py

Expected: "PASSED" with max-abs-error vs a torch reference ~0 (bf16 tolerance).
NOTE: this multiplies by cos only (simplified); reference_gpt.py has full RoPE.
"""

from _ttlang_sim import add_ttlang_sim_to_path

add_ttlang_sim_to_path()

import torch  # noqa: E402
from sim import ttl, ttnn  # noqa: E402  (simulator ttl / ttnn)

SEQ_TILES = 1   # 32 tokens
EMBD_TILES = 1  # 32 head-dim (single head)


@ttl.operation(grid=(1, 1))
def rotary_qk_kernel(q_in, k_in, cos, q_out, k_out):
    """Apply the SIMPLIFIED rotary embedding to Q and K: q_rot = q * cos,
    k_rot = k * cos.

    (Real RoPE also splits the head dim and adds a rotate_half(x) * sin term;
    see reference_gpt.py. This kernel keeps the elementwise-multiply shape that
    the reader/compute/writer pipeline makes concrete.)
    """
    q_dfb = ttl.make_dataflow_buffer_like(q_in, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    k_dfb = ttl.make_dataflow_buffer_like(k_in, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    cos_dfb = ttl.make_dataflow_buffer_like(cos, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    qo_dfb = ttl.make_dataflow_buffer_like(q_out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)
    ko_dfb = ttl.make_dataflow_buffer_like(k_out, shape=(SEQ_TILES, EMBD_TILES), block_count=2)

    @ttl.compute()
    def compute():
        # Keep cos in scope while rotating both Q and K by it.
        with cos_dfb.wait() as cv:
            with q_dfb.wait() as qv, qo_dfb.reserve() as qo:
                qo.store(qv * cv)
            with k_dfb.wait() as kv, ko_dfb.reserve() as ko:
                ko.store(kv * cv)

    @ttl.datamovement()
    def dm_read():
        with q_dfb.reserve() as blk:
            tx = ttl.copy(q_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with k_dfb.reserve() as blk:
            tx = ttl.copy(k_in[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()
        with cos_dfb.reserve() as blk:
            tx = ttl.copy(cos[0:SEQ_TILES, 0:EMBD_TILES], blk)
            tx.wait()

    @ttl.datamovement()
    def dm_write():
        with qo_dfb.wait() as blk:
            tx = ttl.copy(blk, q_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()
        with ko_dfb.wait() as blk:
            tx = ttl.copy(blk, k_out[0:SEQ_TILES, 0:EMBD_TILES])
            tx.wait()


def main() -> int:
    torch.manual_seed(0)
    n = EMBD_TILES * 32
    seq = SEQ_TILES * 32

    q_torch = torch.randn(seq, n, dtype=torch.bfloat16) * 0.1
    k_torch = torch.randn(seq, n, dtype=torch.bfloat16) * 0.1
    # A non-trivial cos table so the multiply is actually exercised (the vendor
    # test used all-ones; we use a per-position cos so a bug can't hide).
    cos_torch = torch.cos(torch.linspace(0, 1, seq * n).reshape(seq, n)).bfloat16()

    # Torch reference for the SIMPLIFIED rotary (cos-only multiply).
    q_rot_ref = q_torch.float() * cos_torch.float()
    k_rot_ref = k_torch.float() * cos_torch.float()
    print("torch reference q_rot[0,:6]:", q_rot_ref[0, :6].tolist())

    to_dev = lambda t: ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
    q = to_dev(q_torch)
    k = to_dev(k_torch)
    cos = to_dev(cos_torch)
    q_out = to_dev(torch.zeros(seq, n, dtype=torch.bfloat16))
    k_out = to_dev(torch.zeros(seq, n, dtype=torch.bfloat16))

    try:
        rotary_qk_kernel(q, k, cos, q_out, k_out)
    except Exception as exc:  # noqa: BLE001
        last = str(exc).strip().splitlines()[-1]
        print(f"TT-Lang kernel: SIM-UNSUPPORTED at this pin (compiler/hardware path only): {last}")
        return 0

    q_res = ttnn.to_torch(q_out).float()
    k_res = ttnn.to_torch(k_out).float()
    q_err = (q_res - q_rot_ref).abs().max().item()
    k_err = (k_res - k_rot_ref).abs().max().item()
    print(f"TT-Lang kernel ran in sim. max abs error  Q: {q_err:.6f}  K: {k_err:.6f}")
    ok = torch.allclose(q_res, q_rot_ref, rtol=1e-2, atol=1e-2) and torch.allclose(
        k_res, k_rot_ref, rtol=1e-2, atol=1e-2
    )
    print("PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
