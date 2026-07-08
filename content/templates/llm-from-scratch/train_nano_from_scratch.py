# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# Lab 5 — Train It & Run for Real: from-scratch training on Blackhole.
#
# This is a THIN runner. It does NOT reinvent the model. It imports the model
# and the training primitives from the canonical, verified upstream example:
#
#     $TT_METAL_HOME/tt-train/sources/examples/nano_gpt/
#         nanogpt_primitives_example.py
#
# ...which builds NanoGPT purely from `ttml.ops` + ttnn primitives (embedding,
# linear, multi-head attention, layernorm, cross-entropy, AdamW) and runs a
# real forward + backward + optimizer step on the device. We reuse that code
# verbatim and drive a short, explicit training loop with the nano config so a
# reader can watch the loss drop live.
#
# PREREQUISITE: `ttml` must be built from a tt-metal source tree and importable.
# See BUILD_TTML.md in this directory for the verified recipe (including the
# `std::bad_cast` ABI fix). There is NO pip wheel for ttml.
#
# Verified on 2026-07-08 on a Blackhole p300c against tt-metal v0.73: a real
# forward+backward+AdamW loop trained on-device, loss dropping ~4.6 -> ~3.3
# over 10 steps, exit 0. Upstream does NOT CI training on Blackhole, so pin your
# tt-metal version and reset the board (`tt-smi -r`) if device open times out.
#
# Run (Blackhole p300c):
#
#     TT_METAL_HOME=/home/ttuser/tt-metal \
#     TT_METAL_RUNTIME_ROOT=/home/ttuser/tt-metal \
#     TT_METAL_ARCH_NAME=blackhole TT_LOGGER_LEVEL=FATAL \
#     python content/templates/llm-from-scratch/train_nano_from_scratch.py \
#         --max_steps 10 --batch_size 2 --data_path /path/to/shakespeare.txt
#
# (On Wormhole use TT_METAL_ARCH_NAME=wormhole_b0.)
# ============================================================================
"""Minimal from-scratch NanoGPT training runner (ttml, on-device)."""

import argparse
import importlib.util
import os
import sys
import time


# --- Nano baseline config (from tt-train/configs/model_configs/nanogpt.yaml) --
# Same shape the verified 4.6 -> 3.3 run used. Scaling to the ~80M "hero" model
# is the same code with bigger knobs (see the README / Lab 5 scaling math).
NANO_EMBEDDING_DIM = 384
NANO_NUM_HEADS = 6
NANO_NUM_BLOCKS = 6
NANO_SEQ_LEN = 256
NANO_DROPOUT = 0.2


def _default_tt_metal_home() -> str:
    return os.environ.get("TT_METAL_HOME", "/home/ttuser/tt-metal")


def _ensure_env(tt_metal_home: str) -> None:
    """Set the env vars ttml/ttnn need if the caller didn't.

    The upstream example aborts immediately without TT_METAL_RUNTIME_ROOT, and
    Blackhole requires TT_METAL_ARCH_NAME=blackhole. We honour anything the user
    already exported (the `:=` semantics) and only fill gaps.
    """
    os.environ.setdefault("TT_METAL_HOME", tt_metal_home)
    os.environ.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    # Guard pattern from the repo CLAUDE.md: default to wormhole_b0 unless the
    # user (or their shell) already set an arch. On this Blackhole box you must
    # pass TT_METAL_ARCH_NAME=blackhole.
    os.environ.setdefault("TT_METAL_ARCH_NAME", "wormhole_b0")
    os.environ.setdefault("TT_LOGGER_LEVEL", "FATAL")


def _load_example_module(tt_metal_home: str):
    """Import the canonical nanogpt_primitives_example.py by path.

    We reuse its PrimitiveNanoGPT model and training primitives rather than
    reimplementing them, so this runner never drifts from the verified source.
    """
    example_dir = os.path.join(
        tt_metal_home, "tt-train", "sources", "examples", "nano_gpt"
    )
    example_path = os.path.join(example_dir, "nanogpt_primitives_example.py")
    if not os.path.isfile(example_path):
        raise FileNotFoundError(
            f"Canonical example not found at {example_path}.\n"
            "Set TT_METAL_HOME to a tt-metal source tree and build ttml "
            "(see BUILD_TTML.md)."
        )
    # The example dir is not a package; add it so relative helpers resolve.
    if example_dir not in sys.path:
        sys.path.insert(0, example_dir)
    spec = importlib.util.spec_from_file_location(
        "nanogpt_primitives_example", example_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thin from-scratch NanoGPT training runner (ttml, on-device)."
    )
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to a plain-text corpus (char-level tokenized).")
    parser.add_argument("--max_steps", type=int, default=10,
                        help="Number of optimizer steps to run (default: 10).")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=NANO_SEQ_LEN)
    parser.add_argument("--embedding_dim", type=int, default=NANO_EMBEDDING_DIM)
    parser.add_argument("--num_heads", type=int, default=NANO_NUM_HEADS)
    parser.add_argument("--num_blocks", type=int, default=NANO_NUM_BLOCKS)
    parser.add_argument("--dropout", type=float, default=NANO_DROPOUT)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=5489)
    parser.add_argument("--tt_metal_home", type=str, default=_default_tt_metal_home())
    args = parser.parse_args()

    _ensure_env(args.tt_metal_home)

    # Import ttml/ttnn and the verified example only AFTER env is set.
    import ttml  # noqa: E402
    import ttnn  # noqa: E402
    from ttml.common.utils import round_up_to_tile  # noqa: E402

    nge = _load_example_module(args.tt_metal_home)

    print("=" * 70)
    print("NanoGPT from-scratch training (thin runner over the verified example)")
    print(f"  arch={os.environ.get('TT_METAL_ARCH_NAME')}  "
          f"runtime_root={os.environ.get('TT_METAL_RUNTIME_ROOT')}")
    print("=" * 70)

    # 1. Data -----------------------------------------------------------------
    if not os.path.isfile(args.data_path):
        print(f"ERROR: data file not found: {args.data_path}")
        return 1
    text = nge.read_file_to_str(args.data_path)
    dataset, tokenizer = nge.create_dataset_from_text(text, args.sequence_length)
    vocab_size = round_up_to_tile(tokenizer.vocab_size, 32)
    print(f"1. Data: {len(dataset)} samples, vocab={tokenizer.vocab_size} "
          f"(padded {vocab_size}), seq_len={args.sequence_length}")

    # 2. Device ---------------------------------------------------------------
    instance = ttml.autograd.AutoContext.get_instance()
    instance.open_device()
    instance.get_device()
    instance.set_seed(args.seed)

    try:
        # 3. Model (built from ttml.ops in the reused example code) -----------
        config = nge.PrimitiveNanoGPTConfig(
            vocab_size=vocab_size,
            block_size=args.sequence_length,
            n_embd=args.embedding_dim,
            n_layer=args.num_blocks,
            n_head=args.num_heads,
            dropout=args.dropout,
            bias=True,
        )
        model = nge.PrimitiveNanoGPT(config)
        total_params = sum(
            p.to_numpy(ttnn.DataType.FLOAT32).size for p in model.parameters().values()
        )
        print(f"2. Model: {args.num_blocks} blocks, {args.embedding_dim} embd, "
              f"{args.num_heads} heads -> {total_params:,} params")

        # 4. Optimizer: AdamW (cross-entropy loss lives inside train_step) ----
        optimizer = ttml.optimizers.create_optimizer(
            {
                "type": "AdamW",
                "lr": args.lr,
                "beta1": 0.9,
                "beta2": 0.999,
                "epsilon": 1.0e-8,
                "weight_decay": 0.01,
                "amsgrad": False,
            },
            model.parameters(),
        )
        print(f"3. Optimizer: {optimizer.get_name()} (lr={optimizer.get_lr()})")

        # 5. Causal mask + training loop --------------------------------------
        mask = nge.create_causal_mask_tensor(args.sequence_length)
        grad_accum = nge.GradientAccumulator(1)  # no accumulation for the demo

        model.train()
        print(f"4. Training for {args.max_steps} steps (batch_size={args.batch_size})")
        print()

        losses = []
        step = 0
        start = time.time()
        # Single pass over the data is plenty for a short from-scratch demo.
        for batch_start in range(0, len(dataset), args.batch_size):
            if step >= args.max_steps:
                break
            batch = dataset[batch_start : batch_start + args.batch_size]
            if len(batch) < args.batch_size:
                break
            input_tokens, target_tokens = nge.collate_fn(
                batch, args.batch_size, args.sequence_length
            )
            loss_float, step_ms, _ = nge.train_step(
                model, optimizer, None, step,
                input_tokens, target_tokens, mask,
                grad_accum, False, 1.0, batch_size=args.batch_size,
            )
            grad_accum.reset()
            losses.append(loss_float)
            print(f"Step: {step:>3}  Loss: {loss_float:.6f}  Time: {step_ms:.1f} ms")
            step += 1

        total = time.time() - start
        print()
        print("=" * 70)
        print("Training completed")
        if losses:
            print(f"  first-step loss: {losses[0]:.4f}   last-step loss: {losses[-1]:.4f}")
        print(f"  steps: {step}   total time: {total:.1f} s")
        print("=" * 70)
    finally:
        # Let ttml close the device cleanly (a killed/partial run triggers a
        # benign teardown abort in MetalContext::destroy_all_instances).
        instance.close_device()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
