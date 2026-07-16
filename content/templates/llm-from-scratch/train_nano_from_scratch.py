# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ============================================================================
# Lab 5 — Train It & Run for Real: the modern Llama-3 from-scratch training run.
#
# This is a THIN, documented runner. It does NOT reinvent the model or the
# training loop — it drives the VERIFIED upstream Llama trainer:
#
#     $TT_METAL_HOME/tt-train/sources/examples/nano_gpt/train_nanogpt.py \
#         --config training_shakespeare_nanollama3_char.yaml
#
# ...with the env vars the run requires, and passes through --max_steps /
# --data_path. That config selects `model_type: llama`, i.e. the modern stack
# the whole arc teaches:
#     RoPE (theta 500000) + RMSNorm + Grouped-Query Attention (6 heads / 3 KV
#     groups) + SwiGLU MLP, embedding_dim 384, 6 blocks, seq 256, char tokenizer
# built from `ttml.models.llama` (transformer.py / gqattn.py).
#
# WHY A DIFFERENT SHAPE THAN THE GPT-2 PATH: unlike GPT-2 (which ships a
# single-file `nanogpt_primitives_example.py` that hand-assembles the model
# from `ttml.ops` primitives), there is NO single-file Llama "primitives"
# example. The canonical, supported way to train the Llama stack from scratch is
# `train_nanogpt.py` + the `training_shakespeare_nanollama3_char.yaml` config
# driving `ttml.models.llama`. So this runner is a launcher around that command,
# not a primitives reimplementation. (reference_gpt.py is the pure-PyTorch
# mirror of the same components, for the "understand" half of the labs.)
#
# PREREQUISITE: `ttml` must be built from a tt-metal source tree and importable.
# See BUILD_TTML.md in this directory for the verified recipe (including the
# `std::bad_cast` ABI fix). There is NO pip wheel for ttml.
#
# VERIFIED on 2026-07-08 on a Blackhole p300c against tt-metal v0.73:
#     train_nanogpt.py --config training_shakespeare_nanollama3_char.yaml \
#         --max_steps 20
#   -> loss 4.69 -> 3.23 over 20 steps, ~65 ms/step, 16.5 TFLOPS, MFU ~11%,
#      exit 0. This is the arc's HERO run. Upstream does NOT CI Llama training
#   on Blackhole, so pin your tt-metal version and reset the board (`tt-smi -r`)
#   if device open times out.
#
# Run (Blackhole p300c):
#
#     python content/templates/llm-from-scratch/train_nano_from_scratch.py \
#         --max_steps 20 --data_path ~/tt-metal/tt-train/data/shakespeare.txt
#
# (Arch resolution: an explicit --arch always wins; otherwise the runner
#  honours an already-exported TT_METAL_ARCH_NAME; otherwise it defaults to
#  blackhole. Pass --arch wormhole_b0 on N-series, or export
#  TT_METAL_ARCH_NAME=wormhole_b0 before running.)
# ============================================================================
"""Thin launcher for the verified nanollama3 from-scratch training run."""

import argparse
import os
import subprocess
import sys


# The training config that selects the modern Llama-3 stack (model_type=llama).
# train_nanogpt.py resolves --config relative to
# $TT_METAL_HOME/tt-train/configs/training_configs, so we pass the bare name.
NANOLLAMA3_CONFIG = "training_shakespeare_nanollama3_char.yaml"


def _default_tt_metal_home() -> str:
    return os.environ.get("TT_METAL_HOME", os.path.expanduser("~/tt-metal"))


def _build_env(tt_metal_home: str, arch: str | None) -> dict:
    """Return an environment dict with the vars ttml/ttnn need.

    - TT_METAL_HOME / TT_METAL_RUNTIME_ROOT: train_nanogpt.py aborts immediately
      without TT_METAL_RUNTIME_ROOT (it needs it to find runtime kernels).
    - TT_METAL_ARCH_NAME: `blackhole` for P-series, `wormhole_b0` for N-series.
      An explicit --arch (arch is not None) always wins. Otherwise we honour
      an already-exported TT_METAL_ARCH_NAME. Otherwise we default to
      `blackhole`, mirroring the repo's
      `: "${TT_METAL_ARCH_NAME:=wormhole_b0}"` guard pattern (default-if-unset).
    - TT_LOGGER_LEVEL=FATAL: keep the on-device log quiet.
    We honour anything the caller already exported and only fill in gaps.
    """
    env = dict(os.environ)
    env.setdefault("TT_METAL_HOME", tt_metal_home)
    env.setdefault("TT_METAL_RUNTIME_ROOT", tt_metal_home)
    if arch is not None:
        env["TT_METAL_ARCH_NAME"] = arch  # explicit --arch overrides everything
    else:
        env.setdefault("TT_METAL_ARCH_NAME", "blackhole")  # honour export, else default
    env.setdefault("TT_LOGGER_LEVEL", "FATAL")
    return env


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thin launcher for the verified nanollama3 (Llama-3) "
                    "from-scratch training run on Tenstorrent hardware."
    )
    parser.add_argument("--data_path", type=str, default="",
                        help="Path to a plain-text corpus (char-tokenized). "
                             "Defaults to the config's data/shakespeare.txt.")
    parser.add_argument("--max_steps", type=int, default=20,
                        help="Number of optimizer steps (default: 20, the "
                             "verified hero-run length).")
    parser.add_argument("--arch", type=str, default=None,
                        choices=["blackhole", "wormhole_b0"],
                        help="TT_METAL_ARCH_NAME. If omitted, honours an "
                             "already-exported TT_METAL_ARCH_NAME, else "
                             "defaults to blackhole (for p300c).")
    parser.add_argument("--tt_metal_home", type=str, default=_default_tt_metal_home())
    parser.add_argument("--dry_run", action="store_true",
                        help="Print the command and environment, then exit "
                             "without launching (no device needed).")
    args = parser.parse_args()

    tt_metal_home = args.tt_metal_home
    train_script = os.path.join(
        tt_metal_home, "tt-train", "sources", "examples", "nano_gpt", "train_nanogpt.py"
    )
    if not os.path.isfile(train_script):
        print(f"ERROR: canonical trainer not found at {train_script}")
        print("Set --tt_metal_home / TT_METAL_HOME to a tt-metal source tree "
              "and build ttml (see BUILD_TTML.md).")
        return 1

    cmd = [
        sys.executable, train_script,
        "--config", NANOLLAMA3_CONFIG,
        "--max_steps", str(args.max_steps),
    ]
    if args.data_path:
        cmd += ["--data_path", args.data_path]

    env = _build_env(tt_metal_home, args.arch)

    print("=" * 70)
    print("nanollama3 from-scratch training (thin launcher over train_nanogpt.py)")
    print(f"  model: Llama-3 stack (RoPE theta=500000 + RMSNorm + GQA 6h/3kv + SwiGLU)")
    print(f"  arch={env['TT_METAL_ARCH_NAME']}  "
          f"runtime_root={env['TT_METAL_RUNTIME_ROOT']}")
    print(f"  config={NANOLLAMA3_CONFIG}  max_steps={args.max_steps}")
    print("  command:")
    print("    " + " ".join(cmd))
    print("=" * 70)

    if args.dry_run:
        print("--dry_run set: not launching.")
        return 0

    # Launch the verified trainer. Its own device open/close + training loop
    # run in the child process (it closes the device cleanly on exit, which
    # avoids the benign MetalContext::destroy_all_instances teardown abort).
    completed = subprocess.run(cmd, env=env)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
