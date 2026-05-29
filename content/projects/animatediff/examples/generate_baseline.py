#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Phase 1 AnimateDiff baseline — correct temporal attention on CPU.

Uses diffusers AnimateDiffPipeline with MotionAdapter. The MotionAdapter
injects temporal attention inside each SD 1.4 UNet transformer block at
320-dim features, which is exactly where mm_sd_v15_v2.ckpt weights operate.
No TT hardware required.

Setup:
    pip install -r requirements.txt
    hf download CompVis/stable-diffusion-v1-4
    hf download guoyww/animatediff-motion-adapter-v1-5-2

Usage:
    python examples/generate_baseline.py
    python examples/generate_baseline.py --prompt "ocean waves" --frames 8 --steps 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Phase 1 — CPU baseline")
    parser.add_argument("--prompt", default="a campfire with crackling flames, cinematic, 4K")
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted", dest="negative_prompt")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames (8 or 16 recommended)")
    parser.add_argument("--steps", type=int, default=25, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/baseline.gif")
    args = parser.parse_args()

    print("AnimateDiff Phase 1 — CPU baseline")
    print(f"  Base model : CompVis/stable-diffusion-v1-4")
    print(f"  Adapter    : guoyww/animatediff-motion-adapter-v1-5-2")
    print(f"  Prompt     : {args.prompt}")
    print(f"  Frames     : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print()

    print("Loading AnimateDiff pipeline (first run downloads ~4.7 GB)...")
    t0 = time.time()
    pipe = create_animatediff_pipeline()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    print(f"Generating {args.frames} frames...")
    t1 = time.time()
    frames = generate(
        pipe,
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)")
    print()

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames → {args.output}")
    print()
    print("What AnimateDiff did:")
    print("  MotionAdapter injected temporal attention into each UNet transformer block.")
    print("  During denoising, each step attends across all frames simultaneously —")
    print("  the motion weights (mm_sd_v15_v2.ckpt) ensure coherent motion.")
    print()
    print("Phase 2 (Blackhole hardware): see examples/generate_blackhole.py")


if __name__ == "__main__":
    main()
