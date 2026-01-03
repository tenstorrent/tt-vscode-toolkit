#!/usr/bin/env python3
"""
Full 16-frame video generation example using AnimateDiff + SD 3.5

This demonstrates the complete AnimateDiff experience:
    - Generate 16-frame animated sequence
    - Apply temporal attention for smooth motion
    - Export to MP4 video

Expected result:
    - Smooth animated video with temporal coherence
    - Objects move realistically across frames
    - Camera or object motion follows physical laws

Usage:
    python examples/generate_16frame_video.py

Note: This example shows the architecture. Full integration with SD 3.5
      pipeline is needed for actual image generation.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import numpy as np

from animatediff_ttnn import create_animatediff_pipeline


def create_test_frames(num_frames: int, size: int = 512) -> list[Image.Image]:
    """Create test frames with synthetic motion for demo purposes.

    In real usage, these would come from SD 3.5 VAE decoding.
    """
    frames = []

    for i in range(num_frames):
        # Create gradient that moves across frames
        img_array = np.zeros((size, size, 3), dtype=np.uint8)

        # Horizontal gradient that shifts
        for y in range(size):
            for x in range(size):
                # Shift the pattern based on frame number
                value = int(((x + i * 20) % size) / size * 255)
                img_array[y, x] = [value, 128, 255 - value]

        frames.append(Image.fromarray(img_array))

    return frames


def generate_16frame_demo():
    """Generate a 16-frame animated sequence with temporal coherence."""

    print("=" * 60)
    print("AnimateDiff 16-Frame Video Generation Demo")
    print("=" * 60)
    print()

    # Configuration
    num_frames = 16
    height = 512
    width = 512
    latent_height = height // 8
    latent_width = width // 8
    channels = 320  # Hidden dimension (NOT latent channels!)

    print(f"Configuration:")
    print(f"  Frames: {num_frames}")
    print(f"  Resolution: {height}x{width}")
    print(f"  Latent size: {latent_height}x{latent_width}")
    print(f"  Hidden dimension: {channels} (NOT latent dim=16)")
    print()

    # Step 1: Create AnimateDiff pipeline
    print("Step 1: Loading AnimateDiff temporal module...")
    pipeline = create_animatediff_pipeline(
        temporal_checkpoint="~/models/animatediff/mm_sd_v15_v2.ckpt"
    )
    print()

    # Step 2: Prepare latents for video
    print("Step 2: Preparing video latents...")
    latents = pipeline.prepare_video_latents(
        batch_size=1,
        num_frames=num_frames,
        height=latent_height,
        width=latent_width,
        num_channels=channels,
        dtype=torch.float32,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Latent shape: {latents.shape}")
    print(f"  Total frames: {num_frames}")
    print()

    # Step 3: Apply temporal coherence
    print("Step 3: Applying temporal attention across 16 frames...")
    print("  This creates smooth motion by attending across the temporal dimension")

    latents_coherent = pipeline.apply_temporal_coherence(
        latents,
        num_frames=num_frames,
    )

    print(f"  ✓ Temporal coherence applied")
    print()

    # Step 4: Analyze temporal consistency
    print("Step 4: Analyzing temporal consistency...")

    correlations = []
    for i in range(num_frames - 1):
        frame_a = latents_coherent[i].flatten()
        frame_b = latents_coherent[i + 1].flatten()
        corr = torch.corrcoef(torch.stack([frame_a, frame_b]))[0, 1].item()
        correlations.append(corr)

    avg_correlation = sum(correlations) / len(correlations)
    print(f"  Average frame-to-frame correlation: {avg_correlation:.4f}")

    if avg_correlation > 0.6:
        print(f"  ✓ Strong temporal coherence (smooth motion expected)")
    elif avg_correlation > 0.4:
        print(f"  ~ Moderate temporal coherence")
    else:
        print(f"  ⚠ Weak temporal coherence (motion may be jerky)")

    print()

    # Step 5: Create test video
    print("Step 5: Creating test video...")
    print("  (Using synthetic frames for demo - replace with SD 3.5 decoding)")

    test_frames = create_test_frames(num_frames, size=512)
    output_path = Path(__file__).parent.parent / "output" / "test_16frame.gif"
    output_path.parent.mkdir(exist_ok=True)

    pipeline.export_video(test_frames, str(output_path), fps=8)
    print()

    # Step 6: Summary
    print("=" * 60)
    print("16-Frame Video Generation Complete!")
    print("=" * 60)
    print()
    print("Results:")
    print(f"  Total frames: {num_frames}")
    print(f"  Temporal coherence: {avg_correlation:.4f}")
    print(f"  Output: {output_path}")
    print()
    print("What happens in full pipeline:")
    print("  1. Generate initial noise latents (16 frames)")
    print("  2. Denoise with SD 3.5 transformer")
    print("  3. Apply temporal attention after each denoising step")
    print("  4. Decode latents with VAE (frame by frame)")
    print("  5. Export to MP4 video")
    print()
    print("Next steps:")
    print("  • Integrate with full SD 3.5 pipeline")
    print("  • Test on actual prompts (e.g., 'butterfly landing')")
    print("  • Optimize for N150 memory constraints")
    print("  • Export high-quality MP4 videos")
    print()


if __name__ == "__main__":
    generate_16frame_demo()
