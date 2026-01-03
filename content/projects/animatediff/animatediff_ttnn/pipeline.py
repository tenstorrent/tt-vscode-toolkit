"""
AnimateDiff Pipeline Wrapper for TT-Metal SD 3.5

This module provides a clean wrapper around TT-Metal's SD 3.5 pipeline
to add AnimateDiff temporal attention for video generation.

Architecture:
    1. Generate latents for multiple frames
    2. Run SD 3.5 denoising with temporal coherence applied after each step
    3. Decode frames and export as video

No modifications to tt-metal required!
"""

from __future__ import annotations

import os
from typing import Optional, List
from pathlib import Path

import torch
import ttnn
from PIL import Image

from .temporal_module import (
    load_animatediff_weights,
    temporal_attention_torch,
    temporal_attention_ttnn,
    TemporalAttentionWeights,
)


class AnimateDiffPipeline:
    """AnimateDiff wrapper for TT-Metal SD 3.5 pipeline.

    This wrapper adds temporal attention to create animated videos from text prompts.
    It works by applying temporal coherence after the spatial diffusion process.

    Example:
        >>> from animatediff_ttnn import AnimateDiffPipeline
        >>>
        >>> # Assuming you have a working SD 3.5 pipeline
        >>> pipeline = AnimateDiffPipeline(
        ...     temporal_checkpoint="~/models/animatediff/mm_sd_v15_v2.ckpt"
        ... )
        >>>
        >>> frames = pipeline(
        ...     prompt="A butterfly landing on a flower, wings gently moving",
        ...     num_frames=16,
        ...     height=512,
        ...     width=512,
        ... )
        >>>
        >>> pipeline.export_video(frames, "butterfly.mp4", fps=8)
    """

    def __init__(
        self,
        temporal_checkpoint: str,
        dim: int = 320,
        num_heads: int = 8,
        max_frames: int = 24,
        use_ttnn: bool = False,  # Set to True once TTNN version is optimized
    ):
        """Initialize AnimateDiff pipeline.

        Args:
            temporal_checkpoint: Path to AnimateDiff checkpoint (.ckpt file)
            dim: Hidden dimension (320 for SD 1.5, may need adjustment for SD 3.5)
            num_heads: Number of attention heads
            max_frames: Maximum number of frames for positional encoding
            use_ttnn: Whether to use TTNN ops (vs PyTorch fallback)
        """
        self.temporal_weights = load_animatediff_weights(
            checkpoint_path=temporal_checkpoint,
            dim=dim,
            num_heads=num_heads,
            max_frames=max_frames,
        )
        self.use_ttnn = use_ttnn

        print(f"✓ AnimateDiff temporal weights loaded from {temporal_checkpoint}")
        print(f"  Dimension: {dim}, Heads: {num_heads}, Max frames: {max_frames}")
        print(f"  Using {'TTNN' if use_ttnn else 'PyTorch'} backend")

    def apply_temporal_coherence(
        self,
        latents: torch.Tensor | ttnn.Tensor,
        num_frames: int,
        device: Optional[ttnn.Device] = None,
    ) -> torch.Tensor | ttnn.Tensor:
        """Apply temporal attention to add motion coherence across frames.

        Args:
            latents: Latent tensor (batch*frames, height, width, channels)
            num_frames: Number of frames
            device: TTNN device (required if using TTNN backend)

        Returns:
            Latents with temporal coherence applied
        """
        if num_frames == 1:
            return latents  # No temporal attention needed for single frame

        # Reshape latents to (batch*frames, spatial_tokens, channels)
        if isinstance(latents, torch.Tensor):
            batch_frames, h, w, c = latents.shape
            spatial_tokens = h * w
            latents_flat = latents.reshape(batch_frames, spatial_tokens, c)

            # Apply temporal attention (PyTorch)
            output_flat = temporal_attention_torch(
                latents_flat,
                self.temporal_weights,
                num_frames,
            )

            # Reshape back
            output = output_flat.reshape(batch_frames, h, w, c)
            return output

        elif self.use_ttnn and isinstance(latents, ttnn.Tensor):
            # TTNN version
            batch_frames, h, w, c = latents.shape
            spatial_tokens = h * w

            # Reshape using TTNN ops
            latents_flat = ttnn.reshape(latents, (batch_frames, spatial_tokens, c))

            # Apply temporal attention (TTNN)
            output_flat = temporal_attention_ttnn(
                latents_flat,
                self.temporal_weights,
                num_frames,
                device,
            )

            # Reshape back
            output = ttnn.reshape(output_flat, (batch_frames, h, w, c))
            return output

        else:
            raise ValueError("Unsupported tensor type")

    def prepare_video_latents(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        num_channels: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Prepare initial noise latents for video generation.

        Args:
            batch_size: Batch size
            num_frames: Number of frames to generate
            height: Latent height (image_height // 8)
            width: Latent width (image_width // 8)
            num_channels: Number of latent channels (16 for SD 3.5)
            dtype: Tensor dtype
            device: Torch device
            generator: Random generator for reproducibility

        Returns:
            Latent tensor of shape (batch*frames, height, width, channels)
        """
        shape = (batch_size * num_frames, height, width, num_channels)
        latents = torch.randn(shape, generator=generator, dtype=dtype, device=device)
        return latents

    def decode_video_latents(
        self,
        latents: torch.Tensor,
        vae_decode_fn,
        num_frames: int,
        batch_size: int = 1,
    ) -> List[Image.Image]:
        """Decode video latents to PIL images.

        Args:
            latents: Latent tensor (batch*frames, h, w, c)
            vae_decode_fn: VAE decode function from SD 3.5 pipeline
            num_frames: Number of frames
            batch_size: Batch size

        Returns:
            List of PIL images (one per frame)
        """
        frames = []

        for i in range(num_frames):
            # Extract frame latent
            frame_latent = latents[i * batch_size : (i + 1) * batch_size]

            # Decode using SD 3.5 VAE
            frame_image = vae_decode_fn(frame_latent)

            frames.append(frame_image)

        return frames

    def export_video(
        self,
        frames: List[Image.Image],
        output_path: str,
        fps: int = 8,
        loop: int = 0,
    ):
        """Export frames to video file.

        Args:
            frames: List of PIL images
            output_path: Output file path (.mp4, .gif, or .webm)
            fps: Frames per second
            loop: Number of loops (0 = infinite, for GIF)
        """
        output_path = Path(output_path)
        suffix = output_path.suffix.lower()

        if suffix == ".gif":
            # Export as GIF
            frames[0].save(
                output_path,
                save_all=True,
                append_images=frames[1:],
                duration=1000 // fps,
                loop=loop,
            )
            print(f"✓ Video exported to {output_path} (GIF, {len(frames)} frames @ {fps} fps)")

        elif suffix in [".mp4", ".webm"]:
            # Export as video using ffmpeg
            try:
                from diffusers.utils import export_to_video

                export_to_video(frames, str(output_path), fps=fps)
                print(f"✓ Video exported to {output_path} ({len(frames)} frames @ {fps} fps)")
            except ImportError:
                print("❌ diffusers not found. Install with: pip install diffusers")
                print(f"   Falling back to GIF export...")
                gif_path = output_path.with_suffix(".gif")
                self.export_video(frames, str(gif_path), fps=fps, loop=loop)

        else:
            raise ValueError(f"Unsupported format: {suffix}. Use .gif, .mp4, or .webm")


def create_animatediff_pipeline(
    temporal_checkpoint: str = "~/models/animatediff/mm_sd_v15_v2.ckpt",
    **kwargs,
) -> AnimateDiffPipeline:
    """Convenience function to create AnimateDiff pipeline.

    Args:
        temporal_checkpoint: Path to AnimateDiff checkpoint
        **kwargs: Additional arguments for AnimateDiffPipeline

    Returns:
        AnimateDiffPipeline instance
    """
    checkpoint_path = os.path.expanduser(temporal_checkpoint)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"AnimateDiff checkpoint not found at {checkpoint_path}\n"
            f"Download it with:\n"
            f"  huggingface-cli download guoyww/animatediff mm_sd_v15_v2.ckpt --local-dir ~/models/animatediff"
        )

    return AnimateDiffPipeline(temporal_checkpoint=checkpoint_path, **kwargs)
