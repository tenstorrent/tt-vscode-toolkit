"""
AnimateDiff for TT-Metal

Native animated video generation on Tenstorrent hardware using AnimateDiff
temporal attention with Stable Diffusion 3.5.

Example:
    >>> from animatediff_ttnn import create_animatediff_pipeline
    >>>
    >>> pipeline = create_animatediff_pipeline()
    >>>
    >>> # Use with your SD 3.5 workflow
    >>> frames = pipeline.apply_temporal_coherence(latents, num_frames=16)
    >>> pipeline.export_video(frames, "output.mp4", fps=8)
"""

__version__ = "0.1.0"

from .pipeline import AnimateDiffPipeline, create_animatediff_pipeline
from .temporal_module import (
    load_animatediff_weights,
    temporal_attention_torch,
    temporal_attention_ttnn,
    TemporalAttentionWeights,
)

__all__ = [
    "AnimateDiffPipeline",
    "create_animatediff_pipeline",
    "load_animatediff_weights",
    "temporal_attention_torch",
    "temporal_attention_ttnn",
    "TemporalAttentionWeights",
]
