"""AnimateDiff for Tenstorrent hardware.

Phase 1 (CPU baseline):
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif

Phase 2 (Blackhole hardware):
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, build_tlist, generate_frames
"""

__version__ = "0.2.0"

from .pipeline import create_animatediff_pipeline, generate, export_gif

__all__ = [
    "create_animatediff_pipeline",
    "generate",
    "export_gif",
]
