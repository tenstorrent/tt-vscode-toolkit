"""Phase 1: AnimateDiff baseline using diffusers AnimateDiffPipeline + MotionAdapter.

Uses CompVis/stable-diffusion-v1-4 with guoyww/animatediff-motion-adapter-v1-5-2.
The MotionAdapter injects temporal attention inside each SD 1.4 UNet transformer
block at 320-dim features — the correct location for mm_sd_v15_v2.ckpt weights.
No TT hardware required.
"""

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter


def create_animatediff_pipeline(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype: torch.dtype = torch.float32,
) -> AnimateDiffPipeline:
    """Create a diffusers AnimateDiffPipeline with MotionAdapter.

    Downloads from HuggingFace cache on first call. No TT hardware required.
    Run `hf download CompVis/stable-diffusion-v1-4` and
    `hf download guoyww/animatediff-motion-adapter-v1-5-2` beforehand.

    Returns:
        diffusers.AnimateDiffPipeline ready for inference
    """
    adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch_dtype)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    return pipe


def generate(
    pipe: AnimateDiffPipeline,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = 42,
) -> List[Image.Image]:
    """Generate an animated sequence from a text prompt.

    Args:
        pipe: AnimateDiffPipeline from create_animatediff_pipeline()
        prompt: Text prompt describing the animation
        negative_prompt: What to avoid in the output
        num_frames: Number of frames (8 or 16 recommended)
        guidance_scale: CFG scale — higher = more prompt-adherent (7.5 is standard)
        num_inference_steps: Denoising steps (25 balances speed and quality)
        seed: Random seed for reproducibility

    Returns:
        List of PIL Images, one per frame, 512x512
    """
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    )
    return output.frames[0]


def export_gif(frames: List[Image.Image], output_path: str, fps: int = 8) -> None:
    """Save a list of PIL Images as an animated GIF.

    Args:
        frames: List of PIL Images (all same size)
        output_path: Destination file path, e.g. 'output/result.gif'
        fps: Frames per second (duration = 1000 // fps ms per frame)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
