"""Phase 2: Blackhole-accelerated video frame generation using TTNN UNet.

Uses the SD 1.4 TTNN UNet from ~/tt-metal (same code runs on Blackhole via
TT_METAL_ARCH_NAME=blackhole). Frames denoised sequentially; temporal
coherence from shared base noise initialization (0.05 per-frame perturbation).

Documented tradeoff: this is TT-hardware-accelerated spatial denoising, not
full AnimateDiff temporal attention. Full integration would require injecting
TemporalTransformer blocks into the TTNN UNet transformer blocks.

Requirements:
    ~/tt-metal present and activated: source ~/tt-metal/python_env/bin/activate
    Blackhole hardware (P100 or P300c)
"""

import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image


TT_METAL_PATH = Path.home() / "tt-metal"


def _ensure_tt_metal_path() -> None:
    """Add ~/tt-metal to sys.path so SD demo module imports work."""
    tt_metal_str = str(TT_METAL_PATH)
    if tt_metal_str not in sys.path:
        if not TT_METAL_PATH.exists():
            raise RuntimeError(
                f"~/tt-metal not found. Activate the tt-metal environment first:\n"
                f"  cd ~/tt-metal && source python_env/bin/activate"
            )
        sys.path.insert(0, tt_metal_str)


def _constant_prop_time_embeddings(timesteps, sample, time_proj):
    """Compute time embeddings for a scalar timestep.

    Equivalent to the function defined in SD demo.py. Accepts a scalar
    timestep tensor, expands to batch size, runs through UNet time_proj.
    """
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    return time_proj(timesteps)


def setup_blackhole():
    """Open a Blackhole TTNN device with SD-appropriate L1 config.

    Sets TT_METAL_ARCH_NAME=blackhole if not already set (os.environ.setdefault,
    so an existing value is never overwritten). Returns the open TTNN device.
    """
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    _ensure_tt_metal_path()

    import ttnn
    from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE

    return ttnn.open_device(device_id=0, l1_small_size=SD_L1_SMALL_SIZE)


def build_tlist(ttnn_scheduler, torch_time_proj, device, latent_h: int = 64, latent_w: int = 64) -> list:
    """Build pre-computed time embeddings for each denoising timestep.

    sd_helper_funcs.run() expects _tlist[i] to be constant_prop_time_embeddings(t_i)
    already converted to a TTNN bfloat16 tensor on device, shape permuted for UNet.

    Args:
        ttnn_scheduler: TtPNDMScheduler with set_timesteps() already called
        torch_time_proj: unet.time_proj from the PyTorch UNet2DConditionModel
        device: TTNN device from setup_blackhole()
        latent_h: Latent height (image_height // 8; 64 for 512px images)
        latent_w: Latent width (image_width // 8; 64 for 512px images)

    Returns:
        List of TTNN tensors, one per timestep
    """
    import ttnn

    dummy = ttnn.from_torch(
        torch.zeros(2, 4, latent_h, latent_w),  # batch=2 matches CFG concat in run()
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = _constant_prop_time_embeddings(t, dummy, torch_time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute: expected shape by TTNN UNet
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)
    return _tlist


def generate_frames(
    device,
    ttnn_model,
    tt_vae,
    config,
    ttnn_scheduler,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    num_frames: int = 8,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> List[Image.Image]:
    """Generate video frames using TTNN UNet on Blackhole.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: UNet2D TTNN model loaded with preprocess_model_parameters
        tt_vae: Vae TTNN object
        config: unet.config from PyTorch UNet2DConditionModel
        ttnn_scheduler: TtPNDMScheduler with set_timesteps() already called
        torch_time_proj: unet.time_proj from PyTorch UNet (used to build _tlist)
        text_embeddings: Shape (2, 96, 768) torch tensor — [uncond, cond] concatenated,
                         padded from 77 to 96 tokens with torch.nn.functional.pad(..., (0,0,0,19))
        num_frames: Number of frames to generate
        guidance_scale: CFG scale (7.5 standard)
        seed: Random seed for shared base noise
        height: Output image height in pixels (512 recommended for single Blackhole)
        width: Output image width in pixels (512 recommended for single Blackhole)

    Returns:
        List of PIL Images, one per frame

    Note:
        Temporal coherence comes from shared base noise initialization (0.05
        per-frame perturbation). This is not full AnimateDiff temporal attention —
        see Phase 1 (generate_baseline.py) for that.
    """
    import ttnn
    from models.demos.wormhole.stable_diffusion.sd_helper_funcs import run

    lh, lw = height // 8, width // 8
    _tlist = build_tlist(ttnn_scheduler, torch_time_proj, device, lh, lw)
    time_step = ttnn_scheduler.timesteps.tolist()

    # Convert text embeddings to TTNN device tensor
    ttnn_text_embeddings = ttnn.from_torch(
        text_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Shared base noise for inter-frame temporal coherence
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)

    frames = []
    for frame_idx in range(num_frames):
        # Small per-frame perturbation keeps frames related but distinct
        frame_noise = base_noise + 0.05 * torch.randn_like(base_noise)
        latents = ttnn.from_torch(
            frame_noise * ttnn_scheduler.init_noise_sigma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        output = run(
            ttnn_model, config, tt_vae,
            input_latents=latents,
            input_encoder_hidden_states=ttnn_text_embeddings,
            _tlist=_tlist,
            time_step=time_step,
            guidance_scale=guidance_scale,
            ttnn_scheduler=ttnn_scheduler,
        )

        # Post-process: TTNN output is (1, 3, H, W) in [-1, 1]
        img = ttnn.to_torch(output.cpu(blocking=True)).squeeze(0)  # (3, H, W)
        img = (img / 2 + 0.5).clamp(0, 1)
        img = (img.float().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))

        print(f"  Frame {frame_idx + 1}/{num_frames} done")

    return frames
