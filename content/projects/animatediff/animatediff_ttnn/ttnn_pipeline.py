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
    if not TT_METAL_PATH.exists():
        raise RuntimeError(
            f"~/tt-metal not found. Activate the tt-metal environment first:\n"
            f"  cd ~/tt-metal && source python_env/bin/activate"
        )
    tt_metal_str = str(TT_METAL_PATH)
    if tt_metal_str not in sys.path:
        sys.path.insert(0, tt_metal_str)


def _constant_prop_time_embeddings(timesteps, sample, time_proj):
    """Compute time embeddings for a scalar timestep.

    Equivalent to the function defined in SD demo.py. Accepts a scalar
    timestep tensor, expands to batch size, runs through UNet time_proj.
    """
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    return time_proj(timesteps)


def setup_blackhole(device_ids: list[int] | None = None):
    """Open all available Blackhole chips as a 1×N MeshDevice.

    Uses open_mesh_device with explicit physical_device_ids so every chip is
    claimed upfront. This prevents the ARC on un-claimed chips from entering a
    degraded state mid-run and avoids the implicit device_id=0 assumption that
    breaks on multi-card systems where PCIe enumeration order is not guaranteed.

    Also checks hwmon sentinel values (temp > 1,000,000 mC = 65536°C) before
    opening — a dead ARC will cause get_num_devices() to block for up to 5
    minutes. Chips with sentinel values are excluded and a RuntimeWarning is
    raised so the caller knows it's running on a degraded set.

    Args:
        device_ids: Physical device IDs to open. Defaults to all healthy chips.

    Returns a MeshDevice compatible with preprocess_model_parameters, UNet2D,
    and to_device() / from_device() below.
    """
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    _ensure_tt_metal_path()

    import ttnn
    from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE

    if device_ids is None:
        import glob as _glob
        live_ids, dead_ids = [], []
        for hwmon in sorted(_glob.glob("/sys/class/hwmon/hwmon*")):
            try:
                if open(f"{hwmon}/name").read().strip() != "blackhole":
                    continue
                temp_mc = int(open(f"{hwmon}/temp1_input").read().strip())
                idx = len(live_ids) + len(dead_ids)
                if temp_mc > 1_000_000:  # UINT32_MAX overflow — ARC dead
                    dead_ids.append(idx)
                else:
                    live_ids.append(idx)
            except (OSError, ValueError):
                pass
        if dead_ids:
            import warnings
            warnings.warn(
                f"Chip(s) {dead_ids} show ARC-dead sentinel values (temp > 1000°C). "
                f"Excluding from mesh. AC power cycle required to recover. "
                f"Using chips: {live_ids}",
                RuntimeWarning, stacklevel=2,
            )
            device_ids = live_ids
        else:
            device_ids = list(range(ttnn.get_num_devices()))

    n = len(device_ids)
    return ttnn.open_mesh_device(
        mesh_shape=ttnn.MeshShape(1, n),
        physical_device_ids=device_ids,
        l1_small_size=SD_L1_SMALL_SIZE,
    )


def to_device(tensor, device, dtype=None, layout=None):
    """Send a torch tensor to device (single Device or MeshDevice).

    For a MeshDevice every chip gets an identical copy — correct for
    data-parallel SD inference where all chips run the same model.
    """
    import ttnn
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if layout is not None:
        kwargs["layout"] = layout
    if isinstance(device, ttnn.MeshDevice):
        kwargs["mesh_mapper"] = ttnn.ReplicateTensorToMesh(device)
    return ttnn.from_torch(tensor, device=device, **kwargs)


def from_device(tensor, device):
    """Retrieve a tensor from device (single Device or MeshDevice) to CPU torch."""
    import ttnn
    if isinstance(device, ttnn.MeshDevice):
        return ttnn.to_torch(tensor, mesh_composer=ttnn.ConcatMeshToTensor(device, dim=0))[0:1]
    return ttnn.to_torch(tensor)


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

    dummy = to_device(
        torch.zeros(2, 4, latent_h, latent_w),
        device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = _constant_prop_time_embeddings(t, dummy, torch_time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute: expected shape by TTNN UNet
        _t = to_device(_t, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT)
        _tlist.append(_t)
    return _tlist


def generate_frames(
    device,
    ttnn_model,
    torch_vae,
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

    UNet denoising runs on Blackhole via TTNN. VAE decode runs on CPU with the
    PyTorch AutoencoderKL — the TTNN VAE OOMs on Blackhole's final conv_out due
    to a grid/L1 size mismatch in the Wormhole-targeted VAE kernel.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: UNet2D TTNN model loaded with preprocess_model_parameters
        torch_vae: PyTorch AutoencoderKL for CPU latent decode
        config: unet.config from PyTorch UNet2DConditionModel
        ttnn_scheduler: TtPNDMScheduler (set_timesteps already called once)
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
    from models.demos.wormhole.stable_diffusion.sd_helper_funcs import tt_guide

    num_steps = ttnn_scheduler.num_inference_steps
    lh, lw = height // 8, width // 8

    # Convert text embeddings to TTNN device tensor once; reused every frame
    ttnn_text_embeddings = to_device(
        text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    # Shared base noise for inter-frame temporal coherence
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)

    # Build time embeddings once — timesteps are the same for every frame
    ttnn_scheduler.set_timesteps(num_steps)
    time_step = ttnn_scheduler.timesteps.tolist()
    _tlist = build_tlist(ttnn_scheduler, torch_time_proj, device, lh, lw)

    frames = []
    for frame_idx in range(num_frames):
        # Reset PNDM scheduler state (counter, ets buffer) before each frame
        ttnn_scheduler.set_timesteps(num_steps)

        # Per-frame perturbation uses the seeded generator so runs are reproducible
        frame_noise = base_noise + 0.05 * torch.randn(base_noise.shape, generator=generator)
        ttnn_latents = to_device(
            frame_noise * ttnn_scheduler.init_noise_sigma,
            device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
        )

        # TTNN UNet denoising loop on Blackhole
        for index in range(len(time_step)):
            ttnn_latent_model_input = ttnn.concat([ttnn_latents, ttnn_latents], dim=0)
            _t = _tlist[index]
            t = time_step[index]
            ttnn_output = ttnn_model(
                ttnn_latent_model_input,
                timestep=_t,
                encoder_hidden_states=ttnn_text_embeddings,
                class_labels=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True,
                config=config,
            )
            noise_pred = tt_guide(ttnn_output, guidance_scale)
            ttnn_latents = ttnn_scheduler.step(noise_pred, t, ttnn_latents).prev_sample

        # Decode with CPU PyTorch VAE — TTNN VAE conv_out OOMs on Blackhole
        latents_cpu = from_device(ttnn_latents, device).to(torch.float32) / 0.18215
        with torch.no_grad():
            decoded = torch_vae.decode(latents_cpu).sample  # (1, 3, H, W) in [-1, 1]
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))

        print(f"  Frame {frame_idx + 1}/{num_frames} done")

    return frames
