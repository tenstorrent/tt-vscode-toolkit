"""Cross-frame temporal attention for AnimateDiff Phase 2.5 on Blackhole.

Applies self-attention across N frames at each denoising step, giving genuine
temporal coherence without requiring the MotionAdapter TemporalTransformer
(which operates on 320-dim UNet intermediate features not accessible from
the standalone TTNN UNet pipeline).

Architecture:
    For step t in [T, T-1, ..., 0]:
        For frame i in [0, N-1]:
            noise_pred[i] = TTNN_UNet(latent[i], t, text_emb)   # Blackhole
        noise_preds = cross_frame_attention(stack(noise_pred))    # CPU (tiny)
        For frame i in [0, N-1]:
            latent[i] = scheduler.step(noise_preds[i], t)        # CPU

Why this works: noise predictions at each step carry structural information
(edges, shapes, motion directions). Attending across frames causes the network
to agree on structure before the scheduler commits to a latent direction.

Phase 2.5 vs Phase 1 (MotionAdapter):
    Phase 1 temporal attention runs INSIDE UNet blocks on 320-dim features —
    full AnimateDiff but CPU-only. Phase 2.5 attention runs at the 4-dim
    noise-prediction level — approximate but Blackhole-accelerated.
"""

import sys
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def cross_frame_attention(noise_preds: torch.Tensor, alpha: float = 0.35) -> torch.Tensor:
    """Self-attention across frames on stacked noise predictions.

    Reshapes [N, 4, H, W] to [H*W, N, C] so each spatial position attends
    across the N frame predictions, then blends attended and original.

    Args:
        noise_preds: Shape [N, C, H, W] — stacked noise preds for all frames
        alpha: Blend weight (0 = no effect, 1 = full attention). 0.35 gives
               strong coherence while preserving frame-to-frame variety.

    Returns:
        Blended tensor of same shape [N, C, H, W]
    """
    N, C, H, W = noise_preds.shape
    if N == 1:
        return noise_preds

    x = noise_preds.permute(2, 3, 0, 1).reshape(H * W, N, C).float()

    # Scaled dot-product self-attention (Q = K = V = x)
    scale = C ** -0.5
    attn = torch.bmm(x, x.transpose(-2, -1)) * scale  # [H*W, N, N]
    attn = torch.softmax(attn, dim=-1)
    attended = torch.bmm(attn, x)                      # [H*W, N, C]

    attended = attended.reshape(H, W, N, C).permute(2, 3, 0, 1)
    blended = (1.0 - alpha) * noise_preds + alpha * attended
    return blended.to(noise_preds.dtype)


def generate_frames_temporal(
    device,
    ttnn_model,
    torch_vae,
    config,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    num_frames: int = 8,
    num_steps: int = 25,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
    temporal_alpha: float = 0.35,
) -> List:
    """Generate temporally-coherent frames on Blackhole with cross-frame attention.

    All N frames are denoised in parallel (one TTNN UNet call per frame per step).
    Cross-frame attention is applied to the stacked noise predictions at each step
    before the scheduler commits to the next latent. Total TTNN UNet calls equals
    Phase 2: num_frames × num_steps.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: Loaded TTNN UNet2D model (from preprocess_model_parameters)
        torch_vae: CPU PyTorch AutoencoderKL for latent → pixel decode
        config: unet.config from PyTorch UNet2DConditionModel
        torch_time_proj: unet.time_proj, used by build_tlist for timestep embeddings
        text_embeddings: Shape (2, 96, 768) — [uncond, cond] concatenated,
                         padded from 77 to 96 tokens
        num_frames: Number of frames to generate
        num_steps: PNDM denoising steps (min 4)
        guidance_scale: CFG scale (7.5 standard)
        seed: RNG seed — shared base noise + per-frame perturbation
        height, width: Output size in pixels (512 × 512 recommended)
        temporal_alpha: Cross-frame attention blend (0 → Phase 2 shared noise,
                        1 → full attention; default 0.35)

    Returns:
        List of PIL Images, length num_frames, with temporal coherence
    """
    import ttnn
    from diffusers import PNDMScheduler
    from PIL import Image
    from animatediff_ttnn.ttnn_pipeline import build_tlist, to_device, from_device
    from models.demos.wormhole.stable_diffusion.sd_helper_funcs import tt_guide
    from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler

    lh, lw = height // 8, width // 8

    # One diffusers PNDMScheduler per frame — independent state, same timesteps
    sched_kwargs = dict(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
    )
    schedulers = [PNDMScheduler(**sched_kwargs) for _ in range(num_frames)]
    for s in schedulers:
        s.set_timesteps(num_steps)

    timesteps = schedulers[0].timesteps
    init_noise_sigma = float(schedulers[0].init_noise_sigma)

    # Shared base noise — same starting point for all frames
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)

    frame_latents = []
    for _ in range(num_frames):
        perturbed = base_noise + 0.05 * torch.randn_like(base_noise)
        frame_latents.append(perturbed * init_noise_sigma)

    # Build TTNN time embeddings — needs a TtPNDMScheduler for timestep tensors
    _tt_sched = TtPNDMScheduler(device=device, **sched_kwargs)
    _tt_sched.set_timesteps(num_steps)
    _tlist = build_tlist(_tt_sched, torch_time_proj, device, lh, lw)

    # Text embeddings to device — same tensor reused for every frame at every step
    ttnn_text_emb = to_device(
        text_embeddings, device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
    )

    # Parallel denoising with cross-frame attention at each step
    for step_idx, t in enumerate(timesteps):
        # Collect TTNN noise predictions for all frames at timestep t
        noise_preds = []
        for i in range(num_frames):
            lat = to_device(
                frame_latents[i], device, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
            )
            # TTNN UNet expects batch=2 for CFG (unconditional + conditional)
            lat_input = ttnn.concat([lat, lat], dim=0)
            ttnn_out = ttnn_model(
                lat_input,
                timestep=_tlist[step_idx],
                encoder_hidden_states=ttnn_text_emb,
                class_labels=None,
                attention_mask=None,
                cross_attention_kwargs=None,
                return_dict=True,
                config=config,
            )
            guided = tt_guide(ttnn_out, guidance_scale)
            noise_preds.append(from_device(guided, device).to(torch.float32))

        # Cross-frame attention — [N, 4, H, W]
        stacked = torch.cat(noise_preds, dim=0)
        attended = cross_frame_attention(stacked, alpha=temporal_alpha)

        # Scheduler step for each frame with the temporally-attended noise_pred
        for i in range(num_frames):
            frame_latents[i] = schedulers[i].step(
                attended[i : i + 1], t, frame_latents[i]
            ).prev_sample

        print(f"  Step {step_idx + 1}/{len(timesteps)}", end="\r", flush=True)

    print()

    # Decode all latents with CPU VAE (TTNN VAE conv_out OOMs on Blackhole)
    frames = []
    for i, latent in enumerate(frame_latents):
        latent_scaled = latent / 0.18215
        with torch.no_grad():
            decoded = torch_vae.decode(latent_scaled).sample  # (1, 3, H, W) in [-1, 1]
        img = (decoded / 2 + 0.5).clamp(0, 1)
        img = (img.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))
        print(f"  Frame {i + 1}/{num_frames} decoded")

    return frames
