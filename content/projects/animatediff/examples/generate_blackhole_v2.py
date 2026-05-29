#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Phase 2.5: Blackhole TTNN UNet with cross-frame temporal attention.

Generates video frames using the SD 1.4 TTNN UNet on Blackhole hardware.
Cross-frame self-attention is applied at each denoising step across all N
frame noise predictions, giving genuine temporal coherence rather than
shared-noise initialization only.

Architecture:
    For step t in [T → 0]:
        For frame i in [0, N]:
            noise_pred[i] = TTNN_UNet(latent[i], t)   # Blackhole hardware
        noise_preds = cross_frame_attention(stack)      # CPU, ~0ms for N=8
        For frame i in [0, N]:
            latent[i] = scheduler.step(noise_pred[i])

Requirements:
    - ~/tt-metal present: cd ~/tt-metal && source python_env/bin/activate
    - Blackhole hardware (P100/P150/P300c/QB2)
    - SD 1.4 cached: hf download CompVis/stable-diffusion-v1-4

Usage:
    python examples/generate_blackhole_v2.py
    python examples/generate_blackhole_v2.py --prompt "ocean waves" --frames 8
    python examples/generate_blackhole_v2.py --temporal-alpha 0 --output output/no_attn.gif
"""

import argparse
import sys
import time
from pathlib import Path

import torch

TT_METAL_PATH = Path.home() / "tt-metal"
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(TT_METAL_PATH))

from animatediff_ttnn.ttnn_pipeline import setup_blackhole
from animatediff_ttnn.temporal_attention import generate_frames_temporal
from animatediff_ttnn.pipeline import export_gif


def load_sd14_ttnn(device):
    """Load SD 1.4 TTNN UNet and PyTorch VAE; return all components for generation.

    TTNN UNet runs on Blackhole; PyTorch VAE decodes on CPU (TTNN VAE OOMs on
    Blackhole's final conv_out due to L1 grid mismatch in Wormhole-targeted kernel).

    Returns (ttnn_model, torch_vae, config, torch_time_proj).
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
    from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        UNet2DConditionModel as UNet2D,
    )

    print("  Loading PyTorch VAE (CPU decode)...")
    torch_vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    torch_vae.eval()

    print("  Loading PyTorch UNet (for config and time_proj)...")
    torch_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    print("  Building TTNN UNet on Blackhole (~2-3 min first run, cached after)...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = UNet2D(device, parameters, 2, 64, 64)

    return ttnn_model, torch_vae, torch_unet.config, torch_unet.time_proj


def encode_prompt(prompt: str, negative_prompt: str = "") -> torch.Tensor:
    """Encode text + negative prompt to (2, 96, 768) using SD 1.4 CLIP.

    Returns [uncond_embeds, cond_embeds] concatenated; padded 77 → 96 tokens
    to match the TTNN UNet's expected sequence length.
    """
    from transformers import CLIPTokenizer, CLIPTextModel

    tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder")
    text_encoder.eval()

    def encode(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(tokens.input_ids)[0]
        return torch.nn.functional.pad(embeds, (0, 0, 0, 19))  # 77 → 96 tokens

    uncond = encode(negative_prompt)
    cond = encode(prompt)
    return torch.cat([uncond, cond], dim=0)  # (2, 96, 768)


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Phase 2.5 — Blackhole + temporal attention")
    parser.add_argument(
        "--prompt",
        default="1939 World's Fair imagined from the year 2099, art deco spires at golden dusk, retro-futurist optimism, cinematic 4K",
    )
    parser.add_argument("--negative-prompt", default="blurry, low quality", dest="negative_prompt")
    parser.add_argument("--frames", type=int, default=8, help="Frames to generate (8 recommended)")
    parser.add_argument("--steps", type=int, default=25, help="Denoising steps (min 4 for PNDM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/blackhole_v2.gif")
    parser.add_argument(
        "--temporal-alpha",
        type=float,
        default=0.35,
        dest="temporal_alpha",
        help="Cross-frame attention blend (0 = shared-noise only, 1 = full attention; default 0.35)",
    )
    args = parser.parse_args()

    print("AnimateDiff Phase 2.5 — Blackhole TTNN UNet + cross-frame temporal attention")
    print(f"  Prompt         : {args.prompt}")
    print(f"  Frames         : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print(f"  Temporal alpha : {args.temporal_alpha}")
    print()

    print("Opening Blackhole device...")
    # SD 1.4 TTNN UNet (wormhole-targeted) uses ttnn.to_torch() without a
    # mesh_composer, which crashes if the tensor is sharded across >1 chip.
    # Restrict to device 0 until the UNet2D model is updated for multi-device.
    device = setup_blackhole(device_ids=[0])
    print()

    try:
        print("Loading SD 1.4 models onto Blackhole...")
        t0 = time.time()
        ttnn_model, torch_vae, config, torch_time_proj = load_sd14_ttnn(device)
        print(f"  Models loaded in {time.time() - t0:.1f}s")
        print()

        print("Encoding prompts with CLIP...")
        text_embeddings = encode_prompt(args.prompt, args.negative_prompt)
        print(f"  Embeddings shape: {text_embeddings.shape}")
        print()

        print(f"Generating {args.frames} frames with temporal attention on Blackhole...")
        t1 = time.time()
        frames = generate_frames_temporal(
            device=device,
            ttnn_model=ttnn_model,
            torch_vae=torch_vae,
            config=config,
            torch_time_proj=torch_time_proj,
            text_embeddings=text_embeddings,
            num_frames=args.frames,
            num_steps=args.steps,
            seed=args.seed,
            temporal_alpha=args.temporal_alpha,
        )
        elapsed = time.time() - t1
        print(f"  Generated in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)")
        print()
    finally:
        import ttnn
        ttnn.close_mesh_device(device)
        print("Device closed.")
        print()

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames → {args.output}")
    print()
    print(f"TTNN UNet spatial denoising: Blackhole hardware")
    print(f"Cross-frame temporal attention (alpha={args.temporal_alpha}): CPU at each step")
    print(f"VAE decode: CPU (TTNN VAE conv_out OOMs on Blackhole)")


if __name__ == "__main__":
    main()
