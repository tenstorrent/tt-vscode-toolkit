#!/usr/bin/env python3
"""Phase 2: Blackhole-accelerated video frame generation using TTNN UNet.

Loads SD 1.4 TTNN UNet and VAE onto a Blackhole device, encodes a text
prompt with CLIP, then generates num_frames sequentially. Temporal coherence
comes from shared base noise initialization (not AnimateDiff motion adapter).

Requirements:
    - ~/tt-metal present: cd ~/tt-metal && source python_env/bin/activate
    - Blackhole hardware (P100 or P300c)
    - SD 1.4 model cached: hf download CompVis/stable-diffusion-v1-4
    - CLIP tokenizer: hf download openai/clip-vit-large-patch14

Usage:
    python examples/generate_blackhole.py
    python examples/generate_blackhole.py --prompt "ocean waves" --frames 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch

TT_METAL_PATH = Path.home() / "tt-metal"
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(TT_METAL_PATH))

from animatediff_ttnn.ttnn_pipeline import setup_blackhole, generate_frames
from animatediff_ttnn.pipeline import export_gif


def load_sd14_ttnn(device):
    """Load SD 1.4 TTNN UNet, VAE, and PNDM scheduler onto device.

    Returns (ttnn_model, tt_vae, config, ttnn_scheduler, torch_unet_time_proj).
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
    from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
    from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        UNet2DConditionModel as UNet2D,
    )
    from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae import Vae

    print("  Loading PyTorch VAE...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tt_vae = Vae(torch_vae=vae, device=device)

    print("  Loading PyTorch UNet (for config and time_proj)...")
    torch_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    print("  Building TTNN UNet (compiles kernels, ~2-3 min first run)...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = UNet2D(device, parameters, 2, 64, 64)

    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    return ttnn_model, tt_vae, torch_unet.config, ttnn_scheduler, torch_unet.time_proj


def encode_prompt(prompt: str, negative_prompt: str = "") -> torch.Tensor:
    """Encode text + negative prompt to (2, 96, 768) tensor using CLIP.

    Returns [uncond_embeds, cond_embeds] concatenated, padded 77->96 tokens.
    """
    from transformers import CLIPTokenizer, CLIPTextModel

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
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
        # Pad from 77 tokens to 96 — TTNN UNet expects 96-token sequence
        return torch.nn.functional.pad(embeds, (0, 0, 0, 19))

    uncond = encode(negative_prompt)
    cond = encode(prompt)
    return torch.cat([uncond, cond], dim=0)  # (2, 96, 768)


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Phase 2 — Blackhole TTNN")
    parser.add_argument("--prompt", default="a campfire with crackling flames, cinematic, 4K")
    parser.add_argument("--negative-prompt", default="blurry, low quality", dest="negative_prompt")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames (8 recommended)")
    parser.add_argument("--steps", type=int, default=25, help="Denoising steps (min 4 for PNDM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/blackhole.gif")
    args = parser.parse_args()

    print("AnimateDiff Phase 2 — Blackhole TTNN UNet")
    print(f"  TT_METAL_ARCH_NAME=blackhole")
    print(f"  Prompt    : {args.prompt}")
    print(f"  Frames    : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print()

    print("Opening Blackhole device...")
    device = setup_blackhole()
    print()

    print("Loading SD 1.4 models onto Blackhole...")
    t0 = time.time()
    ttnn_model, tt_vae, config, ttnn_scheduler, torch_time_proj = load_sd14_ttnn(device)
    ttnn_scheduler.set_timesteps(args.steps)
    print(f"  Models loaded in {time.time() - t0:.1f}s")
    print()

    print("Encoding prompts with CLIP...")
    text_embeddings = encode_prompt(args.prompt, args.negative_prompt)
    print(f"  Embeddings shape: {text_embeddings.shape}")  # (2, 96, 768)
    print()

    print(f"Generating {args.frames} frames on Blackhole...")
    t1 = time.time()
    frames = generate_frames(
        device,
        ttnn_model,
        tt_vae,
        config,
        ttnn_scheduler,
        torch_time_proj,
        text_embeddings,
        num_frames=args.frames,
        seed=args.seed,
    )
    elapsed = time.time() - t1
    print(f"  Generated in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)")
    print()

    import ttnn
    ttnn.close_device(device)
    print("Device closed.")
    print()

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames -> {args.output}")
    print()
    print("Note: Temporal coherence from shared base noise (not AnimateDiff motion adapter).")
    print("For full AnimateDiff temporal attention, see examples/generate_baseline.py (CPU).")


if __name__ == "__main__":
    main()
