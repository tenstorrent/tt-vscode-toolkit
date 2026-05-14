# AnimateDiff on Blackhole — Design Spec

**Date:** 2026-05-14
**Status:** Approved

---

## Problem

The existing `content/projects/animatediff/` implementation has two fundamental bugs:

1. **Wrong base model.** `generate_with_sd35.py` uses SD 3.5's DiT architecture (2432-dim transformer blocks). AnimateDiff's `mm_sd_v15_v2.ckpt` motion weights were trained for SD 1.5's UNet (320-dim transformer blocks). These are architecturally incompatible — no path to adapter injection.

2. **Demo/placeholder mode.** `_generate_frame_latents()` returns random latents; `_decode_latent()` returns gradient images. The pipeline never runs real denoising.

This rewrite fixes both by grounding the implementation in the correct SD 1.4/1.5 UNet architecture and providing two working phases.

---

## Architecture: Two Phases

### Phase 1 — Correct AnimateDiff (CPU Baseline)

Uses the diffusers `AnimateDiffPipeline` with `MotionAdapter`. No TT hardware required. Produces real, temporally coherent video.

**Why this works:** `MotionAdapter("guoyww/animatediff-motion-adapter-v1-5-2")` injects `TemporalTransformer` attention modules at each `BasicTransformerBlock` in the SD 1.4 UNet. The motion weights operate at 320-dim features, which is exactly what they were trained for. Temporal attention propagates inter-frame relationships during spatial denoising — not after.

**Stack:**
- Base model: `CompVis/stable-diffusion-v1-4`
- Motion adapter: `guoyww/animatediff-motion-adapter-v1-5-2`
- Scheduler: `DDIMScheduler` with `clip_sample=False`, `beta_schedule="linear"`, `steps_offset=1`
- diffusers >= 0.32.1 (has `AnimateDiffPipeline` and `MotionAdapter`)

**Output:** 16 frames at 512×512, exported as animated GIF.

### Phase 2 — Blackhole-Accelerated Spatial Diffusion

Replaces the PyTorch UNet with the TTNN UNet from `~/tt-metal/models/demos/wormhole/stable_diffusion/tt/unet_2d_condition_model.py`. The same TTNN UNet code runs on Blackhole (confirmed: `models/demos/blackhole/stable_diffusion/` imports it directly). Frames are denoised sequentially using `sd_helper_funcs.run()`.

**Temporal coherence mechanism:** Shared base noise initialization across frames. Each frame's latent is the base noise plus a small per-frame perturbation. This is a documented simplification — not full AnimateDiff temporal attention.

**Documented scope boundary:** Full AnimateDiff motion module integration on TT hardware would require injecting `TemporalTransformer` blocks into the TTNN UNet transformer blocks, modifying `~/tt-metal/models/demos/wormhole/stable_diffusion/tt/`. That is out of scope for this lesson. Phase 2 is "TT-accelerated video frame generation" — the lesson calls this out explicitly.

**Hardware:** Blackhole (P100/P300c), `TT_METAL_ARCH_NAME=blackhole`, `l1_small_size=SD_L1_SMALL_SIZE`.

**Key implementation detail for `sd_helper_funcs.run()`:**
- Takes `input_latents` as 4-channel 64×64 tensor
- Internally concatenates `[latents, latents]` for CFG (batch=2)
- Runs full denoising loop with TTNN scheduler
- VAE-decodes output, returns 3-channel image tensor
- Called once per frame

---

## File Changes

### Delete
- `content/projects/animatediff/examples/generate_with_sd35.py` — wrong architecture, root cause of original bug

### Rewrite
- `content/projects/animatediff/README.md` — phases, setup, hardware requirements, `hf` CLI throughout
- `content/projects/animatediff/animatediff_ttnn/pipeline.py` — Phase 1: diffusers `AnimateDiffPipeline` wrapper
- `content/projects/animatediff/examples/generate_baseline.py` — Phase 1 end-to-end example
- `content/projects/animatediff/docs/IMPLEMENTATION_STATUS.md` — honest Phase 1/2 status
- `content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md` — `hf` CLI, correct model paths

### New
- `content/projects/animatediff/animatediff_ttnn/ttnn_pipeline.py` — Phase 2: Blackhole TTNN pipeline
- `content/projects/animatediff/examples/generate_blackhole.py` — Phase 2 end-to-end example

### Update
- `content/projects/animatediff/requirements.txt` — `diffusers>=0.32.1`, remove stale deps

---

## Phase 1 Reference Implementation

```python
# animatediff_ttnn/pipeline.py
from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
import torch

def create_animatediff_pipeline(model_path: str = "CompVis/stable-diffusion-v1-4",
                                  adapter_path: str = "guoyww/animatediff-motion-adapter-v1-5-2"):
    adapter = MotionAdapter.from_pretrained(adapter_path)
    scheduler = DDIMScheduler.from_pretrained(
        model_path, subfolder="scheduler",
        clip_sample=False, timestep_spacing="linspace",
        beta_schedule="linear", steps_offset=1
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_path, motion_adapter=adapter, scheduler=scheduler
    )
    return pipe

def generate(pipe, prompt: str, num_frames: int = 16,
             guidance_scale: float = 7.5, num_inference_steps: int = 25,
             seed: int = 42):
    output = pipe(
        prompt=prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    )
    return output.frames[0]  # list of PIL Images
```

---

## Phase 2 Reference Implementation

```python
# animatediff_ttnn/ttnn_pipeline.py
import os
import torch
import ttnn
import sys

SD_L1_SMALL_SIZE = 0x4000  # from Blackhole SD demo

def setup_blackhole():
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    device = ttnn.open_device(device_id=0, l1_small_size=SD_L1_SMALL_SIZE)
    return device

def generate_frames(device, ttnn_unet, ttnn_vae, config, scheduler,
                    text_embeddings, num_frames=8, num_inference_steps=25,
                    guidance_scale=7.5, seed=42, height=512, width=512):
    # Shared base noise for temporal coherence
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, height // 8, width // 8, generator=generator)

    frames = []
    for frame_idx in range(num_frames):
        # Per-frame noise perturbation
        frame_noise = base_noise + 0.05 * torch.randn_like(base_noise)
        scheduler.set_timesteps(num_inference_steps)
        latents = frame_noise * scheduler.init_noise_sigma

        # TTNN denoising loop (sd_helper_funcs.run handles CFG batch=2 internally)
        from models.demos.wormhole.stable_diffusion import sd_helper_funcs
        output = sd_helper_funcs.run(
            ttnn_unet, config, ttnn_vae,
            input_latents=latents,
            input_encoder_hidden_states=text_embeddings,
            _tlist=_build_tlist(scheduler),
            time_step=scheduler.timesteps,
            guidance_scale=guidance_scale,
            ttnn_scheduler=scheduler,
        )
        frames.append(output)
    return frames
```

---

## Model Downloads

```bash
hf download CompVis/stable-diffusion-v1-4
hf download guoyww/animatediff-motion-adapter-v1-5-2
```

Both are open-weight (no gated access). SD 1.4 is ~4GB; motion adapter is ~700MB.

---

## Hardware Requirements

| Phase | Hardware | Setup |
|-------|----------|-------|
| Phase 1 | Any CPU (no TT hardware) | `pip install -r requirements.txt` |
| Phase 2 | Blackhole (P100 or P300c) | `TT_METAL_ARCH_NAME=blackhole` + tt-metal activated |

Phase 2 requires `~/tt-metal` present and the metal environment activated.

---

## Lesson Documentation Strategy

The rewritten `README.md` will:

1. **Lead with Phase 1** — run this first, no hardware needed, see real AnimateDiff output
2. **Explain what AnimateDiff actually does** — MotionAdapter injects temporal attention at the right place (UNet transformer blocks, not post-processing)
3. **Explain what Phase 2 does and doesn't do** — TTNN UNet accelerates spatial denoising; temporal coherence from shared noise, not motion adapter weights; explicit callout: "full integration would require modifying TTNN UNet transformer blocks"
4. **All `hf` CLI** — `hf download`, `hf auth login`, `hf auth whoami` throughout

---

## Out of Scope

- Injecting `TemporalTransformer` blocks into the TTNN UNet (would require modifying `~/tt-metal/models/demos/wormhole/stable_diffusion/tt/`)
- Multi-chip (T3K, Galaxy) distribution of the animation pipeline
- SD 3.5 / DiT-based video generation (separate model family — see WAN2.2 for production video on QB2)
- ControlNet or IP-Adapter combination

---

## Success Criteria

1. `python examples/generate_baseline.py "a campfire"` — produces `output.gif` with visible temporal coherence (flames move consistently frame to frame)
2. `python examples/generate_blackhole.py "a campfire"` — produces `output.gif` on Blackhole hardware, no crashes, correct `TT_METAL_ARCH_NAME` setup
3. No references to `huggingface-cli` anywhere in the project
4. No references to SD 3.5 or `generate_with_sd35.py`
5. `IMPLEMENTATION_STATUS.md` accurately describes Phase 1 as complete, Phase 2 as "TT-accelerated with documented temporal coherence tradeoff"
