# AnimateDiff Implementation Status

**Last Updated:** 2026-05-14
**Version:** v0.2.0

---

## Phase 1 — Correct AnimateDiff (CPU Baseline)

**Status: ✅ Complete**

Uses `diffusers.AnimateDiffPipeline` + `MotionAdapter("guoyww/animatediff-motion-adapter-v1-5-2")`
on `CompVis/stable-diffusion-v1-4`. MotionAdapter injects temporal attention at each UNet
transformer block (320-dim features, matching mm_sd_v15_v2.ckpt). No TT hardware needed.

**Run:** `python examples/generate_baseline.py`

**Produces:** Real, temporally coherent GIF animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

**Status: ✅ Code complete, hardware validation pending**

TTNN UNet (`UNet2D` from tt-metal SD 1.4 demo) denoises frames sequentially on Blackhole.
Temporal coherence via shared base noise. `TT_METAL_ARCH_NAME=blackhole` required.

**Run:** `python examples/generate_blackhole.py` (requires Blackhole hardware + ~/tt-metal)

**Known tradeoff:** Temporal attention is NOT applied during TTNN denoising. For full
AnimateDiff on TT hardware, TemporalTransformer blocks would need to be added to the
TTNN UNet — that is out of scope here and would require modifying tt-metal source.

---

## Root Cause of Original Implementation Failure

The original `examples/generate_with_sd35.py` attempted to use:
- **SD 3.5 DiT** (Diffusion Transformer, 2432-dim features)
- with **mm_sd_v15_v2.ckpt** motion weights trained for SD 1.5 UNet (320-dim features)

These are architecturally incompatible. The motion weights expect transformer blocks
with 320-dim hidden states; the DiT operates at 2432-dim. This mismatch meant no
temporal attention was actually being applied. Additionally, `_generate_frame_latents()`
and `_decode_latent()` were both placeholder implementations returning synthetic data.

The fix: use SD 1.4/1.5 (same UNet architecture as training) via diffusers MotionAdapter.
