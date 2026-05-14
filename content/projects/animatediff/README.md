# AnimateDiff on Tenstorrent Hardware

Two-phase implementation: **Phase 1** generates real, temporally coherent video
on CPU using the correct AnimateDiff architecture. **Phase 2** accelerates spatial
denoising on Blackhole hardware using the TTNN UNet.

---

## Background: Why the Previous Implementation Was Wrong

The original implementation applied `mm_sd_v15_v2.ckpt` (AnimateDiff motion weights)
to SD 3.5's DiT transformer (2432-dim features). These weights were trained for SD 1.5's
UNet (320-dim features). They are architecturally incompatible.

This rewrite uses the correct base: **SD 1.4 UNet** + **MotionAdapter**, where temporal
attention is injected inside each UNet transformer block at the 320-dim level.

---

## Phase 1 — Correct AnimateDiff (CPU, no hardware needed)

Uses `diffusers.AnimateDiffPipeline` with `MotionAdapter`. The MotionAdapter injects
`TemporalTransformer` attention modules at each `BasicTransformerBlock` in the SD 1.4 UNet.
Motion weights operate at 320-dim features during every denoising step — frames are
temporally coherent by design, not post-hoc.

### Setup

```bash
pip install -r requirements.txt
hf download CompVis/stable-diffusion-v1-4
hf download guoyww/animatediff-motion-adapter-v1-5-2
```

### Run

```bash
python examples/generate_baseline.py
python examples/generate_baseline.py --prompt "a sunset over the ocean" --frames 8
```

Expected output: `output/baseline.gif` — 16 frames of temporally coherent animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

Uses the SD 1.4 TTNN UNet from `~/tt-metal/models/demos/wormhole/stable_diffusion/` —
the same code runs on Blackhole via `TT_METAL_ARCH_NAME=blackhole`. Frames are denoised
sequentially using `sd_helper_funcs.run()`. Temporal coherence from shared base noise.

**Documented tradeoff:** This is TT-hardware-accelerated spatial denoising for video
frames, not full AnimateDiff temporal attention. Full integration would require injecting
`TemporalTransformer` blocks into the TTNN UNet transformer blocks.

### Requirements

- Blackhole hardware (P100 or P300c)
- `~/tt-metal` present, environment activated: `source ~/tt-metal/python_env/bin/activate`
- `hf download CompVis/stable-diffusion-v1-4` (also used by Phase 1)
- `hf download openai/clip-vit-large-patch14`

### Run

```bash
source ~/tt-metal/python_env/bin/activate
python examples/generate_blackhole.py
python examples/generate_blackhole.py --prompt "a campfire" --frames 8
```

Expected: `output/blackhole.gif` — 8 frames generated on Blackhole hardware.

---

## Code Structure

```
animatediff_ttnn/
  pipeline.py          Phase 1: thin wrapper around diffusers AnimateDiffPipeline
  ttnn_pipeline.py     Phase 2: TTNN UNet frame generation on Blackhole
  temporal_module.py   Reference only — temporal attention math (kept for study)
  __init__.py          Exports Phase 1 public API

examples/
  generate_baseline.py   Phase 1 end-to-end (CPU, any machine)
  generate_blackhole.py  Phase 2 end-to-end (Blackhole hardware)

tests/
  test_pipeline.py       Phase 1 unit tests
  test_ttnn_pipeline.py  Phase 2 unit tests (hardware-mocked)

docs/
  IMPLEMENTATION_STATUS.md  Current Phase 1/2 status
```

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests mock hardware dependencies and run on any machine.

---

## AnimateDiff Architecture Reference

```
SD 1.4 UNet without MotionAdapter:
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each block has BasicTransformerBlock(spatial attention)

SD 1.4 UNet WITH MotionAdapter (Phase 1):
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each BasicTransformerBlock now has:
             spatial attention (unchanged)
             + TemporalTransformer(cross-frame attention, 320-dim)
                                              ↑
                              This is where mm_sd_v15_v2.ckpt weights live
```

For full AnimateDiff on Blackhole, the TTNN UNet transformer blocks would need
TemporalTransformer layers inserted — a deeper integration than Phase 2 attempts.
