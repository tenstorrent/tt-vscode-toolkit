---
id: animatediff-video-generation
title: Native Video Animation with AnimateDiff
description: >-
  Run Stable Diffusion 1.4 video generation on Blackhole hardware using the TTNN UNet.
  Learn the complete model bring-up workflow — from research to a standalone package
  that accelerates SD frame generation at 15 seconds per frame on P300C/QB2.
category: applications
tags:
  - animation
  - video
  - animatediff
  - stable-diffusion
  - model-bringup
  - blackhole
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p300c
status: validated
validatedOn:
  - p300c
estimatedMinutes: 60
---

# Native Video Animation with AnimateDiff

**Run SD 1.4 video generation on Blackhole — 15 seconds per frame, real images, no CPU fallback on the UNet.**

This lesson walks through three paths to animated video, escalating from CPU baseline to full Blackhole hardware acceleration with cross-frame temporal attention:

- **Phase 1 (any hardware)** — `diffusers` `AnimateDiffPipeline` on CPU, full AnimateDiff temporal attention via MotionAdapter, ~2 min/frame
- **Phase 2 (Blackhole)** — TTNN UNet on Blackhole, ~15 s/frame
- **Phase 2.5 (Blackhole + temporal attention)** — cross-frame self-attention at every denoising step, the canonical production path

Along the way you'll learn the **model bring-up methodology**: how to create standalone packages that integrate with TT-Metal without modifying the core repository.

## What you'll build

These were generated on a single Blackhole P300C — 8 frames × 25 steps each:

| *"World's Fair 2099"* | *"Phosphor Horizon"* | *"Nebula"* | *"Mayan Temple"* |
|---|---|---|---|
| ![world of tomorrow](/assets/img/animatediff_world_of_tomorrow.gif) | ![phosphor horizon](/assets/img/animatediff_phosphor_horizon.gif) | ![nebula](/assets/img/animatediff_nebula.gif) | ![mayan temple](/assets/img/animatediff_mayan_temple.gif) |

And a full 35-GIF cosmic study across the chip's 11×10 Tensix grid — see the [live showcase](https://tenstorrent.github.io/tt-animatediff/).

[![Chip grid — 35 GIFs across 110 Tensix nodes](/assets/img/animatediff_chip_grid.png)](https://tenstorrent.github.io/tt-animatediff/)

---

## What is AnimateDiff?

AnimateDiff adds **temporal attention** to Stable Diffusion 1.4 by injecting `TemporalTransformer` blocks into every `BasicTransformerBlock` in the UNet — at the 320-dim feature level where the motion weights were trained:

```
SD 1.4 UNet WITHOUT MotionAdapter:
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each block: BasicTransformerBlock(spatial attention only)

SD 1.4 UNet WITH MotionAdapter (Phase 1):
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each block: BasicTransformerBlock
                         └── spatial attention (unchanged, 320-dim)
                         └── TemporalTransformer (cross-frame, 320-dim)
                                            ↑
                               mm_sd_v15_v2.ckpt weights here
```

**Why SD 1.4, not SD 3.5?** AnimateDiff motion weights (`mm_sd_v15_v2.ckpt`) were trained for SD 1.5's UNet with 320-dim transformer blocks. SD 3.5 uses a DiT with 2432-dim blocks — architecturally incompatible. The `diffusers` `AnimateDiffPipeline` handles MotionAdapter injection automatically when paired with the SD 1.4 base model.

---

## Step 1: Get the project

The `tt-animatediff` repo is public. Clone it directly — you own your copy:

```bash
git clone --depth 1 --branch v0.1.0 \
    https://github.com/tenstorrent/tt-animatediff.git \
    ~/tt-projects/tt-animatediff

cd ~/tt-projects/tt-animatediff
pip install -e ".[dev]"
```

[📦 Setup AnimateDiff Project](command:tenstorrent.setupAnimateDiffProject)

**Project structure:**

```
tt-animatediff/
├── animatediff_ttnn/
│   ├── pipeline.py            # Phase 1: CPU AnimateDiffPipeline wrapper
│   ├── ttnn_pipeline.py       # Phase 2/2.5: Blackhole TTNN UNet + PNDM scheduler
│   ├── temporal_attention.py  # Phase 2.5: cross-frame self-attention
│   └── temporal_module.py     # Reference — temporal attention math
├── examples/
│   ├── generate.py            # Unified entry point (--mode cpu|blackhole|sim)
│   ├── generate_baseline.py   # Phase 1 CPU shim
│   └── generate_sim.py        # Phase 2.5 on ttsim simulator shim
├── scripts/
│   └── generate_study.py      # Batch generation (35-GIF cosmic study)
├── docs/
│   ├── INTEGRATION_GUIDE.md
│   ├── SIMULATOR.md
│   └── HARDWARE_COMPAT.md
└── tests/                     # 16 CPU/mock tests, no hardware required
```

---

## Step 2: Download the models

```bash
# SD 1.4 — required for all phases
hf download CompVis/stable-diffusion-v1-4

# AnimateDiff motion adapter — Phase 1 only
hf download guoyww/animatediff-motion-adapter-v1-5-2
```

---

## Step 3: Phase 1 — CPU AnimateDiffPipeline

The `diffusers` `AnimateDiffPipeline` loads SD 1.4, injects the MotionAdapter at every transformer block, and denoises all frames simultaneously with temporal attention. This is the **reference implementation** — correct output to compare against.

[🎬 Run Phase 1 (CPU)](command:tenstorrent.runAnimateDiff2Frame)

```bash
cd ~/tt-projects/tt-animatediff
python examples/generate.py --mode cpu \
    --prompt "aurora borealis over arctic ice, green and violet ribbons, cinematic" \
    --frames 8 --steps 25 --output output/phase1.gif
```

**Expected:** ~2 min/frame on CPU. `output/phase1.gif` — 8 frames of temporally coherent animation.

**What happens inside:**

```python
from diffusers import AnimateDiffPipeline, MotionAdapter

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2")
pipe = AnimateDiffPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", motion_adapter=adapter)

# All 8 frames denoised together — temporal attention at every step
result = pipe(prompt=prompt, num_frames=8, num_inference_steps=25)
frames = result.frames[0]  # List of PIL Images
```

---

## Step 4: Phase 2.5 — Blackhole + temporal attention (canonical)

Replaces the PyTorch UNet with the TTNN UNet from `~/tt-metal`, running natively on Blackhole silicon. Cross-frame self-attention is applied at each PNDM step across all N frame latents simultaneously — giving genuine temporal coherence at hardware speed.

**Requires:** Blackhole hardware (P100/P150/P300C/QB2) and `~/tt-metal` built.

[⚡ Run Phase 2.5 (Blackhole)](command:tenstorrent.runAnimateDiff16Frame)

```bash
source ~/tt-metal/python_env/bin/activate
cd ~/tt-projects/tt-animatediff

python examples/generate.py \
    --prompt "1939 World's Fair imagined from the year 2099, art deco spires at golden dusk, retro-futurist optimism, cinematic 4K" \
    --frames 8 --steps 25 --temporal-alpha 0.35 \
    --output output/blackhole.gif
```

**Expected output:**

```
AnimateDiff — Blackhole hardware (TTNN UNet + cross-frame temporal attention)
  Frames : 8  Steps: 25  Temporal alpha: 0.35

Opening Blackhole device...
Loading SD 1.4 models...
  Building TTNN UNet (~2-3 min first run, cached after)...
  Loaded in 7.3s

Generating 8 frame(s)...
  Done in 121.4s (15.2s/frame)

Saved 8 frame(s) → output/blackhole.gif
```

**Performance on P300C:** 7s model load + ~15s/frame. Kernel compilation ~2–3 min on first run, cached after.

---

## What we built — the improvements

The `tt-animatediff` repo went through several rounds of bring-up work to get to this state. Here's what was added beyond the basic TTNN UNet port:

### Cross-frame temporal attention (Phase 2.5)

The original Phase 2 denoised each frame independently with only shared base noise for coherence. Phase 2.5 adds a CPU cross-frame self-attention pass at each denoising step:

```
For step t in [T → 0]:
    For frame i in [0, N]:
        noise_pred[i] = TTNN_UNet(latent[i], t)   # Blackhole hardware
    noise_preds = cross_frame_attention(stack)      # CPU, ~0ms for N=8
    For frame i in [0, N]:
        latent[i] = scheduler.step(noise_pred[i])
```

The cross-frame attention (`animatediff_ttnn/temporal_attention.py`) reshapes the stacked noise predictions so frames attend to each other, then blends the result back via `--temporal-alpha`:

```python
def cross_frame_attention(x: torch.Tensor, alpha: float = 0.35) -> torch.Tensor:
    # x: (N, C, H, W) — one noise prediction per frame
    N, C, H, W = x.shape
    flat = x.view(N, C, H*W).permute(2, 0, 1)  # (H*W, N, C)
    attn_out = F.scaled_dot_product_attention(flat, flat, flat)
    attn_out = attn_out.permute(1, 2, 0).view(N, C, H, W)
    return (1 - alpha) * x + alpha * attn_out
```

### Hardware resilience

`setup_blackhole()` reads hwmon sentinel values before opening the device. If a chip shows a dead-ARC temperature (> 1,000,000 millidegrees in `temp1_input`), a warning is emitted. If you see `Timed out while waiting for active ethernet core`, run:

```bash
tt-smi -r 0 1 2 3 && sleep 8
```

Then retry — this clears hung ethernet cores from prior incomplete teardowns.

### Unified entry point

All three modes are one script with a `--mode` flag:

```bash
python examples/generate.py --mode cpu        # any machine, ~2 min/frame
python examples/generate.py --mode blackhole  # Blackhole hardware, ~15 s/frame
python examples/generate.py --mode sim        # no hardware, ttsim virtual device
```

### tt-metal path tracking

Between firmware 19.5.0 and 19.8.0, the SD 1.4 demo moved from `models.demos.wormhole.stable_diffusion` to `models.demos.vision.generative.stable_diffusion.wormhole`. See `docs/HARDWARE_COMPAT.md` for recovery steps.

---

## Prompt guide

SD 1.4 at 512×512 with the TTNN UNet has a distinct personality. Knowing it gets you better results.

### What it does well

| Category | Examples |
|---|---|
| Cosmic & abstract | nebulae, aurora, galaxies, energy fields, sacred geometry |
| Natural scenes | forests, oceans, deserts, fire, water, sky |
| Painterly styles | oil painting, watercolor, impressionism, concept art |
| Cinematic lighting | golden hour, neon glow, moonlight, candlelight |
| Architecture | temples, ruins, castles, sci-fi structures |
| Retro aesthetics | CRT glow, film grain, vaporwave, cyberpunk |

### What to avoid

- **Photorealistic faces** — anatomy drifts frame-to-frame
- **Text in the image** — SD 1.4 cannot render legible text
- **Specific named real places** — results are impressionistic
- **Very long prompts** — CLIP truncates at 77 tokens (~60 words max)

### Prompt patterns that work

```
# Style before subject — model weights the style heavily
"watercolor painting of ancient ruins at sunset, soft brushstrokes, muted palette"

# Cinematic lighting descriptors unlock quality
"cinematic 4K, dramatic side lighting, volumetric fog, depth of field"

# Cosmic + architecture is the sweet spot for this model
"Mayan pyramid under a swirling nebula, starfield, bioluminescent jungle, cinematic 4K"

# Motion-friendly subjects produce the best animation
"crackling campfire"   "ocean waves"   "swirling clouds"   "aurora borealis"
"shifting cosmos"      "flowing lava"  "drifting smoke"    "mandala blooming"
```

### `--temporal-alpha` tuning

| Value | Effect |
|---|---|
| `0.0` | No cross-frame mixing — shared noise only |
| `0.2–0.3` | Subtle coherence, natural variation |
| **`0.35`** | **Default — good for most subjects** |
| `0.5–0.7` | Strong coherence, background stabilises |
| `1.0` | Maximum blending, very low motion |

Fast motion (fire, water): 0.2–0.35 · Slow drift (cosmos, aurora): 0.4–0.6

### Gallery — cosmic sweet spot

| Sacred geometry | Circuit as nature | Chip as cosmos |
|---|---|---|
| ![mandala](/assets/img/animatediff_mandala.gif) | ![circuit moss](/assets/img/animatediff_circuit_moss.gif) | ![chip cosmos](/assets/img/animatediff_chip_cosmos.gif) |
| `sacred mandala blooming from starfield` | `circuit board growing like moss` | `Blackhole chip glowing with embedded cosmos` |

| Aurora | Mayan temple | Nebula |
|---|---|---|
| ![aurora](/assets/img/animatediff_aurora.gif) | ![mayan temple](/assets/img/animatediff_mayan_temple.gif) | ![nebula](/assets/img/animatediff_nebula.gif) |
| `aurora borealis over arctic ice` | `ancient Mayan temple under shifting cosmos` | `swirling nebula in deep space` |

---

## How Phase 2.5 works

**`animatediff_ttnn/ttnn_pipeline.py`** — the Blackhole denoising loop:

```python
def generate_frames_temporal(device, ttnn_model, torch_vae, config,
                              torch_time_proj, text_embeddings,
                              num_frames, num_steps, seed, temporal_alpha):
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn((1, 4, 64, 64), generator=generator)

    # Per-frame seeded perturbation for variation
    frame_latents = [
        base_noise + 0.05 * torch.randn((1,4,64,64), generator=generator)
        for _ in range(num_frames)
    ]

    for step_idx, t in enumerate(timesteps):
        # TTNN UNet forward pass per frame on Blackhole
        noise_preds = []
        for i in range(num_frames):
            lat = to_device(frame_latents[i], device, ...)
            ttnn_out = ttnn_model(lat, timestep=_tlist[step_idx], ...)
            noise_preds.append(from_device(tt_guide(ttnn_out, guidance_scale), device))

        # Cross-frame attention on CPU — frames agree on structure
        noise_preds = cross_frame_attention(torch.stack(noise_preds), alpha=temporal_alpha)

        # Scheduler step
        for i in range(num_frames):
            frame_latents[i] = pndm_step(noise_preds[i], t, frame_latents[i])

    # VAE decode on CPU — TTNN VAE conv_out OOMs on Blackhole's L1 grid
    return [vae_decode(torch_vae, lat) for lat in frame_latents]
```

**CLIP encoding** uses the text encoder bundled inside SD 1.4 — no separate download. Tokens are padded 77 → 96 to match the TTNN UNet's expected sequence length.

**MeshDevice:** `setup_blackhole()` opens a `MeshDevice(1,1)` on a single chip. The SD 1.4 TTNN UNet uses `ttnn.to_torch()` without a mesh composer, which crashes on multi-chip tensors — single-chip until Phase 3 ships `ShardTensorToMesh` batched dispatch.

---

## The model bring-up methodology

What this project demonstrates is **the complete workflow for integrating any new model**:

### Phase 1: Research (1–2 hours)
1. Clone the reference implementation
2. Read the architecture files — code reveals reality, papers tell the story
3. Document key patterns: reshaping logic, injection points, weight keys
4. Verify on CPU first — a working baseline is your ground truth

### Phase 2: Design (30–60 min)
1. **Standalone package over monorepo modification** — your code, your ownership
2. Define the API surface: what does a caller need to pass in?
3. Identify the TT-Metal integration boundary: which ops stay PyTorch, which go TTNN?

### Phase 3: Implementation (2–4 hours)
1. Start with PyTorch — easier to debug, matches reference
2. Build the TTNN path after PyTorch is validated
3. Keep the CPU path alive as a regression check

### Phase 4: Packaging (1 hour)
- `setup.py` + `requirements.txt` makes it `pip install -e .`-able
- Single unified entry point (`generate.py --mode cpu|blackhole|sim`)
- `docs/HARDWARE_COMPAT.md` documents version-specific gotchas

### Phase 5: Validate on hardware (1–2 hours)
1. First run: expect kernel compilation ~2–3 min — normal, cached after
2. Compare output to Phase 1 CPU baseline at same prompt and seed
3. If hardware hangs: `tt-smi -r 0 1 2 3 && sleep 8`

**Total for a complete new model: 6–10 hours.**

---

## Known limitations

| Issue | Status |
|---|---|
| TTNN VAE OOMs on Blackhole `conv_out` | Workaround: CPU PyTorch VAE decode |
| No TemporalTransformer blocks in TTNN UNet | Phase 2.5 adds CPU cross-frame attention as a bridge |
| Single-chip only (multi-chip crashes on `to_torch()`) | Use `device_ids=[0]` until Phase 3 |
| tt-metal SD path changed in firmware 19.8.0 | See `docs/HARDWARE_COMPAT.md` |
| First run 2–3 min kernel compilation | Expected; cached after first run |

---

## No hardware? Use the simulator

[ttsim](https://github.com/tenstorrent/ttsim) is a bit-exact Blackhole simulator that runs on any Linux/x86_64 machine — same TTNN dispatch path as real hardware.

```bash
python examples/generate.py --mode sim \
    --sim ~/sim/libttsim_bh.so \
    --frames 2 --steps 4 --output output/sim_test.gif
```

See `docs/SIMULATOR.md` in the repo for full setup.

---

## What's next

### Add TemporalTransformer blocks to the TTNN UNet

Full Phase 3 would inject `TemporalTransformer` blocks into the TTNN UNet's `BasicTransformerBlock` instances — native TTNN temporal attention, eliminating the CPU bounce in Phase 2.5.

### Apply this pattern to other models

The standalone package pattern works for any model:
- **ControlNet** — conditioning inputs for SD 1.4
- **LoRA** — weight delta injection into the SD UNet
- **IP-Adapter** — image-conditioned generation
- **Any PyTorch model** — wrap it, validate on CPU, port to TTNN

---

## Resources

- **Repo:** [github.com/tenstorrent/tt-animatediff](https://github.com/tenstorrent/tt-animatediff)
- **Showcase:** [tenstorrent.github.io/tt-animatediff](https://tenstorrent.github.io/tt-animatediff/)
- **AnimateDiff paper:** [arxiv 2307.04725](https://arxiv.org/abs/2307.04725) — Guo, Zheng, Hu et al.
- **SD 1.4:** [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion) — Rombach, Blattmann et al.
- **Hardware compat:** `docs/HARDWARE_COMPAT.md` in the repo
