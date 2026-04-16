---
id: video-generation-ttmetal
title: Video Generation via Frame-by-Frame Diffusion
description: >-
  Create videos by generating frames with Stable Diffusion on Tenstorrent hardware.
  Demonstrates hardware scaling from N150 to T3K - same code, faster performance!
category: serving
tags:
  - hardware
  - video
  - image
  - generation
  - diffusion
  - stable
  - scaling
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p300c
status: draft
estimatedMinutes: 60
---

# Video Generation via Frame-by-Frame Diffusion

## The Idea

**Generate video one frame at a time, then stitch with ffmpeg.**

Rather than a native video generation model, this lesson uses the proven
**Stable Diffusion 1.4** model that runs on every Tenstorrent chip — from
a single N150 to a T3K (8 chips). Each frame is a text-to-image generation
pass. A handful of carefully worded prompts becomes a short film.

**Hardware this works on:**
- **N150 / N300** (Wormhole): `models/demos/wormhole/stable_diffusion/`
- **P100 / P300c** (Blackhole): same demo, set `TT_METAL_ARCH_NAME=blackhole`
- **T3K (8 chips)**: same code, passes more context in parallel

**Requires `~/tt-metal` built from source.** If you don't have that yet, see
[Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"]) first.

---

## What We'll Build

A short video showcasing "Tenstorrent at the 1964–1965 World's Fair" using:
- **Stable Diffusion 1.4** (512×512 images, runs on single-chip hardware)
- **10 frames** generated from creative prompts
- **ffmpeg** to stitch frames into a smooth video

---

## Prerequisites

- ✅ `~/tt-metal` built from source (see Build tt-metal lesson)
- ✅ tt-metal Python venv activated (`source ~/tt-metal/python_env/bin/activate`)
- ✅ Hardware: N150, N300, T3K, P100, or P300c
- ✅ HuggingFace account with `hf auth login` completed (for model download)
- ✅ ffmpeg installed

Install ffmpeg if needed:
```bash
sudo apt-get install -y ffmpeg
```

---

## Step 1: Set Up Environment

```bash
source ~/tt-metal/python_env/bin/activate
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
cd ~/tt-metal
```

**For P100 / P300c (Blackhole):**
```bash
export TT_METAL_ARCH_NAME=blackhole
```

---

## Step 2: Authenticate with HuggingFace

The demo auto-downloads `CompVis/stable-diffusion-v1-4` on first run. You need
a HuggingFace account and to be logged in:

```bash
hf auth login
# Enter your HuggingFace token when prompted
```

Verify:
```bash
hf auth whoami
```

---

## Step 3: Create Your Video Prompts File

Create a JSON file describing your 10 frames. Each object has a single `"prompt"` key.

```bash
mkdir -p ~/tt-scratchpad/worldsfair-video
cat > ~/tt-scratchpad/worldsfair-video/prompts.json << 'EOF'
[
  {"prompt": "Tenstorrent pavilion at 1964 World's Fair, futuristic dome architecture, orange and white corporate colors, crowds in 1960s attire, Kodachrome photo"},
  {"prompt": "vintage 1964 corporate display, Tenstorrent AI accelerator prototype, blinking lights, orange circuit boards, businessmen in suits examining technology, documentary photography"},
  {"prompt": "1960s scientist demonstrating Tenstorrent neural network computer, mainframe-style cabinet with orange panels, oscilloscope displays, amazed visitors, retro-futurism"},
  {"prompt": "1964 Tenstorrent brochure design, geometric mid-century modern graphics, orange and teal color scheme, optimistic corporate advertising aesthetic"},
  {"prompt": "Tenstorrent executives presenting at 1964 World's Fair press conference, vintage microphones, presentation boards, journalists with cameras"},
  {"prompt": "children and families interacting with Tenstorrent AI demonstration, 1960s interactive console, colorful buttons and displays, educational exhibit"},
  {"prompt": "Tenstorrent computing center at World's Fair, rows of AI accelerator cabinets, operators in white coats, blinking lights, 1960s corporate technology photography"},
  {"prompt": "Tenstorrent pavilion at night, illuminated dome, World's Fair Unisphere in background, neon signs, vintage night photography"},
  {"prompt": "futuristic prediction display, 1960s interpretation of future technology, retro-futuristic artwork, optimistic mid-century illustration style"},
  {"prompt": "thank you for visiting Tenstorrent, 1964 corporate signage, World's Fair closing ceremony, nostalgic vintage photograph, orange sunset lighting"}
]
EOF
```

Feel free to replace these with your own theme!

---

## Step 4: Generate Frames

Run the batch demo with your prompts file. The demo downloads `CompVis/stable-diffusion-v1-4`
on first run (a few hundred MB) then compiles kernels before the first image.

```bash
cd ~/tt-metal
pytest --disable-warnings \
  --input-path="$HOME/tt-scratchpad/worldsfair-video/prompts.json" \
  models/demos/wormhole/stable_diffusion/demo/demo.py::test_demo
```

**What happens:**
1. Model weights download (first run only, ~2 min)
2. Kernel compilation runs (first run only, ~5 min)
3. Each frame generates in sequence
4. Images save to the current directory as:
   - `input_data_0_512x512_ttnn.png`
   - `input_data_1_512x512_ttnn.png`
   - ...
   - `input_data_9_512x512_ttnn.png`

> **Note:** The default output path is the directory where you run pytest.
> Run `cd ~/tt-metal` first so images land there, then move them afterward.

**Expected generation time per frame (after compilation):**
- **N150:** ~30–45 seconds per 512×512 frame
- **N300:** ~15–25 seconds per frame (2 chips)
- **T3K:** ~5–10 seconds per frame (8 chips)
- **P100 / P300c:** ~30–45 seconds per frame (similar to N150)

> **First frame is always slower** — kernel compilation adds 3–5 minutes on initial run.
> Subsequent runs use cached compiled kernels and are much faster.

---

## Step 5: Collect Your Frames

Move the generated images to your video directory and rename them sequentially:

```bash
mkdir -p ~/tt-scratchpad/worldsfair-video/frames
cd ~/tt-metal

for i in $(seq 0 9); do
  if [ -f "input_data_${i}_512x512_ttnn.png" ]; then
    mv "input_data_${i}_512x512_ttnn.png" \
       ~/tt-scratchpad/worldsfair-video/frames/frame_$(printf "%03d" $i).png
    echo "Moved frame $i"
  fi
done

ls ~/tt-scratchpad/worldsfair-video/frames/
```

---

## Step 6: Stitch Frames into Video

```bash
cd ~/tt-scratchpad/worldsfair-video

# 2 fps = each frame shows for 0.5 seconds → 10 frames = 5 second video
ffmpeg -framerate 2 \
  -pattern_type glob -i 'frames/frame_*.png' \
  -vf 'format=yuv420p,scale=512:512' \
  -c:v libx264 -crf 18 \
  tenstorrent_worldsfair_1964.mp4

echo "Done! Video saved as tenstorrent_worldsfair_1964.mp4"
```

**ffmpeg parameters:**
- `-framerate 2`: 2 frames per second (slow, cinematic for still images)
- `-pattern_type glob`: match files with wildcard
- `format=yuv420p`: broad player compatibility
- `-crf 18`: high quality (lower = better, 18 is near-lossless)

Try `-framerate 4` or `-framerate 1` for different pacing.

---

## Step 7: Try the Interactive Mode

For a more exploratory workflow — type a prompt, see the image immediately:

```bash
cd ~/tt-metal
pytest models/demos/wormhole/stable_diffusion/demo/demo.py::test_interactive_demo
```

The model stays loaded between prompts. Type a prompt, press Enter, and
`interactive_512x512_ttnn.png` (or `interactive_256x256_ttnn.png`) appears in the
current directory. Type `q` to exit.

Use this to iterate on your prompt wording before committing to a full 10-frame batch.

---

## Understanding Hardware Scaling

The same demo code runs across hardware tiers — the difference is parallelism:

| Hardware | Chips | Relative Speed | Use Case |
|----------|-------|---------------|----------|
| N150 | 1 | 1× (baseline) | Development, testing |
| N300 | 2 | ~2× faster | Faster iteration |
| T3K | 8 | ~6× faster | Production video |
| P100 / P300c | 1 BH | ~1× | Blackhole validation |

**Benchmark example (10 frames at 512×512):**
- N150: ~350 seconds total (~35s/frame after warmup)
- N300: ~200 seconds total
- T3K: ~70 seconds total

This is the TT hardware advantage: **write for N150, scale to T3K with zero code changes.**

---

## Customize Your Video

### Adjust Frame Timing (ffmpeg)
```bash
# Slower: 1 fps (1 second per frame)
ffmpeg -framerate 1 -pattern_type glob -i 'frames/frame_*.png' \
  -vf 'format=yuv420p' -c:v libx264 -crf 18 output_slow.mp4

# Faster: 4 fps
ffmpeg -framerate 4 -pattern_type glob -i 'frames/frame_*.png' \
  -vf 'format=yuv420p' -c:v libx264 -crf 18 output_fast.mp4
```

### Add a Crossfade Transition
```bash
# Use the minterpolate filter to interpolate between frames
ffmpeg -framerate 2 -pattern_type glob -i 'frames/frame_*.png' \
  -vf 'minterpolate=fps=24:mi_mode=mci,format=yuv420p' \
  -c:v libx264 -crf 18 output_smooth.mp4
```

### Different Themes
Replace the prompts JSON with any theme. Some ideas:
- Historical photographs of your city
- A product launch progression
- An abstract art series ("impressionist painting of X in storm/calm/sunset")
- A journey through a landscape

---

## Troubleshooting

**"Generation is very slow (>5 min per frame after warmup)"**

Likely running on CPU, not TT hardware:
1. Check `TT_METAL_HOME` is set: `echo $TT_METAL_HOME`
2. Verify venv is activated: `which python3` should show `tt-metal/python_env/`
3. Check device: `tt-smi -s | grep board_type`

**"Device in bad state after a killed process"**
```bash
tt-smi -r   # Reset device
# Wait ~30 seconds
# Re-run pytest
```

**"Module not found" errors**
```bash
source ~/tt-metal/python_env/bin/activate
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**"huggingface-hub not authenticated"**
```bash
hf auth login
```

**"P300c / QB2: ROW dispatch error"**

The SD demo uses `ttnn.open_device()` without hardcoding dispatch axis, so it should be Blackhole-safe. If you see dispatch errors, ensure `TT_METAL_ARCH_NAME=blackhole` is set.

**"Ran out of frames to resume from"**

If a run was interrupted, the demo can re-run and will overwrite existing files.
Rename already-completed frames before re-running to preserve them.

---

## Key Takeaways

- ✅ **SD 1.4 runs on all single-chip Tenstorrent hardware** (N150, P300c, P100)
- ✅ **Frame-by-frame video** is a practical approach for hardware without native video models
- ✅ **Hardware scaling is automatic** — same pytest command, better hardware = faster
- ✅ **First-run warmup** (download + compilation) is a one-time cost; subsequent runs are fast
- ✅ **ffmpeg stitching** turns a folder of images into a shareable video in seconds

---

## What's Next?

- **AnimateDiff** — For native video animation with temporal consistency between frames:
  [Native Video Animation with AnimateDiff](command:tenstorrent.showLesson?["animatediff-video-generation"])
- **Image Generation** — Single image generation with Stable Diffusion:
  [Image Generation](command:tenstorrent.showLesson?["image-generation"])
- **Explore Metalium** — Understand the architecture running under these demos:
  [Exploring TT-Metalium](command:tenstorrent.showLesson?["explore-metalium"])
