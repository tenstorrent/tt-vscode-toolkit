---
id: video-generation-ttmetal
title: Video Generation via Frame-by-Frame SDXL
description: >-
  Create videos by generating frames with Stable Diffusion XL on Tenstorrent hardware.
  Demonstrates hardware scaling from N150 to Galaxy - same code, exponentially faster performance!
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
status: draft
estimatedMinutes: 30
---

# Video Generation via Frame-by-Frame SDXL

## The Hardware Scaling Story

**Core Philosophy:** Write code for N150 (smallest hardware), watch it run exponentially faster on larger hardware.

This lesson demonstrates the Tenstorrent hardware advantage:
- **Same code** runs on all hardware tiers
- **N150 (1 chip):** Works (baseline)
- **N300 (2 chips):** ~2x faster
- **T3K (8 chips):** ~6x faster
- **Galaxy (32 chips):** ~20x faster

**No code changes needed** - just better hardware!

---

## What We'll Build

A 10-second video showcasing "Tenstorrent at the 1964-1965 World's Fair" using:
- **Stable Diffusion XL Base** (from Lesson 9 - image-generation)
- **Frame-by-frame generation** (10 keyframes at 1024×1024)
- **Hardware verification** (ensuring TT inference, not CPU fallback)
- **ffmpeg stitching** (frames → smooth video)

---

## Prerequisites

- ✅ Completed Lesson 9 (Image Generation with SDXL)
- ✅ tt-metal v0.65.1+ installed at `~/tt-metal`
- ✅ Hardware: N150, N300, T3K, or P100
- ✅ ffmpeg installed (for video stitching)
- ✅ diffusers, transformers packages installed

---

## Step 1: Understanding Video from Frames

**Why frame-by-frame instead of native video?**

Native video generation models (like **Mochi** in `tt_dit`) are available but experimental. Frame-by-frame generation using proven **SDXL** offers:

1. Generate individual 1024×1024 frames with SDXL (proven in Lesson 9)
2. Stitch frames together with ffmpeg
3. Perfect for demonstrating hardware scaling!

**This approach:**
- ✅ Uses production-ready SDXL (validated in Lesson 9)
- ✅ Exercises TT hardware for each frame (true hardware acceleration)
- ✅ Demonstrates linear hardware scaling (2x chips = ~2x faster)
- ✅ Produces high-quality 1024×1024 video

**💡 Note:** For native video generation, check out **Mochi** in `models/experimental/tt_dit/pipelines/mochi/` (experimental as of v0.65.1).

---

## Step 2: Hardware Detection

First, verify your hardware tier:

[🔍 Detect Hardware](command:tenstorrent.runHardwareDetection)

Look for "Board Type" in the output (N150, N300, T3K, P100).

**Set MESH_DEVICE for your hardware:**

### N150 (Wormhole - Single Chip) - Most Common
```bash
export MESH_DEVICE=N150
```
**Expected performance:** ~4 minutes per 1024x1024 frame (first frame ~5 min with compilation)

---

### N300 (Wormhole - Dual Chip)
```bash
export MESH_DEVICE=N300
```
**Expected performance:** ~2 minutes per frame (~2x faster than N150)

---

### T3K (Wormhole - 8 Chips)
```bash
export MESH_DEVICE=T3K
```
**Expected performance:** ~40 seconds per frame (~6x faster than N150)

---

### P100 (Blackhole - Single Chip)
```bash
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole
```
**Expected performance:** ~4 minutes per frame (similar to N150)

---

## Step 3: Create Video Prompts

Create a file with your video's "storyboard" - one prompt per frame:

```bash
mkdir -p ~/tt-scratchpad
cat > ~/tt-scratchpad/video_prompts.txt << 'EOF'
# Tenstorrent at 1964-1965 World's Fair

Tenstorrent pavilion at 1964 World's Fair, futuristic dome architecture, orange and white corporate colors, crowds in 1960s attire, Kodachrome photo

Vintage 1964 corporate display, Tenstorrent AI accelerator prototype, blinking lights, orange circuit boards, businessmen in suits examining technology, documentary photography

1960s scientist demonstrating Tenstorrent neural network computer, mainframe-style cabinet with orange panels, oscilloscope displays, amazed visitors watching, retro-futurism

1964 Tenstorrent brochure design, geometric mid-century modern graphics, orange and teal color scheme, optimistic corporate advertising aesthetic

Tenstorrent executives presenting at 1964 World's Fair press conference, vintage microphones, presentation boards, journalists with cameras

Children and families interacting with Tenstorrent AI demonstration, 1960s interactive console, colorful buttons and displays, educational exhibit aesthetic

Tenstorrent computing center at World's Fair, rows of AI accelerator cabinets, operators in white coats, blinking lights, 1960s corporate technology photography

Tenstorrent pavilion at night, illuminated dome, World's Fair Unisphere in background, crowds, neon signs, vintage night photography

Futuristic prediction display, 1960s interpretation of future technology, retro-futuristic artwork, optimistic mid-century illustration style

Thank you for visiting Tenstorrent, 1964 corporate signage, World's Fair closing ceremony atmosphere, nostalgic vintage photograph, orange sunset lighting
EOF
```

**Customize these prompts** for your own video theme!

---

## Step 4: Generate Frames

Now generate each frame using SD 3.5 interactive mode:

```bash
cd ~/tt-scratchpad
export PYTHONPATH=~/tt-metal:$PYTHONPATH
pytest ~/tt-metal/models/experimental/stable_diffusion_35_large/demo.py
```

[🎨 Start Interactive Generation](command:tenstorrent.startInteractiveImageGen)

**When prompted:**
1. Enter first prompt from `video_prompts.txt`
2. Wait for generation (~30-45s on N150)
3. **Watch the Output Preview panel** - your generated image will automatically appear!
4. Note the output filename
5. Enter next prompt
6. Repeat for all 10 prompts

**The Output Preview Panel:**
- Located in the Tenstorrent sidebar (left panel)
- Automatically displays each generated frame
- Double-click image to open in editor
- Right-click to save or copy

**Tip:** Rename generated files sequentially:
```bash
mv sd35_1024_1024.png frame_000.png
mv sd35_1024_1024.png frame_001.png
# ... etc
```

**Example output:**

Each frame will be a high-quality 1024x1024 image capturing the retro-futuristic vision of Tenstorrent at the 1964-65 World's Fair. The Output Preview panel lets you see your progress in real-time!

---

## Step 5: Verify Hardware Utilization

**Critical:** Ensure frames are generated on TT hardware, not CPU!

During generation, you should see:
```
✓ Model loaded on TT hardware
Generating 1024x1024 image (28 inference steps)...
```

**If you see CPU warnings** or extremely slow generation (>5 minutes), something is wrong.

---

## Step 6: Stitch Frames into Video

Once all frames are generated, use ffmpeg to create the video:

```bash
cd ~/tt-scratchpad

# Stitch at 2 fps (each frame shows for 0.5 seconds)
ffmpeg -framerate 2 -pattern_type glob -i 'frame_*.png' \
  -vf 'format=yuv420p,scale=1024:1024' \
  -c:v libx264 -crf 18 \
  tenstorrent_worldsfair_1964.mp4
```

**Parameters explained:**
- `-framerate 2`: 2 frames per second (10 frames = 5 second video)
- `-pattern_type glob`: Use glob pattern for input files
- `-i 'frame_*.png'`: Input pattern (matches frame_000.png, frame_001.png, etc.)
- `-vf 'format=yuv420p,scale=1024:1024'`: Video filter for compatibility
- `-c:v libx264`: H.264 codec (widely compatible)
- `-crf 18`: Quality (lower = better, 18 is high quality)

**Result:** `tenstorrent_worldsfair_1964.mp4` - your video!

**View your video:**

The completed video will automatically appear in the **Output Preview panel**! The preview pane now supports video playback with controls:
- ▶️ Play/Pause controls
- 🔁 Auto-loop enabled
- 📺 Full video player in the sidebar
- Double-click to open in default video player

Your retro-futuristic Tenstorrent World's Fair video ad is complete!

---

## Step 7: Understanding the Scaling

**Benchmark your generation:**

If you generated 10 frames on N150 and each took ~4 minutes:
- **Total time:** 2400 seconds (~40 minutes)

**Same code on larger hardware:**
- **N300 (2 chips):** ~1200 seconds (~20 minutes) - **2x faster**
- **T3K (8 chips):** ~400 seconds (~6.7 minutes) - **6x faster**
- **Galaxy (32 chips):** ~120 seconds (~2 minutes) - **20x faster**

**This is the TT hardware advantage:** Write for N150, scale to Galaxy with zero code changes!

**Note:** First frame includes model download and kernel compilation (~4-5 minutes). Subsequent frames are faster as compilation is cached. The timings above reflect warm runs after initial setup.

---

## Troubleshooting

### "Generation is very slow (>10 minutes per frame)"
**Likely cause:** Running on CPU, not TT hardware

**Fix:**
1. Check `MESH_DEVICE` is set correctly
2. Verify tt-metal installation: `tt-smi`
3. Check model loaded on TT hardware (look for confirmation in output)

**Note:** Normal N150 performance is ~4 minutes per frame. If you're seeing >10 minutes, it's likely running on CPU.

### "Out of memory errors"
**Solution:** Use smaller hardware config or reduce concurrent operations

### "Frame quality inconsistent"
**Solution:** Use consistent seeds or add more prompt detail

### "Generation stuck or process killed - can't restart"
**Problem:** After killing a stuck generation process, device may be in bad state

**Fix:**
1. Reset the device: `tt-smi -r`
2. Wait for reset to complete (~30 seconds)
3. Rerun your script - it will resume from last completed frame

**How resume works:**
- Script automatically detects existing `frame_*.png` files
- Skips already-generated frames
- Continues from where you left off
- No manual intervention needed!

**Example:**
```bash
# If you have frame_000.png through frame_003.png
# Script automatically resumes from frame_004.png
tt-smi -r  # Reset device first
python3 ~/tt-scratchpad/generate_video_frames.py  # Auto-resumes!
```

---

## Key Takeaways

- ✅ **Frame-by-frame video** works with SD 3.5 on TT hardware
- ✅ **Hardware scaling** is automatic - same code, better performance
- ✅ **Write for N150** - smallest hardware, guaranteed to work everywhere
- ✅ **Production-ready** - proven approach from Lesson 9
- ✅ **High quality** - 1024x1024 frames, state-of-the-art SD 3.5

**The TT hardware story:** Start small, scale big - effortlessly!

---

## What's Next?

### Experiment with:
1. **Different frame rates** - Try 1 fps (slower) or 5 fps (faster)
2. **Transition frames** - Generate intermediate frames between keyframes
3. **Longer videos** - 30+ frames for more complex stories
4. **Custom themes** - Your own video concepts

### Future Improvements:
- Native video generation models (when available)
- Temporal consistency between frames
- Automated batch generation
- Real-time video generation on Galaxy

---

## Resources

- **Lesson 9:** Image Generation with SD 3.5 (foundation for this lesson)
- **tt-metal Docs:** https://docs.tenstorrent.com/
- **ffmpeg Guide:** https://ffmpeg.org/documentation.html
- **SD 3.5:** https://huggingface.co/stabilityai/stable-diffusion-3.5-large

---

**Happy video creating! 🎬**
