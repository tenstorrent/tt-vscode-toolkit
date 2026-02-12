---
id: cookbook-overview
title: Tenstorrent Cookbook Overview
description: >-
  Welcome to the Tenstorrent Cookbook! Build 5 complete projects that teach fundamental TT-Metal techniques: Conway's Game of Life, Audio Signal Processing, Mandelbrot Fractals, Image Filters, and Particle Life. Each recipe is a standalone lesson with full source code and visual output.
category: cookbook
tags:
  - ttnn
  - projects
  - learning
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300
  - galaxy
status: validated
validatedOn:
  - n150
  - p300
estimatedMinutes: 5
---

# Tenstorrent Cookbook: Learn by Building

Welcome to the Tenstorrent Cookbook! This series of hands-on projects teaches you TT-Metal fundamentals through creative, visual applications.

## What You'll Build

Each recipe is a complete, working project that demonstrates core TT-Metal techniques:

### 🎮 [Recipe 1: Conway's Game of Life](command:tenstorrent.showLesson?["cookbook-game-of-life"])
**Learn:** Cellular automata, parallel tile computing, convolution operations

Build the classic cellular automaton where simple rules create complex emergent behavior. Perfect introduction to parallel computation on TT hardware.

**Time:** 30 minutes | **Difficulty:** Beginner

---

### 🎵 [Recipe 2: Audio Signal Processing](command:tenstorrent.showLesson?["cookbook-audio-processor"])
**Learn:** FFT, mel-spectrograms, real-time audio effects

Process audio files on TT hardware - compute spectrograms, detect beats, extract pitch. Foundation for speech recognition models like Whisper.

**Time:** 45 minutes | **Difficulty:** Intermediate

---

### 🌀 [Recipe 3: Mandelbrot Fractal Explorer](command:tenstorrent.showLesson?["cookbook-mandelbrot"])
**Learn:** GPU-style parallel computation, complex number operations

Render beautiful fractals with an interactive zoom explorer. Demonstrates embarrassingly parallel computation - ideal for understanding performance.

**Time:** 30 minutes | **Difficulty:** Beginner-Intermediate

---

### 🖼️ [Recipe 4: Custom Image Filters](command:tenstorrent.showLesson?["cookbook-image-filters"])
**Learn:** 2D convolution, computer vision kernels

Build a library of image filters from edge detection to artistic effects. Same techniques used in ResNet50, MobileNetV2, and ViT models.

**Time:** 30 minutes | **Difficulty:** Beginner-Intermediate

---

### 🌌 [Recipe 5: Particle Life - Emergent Complexity](command:tenstorrent.showLesson?["cookbook-particle-life"])
**Learn:** N² algorithms, physics simulation, multi-device parallelization

Simulate emergent patterns from simple particle interactions. Includes multi-chip acceleration bonus for QuietBox systems!

**Time:** 30 minutes | **Difficulty:** Intermediate

---

## How the Cookbook Works

### 1. Deploy All Projects
[📦 Deploy All Cookbook Projects](command:tenstorrent.createCookbookProjects)

This creates all 5 projects in `~/tt-scratchpad/cookbook/` with one command.

### 2. Pick a Recipe
Choose any recipe - they're independent and can be completed in any order.

### 3. Learn by Doing
Each recipe includes:
- ✅ Complete source code with detailed comments
- ✅ Visual output (animations, plots, fractals)
- ✅ Extension ideas for further exploration
- ✅ One-click execution buttons

### 4. Experiment and Extend
Modify the code, try different parameters, combine projects. The cookbook is your playground!

---

## Building Blocks for Real Models

These recipes teach fundamental techniques used in production models:

| Recipe | Technique | Used In |
|--------|-----------|---------|
| Game of Life | Convolution | YOLO v10-v12, SegFormer |
| Audio Processor | Spectrograms | Whisper speech recognition |
| Mandelbrot | Parallel pixel processing | Stable Diffusion 3.5 |
| Image Filters | 2D convolutions | ResNet50, MobileNetV2, ViT |
| Particle Life | N² algorithms | Physics simulation, pairwise interactions |

---

## What You'll Learn

After completing the cookbook, you'll have:

- ✅ **5 complete, working projects** you built yourself
- ✅ **Deep understanding of TTNN operations** (convolution, FFT, parallel compute)
- ✅ **Experience with parallel tile computing** optimized for TT hardware
- ✅ **Foundation for production applications** and model bring-up

---

## Next Steps After Cookbook

### Combine Projects
- Use audio processor to trigger mandelbrot zoom
- Apply image filters to Game of Life visualization
- Real-time audio-reactive fractal visualizations

### Explore Production Models
- **Convolution** → `models/demos/yolov12x/` (object detection)
- **Transformers** → `models/demos/gemma3/` (multimodal AI)
- **Vision** → `models/demos/mobilenetv2/` (mobile inference)
- **Audio** → `models/demos/whisper/` (speech recognition)

### Contribute
- Submit your projects as examples
- Participate in bounty program
- Help other developers on Discord

---

## Resources

- **Discord**: [discord.gg/tvhGzHQwaj](https://discord.gg/tvhGzHQwaj)
- **GitHub**: [github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)
- **Documentation**: [docs.tenstorrent.com](https://docs.tenstorrent.com)
- **METALIUM_GUIDE.md**: `~/tt-metal/METALIUM_GUIDE.md` - Architecture deep-dive
- **Tech Reports**: `~/tt-metal/tech_reports/` - Research papers and optimizations

---

Ready to start? Pick a recipe above or [deploy all projects](command:tenstorrent.createCookbookProjects) and dive in!

Happy coding! 🚀
