# TT-Metalium Cookbook Projects

This directory contains 4 complete, hands-on projects for learning TT-Metalium programming.

## Projects

### 1. 🎮 Game of Life (`game_of_life/`)
Conway's cellular automaton demonstrating parallel tile computing.

**Files:**
- `game_of_life.py` - Core implementation using TTNN convolution
- `patterns.py` - Classic patterns (glider, blinker, Gosper gun, etc.)
- `visualizer.py` - Matplotlib animation
- `requirements.txt` - Dependencies

**Run:**
```bash
cd game_of_life
pip install -r requirements.txt
python game_of_life.py
```

### 2. 🎵 Audio Processor (`audio_processor/`)
Real-time audio signal processing with mel-spectrograms and effects.

**Files:**
- `processor.py` - Core TTNN audio operations (STFT, mel-spectrogram, MFCC)
- `effects.py` - Audio effects (reverb, pitch shift, echo, chorus)
- `visualizer.py` - Real-time visualization
- `examples/sample.wav` - Example audio file
- `requirements.txt` - Dependencies

**Run:**
```bash
cd audio_processor
pip install -r requirements.txt
python processor.py examples/sample.wav
```

### 3. 🌀 Mandelbrot Explorer (`mandelbrot/`)
Interactive fractal renderer with GPU-style parallel computation.

**Files:**
- `renderer.py` - Mandelbrot/Julia set rendering
- `explorer.py` - Interactive zoom/pan interface
- `requirements.txt` - Dependencies

**Run:**
```bash
cd mandelbrot
pip install -r requirements.txt
python explorer.py
```

### 4. 🖼️ Image Filters (`image_filters/`)
Creative image processing with convolution kernels.

**Files:**
- `filters.py` - Filter bank (edge detect, blur, sharpen, emboss, oil painting)
- `examples/sample.jpg` - Example image
- `requirements.txt` - Dependencies

**Run:**
```bash
cd image_filters
pip install -r requirements.txt
python filters.py examples/sample.jpg
```

## Quick Start

### Deploy All Projects

The VS Code extension can deploy all projects to `~/tt-scratchpad/cookbook/`:

1. Open Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Run: `Tenstorrent: Create Cookbook Projects`
3. Navigate to `~/tt-scratchpad/cookbook/`

### Manual Installation

```bash
# Copy to scratchpad
cp -r . ~/tt-scratchpad/cookbook/

# Install dependencies for all projects
cd ~/tt-scratchpad/cookbook
for project in game_of_life audio_processor mandelbrot image_filters; do
    cd $project
    pip install -r requirements.txt
    cd ..
done
```

## Learning Path

**Beginner:**
1. Start with Game of Life - simple rules, visual output
2. Then Mandelbrot - GPU-style parallel compute

**Intermediate:**
3. Audio Processor - real-world DSP application
4. Image Filters - computer vision foundation

**Advanced:**
- Combine projects (audio-reactive fractals!)
- Extend with custom features
- Optimize performance
- Contribute back to tt-metal

## Key Concepts Taught

- **TTNN Operations**: `conv2d`, `matmul`, `fft`, tensor manipulation
- **Tile-Based Computing**: 32×32 tiles, padding, layout conversion
- **Parallel Execution**: Leveraging multiple Tensix cores
- **Memory Management**: L1 SRAM, DRAM transfers
- **Performance Optimization**: Tile alignment, batch processing

## Common Dependencies

All projects require:
```bash
pip install ttnn torch numpy matplotlib
```

Project-specific:
- Audio: `librosa scipy sounddevice`
- Image: `Pillow opencv-python`

## Troubleshooting

**Device not found:**
```bash
tt-smi  # Check device status
tt-smi -r  # Reset if needed
```

**Out of memory:**
- Reduce grid/image sizes
- Use tile-aligned dimensions (multiples of 32)
- Deallocate tensors when done

**Slow performance:**
- Ensure TILE_LAYOUT is used
- Check tile alignment
- Minimize host↔device transfers

## Resources

- **Lesson 11**: Exploring TT-Metalium architecture
- **Lesson 12**: Complete cookbook with detailed explanations
- **Discord**: [discord.gg/tvhGzHQwaj](https://discord.gg/tvhGzHQwaj)
- **Docs**: [docs.tenstorrent.com](https://docs.tenstorrent.com)
