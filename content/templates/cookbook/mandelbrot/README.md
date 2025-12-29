# Mandelbrot Explorer

Interactive fractal renderer with GPU-style parallel computation on TT hardware.

## Features

- High-resolution Mandelbrot set rendering
- Julia set variations
- Interactive zoom and pan
- Multiple color schemes
- Performance profiling
- VSCode-friendly output (inline + file-based)

## Quick Start

### 🎯 Option 1: Jupyter Notebook (Recommended for VSCode)

Best for interactive exploration with inline visualizations:

```bash
# 1. Ensure ttnn is installed (from TT-Metal)
cd ~/tt-metal
pip install -e .

# 2. Open the notebook in VSCode
code mandelbrot_explorer.ipynb

# 3. Run the first cell - it will:
#    - Auto-check and install numpy/matplotlib
#    - Verify ttnn is available
#    - Auto-install ttnn from ~/tt-metal if needed
#
# 4. Follow any instructions if setup is needed
# 5. Click "Run Cell" on remaining cells to render fractals!
```

The notebook includes:
- **Automatic dependency checking** (numpy, matplotlib, ttnn)
- **Auto-installation** from requirements.txt and ~/tt-metal
- Classic Mandelbrot views
- Zoom sequences into interesting regions
- Julia set comparisons
- Color scheme comparisons
- Performance benchmarking
- Custom exploration template

### 📁 Option 2: Save to Files

For batch rendering or when no display is available:

```bash
# Method 1: Use the dedicated save script
python explorer_save.py
# Outputs saved to ./mandelbrot_outputs/

# Method 2: Use the main script with --save flag
python explorer.py --save
# Outputs saved to ./mandelbrot_outputs/
```

### 🖼️ Option 3: Interactive GUI (if display available)

If you have X11 or are running locally with a display:

```bash
python explorer.py
# Opens interactive matplotlib window with click-to-zoom
```

## File Overview

| File | Purpose |
|------|---------|
| `mandelbrot_explorer.ipynb` | 📓 **Jupyter notebook** - inline visualizations in VSCode |
| `explorer_save.py` | 💾 **Batch renderer** - saves multiple views to files |
| `explorer.py` | 🔄 **Flexible** - interactive or save mode (use `--save` flag) |
| `renderer.py` | ⚙️ **Core engine** - TTNN rendering implementation |

## Usage Examples

### Python API

```python
from renderer import MandelbrotRenderer
from explorer import MandelbrotVisualizer
import ttnn

device = ttnn.open_device(0)
renderer = MandelbrotRenderer(device)

# Render Mandelbrot set
fractal = renderer.render(
    width=2048, height=2048,
    x_min=-2.5, x_max=1.0,
    y_min=-1.25, y_max=1.25,
    max_iter=512
)

# Interactive explorer with click-to-zoom (requires display)
viz = MandelbrotVisualizer(renderer)
viz.interactive_explorer(width=1024, height=1024)

ttnn.close_device(device)
```

### Batch Rendering with File Output

```python
from renderer import MandelbrotRenderer
from explorer_save import MandelbrotVisualizer
import ttnn

device = ttnn.open_device(0)
renderer = MandelbrotRenderer(device)
viz = MandelbrotVisualizer(renderer, output_dir="./my_fractals")

# Render zoom sequence
zoom_points = [
    (-0.5, 0.0, 1),    # Full view
    (-0.7, 0.0, 8),    # 8x zoom
    (-0.75, 0.1, 32),  # 32x zoom
]
viz.render_sequence(zoom_points, width=1024, height=1024)

# Compare Julia sets
c_values = [(-0.4, 0.6), (-0.8, 0.156), (0.285, 0.01)]
viz.compare_julia_sets(c_values, width=512, height=512)

ttnn.close_device(device)
```

## Interactive Controls

When using interactive mode (GUI window):

- **Click**: Zoom into region (4x)
- **R**: Reset to full view
- **C**: Cycle through color maps
- **U**: Undo last zoom
- **Q**: Quit

## VSCode Tips

**For best experience in VSCode:**

1. **Use Jupyter notebook** (`mandelbrot_explorer.ipynb`)
   - Plots appear inline immediately
   - No display server needed
   - Can re-run cells to experiment

2. **File-based output** works great too
   - Run `explorer_save.py` or `explorer.py --save`
   - Click generated PNG files to preview in VSCode
   - Perfect for batch rendering

3. **Image preview** in VSCode
   - Click any `.png` file to see it
   - Use image preview panel for zoom/pan
   - Works on local or remote (SSH) connections

## Interesting Coordinates to Explore

**Mandelbrot set regions:**
- Seahorse Valley: `x=-0.75 to -0.735, y=0.095 to 0.11`
- Elephant Valley: `x=0.25 to 0.35, y=0.0 to 0.1`
- Spiral: `x=-0.7, y=0.27` (zoom in from here)
- Mini Mandelbrot: `x=-0.1592, y=-1.0317` (needs deep zoom)

**Julia set c values:**
- `-0.4 + 0.6i` - Dragon-like pattern
- `-0.8 + 0.156i` - Spiral arms
- `0.285 + 0.01i` - Dendrite structure
- `-0.7269 + 0.1889i` - Douady rabbit
- `-0.835 - 0.2321i` - San Marco fractal

## Complete Implementation

See **Lesson 12** for the full implementation details including:
- Complex number operations with TTNN
- Julia set rendering algorithms
- Interactive explorer with undo/redo
- Performance benchmarking
- Advanced extensions (Burning Ship, 3D Mandelbulb)

## Performance Tips

- Start with 512×512 or 1024×1024 for exploration
- Increase `max_iter` for deeper zooms (typically 2x per zoom level)
- Use `'hot'` or `'viridis'` colormaps for fastest rendering
- Pre-compute multiple zoom levels for smooth animations