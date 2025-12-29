# 🎨 VSCode Quick Start Guide

Three ways to visualize Mandelbrot fractals in VSCode - choose what works best for you!

---

## 🥇 Best: Jupyter Notebook (Inline Viewing)

**Perfect for:** Interactive exploration, experimentation, seeing results immediately

```bash
# 1. Open the notebook
code mandelbrot_explorer.ipynb

# 2. Run cells in order (Shift+Enter or click "Run Cell")
#    - First cell: Checks and installs dependencies automatically
#    - Second cell: Initializes hardware
#    - Remaining cells: Render fractals!
# 3. Plots appear inline automatically!
```

**What you get:**
- ✅ **Automatic dependency installation** (numpy, matplotlib)
- ✅ Inline matplotlib plots (no external windows needed)
- ✅ Classic Mandelbrot views
- ✅ Zoom sequences
- ✅ Julia set comparisons
- ✅ Performance benchmarks
- ✅ Customizable exploration cells

**Requirements:** VSCode with Jupyter extension (should be built-in)

**Note:** The first cell will automatically:
- ✅ Check for and install numpy/matplotlib if missing
- ✅ Check for ttnn (from TT-Metal)
- ✅ Attempt to auto-install ttnn from ~/tt-metal if available
- ⚠️ If ttnn missing, provides clear installation instructions

**If ttnn is not installed:** The cell will detect this and either:
1. Auto-install it from ~/tt-metal (if found)
2. Provide step-by-step instructions to install it manually

Just run the first cell and follow any instructions it displays!

---

## 🥈 Good: Save to Files

**Perfect for:** Batch rendering, creating image libraries, remote work

### Method 1: Dedicated batch script
```bash
python explorer_save.py
```

Generates:
- `./mandelbrot_outputs/mandelbrot_YYYYMMDD_HHMMSS.png` - Classic view
- `./mandelbrot_outputs/mandelbrot_zoom_*.png` - Zoom sequence
- `./mandelbrot_outputs/julia_comparison_*.png` - Julia sets

### Method 2: Main script with flag
```bash
python explorer.py --save
```

Generates:
- `./mandelbrot_outputs/mandelbrot_classic.png`

**View results:** Click any `.png` file in VSCode to preview it!

---

## 🥉 Alternative: Interactive GUI

**Perfect for:** Local machine with display, real-time click-to-zoom

```bash
python explorer.py
```

**Only works if:**
- You're running locally (not SSH)
- You have a display/X11 configured
- VSCode can open GUI windows

**Controls:**
- Click → Zoom 4x
- R → Reset
- C → Cycle colors
- U → Undo
- Q → Quit

---

## 📋 File Summary

| File | Purpose | Output |
|------|---------|--------|
| `mandelbrot_explorer.ipynb` | Interactive notebook | Inline plots |
| `explorer_save.py` | Batch renderer | Multiple PNG files |
| `explorer.py --save` | Quick single render | Single PNG file |
| `explorer.py` | Interactive GUI | GUI window (needs display) |

---

## 💡 Troubleshooting

**Problem:** `python explorer.py` shows no output

**Solution:** You're running headless (no display). Use one of these instead:
```bash
python explorer.py --save          # Save to file
# or
code mandelbrot_explorer.ipynb     # Use notebook
```

---

**Problem:** Dependencies missing in notebook (numpy/matplotlib)

**Solution:** Just run the first cell! It automatically checks and installs numpy/matplotlib if they're missing from your Python environment.

If that doesn't work, manually install:
```bash
pip install -r requirements.txt
# or
pip install numpy matplotlib
```

---

**Problem:** ttnn module not found

**Solution:** The first cell will detect this and try to auto-install from ~/tt-metal.

If auto-install fails or tt-metal not found, install manually:
```bash
# If tt-metal already exists
cd ~/tt-metal
pip install -e .

# If tt-metal doesn't exist yet
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
./install_dependencies.sh
./build_metal.sh
pip install -e .
```

After installation, **restart the notebook kernel** (Kernel → Restart Kernel) and re-run the first cell.

---

**Problem:** Jupyter notebook won't open

**Solution:** Install Jupyter support:
```bash
pip install jupyter ipykernel
# Then reload VSCode
```

---

**Problem:** Images look pixelated in VSCode preview

**Solution:** The images are high-res, but VSCode preview might be zoomed. Try:
- Right-click image → "Open With..." → "Image Preview"
- Use browser: `open mandelbrot_outputs/mandelbrot_*.png`

---

## 🚀 Quick Test

**30-second test to see if everything works:**

```bash
# Install dependencies
pip install -r requirements.txt

# Quick render (saves to file)
python explorer.py --save

# Check output
ls -lh mandelbrot_outputs/
```

Expected: `mandelbrot_classic.png` (several MB)

Then click the file in VSCode to view it! 🎉

---

## 🎯 Recommended Workflow

1. **Start with notebook** (`mandelbrot_explorer.ipynb`)
   - Run the first few cells to see classic views
   - Experiment with the "Custom Region" cell
   - Get a feel for rendering performance

2. **Use save script for production** (`explorer_save.py`)
   - Render high-res images for presentations
   - Generate zoom sequences
   - Create comparison galleries

3. **Interactive mode is optional**
   - Only use if you have local display
   - Great for live demos
   - Not needed for development

---

**Questions? Check `README.md` for full documentation!**
