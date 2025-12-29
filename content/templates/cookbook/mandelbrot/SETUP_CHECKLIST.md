# ✅ Mandelbrot Explorer Setup Checklist

Quick reference for getting everything working.

## Prerequisites

### ✅ TT-Metal Installation

**Required for ttnn:**

```bash
# 1. Clone tt-metal (if not already done)
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal

# 2. Install system dependencies
cd ~/tt-metal
./install_dependencies.sh

# 3. Build tt-metal
./build_metal.sh

# 4. Install Python package
pip install -e .
```

**Verify installation:**
```bash
python -c "import ttnn; print('✅ ttnn installed')"
```

---

## Jupyter Notebook Setup

### Option A: Let the Notebook Handle It (Recommended)

1. **Open notebook:**
   ```bash
   code mandelbrot_explorer.ipynb
   ```

2. **Run cell 2** (dependency check)
   - Checks numpy, matplotlib, ttnn
   - Auto-installs what it can
   - Shows instructions if manual install needed

3. **Follow any instructions shown**
   - If ttnn auto-installed → Restart kernel, re-run cell 2
   - If manual install needed → Follow displayed steps

4. **Proceed to cell 4** when you see:
   ```
   ✅ All required packages are installed!
   💡 Ready to run the notebook!
   ```

---

### Option B: Manual Pre-Installation

1. **Install all dependencies first:**
   ```bash
   # Standard packages
   pip install numpy matplotlib

   # ttnn from tt-metal
   cd ~/tt-metal
   pip install -e .
   ```

2. **Open and run notebook:**
   ```bash
   code mandelbrot_explorer.ipynb
   # Run cell 2 → Should show all ✅
   # Proceed to cell 4 and beyond
   ```

---

## File-Based Scripts Setup

For `explorer.py --save` or `explorer_save.py`:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Install ttnn
cd ~/tt-metal
pip install -e .

# 3. Run scripts
python explorer.py --save
# or
python explorer_save.py
```

---

## Quick Verification

### Test All Dependencies
```bash
python -c "import numpy; print('✅ numpy')"
python -c "import matplotlib; print('✅ matplotlib')"
python -c "import ttnn; print('✅ ttnn')"
```

### Test Notebook
```bash
code mandelbrot_explorer.ipynb
# Run cell 2, expect all ✅
```

### Test File Mode
```bash
python explorer.py --save
ls -lh mandelbrot_outputs/mandelbrot_classic.png
```

---

## Common Issues

### ❌ "ttnn module not found"

**Cause:** ttnn not installed in current Python environment

**Fix:**
```bash
cd ~/tt-metal
pip install -e .
```

If tt-metal doesn't exist:
```bash
git clone https://github.com/tenstorrent/tt-metal.git ~/tt-metal
cd ~/tt-metal
./install_dependencies.sh
./build_metal.sh
pip install -e .
```

---

### ❌ "No module named matplotlib"

**In notebook:** Run cell 2 - it auto-installs

**In terminal:**
```bash
pip install matplotlib
```

---

### ❌ Notebook shows "Kernel Error" or package not found after installation

**Fix:** Restart the notebook kernel
1. Click "Kernel" menu or "..." in VSCode
2. Select "Restart Kernel"
3. Re-run dependency check cell

---

### ❌ `pip install -e .` fails in tt-metal

**Fix:** Make sure you built tt-metal first:
```bash
cd ~/tt-metal
./install_dependencies.sh
./build_metal.sh
# Then try pip install again
pip install -e .
```

---

## Success Indicators

### ✅ Notebook Ready
```
🔍 Checking dependencies...

✅ numpy is installed
✅ matplotlib is installed
✅ ttnn is installed

✅ All required packages are installed!
💡 Ready to run the notebook!
```

### ✅ File Mode Ready
```bash
$ python explorer.py --save
📁 Running in save mode...
Rendering 2048×2048 image...
✅ Saved to: ./mandelbrot_outputs/mandelbrot_classic.png
```

---

## Environment Notes

### Using Virtual Environments

If using venv/conda:
```bash
# Activate your environment
source ~/myenv/bin/activate  # or conda activate myenv

# Install everything IN that environment
pip install numpy matplotlib
cd ~/tt-metal && pip install -e .

# Make sure VSCode uses the same environment
# (Select kernel in notebook: top-right corner)
```

### Using System Python

Works fine, but:
- May need `--user` flag: `pip install --user numpy matplotlib`
- ttnn typically installed per-user anyway

---

## Quick Start Command Sequence

**For impatient users:**

```bash
# Install everything
pip install numpy matplotlib && cd ~/tt-metal && pip install -e . && cd -

# Open notebook
code mandelbrot_explorer.ipynb
# Run cell 2, verify ✅, proceed to cell 4+

# Or use file mode
python explorer.py --save && ls -lh mandelbrot_outputs/
```

Done! 🎉
