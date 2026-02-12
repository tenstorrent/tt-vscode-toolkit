# VSCode Visualization Bridge

**Python→VSCode API integration for matplotlib visualizations**

## Overview

The **TT-VSCode Bridge** enables Python CLI scripts (like cookbook recipes) to display matplotlib plots directly in the VSCode extension's **Output Preview** panel, creating a seamless integrated visualization experience.

**Key Features:**
- 🎯 **Zero configuration** - Just import and use
- 🔄 **Auto-refresh** - Perfect for animations and real-time updates
- 🛡️ **Non-breaking fallback** - Gracefully degrades when VSCode unavailable
- 🚀 **High performance** - Async command execution, non-blocking
- 🎨 **Format support** - PNG, JPG, SVG, GIF, and more
- 📊 **Animation-ready** - Built-in frame update support

## Architecture

### Components

1. **VSCode Extension Side** (`extension.ts`)
   - New command: `tenstorrent.showVisualization`
   - Takes absolute path to image file
   - Displays in `TenstorrentImagePreviewProvider` webview panel

2. **Python Side** (`tt_vscode_bridge.py`)
   - Helper module for Python scripts
   - Auto-detects VSCode availability
   - Calls VSCode command via CLI
   - Graceful fallback to file-based display

### Flow Diagram

```
┌─────────────────────┐
│  Python CLI Script  │
│                     │
│  plt.savefig(...)   │
│  show_plot(path)    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────┐
│  tt_vscode_bridge.py    │
│                         │
│  1. Check VSCode        │
│  2. Call CLI command    │
│  3. Fallback if needed  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────────────┐
│  VSCode CLI                     │
│                                 │
│  code --command                 │
│    tenstorrent.showVisualization│
│    /path/to/image.png           │
└──────────┬──────────────────────┘
           │
           ▼
┌───────────────────────────────────┐
│  VSCode Extension                 │
│                                   │
│  showVisualization(imagePath)     │
│    ├─ Validate path               │
│    ├─ Check file format           │
│    └─ Call imagePreviewProvider   │
└──────────┬────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│  TenstorrentImagePreviewProvider    │
│                                     │
│  showImage(path)                    │
│    └─ Render in webview panel      │
└─────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────┐
│  Output Preview Panel       │
│  (VSCode Sidebar)           │
│                             │
│  📊 [Your Visualization]    │
└─────────────────────────────┘
```

## Usage

### Basic Usage

```python
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend
import matplotlib.pyplot as plt

# Import bridge
from tt_vscode_bridge import show_plot

# Create plot
plt.plot([1, 2, 3], [4, 5, 6])
plt.title("My Plot")

# Save and show
plt.savefig('my_plot.png')
show_plot('my_plot.png')  # Shows in VSCode Output Preview!
```

### Convenience Function

```python
from tt_vscode_bridge import matplotlib_to_vscode

# Create plot
plt.plot([1, 2, 3], [4, 5, 6])

# Save AND show in one step!
matplotlib_to_vscode('my_plot.png', dpi=150)
```

### Animation Support

```python
from tt_vscode_bridge import show_animation_frame

for frame in range(100):
    # Update plot
    plt.imshow(current_frame)

    # Save to SAME file (VSCode auto-refreshes!)
    plt.savefig('animation.png')

    # Show frame with progress
    show_animation_frame('animation.png', frame, 100)

    time.sleep(0.05)  # 20 fps
```

## API Reference

### `show_plot(image_path, fallback=True, verbose=False)`

Display an image in VSCode's Output Preview panel.

**Parameters:**
- `image_path` (str): Absolute or relative path to image file
- `fallback` (bool): Print file path when VSCode unavailable (default: True)
- `verbose` (bool): Print status messages (default: False)

**Returns:**
- `bool`: True if successfully sent to VSCode, False if fallback used

**Example:**
```python
success = show_plot('output.png', verbose=True)
if success:
    print("Sent to VSCode!")
else:
    print("Using fallback - open file manually")
```

### `show_animation_frame(frame_path, frame_number=None, total_frames=None)`

Display an animation frame in VSCode's Output Preview panel.

**Parameters:**
- `frame_path` (str): Path to frame image (reuse same path for animation)
- `frame_number` (int, optional): Current frame number for progress display
- `total_frames` (int, optional): Total frames for progress display

**Returns:**
- `bool`: True if successfully sent to VSCode

**Example:**
```python
for step in range(100):
    # ... update visualization ...
    plt.savefig('frame.png')
    show_animation_frame('frame.png', step, 100)
```

### `matplotlib_to_vscode(save_path='temp_plot.png', dpi=150, show_in_vscode=True, **kwargs)`

Save current matplotlib figure and optionally show in VSCode.

**Parameters:**
- `save_path` (str): Where to save the figure
- `dpi` (int): Resolution for PNG output (default: 150)
- `show_in_vscode` (bool): If True, send to VSCode preview (default: True)
- `**kwargs`: Additional arguments for plt.savefig()

**Returns:**
- `str`: Absolute path to saved image

**Example:**
```python
# One-liner: save and show!
path = matplotlib_to_vscode('my_plot.png', dpi=200)
```

### `is_vscode_available()`

Check if VSCode is available and accessible.

**Returns:**
- `bool`: True if 'code' command available, False otherwise

**Example:**
```python
if is_vscode_available():
    print("VSCode detected!")
else:
    print("VSCode not available, using fallback")
```

## Fallback Behavior

When VSCode is unavailable or the command fails, the bridge:
1. Prints the absolute path to the saved file
2. Provides instructions to open it manually
3. Returns `False` (vs `True` on success)

**Example fallback output:**
```
📊 View your visualization: /home/user/tt-scratchpad/my_plot.png
   In VSCode: Cmd/Ctrl+Click the path above
```

## Use Cases

### 1. Game of Life (Real-time Simulation)

```python
from tt_vscode_bridge import show_animation_frame

for step in range(100):
    # Run simulation step
    grid = evolve(grid)

    # Visualize
    plt.imshow(grid, cmap='binary')
    plt.title(f'Generation {step}')
    plt.savefig('game_of_life.png')

    # Show in VSCode (auto-refreshes!)
    show_animation_frame('game_of_life.png', step, 100)
```

### 2. Mandelbrot Explorer (High-res Static Images)

```python
from tt_vscode_bridge import matplotlib_to_vscode

# Generate fractal
mandelbrot = compute_mandelbrot(resolution=2000)

# Save as SVG for infinite zoom!
plt.imshow(mandelbrot, cmap='hot')
matplotlib_to_vscode('mandelbrot.svg', format='svg')
```

### 3. Audio Processor (Spectrograms)

```python
from tt_vscode_bridge import show_plot

# Compute spectrogram
spectrogram = compute_mel_spectrogram(audio)

# Show in VSCode
plt.imshow(spectrogram, aspect='auto', cmap='viridis')
plt.savefig('spectrogram.png', dpi=150)
show_plot('spectrogram.png', verbose=True)
```

### 4. Particle Life (Multi-species Dynamics)

```python
from tt_vscode_bridge import show_animation_frame

frame_path = 'particle_life.png'

for frame in range(1000):
    # Update physics
    update_particles(particles)

    # Visualize
    plt.scatter(particles[:, 0], particles[:, 1],
                c=species, s=10, alpha=0.6)
    plt.savefig(frame_path)

    # Real-time preview
    show_animation_frame(frame_path, frame, 1000)
```

## Integration with Cookbook Lessons

### Step 1: Copy Bridge Module

The `tt_vscode_bridge.py` module is located at:
```
content/templates/cookbook/tt_vscode_bridge.py
```

When cookbook projects are created, this module is automatically copied to:
```
~/tt-scratchpad/cookbook/tt_vscode_bridge.py
```

### Step 2: Import in Lesson Scripts

All cookbook scripts should import the bridge:

```python
import sys
import os

# Add cookbook directory to path
sys.path.insert(0, os.path.expanduser('~/tt-scratchpad/cookbook'))

from tt_vscode_bridge import show_plot
```

### Step 3: Update Visualization Calls

**Before:**
```python
plt.savefig('output.png')
print(f"Saved to: {os.path.abspath('output.png')}")
# User manually opens file
```

**After:**
```python
from tt_vscode_bridge import matplotlib_to_vscode

matplotlib_to_vscode('output.png')
# Automatically shows in VSCode Output Preview!
```

## Implementation Details

### VSCode Command

**Command ID:** `tenstorrent.showVisualization`

**Registration** (`package.json`):
```json
{
  "command": "tenstorrent.showVisualization",
  "title": "Show Visualization in Output Preview",
  "category": "Tenstorrent"
}
```

**Handler** (`extension.ts`):
```typescript
async function showVisualization(imagePath?: string): Promise<void> {
  // Expand ~ and convert to absolute path
  // Validate file exists and is image format
  // Show in TenstorrentImagePreviewProvider
}
```

### Python CLI Invocation

The Python bridge calls VSCode via:
```bash
code --command tenstorrent.showVisualization /absolute/path/to/image.png
```

**Why this works:**
- VSCode CLI accepts `--command` flag for extension commands
- Commands can take arguments via additional CLI arguments
- Non-blocking execution (returns immediately)
- Works from any terminal/script

### Error Handling

**File not found:**
```python
# Python side checks before calling VSCode
if not os.path.exists(abs_path):
    print("⚠️  Image not found: {abs_path}")
    return False
```

**VSCode command timeout:**
```python
# 5-second timeout prevents hanging
result = subprocess.run([...], capture_output=True, timeout=5)
```

**Extension not loaded:**
```python
# If command fails, fallback gracefully
if result.returncode != 0:
    if fallback:
        print(f"📊 View your visualization: {abs_path}")
    return False
```

## Testing

### Manual Test

```bash
cd ~/tt-scratchpad
python3 test_vscode_viz_integration.py
```

**Test suite includes:**
1. Static plot display
2. SVG format (scalable graphics)
3. Animation (30 frames)
4. Convenience function
5. Fallback behavior

### Unit Test (Python Module)

```bash
cd ~/tt-vscode-toolkit/content/templates/cookbook
python3 tt_vscode_bridge.py
```

**Expected output:**
```
Testing TT-VSCode Bridge...
Image saved to: /tmp/tt_vscode_bridge_test.png
✅ Sent to VSCode Output Preview!
```

### Integration Test (VSCode Command)

```bash
# From VSCode integrated terminal
code --command tenstorrent.showVisualization ~/tt-scratchpad/test.png
```

**Expected:** Image appears in Output Preview panel

## Troubleshooting

### "VSCode not detected"

**Problem:** `is_vscode_available()` returns False

**Solutions:**
1. Ensure VSCode CLI is in PATH: `code --version`
2. Install VSCode CLI: Cmd+Shift+P → "Shell Command: Install 'code' command in PATH"
3. Restart terminal to pick up PATH changes

### "VSCode command failed"

**Problem:** Command returns non-zero exit code

**Solutions:**
1. Ensure Tenstorrent extension is installed and activated
2. Check extension logs: Output panel → "Tenstorrent"
3. Reload VSCode window: Cmd+Shift+P → "Developer: Reload Window"

### "Image preview panel not available"

**Problem:** Extension loaded but webview not registered

**Solutions:**
1. Check sidebar: View → Open View → "Output Preview"
2. Restart extension: Cmd+Shift+P → "Developer: Restart Extension Host"
3. Reinstall extension: `code --install-extension tt-vscode-toolkit.vsix`

### Animation not updating

**Problem:** Frame updates don't show in preview

**Causes:**
1. File not actually changing (check timestamps)
2. Saving to different files instead of same file
3. VSCode file watcher delay

**Solutions:**
```python
# Ensure writing to SAME file each frame
frame_path = 'animation.png'  # CONSTANT path
for frame in range(100):
    plt.savefig(frame_path)  # Overwrite same file
    show_animation_frame(frame_path, frame, 100)
    time.sleep(0.05)  # Give VSCode time to refresh
```

## Performance Considerations

### PNG vs SVG

**PNG:**
- ✅ Fast to generate
- ✅ Fixed resolution
- ⚠️ Large file size for high-res
- Best for: animations, photos, complex plots

**SVG:**
- ✅ Infinite scaling
- ✅ Small file size
- ⚠️ Slower to generate
- Best for: diagrams, fractals, technical drawings

**Recommendation:** Use PNG (150 DPI) for most cases, SVG for zoom-heavy visualizations

### Animation Frame Rate

**Guidelines:**
- **10-20 fps** - Smooth for most visualizations
- **30 fps** - Smooth for fast action
- **60 fps** - Overkill, VSCode can't refresh that fast

**Calculation:**
```python
fps = 20
time.sleep(1.0 / fps)  # 0.05 seconds per frame
```

### File Size

**Typical sizes:**
- **1024×768 PNG @ 150 DPI** - ~100 KB
- **2048×1536 PNG @ 150 DPI** - ~300 KB
- **SVG (simple)** - ~10 KB
- **SVG (complex)** - ~500 KB

**Recommendation:** Keep under 1 MB per image for smooth preview

## Future Enhancements

### Potential Improvements

1. **Direct webview messaging** - Skip file I/O entirely
2. **WebSocket connection** - Real-time streaming of plot data
3. **Interactive controls** - Zoom, pan, play/pause animations
4. **Multi-panel support** - Show multiple plots simultaneously
5. **Plot history** - Navigate previous visualizations
6. **Export options** - Save as PDF, SVG, PNG with custom DPI

### Community Contributions

Contributions welcome! Areas for improvement:
- Additional image format support
- Performance optimizations
- Better error messages
- Cross-platform testing (Windows, Linux, macOS)

## References

### VSCode Extension API
- [Webview API](https://code.visualstudio.com/api/extension-guides/webview)
- [Command API](https://code.visualstudio.com/api/references/vscode-api#commands)
- [File System API](https://code.visualstudio.com/api/references/vscode-api#FileSystemProvider)

### Matplotlib
- [Backends Guide](https://matplotlib.org/stable/users/explain/backends.html)
- [Agg Backend](https://matplotlib.org/stable/api/_as_gen/matplotlib.backends.backend_agg.html)
- [Animation Guide](https://matplotlib.org/stable/api/animation_api.html)

### Tenstorrent
- [Extension README](../README.md)
- [Cookbook Lessons](../content/lessons/cookbook-overview.md)
- [TT-Metal Documentation](https://docs.tenstorrent.com/)

---

**Document Version:** 1.0.0
**Date:** 2026-02-11
**tt-metal Version:** v0.65.1
**Extension Version:** 0.0.310+
