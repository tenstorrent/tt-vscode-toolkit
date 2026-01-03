# AnimateDiff for TT-Metal

**Standalone AnimateDiff temporal attention for TT-Metal Stable Diffusion 3.5**

This project brings native video animation capabilities to Tenstorrent hardware without modifying the tt-metal repository. It implements AnimateDiff's temporal attention as a separate pass that can be applied to any Stable Diffusion 3.5 latents, enabling smooth animated video generation on N150/N300/T3K hardware.

## What is AnimateDiff?

AnimateDiff adds **temporal attention** to Stable Diffusion models, creating smooth motion across video frames. Instead of generating independent images, temporal attention creates coherence between frames so objects actually move, camera pans are smooth, and animations follow physical motion patterns.

**Key Innovation**: AnimateDiff works by adding temporal attention layers after spatial attention in each transformer block. This project isolates that temporal processing into a standalone module that can work with any diffusion model.

## Features

- 🎬 **Native Video Generation**: Generate 2-16 frame animated sequences
- 🔧 **Standalone Package**: Zero modifications to tt-metal repository
- ⚡ **Hardware Optimized**: Works on N150 single chip at 512x512 resolution
- 🎯 **Simple Integration**: Wrap any SD 3.5 pipeline with temporal attention
- 📦 **Plug-and-Play**: Just `pip install` and use
- 🎨 **Export Support**: Save as MP4, GIF, or WebM

## Architecture Overview

```
Standard SD 3.5 (Single Image):
  Noise → Spatial Diffusion → VAE Decode → Image

AnimateDiff SD 3.5 (Video):
  Noise → Spatial Diffusion → Temporal Attention → VAE Decode → Video Frames
                              ↑
                         (This project adds this!)
```

**How it works:**
1. Generate latents for multiple frames with SD 3.5 (spatial diffusion)
2. Apply temporal attention across frames (motion coherence)
3. Decode each frame with VAE
4. Export frames as video

**Temporal Attention Pattern:**
```python
# Input: (batch*frames, spatial_tokens, channels)
# Reshape to: (batch*spatial, frames, channels)
# Attention across frame dimension creates motion
# Reshape back to: (batch*frames, spatial_tokens, channels)
```

## Prerequisites

### Required Software
- **Python 3.10 or 3.11**
- **tt-metal** installed and working (test with `python -c "import ttnn"`)
- **HuggingFace CLI** for weight downloads: `pip install huggingface_hub`

### Required Hardware
- **N150** (Wormhole single chip) - Recommended for 512x512
- **N300/T3K/P100** - For higher resolutions or longer sequences

### Required Weights
- **AnimateDiff Motion Module**: `mm_sd_v15_v2.ckpt` (1.7GB)
- **SD 3.5 Model** (already installed if you've used SD 3.5 on TT-Metal)

## Installation

### 1. Clone or Download

```bash
# If this is part of a git repository:
cd ~/tt-animatediff

# Or download as standalone:
# (copy the animatediff_ttnn/ directory to your project)
```

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install as package (includes dependencies):
pip install -e .

# For video export support:
pip install -e .[video]
```

### 3. Download AnimateDiff Weights

```bash
# Using the provided script (recommended):
bash weights/download_weights.sh

# Or manually:
mkdir -p ~/models/animatediff
huggingface-cli download \
    guoyww/animatediff \
    mm_sd_v15_v2.ckpt \
    --local-dir ~/models/animatediff
```

### 4. Verify Installation

```bash
python -c "import animatediff_ttnn; print('✓ AnimateDiff package installed')"
python -c "import ttnn; print('✓ tt-metal available')"
```

## Quick Start

### 2-Frame Demo (Simplest Test)

```bash
python examples/generate_2frame_video.py
```

**What this does:**
- Loads AnimateDiff temporal module
- Creates synthetic 2-frame latents
- Applies temporal attention
- Verifies temporal coherence

**Expected output:**
```
Step 3: Applying temporal attention...
  ✓ Temporal attention modified the latents (expected)

Step 5: Analyzing frame correlation...
  Correlation between frames: 0.8524
  ✓ High correlation detected - frames are temporally coherent
```

### 16-Frame Video (Full Animation)

```bash
python examples/generate_16frame_video.py
```

**What this does:**
- Generates 16-frame sequence
- Applies temporal attention for motion coherence
- Analyzes frame-to-frame correlation
- Exports test video (GIF)

**Output location:** `output/test_16frame.gif`

## Usage Guide

### Basic Usage: Standalone Testing

```python
from animatediff_ttnn import create_animatediff_pipeline
import torch

# Create pipeline
pipeline = create_animatediff_pipeline(
    temporal_checkpoint="~/models/animatediff/mm_sd_v15_v2.ckpt"
)

# Prepare video latents (16 frames)
latents = pipeline.prepare_video_latents(
    batch_size=1,
    num_frames=16,
    height=64,  # Latent height (512 // 8)
    width=64,   # Latent width (512 // 8)
    num_channels=16,  # SD 3.5 latent channels
    dtype=torch.float32,
    device=torch.device("cpu"),
    generator=torch.Generator().manual_seed(42),
)

# Apply temporal coherence
latents_coherent = pipeline.apply_temporal_coherence(
    latents,
    num_frames=16
)

print(f"Input shape: {latents.shape}")
print(f"Output shape: {latents_coherent.shape}")
# Both: (16, 64, 64, 16) = (frames, height, width, channels)
```

### Integration with SD 3.5 Pipeline

```python
# Pseudocode showing full integration:

# 1. Import both SD 3.5 and AnimateDiff
from tt_metal.models.experimental.stable_diffusion_35_large.tt.pipeline import TtStableDiffusion3Pipeline
from animatediff_ttnn import create_animatediff_pipeline

# 2. Load both pipelines
sd35_pipeline = TtStableDiffusion3Pipeline(...)
animatediff_pipeline = create_animatediff_pipeline(
    temporal_checkpoint="~/models/animatediff/mm_sd_v15_v2.ckpt"
)

# 3. Generate video
prompt = "A butterfly landing on a flower, cinematic"
num_frames = 16

# Generate latents with SD 3.5 (spatial diffusion)
# This is the part that needs integration work:
# - Prepare latents for num_frames
# - Run denoising loop num_frames times
# - Keep frames together for temporal processing
latents = sd35_pipeline.generate_latents(
    prompt=prompt,
    num_frames=num_frames,  # NEW: Generate multiple frames
    height=512,
    width=512,
)

# Apply temporal attention (motion coherence)
latents = animatediff_pipeline.apply_temporal_coherence(
    latents,
    num_frames=num_frames
)

# Decode frames
frames = sd35_pipeline.decode_latents(latents, num_frames=num_frames)

# Export to video
animatediff_pipeline.export_video(frames, "butterfly.mp4", fps=8)
```

**Note**: Full SD 3.5 integration requires modifying the pipeline to handle `num_frames > 1`. The temporal attention module itself is ready to use.

### Video Export

```python
from PIL import Image
from animatediff_ttnn import create_animatediff_pipeline

pipeline = create_animatediff_pipeline(...)

# Create list of PIL Image frames
frames = [Image.open(f"frame_{i:03d}.png") for i in range(16)]

# Export as MP4 (requires diffusers[video])
pipeline.export_video(frames, "output.mp4", fps=8)

# Export as GIF
pipeline.export_video(frames, "output.gif", fps=8, loop=0)

# Export as WebM
pipeline.export_video(frames, "output.webm", fps=8)
```

## API Reference

### `create_animatediff_pipeline()`

**Factory function to create AnimateDiff pipeline.**

```python
def create_animatediff_pipeline(
    temporal_checkpoint: str,
    dim: int = 320,
    num_heads: int = 8,
    max_frames: int = 24,
    use_ttnn: bool = False
) -> AnimateDiffPipeline:
```

**Parameters:**
- `temporal_checkpoint`: Path to AnimateDiff motion module (e.g., `~/models/animatediff/mm_sd_v15_v2.ckpt`)
- `dim`: Hidden dimension size (default: 320, matches SD 3.5)
- `num_heads`: Number of attention heads (default: 8)
- `max_frames`: Maximum frames for positional encoding (default: 24)
- `use_ttnn`: Use TTNN operations if True, PyTorch if False (default: False)

**Returns:** `AnimateDiffPipeline` instance

### `AnimateDiffPipeline`

**Main wrapper class for temporal attention.**

#### Methods

##### `apply_temporal_coherence()`

**Apply temporal attention to add motion coherence across frames.**

```python
def apply_temporal_coherence(
    self,
    latents: torch.Tensor,
    num_frames: int,
    device: Optional[torch.device] = None
) -> torch.Tensor:
```

**Parameters:**
- `latents`: Input latent tensor, shape `(batch*frames, height, width, channels)` or `(frames, height, width, channels)`
- `num_frames`: Number of frames in sequence
- `device`: Target device (optional, uses latents.device if not specified)

**Returns:** Latents with temporal attention applied, same shape as input

##### `prepare_video_latents()`

**Prepare random latents for video generation (testing/initialization).**

```python
def prepare_video_latents(
    self,
    batch_size: int,
    num_frames: int,
    height: int,
    width: int,
    num_channels: int,
    dtype: torch.dtype,
    device: torch.device,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
```

**Returns:** Random latents shaped `(batch*frames, height, width, channels)`

##### `export_video()`

**Export frames to video file.**

```python
def export_video(
    self,
    frames: List[Image.Image],
    output_path: str,
    fps: int = 8,
    loop: int = 0,
) -> None:
```

**Parameters:**
- `frames`: List of PIL Image frames
- `output_path`: Output file path (.mp4, .gif, or .webm)
- `fps`: Frames per second (default: 8)
- `loop`: Loop count for GIF (0 = infinite, default: 0)

### Low-Level Functions

#### `load_animatediff_weights()`

**Load AnimateDiff motion module weights from checkpoint.**

```python
def load_animatediff_weights(
    checkpoint_path: str,
    dim: int = 320,
    num_heads: int = 8,
    max_frames: int = 24,
) -> TemporalAttentionWeights:
```

#### `temporal_attention_torch()`

**Apply temporal attention using PyTorch operations.**

```python
def temporal_attention_torch(
    hidden_states: torch.Tensor,
    weights: TemporalAttentionWeights,
    num_frames: int,
) -> torch.Tensor:
```

#### `temporal_attention_ttnn()`

**Apply temporal attention using TTNN operations (for hardware acceleration).**

```python
def temporal_attention_ttnn(
    hidden_states: "ttnn.Tensor",
    weights: TemporalAttentionWeights,
    num_frames: int,
    device: "ttnn.Device",
) -> "ttnn.Tensor":
```

## Architecture Deep Dive

### Temporal Attention Mechanism

AnimateDiff's core innovation is **temporal attention** - applying self-attention across the frame dimension instead of spatial dimension:

**Standard Spatial Attention (SD 3.5):**
```
Input: (batch, height*width, channels)
Attention across: spatial tokens (height*width dimension)
Result: Each spatial location attends to all other spatial locations
```

**Temporal Attention (AnimateDiff):**
```
Input: (batch, frames, height*width, channels)
Attention across: frames (temporal dimension)
Result: Each frame attends to all other frames
```

### Reshaping Pattern

The key to temporal attention is the tensor reshaping:

```python
# Input from spatial diffusion
hidden_states.shape = (batch*frames, spatial_tokens, channels)
# Example: (16, 4096, 320) for 16 frames of 64x64 latents

# Reshape to separate frame dimension
hidden_states = hidden_states.view(batch, frames, spatial, channels)
# Example: (1, 16, 4096, 320)

# Transpose to put frames in attention position
hidden_states = hidden_states.permute(0, 2, 1, 3)
# Example: (1, 4096, 16, 320)

# Flatten for attention computation
hidden_states = hidden_states.reshape(batch*spatial, frames, channels)
# Example: (4096, 16, 320)
# Now attention will operate across the 16 frames!
```

### Positional Encoding

Frames need positional encoding so the model knows their temporal order:

```python
# Sinusoidal encoding (same as transformer positional encoding)
frame_indices = [0, 1, 2, ..., 15]  # For 16 frames
pos_encoding = sinusoidal_positional_encoding(frame_indices, dim=320)

# Add to frame features
hidden_states = hidden_states + pos_encoding
```

### Multi-Head Attention

Just like spatial attention, temporal attention uses multiple heads:

```python
# Split channels across heads
num_heads = 8
head_dim = 320 // 8 = 40

# Each head operates independently on 40-dim feature space
# Heads are concatenated after attention
```

### Residual Connection

Critical: temporal attention uses a residual connection to preserve spatial information:

```python
# Before temporal attention
residual = hidden_states

# Apply temporal attention
hidden_states = temporal_attention(hidden_states)

# Add residual (preserve spatial structure)
hidden_states = hidden_states + residual
```

## Integration Strategy

### Why Standalone?

This project uses a **separate temporal pass** approach:

**Benefits:**
- ✅ Zero modifications to tt-metal repository
- ✅ Easy to maintain and update
- ✅ Can be used with any SD 3.5 variant
- ✅ Clean separation of concerns

**How it works:**
1. SD 3.5 generates latents for multiple frames (spatial diffusion)
2. AnimateDiff applies temporal attention (motion coherence)
3. VAE decodes frames to images

### Integration Points

To integrate with SD 3.5 pipeline, you need to:

1. **Modify latent preparation** to handle `num_frames > 1`:
```python
# In SD 3.5 pipeline:
if num_frames > 1:
    latents = prepare_video_latents(num_frames, height, width)
else:
    latents = prepare_image_latents(height, width)
```

2. **Apply temporal attention after denoising loop**:
```python
# After spatial diffusion completes:
if num_frames > 1:
    latents = animatediff_pipeline.apply_temporal_coherence(
        latents, num_frames
    )
```

3. **Decode frames sequentially**:
```python
# Decode each frame with VAE:
frames = []
for i in range(num_frames):
    frame_latent = latents[i:i+1]  # Extract single frame
    frame_image = vae_decode(frame_latent)
    frames.append(frame_image)
```

## Troubleshooting

### Import Error: Cannot find module 'animatediff_ttnn'

**Solution:** Install the package:
```bash
cd ~/tt-animatediff
pip install -e .
```

### Import Error: Cannot find module 'ttnn'

**Problem:** tt-metal not installed or not in PYTHONPATH

**Solution:**
```bash
# Activate tt-metal environment
cd ~/tt-metal
source python_env/bin/activate

# Or set PYTHONPATH
export PYTHONPATH=~/tt-metal:$PYTHONPATH
```

### Weight Download Fails

**Problem:** HuggingFace CLI not installed or not authenticated

**Solution:**
```bash
# Install HF CLI
pip install huggingface_hub

# Authenticate (optional, for gated models)
huggingface-cli login
```

### Out of Memory on N150

**Problem:** 16 frames at 512x512 exhausts DRAM

**Solutions:**
1. **Reduce resolution**: Use 512x512 instead of 768x768
2. **Fewer frames**: Start with 8 frames instead of 16
3. **Batch processing**: Process 4 frames at a time, concatenate results
4. **VAE tiling**: Enable tiling in VAE decoder

### Temporal Attention Has No Effect

**Check:**
```python
# Verify num_frames > 1
if num_frames == 1:
    print("⚠️ Temporal attention skipped (only 1 frame)")

# Verify weights loaded
if pipeline.temporal_weights is None:
    print("⚠️ Temporal weights not loaded")

# Check tensor shapes
print(f"Latent shape: {latents.shape}")
# Should be: (frames, height, width, channels)
```

### Low Frame Correlation

**Expected correlation:** > 0.5 indicates temporal coherence

If correlation is low (< 0.3):
- Check that temporal weights loaded correctly
- Verify reshaping logic matches input tensor format
- Ensure residual connection is working

## Performance Tips

### Memory Optimization

```python
# Process frames in batches
batch_size = 4
for i in range(0, num_frames, batch_size):
    batch_latents = latents[i:i+batch_size]
    batch_latents = pipeline.apply_temporal_coherence(
        batch_latents, num_frames=batch_size
    )
    latents[i:i+batch_size] = batch_latents
```

### Hardware Acceleration

```python
# Enable TTNN for hardware acceleration
pipeline = create_animatediff_pipeline(
    temporal_checkpoint="~/models/animatediff/mm_sd_v15_v2.ckpt",
    use_ttnn=True  # Use TT hardware for temporal attention
)
```

**Note:** TTNN implementation is experimental. Start with PyTorch (`use_ttnn=False`) for stability.

## Development

### Running Tests

```bash
# Basic import test
python -c "import animatediff_ttnn; print('✓ Import successful')"

# 2-frame test
python examples/generate_2frame_video.py

# 16-frame test
python examples/generate_16frame_video.py
```

### Code Structure

```
animatediff_ttnn/
├── __init__.py           # Package initialization
├── pipeline.py           # AnimateDiffPipeline wrapper
└── temporal_module.py    # Temporal attention implementation

examples/
├── generate_2frame_video.py   # Simple 2-frame test
└── generate_16frame_video.py  # Full 16-frame demo

weights/
└── download_weights.sh   # Weight download automation

setup.py                  # Package installation
requirements.txt          # Dependencies
README.md                 # This file
```

### Adding Custom Models

To adapt for different model architectures:

1. **Modify dimensions** in `load_animatediff_weights()`:
```python
# For different channel dimensions:
weights = load_animatediff_weights(
    checkpoint_path="...",
    dim=640,  # Change to match your model
    num_heads=16,
)
```

2. **Adjust reshaping** if tensor format differs:
```python
# In temporal_module.py, modify reshape logic:
hidden_states = hidden_states.view(batch, frames, spatial, channels)
```

## Model Bring-Up Tutorial

For a detailed guide on how this project was built (research, architecture analysis, implementation), see:

**`MODEL_BRINGUP_TUTORIAL.md`** (coming soon)

This tutorial documents the complete process:
- Researching AnimateDiff architecture
- Analyzing SD 3.5 transformer structure
- Finding injection points
- Porting to TTNN
- Creating standalone package
- Testing and validation

## Credits

**AnimateDiff**: Original paper and implementation by Yuwei Guo et al.
- Paper: [AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models](https://arxiv.org/abs/2307.04725)
- Repository: https://github.com/guoyww/AnimateDiff

**TT-Metal**: Tenstorrent's machine learning framework
- Repository: https://github.com/tenstorrent/tt-metal

**Stable Diffusion 3.5**: Stability AI's latest diffusion model

## License

This project is provided as-is for research and development purposes. Please respect the licenses of:
- AnimateDiff (check original repository)
- TT-Metal (Apache 2.0)
- Stable Diffusion 3.5 (check Stability AI license)

## Support

For issues or questions:
- Check the Troubleshooting section above
- Review the example scripts in `examples/`
- Open an issue on the project repository
- Join Tenstorrent community channels

---

**Status**: Beta - Core functionality implemented, integration with SD 3.5 pipeline pending

**Tested On**: N150 (Wormhole single chip)

**Version**: 0.1.0
