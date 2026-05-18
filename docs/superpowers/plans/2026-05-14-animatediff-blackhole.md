# AnimateDiff on Blackhole — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite `content/projects/animatediff/` to produce a working two-phase AnimateDiff implementation — Phase 1 as a correct diffusers AnimateDiffPipeline baseline (CPU, no hardware needed), Phase 2 as TTNN UNet frame generation on Blackhole hardware.

**Architecture:** Phase 1 uses `diffusers.AnimateDiffPipeline` with `MotionAdapter("guoyww/animatediff-motion-adapter-v1-5-2")` on `CompVis/stable-diffusion-v1-4` — the correct architecture (MotionAdapter injects temporal attention at each UNet transformer block at 320-dim features, exactly where `mm_sd_v15_v2.ckpt` weights were trained). Phase 2 uses the tt-metal SD 1.4 TTNN UNet (`UNet2D` from `models/demos/wormhole/stable_diffusion/tt/`) running on Blackhole via `TT_METAL_ARCH_NAME=blackhole`, with shared base noise for inter-frame coherence.

**Tech Stack:** Python 3.10+, diffusers>=0.32.1, torch, Pillow, ttnn (from ~/tt-metal), TtPNDMScheduler, CLIPTokenizer/CLIPTextModel (openai/clip-vit-large-patch14), AutoencoderKL+Vae, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `examples/generate_with_sd35.py` | **DELETE** | Wrong architecture — SD 3.5 DiT incompatible with SD 1.5 motion weights |
| `examples/generate_16frame_video.py` | **DELETE** | Demo/placeholder mode |
| `examples/generate_2frame_video.py` | **DELETE** | Demo/placeholder mode |
| `requirements.txt` | **UPDATE** | diffusers>=0.32.1 |
| `tests/__init__.py` | **CREATE** | Test package |
| `animatediff_ttnn/pipeline.py` | **REWRITE** | Phase 1: thin wrapper around diffusers AnimateDiffPipeline |
| `animatediff_ttnn/__init__.py` | **UPDATE** | Export only Phase 1 + Phase 2 public API |
| `animatediff_ttnn/temporal_module.py` | **KEEP AS-IS** | Reference only — no longer exported publicly |
| `animatediff_ttnn/ttnn_pipeline.py` | **CREATE** | Phase 2: TTNN UNet on Blackhole, frame generation |
| `tests/test_pipeline.py` | **CREATE** | Tests for Phase 1 pipeline functions |
| `tests/test_ttnn_pipeline.py` | **CREATE** | Tests for Phase 2 setup_blackhole() and build_tlist() |
| `examples/generate_baseline.py` | **CREATE** | Phase 1 end-to-end example (CPU, no hardware) |
| `examples/generate_blackhole.py` | **CREATE** | Phase 2 end-to-end example (Blackhole required) |
| `output/.gitkeep` | **CREATE** | Keep output/ directory in git |
| `README.md` | **REWRITE** | Two-phase explanation, hf CLI, correct model names |
| `docs/IMPLEMENTATION_STATUS.md` | **REWRITE** | Honest Phase 1/2 status |
| `docs/MODEL_BRINGUP_TUTORIAL.md` | **UPDATE** | Replace huggingface-cli with hf CLI |

---

## Task 1: Delete stale files and update requirements

**Files:**
- Delete: `content/projects/animatediff/examples/generate_with_sd35.py`
- Delete: `content/projects/animatediff/examples/generate_16frame_video.py`
- Delete: `content/projects/animatediff/examples/generate_2frame_video.py`
- Modify: `content/projects/animatediff/requirements.txt`
- Create: `content/projects/animatediff/tests/__init__.py`
- Create: `content/projects/animatediff/output/.gitkeep`

- [ ] **Step 1: Delete stale example files**

```bash
cd content/projects/animatediff
git rm examples/generate_with_sd35.py examples/generate_16frame_video.py examples/generate_2frame_video.py
```

Expected: `rm 'content/projects/animatediff/examples/generate_with_sd35.py'` etc.

- [ ] **Step 2: Rewrite requirements.txt**

```
# AnimateDiff requirements

# Core
torch>=2.0.0
numpy>=1.24.0
Pillow>=9.0.0

# Phase 1: diffusers AnimateDiffPipeline
diffusers>=0.32.1
transformers>=4.30.0
accelerate>=0.20.0

# Testing
pytest>=7.0.0
```

Write this to `content/projects/animatediff/requirements.txt`.

- [ ] **Step 3: Create tests/ directory**

```bash
mkdir -p content/projects/animatediff/tests
touch content/projects/animatediff/tests/__init__.py
touch content/projects/animatediff/output/.gitkeep
```

- [ ] **Step 4: Verify no broken imports from deletions**

```bash
cd content/projects/animatediff
python -c "import animatediff_ttnn; print('imports ok')"
```

Expected: `imports ok` (or ImportError if ttnn not available — that's fine, just verify no syntax errors from our changes).

- [ ] **Step 5: Commit**

```bash
git add content/projects/animatediff/requirements.txt \
        content/projects/animatediff/tests/__init__.py \
        content/projects/animatediff/output/.gitkeep
git commit -m "refactor(animatediff): delete stale SD3.5 examples, update requirements to diffusers>=0.32.1"
```

---

## Task 2: Rewrite animatediff_ttnn/pipeline.py (Phase 1 — diffusers wrapper)

**Files:**
- Rewrite: `content/projects/animatediff/animatediff_ttnn/pipeline.py`
- Create: `content/projects/animatediff/tests/test_pipeline.py`
- Modify: `content/projects/animatediff/animatediff_ttnn/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `content/projects/animatediff/tests/test_pipeline.py`:

```python
from unittest.mock import MagicMock
from PIL import Image
import pytest


def _make_mock_pipe(num_frames=16):
    """Return a mock AnimateDiffPipeline that returns num_frames PIL images."""
    pipe = MagicMock()
    frames = [Image.new("RGB", (512, 512), color=(i * 15, 0, 0)) for i in range(num_frames)]
    pipe.return_value.frames = [frames]
    return pipe


def test_generate_returns_list_of_pil_images():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(16)
    result = generate(mock_pipe, "a campfire", num_frames=16)
    assert isinstance(result, list)
    assert len(result) == 16
    assert all(isinstance(f, Image.Image) for f in result)


def test_generate_passes_num_frames_to_pipe():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(8)
    generate(mock_pipe, "test prompt", num_frames=8, guidance_scale=9.0,
             num_inference_steps=20, seed=77)
    kwargs = mock_pipe.call_args.kwargs
    assert kwargs["num_frames"] == 8
    assert kwargs["guidance_scale"] == 9.0
    assert kwargs["num_inference_steps"] == 20
    assert kwargs["prompt"] == "test prompt"


def test_generate_single_frame():
    from animatediff_ttnn.pipeline import generate
    mock_pipe = _make_mock_pipe(1)
    result = generate(mock_pipe, "a still lake", num_frames=1)
    assert len(result) == 1


def test_export_gif_creates_file(tmp_path):
    from animatediff_ttnn.pipeline import export_gif
    frames = [Image.new("RGB", (64, 64), color=(i * 40, 0, 0)) for i in range(4)]
    output = str(tmp_path / "test.gif")
    export_gif(frames, output, fps=4)
    assert (tmp_path / "test.gif").exists()
    assert (tmp_path / "test.gif").stat().st_size > 0


def test_export_gif_creates_parent_directories(tmp_path):
    from animatediff_ttnn.pipeline import export_gif
    frames = [Image.new("RGB", (32, 32)) for _ in range(2)]
    nested = str(tmp_path / "a" / "b" / "out.gif")
    export_gif(frames, nested)
    import pathlib
    assert pathlib.Path(nested).exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd content/projects/animatediff
python -m pytest tests/test_pipeline.py -v 2>&1 | head -30
```

Expected: `ImportError` or `ModuleNotFoundError` — `generate` and `export_gif` don't exist yet.

- [ ] **Step 3: Rewrite animatediff_ttnn/pipeline.py**

```python
"""Phase 1: AnimateDiff baseline using diffusers AnimateDiffPipeline + MotionAdapter.

Uses CompVis/stable-diffusion-v1-4 with guoyww/animatediff-motion-adapter-v1-5-2.
The MotionAdapter injects temporal attention inside each SD 1.4 UNet transformer
block at 320-dim features — the correct location for mm_sd_v15_v2.ckpt weights.
No TT hardware required.
"""

from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter


def create_animatediff_pipeline(
    model_id: str = "CompVis/stable-diffusion-v1-4",
    adapter_id: str = "guoyww/animatediff-motion-adapter-v1-5-2",
    torch_dtype: torch.dtype = torch.float16,
) -> AnimateDiffPipeline:
    """Create a diffusers AnimateDiffPipeline with MotionAdapter.

    Downloads from HuggingFace cache on first call. No TT hardware required.
    Run `hf download CompVis/stable-diffusion-v1-4` and
    `hf download guoyww/animatediff-motion-adapter-v1-5-2` beforehand.

    Returns:
        diffusers.AnimateDiffPipeline ready for inference
    """
    adapter = MotionAdapter.from_pretrained(adapter_id, torch_dtype=torch_dtype)
    scheduler = DDIMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipe = AnimateDiffPipeline.from_pretrained(
        model_id,
        motion_adapter=adapter,
        scheduler=scheduler,
        torch_dtype=torch_dtype,
    )
    return pipe


def generate(
    pipe: AnimateDiffPipeline,
    prompt: str,
    negative_prompt: str = "",
    num_frames: int = 16,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 25,
    seed: int = 42,
) -> List[Image.Image]:
    """Generate an animated sequence from a text prompt.

    Args:
        pipe: AnimateDiffPipeline from create_animatediff_pipeline()
        prompt: Text prompt describing the animation
        negative_prompt: What to avoid in the output
        num_frames: Number of frames (8 or 16 recommended)
        guidance_scale: CFG scale — higher = more prompt-adherent (7.5 is standard)
        num_inference_steps: Denoising steps (25 balances speed and quality)
        seed: Random seed for reproducibility

    Returns:
        List of PIL Images, one per frame, 512x512
    """
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=num_frames,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        generator=torch.Generator().manual_seed(seed),
    )
    return output.frames[0]


def export_gif(frames: List[Image.Image], output_path: str, fps: int = 8) -> None:
    """Save a list of PIL Images as an animated GIF.

    Args:
        frames: List of PIL Images (all same size)
        output_path: Destination file path, e.g. 'output/result.gif'
        fps: Frames per second (duration = 1000 // fps ms per frame)
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // fps,
        loop=0,
    )
```

- [ ] **Step 4: Update animatediff_ttnn/__init__.py**

```python
"""AnimateDiff for Tenstorrent hardware.

Phase 1 (CPU baseline):
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif

Phase 2 (Blackhole hardware):
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, build_tlist, generate_frames
"""

__version__ = "0.2.0"

from .pipeline import create_animatediff_pipeline, generate, export_gif

__all__ = [
    "create_animatediff_pipeline",
    "generate",
    "export_gif",
]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd content/projects/animatediff
python -m pytest tests/test_pipeline.py -v
```

Expected output:
```
tests/test_pipeline.py::test_generate_returns_list_of_pil_images PASSED
tests/test_pipeline.py::test_generate_passes_num_frames_to_pipe PASSED
tests/test_pipeline.py::test_generate_single_frame PASSED
tests/test_pipeline.py::test_export_gif_creates_file PASSED
tests/test_pipeline.py::test_export_gif_creates_parent_directories PASSED
5 passed
```

- [ ] **Step 6: Commit**

```bash
git add content/projects/animatediff/animatediff_ttnn/pipeline.py \
        content/projects/animatediff/animatediff_ttnn/__init__.py \
        content/projects/animatediff/tests/test_pipeline.py
git commit -m "feat(animatediff): rewrite pipeline.py as Phase 1 diffusers AnimateDiffPipeline wrapper"
```

---

## Task 3: Create examples/generate_baseline.py (Phase 1 end-to-end)

**Files:**
- Create: `content/projects/animatediff/examples/generate_baseline.py`

- [ ] **Step 1: Create examples/generate_baseline.py**

```python
#!/usr/bin/env python3
"""Phase 1 AnimateDiff baseline — correct temporal attention on CPU.

Uses diffusers AnimateDiffPipeline with MotionAdapter. The MotionAdapter
injects temporal attention inside each SD 1.4 UNet transformer block at
320-dim features, which is exactly where mm_sd_v15_v2.ckpt weights operate.
No TT hardware required.

Setup:
    pip install -r requirements.txt
    hf download CompVis/stable-diffusion-v1-4
    hf download guoyww/animatediff-motion-adapter-v1-5-2

Usage:
    python examples/generate_baseline.py
    python examples/generate_baseline.py --prompt "ocean waves" --frames 8 --steps 20
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Phase 1 — CPU baseline")
    parser.add_argument("--prompt", default="a campfire with crackling flames, cinematic, 4K")
    parser.add_argument("--negative-prompt", default="blurry, low quality, distorted", dest="negative_prompt")
    parser.add_argument("--frames", type=int, default=16, help="Number of frames (8 or 16 recommended)")
    parser.add_argument("--steps", type=int, default=25, help="Denoising steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/baseline.gif")
    args = parser.parse_args()

    print("AnimateDiff Phase 1 — CPU baseline")
    print(f"  Base model : CompVis/stable-diffusion-v1-4")
    print(f"  Adapter    : guoyww/animatediff-motion-adapter-v1-5-2")
    print(f"  Prompt     : {args.prompt}")
    print(f"  Frames     : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print()

    print("Loading AnimateDiff pipeline (first run downloads ~4.7 GB)...")
    t0 = time.time()
    pipe = create_animatediff_pipeline()
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print()

    print(f"Generating {args.frames} frames...")
    t1 = time.time()
    frames = generate(
        pipe,
        args.prompt,
        negative_prompt=args.negative_prompt,
        num_frames=args.frames,
        num_inference_steps=args.steps,
        seed=args.seed,
    )
    elapsed = time.time() - t1
    print(f"  Done in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)")
    print()

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames → {args.output}")
    print()
    print("What AnimateDiff did:")
    print("  MotionAdapter injected temporal attention into each UNet transformer block.")
    print("  During denoising, each step attends across all frames simultaneously —")
    print("  the motion weights (mm_sd_v15_v2.ckpt) ensure coherent motion.")
    print()
    print("Phase 2 (Blackhole hardware): see examples/generate_blackhole.py")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify script is importable (syntax check)**

```bash
cd content/projects/animatediff
python -c "import examples.generate_baseline" 2>&1 || python -m py_compile examples/generate_baseline.py && echo "syntax ok"
```

Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add content/projects/animatediff/examples/generate_baseline.py
git commit -m "feat(animatediff): add Phase 1 baseline example (diffusers AnimateDiffPipeline, CPU)"
```

---

## Task 4: Create animatediff_ttnn/ttnn_pipeline.py (Phase 2 — TTNN Blackhole)

**Background context for implementer:**

The SD 1.4 TTNN demo lives at `~/tt-metal/models/demos/wormhole/stable_diffusion/`. Key imports:
- `UNet2D` from `tt.ttnn_functional_unet_2d_condition_model_new_conv` — TTNN UNet
- `TtPNDMScheduler` from `sd_pndm_scheduler` — TTNN-wrapped PNDM scheduler
- `Vae` from `tt.vae.ttnn_vae` — TTNN VAE
- `SD_L1_SMALL_SIZE` from `common` — 21056 on Blackhole, 20928 on Wormhole (auto-detected)
- `sd_helper_funcs.run()` — takes (model, config, tt_vae, input_latents, input_encoder_hidden_states, _tlist, time_step, guidance_scale, ttnn_scheduler) and returns TTNN tensor (1, 3, 512, 512) with values in [-1, 1]

`constant_prop_time_embeddings` is NOT in `sd_helper_funcs` — it's defined locally in demo.py. Define it inline in `ttnn_pipeline.py` (it's 4 lines).

`run()` output post-processing: `(output / 2 + 0.5).clamp(0, 1)` → `(* 255).round().astype("uint8")` → `Image.fromarray()`.

`_tlist[i]` needs `unet.time_proj` from the PyTorch UNet (not TTNN model) to compute time embeddings per timestep.

**Files:**
- Create: `content/projects/animatediff/animatediff_ttnn/ttnn_pipeline.py`
- Create: `content/projects/animatediff/tests/test_ttnn_pipeline.py`
- Modify: `content/projects/animatediff/animatediff_ttnn/__init__.py`

- [ ] **Step 1: Write the failing tests**

Create `content/projects/animatediff/tests/test_ttnn_pipeline.py`:

```python
"""Unit tests for ttnn_pipeline.py — run without Blackhole hardware.

ttnn_pipeline.py uses lazy imports (ttnn imported inside functions), so the
module can be imported on any machine. Tests mock sys.modules at call time.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import torch
import pytest


def test_setup_blackhole_sets_env_var(tmp_path, monkeypatch):
    """setup_blackhole() sets TT_METAL_ARCH_NAME=blackhole if not already set."""
    monkeypatch.delenv("TT_METAL_ARCH_NAME", raising=False)
    (tmp_path).mkdir(exist_ok=True)  # stand in for ~/tt-metal

    from animatediff_ttnn import ttnn_pipeline

    with patch.dict(sys.modules, {
        "ttnn": MagicMock(),
        "models.demos.wormhole.stable_diffusion.common": MagicMock(SD_L1_SMALL_SIZE=21056),
    }):
        with patch.object(ttnn_pipeline, "TT_METAL_PATH", tmp_path):
            ttnn_pipeline.setup_blackhole()

    assert os.environ.get("TT_METAL_ARCH_NAME") == "blackhole"


def test_setup_blackhole_preserves_existing_arch_name(monkeypatch):
    """setup_blackhole() does not overwrite TT_METAL_ARCH_NAME if already set."""
    monkeypatch.setenv("TT_METAL_ARCH_NAME", "wormhole_b0")

    from animatediff_ttnn import ttnn_pipeline

    with patch.object(ttnn_pipeline, "_ensure_tt_metal_path"):
        with patch.dict(sys.modules, {
            "ttnn": MagicMock(),
            "models.demos.wormhole.stable_diffusion.common": MagicMock(SD_L1_SMALL_SIZE=21056),
        }):
            ttnn_pipeline.setup_blackhole()

    assert os.environ.get("TT_METAL_ARCH_NAME") == "wormhole_b0"


def test_build_tlist_returns_one_entry_per_timestep():
    """build_tlist() returns a list with one element per scheduler timestep."""
    from animatediff_ttnn import ttnn_pipeline

    mock_scheduler = MagicMock()
    mock_scheduler.timesteps = torch.tensor([900, 700, 500, 300, 100])
    mock_device = MagicMock()
    mock_time_proj = MagicMock()

    # Patch _constant_prop_time_embeddings at module level so build_tlist() sees it
    with patch.object(ttnn_pipeline, "_constant_prop_time_embeddings",
                      return_value=torch.randn(2, 320)):
        with patch.dict(sys.modules, {"ttnn": MagicMock()}):
            result = ttnn_pipeline.build_tlist(mock_scheduler, mock_time_proj, mock_device)

    assert len(result) == 5


def test_ensure_tt_metal_path_raises_when_absent(tmp_path):
    """_ensure_tt_metal_path() raises RuntimeError when ~/tt-metal is missing."""
    from animatediff_ttnn import ttnn_pipeline

    original = ttnn_pipeline.TT_METAL_PATH
    ttnn_pipeline.TT_METAL_PATH = tmp_path / "nonexistent"
    try:
        with pytest.raises(RuntimeError, match="tt-metal not found"):
            ttnn_pipeline._ensure_tt_metal_path()
    finally:
        ttnn_pipeline.TT_METAL_PATH = original
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd content/projects/animatediff
python -m pytest tests/test_ttnn_pipeline.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError` or `ImportError` for `animatediff_ttnn.ttnn_pipeline`.

- [ ] **Step 3: Create animatediff_ttnn/ttnn_pipeline.py**

```python
"""Phase 2: Blackhole-accelerated video frame generation using TTNN UNet.

Uses the SD 1.4 TTNN UNet from ~/tt-metal (same code runs on Blackhole via
TT_METAL_ARCH_NAME=blackhole). Frames denoised sequentially; temporal
coherence from shared base noise initialization (0.05 per-frame perturbation).

Documented tradeoff: this is TT-hardware-accelerated spatial denoising, not
full AnimateDiff temporal attention. Full integration would require injecting
TemporalTransformer blocks into the TTNN UNet transformer blocks.

Requirements:
    ~/tt-metal present and activated: source ~/tt-metal/python_env/bin/activate
    Blackhole device (P100 or P300c)
"""

import os
import sys
from pathlib import Path
from typing import List

import torch
from PIL import Image


TT_METAL_PATH = Path.home() / "tt-metal"


def _ensure_tt_metal_path() -> None:
    """Add ~/tt-metal to sys.path so SD demo module imports work."""
    tt_metal_str = str(TT_METAL_PATH)
    if tt_metal_str not in sys.path:
        if not TT_METAL_PATH.exists():
            raise RuntimeError(
                f"~/tt-metal not found. Activate the tt-metal environment first:\n"
                f"  cd ~/tt-metal && source python_env/bin/activate"
            )
        sys.path.insert(0, tt_metal_str)


def _constant_prop_time_embeddings(timesteps, sample, time_proj):
    """Compute time embeddings for a scalar timestep.

    Equivalent to the function defined in SD demo.py. Accepts a scalar
    timestep tensor, expands to batch size, runs through UNet time_proj.
    """
    timesteps = timesteps[None]
    timesteps = timesteps.expand(sample.shape[0])
    return time_proj(timesteps)


def setup_blackhole():
    """Open a Blackhole TTNN device with SD-appropriate L1 config.

    Sets TT_METAL_ARCH_NAME=blackhole if not already set (os.environ.setdefault,
    so an existing value is never overwritten). Returns the open TTNN device.
    """
    os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
    _ensure_tt_metal_path()

    import ttnn
    from models.demos.wormhole.stable_diffusion.common import SD_L1_SMALL_SIZE

    return ttnn.open_device(device_id=0, l1_small_size=SD_L1_SMALL_SIZE)


def build_tlist(ttnn_scheduler, torch_time_proj, device, latent_h: int = 64, latent_w: int = 64) -> list:
    """Build pre-computed time embeddings for each denoising timestep.

    sd_helper_funcs.run() expects _tlist[i] to be constant_prop_time_embeddings(t_i)
    already converted to a TTNN bfloat16 tensor on device, shape permuted for UNet.

    Args:
        ttnn_scheduler: TtPNDMScheduler with set_timesteps() already called
        torch_time_proj: unet.time_proj from the PyTorch UNet2DConditionModel
        device: TTNN device from setup_blackhole()
        latent_h: Latent height (image_height // 8; 64 for 512px images)
        latent_w: Latent width (image_width // 8; 64 for 512px images)

    Returns:
        List of TTNN tensors, one per timestep
    """
    import ttnn

    dummy = ttnn.from_torch(
        torch.zeros(2, 4, latent_h, latent_w),  # batch=2 matches CFG concat in run()
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    _tlist = []
    for t in ttnn_scheduler.timesteps:
        _t = _constant_prop_time_embeddings(t, dummy, torch_time_proj)
        _t = _t.unsqueeze(0).unsqueeze(0)
        _t = _t.permute(2, 0, 1, 3)  # pre-permute: expected shape by TTNN UNet
        _t = ttnn.from_torch(_t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
        _tlist.append(_t)
    return _tlist


def generate_frames(
    device,
    ttnn_model,
    tt_vae,
    config,
    ttnn_scheduler,
    torch_time_proj,
    text_embeddings: torch.Tensor,
    num_frames: int = 8,
    guidance_scale: float = 7.5,
    seed: int = 42,
    height: int = 512,
    width: int = 512,
) -> List[Image.Image]:
    """Generate video frames using TTNN UNet on Blackhole.

    Args:
        device: TTNN Blackhole device from setup_blackhole()
        ttnn_model: UNet2D TTNN model loaded with preprocess_model_parameters
        tt_vae: Vae TTNN object
        config: unet.config from PyTorch UNet2DConditionModel
        ttnn_scheduler: TtPNDMScheduler with set_timesteps() already called
        torch_time_proj: unet.time_proj from PyTorch UNet (used to build _tlist)
        text_embeddings: Shape (2, 96, 768) torch tensor — [uncond, cond] concatenated,
                         padded from 77 to 96 tokens with torch.nn.functional.pad(..., (0,0,0,19))
        num_frames: Number of frames to generate
        guidance_scale: CFG scale (7.5 standard)
        seed: Random seed for shared base noise
        height: Output image height in pixels (512 recommended for single Blackhole)
        width: Output image width in pixels (512 recommended for single Blackhole)

    Returns:
        List of PIL Images, one per frame

    Note:
        Temporal coherence comes from shared base noise initialization (0.05
        per-frame perturbation). This is not full AnimateDiff temporal attention —
        see Phase 1 (generate_baseline.py) for that.
    """
    import ttnn
    from models.demos.wormhole.stable_diffusion.sd_helper_funcs import run

    lh, lw = height // 8, width // 8
    _tlist = build_tlist(ttnn_scheduler, torch_time_proj, device, lh, lw)
    time_step = ttnn_scheduler.timesteps.tolist()

    # Convert text embeddings to TTNN device tensor
    ttnn_text_embeddings = ttnn.from_torch(
        text_embeddings,
        dtype=ttnn.bfloat16,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # Shared base noise for inter-frame temporal coherence
    generator = torch.Generator().manual_seed(seed)
    base_noise = torch.randn(1, 4, lh, lw, generator=generator)

    frames = []
    for frame_idx in range(num_frames):
        # Small per-frame perturbation keeps frames related but distinct
        frame_noise = base_noise + 0.05 * torch.randn_like(base_noise)
        latents = ttnn.from_torch(
            frame_noise * ttnn_scheduler.init_noise_sigma,
            dtype=ttnn.bfloat16,
            layout=ttnn.TILE_LAYOUT,
            device=device,
        )

        output = run(
            ttnn_model, config, tt_vae,
            input_latents=latents,
            input_encoder_hidden_states=ttnn_text_embeddings,
            _tlist=_tlist,
            time_step=time_step,
            guidance_scale=guidance_scale,
            ttnn_scheduler=ttnn_scheduler,
        )

        # Post-process: TTNN output is (1, 3, H, W) in [-1, 1]
        img = ttnn.to_torch(output.cpu(blocking=True)).squeeze(0)  # (3, H, W)
        img = (img / 2 + 0.5).clamp(0, 1)
        img = (img.float().permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        frames.append(Image.fromarray(img))

        print(f"  Frame {frame_idx + 1}/{num_frames} done")

    return frames
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd content/projects/animatediff
python -m pytest tests/test_ttnn_pipeline.py -v
```

Expected:
```
tests/test_ttnn_pipeline.py::test_setup_blackhole_sets_env_var PASSED
tests/test_ttnn_pipeline.py::test_setup_blackhole_does_not_overwrite_existing_env PASSED
tests/test_ttnn_pipeline.py::test_build_tlist_returns_one_entry_per_timestep PASSED
tests/test_ttnn_pipeline.py::test_ttnn_pipeline_missing_tt_metal_raises PASSED
4 passed
```

- [ ] **Step 5: Update animatediff_ttnn/__init__.py to add ttnn_pipeline exports**

```python
"""AnimateDiff for Tenstorrent hardware.

Phase 1 (CPU baseline):
    from animatediff_ttnn.pipeline import create_animatediff_pipeline, generate, export_gif

Phase 2 (Blackhole hardware):
    from animatediff_ttnn.ttnn_pipeline import setup_blackhole, build_tlist, generate_frames
"""

__version__ = "0.2.0"

from .pipeline import create_animatediff_pipeline, generate, export_gif

__all__ = [
    "create_animatediff_pipeline",
    "generate",
    "export_gif",
    # Phase 2 — import explicitly: from animatediff_ttnn.ttnn_pipeline import ...
]
```

(Phase 2 symbols are left out of `__all__` intentionally — they require hardware and conditional imports; users import them directly from `ttnn_pipeline`.)

- [ ] **Step 6: Run all tests**

```bash
cd content/projects/animatediff
python -m pytest tests/ -v
```

Expected: all 9 tests pass.

- [ ] **Step 7: Commit**

```bash
git add content/projects/animatediff/animatediff_ttnn/ttnn_pipeline.py \
        content/projects/animatediff/animatediff_ttnn/__init__.py \
        content/projects/animatediff/tests/test_ttnn_pipeline.py
git commit -m "feat(animatediff): add Phase 2 TTNN Blackhole pipeline (ttnn_pipeline.py)"
```

---

## Task 5: Create examples/generate_blackhole.py (Phase 2 end-to-end)

**Files:**
- Create: `content/projects/animatediff/examples/generate_blackhole.py`

- [ ] **Step 1: Create examples/generate_blackhole.py**

```python
#!/usr/bin/env python3
"""Phase 2: Blackhole-accelerated video frame generation using TTNN UNet.

Loads SD 1.4 TTNN UNet and VAE onto a Blackhole device, encodes a text
prompt with CLIP, then generates num_frames sequentially. Temporal coherence
comes from shared base noise initialization (not AnimateDiff motion adapter).

Requirements:
    - ~/tt-metal present: cd ~/tt-metal && source python_env/bin/activate
    - Blackhole hardware (P100 or P300c)
    - SD 1.4 model cached: hf download CompVis/stable-diffusion-v1-4
    - CLIP tokenizer: hf download openai/clip-vit-large-patch14

Usage:
    python examples/generate_blackhole.py
    python examples/generate_blackhole.py --prompt "ocean waves" --frames 8
"""

import argparse
import sys
import time
from pathlib import Path

import torch

TT_METAL_PATH = Path.home() / "tt-metal"
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(TT_METAL_PATH))

from animatediff_ttnn.ttnn_pipeline import setup_blackhole, generate_frames
from animatediff_ttnn.pipeline import export_gif


def load_sd14_ttnn(device):
    """Load SD 1.4 TTNN UNet, VAE, and PNDM scheduler onto device.

    Returns (ttnn_model, tt_vae, config, ttnn_scheduler, torch_unet_time_proj).
    """
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from ttnn.model_preprocessing import preprocess_model_parameters
    from models.demos.wormhole.stable_diffusion.custom_preprocessing import custom_preprocessor
    from models.demos.wormhole.stable_diffusion.sd_pndm_scheduler import TtPNDMScheduler
    from models.demos.wormhole.stable_diffusion.tt.ttnn_functional_unet_2d_condition_model_new_conv import (
        UNet2DConditionModel as UNet2D,
    )
    from models.demos.wormhole.stable_diffusion.tt.vae.ttnn_vae import Vae

    print("  Loading PyTorch VAE...")
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")
    tt_vae = Vae(torch_vae=vae, device=device)

    print("  Loading PyTorch UNet (for config and time_proj)...")
    torch_unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    print("  Building TTNN UNet (compiles kernels, ~2-3 min first run)...")
    parameters = preprocess_model_parameters(
        initialize_model=lambda: torch_unet,
        custom_preprocessor=custom_preprocessor,
        device=device,
    )
    ttnn_model = UNet2D(device, parameters, 2, 64, 64)

    ttnn_scheduler = TtPNDMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        skip_prk_steps=True,
        steps_offset=1,
        device=device,
    )

    return ttnn_model, tt_vae, torch_unet.config, ttnn_scheduler, torch_unet.time_proj


def encode_prompt(prompt: str, negative_prompt: str = "") -> torch.Tensor:
    """Encode text + negative prompt to (2, 96, 768) tensor using CLIP.

    Returns [uncond_embeds, cond_embeds] concatenated, padded 77→96 tokens.
    """
    from transformers import CLIPTokenizer, CLIPTextModel

    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.eval()

    def encode(text):
        tokens = tokenizer(
            text,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(tokens.input_ids)[0]
        # Pad from 77 tokens to 96 — TTNN UNet expects 96-token sequence
        return torch.nn.functional.pad(embeds, (0, 0, 0, 19))

    uncond = encode(negative_prompt)
    cond = encode(prompt)
    return torch.cat([uncond, cond], dim=0)  # (2, 96, 768)


def main():
    parser = argparse.ArgumentParser(description="AnimateDiff Phase 2 — Blackhole TTNN")
    parser.add_argument("--prompt", default="a campfire with crackling flames, cinematic, 4K")
    parser.add_argument("--negative-prompt", default="blurry, low quality", dest="negative_prompt")
    parser.add_argument("--frames", type=int, default=8, help="Number of frames (8 recommended)")
    parser.add_argument("--steps", type=int, default=25, help="Denoising steps (min 4 for PNDM)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="output/blackhole.gif")
    args = parser.parse_args()

    print("AnimateDiff Phase 2 — Blackhole TTNN UNet")
    print(f"  TT_METAL_ARCH_NAME=blackhole")
    print(f"  Prompt    : {args.prompt}")
    print(f"  Frames    : {args.frames}  Steps: {args.steps}  Seed: {args.seed}")
    print()

    print("Opening Blackhole device...")
    device = setup_blackhole()
    print()

    print("Loading SD 1.4 models onto Blackhole...")
    t0 = time.time()
    ttnn_model, tt_vae, config, ttnn_scheduler, torch_time_proj = load_sd14_ttnn(device)
    ttnn_scheduler.set_timesteps(args.steps)
    print(f"  Models loaded in {time.time() - t0:.1f}s")
    print()

    print("Encoding prompts with CLIP...")
    text_embeddings = encode_prompt(args.prompt, args.negative_prompt)
    print(f"  Embeddings shape: {text_embeddings.shape}")  # (2, 96, 768)
    print()

    print(f"Generating {args.frames} frames on Blackhole...")
    t1 = time.time()
    frames = generate_frames(
        device,
        ttnn_model,
        tt_vae,
        config,
        ttnn_scheduler,
        torch_time_proj,
        text_embeddings,
        num_frames=args.frames,
        seed=args.seed,
    )
    elapsed = time.time() - t1
    print(f"  Generated in {elapsed:.1f}s ({elapsed / args.frames:.1f}s/frame)")
    print()

    import ttnn
    ttnn.close_device(device)
    print("Device closed.")
    print()

    export_gif(frames, args.output)
    print(f"Saved {len(frames)} frames → {args.output}")
    print()
    print("Note: Temporal coherence from shared base noise (not AnimateDiff motion adapter).")
    print("For full AnimateDiff temporal attention, see examples/generate_baseline.py (CPU).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

```bash
cd content/projects/animatediff
python -m py_compile examples/generate_blackhole.py && echo "syntax ok"
```

Expected: `syntax ok`

- [ ] **Step 3: Commit**

```bash
git add content/projects/animatediff/examples/generate_blackhole.py
git commit -m "feat(animatediff): add Phase 2 Blackhole example (TTNN UNet, sequential frame generation)"
```

---

## Task 6: Rewrite documentation

**Files:**
- Rewrite: `content/projects/animatediff/README.md`
- Rewrite: `content/projects/animatediff/docs/IMPLEMENTATION_STATUS.md`
- Modify: `content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md`

- [ ] **Step 1: Rewrite README.md**

Replace the entire content of `content/projects/animatediff/README.md` with:

```markdown
# AnimateDiff on Tenstorrent Hardware

Two-phase implementation: **Phase 1** generates real, temporally coherent video
on CPU using the correct AnimateDiff architecture. **Phase 2** accelerates spatial
denoising on Blackhole hardware using the TTNN UNet.

---

## Background: Why the Previous Implementation Was Wrong

The original implementation applied `mm_sd_v15_v2.ckpt` (AnimateDiff motion weights)
to SD 3.5's DiT transformer (2432-dim features). These weights were trained for SD 1.5's
UNet (320-dim features). They are architecturally incompatible.

This rewrite uses the correct base: **SD 1.4 UNet** + **MotionAdapter**, where temporal
attention is injected inside each UNet transformer block at the 320-dim level.

---

## Phase 1 — Correct AnimateDiff (CPU, no hardware needed)

Uses `diffusers.AnimateDiffPipeline` with `MotionAdapter`. The MotionAdapter injects
`TemporalTransformer` attention modules at each `BasicTransformerBlock` in the SD 1.4 UNet.
Motion weights operate at 320-dim features during every denoising step — frames are
temporally coherent by design, not post-hoc.

### Setup

```bash
pip install -r requirements.txt
hf download CompVis/stable-diffusion-v1-4
hf download guoyww/animatediff-motion-adapter-v1-5-2
```

### Run

```bash
python examples/generate_baseline.py
python examples/generate_baseline.py --prompt "a sunset over the ocean" --frames 8
```

Expected output: `output/baseline.gif` — 16 frames of temporally coherent animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

Uses the SD 1.4 TTNN UNet from `~/tt-metal/models/demos/wormhole/stable_diffusion/` —
the same code runs on Blackhole via `TT_METAL_ARCH_NAME=blackhole`. Frames are denoised
sequentially using `sd_helper_funcs.run()`. Temporal coherence from shared base noise.

**Documented tradeoff:** This is TT-hardware-accelerated spatial denoising for video
frames, not full AnimateDiff temporal attention. Full integration would require injecting
`TemporalTransformer` blocks into the TTNN UNet transformer blocks.

### Requirements

- Blackhole hardware (P100 or P300c)
- `~/tt-metal` present, environment activated: `source ~/tt-metal/python_env/bin/activate`
- `hf download CompVis/stable-diffusion-v1-4` (also used by Phase 1)
- `hf download openai/clip-vit-large-patch14`

### Run

```bash
source ~/tt-metal/python_env/bin/activate
python examples/generate_blackhole.py
python examples/generate_blackhole.py --prompt "a campfire" --frames 8
```

Expected: `output/blackhole.gif` — 8 frames generated on Blackhole hardware.

---

## Code Structure

```
animatediff_ttnn/
  pipeline.py          Phase 1: thin wrapper around diffusers AnimateDiffPipeline
  ttnn_pipeline.py     Phase 2: TTNN UNet frame generation on Blackhole
  temporal_module.py   Reference only — temporal attention math (kept for study)
  __init__.py          Exports Phase 1 public API

examples/
  generate_baseline.py   Phase 1 end-to-end (CPU, any machine)
  generate_blackhole.py  Phase 2 end-to-end (Blackhole hardware)

tests/
  test_pipeline.py       Phase 1 unit tests
  test_ttnn_pipeline.py  Phase 2 unit tests (hardware-mocked)

docs/
  IMPLEMENTATION_STATUS.md  Current Phase 1/2 status
```

---

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests mock hardware dependencies and run on any machine.

---

## AnimateDiff Architecture Reference

```
SD 1.4 UNet without MotionAdapter:
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each block has BasicTransformerBlock(spatial attention)

SD 1.4 UNet WITH MotionAdapter (Phase 1):
  Noise → [Down blocks] → [Mid block] → [Up blocks] → Denoised latent
           each BasicTransformerBlock now has:
             spatial attention (unchanged)
             + TemporalTransformer(cross-frame attention, 320-dim)
                                              ↑
                              This is where mm_sd_v15_v2.ckpt weights live
```

For full AnimateDiff on Blackhole, the TTNN UNet transformer blocks would need
TemporalTransformer layers inserted — a deeper integration than Phase 2 attempts.
```

- [ ] **Step 2: Rewrite docs/IMPLEMENTATION_STATUS.md**

Replace the entire content of `content/projects/animatediff/docs/IMPLEMENTATION_STATUS.md` with:

```markdown
# AnimateDiff Implementation Status

**Last Updated:** 2026-05-14
**Version:** v0.2.0

---

## Phase 1 — Correct AnimateDiff (CPU Baseline)

**Status: ✅ Complete**

Uses `diffusers.AnimateDiffPipeline` + `MotionAdapter("guoyww/animatediff-motion-adapter-v1-5-2")`
on `CompVis/stable-diffusion-v1-4`. MotionAdapter injects temporal attention at each UNet
transformer block (320-dim features, matching mm_sd_v15_v2.ckpt). No TT hardware needed.

**Run:** `python examples/generate_baseline.py`

**Produces:** Real, temporally coherent GIF animation.

---

## Phase 2 — Blackhole-Accelerated Frame Generation

**Status: ✅ Code complete, hardware validation pending**

TTNN UNet (`UNet2D` from tt-metal SD 1.4 demo) denoises frames sequentially on Blackhole.
Temporal coherence via shared base noise. `TT_METAL_ARCH_NAME=blackhole` required.

**Run:** `python examples/generate_blackhole.py` (requires Blackhole hardware + ~/tt-metal)

**Known tradeoff:** Temporal attention is NOT applied during TTNN denoising. For full
AnimateDiff on TT hardware, TemporalTransformer blocks would need to be added to the
TTNN UNet — that is out of scope here and would require modifying tt-metal source.

---

## Root Cause of Original Implementation Failure

The original `examples/generate_with_sd35.py` attempted to use:
- **SD 3.5 DiT** (Diffusion Transformer, 2432-dim features)
- with **mm_sd_v15_v2.ckpt** motion weights trained for SD 1.5 UNet (320-dim features)

These are architecturally incompatible. The motion weights expect transformer blocks
with 320-dim hidden states; the DiT operates at 2432-dim. This mismatch meant no
temporal attention was actually being applied. Additionally, `_generate_frame_latents()`
and `_decode_latent()` were both placeholder implementations returning synthetic data.

The fix: use SD 1.4/1.5 (same UNet architecture as training) via diffusers MotionAdapter.
```

- [ ] **Step 3: Fix huggingface-cli in docs/MODEL_BRINGUP_TUTORIAL.md**

Find and replace all occurrences of `huggingface-cli` in `content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md`:

```bash
sed -i 's/huggingface-cli download/hf download/g; s/huggingface-cli login/hf auth login/g; s/huggingface-cli/hf/g' \
    content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md
```

Then verify no `huggingface-cli` remains:

```bash
grep -c "huggingface-cli" content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md
```

Expected: `0`

- [ ] **Step 4: Verify no huggingface-cli remains anywhere in the project**

```bash
grep -r "huggingface-cli" content/projects/animatediff/ 2>/dev/null
```

Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add content/projects/animatediff/README.md \
        content/projects/animatediff/docs/IMPLEMENTATION_STATUS.md \
        content/projects/animatediff/docs/MODEL_BRINGUP_TUTORIAL.md
git commit -m "docs(animatediff): rewrite README/IMPLEMENTATION_STATUS, fix hf CLI throughout"
```

---

## Final Verification

After all tasks complete, run the full test suite from the project root:

```bash
cd content/projects/animatediff
python -m pytest tests/ -v
```

Expected: 9 tests pass (5 pipeline + 4 ttnn_pipeline).

Verify no huggingface-cli anywhere:
```bash
grep -r "huggingface-cli" content/projects/animatediff/
```

Expected: no output.

Verify generate_with_sd35.py is gone:
```bash
ls content/projects/animatediff/examples/
```

Expected: `generate_baseline.py  generate_blackhole.py`
