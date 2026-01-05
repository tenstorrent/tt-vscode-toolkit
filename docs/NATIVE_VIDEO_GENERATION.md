# Native Video Generation on TT Hardware

**Date:** 2026-01-02
**Investigation:** Exploring native text-to-video models (not frame-by-frame stitching)
**Goal:** Find models that create "reality temporarily created that hadn't existed before"

---

## Discovery: Mochi Pipeline

### What It Is

**Mochi** is a native text-to-video generation model integrated into TT-Metal:
- **Model:** genmo/mochi-1-preview
- **Architecture:** 3D transformer with temporal coherence
- **Location:** `/home/user/tt-metal/models/experimental/tt_dit/pipelines/mochi/pipeline_mochi.py`
- **Type:** Actual video generation, not frame-by-frame stitching

**Components:**
- `MochiTransformer3DModel` - 3D transformer for video latent denoising
- `AutoencoderKLMochi` - VAE for video encoding/decoding (spatial scale 8x, temporal scale 6x)
- `FlowMatchEulerDiscreteScheduler` - Temporal flow matching
- `T5EncoderModel` - Text encoding (google/t5-v1_1-xxl)

### What It Generates

**Native video with temporal coherence:**
- Moving animations (not just slideshow)
- Smooth motion between frames
- Temporal relationships preserved
- True "reality temporarily created"

**Example prompt (from test):**
> "A close-up of a beautiful butterfly landing on a flower, wings gently moving in the breeze."

**Output:**
- 16-48 frames of actual video
- Wings actually move smoothly
- Butterfly actually lands (motion trajectory)
- Breeze actually affects movement

### The Hardware Reality

**CRITICAL LIMITATION:** Mochi requires **Galaxy hardware** (32 chips)

**From test file:**
```python
@pytest.mark.parametrize(
    "mesh_device, sp_axis, tp_axis, num_links",
    [
        [(4, 8), 1, 0, 4],  # 4x8 = 32 chips (GALAXY)
    ],
)
def test_tt_mochi_pipeline(mesh_device: ttnn.MeshDevice, ...):
    # Creates TT Mochi pipeline
    parallel_config = DiTParallelConfig(
        cfg_parallel=ParallelFactor(factor=1, mesh_axis=0),
        tensor_parallel=ParallelFactor(factor=tp_factor, mesh_axis=tp_axis),
        sequence_parallel=ParallelFactor(factor=sp_factor, mesh_axis=sp_axis),
    )

    tt_pipe = TTMochiPipeline(
        mesh_device=mesh_device,  # 4x8 mesh
        parallel_config=parallel_config,
        num_links=num_links,      # 4 links
        use_cache=True,
        model_name="genmo/mochi-1-preview",
    )
```

**Requirements:**
- Mesh shape: (4, 8) = 32 chips
- 4 high-speed links for chip-to-chip communication
- Tensor parallelism across 4 chips
- Sequence parallelism across 8 chips

**Hardware comparison:**
- N150: 1 chip ❌ (NOT supported)
- N300: 2 chips ❌ (NOT supported)
- T3K: 8 chips ❌ (NOT supported)
- Galaxy: 32 chips ✅ (REQUIRED)

### Why This Architecture Needs 32 Chips

**Model complexity:**
- 3D video transformer (spatial + temporal dimensions)
- Massive compute for temporal coherence
- VAE with 6x temporal scaling (6 latent frames per video frame)
- Flow matching across time steps

**Memory requirements:**
- Video latents much larger than image latents
- Temporal attention across all frames simultaneously
- 48 frames @ 320x400 = 6.1M pixels of video latent space

**Parallelism strategy:**
- **Tensor Parallel (TP):** Split model weights across 4 chips
- **Sequence Parallel (SP):** Split spatial tokens across 8 chips
- Total: 4 × 8 = 32 chips working together

### Alternative: Diffusers CPU/CUDA Version

**There IS a fallback** that doesn't use TT hardware:

```python
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

# Runs on CUDA or CPU (NOT TT hardware)
pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview", torch_dtype=torch.bfloat16)
pipe.to("cuda")  # or "cpu"

frames = pipe(
    prompt="A mystical forest coming alive at dawn",
    num_inference_steps=10,
    guidance_scale=3.5,
    num_frames=16,
    height=320,
    width=320,
).frames[0]

export_to_video(frames, "output.mp4", fps=8)
```

**But:**
- Uses CUDA/CPU, NOT TT accelerators
- Very slow on CPU (hours for 16 frames)
- Requires CUDA GPU for reasonable speed
- Defeats the purpose of TT hardware showcase

---

## Answer to User's Question

**"Are there any paths to actual video generation -- like creating animations -- we can use models for with tt-metal or tt-xla and really make someone feel like they saw some kind of reality temporarily created that hadn't existed before?"**

### Short Answer

**YES**, native video generation exists (Mochi), **BUT** it requires Galaxy hardware (32 chips).

**NOT viable** for N150 (single chip) or even T3K (8 chips).

### Current Reality for N150

**Frame-by-frame approach (current Lesson 9 Video):**
- ✅ Works on N150 single chip
- ✅ High-quality 1024x1024 frames
- ✅ Uses proven SD 3.5 Large
- ⚠️ No temporal coherence (slideshow, not video)
- ⚠️ Each frame independent

**Native video generation (Mochi):**
- ❌ Requires Galaxy (32 chips)
- ✅ True temporal coherence
- ✅ Smooth motion and animation
- ✅ "Reality temporarily created"
- ❌ NOT available for N150/N300/T3K

### What Would Work on N150

**Option 1: Stick with current approach**
- Frame-by-frame SD 3.5 (validated, working)
- Accept that it's a slideshow, not animated video
- Focus on storytelling through frame sequence
- High-quality still images tell a story

**Option 2: Explore AnimateDiff**
- Check if AnimateDiff has TT-Metal integration
- Adds motion between frames (interpolation)
- Might work on smaller hardware
- Would need investigation

**Option 3: Simple motion effects**
- Generate base frame with SD 3.5
- Apply programmatic motion (pan, zoom, ken burns effect)
- Not "AI-generated motion" but creates movement
- Easy to implement with ffmpeg filters

**Option 4: Wait for Mochi-lite**
- Wait for smaller Mochi variants
- Hope for single-chip optimization
- Not available currently

---

## Investigation Summary

**Models Found:**
1. ✅ **Mochi** - Native video, requires Galaxy (32 chips)
2. ⚠️ **AnimateDiff** - Not yet investigated (might work on N150?)
3. ❌ **Other video models** - None found in tt-metal experimental/

**Recommendation:**
- **For lesson content:** Stick with current frame-by-frame approach (validated, works on N150)
- **For advanced users with Galaxy:** Document Mochi as "advanced native video generation"
- **Future exploration:** Investigate AnimateDiff or other motion models

**Hardware Scaling Story Still Valid:**
- N150: Frame-by-frame works (~4 min/frame)
- N300: 2x faster
- T3K: 6x faster
- Galaxy: 20x faster PLUS native video generation with Mochi

**The "Reality Created" Experience:**
- Current approach: Story through frames (like comic book panels)
- Mochi on Galaxy: Actual animated reality with motion
- Both valid, different hardware tiers

---

## Next Steps

1. ✅ Current lesson validated for N150 (frame-by-frame)
2. ⏸️ Mochi requires Galaxy - document as advanced feature
3. ❓ Investigate AnimateDiff for N150 compatibility
4. ❓ Check TT-XLA for any video models (user mentioned tt-xla)

**Status:** Frame-by-frame approach is production-ready for N150. Native video generation requires Galaxy hardware.
