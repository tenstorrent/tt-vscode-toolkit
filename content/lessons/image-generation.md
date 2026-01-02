---
id: image-generation
title: Image Generation with SD 3.5 Large
description: >-
  Generate high-resolution 1024x1024 images using Stable Diffusion 3.5 Large
  running natively on your Tenstorrent hardware!
category: advanced
tags:
  - hardware
  - image
  - generation
  - diffusion
  - stable
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
status: validated
estimatedMinutes: 20
---

# Image Generation with Stable Diffusion 3.5 Large

Generate images on your Tenstorrent hardware using Stable Diffusion 3.5 Large - turn text prompts into high-resolution images powered by your N150!

## What is Stable Diffusion?

**Stable Diffusion 3.5 Large** is a state-of-the-art text-to-image diffusion model that generates high-quality 1024x1024 images from text descriptions. This version uses a Multimodal Diffusion Transformer (MMDiT) architecture.

**Why Image Generation on Tenstorrent?**
- 🎨 **Native TT Acceleration** - Runs directly on Tenstorrent hardware using tt-metal
- 🔒 **Privacy** - Your prompts and images stay private
- ⚡ **High Resolution** - Generate 1024x1024 images (vs 512x512 in older models)
- 🎓 **Production Ready** - Real hardware acceleration, not CPU fallback

## Journey So Far

- **Lesson 3:** Text generation with Llama
- **Lesson 4-5:** Chat and API servers
- **Lesson 6-7:** Production deployment with vLLM
- **Lesson 8:** Image generation ← **You are here**

## Architecture

Stable Diffusion 3.5 uses a Multimodal Diffusion Transformer (MMDiT):

```text
┌──────────────────────────────────────┐
│     Text Prompt                      │
│  "If Tenstorrent were a company      │
│   in the 1960s and 1970s"            │
└─────────────┬────────────────────────┘
              │
              ▼
     ┌────────────────────────────┐
     │ 3 Text Encoders:           │
     │ • CLIP-L (OpenAI)          │ ← Encode text to embeddings
     │ • CLIP-G (OpenCLIP)        │
     │ • T5-XXL (Google)          │
     └────────┬───────────────────┘
              │
              ▼
     ┌────────────────────────────┐
     │ MMDiT Transformer          │ ← Generate latent representation
     │ (38 blocks)                │    (28 denoising steps)
     │ Running on TT Hardware     │
     └────────┬───────────────────┘
              │
              ▼
     ┌────────────────┐
     │ VAE Decoder    │ ← Convert latents to 1024x1024 pixels
     └────────┬───────┘
              │
              ▼
     ┌────────────────┐
     │ Generated      │
     │ Image (PNG)    │
     └────────────────┘
```

---

## Hardware Compatibility

Stable Diffusion 3.5 Large runs on Tenstorrent hardware with native TT-NN acceleration (not CPU fallback!):

| Hardware | Status | Performance | Notes |
|----------|--------|-------------|-------|
| **N150** (Wormhole) | ✅ Supported | ~12-15 sec/image | Optimized single-chip config |
| **N300** (Wormhole) | ✅ Supported | ~8-10 sec/image | Faster with 2 chips |
| **P100** (Blackhole) | ⚠️ Experimental (as of December 2025) | ~12-15 sec/image | Similar to N150, newer arch |
| **T3K** (Wormhole) | ✅ Supported | ~5-8 sec/image | Production scale (8 chips) |

**All hardware benefits from native TT-NN acceleration!** The model runs directly on Tensix cores using hardware-specific operators.

### Check Your Hardware

**Quick Check:** Not sure which hardware you have?

[🔍 Detect Hardware](command:tenstorrent.runHardwareDetection)

Look for the "Board Type" field in the output (e.g., n150, n300, t3k, p100).

---

## Prerequisites

- tt-metal installed and working (completed Lesson 2)
- Hugging Face account with access to SD 3.5 Large
- Tenstorrent hardware (see compatibility table above)
- ~10-15 GB disk space for model weights

---

## Model: Stable Diffusion 3.5 Large

We'll use **Stable Diffusion 3.5 Large** which runs natively on Tenstorrent hardware using tt-metal.

**Model Details:**
- **Size:** ~10 GB
- **Resolution:** 1024x1024 images (high quality!)
- **Speed:** ~12-15 seconds per image on N150
- **Architecture:** MMDiT (Multimodal Diffusion Transformer)
- **Inference Steps:** 28 (optimized for quality/speed)
- **Hardware:** Runs on TT-NN operators (native acceleration)

## Step 1: Grant Model Access

Stable Diffusion 3.5 Large requires access from Hugging Face:

1. Visit [stabilityai/stable-diffusion-3.5-large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large)
2. Click "Agree and access repository"
3. Login with your Hugging Face account

## Step 2: Authenticate with Hugging Face

Login to download the model (uses token from Lesson 3):

```bash
huggingface-cli login
```

[🔐 Login to Hugging Face](command:tenstorrent.loginHuggingFace)

The model will be automatically downloaded the first time you run it.

## Step 3: Configure for Your Hardware

Set the appropriate mesh device environment variable for your hardware:

<details open style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N150 (Wormhole - Single Chip)</b> - Most common</summary>

```bash
export MESH_DEVICE=N150
```

**Performance:** ~12-15 seconds per 1024x1024 image

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N300 (Wormhole - Dual Chip)</b></summary>

```bash
export MESH_DEVICE=N300
```

**Performance:** ~8-10 seconds per 1024x1024 image (faster with 2 chips!)

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 T3K (Wormhole - 8 Chips)</b></summary>

```bash
export MESH_DEVICE=T3K
```

**Performance:** ~5-8 seconds per 1024x1024 image (production speed!)

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 P100 (Blackhole - Single Chip)</b></summary>

```bash
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole  # Required for Blackhole
```

**Performance:** ~12-15 seconds per 1024x1024 image (similar to N150)

**⚠️ Note:** P100 support is experimental (as of December 2025). Please report any issues!

</details>

---

**What this does:**
- Tells tt-metal to configure for your specific hardware
- Optimizes model parallelization for your chip count
- Enables appropriate memory management

## Step 4: Generate Your First Image

Run the Stable Diffusion 3.5 demo with a sample prompt (using the MESH_DEVICE you set in Step 3):

```bash
mkdir -p ~/tt-scratchpad
cd ~/tt-scratchpad
export PYTHONPATH=~/tt-metal:$PYTHONPATH
# Use the MESH_DEVICE you set in Step 3 (N150, N300, T3K, or P100)

# Run with default prompt
pytest ~/tt-metal/models/experimental/stable_diffusion_35_large/demo.py
```

[🎨 Generate Sample Image](command:tenstorrent.generateRetroImage)

**What you'll see:**

```text
Loading Stable Diffusion 3.5 Large from stabilityai...
Initializing MMDiT transformer on your hardware...
✓ Model loaded on TT hardware

Generating 1024x1024 image (28 inference steps)...
Step 1/28... 2/28... 5/28... 10/28... 15/28... 20/28... 25/28... 28/28
Decoding with VAE...

✓ Image saved to: sd35_1024_1024.png
Generation time: [varies by hardware - see Step 3 performance notes]
```

The generated image will be saved to `~/tt-scratchpad/sd35_1024_1024.png`.

## Step 5: Interactive Mode - Try Your Own Prompts

Run in interactive mode to generate multiple images with custom prompts (using your MESH_DEVICE from Step 3):

```bash
mkdir -p ~/tt-scratchpad
cd ~/tt-scratchpad
export PYTHONPATH=~/tt-metal:$PYTHONPATH
# Use the MESH_DEVICE you set in Step 3

# Run interactive mode
pytest ~/tt-metal/models/experimental/stable_diffusion_35_large/demo.py
```

[🖼️ Start Interactive Image Generation](command:tenstorrent.startInteractiveImageGen)

**When prompted, enter your custom text:**

```text
Enter the input prompt, or q to exit: If Tenstorrent were a company in the 1960s and 1970s, retro corporate office, vintage computers, orange and brown color scheme
```

The model will generate a new image for each prompt and save it to `~/tt-scratchpad/sd35_1024_1024.png`. Type `q` to exit.

**Example prompts to try:**

### Literary & Cultural References

1. **Steinbeck's Computing Dust Bowl:**
```text
   "The Grapes of Wrath reimagined as 1970s computer lab, orange terminals, dusty atmosphere, vintage photograph, film grain"
```

2. **Kerouac's Electric Highway:**
```text
   "On the Road meets Silicon Valley, beat generation aesthetic, vintage mainframe computers, dharma bums coding, 1960s photography"
```

3. **Gertrude Stein's Repetition Machine:**
```text
   "A rose is a rose is a processor, cubist computing, abstract geometric circuit boards, modernist aesthetic, orange and purple"
```

4. **Whole Earth Catalog Computer Lab:**
```text
   "1970s alternative technology workshop, homebrew computer club, Stewart Brand aesthetic, orange and brown, democratic tools, vintage catalog photography"
```

### Classic Movie Computing Quotes

5. **Chocolate-Powered AI:**
```text
   "What would a computer do with a lifetime supply of chocolate? Willy Wonka meets mainframe, whimsical vintage computing, 1970s aesthetic, orange accents"
```

6. **WarGames WOPR:**
```text
   "Would you like to play a game? Cold War computing aesthetic, NORAD command center, green phosphor terminals, dramatic lighting, 1980s photography"
```

### Decidedly Tenstorrent

7. **Tensix Mandelbrot Dreams:**
```text
   "880 RISC-V cores dreaming of fractals, purple and orange silicon wafer, crystalline structure, technical diagram meets abstract art"
```

8. **Orange Silicon Valley:**
```text
   "AI accelerator as California poppy field, orange blooms, Tenstorrent hardware, golden hour lighting, Stanford Foothills, technical beauty"
```

9. **Network-on-Chip Landscape:**
```text
   "NoC topology as ancient trade routes, silicon pathways, orange and purple, cartography meets chip design, vintage map aesthetic"
```

10. **The Tensor Processing Saloon:**
```text
   "Wild West saloon but it's a 1970s computer lab, orange terminals, cowboys coding RISC-V assembly, vintage Americana, film photograph"
```

### Example Output

Here's what you can create with Stable Diffusion 3.5 on Tenstorrent hardware:

![Snowy Cabin - Generated with Stable Diffusion 3.5](../../assets/img/sd35_snowy_cabin.png)

*Generated with prompt: "A cozy cabin in a snowy forest, warm lights in windows, winter evening, oil painting style"*

**Generation details:**
- Resolution: 1024x1024
- Steps: 28
- Hardware: N150 (single Wormhole chip)
- Time: ~2-3 minutes (first run includes model load)

---

## Step 6: Experiment with Code (Advanced)

**Ready to go beyond button-pressing?** Copy the demo to your scratchpad and modify it:

[📝 Copy Demo to Scratchpad](command:tenstorrent.copyImageGenDemo)

This copies `demo.py` to `~/tt-scratchpad/sd35_demo.py` and opens it for editing.

**What you can experiment with:**

1. **Batch generation with variations:**
```python
# Generate multiple images with seed variations
prompts = [
    "Whole Earth Catalog computer lab, 1970s",
    "Kerouac typing on vintage terminal, beat aesthetic",
    "Would you like to play a game? WOPR terminal"
]

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=i  # Different seed for each
    )
    image.save(f"tenstorrent_{i:03d}.png")
```

2. **Parameter exploration:**
```python
# Try different guidance scales to see impact on adherence to prompt
for scale in [2.0, 3.5, 5.0, 7.5]:
    image = pipe(
        prompt="Tenstorrent headquarters, orange architecture",
        guidance_scale=scale
    )
    image.save(f"guidance_{scale}.png")
```

3. **Prompt interpolation:**
```python
# Blend between two concepts
prompts = [
    "1960s mainframe computer room",
    "futuristic AI accelerator lab"
]
# Generate with weighted combination
```

4. **Custom resolution experiments:**
```python
# Try different aspect ratios (must be multiples of 64)
image = pipe(
    prompt="Wide cinematic shot of vintage computing",
    width=1536,  # 16:9 aspect ratio
    height=864
)
```

**Tips for code experiments:**
- Model stays loaded between generations (fast iterations!)
- Save images with descriptive names: `prompt_seed_guidance.png`
- Keep `num_inference_steps=28` (optimized for SD 3.5)
- Experiment with `guidance_scale` between 2.0-7.5
- Use seeds for reproducibility (same seed = same image)

**Make it your own!** The demo is just a starting point - modify, extend, and create your own image generation workflows.

## Understanding the Generation Process

### **Diffusion Process in SD 3.5:**

1. **Text Encoding** - Three encoders (CLIP-L, CLIP-G, T5-XXL) process your prompt
2. **Start with noise** - Begin with random latent representation
3. **Denoise iteratively** - MMDiT transformer removes noise in 28 steps
4. **Each step runs on TT hardware** - Native acceleration on N150
5. **VAE Decoding** - Convert latents to 1024x1024 pixel image

### **Key Parameters:**

**num_inference_steps (28)**
- Optimized for SD 3.5 Large
- Balances quality and speed
- Fixed in the demo (can't be changed without model retraining)

**guidance_scale (3.5)**
- How closely to follow your prompt
- 3.5: Optimized default for SD 3.5
- Lower than SD 1.x because of improved architecture

**image_w, image_h (1024x1024)**
- High resolution output
- Can be adjusted but 1024x1024 is optimal for SD 3.5

**seed (0)**
- Random seed for reproducibility
- Same seed + same prompt = same image
- Useful for iterating on prompts

## Prompt Engineering Tips

**Good prompts include:**
1. **Subject** - What you want to see
2. **Style** - Art style, photography type
3. **Colors** - Color scheme
4. **Lighting** - Lighting conditions
5. **Details** - Specific details to include

**Example:**
```text
"Vintage 1970s office, orange and brown color scheme, retro computers,
warm lighting, film photograph, detailed, high quality"
```

**Keywords that work well:**
- Art styles: `photorealistic`, `digital art`, `oil painting`, `sketch`
- Quality: `detailed`, `high quality`, `8k`, `professional`
- Lighting: `studio lighting`, `natural light`, `dramatic lighting`
- Camera: `35mm photograph`, `wide angle`, `close-up`

## Performance Optimization

**For faster generation on N150:**

1. **Reduce steps:**
   ```bash
   --steps 30  # Instead of 50
```

2. **Lower resolution:**
   ```bash
   --width 256 --height 256  # Instead of 512x512
```

3. **Use attention slicing:**
   The script automatically enables this for N150 to reduce memory usage

## Comparing Generation Speed

| Hardware | Steps | Time | Notes |
|----------|-------|------|-------|
| CPU Only | 50 | ~5-10 min | Very slow |
| **N150** | 50 | ~15-30 sec | Accelerated |
| N300 | 50 | ~10-20 sec | Faster (2 chips) |
| High-end GPU | 50 | ~5-10 sec | Comparison |

## Troubleshooting

**Device reset between models (optional):**

If you experience issues after running other models (like Llama from earlier lessons), you can reset the device:

```bash
tt-smi -r
```

This clears device state and memory. **Usually not needed** between pytest demos, but useful if:
- Previous demo crashed or hung
- You see "out of memory" or device errors
- Device behaves unexpectedly
- Switching between very different workloads

Most pytest tests automatically clean up the device, so this is only needed if something went wrong.

**Model download fails:**
```bash
# Check Hugging Face authentication
huggingface-cli whoami

# Make sure you granted access at HuggingFace
# Visit: https://huggingface.co/stabilityai/stable-diffusion-3.5-large
```

**Slow first generation:**
- First run downloads the model (~10 GB) which takes 5-10 minutes
- First generation loads model into device (2-5 min)
- Subsequent generations are much faster (~12-15 sec)
- This is normal behavior

**Device hangs or crashes:**
```bash
# Reset the device
tt-smi -r

# If that doesn't work, clear device state completely
sudo rm -rf /dev/shm/tenstorrent* /dev/shm/tt_*
tt-smi -r
```

## What You Learned

- ✅ How to set up Stable Diffusion on Tenstorrent hardware
- ✅ Text-to-image generation with custom prompts
- ✅ Understanding diffusion model parameters
- ✅ Prompt engineering for better results
- ✅ Batch generation and optimization

**Key takeaway:** You can generate high-quality images locally on your Tenstorrent hardware, with full control over the generation process and complete privacy.

## Next Steps

**Experiment with:**
1. **Different prompts** - Try various subjects and styles
2. **Parameter tuning** - Adjust steps, guidance_scale, and seed
3. **Batch generation** - Create variations of successful prompts
4. **Image-to-image** - Use generated images as starting points (advanced)

**Advanced topics:**
- Fine-tuning Stable Diffusion on custom images
- Inpainting (editing parts of images)
- ControlNet for precise control
- Integrating with web interfaces

## Resources

- **Stable Diffusion:** [stability.ai](https://stability.ai/)
- **Hugging Face Diffusers:** [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- **Prompt Engineering Guide:** [prompthero.com](https://prompthero.com/)
- **TT-Metal Docs:** [docs.tenstorrent.com](https://docs.tenstorrent.com/)

**Happy generating! 🎨**
