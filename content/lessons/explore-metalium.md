---
id: explore-metalium
title: Exploring TT-Metalium
description: >-
  Discover what's possible with TT-Metalium! Run TTNN operations immediately,
  explore the model zoo, and understand the architecture that powers
  Tenstorrent hardware — from first script to custom kernels.
category: advanced
tags:
  - model
  - ttnn
  - metalium
  - architecture
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: validated
validatedOn:
  - n150
  - p300c
estimatedMinutes: 30
---

# Exploring the TT-Metalium Playground

Welcome to the heart of Tenstorrent development! In this lesson you'll discover what's
possible with **TT-Metalium** and **TTNN**, run real hardware code in minutes, and
understand the architecture that makes it all tick.

## What You'll Do

- ⚡ Run your first TTNN operation on TT hardware in five lines of code
- 🧠 Understand tile-based computing and the Tensix core
- 🏗️ Explore the three-kernel programming model
- 📚 Browse the model zoo and Jupyter tutorials
- 🔧 See the path from TTNN (high-level Python) to TT-Metalium (custom C++ kernels)

---

## Before You Start: Run This Right Now

If you have tt-metal built and your venv activated, you can be running real TTNN code in
60 seconds. No Jupyter, no setup — just Python:

```bash
# Activate the tt-metal Python environment
source ~/tt-metal/python_env/bin/activate
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# Run the first tutorial — adds two tensors on TT hardware
python3 ~/tt-metal/ttnn/tutorials/basic_python/ttnn_add_tensors.py
```

You'll see the device open, the computation run, and the device close. That's real
silicon doing real work. The full tutorial collection lives at:

```
~/tt-metal/ttnn/tutorials/basic_python/
```

**No Jupyter required** — every notebook also has a `.py` companion you can run
directly.

> **Don't have `~/tt-metal` built yet?** Start with
> [Build tt-metal from Source](command:tenstorrent.showLesson?["build-tt-metal"])
> first, then return here.

---

## Why This Hardware is Different

Before diving in, here's what makes Tenstorrent hardware worth exploring:

**Wormhole N150 (single chip, 8 TOPS):**
- Runs Llama 3.1 8B at ~20 tok/s
- Generates 512×512 images in ~30s with Stable Diffusion
- Runs BERT-Large inference at ~400 sentences/sec

**Tenstorrent Galaxy (32 Wormhole chips, 256 TOPS):**
- Runs DeepSeek-V3 (685B parameters) in production
- Stable Diffusion 3.5 Large in 5.6 seconds per image
- Llama 3 70B at hundreds of tok/s

**The same TTNN Python code runs on all of these.** You write for N150, scale to Galaxy
by changing a device count. That's the architecture advantage this lesson explores.

---

## Part 1: Run the Tutorial Scripts

### The Quickest Path: basic_python Scripts

Every TTNN concept has a runnable Python script. These are the best starting point
because they don't require Jupyter and have clear, commented code:

```bash
cd ~/tt-metal
source python_env/bin/activate

# Tensor basics: create, fill, add on device
python3 ttnn/tutorials/basic_python/ttnn_add_tensors.py

# Core operations: element-wise, reductions, broadcasting
python3 ttnn/tutorials/basic_python/ttnn_basic_operations.py

# Matrix multiplication: the workhorse of neural nets
python3 ttnn/tutorials/basic_python/ttnn_basic_matrix_multiplication.py

# 2D convolution on TT hardware
python3 ttnn/tutorials/basic_python/ttnn_basic_conv.py

# Full inference pipeline: MLP on MNIST
python3 ttnn/tutorials/basic_python/ttnn_mlp_inference_mnist.py

# Transformer building block: multi-head attention
python3 ttnn/tutorials/basic_python/ttnn_multihead_attention.py

# CNN inference end-to-end
python3 ttnn/tutorials/basic_python/ttnn_simplecnn_inference.py
```

**Recommended order:** `ttnn_add_tensors` → `ttnn_basic_operations` →
`ttnn_basic_matrix_multiplication` → `ttnn_mlp_inference_mnist`.

### Jupyter Notebooks

If you prefer interactive Jupyter notebooks, the same content is available as `.ipynb`
files in the same directory:

```
~/tt-metal/ttnn/tutorials/
```

[📓 Open TTNN Tutorials](command:tenstorrent.launchTtnnTutorials)

**Available notebooks:**
- `ttnn_intro.ipynb` — Introduction to TTNN concepts
- `ttnn_add_tensors.ipynb` — Tensor creation and addition
- `ttnn_basic_operations.ipynb` — Element-wise ops, reductions
- `ttnn_basic_matrix_multiplication.ipynb` — matmul deep dive
- `ttnn_basic_conv.ipynb` — 2D convolution fundamentals
- `ttnn_mlp_inference_mnist.ipynb` — Complete inference pipeline
- `ttnn_multihead_attention.ipynb` — Transformer building blocks
- `ttnn_simplecnn_inference.ipynb` — End-to-end CNN example
- `ttnn_clip_zero_shot_classification.ipynb` — CLIP model inference

---

## Part 2: The Model Zoo — What Runs Today

Tenstorrent's model repository is one of the most extensive collections of
hardware-optimized AI models available. Here's what you can run right now:

[🔍 Browse Model Zoo](command:tenstorrent.browseModelZoo)

### Production-Ready (models/demos/)

**Language Models:**
- **Llama 3.1 8B** — Chat, code, reasoning (N150/N300)
- **Llama 3 70B** — Large-scale inference (Galaxy, 32 chips)
- **DeepSeek-V3** — State-of-the-art reasoning (Galaxy)
- **Gemma 3 27B** — Multimodal text+image, 128K context (N300/T3K)
- **Qwen 2.5 VL** — Vision-language understanding

**Vision Models:**
- **Stable Diffusion 1.4** — Text-to-image (N150/N300/P100)
- **YOLO v10/v11/v12** — Real-time object detection
- **SegFormer** — Semantic segmentation
- **SigLIP** — Image-text matching
- **ResNet50, MobileNetV2** — Image classification at speed
- **BERT, DistilBERT** — NLP understanding

**Audio:**
- **Whisper** — Speech-to-text transcription

### Experimental (models/experimental/)

- **Stable Diffusion 3.5 Large** — via tt-dit (Galaxy/QuietBox 8+ chips)
- **Flux 1** — Text-to-image generation
- **Mochi-1** — Native video generation
- **Wan 2.2** — Text-to-video model
- **nanoGPT** — Train a GPT from scratch on device
- **Grok** — xAI reasoning model port

### Hardware-Organized Demos

Models are organized by target hardware for easy discovery:

```
models/demos/wormhole/   — N150/N300 optimized
models/demos/t3000/      — T3K (8-chip) configurations
models/demos/blackhole/  — P100/P300c (Blackhole)
models/demos/tg/         — Galaxy (32-chip)
```

**🎯 What's possible:**
1. **Run a 685B parameter model** — DeepSeek-V3 on Galaxy
2. **128K context windows** — Read entire books as context
3. **Real-time object detection** — YOLO v12 on N150
4. **Train models on device** — nanoGPT is buildable from scratch
5. **Native video generation** — Mochi and Wan 2.2 (experimental)

---

## Part 3: Understanding the Architecture

### The Tensix Core

Each Tenstorrent chip contains a grid of **Tensix cores**. Understanding their
architecture helps you write efficient code.

**Inside a Tensix Core:**

```text
┌─────────────────────────────────────────────────┐
│                  Tensix Core                    │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────┐    ┌────────────────┐           │
│  │  5 RISC-V│───▶│  1.5 MB SRAM   │           │
│  │  "Baby"  │    │   (L1 Memory)  │           │
│  │  CPUs    │    └────────────────┘           │
│  └──────────┘            │                     │
│                           │                     │
│       ┌───────────────────┴──────────┐         │
│       │                               │         │
│  ┌────▼─────┐                  ┌─────▼────┐   │
│  │  Matrix  │                  │  Vector  │   │
│  │  Engine  │                  │  Unit    │   │
│  │  (FPU)   │                  │  (SFPU)  │   │
│  │          │                  │          │   │
│  │ 32×32    │                  │ Element- │   │
│  │ Tiles    │                  │  wise    │   │
│  └──────────┘                  └──────────┘   │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │  Network-on-Chip (NoC) - 2 Paths        │ │
│  │  NoC 0: Reads    NoC 1: Writes          │ │
│  └──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
        │                           │
        ▼                           ▼
    DRAM Banks              Other Tensix Cores
```

**Key components:**

1. **5 RISC-V "Baby" CPUs** — Control and orchestration; run your kernel code
2. **1.5 MB L1 SRAM** — Fast local memory, explicitly managed (no cache)
3. **Matrix Engine (FPU)** — Hardware accelerator for 32×32 tile matmul
4. **Vector Unit (SFPU)** — Element-wise ops: ReLU, GELU, Softmax, custom math
5. **Network-on-Chip (NoC)** — Two independent paths; connects DRAM and cores

---

### Tile-Based Computing

**Why 32×32 tiles?**

Traditional GPUs process data in linear layouts. Tenstorrent uses **32×32 tiles** as
the native format because it matches the Matrix Engine hardware perfectly:

```python
import ttnn
import torch

device = ttnn.open_device(device_id=0)

# ROW_MAJOR layout (like NumPy/PyTorch)
row_major = ttnn.from_torch(
    torch.rand((3, 4)),
    layout=ttnn.ROW_MAJOR_LAYOUT,
    device=device
)
print(f"Shape: {row_major.shape}, Padded: {row_major.padded_shape}")
# Output: Shape([3, 4]), Padded: Shape([3, 4])

# TILE_LAYOUT — native format, padded to 32×32 minimum
tile = ttnn.to_layout(row_major, ttnn.TILE_LAYOUT)
print(f"Shape: {tile.shape}, Padded: {tile.padded_shape}")
# Output: Shape([3, 4]), Padded: Shape([32, 32])
# Padding added automatically to fill 32×32 tile!

ttnn.close_device(device)
```

**Performance tip:** Operations on tile-aligned shapes (multiples of 32) are fastest!
Non-aligned shapes work but waste some compute on the padding.

---

### The Three-Kernel Programming Model

Most operations use three kernels working together in a pipeline:

```text
     Reader Kernel              Compute Kernel             Writer Kernel
     (Data Movement)            (Math Operations)         (Data Movement)
            │                          │                         │
┌───────────▼──────────┐   ┌──────────▼─────────┐   ┌──────────▼─────────┐
│  Fetch from DRAM     │──▶│  Process in SRAM   │──▶│  Store to DRAM     │
│  via NoC 0           │   │  (Matrix/Vector)   │   │  via NoC 1         │
└──────────────────────┘   └────────────────────┘   └────────────────────┘

Circular Buffers in L1 SRAM enable pipelining:
- Reader fills buffer while Compute processes previous batch
- Compute fills output buffer while Writer stores previous batch
```

This architecture means there is **no hidden cache thrashing** — every data movement is
explicit. That's why profiling Metalium programs is precise: you know exactly what's
moving where.

---

### Two Levels of Abstraction

**TTNN (Python) — High Level:**

```python
import ttnn

device = ttnn.open_device(device_id=0)

a = ttnn.rand((32, 32), device=device, layout=ttnn.TILE_LAYOUT)
b = ttnn.rand((32, 32), device=device, layout=ttnn.TILE_LAYOUT)

c = ttnn.matmul(a, b)   # Matrix multiply
d = ttnn.add(c, 1.0)    # Add scalar
e = ttnn.gelu(d)        # Activation

result = ttnn.to_torch(e)
ttnn.close_device(device)
```

Use TTNN for: rapid prototyping, standard model inference, Python-first development.

---

**TT-Metalium (C++) — Low Level:**

```cpp
#include "tt_metal/host_api.hpp"

using namespace tt::tt_metal;

int main() {
    Device* device = CreateDevice(0);
    Program program = CreateProgram();

    // Define reader, compute, and writer kernels
    auto reader = CreateKernel(program, "kernels/reader.cpp", core,
                               DataMovementConfig{...});
    auto compute = CreateKernel(program, "kernels/compute.cpp", core,
                                ComputeConfig{...});

    EnqueueProgram(command_queue, program, false);
    Finish(command_queue);
    CloseDevice(device);
}
```

Use TT-Metalium for: maximum performance, custom operations, novel algorithms, research.

---

## Part 4: Programming Examples

### Build and Run Examples

The programming examples demonstrate Metalium kernels from hello world through
multi-core matrix multiply. Build them with:

```bash
cd ~/tt-metal
./build_metal.sh --build-programming-examples
```

**This takes an additional 5–10 minutes but gives you standalone executables.**

**Beginner:**

| Example | What It Teaches |
|---------|-----------------|
| Hello World Compute | Your first compute kernel |
| Hello World Data Movement | Your first reader/writer kernel |
| Add 2 Integers | Basic arithmetic on device |
| DRAM Loopback | Buffer creation, data movement |

```bash
# Run after building with --build-programming-examples
./build/programming_examples/hello_world_compute_kernel
./build/programming_examples/hello_world_datamovement_kernel
./build/programming_examples/add_2_integers_in_compute
```

**Intermediate:**

| Example | What It Teaches |
|---------|-----------------|
| Eltwise Binary | Element-wise ops with circular buffers |
| Eltwise SFPU | Vector operations (SFPU math) |
| Matmul Single Core | Using the matrix engine |
| Matmul Multi Core | Parallel execution across cores |

---

## Hands-On: Tile Padding Experiment

Run this short script to see how TTNN handles the 32×32 tile requirement:

```bash
cat > /tmp/tile_experiment.py << 'EOF'
import ttnn
import torch

device = ttnn.open_device(device_id=0)

cases = [(5, 5), (100, 50), (128, 128), (1024, 1024)]

for shape in cases:
    t = ttnn.from_torch(
        torch.rand(shape),
        layout=ttnn.TILE_LAYOUT,
        device=device
    )
    pad_r = t.padded_shape[-2] - shape[0]
    pad_c = t.padded_shape[-1] - shape[1]
    print(f"{shape[0]:5}×{shape[1]:<5}  →  padded {t.padded_shape[-2]}×{t.padded_shape[-1]}  "
          f"(wasted: {pad_r * t.padded_shape[-1] + pad_c * shape[0]} elements)")

ttnn.close_device(device)
print("\nRule: dimensions always pad to next multiple of 32.")
print("For best performance, design your model shapes to be multiples of 32.")
EOF
cd ~/tt-metal && python3 /tmp/tile_experiment.py
```

**Observe:**
- How much padding each shape requires
- Why 128×128 and 1024×1024 are "free" (already tile-aligned)
- What the padding cost is for 5×5 (nearly 4× the data!)

---

## Key Takeaways

- ✅ **TTNN runs on every Tenstorrent chip** — write once, scale from N150 to Galaxy
- ✅ **Tile-based computing** (32×32) is the native format — align your shapes!
- ✅ **Three-kernel model** (Reader→Compute→Writer) enables pipelined execution
- ✅ **Explicit memory** (L1 SRAM) instead of caches — predictable performance
- ✅ **Production models exist** for LLMs, vision, audio, video, and more
- ✅ **Both levels matter**: TTNN for productivity, Metalium for maximum performance

---

## What's Next?

In the **Metalium Cookbook**, you'll apply these concepts building four creative projects:

1. **Conway's Game of Life** — Cellular automata with parallel tile computing
2. **Audio Processor** — Real-time mel-spectrogram and effects
3. **Mandelbrot Explorer** — GPU-style fractal rendering
4. **Custom Image Filters** — Creative visual effects

[🚀 Continue to JAX Inference with TT-XLA](command:tenstorrent.showLesson?["tt-xla-jax"])

---

## Resources

- **METALIUM_GUIDE.md**: `~/tt-metal/METALIUM_GUIDE.md` ⭐ — Architecture deep-dive
- **Tutorial scripts**: `~/tt-metal/ttnn/tutorials/basic_python/` — Runnable Python files
- **Jupyter notebooks**: `~/tt-metal/ttnn/tutorials/` — Interactive notebooks
- **Programming examples**: `~/tt-metal/tt_metal/programming_examples/`
- **Tech reports**: `~/tt-metal/tech_reports/` — Flash Attention, architecture papers
- **Official docs**: [docs.tenstorrent.com](https://docs.tenstorrent.com)
- **Discord**: [discord.gg/tvhGzHQwaj](https://discord.gg/tvhGzHQwaj)

---

## Troubleshooting

**`ttnn.open_device()` fails:**
```bash
tt-smi    # Check device status
tt-smi -r # Reset if showing errors
```

**Jupyter notebooks won't open:**
```bash
code --install-extension ms-toolsai.jupyter
```

**Out of memory:**
- Reduce batch sizes
- Use tile-aligned dimensions (multiples of 32)
- Release tensors: `ttnn.deallocate(tensor)`

**Slow performance:**
- Non-tile-aligned shapes add padding overhead — use multiples of 32
- Minimize `to_torch()`/`from_torch()` round-trips
- Always set `layout=ttnn.TILE_LAYOUT` for compute-intensive ops
