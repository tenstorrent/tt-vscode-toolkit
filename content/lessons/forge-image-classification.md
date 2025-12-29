---
id: forge-image-classification
title: Image Classification with TT-Forge
description: >-
  Explore TT-Forge - Tenstorrent's MLIR-based compiler that brings PyTorch
  models to TT hardware! Start with validated models like MobileNetV2 for image
  classification. Learn forge.compile(), experiment with supported
  architectures, and understand the path to high-level model deployment.
category: advanced
tags:
  - hardware
  - image
  - forge
  - deployment
  - model
supportedHardware:
  - n150
status: blocked
estimatedMinutes: 20
---

# Lesson 11: Image Classification with TT-Forge

## Welcome to the High-Level Compiler! 🎨

You've been working with **TT-Metal** (low-level kernels) and **vLLM** (production LLM serving). Now meet **TT-Forge**: Tenstorrent's **MLIR-based compiler** that aims to bring PyTorch models to TT hardware with less manual kernel programming.

**The Goal:**
```python
import forge
import torch
import torchvision

# PyTorch model
model = torchvision.models.mobilenet_v2(pretrained=True)
model.eval()

# Compile for TT hardware
sample_input = torch.randn(1, 3, 224, 224)
compiled_model = forge.compile(model, sample_inputs=[sample_input])

# Run on TT accelerators
output = compiled_model(input_tensor)
```

**Why TT-Forge?**
- ✅ **Higher-level API:** Simpler than manual TT-Metal programming
- ✅ **PyTorch integration:** Works with torchvision models
- ✅ **Automatic optimization:** Graph-level transformations
- ⚠️ **Experimental status:** Under active development (as of December 2025), limited model support

---

## What is TT-Forge?

**TT-Forge** is Tenstorrent's **MLIR-based compiler** that attempts to automatically convert PyTorch models to TT hardware:

```text
┌─────────────────────────────────────────┐
│         PyTorch Models                  │  ← Your code
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│        Forge Frontend                   │  ← Graph capture
│     (Trace PyTorch operations)          │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         MLIR Compiler                   │  ← Optimization passes
│   (Fusion, layout, operator lowering)   │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│              TTNN                       │  ← TT Neural Net ops
│        (TT-Metal operations)            │
└──────────────────┬──────────────────────┘
                   │
┌──────────────────▼──────────────────────┐
│         N150 / N300 / T3K               │  ← Hardware (single-chip)
└─────────────────────────────────────────┘
```

**Key Benefits:**
- ✅ **Simpler than TT-Metal:** Automatic graph optimization
- ✅ **PyTorch native:** Use familiar torchvision models
- ✅ **Validated models:** 169 tested examples in tt-forge-models

**Limitations:**
- ⚠️ **Experimental (as of December 2025):** Many models fail to compile
- ⚠️ **Single-chip only:** No multi-chip support yet
- ⚠️ **Limited operators:** Not all PyTorch ops supported

**Compiler Stack Comparison:**

| Compiler | Maturity | Multi-chip | Frameworks | Use Case |
|----------|----------|------------|------------|----------|
| TT-XLA | Production | ✅ Yes (TP/DP) | PyTorch, JAX | Production multi-chip |
| **TT-Forge** | **Beta** | ❌ Single-chip | PyTorch, ONNX | **This lesson - experimental** |
| TT-Metal | Stable | ✅ Yes | Direct API | Low-level control |

**Current Status (December 2025):** TT-Forge is experimental. Start with validated models from tt-forge-models. For production workloads, use TT-Metal directly or wait for TT-XLA maturity.

---

## Why Start with Image Classification?

**Visual feedback and validated models!** You'll:
1. ✅ Run MobileNetV2 - a validated model with known working status
2. ✅ See classification results on real images
3. ✅ Understand the workflow for supported models
4. ✅ Learn which model patterns work reliably

**What works reliably (validated in tt-forge-models):**
- MobileNetV1/V2/V3 (computer vision)
- ResNet variants (classification)
- Some BERT models (NLP)
- Select models from the 169 validated examples

**What might not work:**
- Very new architectures (e.g., recent transformers)
- Models with unsupported operators
- Custom layers or non-standard operations
- Models not in the validated set

**Strategy:** Start with validated examples, then experiment cautiously with other models.

---

## Step 1: Install TT-XLA

**Prerequisites:**
- tt-metal already installed and working (from Lessons 1-10)
- tt-mlir toolchain must be built first
- clang-17 installed (for building)
- Build tools: `sudo apt install build-essential cmake ninja-build clang-17`

**⚠️ IMPORTANT:** TT-XLA currently supports **Nebula boards only** (N150/N300/T3K). Galaxy board support is coming soon (as of December 2025).

---

### Step 1a: Install and Build TT-MLIR Toolchain

TT-XLA depends on the tt-mlir toolchain. You need to build it first:

```bash
# Clone tt-mlir
cd ~
git clone https://github.com/tenstorrent/tt-mlir.git
cd tt-mlir

# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Build tt-mlir toolchain
cmake -G Ninja -B build
cmake --build build

# Set environment variable (add to ~/.bashrc for persistence)
export TTMLIR_TOOLCHAIN_DIR=~/tt-mlir/build
```

**Verify tt-mlir installation:**
```bash
ls $TTMLIR_TOOLCHAIN_DIR/bin/ttmlir-opt
# Should show: /home/user/tt-mlir/build/bin/ttmlir-opt
```

---

### Step 1b: Build TT-XLA from Source

Now build TT-XLA against your tt-mlir toolchain:

```bash
# Clone tt-xla
cd ~
git clone https://github.com/tenstorrent/tt-xla.git
cd tt-xla

# Activate Python environment (or create new one)
python3 -m venv venv
source venv/bin/activate

# Ensure clang-17 is used
export CC=clang-17
export CXX=clang++-17

# Build tt-xla
cmake -G Ninja -B build
cmake --build build

# Install Python package
pip install -e .
```

**Build time:** 15-30 minutes depending on your system.
- **Python 3.11** (will be installed in build steps - required by JAX 0.7.1)

**Build steps:**

```bash
# 0. CRITICAL: Clear environment variables first
unset TT_METAL_HOME
unset TT_METAL_VERSION

# 1. Install Python 3.11 (required by JAX 0.7.1)
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# 2. Create user-owned toolchain directories (avoid permission issues)
mkdir -p ~/ttforge-toolchain ~/ttmlir-toolchain

# 3. Clone tt-forge-fe
cd ~
git clone https://github.com/tenstorrent/tt-forge-fe.git
cd tt-forge-fe

# 4. Configure to use user directories and Python 3.11
export TTFORGE_TOOLCHAIN_DIR=~/ttforge-toolchain
export TTMLIR_TOOLCHAIN_DIR=~/ttmlir-toolchain
export TTFORGE_PYTHON_VERSION=python3.11

# 5. Initialize environment (sets up paths, uses env vars above)
source env/activate

# 6. Initialize submodules (includes tt-mlir)
git submodule update --init --recursive

# 7. Build the environment (creates venv with python3.11, installs deps - takes 10-20 min)
cmake -B env/build env
cmake --build env/build

# 8. Re-activate to ensure venv is active
source env/activate

# 9. Build TT-Forge-FE
cmake -G Ninja -B build -DCMAKE_CXX_COMPILER=clang++-17 -DCMAKE_C_COMPILER=clang-17
cmake --build build

# 10. Install additional dependencies for our classifier
pip install pillow requests tabulate
```

**Note:** If you don't have clang-17, install it:
```bash
wget https://apt.llvm.org/llvm.sh
chmod u+x llvm.sh
sudo ./llvm.sh 17
```

**Why build from source?**
- ✅ Built against YOUR exact tt-metal version (no symbol mismatches)
- ✅ Can update both repos in sync
- ✅ Better for development and experimentation
- ✅ Most reliable for teaching environments
- ✅ Uses user directories (no permission issues)

**Build time:** 10-20 minutes (one-time cost)

**Note:** We use `~/ttforge-toolchain` and `~/ttmlir-toolchain` instead of `/opt/` to avoid permission issues.

[🔨 Build TT-Forge from Source](command:tenstorrent.buildForgeFromSource)

### **Option B: Wheel Installation (Quick but May Fail)**

If you need quick installation and are willing to troubleshoot version issues:

```bash
# CRITICAL: Clear environment variables first
unset TT_METAL_HOME
unset TT_METAL_VERSION

# Create and activate venv
python3 -m venv ~/tt-forge-venv
source ~/tt-forge-venv/bin/activate

# Install wheels
pip install tt_forge_fe --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install tt_tvm --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install pillow torch torchvision requests tabulate
```

**Note:** Wheels are built against specific tt-metal versions. If you get `ImportError: undefined symbol` errors, use Option A instead.

[🚀 Install TT-Forge (Wheel)](command:tenstorrent.installForge)

---

## Step 2: Test Installation

**Quick sanity check:**

Verify `forge` module loads:

**If you built from source (Option A):**
```bash
cd ~/tt-forge-fe
source env/activate
python3 -c "import forge; print(f'✓ TT-Forge {forge.__version__} loaded successfully\!')"
```

**If you used wheels (Option B):**
```bash
source ~/tt-forge-venv/bin/activate
python3 -c "import forge; print(f'✓ TT-Forge {forge.__version__} loaded successfully\!')"
```

Expected output (version will vary):
```python
✓ TT-Forge 0.4.0.dev20250917 loaded successfully!
```

**If you see `ImportError: undefined symbol` errors:**

This usually means TT-Forge was built against a different version of TT-Metal than what's on your system. TT-Forge is under active development (as of December 2025) and version compatibility is still being stabilized.

**Workaround options:**

**Option 1: Skip version check (recommended for now)**
```bash
# Just verify the package installed
source ~/tt-forge-venv/bin/activate
pip list | grep forge
```

You should see `tt_forge_fe` and `tt_tvm` in the list. The actual test will be whether the classifier script works.

**Option 2: Use Docker (if available)**
Docker images have pre-matched versions:
```bash
docker pull ghcr.io/tenstorrent/tt-forge-fe-slim:latest
```

**Option 3: Build from source**
For bleeding-edge compatibility, build tt-forge-fe from source against your tt-metal installation. See: https://github.com/tenstorrent/tt-forge-fe

**Device detection:**

```bash
tt-smi
```

Should show your N150/N300 device is detected.

[🔍 Test Forge Installation](command:tenstorrent.testForgeInstall)

---

## Step 3: Your First Forge Model - MobileNetV2

We'll create a simple image classifier using **MobileNetV2** - one of the validated models in the tt-forge ecosystem.

**Why MobileNetV2?**
- ✅ **Validated:** Confirmed working in tt-forge-models repository
- ✅ **Lightweight:** 3.5M parameters (efficient for hardware)
- ✅ **Standard architecture:** CNN without exotic operators
- ✅ **Fast compilation:** Simpler graph than larger models
- 1000 ImageNet classes (animals, objects, food, vehicles)

**The classifier will:**
1. Load pre-trained MobileNetV2 from torchvision
2. Compile it for TT hardware with `forge.compile()`
3. Classify any image you provide
4. Show top-5 predictions with confidence scores

[📝 Create Image Classifier Script](command:tenstorrent.createForgeClassifier)

This creates `~/tt-scratchpad/tt-forge-classifier.py` with the complete implementation.

---

## Understanding the Code

Let's examine the key parts of `tt-forge-classifier.py`:

**1. Model Loading (Standard PyTorch):**

```python
import torchvision.models as models

# Load pre-trained MobileNetV2 from torchvision
model = models.mobilenet_v2(pretrained=True)
model.eval()  # Set to inference mode
```

Nothing TT-specific yet! This is standard PyTorch.

**2. Compilation for TT Hardware:**

```python
import forge

# Create sample input for shape inference
sample_input = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width

# Attempt compilation for TT hardware
compiled_model = forge.compile(model, sample_inputs=[sample_input])
```

**What happens during `forge.compile()`?**
- Graph capture: Traces PyTorch operations
- Operator validation: Checks if all ops are supported
- Optimization: Applies fusion, layout transforms where possible
- Lowering: Converts to TTNN operations (TT-Metal layer)
- Device mapping: Allocates tensors, schedules execution
- JIT compilation: Generates device kernels

**Compilation can fail if:**
- Unsupported operators are encountered
- Dynamic shapes or control flow patterns
- Memory constraints exceeded
- Operator combinations not yet validated

**3. Image Preprocessing:**

```python
from torchvision import transforms
from PIL import Image

# Standard ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])

img = Image.open(image_path).convert('RGB')
input_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
```

**4. Inference on TT Hardware:**

```python
# Run inference on TT accelerator
output = compiled_model(input_tensor)

# Get predictions
probabilities = torch.nn.functional.softmax(output[0], dim=0)
top5_prob, top5_idx = torch.topk(probabilities, 5)

for i in range(5):
    class_name = imagenet_classes[top5_idx[i]]
    confidence = top5_prob[i].item() * 100
    print(f"{i+1}. {class_name}: {confidence:.2f}%")
```

---

## Step 4: Run Image Classification

**Test with sample image (cat):**

The script downloads a sample cat image and classifies it:

```bash
cd ~/tt-scratchpad
source ~/tt-forge-venv/bin/activate
python tt-forge-classifier.py
```

**Expected output:**

```text
Loading MobileNetV2 model...
✓ Model loaded

Compiling model for TT hardware...
[Forge compiler output...]
✓ Model compiled successfully!

Processing image: sample-cat.jpg
Running inference on TT hardware...

Top 5 Predictions:
1. Tabby cat: 92.34%
2. Egyptian cat: 4.21%
3. Tiger cat: 2.15%
4. Lynx: 0.89%
5. Persian cat: 0.31%

✓ Classification complete!
```

**First compilation takes time (2-5 minutes):**
- Model graph analysis
- Operator lowering
- Kernel generation
- Device memory allocation

**Subsequent runs may be faster if caching works:**
- Compiled artifacts may be cached
- Re-runs primarily inference execution time

**Note:** Performance characteristics are still being optimized. Expect variability across releases.

[🎨 Run Image Classifier](command:tenstorrent.runForgeClassifier)

---

## Step 5: Classify Your Own Images

**Try your own photos!**

```bash
cd ~/tt-scratchpad
source ~/tt-forge-venv/bin/activate

# Classify your dog
python tt-forge-classifier.py --image ~/Pictures/dog.jpg

# Classify food
python tt-forge-classifier.py --image ~/Pictures/pizza.jpg

# Classify objects
python tt-forge-classifier.py --image ~/Pictures/coffee-mug.jpg
```

**ImageNet classes include:**
- 🐾 Animals: 398 classes (dogs, cats, birds, marine life)
- 🍕 Food: 89 classes (fruits, vegetables, meals)
- 🚗 Vehicles: 47 classes (cars, bikes, aircraft)
- 🏠 Objects: 466 classes (furniture, tools, electronics)

Full list: https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a

---

## Step 6: Try Other Validated Models (Carefully)

**Important:** Not all models will work out-of-the-box. Start with validated examples from tt-forge-models.

**Models likely to work (validated in repository):**

**ResNet variants:**
```python
model = models.resnet50(pretrained=True)
```

**Other MobileNet versions:**
```python
model = models.mobilenet_v3_small(pretrained=True)
```

**Experimenting with other models:**

If you want to try a different model:

1. **Check tt-forge-models first:** https://github.com/tenstorrent/tt-forge-models
2. **Look for similar architectures:** If ResNet50 works, ResNet34 likely will too
3. **Start simple:** Smaller models compile faster and hit fewer edge cases
4. **Expect failures:** Compilation errors are normal; file issues on GitHub
5. **Read release notes:** Recent additions may expand operator support

**Example - trying ResNet (validated):**

```python
# Edit tt-forge-classifier.py
# model = models.mobilenet_v2(pretrained=True)  # Comment out
model = models.resnet50(pretrained=True)  # Validated model

# Re-run compilation
python tt-forge-classifier.py
```

**If compilation fails:**
- Check error message for unsupported operators
- Search GitHub issues for similar problems
- Consider filing a bug report with model details
- Fall back to validated models

---

## What You Just Accomplished 🎉

**You learned:**
- ✅ TT-Forge provides higher-level API than TT-Metal
- ✅ `forge.compile()` attempts automatic optimization for TT hardware
- ✅ **Validated models** like MobileNetV2 work reliably
- ✅ Image classification workflow on TT accelerators
- ✅ How to experiment with similar architectures

**Performance insights:**
- First compilation: 2-5 minutes (one-time cost)
- Inference time varies by model and optimization maturity
- Performance improving with each nightly release
- Check benchmarks in tt-forge-models for validated metrics

**Realistic expectations:**
- Not all PyTorch models will compile (yet)
- Start with validated examples
- Operator coverage expanding with each release
- Community contributions help accelerate support

---

## What's in tt-forge-models? (169 Validated Models)

The `tt-forge-models` repository contains **tested and validated** implementations. These are your safest starting points:

**Computer Vision (Validated):**
- YOLOv3/v5 variants (object detection)
- ResNet family (classification)
- MobileNetV1/V2/V3 (efficient CNNs)
- Some segmentation models

**Language Models (Validated):**
- BERT variants (NLP)
- Some GPT-2 configurations
- Select smaller LLMs

**Other Validated Models:**
- Check the repository for current list: https://github.com/tenstorrent/tt-forge-models
- Each model includes loader and example usage
- Models tested on specific hardware configurations

**Standard interface pattern:**
```python
from tt_forge_models.mobilenet_v2.pytorch import ModelLoader

model = ModelLoader.load_model()
inputs = ModelLoader.load_inputs()
# Then compile with forge.compile()
```

**Repository:** https://github.com/tenstorrent/tt-forge-models

---

## Next Steps: Expand Carefully 🚀

**1. Explore Validated Models**

Browse tt-forge-models and try examples that interest you:
- Object detection (YOLOv5 if validated for your hardware)
- Different ResNet sizes
- Other MobileNet variants

**2. Object Detection (If Supported)**

If YOLOv5 is validated for your device:

```python
from tt_forge_models.yolov5.pytorch import ModelLoader
model = ModelLoader.load_model()
compiled_model = forge.compile(model, sample_inputs=[...])
detections = compiled_model(image_tensor)
```

**Use cases:** Security, inventory, autonomous systems

**3. ONNX Export (Alternative Path)**

For models that don't compile directly, try ONNX:

```python
import torch.onnx

# Export PyTorch to ONNX
torch.onnx.export(model, sample_input, "model.onnx",
                  opset_version=17,
                  input_names=['input'],
                  output_names=['output'])

# Try compiling ONNX version
import forge
compiled_model = forge.compile("model.onnx", ...)
```

ONNX may have different operator support than direct PyTorch ingestion.

**4. Multi-Chip Scaling (TT-XLA)**

For production workloads on N300/T3K/TG:
- TT-XLA is more mature for multi-chip configurations
- Better tested for production deployment
- Consider TT-XLA for serious workloads

**5. Contribute to TT-Forge! 💰**

Help expand operator coverage:
- **Bounty program** pays for model adaptations
- File bug reports with reproducible examples
- Share workarounds that work for you
- Contribute operator implementations

**Repository:** https://github.com/tenstorrent/tt-forge-fe/issues

---

## Debugging Tips

**1. ImportError: undefined symbol (MOST COMMON ISSUE)**

**Error:**
```text
ImportError: /path/to/libTTMLIRRuntime.so: undefined symbol: _ZN4ttnn...
```

**ROOT CAUSE #1: Environment Variable Pollution (90% of cases)**

Before anything else, check if you have conflicting environment variables:

```bash
# Check for problematic variables
echo $TT_METAL_HOME
echo $TT_METAL_VERSION

# If either shows a value, UNSET THEM:
unset TT_METAL_HOME
unset TT_METAL_VERSION

# Try running forge again
cd ~/tt-forge-fe && source env/activate
python3 -c "import forge; print('Success!')"
```

**Why this happens:**
- These variables cause forge to load TT-Metal from outdated system paths
- Even if you built from source, forge ignores your build and loads the wrong version
- See [GitHub issue #529](https://github.com/tenstorrent/tt-forge/issues/529) for details

**Make it permanent:**
```bash
# Add to ~/.bashrc to prevent future issues
echo 'unset TT_METAL_HOME' >> ~/.bashrc
echo 'unset TT_METAL_VERSION' >> ~/.bashrc
```

---

**ROOT CAUSE #2: Version Mismatch (10% of cases)**

If unsetting variables doesn't help, then it's a true version mismatch:

**A. Skip forge import test**
The import check is just verification - the actual model compilation might still work. Skip directly to Step 3 (creating the classifier script) and try running it. Many times the runtime compatibility is better than the import-time checks suggest.

**B. Match TT-Metal version**
Check which TT-Metal version your forge was built against:
```bash
# In tt-forge-venv
pip show tt_forge_fe
```

Look at the release date, then checkout the corresponding tt-metal commit from around that time.

**C. Use Docker (if available)**
Docker images have pre-matched versions:
```bash
docker run -it --rm --device /dev/tenstorrent \
  ghcr.io/tenstorrent/tt-forge-fe-slim:latest
```

**D. Build from source**
Clone tt-forge-fe and build against your exact tt-metal installation:
```bash
git clone https://github.com/tenstorrent/tt-forge-fe.git
cd tt-forge-fe
# Follow build instructions in README
```

**2. Compilation Errors**

If `forge.compile()` fails:

```python
# Enable verbose logging
import os
os.environ['FORGE_LOG_LEVEL'] = 'DEBUG'

compiled_model = forge.compile(model, sample_inputs=[...])
```

Common issues:
- **Unsupported operators:** Check error message, search GitHub issues
- **Shape mismatches:** Verify sample_inputs match model expectations
- **Memory errors:** Try smaller batch sizes or models
- **Data types:** Use float32 tensors

**3. Operator Not Supported**

```yaml
Error: Operator 'some_op' not implemented
```

**What to do:**
1. Search GitHub issues: `site:github.com/tenstorrent/tt-forge-fe "some_op"`
2. Check if recent nightly fixed it (try latest wheel)
3. Try ONNX export (different operator mapping)
4. File detailed bug report with minimal reproduction

**4. Silent Failures / Wrong Results**

If model compiles but gives incorrect results:
- Verify preprocessing matches training (normalization, resize)
- Check for numerical precision issues (compare with CPU inference)
- File bug with model + example showing discrepancy

**5. Python Version Error (JAX 0.7.1 not found)**

**Error:**
```yaml
ERROR: Could not find a version that satisfies the requirement jax==0.7.1
ERROR: Ignored the following versions that require a different python version: ... Requires-Python >=3.11
```

**Cause:** JAX 0.7.1 requires Python >=3.11. The build process needs Python 3.11 specifically.

**Solution:** The build steps now include Python 3.11 installation:
```bash
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev
```

Then set `TTFORGE_PYTHON_VERSION` BEFORE sourcing the activate script:
```bash
cd ~/tt-forge-fe
export TTFORGE_PYTHON_VERSION=python3.11
source env/activate
```

The `env/activate` script defaults to `python3.11`, but explicitly setting it ensures the correct version is used when creating the venv.

**6. Performance Issues**

If inference is unexpectedly slow:
- Verify device detected: `tt-smi`
- Check for fallback to CPU (look for warnings in logs)
- Compare with validated benchmark in tt-forge-models
- Profile with: `FORGE_PROFILE=1 python script.py` (if supported)

---

## Comparison: TT-Forge vs TT-Metal

**TT-Metal (Lessons 1-10):**
```python
# Manual kernel programming
import ttnn

input_t = ttnn.from_torch(input_tensor, device=device)
weight_t = ttnn.from_torch(weights, device=device)
output_t = ttnn.matmul(input_t, weight_t)
result = ttnn.to_torch(output_t)
```

**Pros:** Full control, predictable behavior, well-documented operators
**Cons:** Steep learning curve, manual memory management, more code

**TT-Forge:**
```python
# Automatic compilation (when it works)
import forge

compiled_model = forge.compile(model, sample_inputs=[...])
output = compiled_model(input)
```

**Pros:** Simpler API, higher-level abstraction, faster iteration for supported models
**Cons:** Less mature, operator coverage gaps, harder to debug failures

**When to use which?**
- **TT-Forge:** Experimenting with validated models, rapid prototyping
- **TT-Metal:** Production code needing reliability, custom kernels, unsupported operators
- **vLLM (Lesson 6-7):** Production LLM serving (most mature path)

---

## Key Takeaways

**1. Start with Validated Models**
- ✅ 169 models in tt-forge-models are tested
- ✅ MobileNetV2, ResNet are reliable starting points
- ⚠️ Other models may require debugging or workarounds

**2. TT-Forge is Evolving**
- ✅ Daily nightly builds expand capabilities
- ✅ Active development improves operator coverage
- ⚠️ Expect compilation failures for uncommon architectures
- ✅ Community contributions accelerate progress

**3. Multiple Paths to Production**
- TT-Forge-FE: Single-chip, experimental (as of December 2025)
- TT-XLA: Multi-chip, more mature for PyTorch/JAX
- TT-Metal: Direct kernel programming (most control)
- vLLM: Production LLM serving (proven)

**4. Experimentation is Encouraged**
- Try validated models first
- Document what works (and what doesn't)
- File issues to help the community
- Contribute fixes when possible

**You're now equipped to:**
- Run validated PyTorch models on TT hardware
- Understand TT-Forge compilation workflow
- Troubleshoot common issues
- Experiment with the growing model library

---

## Resources

**Official Documentation:**
- TT-Forge Overview: https://github.com/tenstorrent/tt-forge
- TT-Forge-FE: https://github.com/tenstorrent/tt-forge-fe
- TT-Forge-Models (validated examples): https://github.com/tenstorrent/tt-forge-models
- Releases (latest wheels): https://github.com/tenstorrent/tt-forge-fe/releases
- Issue Tracker: https://github.com/tenstorrent/tt-forge-fe/issues

**Community:**
- Discord: https://discord.gg/tenstorrent
- GitHub Issues: Report bugs, check known issues
- Bounty Program: Contribute and get paid

**Realistic Next Steps:**
- Try other validated models from tt-forge-models
- Experiment with similar architectures to working models
- Document your successes and failures (help the community!)
- Consider TT-XLA for multi-chip or production needs

**Remember:** TT-Forge is a powerful tool under active development (as of December 2025). Start with validated models, be patient with edge cases, and contribute back to accelerate progress! 🚀
