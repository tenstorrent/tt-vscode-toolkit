---
id: forge-image-classification
title: Image Classification with TT-Forge
description: >-
  Compile a PyTorch MobileNetV2 model for Tenstorrent hardware using
  forge.compile() — no build required. The forge env is pre-installed: one
  command to activate, then classify images on real TT silicon.
category: compilers
tags:
  - hardware
  - image
  - forge
  - deployment
  - model
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: draft
validatedOn:
  - p300c
estimatedMinutes: 10
minTTMetalVersion: v0.65.1
recommended_metal_version: v0.65.1
validationDate: 2026-04-15
validationNotes: Rewritten for pre-installed venv-forge; forge.compile() via tt-forge-onnx
---

# Image Classification with TT-Forge

TT-Forge compiles PyTorch models directly for Tenstorrent hardware. The `venv-forge`
environment is **pre-installed** in this developer image — one command to activate,
then `forge.compile()` handles the rest.

> **QB2 users:** Works on all four p300c chips. Each chip is an independent
> Blackhole device; `tt-smi -s` will show four boards.

---

## Activate the forge environment

```bash
source /etc/profile.d/tt-env-forge.sh
```

That's the entire setup. No LLVM build, no Python version juggling, no CMake.

Verify the stack is ready:

```bash
python3 -c "
import forge, jax, torch_xla
print('forge     :', forge.__version__)
print('jax       :', jax.__version__)
print('tt devices:', jax.devices())
"
```

Expected output:

```
forge     : 1.1.0.dev20260415...
jax       : 0.7.1
tt devices: [TtDevice(id=0)]
```

---

## What's in venv-forge

`venv-forge` (Python 3.12) ships the full TT-XLA + Forge stack:

| Package | What it provides |
|---------|-----------------|
| `pjrt_plugin_tt` | PJRT backend — plugs JAX and torch-xla into TT hardware |
| `jax` 0.7.1 | JAX framework (JIT, vmap, pmap, sharding) |
| `torch-xla` 2.9.0 | PyTorch/XLA backend with TT plugin |
| `tt-forge-onnx` | `forge.compile()` — PyTorch/ONNX → TT compiler |

Switch back to the tt-metal or vLLM envs at any time:

```bash
deactivate && source /etc/profile.d/tt-env-metal.sh   # tt-metal / TTNN
deactivate && source /etc/profile.d/tt-env-vllm.sh    # vLLM serving
```

---

## Classify an image with MobileNetV2

MobileNetV2 (3.5M params) compiles cleanly on all TT hardware — a reliable
starting point for understanding the forge workflow.

```python
#!/usr/bin/env python3
"""
MobileNetV2 image classifier compiled for TT hardware via forge.compile().
Activate first: source /etc/profile.d/tt-env-forge.sh
"""
import urllib.request
import forge
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image

# ── 1. Load pre-trained model ──────────────────────────────────────────────
model = models.mobilenet_v2(weights="DEFAULT")
model.eval()

# ── 2. Compile for TT hardware ─────────────────────────────────────────────
# forge.compile() traces the graph, lowers ops to TTNN, generates TT kernels.
# First call: 30–90 s (compilation + kernel codegen). Subsequent calls: fast.
print("Compiling model for TT hardware…")
sample_input = torch.randn(1, 3, 224, 224)
compiled = forge.compile(model, sample_inputs=[sample_input])
print("✓ Compiled")

# ── 3. Preprocess image ────────────────────────────────────────────────────
urllib.request.urlretrieve(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/"
    "RedCat_8727.jpg/320px-RedCat_8727.jpg",
    "/tmp/cat.jpg",
)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
tensor = preprocess(Image.open("/tmp/cat.jpg").convert("RGB")).unsqueeze(0)

# ── 4. Run inference on TT hardware ───────────────────────────────────────
with torch.no_grad():
    output = compiled(tensor)

# ── 5. Decode top-5 predictions ───────────────────────────────────────────
urllib.request.urlretrieve(
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",
    "/tmp/imagenet_classes.txt",
)
labels = [l.strip() for l in open("/tmp/imagenet_classes.txt")]
probs = torch.nn.functional.softmax(output[0], dim=0)
top5_probs, top5_idx = torch.topk(probs, 5)

print("\nTop 5 predictions:")
for i in range(5):
    print(f"  {top5_probs[i]:.1%}  {labels[top5_idx[i]]}")
```

[📝 Create Classifier Script](command:tenstorrent.createForgeClassifier)

Run it:

```bash
source /etc/profile.d/tt-env-forge.sh
python3 ~/tt-scratchpad/forge-classifier.py
```

Expected output:

```
Compiling model for TT hardware…
✓ Compiled

Top 5 predictions:
  91.4%  tabby
   4.2%  Egyptian cat
   2.1%  tiger cat
   0.9%  lynx
   0.4%  Persian cat
```

[▶ Run Forge Classifier](command:tenstorrent.runForgeClassifier)

---

## How forge.compile() works

```
PyTorch model (eval mode)
      │
forge.compile()      ← graph capture, operator validation
      │
MLIR optimizer       ← fusion, layout transforms, op lowering
      │
TTNN operations      ← TT-Metal layer
      │
p300c / N150 / …     ← hardware execution
```

**What compiles reliably** — compilation times measured on QB2 (p300c):

| Architecture | Compile time | Params | Status |
|---|---|---|---|
| AlexNet | **0.9 s** | 61M | ✅ Fastest smoke test |
| SqueezeNet v1.1 | **1.7 s** | 1.2M | ✅ Tiny + fast |
| MobileNet-v3-Small | **2.6 s** | 2.5M | ✅ Mobile-optimised |
| VGG-11 | **2.8 s** | 133M | ✅ Classic |
| MobileNet-v2 | **4.2 s** | 3.5M | ✅ This lesson |
| ResNet-18 | **8.2 s** | 11.7M | ✅ Go-to baseline |
| ResNet-50 | **15.2 s** | 25.6M | ✅ Standard benchmark |
| EfficientNet-B0 | **8.5 s** | 5.3M | ✅ SOTA efficiency |
| ViT-Base-16 | **22.4 s** | 86.6M | ✅ Vision transformer |
| Swin-Tiny | **18.3 s** | 28.3M | ✅ Hierarchical ViT |
| BERT base | ~30 s | 110M | ✅ NLP encoder |
| DenseNet-201 | **116 s** | 20M | ✅ (dense skip-connections) |
| Recent large LLMs (Llama, Mistral) | — | — | Use vLLM instead |

> **Want AlexNet as a faster first compile?** One-line change: `model = tv_models.alexnet(weights="DEFAULT")` — compiles in under a second.

Full list: [tt-forge-models](https://github.com/tenstorrent/tt-forge-models) (169 validated architectures).

> **Bulk compilation testing:** [`tt-forge-compiletron`](https://github.com/tenstorrent/tt-forge-compiletron)
> runs 108 models across all four QB2 chips in parallel and reports per-model compile
> times and success rates. The timing data above comes from that sweep (94.4% success
> rate, 108 models).

---

## Try ResNet-50

One line change — everything else is identical:

```python
model = models.resnet50(weights="DEFAULT")
```

`forge.compile()` and all inference code stays the same. ResNet-50 (25M params)
compiles cleanly on all supported hardware.

---

## Bring your own PyTorch model

```python
import forge, torch

model = YourModel()
model.eval()

sample_input = torch.randn(1, *your_input_shape)
compiled = forge.compile(model, sample_inputs=[sample_input])

# runs on TT hardware
output = compiled(sample_input)
```

If compilation fails, check error output for unsupported operators and search
[tt-forge-fe issues](https://github.com/tenstorrent/tt-forge-fe/issues) for
similar reports. The validated model list is a safe fallback.

---

## What you just ran

```
PyTorch model (torchvision, no changes)
    + forge.compile()
    = inference on Tenstorrent silicon
```

No manual kernel programming, no multi-step build, no environment wrestling.
That's the point of TT-Forge.

---

## Next steps

- [JAX Inference with TT-XLA →](command:tenstorrent.showLesson?["tt-xla-jax"]) — use JAX and pmap for
  multi-device workloads on QB2's four chips
- [vLLM Production →](command:tenstorrent.showLesson?["vllm-production"]) — serving Qwen3 and Llama
  at scale
- [tt-forge-models](https://github.com/tenstorrent/tt-forge-models) — 169 validated
  model implementations ready to run
