---
id: tt-xla-jax
title: JAX and PyTorch/XLA on Tenstorrent
description: >-
  Run JAX and PyTorch/XLA computations directly on TT hardware — no install
  needed. venv-forge ships pjrt_plugin_tt, JAX 0.7.1, and torch-xla
  pre-installed. Activate, import, and start dispatching tensors to silicon.
category: compilers
tags:
  - production
  - xla
  - jax
  - inference
  - multi-device
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
validationNotes: Rewritten for pre-installed venv-forge; zero install steps
---

# JAX and PyTorch/XLA on Tenstorrent

The `venv-forge` environment ships JAX, torch-xla, and the TT PJRT plugin
pre-installed. There is no installation step — just activate and start computing.

> **QB2 users:** All four p300c chips appear as TT devices (`jax.devices()` returns
> four entries). `pmap` distributes work across them automatically.

---

## Activate the environment

```bash
source /etc/profile.d/tt-env-forge.sh
```

[▶ Activate Forge Environment](command:tenstorrent.activateForgeEnv)

Expected output:

```
TT devices: [TtDevice(id=0)]          # N150 / p300c
# or
TT devices: [TtDevice(id=0), TtDevice(id=1), TtDevice(id=2), TtDevice(id=3)]   # QB2
```

[▶ Check TT Devices](command:tenstorrent.runHardwareDetection)

---

## JAX — 30 seconds to tensor on silicon

JAX dispatches to TT hardware automatically via the PJRT plugin. No device
placement code needed:

```python
import jax
import jax.numpy as jnp

# Create arrays — they live on TT hardware
a = jnp.ones((1024, 1024))
b = jnp.ones((1024, 1024))

# This runs on your TT chip
c = a @ b
print(c.shape)            # (1024, 1024)
print(c.devices())        # {TtDevice(id=0)}
print(c[0, 0])            # 1024.0
```

[▶ Run JAX Quickstart](command:tenstorrent.runJaxQuickstart)

---

## JIT compilation

`@jax.jit` compiles the function into an XLA program the first time it runs,
then caches it. Subsequent calls hit the compiled path:

```python
import jax
import jax.numpy as jnp

@jax.jit
def scaled_matmul(A, B, scale):
    return scale * (A @ B)

A = jnp.ones((256, 256))
B = jnp.ones((256, 256))

# First call: compiles + runs
result = scaled_matmul(A, B, 2.0)

# Subsequent calls: cached compiled kernel, fast
result = scaled_matmul(A, B, 3.0)
print(result[0, 0])       # 768.0
```

---

## Transformer attention on TT hardware

A minimal multi-head self-attention block — the core of every modern LLM:

```python
import jax
import jax.numpy as jnp

def attention(Q, K, V):
    """Scaled dot-product attention."""
    d_k = Q.shape[-1]
    scores = Q @ K.T / jnp.sqrt(d_k)
    weights = jax.nn.softmax(scores, axis=-1)
    return weights @ V

attention_jit = jax.jit(attention)

seq_len, d_model = 64, 128
Q = jnp.ones((seq_len, d_model))
K = jnp.ones((seq_len, d_model))
V = jnp.ones((seq_len, d_model))

out = attention_jit(Q, K, V)
print(out.shape)          # (64, 128)
print(out.devices())      # {TtDevice(id=0)}
```

---

## Multi-device with pmap (QB2 / N300 / T3K)

`jax.pmap` maps a function over a leading batch dimension, one slice per device.
On QB2 with four p300c chips this uses all four in parallel:

```python
import jax
import jax.numpy as jnp

devices = jax.devices()
n = len(devices)
print(f"Running across {n} TT device(s)")

# Replicate computation across all chips
@jax.pmap
def matmul_per_device(A):
    return A @ A.T

# Leading axis = number of devices
A = jnp.ones((n, 512, 512))
result = matmul_per_device(A)

print(result.shape)       # (4, 512, 512) on QB2
print(result.sharding)    # shows per-device placement
```

[▶ Run Multi-Device pmap Demo](command:tenstorrent.runJaxPmapDemo)

---

## PyTorch/XLA — PyTorch models on TT silicon

`torch-xla` is also pre-installed. Use `xm.xla_device()` to get the TT device
and `.to(device)` to place tensors there — standard PyTorch idiom:

```python
import torch
import torch_xla.core.xla_model as xm

device = xm.xla_device()
print(f"TT device: {device}")

# PyTorch tensors on TT hardware
x = torch.randn(256, 256).to(device)
y = torch.randn(256, 256).to(device)

z = x @ y
xm.mark_step()           # flush the XLA graph

print(z.shape)            # torch.Size([256, 256])
print(z.device)           # xla:0
```

[▶ Run PyTorch/XLA Demo](command:tenstorrent.runPytorchXlaDemo)

### PyTorch model inference

```python
import torch
import torch_xla.core.xla_model as xm
import torchvision.models as models

device = xm.xla_device()

# Standard torchvision model — no code changes needed
model = models.mobilenet_v2(weights="DEFAULT").to(device)
model.eval()

x = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    output = model(x)
    xm.mark_step()

print(output.shape)       # torch.Size([1, 1000])
```

> **Note:** torch-xla (without `forge.compile()`) runs models via the XLA JIT path.
> For the full TT-Forge compiler pipeline with MLIR optimization, see the
> [TT-Forge Image Classification lesson](command:tenstorrent.showLesson?["forge-image-classification"]).

---

## Hardware configuration

Wormhole and Blackhole chips are configured identically at the JAX API level.
`jax.devices()` returns one entry per chip, regardless of board type.

| Hardware | `jax.devices()` | Notes |
|----------|-----------------|-------|
| N150 | `[TtDevice(id=0)]` | Single Wormhole chip |
| N300 | `[TtDevice(id=0), TtDevice(id=1)]` | Two Wormhole chips |
| T3K | `[TtDevice(id=0..7)]` | Eight Wormhole chips |
| p300c | `[TtDevice(id=0)]` | Single Blackhole chip |
| QB2 | `[TtDevice(id=0..3)]` | Four independent p300c chips |
| Galaxy | `[TtDevice(id=0..31)]` | 32 Wormhole chips |

Set `TT_METAL_ARCH_NAME` before activating the env if it isn't already set:

```bash
export TT_METAL_ARCH_NAME=blackhole   # p300c / QB2 / P150
export TT_METAL_ARCH_NAME=wormhole_b0 # N150 / N300 / T3K / Galaxy
source /etc/profile.d/tt-env-forge.sh
```

---

## Run the official tt-forge demos

The `tt-forge` repo has validated GPT-2, ALBERT, ResNet, and OPT demos
using JAX/Flax and PyTorch/XLA:

```bash
git clone https://github.com/tenstorrent/tt-forge.git ~/tt-forge
cd ~/tt-forge/demos/tt-xla/nlp/jax

source /etc/profile.d/tt-env-forge.sh
pip install -r requirements.txt

python gpt_demo.py
```

Expected output:

```
Model Variant: GPT2Variant.BASE
Prompt: Gravity Gravity Gravity Gravity Gravity
Next token: ' Gravity' (id: 24532)
Probability: 0.9876
```

Other demos in `~/tt-forge/demos/tt-xla/`:

| Demo | Path | What it runs |
|------|------|-------------|
| GPT-2 | `nlp/jax/gpt_demo.py` | GPT-2 Base/Medium/Large/XL, next-token prediction |
| ALBERT | `nlp/jax/albert_demo.py` | ALBERT classification |
| OPT | `nlp/jax/opt_demo.py` | Meta OPT language model |
| ResNet | `cnn/` | Image classification with JAX/Flax |

[▶ Clone and Run TT-Forge Demos](command:tenstorrent.runTtXlaDemo)

---

## What you just ran

```
venv-forge (pre-installed)
  pjrt_plugin_tt ─── connects JAX/torch-xla to TT hardware
  jax 0.7.1      ─── framework, JIT, pmap
  torch-xla 2.9  ─── PyTorch XLA backend (TT-patched)

One activation command → tensors on silicon.
```

No new venv, no pip install, no Python version change, no library compilation.

---

## Next steps

- [TT-Forge Image Classification →](command:tenstorrent.showLesson?["forge-image-classification"]) — `forge.compile()` for PyTorch
  models via the full MLIR compiler pipeline
- [vLLM Production →](command:tenstorrent.showLesson?["vllm-production"]) — LLM serving (Qwen3, Llama)
- [JAX documentation](https://jax.readthedocs.io/) — comprehensive JAX tutorials
- [tt-forge demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla) — validated JAX and PyTorch/XLA examples
