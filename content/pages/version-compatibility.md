# Version Compatibility Matrix

**Last updated:** August 2026

This guide documents validated combinations of hardware, software versions, and configurations for the Tenstorrent ecosystem. Use this to troubleshoot compatibility issues or plan your development environment.

---

## 🎯 Quick Recommendations by Use Case

### Just Starting Out (Lessons 1-5)
**Hardware:** n150 (Wormhole single-chip)
**TT-Metalium<sup>™</sup>:** Latest from main branch
**Python:** 3.10 (system default on Ubuntu 22.04)
**Model:** Qwen3-0.6B (1.5GB, no HuggingFace token needed)

### Production Inference (vLLM)
**Hardware:** n150/n300/T3000/p100/p150
**Deployment:** TT-Inference-Server Docker image (recommended)
**Alternative:** Native installation of the standalone `vllm-tt-plugin` platform plugin over **upstream `vllm==0.24.0`** (no Tenstorrent fork), pinned per-model against the tt-metal README LLMs table

### Multi-Chip Development (TT-XLA)
**Hardware:** n150/n300/T3000/Galaxy
**Python:** 3.11
**Installation:** Wheel-based (no source build required)

### Experimental Compiler (TT-Forge<sup>™</sup>)
**Hardware:** n150 only (single-chip)
**Python:** 3.11
**Build time:** 45-60 minutes
**Requirements:** clang-17

---

## 🖥️ Hardware Configurations

### Wormhole<sup>™</sup> Architecture

#### n150 (Single Chip)
- **DRAM:** 12GB
- **Tensix Cores:** 80
- **Best for:** Development, small models (<2B parameters)
- **Recommended models:**
  - Qwen3-0.6B (0.6B params, 1.5GB) ✅ **Easiest first run** — but see the note below
  - Gemma 3-1B-IT (1B params, 2GB)
  - Llama-3.1-8B-Instruct (8B params, 16GB) ⚠️ **Tight fit, may exhaust DRAM** — but the
    model tt-metal actually documents for this class of hardware
- **⚠️ Note on Qwen3-0.6B (verified 2026-08-03):** `0.6B` appears nowhere in
  `tt-metal/models/tt_transformers/`. The only Qwen3 with tt-transformers model parameters is
  **Qwen3-32B** (QuietBox/Galaxy class). Qwen3-0.6B loads and runs under vLLM because the
  plugin maps Qwen3 *by architecture* to `TTQwen3ForCausalLM`, but it runs on generic
  defaults and is **not a validated configuration**. Treat it as a smoke test.
- **Multi-chip support:** No

#### n300 (Dual Chip)
- **DRAM:** 24GB (2x 12GB)
- **Tensix Cores:** 160
- **Best for:** Medium models (8B parameters)
- **Recommended models:**
  - Llama-3.1-8B-Instruct (8B params, 16GB) ✅ **Comfortable**
  - Qwen3-8B (8B params)
- **Production validation:** ✅ Validated for vLLM — Qwen 2.5 7B at batch 32 reaches 22.1 T/S/U and 707.2 T/S (tt-metal LLMs table)
- **Multi-chip support:** Yes (2 chips, selected with `export MESH_DEVICE=N300`)

#### T3000 (8 Chips)
- **DRAM:** 96GB (8x 12GB)
- **Tensix Cores:** 640
- **Best for:** Large models (70B+ parameters)
- **Recommended models:**
  - Llama-3.1-70B
  - Large-scale inference workloads
- **Multi-chip support:** Yes (8 chips)

#### TT-QuietBox (Wormhole-based)
- **Architecture:** Wormhole (not Blackhole<sup>®</sup>)
- **Configuration:** Multi-chip Wormhole system
- **Production validation:** ✅ Validated for vLLM — Qwen 2.5 72B reaches 15.4 T/S/U and 492.8 T/S (tt-metal LLMs table)
- **Best for:** Production inference deployments
- **Reference:** [Tenstorrent TT-QuietBox](https://tenstorrent.com/hardware/tt-quietbox)
- **vLLM compatibility:** Fully validated (see the tt-metal README LLMs table for the exact tt-metal release and vLLM commit this was measured with)

### Blackhole Architecture

#### p100 (Single Chip)
- **DRAM:** ~32GB
- **Tensix Cores:** 140 (14x10 grid, 13x10 available for compute)
- **Best for:** Next-generation single-chip performance
- **Recommended models:**
  - Qwen3-0.6B (0.6B params, 1.5GB) ✅ **Works great**
  - Llama-3.1-8B-Instruct (8B params, 16GB) ✅ **Comfortable fit**
- **Enhanced features:**
  - L1 data cache: 1464 KB with 4x16B cachelines (write-through)
  - Enhanced NoC: 64B reads (vs 32B on Wormhole)
  - Ethernet: 14 cores with 512KB L1, 2x RISC-V per core
  - DRAM: 8 banks with programmable RISC-V, 128KB L1 per bank
- **Requirements:** `export TT_METAL_ARCH_NAME=blackhole`
- **Multi-chip support:** Single chip only

#### p150 (Configurable: 1, 2, 4, or 8 chips)
- **DRAM per chip:** ~32GB
- **Total DRAM:**
  - p150 x1: ~32GB (single chip)
  - p150 x2: ~64GB (2 chips)
  - p150 x4: ~128GB (4 chips)
  - p150 x8: ~256GB (8 chips)
- **Tensix Cores per chip:** 140 (14x10 grid, 13x10 available for compute)
- **Best for:** Scalable multi-chip deployments (70B+ models)
- **Recommended models:**
  - p150 x1: Llama-3.1-8B ✅
  - p150 x2: Llama-3.1-8B (fast), medium models ✅
  - p150 x4: Llama-3.1-70B ✅
  - p150 x8: 70B+ models, large-scale inference ✅
- **Enhanced features:** Same as p100
- **Requirements:** `export TT_METAL_ARCH_NAME=blackhole`
- **Multi-chip support:** Yes (2, 4, or 8 chips via mesh topology)

### Galaxy (Multi-Node)
- **Configuration:** Multiple T3000 nodes
- **Best for:** Massive-scale distributed training/inference
- **Multi-chip support:** Yes (distributed)

---

## 📦 Software Stack Versions

### Core Components (Lessons 1-10)

| Component | Version | Python | Installation Method | Notes |
|-----------|---------|--------|-------------------|-------|
| **TT-Metalium** | Latest (main branch) | 3.10 | Source build | Core low-level API |
| **TT-NN<sup>™</sup>** | Bundled with TT-Metalium | 3.10 | Included | High-level neural network ops |
| **OpenMPI ULFM** | 5.0.7 | N/A | System package | Required for all hardware |
| **PyTorch** | 2.x | 3.10 | pip (in venv) | ML framework |
| **Transformers** | Latest | 3.10 | pip (in venv) | HuggingFace models |

**Environment variables required:**
```bash
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
export MESH_DEVICE=N150  # or N300, T3K, P100, P150, P300, P300x2, TG (Galaxy)
```

### vLLM Production Inference (Lesson 7)

| Deployment Method | Hardware | Status | Notes |
|------------------|----------|--------|-------|
| **TT-Inference-Server (Docker)** | n150/n300/T3000/p100/p150 | ✅ **Recommended** | Pre-validated configurations |
| **Native installation** | n150/n300/T3000 | ⚠️ **Advanced** | Version compatibility challenges |

**How Tenstorrent vLLM support is packaged:** Tenstorrent support is a conformant
out-of-tree vLLM **platform plugin** named `vllm-tt-plugin` (importable as
`vllm_tt_plugin`). It registers itself through vLLM's standard entry-point mechanism and
activates automatically whenever `ttnn` is importable, so **no Tenstorrent fork of vLLM is
involved**. The plugin lives in its own repository —
[`github.com/tenstorrent/vllm-tt-plugin`](https://github.com/tenstorrent/vllm-tt-plugin) —
and its installer pins **upstream `vllm==0.24.0`** from PyPI. It is not itself published on
PyPI and Tenstorrent is not yet listed in upstream vLLM's hardware-plugin documentation, so
installation is from that source checkout.

**Superseded path:** the plugin previously shipped inside the `tenstorrent/vllm` fork on its
`dev` branch at `plugins/vllm-tt-plugin/`. **That path is being retired.** If you have an
older `~/tt-vllm` checkout, migrate to the standalone repo.

**Canonical install** (run from the plugin repo root, with a recent tt-metal Python env
already activated):

```bash
git clone https://github.com/tenstorrent/vllm-tt-plugin.git ~/vllm-tt-plugin
cd ~/vllm-tt-plugin
source docs/install-vllm-tt.sh
```

| Component | Version / source | Notes |
|---|---|---|
| **vLLM** | `vllm==0.24.0` from PyPI, built from sdist (`--no-binary vllm`) | Upstream and unpatched. Other versions untested by the plugin. |
| **`vllm-tt-plugin`** | Source checkout, editable install (`uv pip install -e .`) | Not on PyPI |
| **Python** | `>=3.10,<3.14` | Upstream vLLM 0.24.0's range; 3.10.12 is Ubuntu 22.04's default |
| **`VLLM_TARGET_DEVICE`** | `empty` at **build time only** | `tt` is wrong; the `tt` platform is added at runtime by the plugin |
| **torch** | Whatever the active tt-metal env provides (CPU build) | Not pinned by the plugin |
| **`torchaudio`** | **Uninstalled by the installer** | The CUDA wheel is unloadable next to CPU torch, and `transformers>=5.12` imports it if merely installed |
| **numpy** | `>=1.24.4,<2` — pinned by `docs/vllm-overrides.txt` | See below; this is not optional |
| **`opencv-python-headless`** | `==4.11.0.86` — pinned by the same override file | Last release without a numpy 2 floor |

**🚨 The numpy pin is load-bearing.** `ttnn` requires `numpy<2`, while vLLM's
`requirements/common.txt` asks for `opencv-python-headless>=4.13.0`, which requires
`numpy>=2`. Without `--override docs/vllm-overrides.txt` the resolver upgrades numpy to 2.x,
the install *appears* to succeed, and then **`import ttnn` breaks** — which in turn silently
disqualifies the TT platform. opencv is only reached by vLLM's lazy video-IO path, which no
TT-registered model uses, so pinning it back is safe. Pass the same `--override` on any later
`uv pip install` into this environment.

**Extra dependencies the installer does not cover.** The installer assumes a *full* tt-metal
environment. In a thinner venv or container you also need:

| Package(s) | Symptom if missing |
|---|---|
| `pandas`, `seaborn`, `ml_dtypes`, `graphviz`, `networkx` | `ImportError: Encountered an error while initializing the extension` from `ttnn._ttnn` |
| `torchvision` | `Model architectures ['TT…ForCausalLM'] failed to be inspected` |
| `pytest` | the same "failed to be inspected" error (`tt-metal/models/common/utility_functions.py` imports it at module scope) |

Install them with `--override docs/vllm-overrides.txt` so they cannot pull numpy 2.x back in.

**Docker method (validated):**
```bash
# Uses pre-built image with matched versions
# See Lesson 6 for TT-Inference-Server
# See Lesson 7 for manual vLLM setup
```

**Native installation compatibility matrix:**

| Hardware | TT-Metalium | vLLM | Status | Notes |
|----------|----------|------|--------|-------|
| n150 | Latest (main) | Docker image | ✅ Validated | **Use Docker** |
| n150 | Specific commits | Native build | ⚠️ Complex | Requires model_specs_output.json matching |
| n300+ | Latest (main) | Docker image | ✅ Validated | **Use Docker** |

**Known issues with native installation:**
- **numpy silently upgraded to 2.x**, breaking `import ttnn` — always install with
  `--override docs/vllm-overrides.txt` (see above)
- A CUDA `torchaudio` wheel left installed next to CPU torch (the installer uninstalls it;
  don't reinstall it)
- Missing tracy/inspection dependencies producing opaque import errors (see the table above)
- vLLM version mismatches with TT-Metalium changes — the plugin binds to tt-metal model code
- **Recommendation:** Use Docker unless you have specific requirements for native installation

**Version pinning is per-model, not global:** there is no single "correct" tt-metal + vLLM
pair. The LLMs table in the tt-metal README pairs each model and hardware configuration with
the specific tt-metal release and vLLM commit it was measured against. Pick the row that
matches the model you intend to serve:

| Model | Hardware | TT-Metalium release | vLLM commit |
|-------|----------|---------------------|-------------|
| Llama 3.3 70B | Galaxy | `v0.65.0-rc7` | `59be953` |
| Qwen 2.5 7B | n300 | `v0.62.0-rc35` | `ced0161` |
| Qwen 2.5 72B | TT-QuietBox (Wormhole) | `v0.62.0-rc25` | `e7c329b` |

Always read the current table in the tt-metal README before pinning — these pairs advance
with each release.

**Environment variables (vLLM):**
```bash
# Verbose vLLM logging (helpful while bringing a server up)
export VLLM_CONFIGURE_LOGGING=1

# Generous RPC timeout — TT model compilation on first load is slow
export VLLM_RPC_TIMEOUT=900000

# Multi-chip is selected here, NOT with --tensor-parallel-size
export MESH_DEVICE=N300  # see the MESH_DEVICE table in the FAQ for valid values

# REQUIRED when the model argument is a local path. tt-metal's tt_transformers uses
# HF_MODEL as its checkpoint directory (CKPT_DIR and TOKENIZER_PATH both come from it),
# so it must be the same path you pass to `vllm serve`. An HF org/name also works.
export HF_MODEL=~/models/Llama-3.1-8B-Instruct

# For Blackhole (p100/p150):
export TT_METAL_ARCH_NAME=blackhole
```

**Do not export `VLLM_TARGET_DEVICE` at runtime.** It is a **build-time only** variable, and
the value used when building vLLM for Tenstorrent is `empty` (not `tt`), because the TT
platform is supplied by `vllm-tt-plugin` at runtime and detected automatically. Setting it in
your shell has no useful effect.

If you set `VLLM_PLUGINS` at all — it is optional, and vLLM loads every discovered plugin
when it is unset — it must include **both** plugin names, or TT support will only half-load:

```bash
export VLLM_PLUGINS=tt,tt_model_registry
```

### TT-XLA JAX Compiler (Lesson 12)

| Component | Version | Python | Installation Method | Hardware Support |
|-----------|---------|--------|-------------------|------------------|
| **TT-XLA** | Latest wheel | 3.11 | pip (wheel) | n150/n300/T3000/Galaxy |
| **JAX** | 0.7.1+ | 3.11 | pip | Required dependency |
| **TT-Forge** | Cloned for demos | 3.11 | git clone | Demo code only |

**Status:** ✅ **Production-ready for multi-chip**

**Installation:**
```bash
# Python 3.11 required
python3.11 -m venv ~/tt-xla-venv
source ~/tt-xla-venv/bin/activate
pip install pjrt-plugin-tt --pre --upgrade --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

**Environment isolation (CRITICAL):**
```bash
# MUST unset TT-Metalium variables
unset TT_METAL_HOME
unset LD_LIBRARY_PATH
export PYTHONPATH=~/tt-forge:$PYTHONPATH  # For demo imports only
```

**Use the helper script:**
```bash
source ~/tt-scratchpad/setup-tt-xla.sh
```

### TT-Forge MLIR Compiler (Lesson 11)

| Component | Version | Python | Installation Method | Hardware Support |
|-----------|---------|--------|-------------------|------------------|
| **TT-Forge** | Source build | 3.11 | cmake (45-60 min) | n150 only |
| **clang** | 17 | N/A | apt | Required compiler |
| **LLVM** | Built from submodule | N/A | cmake | 6719 targets (~40 min) |
| **JAX** | 0.7.1 | 3.11 | pip | Build dependency |

**Status:** ⚠️ **Experimental (as of December 2025)**

**Build requirements:**
```bash
# Install prerequisites
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev clang-17

# Create compiler symlinks (CRITICAL)
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
```

**Build time:** 45-60 minutes (LLVM compilation is slow)

**Environment setup (CRITICAL):**
```bash
# MUST unset TT-Metalium variables
unset TT_METAL_HOME
unset TT_METAL_VERSION

# MUST use absolute paths (CMake doesn't expand ~)
export TTFORGE_TOOLCHAIN_DIR=/home/$USER/ttforge-toolchain
export TTMLIR_TOOLCHAIN_DIR=/home/$USER/ttmlir-toolchain
export TTFORGE_PYTHON_VERSION=python3.11
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17
```

**Use the helper script:**
```bash
source ~/tt-scratchpad/setup-tt-forge.sh
```

**Model support:** 169 validated models in tt-forge-models repository
- MobileNetV1/V2/V3 (✅ Recommended starting point)
- ResNet variants (✅ Validated)
- Some BERT models (⚠️ Check repository)

### Stable Diffusion 3.5 (Lesson 9)

| Component | Hardware | Status | Notes |
|-----------|----------|--------|-------|
| **SD 3.5 Large** | n150/n300/T3000/p100 | ✅ Validated | 1024x1024 generation |
| **Generation time** | n150 | ~2-3 minutes | First run, includes model load |
| **Environment** | Standard TT-Metalium | ✅ Works | No special setup needed |

**No special version requirements** - uses standard TT-Metalium environment.

---

## 🔧 Environment Variable Reference

### Always Required (Lessons 1-10)

```bash
# Point to TT-Metalium installation
export TT_METAL_HOME=~/tt-metal

# Add TT-Metalium to Python import path
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# Add OpenMPI libraries (CRITICAL - #1 most common error)
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH

# Specify hardware type
export MESH_DEVICE=N150  # or N300, T3K, P100, P150, P300, P300x2, TG (Galaxy)
```

### Hardware-Specific

**For Blackhole chips (p100/p150):**
```bash
export TT_METAL_ARCH_NAME=blackhole
```

### Application-Specific

**vLLM:**
```bash
export VLLM_CONFIGURE_LOGGING=1
export VLLM_RPC_TIMEOUT=900000

# Required whenever `vllm serve` is given a local model directory: tt-metal's
# tt_transformers reads the checkpoint directory from HF_MODEL.
export HF_MODEL=~/models/Llama-3.1-8B-Instruct

# Optional. If set at all, it must list BOTH TT plugin names.
export VLLM_PLUGINS=tt,tt_model_registry
```

`VLLM_TARGET_DEVICE` is **not** a runtime variable — it only affects how upstream vLLM is
built, and the Tenstorrent install uses `VLLM_TARGET_DEVICE=empty` because the TT platform
comes from the `vllm-tt-plugin` package at runtime.

**Stable Diffusion (non-interactive):**
```bash
export NO_PROMPT=1
```

**TT-XLA (isolation required):**
```bash
unset TT_METAL_HOME
unset LD_LIBRARY_PATH
export PYTHONPATH=~/tt-forge:$PYTHONPATH
```

**TT-Forge (isolation required):**
```bash
unset TT_METAL_HOME
unset TT_METAL_VERSION
export TTFORGE_TOOLCHAIN_DIR=/home/$USER/ttforge-toolchain
export TTMLIR_TOOLCHAIN_DIR=/home/$USER/ttmlir-toolchain
export TTFORGE_PYTHON_VERSION=python3.11
export CC=/usr/bin/clang-17
export CXX=/usr/bin/clang++-17
```

---

## 🐛 Common Compatibility Issues

### Issue 1: "undefined symbol: MPIX_Comm_revoke"

**Error:**
```
ImportError: /home/user/tt-metal/build/tt_metal/libtt_metal.so: undefined symbol: MPIX_Comm_revoke
```

**Cause:** OpenMPI library path not set

**Fix:**
```bash
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

**Prevalence:** #1 most common error in cloud environments

**Make permanent:**
```bash
echo 'export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
```

### Issue 2: vLLM Version Mismatch

**Error:**
```
TypeError: block_size has unsupported type list[int]
```

**Cause:** PyTorch/vLLM type hint incompatibility

**Fix:** Use Docker image (validated configuration)
```bash
# See Lesson 6 for TT-Inference-Server Docker setup
# See Lesson 7 for manual Docker approach
```

**Alternative:** Pin a matched pair of commits (advanced). Pinning is **per-model**, not a
single global version — look up the model and hardware you are serving in the tt-metal README
LLMs table and check out the tt-metal release and vLLM commit that row names. See the
per-model pinning table in the vLLM section above for examples.

### Issue 3: TT-Forge Import Failure

**Error:**
```
ImportError: /path/to/libTTMLIRRuntime.so: undefined symbol: _ZN4ttnn...
```

**Cause:** Environment variable pollution (TT_METAL_HOME conflicts)

**Fix:**
```bash
source ~/tt-scratchpad/setup-tt-forge.sh
```

The script automatically unsets conflicting variables.

### Issue 4: CMake Build Errors (TT-Forge)

**Error:**
```
CMake Error: CMAKE_C_COMPILER: clang not found
```

**Cause:** Compiler symlinks not created

**Fix:**
```bash
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
```

### Issue 5: JAX 0.7.1 Not Found

**Error:**
```
ERROR: Could not find a version that satisfies the requirement jax==0.7.1
ERROR: Ignored the following versions that require a different python version: ... Requires-Python >=3.11
```

**Cause:** Python version mismatch (needs 3.11)

**Fix:**
```bash
# Install Python 3.11
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev

# Create venv with 3.11
python3.11 -m venv ~/tt-xla-venv
source ~/tt-xla-venv/bin/activate
```

---

## 📊 Model Size vs Hardware Matrix

### Recommended Model-Hardware Combinations

| Model | Parameters | Disk Size | n150 (12GB) | n300 (24GB) | T3000 (96GB) |
|-------|-----------|-----------|-------------|-------------|------------|
| **Qwen3-0.6B** | 0.6B | 1.5GB | ✅ **Perfect** | ✅ Excellent | ✅ Excellent |
| **Gemma 3-1B-IT** | 1B | 2GB | ✅ **Good** | ✅ Excellent | ✅ Excellent |
| **Llama-3.1-8B** | 8B | 16GB | ⚠️ **Tight** | ✅ **Good** | ✅ Excellent |
| **Qwen3-8B** | 8B | 16GB | ⚠️ **Tight** | ✅ **Good** | ✅ Excellent |
| **Llama-3.1-70B** | 70B | 140GB | ❌ Too large | ❌ Too large | ✅ **Good** |

**Legend:**
- ✅ **Perfect**: Fast, reliable, recommended
- ✅ **Good**: Works well, stable
- ⚠️ **Tight**: May work but can exhaust DRAM under load
- ❌ **Too large**: Model won't fit

**n150 recommendation:** Start with **Qwen3-0.6B** (0.6B parameters, 1.5GB)
- Ultra-lightweight (13x smaller than Llama-3.1-8B)
- No HuggingFace token needed (ungated)
- Dual thinking modes (reasoning-capable)
- Perfect for learning and many production use cases

---

## 🎓 Learning Path Recommendations

### Path 1: Beginner (First Time with Tenstorrent)
1. Hardware: n150
2. Start with: Lessons 1-5 (Direct TT-Metalium API)
3. Model: Qwen3-0.6B
4. Time to first inference: ~30 minutes
5. Environment: Standard TT-Metalium (Python 3.10)

### Path 2: Production Deployment
1. Hardware: n150/n300/T3000 depending on model size
2. Start with: Lessons 1-5 (understand the stack)
3. Then: Lesson 6 (TT-Inference-Server Docker)
4. Model: Match to hardware capacity
5. Environment: Docker (validated configurations)

### Path 3: Model Developer
1. Hardware: n150 (development), scale up for testing
2. Start with: Lessons 1-5 (foundation)
3. Then: Lesson 13 (Bounty Program contribution workflow)
4. Model: Bring your own architecture
5. Environment: Standard TT-Metalium + git workflow

### Path 4: Compiler Explorer
1. Hardware: n150 (single-chip)
2. Start with: Lessons 1-5 (baseline understanding)
3. Then: Lesson 12 (TT-XLA, production-ready)
4. Optional: Lesson 11 (TT-Forge, experimental)
5. Environment: Isolated (separate Python 3.11 venvs)

---

## 🔍 Validation Status by Lesson

| Lesson | Hardware Tested | Status | Notes |
|--------|----------------|--------|-------|
| 1-5 | n150 | ✅ Validated | Zero issues after install_dependencies.sh |
| 6 | n150 | ✅ Validated | TT-Inference-Server Docker |
| 7 | n150 | ⚠️ Docker recommended | Native install has version challenges |
| 8 | n150 | ✅ Validated | VSCode chat integration |
| 9 | n150 | ✅ Validated | Stable Diffusion 3.5, ~2.5 min generation |
| 10 | n150 | ✅ Validated | Coding assistant |
| 11 | n150 | ⚠️ Experimental | TT-Forge 45-60 min build, limited model support |
| 12 | n150 | ✅ Validated | TT-XLA wheel install, GPT-2 XL working |
| 13-14 | n150 | 📋 Documentation | Bounty program, RISC-V programming |
| 15 | n150 | ✅ Validated | TT-Metalium cookbook projects |

---

## 📚 Additional Resources

**Official Documentation:**
- TT-Metalium: https://github.com/tenstorrent/tt-metal
- TT-Inference-Server: https://github.com/tenstorrent/tt-inference-server
- TT-XLA: https://github.com/tenstorrent/tt-xla
- TT-Forge: https://github.com/tenstorrent/tt-forge-fe

**Community:**
- Discord: https://discord.gg/tenstorrent
- Model contributions: Lesson 13 (Bounty Program)

**Troubleshooting:**
- FAQ page (in this extension)
- Step Zero guide (comprehensive tech stack explanation)
- Lesson-specific debugging sections

---

**Remember:** When in doubt, start with the recommended path for your hardware. The most reliable configurations are thoroughly documented in Lessons 1-5, which work on all hardware.
