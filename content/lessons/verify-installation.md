---
id: verify-installation
title: Verify tt-metal Installation
description: >-
  Test your tt-metal installation by running a sample operation on your
  Tenstorrent device.
category: first-inference
tags:
  - installation
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: validated
validatedOn:
  - n150
estimatedMinutes: 5
minTTMetalVersion: v0.65.1
recommended_metal_version: v0.65.1
validationDate: 2026-02-10
validationNotes: UX improvements added - 4 command buttons for easier setup
---

# Verify tt-metal Installation

Test your tt-metal installation by running a sample operation on your Tenstorrent device.

## What This Does

This step runs a simple test program that:
- Initializes the Tenstorrent device
- Performs a basic tensor operation
- Verifies the software stack is properly configured

## Prerequisites: Install System Dependencies

**⚠️ Important:** Before testing tt-metal, ensure all system-level dependencies are installed. This prevents common build and runtime errors.

If you haven't already run this command, do it now:

```bash
cd ~/tt-metal && sudo ./install_dependencies.sh
```

[⚙️ Install System Dependencies](command:tenstorrent.installDependencies)

**What this script does:**
- Installs required system libraries (build tools, kernel modules, etc.)
- Sets up device drivers
- Configures system settings for optimal performance
- Only needs to be run once (or after system updates)

**Time:** ~2-5 minutes depending on your system

**After installation completes**, proceed to setting up your environment below.

---

## Set Up Environment Variables

**Before running verification**, you need to set up environment variables that tell the system where to find tt-metal and its dependencies.

### Required Environment Variables

Add these to your current terminal session:

```bash
# 1. Point to tt-metal installation
export TT_METAL_HOME=~/tt-metal

# 2. Add tt-metal to Python import path
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH

# 3. Add OpenMPI libraries to library path (CRITICAL!)
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

[📋 Copy Environment Setup](command:tenstorrent.copyEnvironmentSetup)

> **💡 Extension Tip:** Use the "Copy Environment Setup" button above to quickly paste these `export` commands into your terminal. The terminal commands in this walkthrough assume these variables are already set.

### Why These Matter

**`TT_METAL_HOME`** - Base directory for tt-metal installation
- Used by model loaders, kernel compilers, and Python imports
- Must match where you cloned/installed tt-metal

**`PYTHONPATH`** - Where Python looks for importable modules
- Allows `import ttnn` and `from models.tt_transformers import ...` to work
- Without this, Python can't find tt-metal modules

**`LD_LIBRARY_PATH`** - Where Linux looks for shared libraries (.so files)
- **CRITICAL:** tt-metal depends on OpenMPI ULFM libraries
- Without this, you'll get: `ImportError: undefined symbol: MPIX_Comm_revoke`
- Even single-chip (N150) operation requires OpenMPI

### Making Changes Permanent (Optional)

To avoid setting these every time, add them to your `~/.bashrc`:

```bash
echo 'export TT_METAL_HOME=~/tt-metal' >> ~/.bashrc
echo 'export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc  # Reload configuration
```

[💾 Add to ~/.bashrc (Permanent)](command:tenstorrent.persistEnvironment)

**Note:** The extension's terminal commands automatically set these variables, but it's good to understand what they do!

---

## Run the Verification

After setting environment variables, run this test operation on your Tenstorrent device:

```bash
python3 -m ttnn.examples.usage.run_op_on_device
```

[✓ Verify TT-Metal Installation](command:tenstorrent.verifyInstallation)

## Expected Output

You should see output indicating successful device initialization and operation completion. This confirms that:
- The tt-metal software is correctly installed
- Your device is properly configured
- You can run programs on the Tenstorrent hardware

## Try More Examples

Once verification succeeds, you can explore more examples:

- **[TT-NN Basic Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html#basic-examples)** - Learn fundamental tensor operations
- **[Simple Kernels on TT-Metalium](https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/index.html)** - Write custom compute kernels

---

## Alternative: Build tt-metal from Source

The instructions above assume you're using a pre-installed tt-metal (via tt-installer or system package). For development work, custom modifications, or accessing the latest features, you can build tt-metal from source.

### When to Build from Source

**✅ Build from source when you need:**
- Latest features from the main branch (not yet in releases)
- To develop custom kernels or operations
- To modify tt-metal internals
- Deep debugging capability
- Access to full source code

**❌ Use pre-installed when you need:**
- Quick setup for inference work
- Production stability
- Automatic updates
- Minimal disk space usage

### Build Process Overview

**Total time:** ~60 minutes
**Disk space:** ~5 GB

### Step 1: Clone Repository (if not done)

```bash
cd ~
git clone --recursive https://github.com/tenstorrent/tt-metal.git
cd tt-metal
```

**Note:** The `--recursive` flag automatically initializes git submodules. If you already cloned without this flag, run:

```bash
git submodule update --init --recursive
```

⚠️ **Critical:** Do not skip submodule initialization. The build will fail with "Missing submodules" error.

### Step 2: Install System Dependencies

```bash
cd ~/tt-metal
sudo ./install_dependencies.sh
```

**Time:** ~5 minutes

This installs:
- Compilers: clang-20, gcc-14, g++-14
- Build tools: cmake (3.28+), ninja-build
- Libraries: boost, yaml-cpp, protobuf, capnproto
- OpenMPI ULFM 5.0.7 (critical for multi-device support)

### Step 3: Build tt-metal

```bash
cd ~/tt-metal
./build_metal.sh
```

**Time:** 30-60 minutes (varies by CPU)

Monitor progress - you'll see compilation steps counting up to ~1008 targets.

**Build configuration:**
- Build type: Release (optimized)
- Compiler: Clang 20 with libstdc++
- Python bindings: nanobind
- Distributed support: ON (multi-device)
- Unity builds: ON (faster compilation)

**Build artifacts created:**
- `~/tt-metal/build_Release/lib/` - Compiled libraries
- `~/tt-metal/build_Release/libexec/` - SFPI compiler toolchain

### Step 4: Create Python Environment

⚠️ **Important:** The Python version must match what the build was configured for. Check the build log for `DPython3_INCLUDE_DIR=/usr/include/pythonX.Y` to see which Python version was used.

**For Python 3.12 (most common on Ubuntu 24.04):**

```bash
cd ~/tt-metal
python3.12 -m venv python_env_3.12
source python_env_3.12/bin/activate
pip install --upgrade pip
pip install -e .
pip install torch  # If you need PyTorch integration
```

**For Python 3.10 (Ubuntu 22.04 or older):**

```bash
cd ~/tt-metal
python3.10 -m venv python_env
source python_env/bin/activate
pip install --upgrade pip
pip install -e .
```

**Time:** ~3 minutes

### Step 5: Set Environment Variables

These environment variables are **required** for the built tt-metal to work:

```bash
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

⚠️ **Critical:** The `LD_LIBRARY_PATH` is mandatory, even for single-device operations. Without it, you'll get:
```
ImportError: undefined symbol: MPIX_Comm_revoke
```

**To make permanent:** Add these to `~/.bashrc`:

```bash
echo 'export TT_METAL_HOME=~/tt-metal' >> ~/.bashrc
echo 'export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

### Step 6: Verify the Build

```bash
cd ~/tt-metal
source "${VENV_NAME}/bin/activate"  # e.g., python_env_3.12 for Python 3.12 or python_env for Python 3.10
python3 -c "import ttnn; print('✓ TTNN imported successfully')"
```

If successful, you should see: `✓ TTNN imported successfully`

### Step 7: Run Validation Test

The extension can generate and run a validation test for you:

[🧪 Generate and Run Validation Test](command:tenstorrent.generateValidationTest)

This will create `~/tt-scratchpad/test_build.py` with the following test:

```python
import ttnn
import torch

print("Checking device availability...")
num_devices = ttnn.GetNumAvailableDevices()
print(f"✓ Found {num_devices} device(s)")

print("Opening device 0...")
device = ttnn.open_device(device_id=0)
print("✓ Device opened")

print("Creating test tensor...")
torch_tensor = torch.randn(1, 1, 32, 32)
tt_tensor = ttnn.from_torch(torch_tensor, dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=device)
print("✓ Tensor created on device")

print("Performing operation...")
result = ttnn.add(tt_tensor, tt_tensor)
print("✓ Operation completed")

ttnn.close_device(device)
print("✓ Device closed - validation successful!")
```

**To create and run manually:**
```bash
# Create the script
cat > ~/tt-scratchpad/test_build.py << 'EOF'
[script content from above]
EOF

# Run it
python ~/tt-scratchpad/test_build.py
```

### Build Troubleshooting

#### Error: "Missing submodules"

**Cause:** Git submodules not initialized.

**Fix:**
```bash
cd ~/tt-metal
git submodule update --init --recursive
```

#### Error: "undefined symbol: PyObject_Vectorcall"

**Cause:** Python version mismatch between build and runtime.

**Symptom:** Build used Python 3.12, but you're running with Python 3.10 (or vice versa).

**Fix:** Use the correct Python version:
```bash
# Check what Python was used in build (look in build log)
# Then create matching venv:
python3.12 -m venv python_env_3.12  # Match the version!
source python_env_3.12/bin/activate
pip install -e .
```

#### Error: "undefined symbol: MPIX_Comm_revoke"

**Cause:** OpenMPI ULFM libraries not in library path.

**Fix:**
```bash
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

Add to `~/.bashrc` to make permanent.

#### Error: "api/debug/assert.h: No such file or directory"

**Cause:** Mixing pre-installed ttnn with built tt-metal in the same environment.

**Fix:** Use a clean Python environment with only the built version:
```bash
deactivate  # Exit any active venv
cd ~/tt-metal
source python_env_3.12/bin/activate  # Use ONLY the built environment
unset PYTHONPATH  # Clear conflicts
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

#### Build fails with compilation errors

**Try a clean rebuild:**
```bash
cd ~/tt-metal
./build_metal.sh --clean      # Remove all build artifacts
sudo ./install_dependencies.sh     # Refresh dependencies
./build_metal.sh              # Rebuild from scratch
```

**Enable ccache for faster rebuilds:**
```bash
./build_metal.sh --enable-ccache
```

### Hardware-Specific Notes

#### QuietBox 2 (2x P300C Blackhole)

**Validated Configuration:**
- OS: Ubuntu 24.04
- Firmware: 19.4.2.0
- Driver: TT-KMD 2.6.0-rc1
- Python: 3.12.3
- Build time: ~45 minutes

**Status:** ✅ Fully validated (2026-01-27)

The manual build process works perfectly on P300C hardware. No hardware-specific build flags needed.

**Performance:** Identical to pre-installed ttnn (no performance penalty for building from source).

### Comparing Build vs. Pre-installed

| Aspect | Pre-installed (tt-installer) | Built from Source |
|--------|------------------------------|-------------------|
| **Setup time** | 0 min (already present) | ~60 min |
| **Disk space** | ~2 GB | ~5 GB |
| **Use case** | Production inference | Development |
| **Updates** | Automatic (tt-installer) | Manual (git pull + rebuild) |
| **Flexibility** | Fixed release | Latest features |
| **Modifications** | Not possible | Full source access |
| **Stability** | Production-tested | Bleeding edge |

### When to Use Which

**Use pre-installed (via tt-installer or system package) for:**
- Running inference workloads
- Production deployments
- Learning and experimentation
- Quick setup and prototyping

**Build from source for:**
- Contributing to tt-metal
- Accessing unreleased features
- Custom kernel development
- Deep debugging needs
- Research and development

**Hybrid approach (advanced users):**
- Keep both installed in separate environments
- Use pre-installed for production
- Use built version for development
- Switch between environments as needed

---

## Troubleshooting

If the verification fails, try these steps:

### Common Error 1: "undefined symbol: MPIX_Comm_revoke"

**Full error:**
```
ImportError: /home/user/tt-metal/build/tt_metal/libtt_metal.so: undefined symbol: MPIX_Comm_revoke
```

**What it means:** OpenMPI libraries are not in the system library path.

**Fix:**
```bash
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

Then retry the verification command. This is the **#1 most common issue** in cloud environments.

### Common Error 2: "cannot import name 'ttnn'"

**What it means:** Python can't find ttnn module.

**Fix:**
```bash
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
source ~/tt-metal/python_env/bin/activate
```

Then retry the verification.

### Common Error 3: "Device not found" or "No devices detected"

**Check device detection:**
```bash
tt-smi -s
```

Make sure your device shows up with no errors. If tt-smi itself fails, the hardware may not be properly connected or drivers not loaded.

### Last Resort: Clean Rebuild

If all else fails, clean and rebuild tt-metal:

```bash
cd ~/tt-metal
./build_metal.sh --clean      # Remove all build artifacts
sudo ./install_dependencies.sh     # Ensure dependencies are current
./build_metal.sh              # Rebuild from scratch
```

**Why `--clean`?** Sometimes old build artifacts can cause issues. The `--clean` flag removes all cached builds and forces a complete rebuild.

**Time:** 5-15 minutes for full rebuild.

## Learn More

For installation troubleshooting and detailed documentation, visit the [tt-metal installation guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
