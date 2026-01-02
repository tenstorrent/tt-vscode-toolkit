---
id: verify-installation
title: Verify tt-metal Installation
description: >-
  Test your tt-metal installation by running a sample operation on your
  Tenstorrent device.
category: advanced
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
estimatedMinutes: 5
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
