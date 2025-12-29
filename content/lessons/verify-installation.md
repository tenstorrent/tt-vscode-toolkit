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

**After installation completes**, proceed to verification below.

---

## Run the Verification

This command will run a test operation on your Tenstorrent device:

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

**1. Clean and rebuild tt-metal:**
```bash
cd ~/tt-metal
./build_metal.sh --clean      # Remove all build artifacts
sudo ./install_dependencies.sh     # Ensure dependencies are current
./build_metal.sh              # Rebuild from scratch
```

**Why `--clean`?** Sometimes old build artifacts can cause issues. The `--clean` flag removes all cached builds and forces a complete rebuild.

**2. Check device detection:**
```bash
tt-smi
```

Make sure your device is detected and shows no errors.

**3. Verify Python environment:**
```bash
python3 -c "import ttnn; print('✓ ttnn import successful')"
```

If imports fail, check that your Python environment has access to tt-metal.

## Learn More

For installation troubleshooting and detailed documentation, visit the [tt-metal installation guide](https://github.com/tenstorrent/tt-metal/blob/main/INSTALLING.md).
