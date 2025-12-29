---
id: tt-installer
title: Modern Setup with tt-installer 2.0
description: >-
  The fastest way to get started with Tenstorrent! Use tt-installer 2.0 for
  one-command installation of the full stack including drivers, firmware,
  tt-metalium containers, and Python environment.
category: advanced
tags:
  - installation
  - setup
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: validated
estimatedMinutes: 15
---

# Modern Setup with tt-installer 2.0

**Welcome to the fastest way to get started with Tenstorrent!**

tt-installer 2.0 is Tenstorrent's official one-command installation tool that sets up your entire development environment in minutes. It replaces manual installation with an automated, production-tested approach.

## What is tt-installer 2.0?

tt-installer 2.0 is a comprehensive installation script that:

- ✅ **Installs the full Tenstorrent stack** - Everything you need in one command
- ✅ **Uses modern containers** - tt-metalium runs in Podman containers (no complex builds)
- ✅ **Configures your system** - Kernel drivers, HugePages, firmware automatically set up
- ✅ **Provides production tools** - tt-smi, tt-flash, tt-inference-server included
- ✅ **Supports multiple hardware** - Works with N150, N300, T3K, and Galaxy systems
- ✅ **Offers flexibility** - Interactive or non-interactive modes, customizable options

## Why Use tt-installer Instead of Manual Setup?

**Traditional approach (manual):**
- Install system packages individually
- Clone and build tt-metal from source (20+ minutes)
- Configure Python environments manually
- Install kernel drivers with DKMS
- Set up HugePages in /etc/sysctl.conf
- Update firmware with tt-flash
- Install tt-smi separately
- Debug dependency issues
- **Total time:** 1-2 hours (if everything goes right)

**Modern approach (tt-installer 2.0):**
- Run one command
- Answer a few prompts
- Get coffee
- **Total time:** 5-15 minutes

## What Gets Installed

tt-installer 2.0 sets up:

1. **System packages** - Build tools, dependencies (via apt/yum)
2. **Python environment** - Virtual environment with pip/pipx
3. **Kernel-Mode Driver (KMD)** - Tenstorrent hardware driver
4. **Firmware updater (tt-flash)** - Updates your card's firmware to latest
5. **HugePages** - Kernel memory configuration for fast hardware access
6. **System Management Interface (tt-smi)** - Monitor your Tenstorrent devices
7. **Podman** - Container runtime for tt-metalium
8. **tt-metalium containers** - Two options:
   - **Standard container** (1GB) - For TT-NN inference and development
   - **Model Demos container** (10GB) - Includes full tt-metal build and demos

9. **tt-inference-server** - Production inference serving
10. **SFPI** - Scalar Floating Point Interface for kernel development

## Quick Start: One-Command Installation

The fastest way to get started:

```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

**What happens:**
1. Downloads the latest installer script
2. Prompts you to choose Python environment strategy
3. Asks if you want the Model Demos container (10GB) or just standard (1GB)
4. Installs everything automatically
5. May ask to reboot (required for kernel driver)

**Safety note:** Always review scripts before running them. You can inspect the installer at:
https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh

[🚀 Run Quick Install](command:tenstorrent.runQuickInstall)

## Step 1: Download and Inspect the Installer

For more control, download the installer first:

```bash
cd ~
curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh -O
chmod +x install.sh
```

Then review it:

```bash
less install.sh
```

[📥 Download Installer](command:tenstorrent.downloadInstaller)

## Step 2: Run Interactive Installation

Run the installer with prompts to customize your setup:

```bash
cd ~
./install.sh
```

**You'll be asked:**
- **Python environment choice** - Use active venv, create new venv, use system Python, or use pipx
- **Model Demos container** - Install 10GB container with full tt-metal build? (yes/no)
- **Reboot** - Reboot now, later, or never (reboot required for kernel driver)

**Recommended choices for most users:**
- Python: `new-venv` (creates `~/.tenstorrent-venv`)
- Model Demos: `no` (can install later if needed)
- Reboot: `yes` (required for kernel driver to load)

[⚙️ Run Interactive Install](command:tenstorrent.runInteractiveInstall)

## Step 3: Non-Interactive Installation (Optional)

For automated deployments or cloud environments, use non-interactive mode:

```bash
./install.sh --mode-non-interactive \
  --python-choice=new-venv \
  --install-metalium-models-container=off \
  --reboot-option=never
```

**Key flags:**
- `--mode-non-interactive` - No prompts, uses defaults or provided flags
- `--python-choice` - Options: `active-venv`, `new-venv`, `system-python`, `pipx`
- `--install-metalium-models-container=on/off` - Install 10GB container with demos
- `--reboot-option` - Options: `ask`, `never`, `always`
- `--no-install-kmd` - Skip kernel driver (useful in containers)
- `--no-install-hugepages` - Skip HugePages config
- `--metalium-image-tag=latest-rc` - Pin specific container version

[🤖 Run Non-Interactive Install](command:tenstorrent.runNonInteractiveInstall)

## Step 4: Verify Installation

After installation (and reboot if prompted), verify everything works:

### Check Hardware Detection

```bash
tt-smi
```

You should see your Tenstorrent device(s) listed with:
- Board Type (N150, N300, T3K, etc.)
- PCI Bus ID
- Firmware version
- Temperature
- Power draw

[🔍 Run tt-smi](command:tenstorrent.runHardwareDetection)

### Test tt-metalium Container

Run a simple test inside the container:

```bash
tt-metalium "python3 -c 'import ttnn; print(ttnn.__version__)'"
```

This verifies:
- ✅ Container launches successfully
- ✅ TTNN library is available
- ✅ Python environment is configured

[🧪 Test tt-metalium](command:tenstorrent.testMetaliumContainer)

## Using tt-metalium Containers

tt-installer provides two ways to use tt-metalium:

### Interactive Shell

Launch an interactive session:

```bash
tt-metalium
```

This:
- Starts a bash shell inside the container
- Mounts your home directory (access your files)
- Provides full tt-metal environment
- Use `exit` to leave the container

### Run Commands Directly

Execute commands without entering the shell:

```bash
# Check TTNN version
tt-metalium "python3 -c 'import ttnn; print(ttnn.__version__)'"

# Run a Python script
tt-metalium "python3 ~/my-inference-script.py"

# Use pytest (for demos)
tt-metalium "pytest models/demos/wormhole/llama31_8b/demo/demo.py"
```

**Key benefit:** Your files in `~` are automatically accessible inside the container!

### Standard vs Model Demos Container

**Standard container** (1GB) - `tt-metalium`:
- ✅ TTNN library for inference
- ✅ Python 3.10+ environment
- ✅ Fast to download and update
- ✅ Best for production inference
- ❌ No model demos included
- ❌ No tt-metal source code

**Model Demos container** (10GB) - `tt-metalium-models`:
- ✅ Full tt-metal repository with demos
- ✅ Pre-compiled examples
- ✅ All model demos (LLMs, vision, audio)
- ✅ Source code for learning
- ❌ Large download (10GB)
- ❌ Slower to update

**Recommendation:**
- Start with standard container (1GB)
- Install Model Demos later if you want to explore examples
- You can have both installed simultaneously

## Advanced Options

### Pin Specific Versions

Install specific versions of components:

```bash
./install.sh \
  --kmd-version=1.34 \
  --fw-version=80.18.3.0 \
  --smi-version=2.0.0 \
  --metalium-image-tag=v0.53.0-rc36
```

### Skip Components

Customize what gets installed:

```bash
./install.sh \
  --no-install-kmd \              # Skip kernel driver (for containers)
  --no-install-hugepages \        # Skip HugePages config
  --no-install-podman \           # Skip Podman (if you have Docker)
  --no-install-metalium-container # Skip container download
```

### Container Mode

When running inside a container (like Docker), use container mode:

```bash
./install.sh --mode-container
```

This automatically:
- Skips KMD installation (must be on host)
- Skips HugePages configuration (must be on host)
- Skips Podman installation (no nested containers)
- Never attempts reboot

### Custom Python Environment

Specify where to create the Python venv:

```bash
./install.sh \
  --python-choice=new-venv \
  --new-venv-location=$HOME/my-custom-venv
```

### Custom Container Image

Use a different container image or registry:

```bash
./install.sh \
  --metalium-image-url=myregistry.example.com/tt-metalium \
  --metalium-image-tag=custom-build-123
```

### Use UV Instead of Pip

For faster Python package installation:

```bash
./install.sh --use-uv=on
```

[uv](https://github.com/astral-sh/uv) is a faster alternative to pip written in Rust.

## Post-Installation: Next Steps

After installation completes, you're ready to:

1. **Explore Lessons 3-12** - This walkthrough teaches you how to:
   - Download and run LLMs (Lesson 3)
   - Build chat interfaces (Lessons 4-5)
   - Deploy production vLLM servers (Lessons 6-7)
   - Generate images with Stable Diffusion (Lesson 8)
   - Create coding assistants (Lesson 9)
   - Use TT-Jukebox for model management (Lesson 10)
   - Compile models with TT-Forge (Lesson 11)
   - Use JAX with TT-XLA (Lesson 12)

2. **Try Model Demos** (if you installed Model Demos container):
   ```bash
   tt-metalium-models
   cd tt-metal/models/demos
   pytest wormhole/llama31_8b/demo/demo.py
```

3. **Read Official Documentation**:
   - [TT-Metalium Docs](https://docs.tenstorrent.com/tt-metal/latest/)
   - [TTNN Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html)
   - [tt-installer Wiki](https://github.com/tenstorrent/tt-installer/wiki)

4. **Join the Community**:
   - [Discord](https://discord.gg/tenstorrent)
   - [GitHub Discussions](https://github.com/tenstorrent/tt-metal/discussions)

## Troubleshooting

### Installation fails with "Permission denied"

**Problem:** Installer requires sudo permissions.

**Solution:** Run with sudo or ensure your user is in sudoers file:
```bash
sudo ./install.sh
```

### "tt-smi: command not found" after installation

**Problem:** Python environment not activated or PATH not updated.

**Solution:**
```bash
# If using pipx (default)
pipx ensurepath
source ~/.bashrc

# If using venv
source ~/.tenstorrent-venv/bin/activate
```

### Kernel driver not loading

**Problem:** Reboot required after KMD installation.

**Solution:**
```bash
sudo reboot
# After reboot, verify:
lsmod | grep tenstorrent
```

### Container fails to start

**Problem:** Podman not configured or device permissions issue.

**Solution:**
```bash
# Check Podman status
podman info

# Check device permissions
ls -l /dev/tenstorrent/

# Re-run installer to fix permissions
./install.sh --no-install-kmd --no-install-hugepages
```

### Firmware update fails

**Problem:** Device not detected or firmware file corrupted.

**Solution:**
```bash
# Verify device detection first
tt-smi

# Force firmware update
./install.sh --update-firmware=force --no-install-kmd
```

### HugePages not configured

**Problem:** Installation skipped HugePages or reboot didn't apply changes.

**Solution:**
```bash
# Check current HugePages
cat /proc/meminfo | grep Huge

# Re-run installer to configure
./install.sh --no-install-kmd --no-install-podman

# Then reboot
sudo reboot
```

### Model Demos container too large

**Problem:** 10GB download too slow or disk space limited.

**Solution:** Use standard container only:
```bash
./install.sh --install-metalium-models-container=off
```

You can always add it later:
```bash
./install.sh --no-install-kmd --no-install-hugepages --install-metalium-models-container=on
```

## Supported Operating Systems

| OS     | Version     | Status | Notes |
|--------|-------------|--------|-------|
| Ubuntu | 24.04 LTS   | ✅ Recommended | Latest Ubuntu LTS |
| Ubuntu | 22.04 LTS   | ✅ Recommended | Most tested, preferred by Tenstorrent |
| Ubuntu | 20.04 LTS   | ⚠️ Deprecated | Support will be removed; Metalium cannot be installed |
| Debian | 12.10.0     | ✅ Supported | Requires curl and rustup for modern Rust |
| Fedora | 41-42       | ✅ Supported | May require restart after base packages |
| Other DEB | Various   | ⚠️ Unsupported | May work but not tested |
| Other RPM | Various   | ⚠️ Unsupported | May work but not tested |

**Recommended:** Ubuntu 22.04.5 LTS for best compatibility.

## Comparison: tt-installer vs Manual Setup

| Feature | tt-installer 2.0 | Manual Setup (Old Lessons 1-2) |
|---------|------------------|--------------------------------|
| **Time** | 5-15 minutes | 1-2 hours |
| **Complexity** | One command | 15+ manual steps |
| **Kernel driver** | ✅ Automatic | ❌ Manual DKMS configuration |
| **Firmware** | ✅ Auto-updated | ❌ Manual tt-flash usage |
| **HugePages** | ✅ Auto-configured | ❌ Manual sysctl.conf editing |
| **tt-metalium** | ✅ Container (1GB) | ❌ Build from source (20+ min) |
| **Python env** | ✅ Auto-created | ❌ Manual venv setup |
| **Updates** | ✅ Re-run installer | ❌ Rebuild everything |
| **Rollback** | ✅ Version pinning | ❌ Complex git operations |
| **Production** | ✅ Ready | ❌ Requires hardening |

## When to Use Manual Setup Instead

tt-installer is recommended for most users, but manual setup may be preferred if:

- ❌ You need bleeding-edge unreleased features (build from main branch)
- ❌ You're developing tt-metal itself (need source code access)
- ❌ Your OS is unsupported by tt-installer
- ❌ You need custom compiler flags or build options
- ❌ You're debugging kernel driver issues (need to build KMD yourself)

**For 95% of users, tt-installer is the right choice.**

## Resources

**Official Documentation:**
- [tt-installer GitHub](https://github.com/tenstorrent/tt-installer)
- [tt-installer Wiki](https://github.com/tenstorrent/tt-installer/wiki)
- [Customizing Installation](https://github.com/tenstorrent/tt-installer/wiki/Customizing-your-installation)
- [Using tt-metalium Container](https://github.com/tenstorrent/tt-installer/wiki/Using-the-tt%E2%80%90metalium-container)

**Community:**
- [Discord](https://discord.gg/tenstorrent)
- [GitHub Issues](https://github.com/tenstorrent/tt-installer/issues)

**Next Steps:**
- Continue to Lesson 1: Hardware Detection (verify installation)
- Skip to Lesson 3: Download Model (start running inference)
- Explore Lessons 6-7: Production vLLM deployment

---

**Congratulations! You now have a complete Tenstorrent development environment.**

The next lessons will teach you how to use this environment to run inference, build applications, and deploy production services.
