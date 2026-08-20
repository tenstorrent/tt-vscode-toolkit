---
id: tt-installer
title: Modern Setup with TT-Installer 2.0
description: >-
  The fastest way to get started with Tenstorrent! Use TT-Installer 2.0 for
  one-command installation of the full stack including drivers, firmware,
  TT-Metalium containers, and Python environment.
category: first-inference
tags:
  - installation
  - setup
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
estimatedMinutes: 15
---

# Modern Setup with TT-Installer 2.0

**Welcome to the fastest way to get started with Tenstorrent!**

TT-Installer 2.0 is Tenstorrent's official one-command installation tool that sets up your entire development environment in minutes. It replaces manual installation with an automated, production-tested approach.

## What is TT-Installer 2.0?

TT-Installer 2.0 is a comprehensive installation script that:

- ✅ **Installs the full Tenstorrent stack** - Everything you need in one command
- ✅ **Uses modern containers** - TT-Metalium<sup>™</sup> runs in Podman containers (no complex builds)
- ✅ **Configures your system** - Kernel drivers, HugePages, firmware automatically set up
- ✅ **Provides production tools** - tt-smi, tt-flash, TT-Inference-Server included
- ✅ **Supports multiple hardware** - Works with n150, n300, T3000, Galaxy, p100, p150, p300c, and TT-QuietBox<sup>®</sup> 2
- ✅ **Offers flexibility** - Interactive or non-interactive modes, customizable options

> **⚠️ IMPORTANT: Cloud and Container Environments**
>
> If you're running in a **cloud VM**, **Kubernetes**, **Docker**, or any **containerized environment**:
> - Use `--mode-container` flag or explicitly skip firmware/KMD updates
> - **DO NOT** attempt to update firmware or kernel drivers unless you have full system control
> - Host-level changes (KMD, HugePages, firmware) must be done on bare metal or with explicit cloud provider support
> - See [Container Mode](#container-mode) section below for detailed guidance
>
> **Rule of thumb**: If you're SSH'd into a cloud instance or running inside a container, use `--mode-container` or `--no-install-kmd --no-update-firmware` flags.

## Why Use TT-Installer Instead of Manual Setup?

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

**Modern approach (TT-Installer 2.0):**
- Run one command
- Answer a few prompts
- Get coffee
- **Total time:** 5-15 minutes

## What Gets Installed

TT-Installer 2.0 sets up:

1. **System packages** - Build tools, dependencies (via apt/yum), plus the
   **Tenstorrent package repository** and its signing key
   (see [The Tenstorrent Package Repository](#the-tenstorrent-package-repository-ppa))
2. **Python environment** - Virtual environment with pip/pipx
3. **Kernel-Mode Driver (KMD)** - Tenstorrent hardware driver
4. **Firmware updater (tt-flash)** - Updates your card's firmware to latest
5. **HugePages** - Kernel memory configuration for fast hardware access
6. **System Management Interface (tt-smi)** - Monitor your Tenstorrent devices
7. **Podman** - Container runtime for TT-Metalium
8. **TT-Metalium containers** - Two options:
   - **Standard container** (1GB) - For TT-NN<sup>™</sup> inference and development
   - **Model Demos container** (10GB) - Includes full TT-Metalium build and demos

9. **TT-Inference-Server** - Production inference serving
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
- **Model Demos container** - Install 10GB container with full TT-Metalium build? (yes/no)
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

After installation (and reboot if prompted), confirm the pieces are in place.

**Detecting your hardware** (`tt-smi`) is covered in depth in the next lesson —
see [Hardware Detection](command:tenstorrent.showLesson?["hardware-detection"]).
Once that shows your device, come back and test the container below.

### Test TT-Metalium Container

Run a simple test inside the container:

```bash
tt-metalium "python3 -c 'import ttnn; print(ttnn.__version__)'"
```

This verifies:
- ✅ Container launches successfully
- ✅ TT-NN library is available
- ✅ Python environment is configured

> **⚠️ TT-QuietBox 2 (and other pre-configured images):** this test needs the
> TT-Metalium **container** (Podman + the `tt-metalium` wrapper) that the
> installer sets up. Pre-built QB2 images ship TT-NN and vLLM directly but may
> **not** include Podman or the container wrapper — in that case `tt-metalium`
> won't be found. Verify TT-NN directly instead:
> ```bash
> python3 -c 'import ttnn; print(ttnn.__version__)'
> ```

[🧪 Test TT-Metalium](command:tenstorrent.testMetaliumContainer)

## The Tenstorrent Package Repository (PPA)

Before it can install `tenstorrent-dkms`, `tt-smi`, `tt-flash` and friends, the
installer adds Tenstorrent's package repository at **https://ppa.tenstorrent.com**
to your package manager — and, critically, installs the repository's **signing key**
to `/etc/apt/keyrings/tt-pkg-key.asc`. apt refuses to install from a repository it
cannot verify, so the key is not optional.

TT-Installer does this for you. The commands below are what it runs — useful if you
want to add the repository without running the full installer, or if you need to
repair a broken setup.

### Ubuntu

```bash
# 1. Create the keyring directory
sudo mkdir -p /etc/apt/keyrings
sudo chmod 755 /etc/apt/keyrings

# 2. Download the Tenstorrent package signing key
sudo curl -fsSL -o /etc/apt/keyrings/tt-pkg-key.asc https://ppa.tenstorrent.com/tt-pkg-key.asc
sudo chmod 644 /etc/apt/keyrings/tt-pkg-key.asc
# 3. Add the repository, pinned to that key
echo "deb [signed-by=/etc/apt/keyrings/tt-pkg-key.asc] https://ppa.tenstorrent.com/ubuntu/ $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
  | sudo tee /etc/apt/sources.list.d/tenstorrent.list > /dev/null

# 4. Refresh the package lists
sudo apt-get update
```

### Debian

Identical, except the repository path is `/debian/` instead of `/ubuntu/`:

```bash
sudo mkdir -p /etc/apt/keyrings
sudo chmod 755 /etc/apt/keyrings
sudo curl -fsSL -o /etc/apt/keyrings/tt-pkg-key.asc https://ppa.tenstorrent.com/tt-pkg-key.asc
echo "deb [signed-by=/etc/apt/keyrings/tt-pkg-key.asc] https://ppa.tenstorrent.com/debian/ $(. /etc/os-release && echo "$VERSION_CODENAME") main" \
  | sudo tee /etc/apt/sources.list.d/tenstorrent.list > /dev/null
sudo apt-get update
```

### Fedora / RHEL / CentOS

dnf reads the key straight from the URL, so there is no keyring file to manage:

```bash
sudo tee /etc/yum.repos.d/tenstorrent.repo > /dev/null << 'EOF'
[Tenstorrent]
name=Tenstorrent
baseurl=https://ppa.tenstorrent.com/fedora
enabled=1
gpgcheck=1
gpgkey=https://ppa.tenstorrent.com/tt-pkg-key.asc
EOF
```

> **Note:** RHEL and CentOS are not officially supported — they use the Fedora
> repository, which may or may not match your system's libraries.

### Verify it worked

```bash
# The key should be a PGP public key block
head -1 /etc/apt/keyrings/tt-pkg-key.asc
# → -----BEGIN PGP PUBLIC KEY BLOCK-----

# The repository line should reference that key
cat /etc/apt/sources.list.d/tenstorrent.list

# And packages should now resolve to ppa.tenstorrent.com
apt-cache policy tt-smi
```

### What's in the repository

Confirmed packages on the Ubuntu 24.04 (noble) channel:

| Package | What it is |
|---------|-----------|
| `tenstorrent-dkms` | Kernel-Mode Driver (KMD), built by DKMS on kernel updates |
| `tt-smi` | System Management Interface — device status and reset |
| `tt-flash`, `tt-burnin` | Firmware flashing and burn-in |
| `tt-topology` | Multi-chip topology configuration |
| `tt-metalium`, `tt-metalium-dev`, `tt-metalium-examples`, `tt-metalium-jit` | TT-Metalium<sup>™</sup> runtime and headers |
| `tt-nn`, `tt-nn-dev`, `tt-nn-examples` | TT-NN<sup>™</sup> libraries |
| `sfpi` | Scalar Floating Point Interface (kernel development) |
| `tt-toplike`, `tt-toplike-app` | `top`-style live device monitor |
| `tt-tools-common`, `python3-tt-tools-common`, `python3-pyluwen` | Shared Python tooling |

Install any of them the usual way once the repository is configured:

```bash
sudo apt-get install tt-smi tt-toplike
```

## Using TT-Metalium Containers

TT-Installer provides two ways to use TT-Metalium:

### Interactive Shell

Launch an interactive session:

```bash
tt-metalium
```

This:
- Starts a bash shell inside the container
- Mounts your home directory (access your files)
- Provides full TT-Metalium environment
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
- ✅ TT-NN library for inference
- ✅ Python 3.10+ environment
- ✅ Fast to download and update
- ✅ Best for production inference
- ❌ No model demos included
- ❌ No tt-metal source code

**Model Demos container** (10GB) - `tt-metalium-models`:
- ✅ Full TT-Metalium repository with demos
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

When running inside a container (like Docker) or cloud environment, use container mode:

```bash
./install.sh --mode-container
```

This automatically:
- Skips KMD installation (must be on host)
- Skips HugePages configuration (must be on host)
- Skips Podman installation (no nested containers)
- Skips firmware updates (requires host access)
- Never attempts reboot

#### Cloud Environment Best Practices

**When running in cloud VMs (AWS, GCP, Azure, etc.):**

1. **Bare Metal Instances Only**: Tenstorrent hardware requires PCIe passthrough - only works on bare metal instances, not virtualized VMs
2. **Provider-Managed Drivers**: If your cloud provider pre-installs KMD and firmware, use:
   ```bash
   ./install.sh --no-install-kmd --no-update-firmware --no-install-hugepages
   ```
3. **Container Orchestration (Kubernetes)**: For pods running TT-Metalium:
   ```bash
   ./install.sh --mode-container
   ```
   - KMD must be installed on host nodes
   - HugePages configured via Kubernetes DaemonSet
   - Firmware managed by cluster admins

**When NOT to tamper with firmware/KMD:**
- ❌ Inside Docker/Podman containers
- ❌ Kubernetes pods without privileged access
- ❌ Cloud instances where provider manages hardware
- ❌ Shared infrastructure where you're not the admin
- ❌ Any environment where you can't reboot the host

**Safe operations in restricted environments:**
- ✅ Installing Python packages (tt-smi, TT-Inference-Server)
- ✅ Running TT-Metalium containers (if host has KMD)
- ✅ Using tt-smi to monitor devices
- ✅ Running inference workloads

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
   - Compile models with TT-Forge<sup>™</sup> (Lesson 11)
   - Use JAX with TT-XLA (Lesson 12)

2. **Try Model Demos** (if you installed Model Demos container):
   ```bash
   tt-metalium-models
   cd tt-metal/models/demos
   pytest wormhole/llama31_8b/demo/demo.py
```

3. **Read Official Documentation**:
   - [TT-Metalium Docs](https://docs.tenstorrent.com/tt-metal/latest/)
   - [TT-NN Examples](https://docs.tenstorrent.com/tt-metal/latest/ttnn/ttnn/usage.html)
   - [TT-Installer Wiki](https://github.com/tenstorrent/tt-installer/wiki)

4. **Join the Community**:
   - [Discord](https://discord.gg/tenstorrent)
   - [GitHub Discussions](https://github.com/tenstorrent/tt-metal/discussions)

## Troubleshooting

### apt errors about the Tenstorrent repository signature

**Problem:** `apt-get update` or `apt-get install` fails with one of:

```
E: The repository 'https://ppa.tenstorrent.com/ubuntu noble InRelease' is not signed.
W: GPG error: ... NO_PUBKEY ...
E: Failed to fetch ... 403 Forbidden
N: Updating from such a repository can't be done securely, and is therefore disabled by default.
```

**Cause:** the repository line in `/etc/apt/sources.list.d/tenstorrent.list` points at
`/etc/apt/keyrings/tt-pkg-key.asc`, and that key file is missing, empty, truncated, or
unreadable. This happens when the repository was added by hand without the key, when the
key download was interrupted, or when an older setup used the deprecated
`apt-key add` path.

**Solution:** re-download the key and refresh:

```bash
sudo mkdir -p /etc/apt/keyrings
sudo chmod 755 /etc/apt/keyrings
sudo curl -fsSL -o /etc/apt/keyrings/tt-pkg-key.asc https://ppa.tenstorrent.com/tt-pkg-key.asc

# Sanity-check: must be a PGP block, and world-readable (644)
head -1 /etc/apt/keyrings/tt-pkg-key.asc
ls -l /etc/apt/keyrings/tt-pkg-key.asc

sudo apt-get update
```

**Still failing?** Check these in order:

- **`curl: command not found`** — `sudo apt-get install -y curl` first.
- **Zero-byte key file** — a proxy or captive portal returned an error page.
  `curl -fsSL https://ppa.tenstorrent.com/tt-pkg-key.asc | head -1` should print
  `-----BEGIN PGP PUBLIC KEY BLOCK-----`; if it prints HTML, fix the proxy
  (`HTTPS_PROXY`) and retry. Note `sudo` does not inherit your proxy variables
  by default — use `sudo -E curl ...`.
- **Permissions** — the file must be readable by `_apt`. If you see
  `Could not open file ... Permission denied`, run
  `sudo chmod 644 /etc/apt/keyrings/tt-pkg-key.asc`.
- **Wrong path** — the `signed-by=` path in `tenstorrent.list` must match the file
  you downloaded exactly, including the `.asc` extension. Compare the two:
  `cat /etc/apt/sources.list.d/tenstorrent.list`.
- **Wrong codename** — the repository must reference your release
  (`noble`, `jammy`, …). Confirm with
  `. /etc/os-release && echo "$VERSION_CODENAME"`.
- **Legacy `apt-key`** — if an old trusted key is also present, remove it
  (`sudo apt-key del <keyid>`); the `signed-by=` pin is the supported mechanism.

See [The Tenstorrent Package Repository](#the-tenstorrent-package-repository-ppa)
for the full setup.

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

## Comparison: TT-Installer vs Manual Setup

| Feature | TT-Installer 2.0 | Manual Setup (Old Lessons 1-2) |
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

TT-Installer is recommended for most users, but manual setup may be preferred if:

- ❌ You need bleeding-edge unreleased features (build from main branch)
- ❌ You're developing TT-Metalium itself (need source code access)
- ❌ Your OS is unsupported by TT-Installer
- ❌ You need custom compiler flags or build options
- ❌ You're debugging kernel driver issues (need to build KMD yourself)

**For 95% of users, TT-Installer is the right choice.**

## Frequently Asked Questions

### Q: Can I use TT-Installer inside a Docker container?

**A:** Yes, but use `--mode-container` flag:
```bash
./install.sh --mode-container
```
This skips host-level changes (KMD, firmware, HugePages) that must be done on the host system. The container can still run TT-Metalium and use devices if the host has proper drivers installed.

### Q: I'm running in AWS/GCP/Azure - should I update firmware?

**A:** **No, not unless you're on a bare metal instance and have full control.** Cloud providers may manage firmware and drivers for you. Use:
```bash
./install.sh --no-install-kmd --no-update-firmware
```
Check with your cloud provider's documentation for Tenstorrent support.

### Q: What's the difference between `--mode-container` and `--no-install-kmd`?

**A:**
- `--mode-container` - Skips ALL host-level changes (KMD, firmware, HugePages, reboot). Use inside Docker/Kubernetes.
- `--no-install-kmd` - Skips only KMD installation. Use when KMD is already installed or managed elsewhere.

You can combine flags for fine-grained control:
```bash
./install.sh --no-install-kmd --no-update-firmware --no-install-hugepages
```

### Q: Can I run TT-Metalium without KMD installed?

**A:** No. The Tenstorrent kernel driver must be installed on the **host** system for hardware access. Containers can use devices if the host has KMD loaded.

### Q: How do I know if I'm in a restricted environment?

**A:** If any of these are true, you're likely restricted:
- You can't run `sudo reboot` without approval
- You're inside a Docker/Podman container (`/.dockerenv` file exists)
- You're in a Kubernetes pod (`/var/run/secrets/kubernetes.io` exists)
- You don't have sudo access
- Cloud provider manages your instance's kernel/firmware

Use `--mode-container` or skip flags in these cases.

### Q: What if firmware update fails in my cloud environment?

**A:** Firmware updates require direct hardware access and may not work in cloud environments. Options:
1. Skip firmware updates: `./install.sh --no-update-firmware`
2. Use cloud provider's firmware (if pre-installed)
3. Contact cloud provider support for Tenstorrent firmware management
4. If on bare metal with full control, ensure proper PCIe access

### Q: Can I use TT-Installer in Kubernetes?

**A:** Yes, with proper setup:
- Install KMD and firmware on **host nodes** (outside containers)
- Configure HugePages on host nodes
- Inside pods, use: `./install.sh --mode-container`
- Mount `/dev/tenstorrent` devices into pods
- Use privileged security context if needed

See [Using TT-Metalium Container](https://github.com/tenstorrent/tt-installer/wiki/Using-the-tt%E2%80%90metalium-container) for Kubernetes examples.

## Resources

**Official Documentation:**
- [TT-Installer GitHub](https://github.com/tenstorrent/tt-installer)
- [TT-Installer Wiki](https://github.com/tenstorrent/tt-installer/wiki)
- [Customizing Installation](https://github.com/tenstorrent/tt-installer/wiki/Customizing-your-installation)
- [Using TT-Metalium Container](https://github.com/tenstorrent/tt-installer/wiki/Using-the-tt%E2%80%90metalium-container)

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
