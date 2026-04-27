# TT Developer Toolkit — Extension Guide

**TT Developer Toolkit** (`Tenstorrent.tt-vscode-toolkit`) is a VS Code extension that puts the entire Tenstorrent developer experience inside your editor. It ships interactive lessons that run shell commands for you, a hardware-aware lesson browser, built-in walkthroughs, and code templates — all targeting developers working with Wormhole (N150, N300, T3K) and Blackhole (P100, P150, P300, P300C, QB2) hardware.

---

## Installing from the Marketplace

The easiest way to install is directly from the VS Code Marketplace:

1. Open the Extensions panel (`Ctrl+Shift+X` / `⌘+Shift+X`)
2. Search for **TT Developer Toolkit**
3. Click **Install**

Or use the CLI:

```bash
code --install-extension Tenstorrent.tt-vscode-toolkit
```

→ [Open in VS Code Marketplace](https://marketplace.visualstudio.com/items?itemName=Tenstorrent.tt-vscode-toolkit)

---

## Installing from a GitHub Release

Releases are published at:

**[github.com/tenstorrent/tt-vscode-toolkit/releases](https://github.com/tenstorrent/tt-vscode-toolkit/releases)**

1. Open the **latest release** and download the `.vsix` file (e.g. `tt-vscode-toolkit-<version>.vsix`).

2. Install it in your editor — pick the method that matches your setup:

### VS Code (GUI)

Open the Extensions panel (`Ctrl+Shift+X` / `⌘+Shift+X`), click the **`···`** menu at the top-right of the panel, choose **Install from VSIX…**, and select the downloaded file.

### VS Code (terminal)

```bash
code --install-extension tt-vscode-toolkit-*.vsix
```

### Cursor / Windsurf

The GUI and CLI paths are identical to VS Code — both editors are compatible with `.vsix` packages.

### code-server (GUI)

Open the Extensions panel, click **`···`**, choose **Install from VSIX…**.

### code-server (terminal)

```bash
code-server --install-extension tt-vscode-toolkit-*.vsix
```

### One-liner (GitHub CLI)

If you have the `gh` CLI installed and authenticated:

```bash
gh release download --repo tenstorrent/tt-vscode-toolkit --pattern '*.vsix' \
  && code --install-extension tt-vscode-toolkit-*.vsix
```

---

## First Launch

After installation, the **TT Toolkit** icon appears in the VS Code Activity Bar (left sidebar). Click it to open the **Lessons** panel.

The extension also registers the **"Open Setup Walkthrough"** command (`Ctrl+Shift+P` → `Tenstorrent: Open Setup Walkthrough`), which guides you through initial environment setup step by step.

---

## The Lesson Library

The extension ships **42+ interactive lessons** across eight tracks. Every lesson includes one-click buttons that run the exact shell commands described in the text — no copy-pasting required.

### 🚀 First Inference *(7 lessons, all validated)*

Get from zero to a running inference server.

| Lesson | Hardware |
|---|---|
| Modern Setup with tt-installer 2.0 | N150 · P300C |
| Hardware Detection | N150 · P300C |
| Download Model and Run Inference | N150 · P300C |
| Verify Your Setup | N150 · P300C |
| Interactive Chat with Direct API | N150 |
| HTTP API Server with Direct API | N150 |
| Build tt-metal from Source | N150 · P300C |

### 🏭 Serving Models *(4 lessons)*

Production-grade inference serving.

| Lesson | Hardware |
|---|---|
| Production Inference with tt-inference-server | N150 · P100 |
| Production Inference with vLLM | N150 · P300C |
| Image Generation with Stable Diffusion XL | N150 |
| Video Generation via Frame-by-Frame Diffusion | *(draft)* |

### 🎬 Applications *(4 lessons)*

Real demos you can run end-to-end.

| Lesson | Hardware |
|---|---|
| Coding Assistant with Aider | N150 · N300 |
| Native Video Animation with AnimateDiff | N150 |
| OpenClaw AI Assistant on QuietBox 2 | QB2 |
| Generating Video on QuietBox 2 | QB2 |

### 🍳 Cookbook *(6 lessons, all validated)*

Self-contained programs that showcase Tenstorrent capabilities with vivid visual output — great for demos and first explorations.

| Lesson | What it builds |
|---|---|
| Tenstorrent Cookbook Overview | Index and orientation |
| Recipe 1: Conway's Game of Life | Classic cellular automaton on TT cores |
| Recipe 2: Audio Signal Processing | FFT and mel-spectrogram pipeline |
| Recipe 3: Mandelbrot Fractal Explorer | Interactive fractal renderer |
| Recipe 4: Custom Image Filters | Convolution and edge detection |
| Recipe 5: Particle Life Simulator | Emergent behavior simulation |

### 🔧 Compilers & Tools *(2 lessons, in progress)*

| Lesson | Focus |
|---|---|
| Image Classification with TT-Forge | PyTorch model compilation |
| JAX and PyTorch/XLA on Tenstorrent | XLA/PJRT plugin path |

### 🧠 Advanced *(2 lessons)*

| Lesson | Focus |
|---|---|
| Exploring TT-Metalium | Low-level kernel programming, validated |
| Bounty Program: Model Bring-Up | Open-source contribution workflow |

### 🎓 CS Fundamentals *(7 modules, in development)*

A structured computer-science curriculum designed for the Tenstorrent architecture — covering RISC-V, memory hierarchies, parallel computing, networks, synchronization, abstraction, and computational complexity.

### 🚢 Deployment *(2 lessons, validated)*

| Lesson | Focus |
|---|---|
| Deploy tt-vscode-toolkit to Koyeb | Hosting the extension as a code-server instance |
| Deploy Your Work to Koyeb | Deploying your own TT-accelerated applications |

---

## Hardware Filtering

The **Lessons panel** includes a hardware filter. Select your card (N150, N300, T3K, P100, QB2, etc.) and the list narrows to only the lessons validated on your hardware. The filter state is remembered across sessions.

---

## Built-in Commands

Beyond lessons, the extension registers 70+ commands accessible via the Command Palette (`Ctrl+Shift+P`):

| Category | Examples |
|---|---|
| **Setup** | Run tt-installer Quick Install, Download tt-installer Script |
| **Hardware** | Run Hardware Detection (tt-smi), Reset Device, Clear Device State |
| **Models** | Set Hugging Face Token, Login to HuggingFace, Download Model |
| **Inference** | Start tt-inference-server (N150/N300), Start vLLM Server (T3K/P100) |
| **Compilers** | Activate TT-Forge Environment, Install TT-XLA PJRT Plugin, Test TT-Forge Installation |
| **Metalium** | Build Programming Examples, Run RISC-V Addition Example |
| **Video gen** | Clone tt-local-generator, Start Wan2.2 Video Server, Launch tt-gen GUI |
| **Templates** | Create Coding Assistant Script, Create API Server Script, Create Chat Script |
| **Deployment** | Clone tt-local-generator, Deploy to Koyeb |

Each command opens a terminal and runs the relevant shell invocation — the same command that appears in the lesson buttons.

---

## Keeping Up to Date

The extension does not auto-update when installed from a `.vsix`. To update:

1. Download the latest release from the [releases page](https://github.com/tenstorrent/tt-vscode-toolkit/releases).
2. Re-run the install command — VS Code and code-server replace the existing installation in place.

```bash
# Quick re-install from the repo (requires gh CLI)
gh release download --repo tenstorrent/tt-vscode-toolkit --pattern '*.vsix' --clobber \
  && code --install-extension tt-vscode-toolkit-*.vsix
```

---

## Reporting Issues

File bugs and feature requests at:

**[github.com/tenstorrent/tt-vscode-toolkit/issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues)**
