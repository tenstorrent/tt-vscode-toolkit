# Tenstorrent VSCode Toolkit

**Interactive learning and development tools for Tenstorrent AI accelerators**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![VSCode](https://img.shields.io/badge/VSCode-1.93%2B-blue.svg)](https://code.visualstudio.com/)

Learn Tenstorrent hardware and software through 48 interactive lessons with guided hands-on exercises, production-ready code templates, and intelligent hardware detection. Perfect for developers new to Tenstorrent hardware and teams building production AI inference and **custom training** pipelines.

![Screen capture of tt-vscode-toolkit in action](assets/img/screenshot.png)

---

## Overview

The Tenstorrent VSCode Toolkit is an educational extension that provides:

- ✅ **48 Interactive Lessons** - From hardware detection to custom training, organized by skill level
- ✅ **Custom Training Ready** - Train models from scratch or fine-tune existing ones (validated on hardware!)
- ✅ **Click-to-Run Commands** - Execute lessons step-by-step without copy-pasting commands
- ✅ **Hardware Auto-Detection** - Automatically detects your Tenstorrent device and provides tailored guidance
- ✅ **Production Templates** - Real, tested code you can customize for your projects
- ✅ **Multi-Framework Support** - Learn vLLM, TT-Forge, TT-XLA, and TT-Metal
- ✅ **Live Device Monitoring** - Real-time temperature, power, and health status in the status bar

**Target Audience:**
- Developers new to Tenstorrent hardware
- AI engineers deploying models on TT accelerators
- Teams building production inference pipelines
- ML researchers training custom models
- Contributors to the Tenstorrent ecosystem

---

## Quick Start

### Try in Docker (No Installation)

Run the IDE locally in your browser:

```bash
docker run -d -p 8080:8080 -e PASSWORD=demo \
  ghcr.io/tenstorrent/tt-vscode-toolkit:latest
```

Access at: http://localhost:8080 (password: `demo`)

### Deploy to Cloud with Real Hardware

Get access to Tenstorrent N300 hardware in 60 seconds:

```bash
# Install Koyeb CLI
curl -fsSL https://cli.koyeb.com/install.sh | sh

# Login
koyeb login

# Deploy
koyeb services create vscode \
  --app tt-vscode-toolkit \
  --docker ghcr.io/tenstorrent/tt-vscode-toolkit:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=yourpass \
  --env MESH_DEVICE=N300 \
  --regions na \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

See [Deploy to Koyeb](./content/lessons/deploy-vscode-to-koyeb.md) lesson for details.

---

## Installation

### Prerequisites

**Hardware:**
- Tenstorrent accelerator (N150, N300, T3K, P100, P150, or Galaxy)
- 32GB+ RAM recommended (16GB minimum)
- 100GB+ free disk space for models

**Software:**
- Linux (Ubuntu 20.04+, RHEL 8+, or compatible)
- Python 3.10+ (3.11 for TT-XLA)
- VSCode 1.93+
- tt-metal installed and configured

**Verify your environment:**
```bash
tt-smi                                           # Hardware detected?
python3 --version                                # Python 3.10+?
python3 -c "import ttnn; print('✓ Ready')"       # tt-metal working?
```

### Installation

#### Option 1: Install from VSIX Package (Recommended)

```bash
# Download the latest release
# Install extension
code --install-extension tt-vscode-toolkit-*.vsix

# Restart VSCode
```

#### Option 2: Build from Source

```bash
# Clone repository
git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
cd tt-vscode-toolkit

# Install dependencies
npm install

# Build extension
npm run build

# Package extension
npm run package

# Install
code --install-extension tt-vscode-toolkit-*.vsix
```

### First Steps

1. **Open VSCode** - The extension activates automatically on startup
2. **Open Tenstorrent Sidebar** - Click the Tenstorrent icon in the activity bar
3. **Start Learning** - Begin with "Hardware Detection" lesson or open the Welcome page

**Configuration:** By default, all lessons are visible. To show only validated lessons, disable "Show Unvalidated Lessons" in settings.

---

## Learning Paths

### 🎯 Beginner Path (4-6 hours)
*Perfect for first-time users*

```
1. Hardware Detection      (5 min)  → Verify your hardware setup
2. Verify Installation     (5 min)  → Test tt-metal installation
3. Download Model          (30 min) → Get Llama-3.1-8B or Qwen3-0.6B
4. vLLM Production         (20 min) → Launch production server
```

**What you'll learn:** Hardware detection, environment verification, model downloading, and production inference serving with OpenAI-compatible API.

### 🚀 Intermediate Path (6-8 hours)
*For experienced developers*

```
1. Hardware Detection      → Verify setup
2. vLLM Production        → Production serving
3. Image Generation       → Stable Diffusion on TT hardware
4. TT-Forge               → PyTorch model compilation
5. Coding Assistant       → Build an AI coding tool
```

**What you'll learn:** Production deployment patterns, multi-modal inference (text + images), compiler workflows, and practical AI applications.

### 💡 Advanced Path (10-15 hours)
*For contributors and power users*

```
1. TT-XLA                 → JAX production compiler
2. RISC-V Programming     → Low-level Tensix core programming
3. Bounty Program         → Model bring-up opportunities
4. TT-Metalium Cookbook   → Custom hardware projects
```

**What you'll learn:** Advanced compiler usage, low-level hardware programming, model bring-up workflows, and custom kernel development.

---

## Lesson Catalog

**39 lessons** organized by category. Hardware badges show validated platforms (✅ = tested and working).

<!-- LESSON_CATALOG_START -->
<!-- This section is auto-generated by scripts/generate-lesson-registry.js -->
<!-- Do not edit manually - your changes will be overwritten -->

### 👋 Your journey begins here
*0 lessons, 0 validated*


### 🚀 Your First Inference
*5 lessons, 5 validated*

- **Hardware Detection** — `N150`
- **Verify tt-metal Installation** — `N150`
- **Download Model and Run Inference** — `N150`
- **Interactive Chat with Direct API** — `N150`
- **HTTP API Server with Direct API** — `N150`

### 🏭 Serving Models
*4 lessons, 3 validated*

- **Production Inference with tt-inference-server** — `N150`
- **Production Inference with vLLM** — `N150`
- **Image Generation with Stable Diffusion XL** — `N150`
- **Video Generation via Frame-by-Frame SDXL** — *draft*

### 🔧 Compilers & Tools
*2 lessons, 0 validated*

- **Image Classification with TT-Forge** — *⚠️ blocked*
- **JAX Inference with TT-XLA** — *draft*

### 🎯 Applications
*2 lessons, 1 validated*

- **Coding Assistant with Prompt Engineering** — *draft*
- **Native Video Animation with AnimateDiff** — `N150`

### 🎓 Advanced Topics
*3 lessons, 1 validated*

- **Modern Setup with tt-installer 2.0** — `N150`
- **Bounty Program: Model Bring-Up** — *draft*
- **Exploring TT-Metalium** — *draft*

### 🎓 Custom Training
*8 lessons, 8 validated*

- **Understanding Custom Training** — `N150`
- **Dataset Fundamentals** — `N150`
- **Configuration Patterns** — `N150`
- **Fine-tuning Basics** — `N150`
- **Multi-Device Training** — `N150`
- **Experiment Tracking** — `N150`
- **Model Architecture Basics** — `N150`
- **Training from Scratch** — `N150`

### ☁️ Deployment
*2 lessons, 2 validated*

- **Deploy tt-vscode-toolkit to Koyeb** — `N150`
- **Deploy Your Work to Koyeb** — `N150`

### 👨‍🍳 Tenstorrent Cookbook
*6 lessons, 6 validated*

- **Tenstorrent Cookbook Overview** — `N150` `P300`
- **Recipe 1: Conway's Game of Life** — `N150` `P300`
- **Recipe 2: Audio Signal Processing** — `N150` `P300`
- **Recipe 3: Mandelbrot Fractal Explorer** — `N150` `P300`
- **Recipe 4: Custom Image Filters** — `N150` `P300`
- **Recipe 5: Particle Life Simulator** — `N150` `P300`

### 🧠 CS Fundamentals
*7 lessons, 0 validated*

- **Module 1: RISC-V & Computer Architecture** — *draft*
- **Module 2: The Memory Hierarchy** — *draft*
- **Module 3: Parallel Computing** — *draft*
- **Module 4: Networks and Communication** — *draft*
- **Module 5: Synchronization** — *draft*
- **Module 6: Abstraction Layers** — *draft*
- **Module 7: Computational Complexity in Practice** — *draft*

<!-- LESSON_CATALOG_END -->

---

## Key Features

### Intelligent Hardware Detection
- Auto-detects device type (N150, N300, T3K, P100, P150, Galaxy)
- Provides hardware-specific commands and configurations
- Real-time telemetry monitoring (temperature, power, clock speed)
- Multi-device support with aggregate health status

### Interactive Learning Experience
- Click-to-run commands from lesson content
- Persistent terminal sessions maintain environment state
- Visual progress tracking
- Hierarchical organization by difficulty and category

### Production-Ready Code
- Tested templates for common workflows
- Best practices from Tenstorrent engineering team
- Scripts saved to `~/tt-scratchpad/` for easy customization
- Hardware-specific optimization examples

### Multi-Framework Coverage
| Framework | Purpose | Use Case |
|-----------|---------|----------|
| **vLLM** | Production LLM serving | OpenAI-compatible API, high throughput |
| **TT-Forge** | PyTorch compilation | MLIR-based experimental compiler |
| **TT-XLA** | JAX/PyTorch XLA | Production compiler for JAX workflows |
| **TT-Metal** | Low-level kernels | Custom ops and hardware programming |

### Hands-On Cookbook Projects

The Cookbook (Lesson 16) includes 5 interactive projects that run directly on Tenstorrent hardware:

<table>
<tr>
<td width="50%">
<a href="https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/game_of_life.gif"><img src="assets/img/game_of_life_preview.png" alt="Conway's Game of Life running on Tenstorrent hardware" /></a>
<p align="center"><b>Game of Life</b> - Classic cellular automaton with TT-NN acceleration<br/><sup><a href="https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/game_of_life.gif">View full animation →</a></sup></p>
</td>
<td width="50%">
<a href="https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/particle_life.gif"><img src="assets/img/particle_life_preview.png" alt="Particle Life simulation on Tenstorrent" /></a>
<p align="center"><b>Particle Life</b> - Physics simulation with 10,000+ particles<br/><sup><a href="https://github.com/tenstorrent/tt-vscode-toolkit/blob/main/assets/img/particle_life.gif">View full animation →</a></sup></p>
</td>
</tr>
<tr>
<td width="50%">
<img src="assets/img/mandelbrot.png" alt="Mandelbrot set fractal rendering" />
<p align="center"><b>Mandelbrot Set</b> - Fractal rendering with hardware acceleration</p>
</td>
<td width="50%">
<img src="assets/img/mel_spectrogram.png" alt="Audio mel spectrogram processing" />
<p align="center"><b>Audio Processing</b> - Mel spectrogram computation</p>
</td>
</tr>
</table>

**Plus:** Image filters (blur, sharpen, edge detection) - all with complete source code and interactive tutorials.

---

## Documentation

### User Documentation
- **[FAQ.md](content/pages/FAQ.md)** - Comprehensive troubleshooting (covers 90% of common issues)
- **Lesson Content** - Interactive lessons accessible via the extension
- **[CHANGELOG.md](CHANGELOG.md)** - Version history and release notes

### Developer Documentation
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Development setup, architecture, and contribution guidelines
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture and design principles
- **[docs/TESTING.md](docs/TESTING.md)** - Testing guide (134+ automated tests)
- **[docs/PACKAGING.md](docs/PACKAGING.md)** - Build and distribution workflow

### Community & Support
- **[Tenstorrent Documentation](https://docs.tenstorrent.com)** - Official technical documentation
- **[Discord Community](https://discord.gg/tenstorrent)** - Live discussions and community support
- **[GitHub Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues)** - Bug reports and feature requests
- **[Code of Conduct](CODE_OF_CONDUCT.md)** - Community standards
- **[Security Policy](SECURITY.md)** - Vulnerability reporting

---

## Common Issues

### "No hardware detected"
```bash
tt-smi -r      # Reset and rescan
sudo tt-smi    # Try with elevated permissions
```
*See [FAQ.md](content/pages/FAQ.md) for complete diagnostic steps.*

### "ImportError: undefined symbol" (TT-XLA)
```bash
unset TT_METAL_HOME
unset TT_METAL_VERSION
```
*TT-XLA requires clean environment. See Lesson 12 for details.*

### "vLLM won't start"
```bash
echo $TT_METAL_HOME    # Should be ~/tt-metal
echo $MESH_DEVICE      # Should match your hardware (e.g., N150)
```
*See [FAQ.md](content/pages/FAQ.md) for systematic vLLM debugging.*

For more troubleshooting, check the **[FAQ](content/pages/FAQ.md)** or join **[Discord](https://discord.gg/tenstorrent)**.

---

## Contributing

We welcome contributions! Here's how to get involved:

1. **Report Issues** - Use our [issue templates](.github/ISSUE_TEMPLATE/) for bugs, content issues, feature requests, or new lesson ideas
2. **Improve Content** - Lessons are in `content/lessons/*.md` - submit PRs for corrections or improvements
3. **Add Features** - See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
4. **Validate Lessons** - Test lessons on hardware and update metadata
5. **Join Discussions** - Participate in [Discord](https://discord.gg/tenstorrent) and [GitHub Discussions](https://github.com/tenstorrent/tt-vscode-toolkit/discussions)

**See [CONTRIBUTING.md](CONTRIBUTING.md) for:**
- Development setup instructions
- Architecture and design principles
- Code style and standards
- Testing requirements
- Pull request workflow
- Packaging and distribution

---

## Release Information

### Latest Release: v0.0.268 (2026-01-09)

**Highlights:**
- 🐳 Docker support with container orchestration
- 📊 Consolidated status bar with sysfs telemetry
- 🎨 Color-coded hardware validation badges
- 🤝 Open-source documentation and GitHub workflows
- 🔧 CS Fundamentals lesson fixes and ttnn API improvements

### Previous Release: v0.0.243 (2026-01-09)

**Highlights:**
- 📚 New technical documentation structure (ARCHITECTURE, TESTING, PACKAGING)
- 🎯 README streamlined (50% reduction in size)
- 🎨 Monochrome sidebar icon for better theme compatibility

**See [CHANGELOG.md](CHANGELOG.md) for complete version history.**

### Version Support

| Version | Status | Notes |
|---------|--------|-------|
| 0.1.x | ✅ Current | Active development, full support |
| 0.0.x | ⚠️ Legacy | Community support only |

---

## License

This project is licensed under the **Apache License 2.0** - see the [LICENSE](LICENSE) file for details.

### License Understanding

This software assists in programming Tenstorrent products. However, making, using, or selling hardware, models, or IP may require the license of rights (such as patent rights) from Tenstorrent or others. See [LICENSE_understanding.txt](LICENSE_understanding.txt) for details.

### Third-Party Licenses

This extension depends on several open-source projects:
- **marked** (MIT) - Markdown parsing
- **DOMPurify** (Apache-2.0 or MPL-2.0) - HTML sanitization
- **mermaid** (MIT) - Diagram rendering
- **jsdom** (MIT) - DOM implementation

Run `npm list --prod` to see all production dependencies.

---

## Acknowledgments

Built by the Tenstorrent community with contributions from:
- Tenstorrent Developer Relations and Engineering teams
- Open-source contributors worldwide
- Community members providing feedback and hardware validation

**Special thanks to:**
- Beta testers who validated lessons on real hardware across all device types
- Documentation contributors who improved clarity and caught errors
- Bug reporters who helped us fix issues quickly
- Community members suggesting new lessons and features

---

## Related Projects

**Tenstorrent Ecosystem:**
- **[tt-metal](https://github.com/tenstorrent/tt-metal)** - Core runtime and kernel library
- **[vLLM](https://github.com/tenstorrent/vllm)** - High-performance LLM serving (TT fork)
- **[tt-forge](https://github.com/tenstorrent/tt-forge)** - MLIR-based compiler for PyTorch
- **[tt-xla](https://github.com/tenstorrent/tt-xla)** - XLA compiler plugin for JAX
- **[tt-inference-server](https://github.com/tenstorrent/tt-inference-server)** - Production inference automation

---

**Ready to start building AI on Tenstorrent hardware? Install the extension and open the Welcome page!** 🚀

*Questions? Check the [FAQ](content/pages/FAQ.md) or join our [Discord community](https://discord.gg/tenstorrent)!*
