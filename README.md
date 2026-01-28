# Tenstorrent VSCode Toolkit

**Interactive learning and development tools for Tenstorrent AI accelerators**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![VSCode](https://img.shields.io/badge/VSCode-1.93%2B-blue.svg)](https://code.visualstudio.com/)

Learn Tenstorrent hardware and software through 16 interactive lessons with guided hands-on exercises, production-ready code templates, and intelligent hardware detection. Perfect for developers new to Tenstorrent hardware and teams building production AI inference pipelines.

---

## Overview

The Tenstorrent VSCode Toolkit is an educational extension that provides:

- ✅ **16 Interactive Lessons** - From hardware detection to production deployment, organized by skill level
- ✅ **Click-to-Run Commands** - Execute lessons step-by-step without copy-pasting commands
- ✅ **Hardware Auto-Detection** - Automatically detects your Tenstorrent device and provides tailored guidance
- ✅ **Production Templates** - Real, tested code you can customize for your projects
- ✅ **Multi-Framework Support** - Learn vLLM, TT-Forge, TT-XLA, and TT-Metal
- ✅ **Live Device Monitoring** - Real-time temperature, power, and health status in the status bar

**Target Audience:**
- Developers new to Tenstorrent hardware
- AI engineers deploying models on TT accelerators
- Teams building production inference pipelines
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

### 👋 Welcome & Getting Started
- **Step Zero** - Resources, quickstart, and ecosystem overview
- **FAQ** - Comprehensive troubleshooting guide

### 🚀 First Inference (Lessons 1-5)
- **Hardware Detection** - Detect and verify Tenstorrent devices
- **Verify Installation** - Test tt-metal and TTNN functionality
- **Download Model** - Download and configure LLMs from HuggingFace
- **First Inference** - Run direct API inference with Llama
- **Interactive Chat** - Build a command-line chat application

### 🏭 Production Serving (Lessons 6-9)
- **Production Serving** - Enterprise deployment with tt-inference-server
- **vLLM Production** - High-performance serving with vLLM (OpenAI-compatible)
- **VSCode Chat Integration** - Integrate with VSCode's chat panel
- **Image Generation** - Stable Diffusion 3.5 for image generation

### 🔧 Compilers & Tools (Lessons 10-12)
- **Coding Assistant** - Build an AI-powered code assistant
- **TT-Forge** - PyTorch MLIR compiler (experimental)
- **TT-XLA** - JAX and PyTorch/XLA production compiler

### 🎓 Advanced Topics (Lessons 13-16)
- **RISC-V Programming** - Low-level Tensix core programming
- **Bounty Program** - Model bring-up opportunities ($500-$5000)
- **TT-Metalium Exploration** - Model zoo, tutorials, and programming examples
- **The Cookbook** - 5 hands-on projects (Game of Life, Mandelbrot, audio, images, particle physics)

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

### Latest Release: v0.0.268 (2025-01-09)

**Highlights:**
- 🐳 Docker support with container orchestration
- 📊 Consolidated status bar with sysfs telemetry
- 🎨 Color-coded hardware validation badges
- 🤝 Open-source documentation and GitHub workflows
- 🔧 CS Fundamentals lesson fixes and ttnn API improvements

### Previous Release: v0.0.243 (2024-01-09)

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
