# Tenstorrent VSCode Toolkit

This extension helps you explore Tenstorrent hardware and software through guided lesson content, helpful commands, and an explorative heart. Working with our RISC-V-based AI accelerators is an exciting journey and we hope you find these tools useful.

Learn Tenstorrent by doing: from detecting hardware to deploying production LLM servers.

**Perfect for:**
- ✅ Developers new to Tenstorrent hardware
- ✅ AI engineers deploying models on TT accelerators
- ✅ Teams building production inference pipelines
- ✅ Contributors to the Tenstorrent ecosystem

Much of the content you'll find here was written with LLM assistance. The sentence you're reading right now was written by me, a member of our developer relations team. We made this toolkit with love for developers.

---

## 🚀 Quick Start

### From Source (Development)

```bash
# Clone and install
git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
cd tt-vscode-toolkit
npm install

# Open in VSCode and launch
code .
# Press F5 to launch Extension Development Host
# The welcome page opens automatically!
```

### Install from Package

```bash
# Build package
npm run package

# Install extension
code --install-extension tt-vscode-toolkit-*.vsix
```

**Note:** By default, only validated lessons are shown. Enable "Show Unvalidated Lessons" in settings to see draft/experimental content during development.

---

## 📚 What You'll Learn

This extension guides you through the Tenstorrent ecosystem with 16 interactive lessons across 6 categories:

- **👋 Welcome** - Resources, FAQ, and community connections
- **🚀 First Inference** - Hardware detection, installation verification, model download, and your first chat application
- **🏭 Production Serving** - Enterprise-ready inference with tt-inference-server and vLLM, plus image generation
- **🔧 Compilers & Tools** - TT-Forge (MLIR) and TT-XLA (JAX) for model compilation
- **🎯 Applications** - Build real-world AI applications like coding assistants
- **🎓 Advanced Topics** - RISC-V programming, bounty program, TT-Metalium exploration, and the cookbook

*Open the extension to explore all lessons with click-to-run commands and interactive guidance.*

---

## 🌟 Key Features

### Interactive Learning
- ✅ Click-to-run commands from lessons
- ✅ Built-in terminal integration with persistent sessions
- ✅ Visual feedback and progress tracking
- ✅ Hierarchical organization by category

### Production-Ready Code
- ✅ Real templates you can customize
- ✅ Best practices from the Tenstorrent team
- ✅ Scripts saved to `~/tt-scratchpad/` for easy access

### Hardware-Aware
- ✅ Auto-detects your device (N150, N300, T3K, P100, P150, Galaxy)
- ✅ Provides hardware-specific guidance and configurations
- ✅ Real-time device monitoring in statusbar

### Multi-Framework Support
- ✅ **vLLM** - Production LLM serving with OpenAI-compatible API
- ✅ **TT-Forge** - PyTorch MLIR compiler (experimental)
- ✅ **TT-XLA** - JAX and PyTorch/XLA support
- ✅ **TT-Metal** - Low-level kernel development

---

## 🎓 Learning Paths

### Beginner Path (First-time users)
```
1. Hardware Detection      (5 min)  → Verify your setup
2. Verify Installation     (5 min)  → Test tt-metal works
3. Download Model          (30 min) → Get Llama-3.1-8B or Qwen3-0.6B
4. vLLM Production         (20 min) → Production server
```

### Intermediate Path (Experienced developers)
```
1. Hardware Detection      (verify only)
2. vLLM Production        (production serving)
3. Image Generation       (Stable Diffusion on TT hardware)
4. TT-Forge               (PyTorch compilation)
```

### Advanced Path (Contributors)
```
1. TT-XLA                 (JAX production compiler)
2. RISC-V Programming     (low-level Tensix core programming)
3. Bounty Program         (model bring-up)
4. TT-Metalium Cookbook   (custom projects)
```

**Total time to complete:** 4-6 hours (depending on download speeds and depth of exploration)

---

## 🛠️ Prerequisites

**Hardware:**
- Tenstorrent accelerator (N150, N300, T3K, P100, P150, or Galaxy)
- 32GB+ RAM recommended (16GB minimum)
- 100GB+ free disk space
- Fast network connection (will download 20-40GB models)

**Software:**
- Linux (Ubuntu 20.04+, RHEL 8+, or compatible)
- Python 3.10+ (3.11 for TT-XLA)
- VSCode 1.93+
- tt-metal installed and working

**Quick Check:**
```bash
tt-smi                                           # Hardware detected?
python3 --version                                # Python 3.10+?
python3 -c "import ttnn; print('✓ Ready')"       # tt-metal working?
```

---

## 📖 Documentation & Resources

### This Repository
- **[FAQ.md](FAQ.md)** - Comprehensive troubleshooting
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Developer guide
- **[docs/](docs/)** - Architecture, testing, and packaging guides
- **Lesson files** - `content/lessons/*.md` (editable by technical writers)

### External Resources
- [Tenstorrent Documentation](https://docs.tenstorrent.com)
- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal)
- [vLLM for TT](https://github.com/tenstorrent/vllm)
- [TT-Forge](https://github.com/tenstorrent/tt-forge)
- [Discord Community](https://discord.gg/tenstorrent)

---

## 📊 What's New

### Version 0.0.243 (Latest)
- 🎨 Updated sidebar icon to monochrome symbol
- ✨ README revision with better organization
- 📚 New docs/ structure for technical details

### Version 0.0.241
- 🗜️ Package size reduction: excluded large .gif files (60MB → 31MB)
- 📦 Better .vscodeignore configuration

### Version 0.0.239
- ✨ Mermaid diagram rendering fully working
- 🐛 Fixed plugin ordering issue with marked.js

### Version 0.0.225
- ✅ Added mermaid validation tests
- 🔧 Fixed stroke property issues in diagrams

### Version 0.0.121
- ✨ Configuration option for showing unvalidated lessons
- 🏷️ Lesson metadata system with validation tracking
- ✅ Comprehensive test suite (134 tests)

*See [CHANGELOG.md](CHANGELOG.md) for full version history.*

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Report Issues** - Use our [GitHub issue tracker](https://github.com/tenstorrent/tt-vscode-toolkit/issues)
2. **Improve Documentation** - Lessons are in `content/lessons/*.md`
3. **Add Features** - See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup
4. **Validate Lessons** - Test on hardware and update metadata
5. **Join the Bounty Program** - See the Bounty Program lesson for opportunities

For technical details on architecture, testing, and packaging, see the [docs/](docs/) directory.

---

## 💬 Community & Support

### Get Help
- **FAQ:** Check [FAQ.md](FAQ.md) first - covers 90% of common issues
- **Discord:** Join the [Tenstorrent community](https://discord.gg/tenstorrent) for live discussions
- **GitHub:** Search [issues](https://github.com/tenstorrent/tt-metal/issues) or file a new one
- **Documentation:** Browse [docs.tenstorrent.com](https://docs.tenstorrent.com)

### Common Issues

**"No hardware detected"**
```bash
tt-smi -r      # Reset and rescan
sudo tt-smi    # Try with elevated permissions
```
*See FAQ.md for full diagnostic steps.*

**"ImportError: undefined symbol"** (TT-XLA)
```bash
unset TT_METAL_HOME
unset TT_METAL_VERSION
```
*See TT-XLA lesson for details.*

**"vLLM won't start"**
```bash
echo $TT_METAL_HOME    # Should be ~/tt-metal
echo $MESH_DEVICE      # Should match your hardware
```
*See FAQ.md for systematic debugging.*

### Connect With Us
- **Discord:** https://discord.gg/tenstorrent (most active)
- **GitHub:** https://github.com/tenstorrent
- **Documentation:** https://docs.tenstorrent.com
- **Twitter:** [@Tenstorrent](https://twitter.com/tenstorrent)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built by the Tenstorrent community with contributions from:
- Tenstorrent engineering team
- Open-source contributors
- Community members providing feedback and testing

**Special thanks to:**
- All beta testers who helped validate lessons on real hardware
- Documentation contributors who improved clarity and accuracy
- Bug reporters who helped us fix issues quickly

---

**Ready to build AI on Tenstorrent hardware? Let's go! 🚀**

*Questions? Check [FAQ.md](FAQ.md) or join our [Discord](https://discord.gg/tenstorrent)!*
