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

### Installation

1. **Clone this repository:**
   ```bash
   git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
   cd tt-vscode-toolkit
   ```

2. **Install dependencies:**
   ```bash
   npm install
   ```

3. **Run in development mode:**
   - Open in VSCode
   - Press `F5` to launch Extension Development Host
   - The welcome page opens automatically

### Or Install from .vsix

```bash
npm run package
code --install-extension tt-vscode-toolkit-0.0.121.vsix
```

**Production builds:** By default, only validated lessons are shown. Enable "Show Unvalidated Lessons" in settings to see draft/experimental content during development.

---

## 📚 What You'll Learn

### 👋 **Welcome to Tenstorrent**
Get started with resources and documentation.

- **Welcome Page** - Introduction and quick navigation
- **FAQ** - Comprehensive troubleshooting and answers

### 🚀 **Your First Inference**
Set up hardware and run your first model.

- **Hardware Detection** - Scan for connected Tenstorrent devices
- **Verify Installation** - Test your tt-metal setup works correctly
- **Download Model** - Get Llama-3.1-8B-Instruct from HuggingFace
- **Interactive Chat** - Build a custom chat app using Generator API
- **HTTP API Server** - Create a production-ready Flask API

### 🏭 **Serving Models**
Production deployment and serving infrastructure.

- **tt-inference-server** - Official inference server (simple CLI configuration)
- **vLLM Production** - OpenAI-compatible APIs, continuous batching, enterprise features
- **Image Generation** - Stable Diffusion 3.5 Large for high-resolution images

### 🔧 **Compilers & Tools**
TT-Forge, TT-XLA, and compiler toolchains.

- **TT-Forge Image Classification** - MLIR-based compiler for PyTorch models
- **JAX Inference with TT-XLA** - Production-ready XLA compiler with PJRT integration

### 🎯 **Applications**
Build real-world applications with AI.

- **Coding Assistant** - AI coding assistant with prompt engineering

### 🎓 **Advanced Topics**
Low-level programming, installation, and community contributions.

- **Modern Setup with tt-installer** - One-command installation of the full stack
- **RISC-V Programming** - Program 880 RISC-V cores on a single chip
- **Bounty Program** - Contribute models to the ecosystem and earn rewards
- **Exploring TT-Metalium** - TTNN tutorials, model zoo, Jupyter notebooks
- **TT-Metalium Cookbook** - 4 complete projects (Game of Life, Audio, Fractals, Filters)

---

## 🌟 Key Features

### **Interactive Learning**
- ✅ Click-to-run commands from lessons
- ✅ Built-in terminal integration
- ✅ Step-by-step progression with completion tracking
- ✅ Visual feedback and validation
- ✅ Hierarchical lesson organization by category

### **Production-Ready Code**
- ✅ Real templates you can customize
- ✅ Best practices from the Tenstorrent team
- ✅ Copy-paste ready examples
- ✅ Scripts saved to `~/tt-scratchpad/`

### **Hardware-Aware**
- ✅ Auto-detects your device (N150, N300, T3K, P100, P150, Galaxy)
- ✅ Provides hardware-specific guidance
- ✅ Real-time device monitoring in statusbar
- ✅ Configuration tuned for your device

### **Lesson Validation System**
- ✅ Each lesson has metadata for hardware compatibility
- ✅ Validation status tracking (validated, draft, blocked)
- ✅ Production builds show only validated lessons by default
- ✅ Configurable to show draft/experimental content during development

### **Multi-Framework Support**
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

### Hardware Requirements
- **Tenstorrent accelerator:** N150, N300, T3K (Wormhole) or P100, P150 (Blackhole), or Galaxy
- **RAM:** 32GB+ recommended (16GB minimum)
- **Disk space:** 100GB+ free
- **Network:** Fast connection (will download ~20-40GB models)

### Software Requirements
- **OS:** Linux (Ubuntu 20.04+, RHEL 8+, or compatible)
- **Python:** 3.10+ (3.11 for TT-XLA)
- **Node.js:** v16+ (for extension development)
- **tt-metal:** Installed and working
- **VSCode:** 1.93+

### Quick Check
```bash
# Hardware detected?
tt-smi

# Python version OK?
python3 --version

# tt-metal working?
python3 -c "import ttnn; print('✓ Ready')"
```

---

## 📖 Documentation

- **[FAQ.md](FAQ.md)** - Comprehensive FAQ with troubleshooting
- **[CLAUDE.md](CLAUDE.md)** - Technical implementation details
- **[LESSON_METADATA.md](LESSON_METADATA.md)** - Lesson metadata system documentation
- **Lesson files** - `content/lessons/*.md` (editable by technical writers)

### External Resources
- [Tenstorrent Documentation](https://docs.tenstorrent.com)
- [tt-metal GitHub](https://github.com/tenstorrent/tt-metal)
- [vLLM for TT](https://github.com/tenstorrent/vllm)
- [TT-Forge](https://github.com/tenstorrent/tt-forge)
- [Discord Community](https://discord.gg/tenstorrent)

---

## 🏗️ Architecture

### Extension Structure

```
tt-vscode-extension/
├── content/
│   ├── lessons/          # 16 markdown lesson files
│   ├── templates/        # Python script templates (8 templates)
│   ├── pages/            # Welcome page, FAQ templates
│   └── lesson-registry.json  # Lesson metadata and categories
├── src/
│   ├── commands/         # Terminal command definitions
│   ├── config/           # Model registry and shared config
│   ├── renderers/        # Markdown and command button renderers
│   ├── state/            # Progress tracking and state management
│   ├── types/            # TypeScript types and interfaces
│   ├── utils/            # Lesson registry utilities
│   ├── views/            # Tree view and webview managers
│   ├── webview/          # Webview resources
│   └── extension.ts      # Main extension logic
├── test/
│   └── lesson-tests/     # Automated validation tests
│       ├── markdown-validation.test.ts  # 96 tests for markdown quality
│       ├── templates.test.ts            # Python template validation
│       └── config-extraction.test.ts    # Model registry tests
├── vendor/               # Reference repos (NOT deployed)
│   ├── tt-metal/         # Main reference: demos, APIs, examples
│   ├── vllm/             # Production inference patterns
│   ├── tt-xla/           # JAX/TT-XLA examples
│   ├── tt-forge-fe/      # TT-Forge reference
│   ├── tt-inference-server/  # Production deployment
│   └── tt-installer/     # Installation workflows
├── package.json          # Extension manifest + configuration
├── FAQ.md                # Comprehensive FAQ
└── README.md             # This file
```

### Generated Files (User System)

The extension creates files in your home directory:

```
~/tt-scratchpad/          # All generated scripts
├── tt-chat-direct.py
├── tt-api-server-direct.py
├── tt-forge-classifier.py
├── start-vllm-server.py
└── ...

~/models/                 # Downloaded models
└── Llama-3.1-8B-Instruct/

~/tt-vllm/               # vLLM repository
~/tt-metal/              # TT-Metal repository
```

### Design Principles

- **Content-first:** Lessons are markdown files - easy for technical writers to edit
- **No custom UI:** Uses VSCode's native TreeView and Webview APIs
- **Terminal-integrated:** Commands run in persistent terminals
- **Stateless commands:** Each command can run independently
- **Hardware-aware:** Detects your device and adjusts instructions
- **Validation-aware:** Lessons tagged with hardware compatibility and validation status

---

## 🧪 Testing

The extension includes comprehensive automated tests for quality assurance.

### Running Tests

```bash
# Run all tests
npm test

# Run specific test suites
npm run test:templates
npm run test:markdown
npm run test:config

# Watch mode (re-run on changes)
npm run test:watch
```

### What Gets Tested

✅ **Markdown Validation** (96 tests)
- Code block fence matching (opening/closing)
- Language specifiers on opening fences only
- Empty code blocks detection
- YAML frontmatter validation
- Trailing spaces/tabs
- Lesson structure (headings, content)

✅ **Python Template Validation** (8 tests)
- Python syntax validation for all script templates
- Import statement verification
- Python 3 compatibility checks
- File structure and documentation

✅ **Configuration Extraction** (13 tests)
- Model registry structure validation
- Default model configuration
- Path generation
- No duplication checks

### Test Files

```
test/
├── .mocharc.json          # Mocha configuration
└── lesson-tests/
    ├── markdown-validation.test.ts  # Markdown quality tests
    ├── templates.test.ts            # Template validation suite
    └── config-extraction.test.ts    # Config tests
```

### CI/CD Integration

Tests run automatically on:
- Pull requests
- Commits to main branch
- Pre-publish builds

**Note:** Tests run on any platform (no hardware required). All 134 tests validate content quality, not runtime behavior.

---

## 📦 Packaging and Distribution

### Production Builds

Production builds include only validated lessons by default:

```bash
# Build extension
npm run build

# Package as .vsix
npm run package
```

The `lesson-registry.json` includes validation metadata for each lesson:

```json
{
  "id": "vllm-production",
  "status": "validated",
  "validatedOn": ["n150", "n300", "t3k"],
  "supportedHardware": ["n150", "n300", "t3k", "p100"]
}
```

**Status values:**
- `validated` - Tested and ready for production (shown by default)
- `draft` - In development (hidden by default)
- `blocked` - Known issue (hidden by default)

### Configuration

Users can enable draft/experimental lessons in VSCode settings:

```
Tenstorrent › Show Unvalidated Lessons
Show lessons that haven't been validated on hardware yet.
```

This allows developers and testers to see all content while keeping production builds clean.

---

## 🤝 Contributing

We welcome contributions! Here's how:

### Report Issues
- Use the [GitHub issue tracker](https://github.com/tenstorrent/tt-vscode-toolkit/issues)
- Include hardware type, OS, and error messages
- Check [FAQ.md](FAQ.md) first

### Improve Documentation
- Lessons are in `content/lessons/*.md`
- Follow [Microsoft Writing Style Guide](https://learn.microsoft.com/en-us/style-guide/welcome/)
- Update lesson metadata (supportedHardware, status, validatedOn)

### Add Features
1. Fork the repository
2. Create a feature branch
3. Test thoroughly (`F5` in VSCode, `npm test`)
4. Update documentation
5. Submit a pull request

### Validate Lessons
When testing lessons on hardware:
1. Complete the lesson end-to-end
2. Update `validatedOn` array in `lesson-registry.json`
3. Change status from `draft` to `validated` if fully working
4. Document any hardware-specific issues in lesson content

### Join the Bounty Program
- See the **Bounty Program** lesson for model bring-up opportunities
- Contribute to the ecosystem
- Earn rewards

---

## 🐛 Troubleshooting

### Common Issues

**"No hardware detected"**
```bash
# Try:
tt-smi -r
sudo tt-smi
# See FAQ.md for full diagnostic steps
```

**"ImportError: undefined symbol"**
```bash
# Fix environment pollution:
unset TT_METAL_HOME
unset TT_METAL_VERSION
# See TT-XLA lesson for details
```

**"vLLM won't start"**
```bash
# Check environment:
echo $TT_METAL_HOME    # Should be ~/tt-metal
echo $MESH_DEVICE      # Should match hardware
# See FAQ.md for systematic debugging
```

**Need more help?**
- Check [FAQ.md](FAQ.md) - covers 90% of issues
- Join [Discord](https://discord.gg/tenstorrent)
- Search [GitHub issues](https://github.com/tenstorrent/tt-metal/issues)

---

## 📊 What's New

### Version 0.0.121 (Current)
- ✨ Added configuration option for showing unvalidated lessons (defaults to false)
- ✨ Reorganized lessons into 6 clear categories
- ✅ Comprehensive test suite (134 tests) for markdown, templates, and config
- 📖 Updated README with accurate project structure and validation documentation
- 🏷️ Lesson metadata system with validation tracking
- 🔧 Tree view now respects configuration and filters lessons by validation status

### Version 0.0.86-0.0.99
- ✨ Lesson metadata system with hardware compatibility tracking
- ✨ vLLM starter script with smart defaults and hardware auto-detection
- ✨ Qwen3-0.6B focus for N150 (ultra-lightweight, reasoning-capable)
- 🔧 Fixed install_dependencies.sh commands across all lessons

### Version 0.0.79
- 📖 Added FAQ command - accessible from welcome page, command menu, and Command Palette
- ✅ Fixed all Jukebox references (now standalone tool)
- 🎯 Made TT-XLA lesson visible on welcome page
- 📝 Updated lesson numbering throughout

### Version 0.0.65-0.78
- ✨ Two-terminal strategy implementation
- ✨ Auto-configured UX (theme, terminal, extensions)
- ✨ Statusbar device monitoring
- 📚 Major documentation updates
- ✨ TT-XLA lesson (JAX support)
- 🔧 Improved Forge lesson with environment variable fixes

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 💬 Community

- **Discord:** https://discord.gg/tenstorrent (most active)
- **GitHub:** https://github.com/tenstorrent
- **Documentation:** https://docs.tenstorrent.com
- **Twitter:** [@Tenstorrent](https://twitter.com/tenstorrent)

---

## 🙏 Acknowledgments

Built by the Tenstorrent community with contributions from:
- Tenstorrent engineering team
- Open-source contributors
- Community members providing feedback and testing

**Special thanks to:**
- All beta testers
- Documentation contributors
- Bug reporters

---

## 🎯 Get Started Now

```bash
# Clone and open in VSCode
git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
cd tt-vscode-toolkit
code .

# Press F5 to launch
# The welcome page opens automatically!
```

**Ready to build AI on Tenstorrent hardware? Let's go! 🚀**

---

**Questions?** Check [FAQ.md](FAQ.md) or join our [Discord](https://discord.gg/tenstorrent)!
