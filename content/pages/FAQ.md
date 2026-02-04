# Tenstorrent Developer Extension - FAQ

**Frequently Asked Questions** - Your quick reference for common questions, troubleshooting, and tips from all 48 lessons.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Remote Development & SSH](#remote-development--ssh)
- [Hardware & Detection](#hardware--detection)
- [Installation & Setup](#installation--setup)
- [Models & Downloads](#models--downloads)
- [Inference & Serving](#inference--serving)
- [Custom Training](#custom-training)
- [Compilers & Tools](#compilers--tools)
- [Troubleshooting](#troubleshooting)
- [Performance & Optimization](#performance--optimization)
- [Community & Support](#community--support)

---

## Getting Started

### Q: Which lesson should I start with?

**A:** Start with **[Hardware Detection](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22hardware-detection%22%7D)** if you're brand new. The 48 lessons are organized into 9 categories:

**🚀 Your First Inference (5 lessons)**
1. Hardware Detection → Verify Installation → Download Model → Interactive Chat → API Server

**🏭 Serving Models (4 lessons)**
Production servers (tt-inference-server, vLLM) and generation (Image, Video)

**🎓 Custom Training (8 lessons)** ⭐ NEW!
Fine-tune models or train from scratch - validated on hardware with both workflows working!

**🎯 Applications (2 lessons)**
Coding Assistant, AnimateDiff Video Generation

**👨‍🍳 Tenstorrent Cookbook (6 lessons)**
Game of Life, Audio, Mandelbrot, Image Filters, Particle Life + Overview

**🔧 Compilers & Tools (2 lessons)**
TT-Forge, TT-XLA

**🧠 CS Fundamentals (7 lessons)**
Computer Architecture, Memory, Parallelism, Networks, Synchronization, Abstraction, Complexity

**🎓 Advanced Topics (5 lessons)**
tt-installer, Bounty Program, Explore Metalium, Koyeb Deployment (2)

**Can I skip lessons?** Yes! Categories are independent - jump to what interests you.

### Q: Do I need to complete lessons in order?

**A:** Not strictly, but:
- **Hardware Detection, Verify Installation, and Download Model** are foundational - most later lessons assume you've done these
- **Interactive Chat through Image Generation** build on each other but can be done selectively
- **Advanced topics** (compilers, RISC-V, bounty program) are more independent

**Quick start for experienced users:**
1. Run [Hardware Detection](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22hardware-detection%22%7D) (2 minutes - verify hardware)
2. Skip to [Production Inference with vLLM](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D) (production serving)
3. Explore advanced topics (compilers, RISC-V, bounty program)

### Q: What's the difference between the different tools?

**A:** Tenstorrent has several tools serving different purposes:

| Tool | Purpose | When to Use | Maturity |
|------|---------|-------------|----------|
| **tt-metal** | Low-level framework | Custom kernels, maximum control | Stable |
| **vLLM** | LLM serving | Production LLM deployment | Production |
| **TT-Forge** | MLIR compiler | PyTorch models (experimental) | Beta |
| **TT-XLA** | XLA compiler | JAX/PyTorch (production) | Production |

**Simple guide:**
- Need to run LLMs? → **[Production Inference with vLLM](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D)**
- Want to experiment with PyTorch? → **[Image Classification with TT-Forge](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22forge-image-classification%22%7D)**
- Need JAX support? → **[JAX Inference with TT-XLA](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22tt-xla-jax%22%7D)**
- Building custom kernels? → **tt-metal** (Hardware Detection, Verify Installation, Download Model, RISC-V Programming)

---

## Remote Development & SSH

### Q: Can I use this extension from my Mac/Windows laptop to access remote Tenstorrent hardware?

**A:** **Yes!** Use VSCode's **Remote-SSH extension** - the industry-standard solution for remote development.

**This is the recommended approach for:**
- Developing on macOS/Windows while hardware is on Linux
- Working from laptop with hardware in datacenter/cloud
- Team development with shared hardware resources

**Why Remote-SSH is perfect for this:**
- ✅ **Zero extension changes needed** - Everything "just works"
- ✅ **Transparent experience** - Feels like local development
- ✅ **All features work** - Terminal commands, file operations, debugging
- ✅ **Battle-tested** - Used by millions of developers daily

### Q: How do I set up Remote-SSH for Tenstorrent development?

**A:** Quick setup guide:

**Step 1: Install Remote-SSH extension**
1. Open VSCode on your local machine (Mac/Windows)
2. Open Extensions panel (`Cmd+Shift+X` or `Ctrl+Shift+X`)
3. Search for "Remote - SSH"
4. Install the official Microsoft extension

**Step 2: Configure SSH connection**

Add your Tenstorrent machine to SSH config:

```bash
# On your local machine, edit ~/.ssh/config
# (Cmd+Shift+P → "Remote-SSH: Open Configuration File")

Host tenstorrent-dev
  HostName 192.168.1.100        # Your hardware machine IP
  User ubuntu                   # Your username
  IdentityFile ~/.ssh/id_rsa    # Your SSH key
  ForwardAgent yes              # Optional: Forward SSH agent
```

**Step 3: Connect to remote machine**
1. `Cmd+Shift+P` (or `Ctrl+Shift+P`) → "Remote-SSH: Connect to Host"
2. Select "tenstorrent-dev"
3. New VSCode window opens connected to remote machine

**Step 4: Install Tenstorrent extension on remote**
1. In the remote VSCode window, go to Extensions
2. Search for "Tenstorrent Developer Extension"
3. Click "Install in SSH: tenstorrent-dev"

**Step 5: Start using lessons!**
- All terminal commands run on remote machine
- All file operations work on remote filesystem
- Hardware detection works automatically
- Models download to remote machine

### Q: Do the lessons work through Remote-SSH?

**A:** **Yes, perfectly!** Remote-SSH makes everything transparent:

**What works automatically:**
- ✅ All terminal commands run on remote machine
- ✅ File operations (`Read`, `Write`, `Edit`) work on remote filesystem
- ✅ Hardware detection (`tt-smi`) works
- ✅ Model downloads go to remote machine
- ✅ Inference runs on remote hardware
- ✅ Port forwarding automatic (access servers on localhost)

**Example workflow:**
1. Connect via Remote-SSH from your Mac
2. Open Tenstorrent walkthrough (works like local)
3. Run Hardware Detection → `tt-smi` runs on remote
4. Download model → Saves to remote `~/models/`
5. Start vLLM server → Runs on remote, port auto-forwarded
6. Test from local browser → `http://localhost:8000` works!

**No code changes needed** - The extension doesn't know or care that you're remote!

### Q: What about SSH without Remote-SSH extension?

**A:** **Not recommended.** Manual SSH has major problems:

❌ **File operations break** - Extension reads/writes local filesystem, not remote
❌ **Path mismatches** - `~/models/` on Mac ≠ `~/models/` on remote
❌ **Complex escaping** - Terminal commands get mangled through SSH
❌ **No port forwarding** - Can't access servers on `localhost`
❌ **Poor UX** - Feels disconnected, hard to debug

**Example of problems:**

If you manually SSH in terminal:

```bash
# This command in lesson creates file on your MAC, not remote!
cat > ~/tt-scratchpad/script.py << 'EOF'
...
EOF
```

Then this fails because the file is on the wrong machine:

```bash
ssh user@remote python3 ~/tt-scratchpad/script.py
```

**With Remote-SSH:** Both operations happen on remote automatically.

### Q: Can multiple people share the same remote hardware?

**A:** Yes, but with considerations:

**Shared hardware works best with:**
- ✅ **Resource coordination** - Don't run multiple large models simultaneously
- ✅ **User directories** - Each user has own `~/models/`, `~/tt-scratchpad/`
- ✅ **Port management** - Use different ports (8000, 8001, 8002...)
- ✅ **Communication** - Team chat to coordinate who's using hardware

**Limitations:**
- ⚠️ Only one model can load on device at a time
- ⚠️ Large models need device reset between users
- ⚠️ `/dev/shm` shared memory might need cleanup

**Best practice for teams:**
```bash
# User 1
vllm ... --port 8001

# User 2
vllm ... --port 8002

# Each user accesses their own server
curl http://localhost:8001/...  # User 1
curl http://localhost:8002/...  # User 2
```

### Q: What about Tenstorrent Cloud? Does Remote-SSH work?

**A:** **Yes!** Tenstorrent Cloud instances are perfect for Remote-SSH:

**Typical setup:**
1. Get Tenstorrent Cloud instance (pre-configured with hardware)
2. Receive SSH credentials
3. Add to `~/.ssh/config` on your laptop
4. Connect via Remote-SSH
5. Start developing!

**Cloud benefits:**
- ✅ Pre-installed tt-metal and drivers
- ✅ Pre-configured environment
- ✅ No hardware setup needed
- ✅ Access from anywhere

**Example cloud SSH config:**
```bash
Host tt-cloud
  HostName cloud.instance.tenstorrent.com
  User your-username
  IdentityFile ~/.ssh/tt-cloud-key
  ForwardAgent yes
```

### Q: Are there performance considerations with Remote-SSH?

**A:** Remote-SSH is **very efficient**:

**Fast operations (no noticeable latency):**
- Terminal commands (SSH is fast)
- File editing (only changes sync)
- Running inference (happens on remote)
- Model downloads (direct from HuggingFace to remote)

**What uses bandwidth:**
- File tree indexing (one-time)
- Large file transfers (if you copy files between machines)
- Extension updates (rare)

**Best practices:**
- ✅ Use wired connection or good WiFi
- ✅ Keep large models on remote (don't transfer)
- ✅ Use compression in SSH config: `Compression yes`

**Real-world experience:**
- **Feels instant** on good connection (10+ Mbps)
- **Usable** on moderate connection (1-5 Mbps)
- **Not recommended** on very slow connections (<1 Mbps)

### Q: How do I disconnect from remote machine?

**A:** Several options:

**Graceful disconnect:**
- Close the remote VSCode window
- Connection closes, remote processes continue running

**From command palette:**
- `Cmd+Shift+P` → "Remote-SSH: Close Remote Connection"

**Important:** vLLM servers keep running after disconnect!
```bash
# Before disconnecting, you may want to:
docker ps                    # Note container IDs
docker stop <container-id>   # Stop servers

# Or leave them running and reconnect later
```

**Reconnecting:**
- Just repeat: "Remote-SSH: Connect to Host" → Select your host
- Everything exactly as you left it

---

## Hardware & Detection

### Q: Can I try Tenstorrent development without hardware?

**A:** **Yes!** Use **ttsim** - Tenstorrent's full-system simulator.

**What is ttsim:**
- Virtual Wormhole or Blackhole device that runs on any Linux/x86_64 system
- No physical hardware needed
- Slower than silicon but fast enough for learning and experimentation
- Perfect for exploring before purchasing hardware

**Quick Start:**

```bash
# Download simulator (replace vX.Y with latest version)
mkdir -p ~/sim
cd ~/sim
wget https://github.com/tenstorrent/ttsim/releases/latest/download/libttsim_wh.so

# Copy SOC descriptor
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# Set environment variable
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so

# Run in slow dispatch mode (required for simulator)
export TT_METAL_SLOW_DISPATCH_MODE=1

# Test it works
cd $TT_METAL_HOME
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

**What you CAN do with ttsim:**
- ✅ Learn TT-Metal programming model
- ✅ Run programming examples and tests
- ✅ Develop and debug kernels
- ✅ Test TTNN operations
- ✅ Explore Tenstorrent architecture

**What you CAN'T do (too slow):**
- ❌ Run full model inference (vLLM, large models)
- ❌ Production workloads
- ❌ Performance benchmarking
- ❌ Real-time applications

**Which lessons work with ttsim:**
- **Hardware Detection:** Partial support - `ttnn` works, `tt-smi` won't detect simulated device
- **Verify Installation:** Yes - programming examples work great
- **RISC-V Programming:** Yes - perfect for learning low-level programming
- **Model Inference lessons:** No - too slow for practical use (Interactive Chat through Image Generation)
- **Compiler lessons:** Limited - depends on workload (TT-Forge, TT-XLA)

**Resources:**
- GitHub: https://github.com/tenstorrent/ttsim
- Releases: https://github.com/tenstorrent/ttsim/releases/latest

**Tip:** Use ttsim for learning and kernel development, then move to real hardware for model inference and production workloads.

### Q: Which hardware do I have?

**A:** Run this command:

```bash
tt-smi -s | grep -o '"board_type": "[^"]*"'
```

**Output tells you:**
- **N150** - Single Wormhole chip (development, 64K context)
- **N300** - Dual Wormhole chips (128K context, TP=2)
- **T3K** - Eight Wormhole chips (large models, TP=8)
- **P100** - Single Blackhole chip (newer architecture)
- **P150** - Dual Blackhole chips (TP=2)

### Q: tt-smi says "No devices found" - what do I do?

**A:** Try these steps in order:

1. **Check PCIe detection:**
   ```bash
   lspci | grep -i tenstorrent
   ```
   Should show: `Processing accelerators: Tenstorrent Inc.`

2. **Try with sudo:**
   ```bash
   sudo tt-smi
   ```
   If this works, you have a permissions issue.

3. **Reset the device:**
   ```bash
   tt-smi -r
   ```

4. **Full cleanup (if still failing):**
   ```bash
   sudo pkill -9 -f tt-metal
   sudo pkill -9 -f vllm
   sudo rm -rf /dev/shm/tenstorrent* /dev/shm/tt_*
   tt-smi -r
   ```

**Still not working?** Check the [Hardware Detection](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22hardware-detection%22%7D) lesson troubleshooting section for detailed steps.

### Q: What's the difference between Wormhole and Blackhole?

**A:**
- **Wormhole (N150, N300, T3K)** - 2nd generation, well-validated, most models tested
- **Blackhole (P100, P150)** - Latest generation, newer architecture, some experimental models

**For production:** Stick with Wormhole (N150/N300/T3K) - more models validated.

**For experimentation:** Blackhole offers newer features but check model compatibility.

### Q: How do I know what my hardware can run?

**A:** Quick reference:

| Hardware | Max Model Size | Max Context | Multi-chip | Best For |
|----------|---------------|-------------|------------|----------|
| N150, P100 | 8B | 64K | No (TP=1) | Development, prototyping |
| N300, P150 | 13B | 128K | Yes (TP=2) | Medium models, multi-user |
| T3K | 70B+ | 128K | Yes (TP=8) | Large models, production |

---

## Installation & Setup

### Q: How do I verify tt-metal is working?

**A:** Run this quick test:

```bash
python3 -c "import ttnn; print('✓ tt-metal ready')"
```

**If it fails:**
- Check `PYTHONPATH` includes tt-metal directory
- Verify tt-metal is built: `ls ~/tt-metal/build/lib`
- Rebuild if needed: `cd ~/tt-metal && ./build_metal.sh`

### Q: Which Python version do I need?

**A:**
- **Minimum:** Python 3.9
- **Recommended:** Python 3.10+
- **For TT-Forge:** Python 3.11+ (requirement)

Check your version:
```bash
python3 --version
```

### Q: Where should models be installed?

**A:** Standard locations:

- **Recommended:** `~/models/[model-name]/`
  - Example: `~/models/Llama-3.1-8B-Instruct/`
  - Used by most lessons

- **HuggingFace cache:** `~/.cache/huggingface/hub/`
  - Automatic when using `huggingface-cli`
  - Takes more disk space (keeps multiple versions)

**Both formats needed for some lessons:**
- Meta format: `~/models/[model]/original/` (for Lessons 3-5)
- HuggingFace format: `~/models/[model]/` (for Lessons 6-9)

### Q: How much disk space do I need?

**A:** Plan for:
- **tt-metal:** ~5GB (source + build artifacts)
- **vLLM:** ~20GB (including dependencies)
- **Per model:**
  - Small models (1-3B): 10-15GB
  - Medium models (7-8B): 30-40GB
  - Large models (70B): 140GB+

**Minimum for this extension:** 100GB free space

---

## Models & Downloads

### Q: Which model should I download first?

**A:** **Llama-3.1-8B-Instruct** - covered in [Download Model](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22download-model%22%7D).

**Why this model:**
- ✅ Works on N150 (most common hardware)
- ✅ Good performance for 8B size
- ✅ Supports all lessons (4-9)
- ✅ Well-tested and documented

**Download command:**
```bash
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
  --local-dir ~/models/Llama-3.1-8B-Instruct
```

### Q: How do I handle HuggingFace authentication?

**A:** Three options:

**Option 1: Environment variable (recommended for scripts)**
```bash
export HF_TOKEN=your_token_from_huggingface
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct --local-dir ~/models/Llama-3.1-8B-Instruct
```

**Option 2: Interactive login (recommended for manual use)**
```bash
huggingface-cli login
# Paste your token when prompted
```

**Option 3: In code**
```python
from huggingface_hub import login
login(token="your_token_from_huggingface")
```

**Get a token:** https://huggingface.co/settings/tokens

### Q: Download failed with "repository not found" - why?

**A:** Gated models require access request:

1. Go to model page on HuggingFace
2. Click "Request access" button
3. Wait for approval (usually instant for Llama)
4. Ensure you're authenticated (see question above)

**For Llama models:** Must accept Meta's license agreement.

### Q: Can I use models from other sources?

**A:** Yes, but:
- **HuggingFace format** required for vLLM (Production Inference lessons)
- **Meta checkpoint format** required for Direct API (Interactive Chat, API Server)
- **ONNX/PyTorch format** for TT-Forge (Image Classification)

**Recommendation:** Stick with HuggingFace - most compatible.

---

## Inference & Serving

### Q: Which inference method should I use?

**A:** Depends on your goal:

| Method | Lesson | Best For | Speed (after load) |
|--------|--------|----------|-------------------|
| **One-shot demo** | Download Model | Testing, verification | 2-5 min per query |
| **Interactive chat** | Interactive Chat | Learning, prototyping | 1-3 sec per query |
| **Flask API** | API Server | Simple custom APIs | 1-3 sec per query |
| **vLLM** | Production Inference | Production serving | 1-3 sec per query |

**Quick guide:**
- Just testing? → **[Download Model](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22download-model%22%7D)** (one-shot demo)
- Learning/experimenting? → **[Interactive Chat](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22interactive-chat%22%7D)** (interactive)
- Building custom app? → **[API Server](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22api-server%22%7D)** (Flask API)
- Production deployment? → **[Production Inference with vLLM](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D)** (vLLM)

### Q: Why does first load take 2-5 minutes?

**A:** Model initialization involves:
1. Loading weights from disk (~16GB for Llama-8B)
2. Converting to TT-Metal format
3. Distributing to hardware cores
4. JIT compilation of kernels

**This is normal and only happens once.**

**Subsequent queries are fast (1-3 seconds)** because model stays in memory.

### Q: Can I run multiple models simultaneously?

**A:** On same hardware: **No** (one model at a time per device)

**Workarounds:**
- Use model switching (stop one, start another)
- Use multiple hardware devices
- Use different hardware for different models (N150 for model A, N300 for model B)

### Q: What does "context length" mean and why does it matter?

**A:**
- **Context length** = Maximum tokens (words/subwords) model can process at once
- Includes both input (prompt) + output (response)

**Hardware limits:**
- N150/P100: 64K tokens (~48K words)
- N300/T3K: 128K tokens (~96K words)

**Exceeding context?**
```
RuntimeError: Input sequence length exceeds maximum
```

**Solutions:**
- Shorten your prompts
- Use summarization for long documents
- Switch to hardware with larger context support

### Q: Getting PyTorch dataclass errors with vLLM - how do I fix them?

**A:** This error (`TypeError: must be called with a dataclass type or instance`) is caused by PyTorch version mismatches.

**Error looks like:**
```
TypeError: must be called with a dataclass type or instance
# ... torch/_inductor/runtime/hints.py errors
```

**Root cause:** vLLM on Tenstorrent hardware requires **PyTorch 2.5.0+cpu** specifically. Other versions (2.4.x, 2.7.x) cause compatibility issues.

**Solution: Recreate your vLLM environment**
```bash
bash ~/tt-scratchpad/setup-vllm-env.sh
```

This automated script:
- ✅ Creates environment at correct location (`~/tt-metal/build/python_env_vllm`)
- ✅ Installs PyTorch 2.5.0+cpu (exact version)
- ✅ Installs all required dependencies
- ✅ Validates installation before completion

**Verify your environment:**
```bash
source ~/activate-vllm-env.sh
python3 -c "import torch; print('PyTorch version:', torch.__version__)"
# Should print: PyTorch version: 2.5.0+cpu
```

**Why the specific version?** TT-Metal hardware drivers are built against PyTorch 2.5.0+cpu APIs. Other versions have incompatible dataclass implementations.

---

## Custom Training

### Q: Can I train models on Tenstorrent hardware?

**A:** Yes! The extension now includes 8 complete Custom Training lessons (CT1-CT8) that are fully validated on hardware.

**What's working:**
- ✅ **From-scratch training:** NanoGPT (11M params) - 136 steps in 76 seconds on N150
- ✅ **Fine-tuning:** Train custom models on your own datasets
- ✅ **Complete toolkit:** Setup scripts, validation, and tested templates
- ✅ **Production-ready:** Both training workflows validated end-to-end

**Recommended version:** tt-metal v0.66.0-rc7 (fully tested)

### Q: What hardware do I need for training?

**A:** Training requirements depend on model size:

**N150 (Wormhole single-chip):**
- ✅ Perfect for NanoGPT (11M params, 6 layers, 384 dim)
- ✅ From-scratch training on Shakespeare, custom datasets
- ❌ TinyLlama-1.1B OOM (needs 2GB DRAM, only 1GB available)

**N300+ (Wormhole dual-chip or higher):**
- ✅ Everything N150 can do
- ✅ TinyLlama-1.1B fine-tuning (2GB+ DRAM available)
- ✅ Larger models and batch sizes

**Recommendation:** Start with N150 and NanoGPT to learn the workflow!

### Q: What's the difference between fine-tuning and training from scratch?

**A:**

**Fine-tuning (CT4):**
- Start with pre-trained model (e.g., TinyLlama-1.1B)
- Train on small custom dataset (50-1000 examples)
- Adapts model to your specific task
- Faster (minutes to hours)
- Good for: Q&A bots, domain-specific assistants

**Training from Scratch (CT8):**
- Build model from random initialization
- Train on large dataset (Shakespeare, your own data)
- Learn patterns from ground up
- Slower (hours to days)
- Good for: Understanding training deeply, custom architectures

**Which should I start with?** CT8 (from-scratch) - it's faster on N150 with NanoGPT and teaches fundamentals!

### Q: What tt-metal version do I need for training?

**A:** Training requires **v0.66.0-rc5 or later**

**Why:**
- v0.64.5 and earlier: C++ tt-train only ❌
- v0.66.0-rc5+: Python ttml module available ✅
- v0.66.0-rc7: Fully validated and recommended ✅

**Check your version:**
```bash
cd $TT_METAL_HOME && git describe --tags
```

**See CT4 and CT8 lessons for complete setup instructions!**

---

## Compilers & Tools

### Q: What's the difference between TT-Forge and TT-XLA?

**A:**

| Feature | TT-Forge | TT-XLA |
|---------|----------|--------|
| **Status** | Experimental | Production-ready |
| **Multi-chip** | Single only | Yes (TP/DP) |
| **Frameworks** | PyTorch, ONNX | JAX, PyTorch/XLA |
| **Model support** | Limited (169 validated) | Broader |
| **Installation** | Complex (build from source) | Simple (pip) |

**When to use TT-Forge:**
- Experimenting with PyTorch models
- Learning MLIR compilation
- Working with validated models list

**When to use TT-XLA:**
- Production multi-chip workloads
- JAX workflows
- Need stability and support

### Q: Why did my model fail to compile in TT-Forge?

**A:** TT-Forge is experimental. Common reasons:

1. **Unsupported operators**
   - Not all PyTorch ops implemented
   - Check tt-forge-models for validated examples

2. **Model architecture**
   - Very new architectures may not work
   - Dynamic shapes not supported
   - Control flow limited

3. **Environment variable pollution** (most common!)
   ```bash
   unset TT_METAL_HOME
   unset TT_METAL_VERSION
   # Then try again
   ```

**Recommendation:** Start with MobileNetV2 (Image Classification with TT-Forge default) - known to work.

### Q: How do I know if my model is supported?

**A:**

**For TT-Forge:**
- Check [tt-forge-models repository](https://github.com/tenstorrent/tt-forge-models)
- 169 validated models listed
- Start with these before trying others

**For vLLM:**
- Llama family well-supported (2, 3, 3.1, 3.2)
- Mistral supported
- Qwen supported (needs N300+ for larger models)
- Check documentation for your specific model

**For TT-XLA:**
- Most JAX/Flax models work
- PyTorch/XLA support growing
- GPT-2 demo included (JAX Inference with TT-XLA)

---

## Troubleshooting

### Q: Command failed with "ImportError: undefined symbol"

**A:** This is almost always environment variable pollution.

**Fix:**
```bash
unset TT_METAL_HOME
unset TT_METAL_VERSION
# Retry your command
```

**Make permanent:**
Add to `~/.bashrc`:
```bash
# Prevent TT-Metal environment pollution
unset TT_METAL_HOME
unset TT_METAL_VERSION
```

**Why this happens:** Different versions of libraries loaded due to environment variables overriding build paths.

### Q: vLLM server won't start - what do I check?

**A:** Systematic debugging:

**1. Check environment variables:**
```bash
echo $TT_METAL_HOME    # Should be ~/tt-metal
echo $MESH_DEVICE      # Should match your hardware (N150, etc.)
echo $PYTHONPATH       # Should include $TT_METAL_HOME
```

**2. Verify model path:**
```bash
ls ~/models/Llama-3.1-8B-Instruct/config.json
```

**3. Check for other processes:**
```bash
ps aux | grep -E "tt-metal|vllm"
# Kill if needed:
# pkill -9 -f vllm
```

**4. Verify vLLM installation:**
```bash
source ~/tt-vllm-venv/bin/activate
python3 -c "import vllm; print(vllm.__version__)"
```

**5. Check device availability:**
```bash
tt-smi
# Should show your device
```

### Q: "Out of memory" errors - what can I do?

**A:** Several strategies:

**1. Reduce context length:**
```bash
# Instead of:
--max-model-len 65536

# Try:
--max-model-len 32768
```

**2. Reduce batch size:**
```bash
# Instead of:
--max-num-seqs 32

# Try:
--max-num-seqs 16
```

**3. Use smaller model:**
- 8B → 3B (Llama-3.2-3B)
- 8B → 1B (Llama-3.2-1B)

**4. Clear device state:**
```bash
sudo pkill -9 -f tt-metal
sudo rm -rf /dev/shm/tenstorrent* /dev/shm/tt_*
tt-smi -r
```

### Q: Build failed - where do I look?

**A:**

**tt-metal build issues:**
```bash
cd ~/tt-metal
./build_metal.sh 2>&1 | tee build.log
# Check build.log for errors
```

**Common build failures:**
- **Missing dependencies:** `sudo apt-get install build-essential cmake`
- **Python version:** Need 3.9+ (check with `python3 --version`)
- **Disk space:** Need 10GB+ free
- **Memory:** Need 16GB+ RAM for building

**TT-Forge build issues:**
- **Python 3.11 required:** Can't use older Python
- **clang-17 required:** `sudo apt-get install clang-17`
- **Environment variables:** Must unset TT_METAL_HOME first

### Q: TTNN import errors or symbol undefined errors in cloud environments - how do I fix them?

**A:** After rolling back or updating tt-metal versions, TTNN bindings may become incompatible.

**Symptoms:**
- `ImportError: undefined symbol: _ZN2tt9tt_fabric15SetFabricConfigENS0...`
- `ImportError: undefined symbol: MPIX_Comm_revoke`
- TTNN examples that previously worked now fail

**Common Cause:**
Rolling back or updating tt-metal versions (for example, to match specific vLLM compatibility) can break TTNN bindings.

**Solution - Clean Rebuild to Known-Good Version:**

1. **Note your original working commit:**
   ```bash
   cd ~/tt-metal
   git log --oneline | head -5
   # Save the commit hash that was working
   ```

2. **Checkout the known-good version:**
   ```bash
   cd ~/tt-metal
   git checkout 5143b856eb  # Replace with your working commit
   git submodule update --init --recursive
   ```

3. **Complete clean rebuild:**
   ```bash
   cd ~/tt-metal
   # Clean all build artifacts
   rm -rf build build_Release

   # Reinstall dependencies
   sudo ./install_dependencies.sh

   # Rebuild from scratch
   ./build_metal.sh
   ```

4. **Test TTNN:**
   ```bash
   source ~/tt-metal/python_env/bin/activate
   export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
   export PYTHONPATH=~/tt-metal:$PYTHONPATH
   python3 -m ttnn.examples.usage.run_op_on_device
   ```

**Important Notes:**
- The original/untouched tt-metal version is often the most stable
- Rolling back to older commits can create incompatible bindings
- Always do a **complete clean rebuild** after changing commits
- OpenMPI library path is required: `/opt/openmpi-v5.0.7-ulfm/lib`

**Known-Good Commit (as of Dec 2024):**
- `5143b856eb` (Oct 28, 2024) - Stable TTNN, validated on N150

### Q: Getting OpenMPI errors - how do I fix them?

**A:** OpenMPI library path errors are common and easy to fix.

**Symptoms:**
- Errors mentioning "libmpi.so" or "OpenMPI"
- "ImportError: cannot open shared object file"
- Commands fail with MPI-related errors

**Fix:**
```bash
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

**Make permanent:**
Add to `~/.bashrc`:
```bash
# OpenMPI library path for Tenstorrent
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
```

Then reload:
```bash
source ~/.bashrc
```

**Why this happens:** The OpenMPI library installation isn't in the system's default library search path, so you need to explicitly tell the dynamic linker where to find it.

**Alternative OpenMPI paths:**
If the above doesn't work, try:
```bash
# Find your OpenMPI installation
find /opt -name "libmpi.so*" 2>/dev/null

# Use the directory containing the .so files
export LD_LIBRARY_PATH=/path/to/openmpi/lib:$LD_LIBRARY_PATH
```

### Q: Downloads are slow or failing

**A:**

**Slow downloads:**
- HuggingFace throttles anonymous requests
- Solution: Login with `huggingface-cli login`
- Consider downloading overnight for large models

**Failing downloads:**
1. **Check internet connection**
2. **Verify HF authentication** (see authentication question above)
3. **Check disk space:** `df -h ~`
4. **Try resuming:**
   ```bash
   huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
     --local-dir ~/models/Llama-3.1-8B-Instruct \
     --resume-download
   ```

---

## Performance & Optimization

### Q: How can I speed up inference?

**A:**

**After first load (model in memory):**
- **Already fast:** 1-3 seconds per query typical
- **Can't improve much:** Hardware-optimized already

**For batch processing:**
- Use vLLM's batching: `--max-num-seqs 32`
- Process multiple requests together
- 3-5x throughput improvement

**For lower latency:**
- Reduce `max_tokens` parameter (shorter responses = faster)
- Use smaller model (8B → 3B)
- Consider hardware upgrade (N150 → N300)

### Q: What are good vLLM server parameters?

**A:** Recommended by hardware:

**N150 (single chip):**
```bash
--max-model-len 65536   # Full 64K context
--max-num-seqs 16       # Moderate batching
--block-size 64         # Standard
```

**N300 (dual chip):**
```bash
--max-model-len 131072  # Full 128K context
--max-num-seqs 32       # Higher batching
--block-size 64
--tensor-parallel-size 2  # Use both chips
```

**T3K (8 chips):**
```bash
--max-model-len 131072
--max-num-seqs 64       # High batching
--block-size 64
--tensor-parallel-size 8  # Use all chips
```

**Conservative (if OOM errors):**
- Reduce `max-model-len` by 50%
- Reduce `max-num-seqs` by 50%
- Test incrementally

### Q: How do I monitor performance?

**A:**

**Token generation speed:**
```bash
# In vLLM output, look for:
"Generated 150 tokens in 2.5 seconds (60 tokens/sec)"
```

**Server metrics:**
```bash
# vLLM exposes Prometheus metrics:
curl http://localhost:8000/metrics
```

**System monitoring:**
```bash
# GPU-like monitoring for TT:
watch -n 1 tt-smi
```

**Load testing:**
```bash
# Install hey:
go install github.com/rakyll/hey@latest

# Test throughput:
hey -n 100 -c 10 -m POST \
  -H "Content-Type: application/json" \
  -d '{"model": "...", "messages": [...]}' \
  http://localhost:8000/v1/chat/completions
```

---

## Community & Support

### Q: Where can I get help?

**A:**

**Official channels:**
- **Discord:** https://discord.gg/tenstorrent (most active)
- **GitHub Issues:**
  - tt-metal: https://github.com/tenstorrent/tt-metal/issues
  - vLLM: https://github.com/tenstorrent/vllm/issues
  - TT-Forge: https://github.com/tenstorrent/tt-forge/issues
- **Documentation:** https://docs.tenstorrent.com

**When asking for help, include:**
1. Hardware type (N150/N300/T3K/P100)
2. Error message (full text)
3. Command you ran
4. Output of `tt-smi`
5. Which lesson you're on

### Q: How do I report a bug?

**A:**

**Before reporting:**
1. Search existing issues on GitHub
2. Verify hardware works (`tt-smi`)
3. Try reset (`tt-smi -r`)
4. Check you're on latest tt-metal/vLLM

**When reporting, include:**
```
Hardware: N150
OS: Ubuntu 22.04
tt-metal version: [git rev-parse HEAD output]
vLLM version: [pip show vllm]
Error: [paste full error]
Steps to reproduce: [numbered list]
```

**Good issue = faster fix!**

### Q: Can I contribute?

**A:** Yes! Several ways:

**1. [Bounty Program](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22bounty-program%22%7D)**
- Bring up new models
- Earn rewards
- Official contribution path

**2. Documentation**
- Fix typos/errors
- Add examples
- Improve tutorials

**3. Code contributions**
- Bug fixes
- Performance improvements
- New features

**Start here:**
- Join Discord #contributing channel for guidance
- Ask about "good first issue" opportunities
- Review documentation at https://docs.tenstorrent.com

### Q: Is this production-ready?

**A:** Depends on component:

**Production-ready (✅):**
- **tt-metal** - Stable, tested
- **vLLM** - Production-grade serving
- **TT-XLA** - Production compiler

**Experimental (⚠️):**
- **TT-Forge** - Beta, limited model support
- **Some models** - Check validation status

**Recommendation:**
- **For production:** Stick with vLLM + validated models
- **For experimentation:** Try TT-Forge, new models
- **Always test** thoroughly before production deployment

---

## Quick Reference

### Essential Commands

```bash
# Hardware
tt-smi                                    # Check hardware
tt-smi -s                                # Structured output
tt-smi -r                                # Reset device

# Model info
ls ~/models/                            # List installed models
du -sh ~/models/*                       # Check model sizes

# Environment
python3 -c "import ttnn; print('✓')"   # Test tt-metal
which huggingface-cli                   # Check HF CLI

# vLLM
source ~/tt-vllm-venv/bin/activate      # Activate venv
curl http://localhost:8000/health       # Check server
curl http://localhost:8000/metrics      # Get metrics

# Cleanup
sudo pkill -9 -f "tt-metal|vllm"       # Kill processes
sudo rm -rf /dev/shm/tt_*              # Clear shared memory
tt-smi -r                               # Reset hardware
```

### Quick Diagnostic

Run this to check your setup:

```bash
#!/bin/bash
echo "=== Tenstorrent Diagnostic ==="
echo ""
echo "Hardware:"
tt-smi -s 2>&1 | grep -o '"board_type": "[^"]*"' || echo "❌ No hardware detected"
echo ""
echo "tt-metal:"
python3 -c "import ttnn; print('✓ Working')" 2>&1 || echo "❌ Not working"
echo ""
echo "Models:"
ls ~/models/ 2>/dev/null | head -3 || echo "❌ No models found"
echo ""
echo "Disk space:"
df -h ~ | grep -v Filesystem
echo ""
echo "Python:"
python3 --version
```

---

## Advanced Learning Resources

### Q: Where can I learn about low-level RISC-V programming on Tenstorrent hardware?

**A:** Check out the **[CS Fundamentals series](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22cs-fundamentals-01-computer%22%7D)** - Module 1 covers RISC-V & Computer Architecture!

Each Tensix core contains **five RISC-V processors** (RV32IM ISA):
- **BRISC (RISCV_0)** - Primary data movement
- **NCRISC (RISCV_1)** - Network operations
- **TRISC0/1/2** - Compute pipeline (unpack, math, pack)

With 176 Tensix cores on Wormhole, that's **880 RISC-V cores** you can program directly!

**What Module 1 includes:**
- ✅ Von Neumann architecture & fetch-decode-execute cycle
- ✅ RISC-V ISA fundamentals
- ✅ Hands-on example: Add two integers in RISC-V assembly
- ✅ Build and run tt-metal programming examples
- ✅ Explore kernel source code
- ✅ Comprehensive exploration guide (60+ pages)

**Topics covered across 7 CS Fundamentals modules:**
- RISC-V architecture and memory maps
- Memory hierarchy and cache locality
- Parallel computing (scale from 1 to 880 cores!)
- NoC (Network-on-Chip) programming
- Synchronization and barriers
- Abstraction layers and compilation
- Computational complexity in practice

**Access the series:**
- From Welcome page → CS Fundamentals section
- Or start with [Module 1: RISC-V & Computer Architecture](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22cs-fundamentals-01-computer%22%7D)

**View the full guide:**
- [Open RISC-V Exploration Guide](command:tenstorrent.showRiscvGuide) - Comprehensive deep-dive documentation

**Perfect for:**
- Developers who want to understand the hardware at the lowest level
- Embedded systems programmers exploring RISC-V at scale
- Computer architecture enthusiasts
- Anyone optimizing kernel performance

---

## Still Have Questions?

**Check:**
1. Specific lesson troubleshooting sections
2. CLAUDE.md for detailed technical info
3. Discord #help channel

**Remember:** Most issues are:
- Environment variables (unset TT_METAL_HOME)
- Permissions (try sudo or add to tenstorrent group)
- Device state (reset with tt-smi -r)

**When in doubt:**
```bash
tt-smi -r
sudo rm -rf /dev/shm/tt_*
# Then retry
```

---

**Last updated:** January 2026
**Extension version:** 0.0.283

**Found an error in this FAQ?** Please report it on GitHub or Discord!
