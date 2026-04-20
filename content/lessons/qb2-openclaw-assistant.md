---
id: qb2-openclaw-assistant
title: OpenClaw AI Assistant on QuietBox 2
description: Deploy a local AI assistant with Tenstorrent expertise, memory search, and 70B reasoning powered by your QB2 hardware
category: applications
tags:
  - qb2
  - p300x2
  - openclaw
  - ai-assistant
  - memory-search
  - 70b
  - agent-framework
supportedHardware:
  - p300x2
  - n150
status: validated
validatedOn:
  - p300x2
estimatedMinutes: 45
---

# OpenClaw AI Assistant on QuietBox 2

Transform your QuietBox 2 into an expert AI assistant that knows everything about Tenstorrent hardware, deployment, and programming - with citations from 45+ interactive lessons and official documentation.

## What You'll Build

**OpenClaw** is an enterprise-grade AI agent framework that runs entirely on your QB2. Once configured, you get:

- 🧠 **TT Expert Mode**: Ask questions about hardware, deployment, lessons - get detailed answers with citations
- 📚 **Memory Search**: Automatically searches 45+ lessons + official TT docs (1,200+ knowledge chunks)
- 🎮 **Agent Framework**: Build custom AI agents (adventure games included as example)
- 🔧 **Tool Calling**: Agents can execute commands, search files, manage system state
- 💬 **Multiple Interfaces**: Terminal UI, WebSocket API, channel integrations

**What makes this special:**
- **Runs locally** - Your data never leaves your machine
- **Hardware-accelerated** - 70B model on your QB2's 4x Blackhole chips
- **Self-documenting** - Knows all tt-vscode-toolkit lessons by heart
- **Production-ready** - Used for booth demos and development

## Architecture

```
┌─────────────┐
│  User (TUI) │ Ask: "What is QB2?"
└──────┬──────┘
       │ WebSocket to ws://127.0.0.1:18789
       ▼
┌──────────────────┐
│ OpenClaw Gateway │ Process query, search memory, call LLM
└────────┬─────────┘
         │ HTTP POST to http://127.0.0.1:8000/v1/chat/completions
         ▼
┌──────────────────┐
│ vLLM :8000       │ Llama-3.3-70B-Instruct with tool calling
│ (Docker)         │ (accepts all OpenClaw API fields natively)
└────────┬─────────┘
         │ Inference with TT-Metal + vLLM optimizations
         ▼
┌──────────────────┐
│ 4x P300C Chips   │ 480 Tensix cores, 2,654 TFLOPS, 128K context
│ (Blackhole)      │
└──────────────────┘
```

**Performance on QB2 (70B model):**
- First response: ~14 seconds (includes memory search)
- Follow-ups: ~8-10 seconds
- Context window: 131,072 tokens (128K)
- Max concurrent: 32 sequences
- Quality: Excellent reasoning and long-form responses

---

## Prerequisites

### Hardware Requirements

**For QB2 (P300X2) - 70B Model:**
- 4x Blackhole ASICs (P300C boards)
- 236 GB RAM (175 GB required for 70B)
- 2.3 TB disk space (160 GB required)
- Firmware: 19.4.2.0+
- KMD: 2.7.0+

**Verify your QB2:**

```bash
# Check hardware detection
tt-smi -s
# Should show: 4 Blackhole chips, all healthy

# Check available RAM
free -h
# Should show: ~236 GB total
```

**🔍 Check QB2 Hardware**

### Software Requirements

- **tt-inference-server**: `~/code/tt-inference-server` (cloned and working)
- **OpenClaw**: v2026.3.2+ installed
- **Python**: 3.8+ with venv support
- **Docker**: For containerized vLLM deployment

**Quick verification:**

```bash
# tt-inference-server available?
ls ~/code/tt-inference-server/run.py

# OpenClaw installed?
which openclaw || ls ~/openclaw/openclaw.sh

# Docker working?
docker --version
```

---

## Step 1: Deploy vLLM with 70B Model

The first step is getting Llama-3.3-70B-Instruct running on your QB2 with tool calling support (required for OpenClaw agents).

### Understanding the Deployment

**What happens during deployment:**
1. **Environment preparation** (30 seconds) - Reset hardware, stop existing containers
2. **Model download** (10-30 minutes, one-time) - Downloads 140 GB from HuggingFace
3. **Docker startup** (2 minutes) - Initializes container with TT-Metal environment
4. **Model loading** (10-20 minutes) - Loads weights to TT hardware (silent phase)
5. **Warmup** (5 minutes) - Compiles kernels for P300X2 configuration
6. **Ready** - Health endpoint responds, ready for inference

**Total time:** 15-60 minutes (depending on if model is cached)

### The Deployment Command

This exact configuration is validated and working:

```bash
# Clean environment
docker stop $(docker ps -q) 2>/dev/null || true
tt-smi -r
sleep 5

# Deploy 70B with tool calling
cd ~/code/tt-inference-server

python3 run.py \
    --model Llama-3.3-70B-Instruct \
    --tt-device p300x2 \
    --workflow server \
    --docker-server \
    --dev-mode \
    --host-hf-cache ~/.cache/huggingface \
    --skip-system-sw-validation \
    --no-auth \
    --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.9.0-e867533-22be241 \
    --vllm-override-args '{"enable_auto_tool_choice": true, "tool_call_parser": "llama3_json"}'
```

**🚀 Deploy 70B Model**

### Critical Flags Explained

**`--enable-auto-tool-choice`** (REQUIRED for OpenClaw)
- Allows agents to automatically select which tools to use
- Without this, you get "400 auto tool choice requires..." errors
- Enables the memory_search, file operations, and system commands

**`--tool-call-parser llama3_json`** (REQUIRED for OpenClaw)
- Uses Llama 3's native JSON tool calling format
- Ensures reliable tool invocation parsing
- Matches OpenClaw's expected tool response format

**`--override-docker-image`** (Exact version)
- This specific image (0.9.0-e867533-22be241) is validated
- Includes necessary patches for P300X2 multi-chip support
- Different versions may have compatibility issues

**`--no-auth`** (Simplifies local setup)
- No API keys needed for local-only deployment
- Safe for single-user local deployments
- Safe for single-user systems

**`--dev-mode`** (Enables advanced features)
- Allows mounting of custom model specs
- Provides more verbose logging
- Useful for troubleshooting

### Interactive Prompts

During deployment, you'll see prompts:

**1. Model source:**
```
Select model source:
  1. Download from HuggingFace (requires HF_TOKEN)
  2. Use local model files

Choice: 1 or 2
```

Choose **1** for first run (will download 140 GB), **2** for subsequent runs if model is cached.

**2. JWT secret:**
```
Enter JWT secret (press Enter for default):
```

Just press **Enter** (we're using `--no-auth`).

### Monitoring Deployment

**Watch for these phases:**

```bash
# In another terminal, monitor container logs
docker logs -f $(docker ps -q --filter ancestor=*vllm-tt-metal*)

# Key messages to watch for:
# "Fabric initialized" - Hardware ready
# "Loading checkpoint shards" - Model loading started
# (10-20 min silence) - Weight loading (normal!)
# "Allocated TT KV caches" - Compilation starting
# "Readiness file created" - Ready for inference!
```

**Silent phases are normal!** Weight loading takes 10-20 minutes with no log output.

### Verify Deployment

Once you see "Readiness file created", test the endpoint:

```bash
# Check model is loaded
curl http://localhost:8000/v1/models

# Expected output:
# {"object":"list","data":[{"id":"meta-llama/Llama-3.3-70B-Instruct",...}]}

# Test inference (30-second timeout)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Hello",
    "max_tokens": 10
  }'
```

**✅ Test vLLM**

---

## Step 2: Clone tt-claw Repository

The tt-claw repository provides the unified OpenClaw configuration and helper scripts.

```bash
# Clone the repository
git clone https://github.com/tsingletaryTT/tt-claw.git ~/tt-claw

# Review structure
cd ~/tt-claw
ls -la

# Expected directories:
# bin/        - Wrapper scripts (openclaw, services, adventure-menu)
# runtime/    - Configuration and data (openclaw.json, agents/, memory/)
# proxy/      - vLLM compatibility proxy (vllm-proxy.py)
# docs/       - Documentation and troubleshooting guides
```

**📦 Clone tt-claw**

### Understanding the Structure

**`bin/openclaw`**
- Portable wrapper script with environment variables
- Sets `OPENCLAW_STATE_DIR` and `OPENCLAW_CONFIG_PATH`
- Works from any directory

**`runtime/openclaw.json`**
- Main configuration file
- Provider definitions (vLLM endpoint)
- Memory search configuration (45+ lessons)
- Agent list (main, chip-quest, terminal-dungeon, conference-chaos)

**`runtime/agents/`**
- Per-agent configuration directories
- System prompts (system.md or SOUL.md)
- Workspace directories
- Memory databases (auto-generated)

**`proxy/vllm-proxy.py`** *(legacy, no longer needed)*
- Originally stripped `strict`, `store`, `prompt_cache_key` for an older image
- All current TT vLLM images accept these fields natively (`extra="allow"`)
- OpenClaw connects directly to port 8000 now

---

## Step 3: Install OpenClaw

If OpenClaw isn't already installed:

```bash
# Download and install OpenClaw v2026.3.2
npm install -g openclaw@2026.3.2

# Verify installation
openclaw --version
# Should show: 2026.3.2

# Create wrapper script (optional)
mkdir -p ~/openclaw
cat > ~/openclaw/openclaw.sh << 'EOF'
#!/bin/bash
# OpenClaw wrapper with tt-claw runtime config
export OPENCLAW_STATE_DIR="$HOME/tt-claw/runtime"
export OPENCLAW_CONFIG_PATH="$HOME/tt-claw/runtime/openclaw.json"
exec openclaw "$@"
EOF
chmod +x ~/openclaw/openclaw.sh
```

**📥 Install OpenClaw**

---

## Step 4: Configure Memory Search

The configuration is already set up in `~/tt-claw/runtime/openclaw.json`, but let's verify the paths exist:

```bash
# Check that indexed documentation exists
ls ~/code/tt-vscode-toolkit/content/lessons/*.md | wc -l
# Should show: 45+ lessons

ls ~/tt-metal/METALIUM_GUIDE.md
# Should exist

ls ~/code/tt-inference-server/README.md
# Should exist
```

**What gets indexed:**

1. **45+ Interactive Lessons** (1.1 MB markdown)
   - Hardware detection and setup
   - Model deployment and optimization
   - Cookbook examples (Game of Life, Mandelbrot, audio)
   - TT-Forge, TT-XLA, TT-Metal frameworks
   - API servers, chat interfaces

2. **TT-Metal Documentation**
   - METALIUM_GUIDE.md - Core framework documentation
   - Release notes and version history
   - Contributing and development guides

3. **TT-Inference-Server Docs**
   - Deployment procedures
   - Model bringup guides
   - Workflow documentation

4. **OpenClaw Journey** (CLAUDE.md)
   - Complete installation story
   - 70B deployment process
   - vLLM compatibility research
   - Troubleshooting database

**Memory search configuration (already in openclaw.json):**

```json
{
  "agents": {
    "defaults": {
      "memorySearch": {
        "extraPaths": [
          "/home/ttuser/code/tt-vscode-toolkit/content/lessons",
          "/home/ttuser/tt-metal/METALIUM_GUIDE.md",
          "/home/ttuser/tt-metal/releases",
          "/home/ttuser/code/tt-inference-server/docs",
          "/home/ttuser/tt-claw/CLAUDE.md"
        ],
        "provider": "local",
        "fallback": "none"
      }
    }
  }
}
```

**How it works:**
- **Local embeddings** - Uses node-llama-cpp (built-in, no external APIs)
- **Vector search** - SQLite with sqlite-vec for fast semantic search
- **Auto-indexing** - Runs on first gateway startup (1-2 minutes)
- **Persistent** - Databases stored in `runtime/memory/*.sqlite`

---

## Step 5: Start OpenClaw Gateway

The gateway is the WebSocket server that manages agents, memory, and LLM interactions.

```bash
# Set environment variables (if not using wrapper)
export OPENCLAW_STATE_DIR="$HOME/tt-claw/runtime"
export OPENCLAW_CONFIG_PATH="$HOME/tt-claw/runtime/openclaw.json"

# Start gateway
cd ~/openclaw
./openclaw.sh gateway run

# Wait for startup message:
# [gateway] listening on ws://127.0.0.1:18789
```

**⚡ Start Gateway**

**First startup takes 1-2 minutes:**
- Downloads local embedding models (~500 MB one-time)
- Indexes all documentation (45+ lessons)
- Creates vector databases (~50 MB)
- Subsequent startups are fast (<10 seconds)

**Watch for these messages:**
```
✓ Loading configuration from ~/tt-claw/runtime/openclaw.json
✓ Indexing documentation: 45 lessons + TT docs
✓ Memory databases created: main.sqlite (50 MB)
✓ Provider "vllm" ready: http://127.0.0.1:8000/v1
✓ Gateway listening on ws://127.0.0.1:18789
```

---

## Step 7: Launch Terminal UI

Now connect to the gateway with the Terminal UI:

```bash
# In a new terminal window
cd ~/openclaw
./openclaw.sh tui

# TUI interface should appear
```

**💬 Launch TUI**

---

## Step 8: Test TT Expert Mode

Try these queries to verify everything works:

### Basic Hardware Question

**You ask:**
```
What is QB2?
```

**Expected response (with citations):**
```
QuietBox 2 (QB2) is TT-QuietBox™ 2, a liquid-cooled, desk-friendly AI
workstation that runs models up to 120 billion parameters locally with a
fully open-source software stack. It features:

- 4x Blackhole ASICs (P300C boards)
- 480 Tensix cores (120 cores per chip)
- 2,654 TFLOPS of compute (663 TFLOPS per chip)
- 200 GB/s memory bandwidth per chip
- 120+ billion parameter model support

It's the industry's first desktop AI workstation built on RISC-V
architecture, designed for AI researchers and developers who want
local inference without cloud dependencies.

Source: ../../code/tt-vscode-toolkit/content/lessons/hardware-detection.md#L45-L62
```

### Deployment Question

**You ask:**
```
How do I deploy vLLM on Tenstorrent?
```

**Expected response:**
```
To deploy vLLM on Tenstorrent hardware:

1. Ensure tt-inference-server is installed:
   cd ~/code/tt-inference-server

2. Run the deployment script with your model and device:
   python3 run.py \
     --model Llama-3.1-8B-Instruct \
     --tt-device n300 \
     --workflow server \
     --docker-server \
     --no-auth

3. Wait for warmup (5-15 minutes for model loading)

4. Test the endpoint:
   curl http://localhost:8000/v1/models

For larger models (70B) on P300X2, add tool calling support:
   --vllm-override-args '{"enable_auto_tool_choice": true, "tool_call_parser": "llama3_json"}'

The deployment process handles Docker containerization, TT-Metal
environment setup, and hardware initialization automatically.

Source: ../../code/tt-vscode-toolkit/content/lessons/api-server.md#L102-L125
```

### Technical Deep Dive

**You ask:**
```
What cookbook examples can I run on my QB2?
```

**Expected response:**
```
The tt-vscode-toolkit includes several cookbook examples you can run:

1. **Game of Life** - Conway's cellular automaton on TT hardware
   - Demonstrates matrix operations on Tensix cores
   - Real-time visualization of compute patterns

2. **Mandelbrot Set** - Fractal generation with parallel compute
   - Shows how to map complex algorithms to TT architecture
   - Achieves significant speedup over CPU

3. **Audio Processing** - Real-time audio filtering and effects
   - Uses TT-Metal for low-latency DSP operations
   - Examples: reverb, EQ, compression

4. **Image Filters** - Convolution and image transformations
   - Demonstrates TT tensor operations on image data
   - Supports various filter kernels (blur, sharpen, edge detect)

5. **Particle Life** - Physics simulation with emergent behavior
   - Many-body particle interactions
   - Beautiful emergent patterns from simple rules

All examples include source code, documentation, and performance
comparisons vs CPU/GPU.

Source: ../../code/tt-vscode-toolkit/content/lessons/cookbook-overview.md#L28-L89
```

---

## Alternative: N150 with 8B Model

If you have an N150 (single chip) instead of QB2, you can run OpenClaw with a smaller model.

### N150 Specifications

- **Chip**: Single Blackhole ASIC
- **Compute**: 663 TFLOPS
- **Memory**: 8 GB GDDR6
- **Model size**: Up to 8B parameters
- **Context**: 8,192 tokens
- **Power**: ~75W (vs ~300W for QB2)

### Deployment for N150

```bash
# Same process, different model and device
cd ~/code/tt-inference-server

python3 run.py \
    --model Llama-3.1-8B-Instruct \
    --tt-device n150 \
    --workflow server \
    --docker-server \
    --dev-mode \
    --host-hf-cache ~/.cache/huggingface \
    --no-auth \
    --override-docker-image ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-dev-ubuntu-22.04-amd64:0.9.0-e867533-22be241 \
    --vllm-override-args '{"enable_auto_tool_choice": true, "tool_call_parser": "llama3_json"}'
```

### Update OpenClaw Configuration

Edit `~/tt-claw/runtime/openclaw.json`:

```json
{
  "models": {
    "providers": {
      "vllm": {
        "models": [{
          "id": "meta-llama/Llama-3.1-8B-Instruct",
          "name": "Llama 3.1 8B Instruct",
          "contextWindow": 8192,
          "maxTokens": 2048
        }]
      }
    }
  },
  "agents": {
    "defaults": {
      "model": {
        "primary": "vllm/meta-llama/Llama-3.1-8B-Instruct"
      }
    }
  }
}
```

### Performance Comparison: 70B vs 8B

| Feature              | QB2 (70B)          | N150 (8B)         |
|----------------------|--------------------|-------------------|
| Response time        | 14+ seconds        | 2-3 seconds       |
| Context window       | 131K tokens        | 8K tokens         |
| Reasoning depth      | Excellent          | Good              |
| Hardware required    | 4x P300C           | 1x N150           |
| Power consumption    | ~300W              | ~75W              |
| Model loading time   | 10-20 minutes      | 2-5 minutes       |
| Best for             | Complex reasoning  | Fast responses    |

**When to use 8B:**
- ✅ Quick Q&A and chat
- ✅ Simple code generation
- ✅ Lower power/noise requirements
- ✅ Faster iteration during development

**When to use 70B:**
- ✅ Complex multi-step reasoning
- ✅ Long-form content generation
- ✅ Advanced code review and refactoring
- ✅ Maximum quality output

---

## Troubleshooting

### Gateway Fails to Start

**Symptom:** Gateway exits with "Failed to connect to provider"

**Solution:**
```bash
# 1. Verify vLLM is running
curl http://localhost:8000/v1/models
# Should return model list

# 2. Check openclaw.json points to port 8000 (not 8001)
grep baseUrl ~/tt-claw/runtime/openclaw.json
# Should show: "baseUrl": "http://127.0.0.1:8000/v1"

# 3. Restart gateway
pkill -f 'openclaw.*gateway'
cd ~/openclaw && ./openclaw.sh gateway run
```

### No Responses from Model

**Symptom:** Gateway starts but queries timeout or return empty

**Solution:**
```bash
# 1. Test vLLM directly
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }'
# Should return completion within 10-20 seconds

# 2. Check if model is still warming up
docker logs $(docker ps -q) | tail -50
# Look for "Readiness file created"

# 3. Check context window setting
grep contextWindow ~/tt-claw/runtime/openclaw.json
# Should be: 131072 (not 16000 or lower)

# 4. Verify gateway logs
tail -100 /tmp/openclaw-*/openclaw-*.log | jq 'select(.level=="error")'
```

### Memory Search Not Working

**Symptom:** Agent responds but doesn't use indexed knowledge

**Solution:**
```bash
# 1. Check memory databases were created
ls -lh ~/tt-claw/runtime/memory/*.sqlite
# Should show 2-3 files, ~50 MB each after first run

# 2. Verify paths exist
ls ~/code/tt-vscode-toolkit/content/lessons/*.md | wc -l
# Should show 45+ files

# 3. Check indexing logs
grep -i "index" /tmp/openclaw-*/openclaw-*.log | head -20
# Should show "Indexed X documents"

# 4. Restart gateway to re-index
pkill -f 'openclaw.*gateway'
rm ~/tt-claw/runtime/memory/*.sqlite  # Force re-index
cd ~/openclaw && ./openclaw.sh gateway run
# Wait 1-2 minutes for indexing
```

### TUI Shows "Connecting..."

**Symptom:** TUI stuck at "Connecting to gateway..."

**Solution:**
```bash
# 1. Verify gateway is running
ps aux | grep 'openclaw.*gateway'

# 2. Check gateway is listening
netstat -tlnp | grep 18789
# Should show: LISTEN on 0.0.0.0:18789

# 3. Check gateway logs for startup errors
tail -50 /tmp/openclaw-*/openclaw-*.log | grep -i error

# 4. Verify token matches (if auth is enabled)
grep '"token":' ~/tt-claw/runtime/openclaw.json
# Should match in both "auth" and "remote" sections

# 5. Try connecting with curl (WebSocket test)
curl -i -N \
  -H "Connection: Upgrade" \
  -H "Upgrade: websocket" \
  http://127.0.0.1:18789/
# Should return: "101 Switching Protocols"
```

### Docker Container Keeps Restarting

**Symptom:** vLLM container crashes during warmup

**Solution:**
```bash
# 1. Check available RAM
free -h
# Need at least 175 GB free for 70B model

# 2. Check for OOM killer
dmesg | grep -i "out of memory"
# If present, system ran out of RAM

# 3. Reset hardware and retry
tt-smi -r
sleep 5
docker stop $(docker ps -q)
docker system prune -f
# Then re-run deployment from Step 1

# 4. Check Docker logs for specific error
docker logs $(docker ps -aq | head -1) | tail -100
```

### Gateway Gets 400 Bad Request from vLLM

**Symptom:** Gateway connects but gets "400 Bad Request" from vLLM

All current TT vLLM images accept unknown fields silently (including `strict`, `store`,
`prompt_cache_key` that OpenClaw sends). If you're seeing 400 errors, the issue is
likely a model name mismatch or malformed payload, not the extra fields.

**Solution:**
```bash
# 1. Test vLLM directly with the exact model name OpenClaw uses
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hi"}],
    "strict": true,
    "store": false
  }'
# Both strict and store are accepted — should return 200 OK

# 2. Verify the model ID in openclaw.json matches what vLLM reports
curl http://localhost:8000/v1/models | python3 -m json.tool
grep '"id"' ~/tt-claw/runtime/openclaw.json
# IDs must match exactly
```

---

## What You've Built

### Complete AI Assistant Stack

```
User Interface Layer:
  ├─ Terminal UI (interactive chat)
  ├─ WebSocket API (port 18789)
  └─ Channel Integrations (optional: Slack, Discord, etc.)

Agent Framework Layer:
  ├─ OpenClaw Gateway (agent management, memory, tools)
  ├─ Memory Search (1,217 chunks from 45+ lessons)
  ├─ Tool Calling (file ops, system commands, memory ops)
  └─ Multi-Agent Support (main + 3 adventure game agents)

LLM Inference Layer:
  ├─ vLLM Server (inference engine, port 8000)
  └─ Model: Llama-3.3-70B-Instruct (128K context, tool calling)

Hardware Layer:
  ├─ TT-Metal Framework (kernel compilation, device management)
  ├─ Firmware (19.4.2.0+ on each chip)
  └─ 4x Blackhole ASICs (480 Tensix cores, 2,654 TFLOPS)
```

### Capabilities

**Ask about Tenstorrent:**
- Hardware specifications and architecture
- Deployment procedures and best practices
- Available lessons and cookbook examples
- Troubleshooting and debugging

**Technical assistance:**
- Code review and optimization
- Model deployment guidance
- System configuration help
- Performance tuning advice

**Creative applications:**
- Adventure games (chip-quest, terminal-dungeon, conference-chaos)
- Custom agent development
- Tool integration
- Channel bots (Slack, Discord, etc.)

### Key Features

✅ **Fully Local** - No data leaves your machine
✅ **Hardware Accelerated** - 70B model on QB2's Blackhole chips
✅ **Self-Documenting** - Knows all 45+ lessons by heart
✅ **Extensible** - Build custom agents and skills
✅ **Production Ready** - Used for real demos and development
✅ **Open Source** - All code and configs available

---

## Next Steps

### Explore Built-in Agents

OpenClaw includes 3 adventure game agents as examples:

```bash
# Launch adventure menu
cd ~/tt-claw
./bin/adventure-menu

# Or use TUI to switch agents:
# 1. In TUI, type: /agent chip-quest
# 2. Say: "start the adventure"
# 3. Enjoy educational adventure inside a TT chip!
```

### Build Custom Agents

Create your own agents by:
1. Adding agent definition to `openclaw.json`
2. Creating agent directory in `runtime/agents/`
3. Writing system prompt (SOUL.md or system.md)
4. Testing with TUI

See `runtime/agents/chip-quest/` for example structure.

### Integrate with Channels

Connect OpenClaw to external services:
- Slack workspace bot
- Discord server bot
- Telegram bot
- WhatsApp integration
- SMS gateway

See OpenClaw docs: https://openclaw.io/docs/channels

### Optimize Performance

**For faster responses:**
- Use 8B model on N150 (2-3 seconds vs 14+ seconds)
- Reduce max_tokens for shorter outputs
- Pre-warm specific query patterns
- Use batching for multiple requests

**For better quality:**
- Increase temperature for creative responses
- Use longer context window for complex queries
- Add more documentation to memory search
- Fine-tune system prompts for specific tasks

---

## Resources

**Documentation:**
- tt-claw README: `~/tt-claw/README.md`
- Architecture guide: `~/tt-claw/docs/ARCHITECTURE.md`
- Complete journey: `~/tt-claw/CLAUDE.md`
- OpenClaw docs: https://openclaw.io/docs

**Community:**
- Tenstorrent Discord: https://discord.gg/tenstorrent
- tt-inference-server: https://github.com/tenstorrent/tt-inference-server
- OpenClaw GitHub: https://github.com/OpenClawIO/openclaw

**Support:**
- Hardware Detection: [Hardware Detection](command:tenstorrent.showLesson?["hardware-detection"])
- Model Deployment: [Deploy Models](command:tenstorrent.showLesson?["model-deployment"])

---

## Summary

You've deployed a complete AI assistant on your QuietBox 2:

- ✅ **70B model running** on 4x Blackhole chips with tool calling
- ✅ **Memory search** indexing 45+ lessons and TT documentation
- ✅ **OpenClaw gateway** managing agents and LLM interactions
- ✅ **Direct vLLM connection** — no proxy needed, all API fields accepted natively
- ✅ **Terminal UI** for interactive queries

**Test it now:**
```bash
cd ~/openclaw && ./openclaw.sh tui
```

Ask: "What is QB2?" or "How do I deploy vLLM?" and watch it respond with citations!

Your QuietBox 2 is now an expert on Tenstorrent hardware, deployment, and programming. 🎉
