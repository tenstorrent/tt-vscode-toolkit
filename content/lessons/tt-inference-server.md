---
id: tt-inference-server
title: Production Inference with tt-inference-server
description: >-
  Use Tenstorrent's official inference server for production deployments with
  simple CLI configuration.
category: advanced
tags:
  - production
  - deployment
  - inference
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
status: blocked
estimatedMinutes: 30
---

# Production Inference with tt-inference-server

Learn to use tt-inference-server, Tenstorrent's official workflow automation tool for deploying vLLM inference servers with hardware-aware configuration.

## What is tt-inference-server?

tt-inference-server is Tenstorrent's official **workflow automation tool** that simplifies running vLLM inference servers on Tenstorrent hardware. It's NOT a standalone server - it's a smart wrapper that:

**Key capabilities:**
- ✅ **Automated vLLM deployment** - Starts vLLM servers in Docker with correct configs
- ✅ **Hardware-aware** - Automatically configures for N150/N300/T3K/Galaxy
- ✅ **Model-specific** - Each model has validated configuration (from MODEL_SPECS)
- ✅ **Multiple workflows** - server, benchmarks, evals, reports, release
- ✅ **Official support** - Maintained by Tenstorrent, tested with each release
- ✅ **Simple CLI** - One command with model name and device type

**Important:** tt-inference-server is a workflow runner that wraps vLLM. When you use `--workflow server`, it starts a vLLM server in Docker and exits, leaving the server running.

## Where Does tt-inference-server Fit?

Let's see how it compares to what you've learned so far:

| Approach | What It Does | When to Use |
|----------|--------------|-------------|
| **Direct API (Lesson 4-5)** | Manual Python with Generator API | Learning, custom logic, prototyping |
| **Custom Flask (Lesson 5)** | Your Flask server wrapping Generator | Custom applications, full control |
| **tt-inference-server (This lesson)** | Automated vLLM deployment | Quick production setup with validated configs |
| **vLLM directly (Lesson 7)** | Manual vLLM installation/config | Custom vLLM deployments, advanced tuning |

**tt-inference-server is ideal when:**
- You want production-ready serving without manual configuration
- You want Tenstorrent-validated model configurations
- You want Docker-based deployment
- You don't need custom inference logic

## Prerequisites

Before starting, ensure you have:

1. **tt-inference-server installed** (included with tt-installer 2.0)
   ```bash
   which tt-inference-server
   # or check if run.py exists
   ls ~/.local/lib/tt-inference-server/run.py
```

   **If not found:** Install with tt-installer (see Setup Information in welcome page)

2. **Docker installed** (required for --docker-server)
   ```bash
   docker --version
```

   **Expected:** Docker 20.10+ or Podman 3.0+

3. **Llama model available** (will be downloaded if not present)
   - tt-inference-server can auto-download from HuggingFace
   - Requires HF_TOKEN for gated models

4. **Hardware detected** (from Lesson 1)
   ```bash
   tt-smi
```

   **Expected:** Your Tenstorrent device shown (N150, N300, T3K, etc.)

[✅ Verify Prerequisites](command:tenstorrent.verifyInferenceServerPrereqs)

---

## Step 1: Understand the Command Format

tt-inference-server uses this command structure:

```bash
tt-inference-server --model <model-name> --device <device> --workflow <workflow> [options]
```

**Required arguments:**
- `--model` - Model NAME (e.g., "Llama-3.1-8B-Instruct"), NOT a path
- `--device` - Hardware type: `n150`, `n300`, `t3k`, `galaxy` (lowercase!)
- `--workflow` - What to run: `server`, `benchmarks`, `evals`, `reports`, `release`

**Common options:**
- `--docker-server` - Run vLLM in Docker container (recommended)
- `--local-server` - Run vLLM on localhost (not yet implemented)
- `--service-port PORT` - Service port (default: 8000)
- `--dev-mode` - Mount local code into container for development

**Example:**
```bash
# Start vLLM server for Llama 3.1 8B on N150
tt-inference-server \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server
```

**What this does:**
1. Looks up Llama-3.1-8B-Instruct in MODEL_SPECS
2. Gets hardware-specific configuration for N150
3. Downloads model weights if not present
4. Starts vLLM server in Docker container
5. Exits, leaving server running in background

---

## Step 2: Start Your First Server

Let's start a vLLM server for Llama 3.1 8B on your hardware.

**Quick Check:** Not sure which hardware you have?

[🔍 Detect Hardware](command:tenstorrent.runHardwareDetection)

---

**Choose your hardware configuration:**

<details open style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N150 (Wormhole - Single Chip)</b></summary>

```bash
cd ~/tt-inference-server  # or wherever it's installed
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server
```

**Configuration:** Optimized for single-chip development and testing

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N300 (Wormhole - Dual Chip)</b></summary>

```bash
cd ~/tt-inference-server
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n300 \
  --workflow server \
  --docker-server
```

**Configuration:** Tensor parallelism across 2 chips for higher throughput

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 T3K (Wormhole - 8 Chips)</b></summary>

```bash
cd ~/tt-inference-server
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device t3k \
  --workflow server \
  --docker-server
```

**Configuration:** Production-scale deployment across 8 chips

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 Galaxy (32 Chips)</b></summary>

```bash
cd ~/tt-inference-server
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device galaxy \
  --workflow server \
  --docker-server
```

**Configuration:** Data center scale with 32-chip mesh

</details>

---

**Expected output:**
```text
============================================================
tt-inference-server run.py CLI args summary
============================================================

Model Options:
  model:                      Llama-3.1-8B-Instruct
  device:                     n150
  workflow:                   server

Starting inference server...
Created Docker container ID: abc123def456
Access container logs via: docker logs -f abc123def456
Stop running container via: docker stop abc123def456
```

**⏱️ First run:** 5-15 minutes (downloads Docker image, downloads model weights)
**⏱️ Subsequent runs:** 2-5 minutes (uses cached image and model)

[🚀 Start tt-inference-server](command:tenstorrent.startTtInferenceServer)

**Note:** The command exits after starting the server, but the Docker container keeps running in the background.

---

## Step 3: Check Container Status

After starting, verify the container is running:

```bash
# List running containers
docker ps

# Expected output
CONTAINER ID   IMAGE                          STATUS         PORTS
abc123def456   ghcr.io/tenstorrent/vllm...   Up 2 minutes   0.0.0.0:8000->8000/tcp
```

**View server logs:**
```bash
docker logs -f abc123def456  # Replace with your container ID
```

**Expected in logs:**
```yaml
INFO: Loading model Llama-3.1-8B-Instruct...
INFO: Initializing Tenstorrent device (N150)...
INFO: Model loaded successfully
INFO: Starting vLLM server on port 8000
INFO: Server ready to handle requests
```

---

## Step 4: Test the Server

The server provides an OpenAI-compatible API. Test with curl:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "prompt": "Explain what a Tenstorrent AI accelerator is in one sentence.",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Expected response:**
```json
{
  "id": "cmpl-abc123",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Llama-3.1-8B-Instruct",
  "choices": [
    {
      "text": "A Tenstorrent AI accelerator is a specialized hardware chip designed to efficiently run deep learning workloads with high performance and energy efficiency.",
      "index": 0,
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 28,
    "total_tokens": 40
  }
}
```

[🧪 Test Server (Simple)](command:tenstorrent.testTtInferenceServerSimple)

---

## Step 5: Understand Command-Line Options

tt-inference-server provides many options. Here are the most important:

### Required Arguments

```bash
--model MODEL_NAME          # Model name from MODEL_SPECS
                            # Examples: "Llama-3.1-8B-Instruct", "Qwen2.5-7B-Instruct"

--device DEVICE_TYPE        # Hardware type (lowercase!)
                            # Options: n150, n300, t3k, galaxy

--workflow WORKFLOW         # What to run
                            # Options: server, benchmarks, evals, reports, release
```

### Optional Arguments

**Server Deployment:**
```bash
--docker-server             # Run in Docker container (recommended)
--local-server              # Run on localhost (not yet implemented)
--service-port PORT         # Service port (default: 8000)
--dev-mode                  # Mount local files into container for development
```

**Docker Configuration:**
```bash
--override-docker-image IMG # Use custom Docker image instead of default
-it, --interactive          # Run Docker in interactive mode
```

**Advanced:**
```bash
--device-id IDS             # Specific device IDs (e.g., "0,1,2,3")
--disable-trace-capture     # Skip trace capture for faster startup
--override-tt-config JSON   # Override TT config as JSON
--vllm-override-args JSON   # Override vLLM arguments as JSON
```

**For debugging:**
```bash
--reset-venvs               # Remove .workflow_venvs/ if dependencies broken
--skip-system-sw-validation # Skip tt-smi/tt-topology verification
```

### Example Commands

**N150 basic server:**
```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow server \
  --docker-server
```

**N300 with custom port:**
```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n300 \
  --workflow server \
  --docker-server \
  --service-port 8001
```

**T3K with specific devices:**
```bash
python3 run.py \
  --model Llama-3.3-70B-Instruct \
  --device t3k \
  --workflow server \
  --docker-server \
  --device-id 0,1,2,3,4,5,6,7
```

[⚙️ Start with N150 Config](command:tenstorrent.startTtInferenceServerN150)

[⚙️ Start with N300 Config](command:tenstorrent.startTtInferenceServerN300)

---

## Step 6: Advanced Testing

### Test with Streaming

Request streaming responses (token-by-token):

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "prompt": "Write a haiku about AI acceleration:",
    "max_tokens": 100,
    "stream": true
  }'
```

**Expected:** Server-Sent Events stream

```text
data: {"choices":[{"text":"Silicon","finish_reason":null}]}

data: {"choices":[{"text":" minds","finish_reason":null}]}

data: {"choices":[{"text":" awakening","finish_reason":null}]}
...
data: [DONE]
```

[🌊 Test Streaming](command:tenstorrent.testTtInferenceServerStreaming)

### Test with Different Sampling

**High temperature (creative):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "prompt": "Once upon a time",
    "max_tokens": 50,
    "temperature": 1.2,
    "top_p": 0.95
  }'
```

**Low temperature (deterministic):**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.1-8B-Instruct",
    "prompt": "The capital of France is",
    "max_tokens": 10,
    "temperature": 0.1
  }'
```

[🎲 Test Sampling Parameters](command:tenstorrent.testTtInferenceServerSampling)

### Test with Python Client

Create a Python client using the OpenAI SDK:

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"  # Not used, but required by SDK
)

# Generate text
response = client.completions.create(
    model="Llama-3.1-8B-Instruct",
    prompt="Explain quantum computing to a 5-year-old:",
    max_tokens=100,
    temperature=0.8
)

print(response.choices[0].text)
```

[📝 Create Python Client](command:tenstorrent.createTtInferenceServerClient)

---

## Step 7: Explore Other Workflows

tt-inference-server supports multiple workflows beyond `server`:

### Benchmarks Workflow

Run performance benchmarks on your model:

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow benchmarks \
  --docker-server
```

**What this does:**
1. Starts vLLM server in Docker
2. Runs benchmark client (random prompts, various lengths)
3. Measures throughput, latency, tokens/sec
4. Saves results to `workflow_logs/benchmarks_output/`

### Evals Workflow

Run accuracy evaluations:

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow evals \
  --docker-server
```

**What this does:**
1. Starts vLLM server in Docker
2. Runs evaluation tasks (MMLU, HellaSwag, etc.)
3. Scores model accuracy
4. Saves results to `workflow_logs/evals_output/`

### Release Workflow

Run full validation (benchmarks + evals + reports):

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --device n150 \
  --workflow release \
  --docker-server
```

**Use this before deploying a new model to production!**

---

## Understanding MODEL_SPECS

tt-inference-server uses MODEL_SPECS to define validated model configurations.

**When you run:**
```bash
python3 run.py --model Llama-3.1-8B-Instruct --device n150 --workflow server --docker-server
```

**tt-inference-server looks up:**
- Docker image: `ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.3.0-9b67e09-a91b644`
- tt-metal commit: `9b67e09`
- vLLM commit: `a91b644`
- Hardware config for N150: max context, max concurrency, etc.
- HuggingFace repo: `meta-llama/Llama-3.1-8B-Instruct`

**Supported models (see vendor/tt-inference-server/README.md):**
- Llama 3.1 8B (n150, n300, t3k)
- Llama 3.1 70B (t3k, galaxy)
- Llama 3.2 1B/3B/11B/90B Vision
- Qwen 2.5 7B/72B
- Mistral 7B
- And many more...

Each model has a validated configuration tested by Tenstorrent.

---

## Managing Running Servers

### List Running Containers

```bash
docker ps
```

**Output:**
```text
CONTAINER ID   IMAGE                          STATUS         PORTS
abc123def456   ghcr.io/tenstorrent/vllm...   Up 10 minutes  0.0.0.0:8000->8000/tcp
```

### View Logs

```bash
docker logs -f abc123def456  # Follow logs in real-time
docker logs abc123def456      # View all logs
```

### Stop Server

```bash
docker stop abc123def456
```

### Remove Container

```bash
docker rm abc123def456
```

### Clean Up All

```bash
# Stop all tt-inference-server containers
docker ps | grep vllm-tt-metal | awk '{print $1}' | xargs docker stop

# Remove all stopped containers
docker container prune
```

---

## Comparison: tt-inference-server vs Other Approaches

| Feature | Direct API | Flask Server | tt-inference-server | vLLM Manual |
|---------|------------|--------------|---------------------|-------------|
| **Setup complexity** | Low | Medium | Low | High |
| **Code required** | ✅ Yes | ✅ Yes | ❌ No | ⚠️ Minimal |
| **Configuration** | Manual | Manual | Automated | Manual |
| **Docker deployment** | ❌ Manual | ❌ Manual | ✅ Built-in | ⚠️ Manual |
| **Validated configs** | ❌ No | ❌ No | ✅ Yes | ❌ No |
| **Official support** | ✅ Yes | ❌ No | ✅ Yes | ✅ Yes |
| **Customization** | ✅ Full | ✅ Full | ⚠️ Limited | ✅ Full |
| **OpenAI API compat** | ❌ No | ❌ No | ✅ Yes (vLLM) | ✅ Yes |
| **Best for** | Learning | Prototyping | Quick production | Custom deployment |

**Choose tt-inference-server when:**
- ✅ You want Tenstorrent-validated configurations
- ✅ You want quick production deployment
- ✅ You want Docker-based deployment
- ✅ You don't need custom inference logic

**Choose vLLM directly (next lesson) when:**
- ✅ You need custom vLLM configuration
- ✅ You want to understand vLLM internals
- ✅ You're deploying outside Docker
- ✅ You need maximum control

---

## Troubleshooting

### "tt-inference-server: command not found"

**Problem:** Server not installed.

**Solution:** Install with tt-installer:
```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

### "Cannot find model module 'Llama-3.1-8B-Instruct'"

**Problem:** Model name not in MODEL_SPECS.

**Solution:** Check supported models:
```bash
# List all supported models
python3 run.py --help | grep -A 100 "Available models"
```

Or check vendor/tt-inference-server/README.md for full model list.

### "Docker daemon not running"

**Problem:** Docker service not started.

**Solution:**
```bash
sudo systemctl start docker
# or
sudo service docker start
```

### "Failed to detect Tenstorrent device"

**Problem:** Hardware not detected.

**Solution:**
```bash
# Check hardware
tt-smi

# Reset device if needed
tt-smi -r

# Skip validation (for debugging)
python3 run.py ... --skip-system-sw-validation
```

### "Port 8000 already in use"

**Problem:** Another service using port 8000.

**Solution:** Use different port:
```bash
python3 run.py ... --service-port 8001
```

### Container starts but requests timeout

**Problem:** Model not fully loaded yet.

**Solution:** Wait for server to be fully ready:
```bash
# Watch logs until you see "Server ready"
docker logs -f <container-id>

# Health check
curl http://localhost:8000/health
```

### "out of memory" in container

**Problem:** Model too large for hardware.

**Solution:** Use smaller model:
- N150: Llama 3.1 8B, Llama 3.2 1B/3B
- N300: Llama 3.1 8B, Qwen 2.5 7B
- T3K: Llama 3.1 70B, Qwen 2.5 72B

---

## Log Files and Debugging

tt-inference-server creates organized logs:

```text
~/tt-inference-server/workflow_logs/
├── run_logs/                  # Main run logs
│   └── run_2025-01-15_10-30-00_Llama-3.1-8B-Instruct_n150_server.log
├── docker_server/             # Docker container logs
│   └── vllm_2025-01-15_10-30-00_Llama-3.1-8B-Instruct_n150_server.log
├── benchmarks_output/         # Benchmark results
├── evals_output/              # Evaluation results
└── run_specs/                 # Model spec JSON files
```

**Useful for debugging:**
```bash
# Latest run log
ls -lt ~/tt-inference-server/workflow_logs/run_logs/ | head -2

# View specific run
tail -f ~/tt-inference-server/workflow_logs/run_logs/run_*.log
```

---

## What You Learned

- ✅ What tt-inference-server is (workflow automation tool for vLLM)
- ✅ How to start a vLLM server with validated configuration
- ✅ Understanding --model (name), --device (lowercase), --workflow
- ✅ Testing with OpenAI-compatible API
- ✅ Managing Docker containers
- ✅ Using different workflows (server, benchmarks, evals, release)
- ✅ Understanding MODEL_SPECS and validated configurations

**Key insight:** tt-inference-server is NOT a standalone server - it's a smart wrapper that deploys vLLM with Tenstorrent-validated configurations. This saves you from manual Docker image selection, model configuration, and hardware optimization.

**Next step:** Learn vLLM directly for maximum control and customization beyond what tt-inference-server provides.

Continue to Lesson 7: Production Inference with vLLM!

---

## Learn More

**Documentation:**
- tt-inference-server repo: [github.com/tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server)
- Workflows User Guide: `vendor/tt-inference-server/docs/workflows_user_guide.md`
- MODEL_SPECS: `vendor/tt-inference-server/workflows/model_spec.py`

**Community:**
- Discord: [discord.gg/tenstorrent](https://discord.gg/tenstorrent)
- GitHub Issues: [github.com/tenstorrent/tt-inference-server/issues](https://github.com/tenstorrent/tt-inference-server/issues)

**Related Lessons:**
- Lesson 4-5: Direct API (custom inference logic)
- Lesson 7: vLLM Production (manual vLLM deployment)
- Lesson 8: VSCode Chat (using vLLM server)
