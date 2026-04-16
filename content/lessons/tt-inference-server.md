---
id: tt-inference-server
title: Production Inference with tt-inference-server
description: >-
  Deploy Llama-3.1-8B on any Tenstorrent hardware in minutes — N150, N300, T3K,
  P100, p300c, or QB2. tt-inference-server automates Docker image selection,
  model download, and server startup with a single command. OpenAI-compatible
  API ready immediately.
category: serving
tags:
  - production
  - deployment
  - inference
  - llama
  - docker
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
  - p100
estimatedMinutes: 20
minTTMetalVersion: v0.65.1
recommended_metal_version: v0.65.1
validationDate: 2026-04-15
validationNotes: >-
  Rewritten for v0.12.0 Docker images; --tt-device auto-detection; Llama-3.1-8B
  validated Complete on WH (N150/N300/T3K) and Experimental on BH (P100/p300c).
---

# Production Inference with tt-inference-server

[tt-inference-server](https://github.com/tenstorrent/tt-inference-server) is
Tenstorrent's official workflow automation tool. Give it a model name and your
hardware type and it handles everything: pulls the right Docker image (pre-built
tt-metal + vLLM), downloads model weights, and starts an OpenAI-compatible
inference server.

> **QB2 / p300c users:** Llama-3.1-8B is supported on P100/P150 hardware
> (🛠️ Experimental status). Use `--tt-device p100` for p300c or QB2.

---

## Prerequisites

### Install tt-inference-server

**QB2 / pre-configured images:** tt-inference-server is pre-installed at
`~/.local/lib/tt-inference-server`. Skip to the next section.

**All other hardware (N150/N300/T3K/P100/P150):** Clone it:

```bash
git clone https://github.com/tenstorrent/tt-inference-server.git \
  ~/.local/lib/tt-inference-server
```

Verify:
```bash
ls ~/.local/lib/tt-inference-server/run.py
```

---

### Other Prerequisites

- **HF token** — Llama is gated on HuggingFace. Set once:
  ```bash
  export HF_TOKEN=hf_...          # your HuggingFace access token
  ```
- **Docker** — the server runs in a container. Verify:
  ```bash
  docker --version
  ```
- **Hardware detected** — confirm your card is visible:

[▶ Detect Hardware](command:tenstorrent.runHardwareDetection)

[▶ Verify Prerequisites](command:tenstorrent.verifyInferenceServerPrereqs)

---

## The Model: Llama-3.1-8B

Llama-3.1-8B is the widest-coverage model in tt-inference-server — it runs on
every current Tenstorrent board:

| Hardware | Device flag | Status | Max context |
|----------|-------------|--------|-------------|
| N150 | `--tt-device n150` | 🟢 Complete | 64 K |
| N300 | `--tt-device n300` | 🟢 Complete | 128 K |
| T3K (WH QuietBox/LoudBox) | `--tt-device t3k` | 🟢 Complete | 128 K |
| P100 / p300c / QB2 | `--tt-device p100` | 🛠️ Experimental | 64 K |
| P150 | `--tt-device p150` | 🛠️ Experimental | 64 K |
| Galaxy | `--tt-device galaxy` | 🟢 Complete | — |

Two weight variants are available via MODEL_SPECS:
- `Llama-3.1-8B` — base model (default)
- `Llama-3.1-8B-Instruct` — instruction-tuned (use for chat)

---

## Start the Server

### Option A — Automated (run.py)

`run.py` selects the correct Docker image for your hardware, handles the
volume, and downloads weights inside the container on first run.

`--tt-device` can be omitted and will be **auto-detected** from your hardware
via `tt-smi`:

```bash
cd ~/.local/lib/tt-inference-server

python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --workflow server \
  --docker-server \
  --no-auth
```

Add `--tt-device <device>` if auto-detection doesn't match your hardware.

---

### Hardware-specific commands

#### N150 (Wormhole — single chip, 64 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth
```

[▶ Start Server (N150)](command:tenstorrent.startTtInferenceServerN150)

---

#### N300 (Wormhole — dual chip, 128 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n300 \
  --workflow server \
  --docker-server \
  --no-auth
```

[▶ Start Server (N300)](command:tenstorrent.startTtInferenceServerN300)

---

#### T3K — WH QuietBox / LoudBox (8 chips, 128 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device t3k \
  --workflow server \
  --docker-server \
  --no-auth
```

---

#### P100 / p300c / QB2 (Blackhole — 64 K context, Experimental)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device p100 \
  --workflow server \
  --docker-server \
  --no-auth
```

> QB2 exposes each p300c chip as an independent `p100` device. Run one server
> per chip, each on a different `--service-port`, or use the T3K-class
> configurations when available on future firmware.

---

### Option B — Direct docker run

For full transparency, or when you want to run without `run.py`, use the
container directly. Pass `--model` and `--tt-device` as container args; the
container resolves the config from its bundled model spec catalog.

**Wormhole (N150 / N300 / T3K):**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-25305db-6e67d2d \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150
```

Change `--tt-device` to `n300` or `t3k` for those boards — same image.

**Blackhole (P100 / p300c / QB2):**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-55fd115-aa4ae1e \
  --model Llama-3.1-8B-Instruct \
  --tt-device p100
```

Use `--print-docker-cmd` with `run.py` to see the exact command it would run:

```bash
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow server --docker-server --no-auth --print-docker-cmd
```

---

## First Run

**⏱️ First run:** 10–20 minutes — Docker image pull (~10 GB) + model weight download (~16 GB) inside the container.

**⏱️ Subsequent runs:** 2–5 minutes — image and weights are cached.

Watch for this in run.py output when the container is up:

```
INFO: Created Docker container ID: 6b8c7038a44a
INFO: Access container logs via: docker logs -f 6b8c7038a44a
INFO: Stop running container via: docker stop 6b8c7038a44a
```

Then watch the container logs until vLLM is ready:

```bash
docker logs -f 6b8c7038a44a
```

Look for:

```
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

---

## Test the Server

Once vLLM is ready, the server exposes a standard OpenAI-compatible API.

**Quick test:**

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.1-8B-Instruct",
       "prompt": "Tenstorrent accelerators are designed for",
       "max_tokens": 60}'
```

[▶ Test Server](command:tenstorrent.testTtInferenceServerSimple)

**Streaming (token-by-token):**

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Llama-3.1-8B-Instruct",
       "prompt": "Write a haiku about silicon:",
       "max_tokens": 40,
       "stream": true}'
```

[▶ Test Streaming](command:tenstorrent.testTtInferenceServerStreaming)

**Python client (OpenAI SDK):**

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

response = client.chat.completions.create(
    model="Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What is a Tenstorrent accelerator?"}],
    max_tokens=100,
)
print(response.choices[0].message.content)
```

[▶ Create Python Client](command:tenstorrent.createTtInferenceServerClient)

---

## Managing the Running Server

```bash
# List running containers
docker ps

# Follow logs in real time
docker logs -f <container-id>

# Stop server
docker stop <container-id>

# Stop all tt-inference-server containers at once
docker ps --filter ancestor=ghcr.io/tenstorrent/tt-inference-server \
  --format '{{.ID}}' | xargs docker stop
```

---

## Beyond the Server: Other Workflows

With the container already running, client-side workflows can run against it
without restarting:

```bash
cd ~/.local/lib/tt-inference-server

# Quick smoke test (reduced samples)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow benchmarks --limit-samples-mode smoke-test

# Full accuracy evals
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow evals

# Benchmarks + evals + reports in one pass (release certification)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow release --docker-server --no-auth
```

Results land in `~/.local/lib/tt-inference-server/workflow_logs/`.

---

## Tuning vLLM Arguments

tt-inference-server's model specs set reasonable defaults for each model/device
pair (block size 64, full context window, 32 concurrent sequences). Override
any of them without rebuilding the container.

### Tool Use / Function Calling

Enable the OpenAI tool-calling API by passing two flags to vLLM:

**Via `run.py`:**

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth \
  --vllm-override-args '{"enable-auto-tool-choice": true, "tool-call-parser": "llama3_json"}'
```

**Via direct `docker run`** (remaining args pass straight through to `vllm serve`):

```bash
docker run ... \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.12.0-25305db-6e67d2d \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

**Parser by model family:**

| Model family | `tool-call-parser` |
|-------------|---------------------|
| Llama 3.x | `llama3_json` |
| Qwen / Hermes-format | `hermes` |
| Mistral | `mistral` |

**Current limitation:** `tool_choice="none"` and `tool_choice="required"` are
not yet supported in the TT vLLM fork. Only `tool_choice="auto"` works reliably.

Once the server is running with tool choice enabled, use the API normally:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")

tools = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather in a city",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}]

response = client.chat.completions.create(
    model="Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "What's the weather in Austin?"}],
    tools=tools,
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls)
```

---

### Reducing Context Length

By default the server uses the full context window supported by the hardware
(64 K on N150/P100, 128 K on N300/T3K). Reducing it lowers DRAM usage and
can speed up model load:

```bash
# Via run.py
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth \
  --vllm-override-args '{"max-model-len": 8192}'

# Via docker run (passthrough)
... <image> --model Llama-3.1-8B-Instruct --tt-device n150 --max-model-len 8192
```

`max-model-len` must be a multiple of `block-size` (default 64). Values like
4096, 8192, 16384, 32768 all work cleanly.

---

### Concurrency and Batch Limits

The model spec sets `max-num-seqs` (concurrent in-flight sequences) and
`max-num-batched-tokens` (tokens per forward pass). Lower them to reduce
peak memory or raise them when throughput matters more than latency:

```bash
# Reduce to 8 concurrent users (lower memory, lower throughput)
--vllm-override-args '{"max-num-seqs": 8}'

# Increase for high-throughput batch workloads (N300/T3K only — needs headroom)
--vllm-override-args '{"max-num-seqs": 64, "max-num-batched-tokens": 65536}'
```

Defaults for Llama-3.1-8B: `max-num-seqs=32`, `max-num-batched-tokens=65536`
(N150) / `131072` (N300/T3K).

---

### Combining Multiple Overrides

JSON keys are merged, so all overrides can go in one `--vllm-override-args`:

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth \
  --vllm-override-args '{
    "enable-auto-tool-choice": true,
    "tool-call-parser": "llama3_json",
    "max-model-len": 16384,
    "max-num-seqs": 8
  }'
```

Use `--print-docker-cmd` to verify the generated docker command before
launching:

```bash
python3 run.py ... --print-docker-cmd
```

---

## Models Outside MODEL_SPECS

The container resolves model configuration from a bundled `model_spec.json`
with 60+ validated models. If your model isn't there, you have three options:

### 1. Check by short name

The container resolves both full HF repo IDs and short names (the last segment
of the path). So `Llama-3.1-8B-Instruct` and
`meta-llama/Llama-3.1-8B-Instruct` both work:

```bash
# These are equivalent
... --model Llama-3.1-8B-Instruct --tt-device n150
... --model meta-llama/Llama-3.1-8B-Instruct --tt-device n150
```

The full catalog includes Llama, Qwen, Mistral, Gemma, DeepSeek, Whisper,
Stable Diffusion, FLUX, Mochi video, and more — see
[model support docs](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/README.md).

### 2. Run vLLM directly (no MODEL_SPECS constraint)

The [vLLM Production lesson](command:tenstorrent.showLesson?["vllm-production"])
shows how to run vLLM directly on the host without tt-inference-server. This
accepts any model path or HF repo and gives you full control over every vLLM
flag — useful for models in development or private repos.

### 3. Request official support

Open an issue at
[github.com/tenstorrent/tt-inference-server](https://github.com/tenstorrent/tt-inference-server/issues)
to request a new model be added to MODEL_SPECS. Include your hardware type,
model name, and any performance requirements.

---

## Non-Container Deployment (--local-server)

If you have a built tt-metal checkout (e.g. via the Build tt-metal lesson),
you can run vLLM directly on the host — no Docker required:

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --local-server \
  --tt-metal-home /opt/tt-metal \
  --host-hf-cache       # reuse your existing HF cache
```

`--local-server` uses `REPO_ROOT/persistent_volume/` for logs and caches, and
runs as the invoking user (no Docker volume permissions to manage).

---

## HF Cache Tips

If you've already downloaded model weights (e.g. via `hf download`), point
`run.py` at them to skip the in-container download:

```bash
# Reuse ~/.cache/huggingface (bare flag uses HF_HOME / ~/.cache/huggingface)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow server --docker-server --no-auth \
  --host-hf-cache

# Or point at a specific directory
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow server --docker-server --no-auth \
  --host-weights-dir ~/models/meta-llama/Llama-3.1-8B-Instruct
```

---

## Cache Persistence

tt-inference-server uses two separate cache directories inside the container —
knowing how each is stored makes the difference between a 2-minute startup and
a 10-minute one.

### What gets cached where

```
cache_root/
  weights/{model_name}/                              # HF model weights
  tt_metal_cache/cache_{model_name}/{device_type}/   # compiled TT Metal kernels
  tt_dit_cache/                                      # compiled WAN/Mochi tensor weights
  logs/                                              # vLLM server logs
```

- **TT Metal kernels** (`tt_metal_cache/`) — compiled by vLLM on first run.
  Subsequent starts load from this cache: ~2–5 min instead of 10–20 min.
- **Media model tensor weights** (`tt_dit_cache/`) — compiled by video/image
  models (WAN 2.2, Mochi). **Not cached by default** — stored in `/tmp/TT_DIT_CACHE`
  inside the container and lost when the container stops.

---

### Docker named volumes (default — TT Metal kernels)

By default `run.py` mounts a Docker named volume at `cache_root`. The TT Metal
kernel cache survives container restarts automatically — no extra flags needed.

You can verify the volume exists after the first run:

```bash
docker volume ls | grep Llama
```

---

### Persisting media model caches (`TT_DIT_CACHE_DIR`)

For video and image models (WAN 2.2, Mochi, FLUX) the container compiles tensor
weights at startup and stores them in `TT_DIT_CACHE_DIR`. The default is
`/tmp/TT_DIT_CACHE`, which is lost when the container stops.

**First run without cache:** ~525 seconds (WAN 2.2 on QB2)
**Subsequent runs with cache:** ~5 minutes

Move the cache under `cache_root` so it lives in the persistent Docker volume:

```bash
# In your .env file (or export before running)
TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache
```

With `run.py`:

```bash
TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache \
python3 run.py --model Wan-2.2-T2V-1.3B --tt-device p100 \
  --workflow server --docker-server --no-auth
```

With direct `docker run`, pass it as `-e`:

```bash
docker run \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache" \
  --volume volume_id_Wan-2.2:/home/container_app_user/cache_root \
  ... <image> --model Wan-2.2-T2V-1.3B --tt-device p100
```

---

### Full host-side persistence (`--host-volume`)

To survive **Docker image updates** (which create new named volumes), bind the
entire `cache_root` to a host directory. All weights and all caches land on the
host filesystem:

```bash
# Ensure the host directory is writable by UID 1000 (container user)
sudo mkdir -p ~/tt-cache
sudo chown 1000 ~/tt-cache

python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth \
  --host-volume ~/tt-cache
```

With `--host-volume`, `TT_DIT_CACHE_DIR` should still be set explicitly to keep
it within the bound directory:

```bash
TT_DIT_CACHE_DIR=~/tt-cache/tt_dit_cache \
python3 run.py --model Wan-2.2-T2V-1.3B --tt-device p100 \
  --workflow server --docker-server --no-auth \
  --host-volume ~/tt-cache
```

---

### Skip HF hub checks at startup (`HF_HUB_OFFLINE`)

After weights are downloaded, the HF library still pings the hub at startup to
check for updates. Disable this to cut several seconds off every startup:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python3 run.py --model Llama-3.1-8B-Instruct --tt-device n150 \
  --workflow server --docker-server --no-auth
```

Or add to your `.env`:

```bash
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

With direct `docker run`:

```bash
docker run \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "HF_HUB_OFFLINE=1" \
  -e "TRANSFORMERS_OFFLINE=1" \
  ... <image> --model Llama-3.1-8B-Instruct --tt-device n150
```

> **Note:** Set these only after the model is fully downloaded. With
> `HF_HUB_OFFLINE=1` the container cannot fetch weights that aren't cached.

---

### Sharing caches between `--docker-server` and `--local-server`

`--docker-server` writes caches as UID 1000 (the container user).
`--local-server` writes as your host user. If you switch between them without
fixing ownership, the other mode can't read the cache.

Fix ownership before switching:

```bash
# Switching from --docker-server → --local-server
sudo chown -R $USER ~/tt-cache

# Switching from --local-server → --docker-server
sudo chown -R 1000 ~/tt-cache
```

When using `--local-server`, caches land in `REPO_ROOT/persistent_volume/`
unless overridden. Point `TT_DIT_CACHE_DIR` at a shared path if you want
`--local-server` to reuse what Docker compiled:

```bash
TT_DIT_CACHE_DIR=~/tt-cache/tt_dit_cache \
python3 run.py --model Wan-2.2-T2V-1.3B --tt-device p100 \
  --workflow server --local-server \
  --tt-metal-home /opt/tt-metal
```

---

## Dev Branch — What's Coming (as of 2026-04-15)

The `dev` branch of tt-inference-server (VERSION 0.12.0) contains work that is
not yet in a release. Here's a snapshot of what's active there:

### C++ inference server with IPC

A new `tt-cppserver` backend provides a C++ server communicating with the vLLM
Python layer via IPC, replacing the single-process Python server. Includes a
`MemoryManager` that tracks device DRAM allocation and enables explicit cache
management.

### Session manager

A persistent session manager that maintains model state across requests —
useful for multi-turn chat without re-loading the model between conversations.

### Disaggregated prefill / decode

Separate prefill and decode stages can now run on different devices or pods,
enabling higher throughput at scale. This matches the architecture used in
large-scale deployments.

### Grafana metrics dashboard

Container exposes Prometheus-compatible metrics at `/metrics`; a bundled
Grafana dashboard visualises throughput, latency, and token rates. Connect
Grafana at `http://localhost:3000` after starting with the metrics profile.

### OpenAI `/v1/responses` endpoint

The server now implements the newer OpenAI Responses API (in addition to
`/v1/completions` and `/v1/chat/completions`), allowing compatibility with the
latest OpenAI SDK streaming patterns.

### Multi-host deployment

`--multihost` flag and companion documentation (`docs/multihost_deployment.md`)
support deploying prefill and decode on separate machines connected via
high-speed fabric.

> These features are under active development. Pin a dev branch commit if you
> want to experiment — `run.py` accepts `--override-docker-image` to use a
> custom build.

---

## Next Steps

- [vLLM Production →](command:tenstorrent.showLesson?["vllm-production"]) — run vLLM directly without the workflow wrapper
- [VSCode Chat →](command:tenstorrent.showLesson?["vscode-chat"]) — connect the inference server to the VSCode @tenstorrent chat participant
- [tt-inference-server docs](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/workflows_user_guide.md) — full CLI reference
