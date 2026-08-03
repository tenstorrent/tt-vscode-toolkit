---
id: tt-inference-server
title: Production Inference with TT-Inference-Server
description: >-
  Deploy Llama-3.1-8B on any Tenstorrent hardware in minutes — n150, n300, T3000,
  p150, p300, or TT-QuietBox 2 (p300x2). TT-Inference-Server automates Docker image
  selection, model download, and server startup with a single command.
  OpenAI-compatible API ready immediately.
category: serving
tags:
  - production
  - deployment
  - inference
  - llama
  - docker
  - p300x2
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - p300x2
  - galaxy
status: validated
validatedOn:
  - n150
  - p100
estimatedMinutes: 20
minTTMetalVersion: v0.65.1
recommended_metal_version: v0.65.1
validationDate: 2026-08-03
validationNotes: >-
  Reviewed against tt-inference-server v0.19.0. --tt-device is the documented flag
  (--device is a hidden legacy alias). TT-QuietBox 2 is a first-class device,
  --tt-device p300x2 ("BH QuietBox 2", 4 Blackhole chips in a 2x2 mesh); the
  Llama-3.1-8B / -Instruct spec for P300X2 is status FUNCTIONAL and carries an
  upstream *nightly* CI entry, which is the basis for the p300x2 entry in
  validatedOn (upstream CI evidence, not local hardware validation).
  setup.sh, --workflow reports, --workflow tests, ~/models, and the
  vllm-tt-metal-llama3/ path are all gone upstream.
---

# Production Inference with TT-Inference-Server

[TT-Inference-Server](https://github.com/tenstorrent/tt-inference-server) is
Tenstorrent's official workflow automation tool. Give it a model name and your
hardware type and it handles everything: pulls the right Docker image (pre-built
TT-Metalium<sup>™</sup> + vLLM), downloads model weights, and starts an OpenAI-compatible
inference server.

> **TT-QuietBox<sup>®</sup> 2 users:** TT-QuietBox 2 is a **first-class device** — pass
> `--tt-device p300x2`. Upstream names it `P300X2` ("BH QuietBox 2"): 2× P300 cards =
> 4 Blackhole<sup>®</sup> chips wired as a 2×2 mesh. Do **not** pretend it is a `p100`;
> that would give you one chip out of four. Lead model is `Llama-3.1-8B`
> (🟡 FUNCTIONAL, 32 concurrent sequences, 131072-token context).

> **Flag name matters: `--tt-device`, not `--device`.** `--tt-device` is the documented
> flag. `--device` still works as a *hidden legacy alias* in `run.py`
> (`help=argparse.SUPPRESS`), but it is unadvertised and easy to confuse with Docker's
> own `--device`. See the collision warning in the direct-`docker run` section below.

---

## Prerequisites

### Install TT-Inference-Server

**TT-QuietBox 2 / pre-configured images:** TT-Inference-Server is pre-installed at
`~/.local/lib/tt-inference-server`. Skip to the next section.

**All other hardware (n150/n300/T3000/p100/p150/p300):** Clone it:

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
  export HF_TOKEN=hf_...          # your HuggingFace access token — never commit this
  ```
  `run.py` requires `HF_TOKEN` for server workflows and `--docker-server`. If it is
  unset, `run.py` prompts for it interactively (via `getpass`) and writes it to
  `<repo_root>/.env`. Six client-only workflows are exempt.
- **`pyyaml` in your *host* `python3`** — undocumented but real: `run.py` does a bare
  `import yaml` at module import time, before any venv bootstrapping. This is fine on a
  stock Ubuntu 24.04 desktop, but it bites minimal containers and slim base images:
  ```bash
  python3 -c 'import yaml; print("pyyaml OK")'   # if this fails, run.py cannot start
  ```
- **Docker** — the server runs in a container. Verify:
  ```bash
  docker --version
  ```
- **Hardware detected** — confirm your card is visible:

[▶ Detect Hardware](command:tenstorrent.runHardwareDetection)

[▶ Verify Prerequisites](command:tenstorrent.verifyInferenceServerPrereqs)

---

## Reading Model Status

Every model/device pair in the catalog carries a **status**. Learn the four values
before you pick a model — they are your single best predictor of whether a run will
just work:

| Status | Badge | What it means in practice |
|--------|-------|---------------------------|
| `EXPERIMENTAL` | 🛠️ | Brings up and generates, but expect rough edges. Often concurrency 1. |
| `FUNCTIONAL` | 🟡 | Correct output at real concurrency. Not perf-tuned; may carry known eval issues. |
| `COMPLETE` | 🟢 | Functional **and** accuracy-validated against reference scores. |
| `TOP_PERF` | 🚀 | Complete **and** hitting published performance targets. |

Don't read these as marketing tiers — read them as "how much has actually been
measured." A 🟡 model at concurrency 32 is a perfectly good production starting point;
a 🛠️ model at concurrency 1 is a demo.

**CI coverage is a separate axis, and it is uneven.** Checking
`.github/workflows/models-ci-config.json` upstream:

- **`p300x2` (TT-QuietBox 2) has real nightly CI** — including `Llama-3.1-8B-Instruct`,
  plus Qwen3.6-27B (which also gates the `release` workflow), gemma-4-31B-it,
  gemma-4-12B-it, gpt-oss-120b, Llama-3.3-70B-Instruct, FLUX ×2, Motif, Wan2.2 ×2,
  Z-Image, and speecht5.
- **Single-card `p300` and `p100` are barely covered.** Across the entire catalog,
  `Llama-3.1-8B-Instruct` is the *only* model with a `P300` (nightly) or `P100`
  (weekly) CI entry. Everything else on those two devices is untested by CI. Treat
  p100/p300 as bring-up targets, not validated ones.

---

## The Model: Llama-3.1-8B

Llama-3.1-8B is the widest-coverage model in TT-Inference-Server — it runs on
every current Tenstorrent board:

| Hardware | Device flag | Status | Concurrency | Max context |
|----------|-------------|--------|-------------|-------------|
| n150 | `--tt-device n150` | 🟢 COMPLETE | 32 | 64 K |
| n300 | `--tt-device n300` | 🟢 COMPLETE | 32 | 128 K |
| T3000 (WH TT-QuietBox/LoudBox) | `--tt-device t3k` | 🟢 COMPLETE | 32 | 128 K |
| **TT-QuietBox 2** | **`--tt-device p300x2`** | 🟡 FUNCTIONAL | 32 | 128 K |
| p300 (single card, 2 dies) | `--tt-device p300` | 🟡 FUNCTIONAL | 32 | 128 K |
| p100 | `--tt-device p100` | 🛠️ EXPERIMENTAL | 32 | 64 K |
| p150 | `--tt-device p150` | 🛠️ EXPERIMENTAL | 32 | 64 K |
| BH 4×P150 | `--tt-device p150x4` | 🟢 COMPLETE | 128 | 128 K |
| BH LoudBox (8×P150) | `--tt-device p150x8` | 🟡 FUNCTIONAL | 256 | 128 K |
| Galaxy | `--tt-device galaxy` | 🟡 FUNCTIONAL | 128 | 128 K |

> **Firmware / KMD floor on p300 and p300x2.** The Blackhole 4-chip specs declare
> *strict* system requirements: firmware `>= 19.2.0` and `tt-kmd >= 2.5.0`. `run.py`
> validates these before launching and will refuse to start on older stacks — check
> with `tt-smi -s` rather than guessing.

### The `-Instruct` variants (valid, but undocumented upstream)

Two weight variants exist for this model:

- `Llama-3.1-8B` — base model
- `Llama-3.1-8B-Instruct` — instruction-tuned (**use this for chat**)

Upstream expresses these as a two-entry `weights:` list on a *single* YAML spec
template, and the generated documentation only emits a page for the base name. So
`Llama-3.1-8B-Instruct` looks undocumented — but it is fully valid:

```bash
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow server --docker-server
```

Both short names and full HF repo IDs resolve, so
`meta-llama/Llama-3.1-8B-Instruct` works identically.

---

## Small-Model Starting Points (n150)

⚠️ **`Qwen3-0.6B` is not servable through TT-Inference-Server.** It does not exist in
the model spec catalog at all — it appears only as a row in a *theoretical benchmark
target* reference table, which is not a servable spec. `--model Qwen3-0.6B` will fail
to resolve. The smallest text Qwen served by vLLM here is **Qwen3-8B**; the smallest
Qwen of any kind is Qwen3-4B on the **forge** engine (🛠️, context 2048, concurrency 1).

(Qwen3-0.6B remains an excellent choice on the *native* vLLM / TT-Metalium<sup>™</sup>
path — see the vLLM Production lesson. It just isn't reachable from this one.)

For a genuinely small first model on n150, use one of these instead:

| Model | Status | Concurrency | Max context |
|-------|--------|-------------|-------------|
| `Llama-3.2-1B-Instruct` | 🟡 FUNCTIONAL | 32 | 128 K |
| `gemma-3-1b-it` | 🛠️ EXPERIMENTAL | 32 | 32 K |
| `Mistral-7B-Instruct-v0.3` | 🟢 COMPLETE | 32 | 32 K |
| `Llama-3.1-8B-Instruct` | 🟢 COMPLETE | 32 | 64 K |

Note that `Llama-3.2-1B-Instruct` is the leaner option but is only 🟡, while the 8B
Llama is 🟢 — on n150 the bigger model is actually the better-validated one.

**Llama 4 is not in this catalog at all.** Don't reach for it here.

---

## What Else Runs on TT-QuietBox 2 (`p300x2`)

Once Llama-3.1-8B is working, this is the rest of the LLM catalog for `p300x2`. All of
it is served with the same command shape — only `--model` changes:

| `--model` | Status | Concurrency | Max context |
|-----------|--------|-------------|-------------|
| `--model` | Status | Concurrency | Max context | Implementation |
|-----------|--------|-------------|-------------|----------------|
| `Llama-3.1-8B` / `-Instruct` | 🟡 FUNCTIONAL | 32 | 128 K | `tt-transformers` |
| `Llama-3.3-70B-Instruct` | 🟡 FUNCTIONAL | 32 | 128 K | `tt-transformers` |
| `Llama-3.1-70B` / `-Instruct` | 🟡 FUNCTIONAL | 32 | 128 K | `tt-transformers` |
| `DeepSeek-R1-Distill-Llama-70B` | 🟡 FUNCTIONAL | 32 | 128 K | `tt-transformers` |
| `Qwen3-32B` | 🟡 FUNCTIONAL | 32 | 128 K | `tt-transformers` |
| `Qwen3.6-27B` | 🛠️ EXPERIMENTAL | 1 | 256 K | `qwen36-blackhole` |
| `gpt-oss-120b` | 🛠️ EXPERIMENTAL | 1 | 128 K | `gpt-oss` |
| `gemma-4-31B-it` | 🛠️ EXPERIMENTAL | 1 | 48 K | `tt-transformers` |

Note the shape of that list. Every 🟡 `tt-transformers` model — 8B through 70B — serves
**32 concurrent sequences at 128 K context**, so stepping up in size costs you latency
and memory, not batching. The 🛠️ entries are a different animal: concurrency 1, single
custom implementations, and in gemma-4's case a deliberately reduced 48 K context to fit
Blackhole DRAM. Start with Llama-3.1-8B, then step up within the 🟡 group before you
touch the 🛠️ ones.

Media models also run here — `Wan2.2-T2V-A14B-Diffusers` and `mochi-1-preview`
(both 🟢 COMPLETE on `p300x2`), plus FLUX, Motif, and Z-Image, which is why
TT-QuietBox 2 carries the broadest nightly CI of any Blackhole configuration.

> **Generated per-model docs lag the catalog.** Upstream auto-generates a page per
> model/device pair (e.g. `docs/model_support/llm/Llama-3.1-8B_p300x2.md`), but the
> generator trails the YAML — as of v0.19.0 there is no page for `Llama-3.1-70B` or
> `DeepSeek-R1-Distill-Llama-70B` on `p300x2` even though both are in the catalog and
> will run. Absence of a doc page is not absence of support.

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

#### n150 (Wormhole<sup>™</sup> — single chip, 64 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --workflow server \
  --docker-server \
  --no-auth
```

[▶ Start Server (n150)](command:tenstorrent.startTtInferenceServerN150)

---

#### n300 (Wormhole — dual chip, 128 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device n300 \
  --workflow server \
  --docker-server \
  --no-auth
```

[▶ Start Server (n300)](command:tenstorrent.startTtInferenceServerN300)

---

#### T3000 — WH TT-QuietBox / LoudBox (8 chips, 128 K context)

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device t3k \
  --workflow server \
  --docker-server \
  --no-auth
```

---

#### TT-QuietBox 2 (Blackhole<sup>®</sup> — `p300x2`, 4 chips, 128 K context)

This is the canonical TT-QuietBox 2 command:

```bash
python3 run.py \
  --model Llama-3.1-8B \
  --tt-device p300x2 \
  --workflow server \
  --docker-server
```

Add `--no-auth` for a quick local run without tokens, or `-Instruct` for the
chat-tuned weights:

```bash
python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device p300x2 \
  --workflow server \
  --docker-server \
  --no-auth
```

Upstream registers this device as `P300X2` — *"2x P300 cards = 4 chips (2,2 mesh)"* —
and its product string is literally **"BH QuietBox 2"**. All four chips are driven as
one mesh by a single server process. There is no "one server per chip" arrangement, and
`--tt-device p100` is **not** a substitute: it would bind a single chip and leave three
idle.

---

#### Single-card Blackhole (`p300`, `p150`, `p100`)

```bash
# p300 — one P300 card (2 dies), 🟡 FUNCTIONAL
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300 \
  --workflow server --docker-server --no-auth

# p150 / p100 — single Blackhole chip, 🛠️ EXPERIMENTAL
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p150 \
  --workflow server --docker-server --no-auth
```

Remember the CI reality from the status section: on `p300` and `p100`,
`Llama-3.1-8B-Instruct` is the only model anything upstream regularly tests. If you
need a Blackhole path you can lean on, it's TT-QuietBox 2 (`p300x2`).

---

### Never Hand-Set `MESH_DEVICE` or `TT_MESH_GRAPH_DESC_PATH`

You may have seen these environment variables in TT-Metalium or tt-train
documentation. **On the TT-Inference-Server path, do not set them.** `run.py` derives
`MESH_DEVICE`, `ARCH_NAME`, `WH_ARCH_YAML`, and (where needed)
`TT_MESH_GRAPH_DESC_PATH` from `--tt-device` plus the model spec. Anything you export
by hand either gets overridden or, worse, wins and silently mis-maps the fabric.

This matters most on TT-QuietBox 2, where the correct `MESH_DEVICE` is **per-model** —
there is no single right answer to hardcode:

```
  Llama-3.1-8B on p300x2      ->  MESH_DEVICE = P300x2
  Qwen3.6-27B  on p300x2      ->  MESH_DEVICE = (1, 4)
  gpt-oss-120b on p300x2      ->  MESH_DEVICE = (1, 4)
  gemma-4-31B-it on p300x2    ->  MESH_DEVICE = P150x4   <-- deliberately not P300x2
```

That last one is the cautionary tale. Upstream found that on `gemma-4-31B-it` the
custom `p300_x2` mesh graph descriptor laid the tensor-parallel collectives over the
**wrong fabric links and corrupted decode logits** — the model produced plausible-looking
but wrong output rather than crashing. The fix was to select the `p150_x4` descriptor by
setting `MESH_DEVICE: P150x4` in that model's spec. The spec knows this; you don't.

Note the variable is `ARCH_NAME` here (`blackhole` on P-series), **not**
`TT_METAL_ARCH_NAME` as used on the bare TT-Metalium path. Another reason to let the
tool set it.

---

### Option B — Direct docker run

For full transparency, or when you want to run without `run.py`, use the
container directly. Pass `--model` and `--tt-device` as container args; the
container resolves the config from its bundled model spec catalog.

> **⚠️ `--device` vs `--tt-device` on the same command line**
>
> This is the single easiest way to break a hand-written `docker run`. The two flags
> look alike and mean completely different things:
>
> ```
> ╔══════════════════════════════════════════════════════════════
> ║  docker run --device /dev/tenstorrent ...  <image>  --tt-device p300x2
> ║             ^^^^^^^^                                ^^^^^^^^^^^
> ║             Docker's flag.                          Container argument.
> ║             Passes the TT device node               Tells the server which
> ║             through to the container.               mesh topology to build.
> ║             Consumed by Docker,                     Consumed by the app,
> ║             BEFORE the image name.                  AFTER the image name.
> ╚══════════════════════════════════════════════════════════════
> ```
>
> Everything before the image name belongs to Docker; everything after it is passed to
> the server process inside. Putting `--tt-device` before the image makes Docker reject
> it; putting `--device` after the image makes the server reject it.
>
> `run.py` also accepts a bare `--device` as a **hidden legacy alias** for
> `--tt-device`. It is suppressed from `--help` and reconciled internally. Don't use it —
> when you later paste a line into a `docker run` context, `--device` will mean the
> *other* thing.

**Wormhole (n150 / n300 / T3000):**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.9.0-25305db-6e67d2d \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tt-device n150
```

Change `--tt-device` to `n300` or `t3k` for those boards — same image.

**TT-QuietBox 2 (`p300x2`):**

```bash
docker run \
  --env "HF_TOKEN=$HF_TOKEN" \
  --ipc host \
  --publish 8000:8000 \
  --device /dev/tenstorrent \
  --mount type=bind,src=/dev/hugepages-1G,dst=/dev/hugepages-1G \
  --volume volume_id_Llama-3.1-8B:/home/container_app_user/cache_root \
  ghcr.io/tenstorrent/tt-inference-server/vllm-tt-metal-src-release-ubuntu-22.04-amd64:0.19.0-b204341-9bd099c \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tt-device p300x2
```

> **Don't memorise image tags.** There is no single pinned vLLM version any more —
> every model/device spec pins its own, and the tag encodes them as
> `<spec-version>-<tt_metal_commit>-<vllm_commit>`. The two tags above differ because
> the n150 and p300x2 specs for the same model are pinned to different commits. Tags
> also drift with every release.

The reliable way to get a correct `docker run` line is to ask `run.py` to print the one
it would have used — it fills in the right image, volume name, env vars, and device
mounts for your exact model and hardware:

```bash
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow server --docker-server --no-auth --print-docker-cmd
```

`--print-docker-cmd` prints and exits without starting anything, so it's safe to run
any time.

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

The examples in this section assume you started with `--no-auth`, which is why
`api_key` can be a throwaway string. For an authenticated server, see
[Authentication](#authentication-two-schemes) below.

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

## Authentication: Two Schemes

There is no single API key. TT-Inference-Server now runs **two different auth schemes**
depending on which engine serves your model, and mixing them up produces a confusing
401 against a server that is otherwise perfectly healthy.

### 1. vLLM engine — HS256 JWT derived from `JWT_SECRET`

Text/LLM models served by vLLM expect a **signed JWT**, not a plain secret. The server
derives the expected token by HS256-signing a fixed payload with your `JWT_SECRET`;
your client must present the same signed token as a bearer.

Key resolution order on the client side:

```
  VLLM_API_KEY   ->  used literally, if set
  JWT_SECRET     ->  used to *derive* the HS256 JWT
  (neither)      ->  no authorization header sent
```

Generate the bearer token and call the server (needs `pyjwt`):

```bash
export JWT_SECRET="my-secret-string"

export BEARER_TOKEN=$(python -c 'import os, json, jwt; print(jwt.encode({"team_id": "tenstorrent", "token_id": "debug-test"}, os.getenv("JWT_SECRET"), algorithm="HS256"))')

curl -s --no-buffer -X POST "http://0.0.0.0:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $BEARER_TOKEN" \
  -d '{"model":"meta-llama/Llama-3.1-8B-Instruct",
       "messages":[{"role":"user","content":"What is Tenstorrent?"}],
       "max_tokens":256}'
```

The payload `{"team_id": "tenstorrent", "token_id": "debug-test"}` is not a placeholder
you may edit — it is the exact payload the server signs, so it must match byte for byte.

With the OpenAI SDK, pass the *derived token* (not the secret) as `api_key`:

```python
import os, json, jwt
from openai import OpenAI

bearer = jwt.encode(
    {"team_id": "tenstorrent", "token_id": "debug-test"},
    os.environ["JWT_SECRET"],
    algorithm="HS256",
)
client = OpenAI(base_url="http://localhost:8000/v1", api_key=bearer)
```

### 2. Media / Forge engines — literal shared secret `API_KEY`

Image, video, audio and Forge-served models take a **plain shared secret**, passed
through verbatim:

```bash
export API_KEY="choose-your-own-secret"     # do not ship the upstream default
curl -H "Authorization: Bearer $API_KEY" http://localhost:8000/...
```

Upstream's default value for this is the literal string `your-secret-key`. That is a
development placeholder — always override it on anything reachable from a network.

### 3. `--no-auth` — turn both off

```bash
python3 run.py --model Llama-3.1-8B --tt-device p300x2 \
  --workflow server --docker-server --no-auth
```

`--no-auth` disables authorization for both schemes and removes the `JWT_SECRET`
requirement entirely. Convenient on a workstation you trust; never on a shared host.

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

`--workflow` is required, and only these ten values are accepted:

```
  benchmarks   evals        stress_tests   server        release
  spec_tests   agentic      agentic_traces serving_bench prefill_decode
```

> **`reports` and `tests` no longer exist.** Both were removed upstream. Reports are now
> **emitted automatically at the end of every workflow**, so there is nothing to invoke;
> and `tests` was folded into `spec_tests`. If you find `--workflow reports` or
> `--workflow tests` in a guide — including some pages of tt-inference-server's *own*
> docs, which are stale on this point — that guide predates the change and `run.py` will
> reject the value outright.

With the container already running, client-side workflows can run against it
without restarting:

```bash
cd ~/.local/lib/tt-inference-server

# Quick smoke test (reduced sample count — fastest way to prove the loop works)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow benchmarks --limit-samples-mode smoke-test

# Full accuracy evals
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow evals

# Benchmarks + evals in one pass, report written at the end (release certification)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow release --docker-server --no-auth
```

Results land in `~/.local/lib/tt-inference-server/workflow_logs/`.

### Choosing a benchmark client (`--tools`)

The `benchmarks` workflow can drive the server with any of four load generators:

```bash
--tools vllm      # default — vLLM's own benchmark_serving.py
--tools genai     # genai-perf (Triton SDK)
--tools aiperf    # AIPerf
--tools guidellm  # GuideLLM
```

All four are ordinary HTTP clients pointed at the OpenAI-compatible endpoint, so the
choice affects only how load is shaped and reported — not how the model runs.

### Benchmarking a server you didn't start (`--server-url`)

If a server is already running — on this machine or another one — skip
`--docker-server` / `--local-server` entirely and point the client workflows at it:

```bash
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow benchmarks --server-url http://192.168.1.50 --service-port 8000
```

`--server-url` is mutually exclusive with `--docker-server` and `--local-server`; it
means "don't launch anything, just talk to this."

---

## Tuning vLLM Arguments

TT-Inference-Server's model specs set reasonable defaults for each model/device
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
  <image>  \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --tt-device n150 \
  --enable-auto-tool-choice \
  --tool-call-parser llama3_json
```

(Get `<image>` from `--print-docker-cmd` — see the tag note above.)

**Parser by model family:**

| Model family | `tool-call-parser` |
|-------------|---------------------|
| Llama 3.x | `llama3_json` |
| Qwen / Hermes-format | `hermes` |
| Mistral | `mistral` |

**Current limitation:** `tool_choice="none"` and `tool_choice="required"` are
not yet supported by the Tenstorrent vLLM plugin. Only `tool_choice="auto"` works
reliably.

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

By default the server uses the full context window declared in the model spec for your
device (for Llama-3.1-8B: 64 K on n150/p100/p150, 128 K on n300/T3000/p300/**p300x2**).
Reducing it lowers DRAM usage and can speed up model load:

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

# Increase for high-throughput batch workloads (n300/T3000 only — needs headroom)
--vllm-override-args '{"max-num-seqs": 64, "max-num-batched-tokens": 65536}'
```

Defaults for Llama-3.1-8B: `max-num-seqs=32` on every single-host device including
`p300x2`; `max-num-batched-tokens=65536` (n150) / `131072`
(n300/T3000/p300/p300x2). The larger multi-card meshes raise concurrency instead —
128 on `p150x4` and Galaxy, 256 on `p150x8`.

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

## Models Outside the Spec Catalog

Model configuration comes from YAML spec catalogs in the repo
(`workflows/model_specs/prod/` — `llm.yaml`, `image.yaml`, `video.yaml`, `vlm.yaml`,
`cnn.yaml`, `embedding.yaml`, `audio_tts.yaml`, `training.yaml`), which `run.py`
resolves and hands to the container. **These YAML files are the source of truth** —
more current than any generated documentation page. If your model isn't in them, you
have three options:

### 1. Check by short name

Both full HF repo IDs and short names (the last path segment) resolve. So
`Llama-3.1-8B-Instruct` and `meta-llama/Llama-3.1-8B-Instruct` both work:

```bash
# These are equivalent
... --model Llama-3.1-8B-Instruct --tt-device p300x2
... --model meta-llama/Llama-3.1-8B-Instruct --tt-device p300x2
```

The full catalog includes Llama, Qwen, Mistral, Gemma, DeepSeek, gpt-oss, Whisper,
Stable Diffusion, FLUX, Wan 2.2 and Mochi video, and more — see
[model support docs](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/README.md),
and the generated per-device pages alongside it (e.g.
[`Llama-3.1-8B_p300x2.md`](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/model_support/llm/Llama-3.1-8B_p300x2.md)).
When a doc page and the YAML disagree, believe the YAML.

### 2. Run vLLM directly (no spec-catalog constraint)

The [vLLM Production lesson](command:tenstorrent.showLesson?["vllm-production"])
shows how to run vLLM directly on the host without TT-Inference-Server. This
accepts any model path or HF repo and gives you full control over every vLLM
flag — useful for models in development or private repos.

### 3. Request official support

Open an issue at
[TT-Inference-Server issue tracker](https://github.com/tenstorrent/tt-inference-server/issues)
to request a new model be added to the spec catalog. Include your hardware type,
model name, and any performance requirements.

---

## Non-Container Deployment (--local-server)

If you have a built TT-Metalium checkout (e.g. via the Build TT-Metalium lesson),
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
# Reuse ~/.cache/huggingface (bare flag resolves HOST_HF_HOME, then HF_HOME,
# then ~/.cache/huggingface)
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow server --docker-server --no-auth \
  --host-hf-cache

# Or point at a specific directory of pre-downloaded weights
python3 run.py --model Llama-3.1-8B-Instruct --tt-device p300x2 \
  --workflow server --docker-server --no-auth \
  --host-weights-dir ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct
```

Download weights ahead of time with the `hf` CLI (the modern client — the old
`huggingface-cli` name is gone from tt-inference-server entirely):

```bash
hf auth login
hf download meta-llama/Llama-3.1-8B-Instruct --exclude 'original/**'
```

### ⚠️ There is no `~/models` any more

Older guides told you to stage weights in `~/models`. That convention is **dead** —
there are zero references to it anywhere in tt-inference-server. The layout now is:

```
╔══════════════════════════════════════════════════════════
║  In-container cache root
║    /home/container_app_user/cache_root
║
║  Host side (--local-server, or --host-volume with no path)
║    <repo_root>/persistent_volume/
╚══════════════════════════════════════════════════════════
```

You choose how the host provides storage with exactly **one** of three mutually
exclusive flags — `run.py` errors out if you pass more than one:

| Flag | What it does | Mount mode |
|------|--------------|-----------|
| `--host-volume [PATH]` | Binds the whole `cache_root` to the host. Bare flag → `<repo_root>/persistent_volume/` | read-write |
| `--host-hf-cache [PATH]` | Reuses an existing HF cache for weights only | read-only |
| `--host-weights-dir PATH` | Reuses a specific pre-downloaded weights directory | read-only |

Passing none of them (the default for `--docker-server`) uses a Docker **named
volume**, which needs no host permission setup at all. The relevant weights env var
inside the vLLM container is `MODEL_WEIGHTS_DIR`; `MODEL_WEIGHTS_PATH` is
media-engine-only.

---

## Cache Persistence

TT-Inference-Server uses two separate cache directories inside the container —
knowing how each is stored makes the difference between a 2-minute startup and
a 10-minute one.

### What gets cached where

```
cache_root/
  weights/{model_name}/                              # HF model weights
  tt_metal_cache/cache_{model_name}/{device_type}/   # compiled TT-Metalium kernels
  tt_dit_cache/                                      # compiled WAN/Mochi tensor weights
  logs/                                              # vLLM server logs
```

- **TT-Metalium kernels** (`tt_metal_cache/`) — compiled by vLLM on first run.
  Subsequent starts load from this cache: ~2–5 min instead of 10–20 min.
- **Media model tensor weights** (`tt_dit_cache/`) — compiled by video/image
  models (WAN 2.2, Mochi). **Not cached by default** — stored in `/tmp/TT_DIT_CACHE`
  inside the container and lost when the container stops.

---

### Docker named volumes (default — TT-Metalium kernels)

By default `run.py` mounts a Docker named volume at `cache_root`. The TT-Metalium
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

**First run without cache:** ~525 seconds (WAN 2.2 on TT-QuietBox 2)
**Subsequent runs with cache:** ~5 minutes

Move the cache under `cache_root` so it lives in the persistent Docker volume:

```bash
# In your .env file (or export before running)
TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache
```

With `run.py`:

```bash
TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache \
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --tt-device p300x2 \
  --workflow server --docker-server --no-auth
```

With direct `docker run`, pass it as `-e`:

```bash
docker run \
  -e "HF_TOKEN=$HF_TOKEN" \
  -e "TT_DIT_CACHE_DIR=/home/container_app_user/cache_root/tt_dit_cache" \
  --volume volume_id_Wan2.2-T2V-A14B-Diffusers:/home/container_app_user/cache_root \
  ... <image> --model Wan2.2-T2V-A14B-Diffusers --tt-device p300x2
```

---

### Full host-side persistence (`--host-volume`)

To survive **Docker image updates** (which create new named volumes), bind the
entire `cache_root` to a host directory. All weights and all caches land on the
host filesystem:

> **⚠️ Breaking change in v0.11.0 — you now own the permissions**
>
> Before v0.11.0 the vLLM image had a root entrypoint (`docker-entrypoint.sh` + `gosu`)
> that could `chown` the mounted volume for you on startup. **That entrypoint is gone** —
> the image now execs Python directly as a non-root user with no root-level step at all.
>
> Consequence: with `--host-volume`, the host directory must **already** be writable by
> the container's UID before you launch. Nothing inside the container will fix it, and
> the failure surfaces as permission errors partway through weight download rather than
> as a clean startup error. Do this yourself, once:
>
> ```bash
> sudo mkdir -p ~/tt-cache
> sudo chown 1000 ~/tt-cache
> ```
>
> `--image-user` controls the UID passed to `docker run --user` and **defaults to
> `1000`**, matching the UID that default release images are built with. Change it only
> if you built a custom image with a different UID — and if you do, `chown` to *that*
> UID instead. A mismatch between `--image-user` and the image's actual UID is its own
> class of permission failure.
>
> `--host-hf-cache` and `--host-weights-dir` are mounted **read-only**, so they need
> only read access and no `chown`. The default named-volume path needs nothing at all.
> `run.py` also refuses to run a pre-0.11 vLLM image, since the old image cannot speak
> the new argument contract.

```bash
# Ensure the host directory is writable by UID 1000 (the container user)
sudo mkdir -p ~/tt-cache
sudo chown 1000 ~/tt-cache

python3 run.py \
  --model Llama-3.1-8B-Instruct \
  --tt-device p300x2 \
  --workflow server \
  --docker-server \
  --no-auth \
  --host-volume ~/tt-cache
```

With `--host-volume`, `TT_DIT_CACHE_DIR` should still be set explicitly to keep
it within the bound directory:

```bash
TT_DIT_CACHE_DIR=~/tt-cache/tt_dit_cache \
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --tt-device p300x2 \
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
python3 run.py --model Wan2.2-T2V-A14B-Diffusers --tt-device p300x2 \
  --workflow server --local-server \
  --tt-metal-home /opt/tt-metal
```

---

## What Changed Recently (reviewed against v0.19.0, 2026-08-03)

> **Note:** The `dev` branch has been retired. All active development lands directly on
> **`main`** before release, so clone `main` for the latest features.

Things worth knowing if you last used this tool a few releases ago. Each of these has
already been folded into the sections above — this is the short list of what would
otherwise silently break a command you had memorised:

| Change | What to do differently |
|--------|------------------------|
| `--device` → **`--tt-device`** | Use `--tt-device`. The old name is a hidden alias and collides with Docker's `--device`. |
| **`setup.sh` deleted** (Nov 2025) | Gone and unsupported. There is no setup script — `run.py` handles host setup itself. |
| **`--workflow reports` / `tests` removed** | Reports are emitted automatically at the end of every workflow; `tests` became `spec_tests`. |
| **`~/models` retired** | Cache root is `/home/container_app_user/cache_root`; host side is `persistent_volume/`. Use `--host-volume` / `--host-hf-cache` / `--host-weights-dir`. |
| **`vllm-tt-metal-llama3/` renamed** (Mar 2026) | The in-repo path is now **`vllm-tt-metal/`**. Old links 404. |
| **v0.11.0 entrypoint change** | No more root entrypoint — you `sudo chown 1000` your `--host-volume` path yourself. Pre-0.11 images are refused. |
| **No single pinned vLLM version** | Each model/device spec pins its own tt-metal and vLLM commits; the image tag encodes them. Read tags off `--print-docker-cmd`. |
| **TT-QuietBox 2 is `p300x2`** | A first-class 4-chip device, not four `p100`s. |
| **`huggingface-cli` gone** | The repo uses the `hf` CLI throughout. |

### Also newer, and genuinely useful

- **`--print-docker-cmd`** — print the exact `docker run` line and exit without starting
  anything. The best way to learn what `run.py` actually does.
- **`--no-auth`** — disable both auth schemes for local experimentation.
- **`--server-url`** — run client workflows against an already-running server, local or
  remote, instead of launching one.
- **`--local-server --tt-metal-home PATH`** — run on the host with no Docker at all,
  against a TT-Metalium checkout you built yourself.
- **`--tools {vllm,genai,aiperf,guidellm}`** — pick the benchmark load generator.
- **`--limit-samples-mode smoke-test`** — a fast reduced-sample preset for proving the
  pipeline works before committing to a full sweep.
- **`--engine {vllm,media,forge}`** and **`--impl`** — select the serving engine and the
  model implementation (e.g. `tt-transformers`, `qwen36-blackhole`, `gpt-oss`) when a
  model offers more than one.
- **Multi-host deployment** — supported, but note there is **no `--multihost` flag**. It
  is inferred from the device type: `dual_galaxy`, `quad_galaxy`, and `super_cluster` are
  the multi-host devices, and passing one of those to `--tt-device` selects the
  multi-host image and orchestration path. See `docs/multihost_deployment.md`.

### Reading upstream docs critically

tt-inference-server's own documentation has drifted behind its code in a few specific
places. If you hit a contradiction, **the code wins** — `workflows/workflow_types.py`
for valid `--workflow` and `--tt-device` values, `run.py` for flags, and
`workflows/model_specs/prod/*.yaml` for what actually runs where. Known stale spots:
the workflow list still shows the removed `reports`/`tests`, and the device list omits
every `p300*` and `p150x*` value — including `p300x2`, which very much exists.

---

## Next Steps

- [vLLM Production →](command:tenstorrent.showLesson?["vllm-production"]) — run vLLM directly without the workflow wrapper. This is also where the very small models live: **Qwen3-0.6B** is a great fit on that path, and is the one place it's a valid choice.
- [VSCode Chat →](command:tenstorrent.showLesson?["vscode-chat"]) — connect the inference server to the VSCode @tenstorrent chat participant
- [TT-Inference-Server workflows guide](https://github.com/tenstorrent/tt-inference-server/blob/main/docs/workflows_user_guide.md) — full CLI reference (read it with the "stale spots" list above in mind)
- [`vllm-tt-metal/` README](https://github.com/tenstorrent/tt-inference-server/blob/main/vllm-tt-metal/README.md) — the vLLM integration itself, including the canonical JWT auth example. Note the directory name: it was `vllm-tt-metal-llama3/` until March 2026, so older bookmarks 404.
