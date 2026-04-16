---
id: qb2-video-generation
title: Generating Video on QuietBox 2
description: Go from a fresh QB2 to AI-generated video clips in one session — Wan2.2-T2V-A14B on 4× Blackhole chips with a GTK4 GUI and automated prompt generation
category: applications
tags:
  - qb2
  - p300x2
  - video-generation
  - wan2-2
  - text-to-video
  - diffusers
  - skyreels
  - docker
supportedHardware:
  - p300x2
status: validated
validatedOn:
  - p300x2
estimatedMinutes: 90
---

# Generating Video on QuietBox 2

> **QB2-only lesson.** Everything here is validated on QuietBox 2 (P300X2 — 4× Blackhole chips in a (2,2) mesh). The model, the patch set, the timing data, and TT-TV attractor mode were all built and tested directly on QB2 hardware.

Your QB2 can generate original AI video — 5–7 minute clips of cinematic footage driven by natural language prompts, running completely offline with no cloud API required. This lesson takes you from a fresh Ubuntu 24.04 install to a running GPU-accelerated video generation studio.

## What You'll Build

```
┌─────────────────────────────────────────────────────┐
│               TT-TV Attractor Mode                  │
│  (continuous generation + fullscreen playback loop) │
└────────────────────────┬────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────┐
│                  tt-gen (GTK4 GUI)                  │
│  Prompt ──▶ Generate ──▶ Gallery ──▶ Export         │
│  ✨ Prompt generator | 🎬 Hover previews | History  │
└────────────────────────┬────────────────────────────┘
                         │ HTTP POST /generate
┌────────────────────────▼────────────────────────────┐
│  tt-inference-server (Docker)                       │
│  Wan2.2-T2V-A14B-Diffusers  ·  WAN transformer     │
│  Port 8000                                          │
└────────────────────────┬────────────────────────────┘
                         │ TTNN dispatch
┌────────────────────────▼────────────────────────────┐
│  QuietBox 2 Hardware                                │
│  2× P300 cards = 4× Blackhole chips = (2,2) mesh   │
│  480 Tensix cores  ·  2,654 TFLOPS  ·  4× GDDR6    │
└─────────────────────────────────────────────────────┘
```

**Secondary stack** — runs alongside, no TT hardware needed:

```
Qwen3-0.6B (CPU, port 8001)  ←  Prompt polish server
word_banks.py + Markov chain  ←  Algorithmic stage
```

**What you get at the end:**

- Wan2.2-T2V-A14B generating 480×832 video at ~370 s/clip median
- Automated prompt generator (algorithmic → Markov → LLM polish)
- GUI gallery with hover previews, queue, history export
- **TT-TV attractor mode** — fullscreen screensaver that generates and plays video continuously

---

## Performance at a Glance

| What | Time | Notes |
|---|---|---|
| Clone & vendor setup | ~2 min | One-time |
| Wan2.2 download | ~64 min | One-time, ~118 GB |
| Qwen3-0.6B download | ~2 min | One-time, ~1.2 GB |
| First-run warmup | **~9 min (525 s)** | TT kernel compilation |
| Per-restart warmup | ~5 min | Compiled kernels cached |
| First video generated | ~6 min after warmup |  |
| Steady-state per clip | **~370 s (~6 min)** | 480×832, 80 frames |

> **The script says "~5 min" during warmup. Ignore it.** Measured time to `Application startup complete` on QB2 hardware is consistently 525 s on the first cold start after reboot.

---

## Prerequisites

- QB2 hardware detected and healthy (`tt-smi` showing 4 chips)
- Tenstorrent PPA installed (firmware + KMD current)
- Docker CE installed and running
- `python3-gi`, `python3-gi-cairo`, `gir1.2-gtk-4.0` installed (system apt, **not venv**)
- ~180 GB free disk (`~/.cache/huggingface` + `TT_DIT_CACHE_DIR`)
- HuggingFace account (Wan2.2 requires `hf auth login`)

**Verify your QB2 hardware:**

```bash
tt-smi -s | python3 -m json.tool
# Expect: 4 chips, all status "OK"
```

[Verify Hardware](command:tenstorrent.runHardwareDetection)

---

## Step 1: Verify QB2 Hardware

Before starting, confirm all four Blackhole chips are healthy:

```bash
tt-smi -s
```

[Run Hardware Detection](command:tenstorrent.runHardwareDetection)

Look for four entries with `"status": "OK"` and no `"error"` fields. If chips are missing or unhealthy:

```bash
# Reset all TT devices and retry
tt-smi -r
sleep 5
tt-smi -s
```

[Reset Device](command:tenstorrent.resetDevice)

---

## Step 2: Clone tt-local-generator

```bash
git clone https://github.com/tenstorrent/tt-local-generator.git ~/code/tt-local-generator
```

[Clone tt-local-generator](command:tenstorrent.cloneTtLocalGenerator)

This clones the GTK4 video generation UI. The repo is small (~5 MB); the heavy content (model weights, Docker image) is downloaded separately.

**Install GTK4 system dependencies** (if not already present):

```bash
sudo apt install python3-gi python3-gi-cairo gir1.2-gtk-4.0
```

> The app uses **system** `python3`, not a venv. GTK bindings are invisible inside virtualenvs.

---

## Step 3: Set Up the Vendored Inference Server

```bash
cd ~/code/tt-local-generator && ./bin/setup_vendor.sh
```

[Set Up Vendored Server](command:tenstorrent.setupVideoGenVendor)

This clones a shallow copy of `tt-inference-server` at a pinned, tested commit into `vendor/tt-inference-server/`. It never touches `~/code/tt-inference-server` if you have one — the vendor copy is isolated.

```
vendor/
  VENDOR_SHA                    ← pinned commit SHA (tracked in git)
  tt-inference-server/          ← shallow clone (gitignored)
```

Verify it worked:

```bash
cat ~/code/tt-local-generator/vendor/VENDOR_SHA
# Should print a 40-char SHA like: e7a2322f82d...
```

---

## Step 4: Apply QB2 Hotpatches

```bash
cd ~/code/tt-local-generator && ./bin/apply_patches.sh
```

[Apply QB2 Patches](command:tenstorrent.applyVideoGenPatches)

This injects two sets of patches into the vendored server:

| Patch | Path | Effect |
|---|---|---|
| Config overrides | `patches/media_server_config/` | P300X2 device shape, request timeouts |
| Pipeline fixes | `patches/tt_dit/` | WAN transformer forward pass fixes |

**Always run `apply_patches.sh` after `setup_vendor.sh`.** The patches are what make the server work correctly on QB2.

---

## Step 5: Authenticate with HuggingFace

Wan2.2-T2V-A14B requires a HuggingFace account. Get a token at https://huggingface.co/settings/tokens and accept the model terms at https://huggingface.co/Wan-AI.

[Set HuggingFace Token](command:tenstorrent.setHuggingFaceToken)

```bash
hf auth login --token "$HF_TOKEN"
```

[Login to HuggingFace](command:tenstorrent.loginHuggingFace)

Verify auth works:

```bash
hf whoami
```

---

## Step 6: Download the Models

### Wan2.2-T2V-A14B-Diffusers (~118 GB, one-time)

```bash
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers
```

[Download Wan2.2 Model](command:tenstorrent.downloadWan22Model)

This streams to `~/.cache/huggingface/hub/`. The Docker container bind-mounts this directory, so the weights are available to the server without copying.

> **One download, forever.** The model is cached and reused on every server start. On a fast connection expect ~60–90 minutes.

### Qwen3-0.6B (~1.2 GB, one-time)

```bash
hf download Qwen/Qwen3-0.6B
```

[Download Qwen3-0.6B](command:tenstorrent.downloadQwen3Small)

Qwen3-0.6B is the prompt polish LLM. It runs on CPU (no TT hardware required) and uses only ~2.9 GB RAM. It loads in under 30 seconds.

---

## Step 7: Configure the Environment File

The Docker container reads its config from `vendor/tt-inference-server/.env`. The most important setting is the compiled-weight cache directory:

```bash
# Edit the .env file
nano ~/code/tt-local-generator/vendor/tt-inference-server/.env
```

Set or confirm these values:

```bash
# QB2-specific: cache compiled TT weights across container restarts (~66 GB after first run)
TT_DIT_CACHE_DIR=/home/ttuser/.cache/tt_dit_cache

# Keep HF offline after download (prevents startup delays)
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

> **`TT_DIT_CACHE_DIR` matters a lot.** First-run warmup compiles Blackhole kernels for the WAN transformer (~525 s). With the cache set, subsequent starts use the compiled kernels and warm up in ~5 min instead.

---

## Step 8: Start the Servers

Open **two terminals** — one for each server.

### Terminal 1: Video Inference Server

```bash
cd ~/code/tt-local-generator && ./bin/start_wan_qb2.sh
```

[Start Wan2.2 Server](command:tenstorrent.startWan22Server)

Watch the output. You will see phases:

```
Starting container ghcr.io/tenstorrent/tt-media-inference-server:0.11.1-...
[silence for 2-3 min while weights load from DRAM]
Compiling TT kernels... (this is the 525 s phase on first run)
Application startup complete.
```

**The server is ready when you see `Application startup complete`.** Do not try to generate before this.

### Terminal 2: Prompt Generation Server

```bash
cd ~/code/tt-local-generator && ./bin/start_prompt_gen.sh
```

[Start Prompt Server](command:tenstorrent.startPromptGenServer)

This starts Qwen3-0.6B on port 8001. It loads in ~30 seconds. The prompt server is optional but makes the ✨ Generate Prompt button in the UI work.

---

## Step 9: Verify Both Servers Are Ready

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
# Expect: {"status": "ok"} or model info

curl -s http://localhost:8001/health | python3 -m json.tool
# Expect: {"status": "ok", "model_ready": true}
```

[Check Server Health](command:tenstorrent.checkVideoServerHealth)

If the video server returns nothing or an error, it's still warming up — wait and retry.

---

## Step 10: Launch tt-gen

```bash
cd ~/code/tt-local-generator && ./tt-gen
```

[Launch tt-gen](command:tenstorrent.launchTtGen)

The GTK4 window opens. Confirm the server status indicator in the toolbar shows green (healthy). If it shows red, wait for warmup to finish and the indicator will update automatically.

---

## Step 11: Generate Your First Video

1. Click the **Video** tab (if not already selected)
2. Confirm **Wan2.2** is selected in the model dropdown
3. Type a prompt, or click **✨ Generate Prompt** for an auto-generated one
4. Click **Generate**

A pending card appears in the gallery. Expect **370 s (~6 min)** for the first clip. The card shows a live elapsed timer.

**Good starter prompts for Wan2.2:**

```
Aerial view of a coastal city at golden hour, calm ocean, sailboats, cinematic drone shot

A lone wolf walking through a snowy pine forest at dusk, mist rising from the ground, ethereal lighting

Time-lapse of clouds moving over mountain peaks, dramatic shadows, 4K nature documentary style

Underwater coral reef at dawn, schools of colorful fish, shafts of morning light, slow camera drift
```

**Parameters that matter:**

| Setting | Default | Notes |
|---|---|---|
| Resolution | 480×832 | Native for Wan2.2 on QB2 |
| Frames | 80 | ~3.3 s at 24 fps |
| Steps | 50 | More = better quality, linearly slower |
| Guidance scale | 7.0 | Higher = follows prompt more strictly |

---

## The Prompt Generation System

The **✨ Generate Prompt** button runs a three-tier pipeline:

```
Tier 1 — Algorithmic (always available)
  word_banks.py selects slot-by-slot:
  subject + action + setting + lighting + camera + style

Tier 2 — Markov chain (markovify package)
  Trained on prompts/markov_seed.txt and any
  prompts you append to prompts/markov_output.txt

Tier 3 — LLM polish (Qwen3-0.6B, port 8001)
  Takes the raw slug from Tier 1/2, makes it flow
  naturally — does NOT re-select elements
  Falls back gracefully if port 8001 is down
```

**Grow the corpus over time:**

Append good prompts to `prompts/markov_output.txt` — they feed back into Tier 2 immediately on the next generation, with no restart required:

```bash
echo "video|cinematic tracking shot through an ancient Roman market, golden hour, busy merchants" \
  >> ~/code/tt-local-generator/app/prompts/markov_output.txt
```

**CLI usage (generate prompts without the GUI):**

```bash
# Default: algo + LLM polish, video type
python3 ~/code/tt-local-generator/app/generate_prompt.py

# Five prompts, no LLM, SkyReels type
python3 ~/code/tt-local-generator/app/generate_prompt.py \
  --type skyreels --count 5 --no-enhance

# Raw slug only (no JSON wrapper)
python3 ~/code/tt-local-generator/app/generate_prompt.py --raw
```

---

## TT-TV: The Attractor

Once you have a few clips in the gallery, you can activate **TT-TV** — an attractor mode that:

- Generates new prompts automatically (using the three-tier system)
- Submits generation jobs continuously
- Shows generated clips in fullscreen with fade transitions
- Cycles through your gallery when no new clip is ready

To activate TT-TV:

1. Build up at least a few clips in the gallery (5+ recommended)
2. Click the **TT-TV** button in the toolbar (looks like a television icon)
3. The window goes fullscreen — move the mouse to see the HUD overlay

TT-TV is the reason to keep the servers running overnight. Wake up to a gallery of AI video your QB2 generated while you slept.

> **Disk space:** TT-TV stops generating when free disk drops below 18 GB. Check `df -h ~` if generation pauses.

---

## Other Video Models

Once you have the infrastructure running, switching models is straightforward. The same `apply_patches.sh + start_*.sh` pattern works for all of them.

### SkyReels-V2-DF-1.3B-540P (Fast clips, same hardware)

SkyReels uses a smaller 1.3B diffusion transformer trained on the same WAN backbone. **~28 s/clip** vs ~370 s for Wan2.2 — 13× faster, at 540P resolution:

```bash
./bin/start_skyreels.sh
```

In the GUI, click the **SkyReels** model button in the Video tab. Supported frame counts: 9, 33, 65, 97 (set in Preferences → SkyReels → Frame Count).

### Mochi-1 (Different architecture)

```bash
./bin/start_mochi.sh
```

### Wan2.2-Animate-14B (Character animation, video-to-video)

The Animate model takes a motion video + character image and produces an animation. This is the **💃 Animate** source toggle in the GUI:

```bash
./bin/start_animate.sh
```

Required inputs:
- **Motion video** — MP4 supplying the motion pattern
- **Character image** — PNG/JPG of the subject
- **Mode** — `animation` (character mimics motion) or `replacement` (character replaces person)

---

## Troubleshooting

### Video server never prints "Application startup complete"

**Most likely cause:** First-run kernel compilation is still in progress. The 525 s warmup appears silent — no progress output. Wait up to 10 minutes.

If it has been more than 15 minutes:

```bash
# Check if the container is still running
docker ps | grep tt-media

# Check container logs
docker logs -f $(docker ps -q --filter "ancestor=ghcr.io/tenstorrent/tt-media")
```

If the container has exited, check logs for OOM or device errors.

### "Cannot open display" when launching tt-gen

The GTK4 app needs a display. If you're SSH-connected without X11 forwarding:

```bash
# Check your display
echo $DISPLAY

# Enable X11 forwarding in your SSH session (reconnect with -X)
ssh -X ttuser@quietbox2

# Or if on the machine directly, confirm DISPLAY is set
export DISPLAY=:0
./tt-gen
```

### python3-gi import error inside venv

The GTK bindings are installed as **system apt packages**. Always launch with the system Python:

```bash
/usr/bin/python3 ~/code/tt-local-generator/app/main.py
# or use the launcher:
./tt-gen   # uses /usr/bin/python3 internally
```

### Prompt server shows "algo only" from a remote client

If you're running the GUI from a Mac and connecting to QB2 via `--server http://quietbox2:8000`, the prompt server must also bind to `0.0.0.0`:

```bash
# Check if it's bound to 127.0.0.1 (loopback only)
ss -tlnp | grep 8001

# Restart with network binding
PROMPT_HOST=0.0.0.0 ./bin/start_prompt_gen.sh
```

### Vendor patches not applied / server doesn't respond correctly

Re-run the patch script any time you reset the vendor directory:

```bash
./bin/setup_vendor.sh --force   # re-clone at pinned SHA
./bin/apply_patches.sh          # re-apply patches
```

### Generated video shows blank / ⊘ icon in gallery

The video file exists but GTK4's GStreamer backend can't play it. On Linux this is usually a missing codec:

```bash
# Check GStreamer plugins
gst-inspect-1.0 --exists mp4mux && echo "MP4 OK" || echo "MP4 missing"
gst-inspect-1.0 --exists avenc_h264 && echo "H264 OK" || echo "H264 missing"

# Install missing plugins
sudo apt install gstreamer1.0-libav gstreamer1.0-plugins-ugly
```

---

## CLI Control

`tt-ctl` manages all services without touching the GUI:

```bash
# Start recommended services (wan2.2 + prompt-server)
./tt-ctl start all

# Check health of every managed service
./tt-ctl servers

# Stop everything
./tt-ctl stop wan2.2
./tt-ctl stop prompt-server
```

---

## What's Next

- **Explore the kernel level:** The `~/code/skyreels-ttlang/` directory has hand-written Tensix kernels for the WAN transformer block, verified in the functional simulator at SkyReels-1.3B production dimensions — a starting point for pushing throughput beyond TTNN dispatch
- **Build on the API:** `api_client.py` exposes a simple `generate()` call — write scripts that feed prompts from any source and collect the output MP4s
- **Automate at scale:** Hook `generate_prompt.py --raw` into a cron job; pipe output to `tt-ctl generate` for hands-off overnight batch runs
