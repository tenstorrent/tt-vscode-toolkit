---
id: deploy-vscode-to-koyeb
title: "Deploy tt-vscode-toolkit to Koyeb"
description: >-
  Deploy your own cloud-based VSCode IDE with the Tenstorrent extension pre-installed. Run on Koyeb with optional N300 hardware access.
category: advanced
tags:
  - deployment
  - koyeb
  - cloud
  - vscode
  - ide
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: draft
estimatedMinutes: 20
---

# Deploy tt-vscode-toolkit to Koyeb

Get your own cloud-based VSCode IDE with the Tenstorrent extension pre-installed, running on Koyeb with access to N300 hardware.

## What You'll Build

A browser-accessible VSCode environment with:
- ✅ Tenstorrent extension pre-installed
- ✅ tt-smi for hardware monitoring
- ✅ Access to Tenstorrent N300 accelerator hardware
- ✅ Persistent development environment
- ✅ Accessible from anywhere via HTTPS

## Prerequisites

- Koyeb account (free tier available)
- Git clone of tt-vscode-toolkit repository
- ~10-15 minutes for initial deployment

## Why Deploy to Koyeb?

Koyeb offers **direct access to Tenstorrent N300 hardware** (`gpu-tenstorrent-n300s` instance type) with:
- Native N300 support (2 Wormhole chips)
- `/dev/tenstorrent/` device access
- Dedicated hardware for your workloads
- HTTPS endpoints with automatic SSL
- Simple deployment from source code

Perfect for:
- Learning Tenstorrent development without local hardware
- Remote development with real hardware
- Sharing reproducible environments
- Testing before purchasing hardware

---

## Step 1: Install Koyeb CLI

```bash
curl -fsSL https://cli.koyeb.com/install.sh | sh
```

Verify installation:
```bash
koyeb version
```

---

## Step 2: Login to Koyeb

```bash
koyeb login
```

This opens your browser for authentication.

Verify you're logged in:
```bash
koyeb profile show
```

---

## Step 3: Clone the Repository

```bash
git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
cd tt-vscode-toolkit
```

---

## Step 4: Quick Deploy (Recommended)

The repository includes a deployment script that handles everything:

```bash
./scripts/koyeb-deploy-direct.sh
```

**What this does:**
1. Builds the VSCode extension locally
2. Uploads your directory to Koyeb (84MB compressed)
3. Koyeb builds the Docker image remotely
4. Deploys with N300 hardware automatically
5. Outputs your access URL with password

**Output:**
```
🚀 Koyeb Direct Deploy (No Registry!)

App:     tt-vscode-toolkit
Service: vscode
Pass:    abc123xyz456

1/2 Building extension...
2/2 Deploying to Koyeb...

✅ Deployed!

🌐 https://vscode-<hash>.koyeb.app?password=abc123xyz456
   (URL includes password for easy access)

🔑 Password: abc123xyz456
```

**⚠️ Save your password!** You'll need it to access your IDE.

---

## Step 5: Access Your IDE

Click the URL from the deployment output. The password is included in the URL for convenience:

```
https://vscode-<your-hash>.koyeb.app?password=abc123xyz456
```

**First-time setup** (30-60 seconds):
- Container starts
- Extension activates
- Hardware permissions configured
- Welcome page opens

---

## Step 6: Verify Hardware Access

Open a new terminal in your cloud IDE (Terminal → New Terminal) and run:

```bash
tt-smi
```

**Expected output:**
```
 Board type: p300c
 Num devices: 2

 Device 0:
   Coordinates (rack,shelf,y,x): (0,0,0,0)
   PCI domain:bus:device.function: 0000:01:00.0
   Board type: n300
   ...
```

**Note:** N300 reports as 2 devices (2 Wormhole chips on one board) - this is correct!

---

## Step 7: Explore the Extension

The Tenstorrent extension is pre-installed and active:

1. **Click the Tenstorrent icon** in the left sidebar (TT logo)
2. **Browse lessons** - All 16+ interactive lessons available
3. **Open Welcome Page** - Click "Welcome Page" in the sidebar
4. **Try a lesson** - Start with "Your First Inference" or "Hardware Detection"

---

## What's Included in the Deployment

Your cloud IDE includes:

**Software Stack:**
- Ubuntu 24.04 (noble)
- code-server (VSCode in browser)
- tt-smi v3.1.1+ (hardware monitoring)
- Tenstorrent VSCode extension
- Git, Python 3.12, build tools

**Hardware:**
- Tenstorrent N300 (2x Wormhole chips)
- Device access via `/dev/tenstorrent/0` and `/dev/tenstorrent/1`

**CLI Tools:**
- `hf` - HuggingFace CLI for downloading models
- `claude` - Claude Code CLI for AI-assisted development (requires ANTHROPIC_API_KEY)
- `tt-smi` - Tenstorrent hardware monitoring
- Standard dev tools (git, python3, npm, etc.)
- Full hardware permissions configured

**Configuration:**
- Auto-configured theme (Tenstorrent Dark)
- Device permissions (sudo, video, render groups)
- Privileged container mode for hardware access
- No telemetry, auto-updates disabled

---

## Managing Your Deployment

### Check Status

```bash
koyeb services get vscode
```

### Watch Logs

```bash
koyeb services logs vscode -f
```

### Update Deployment

After making changes to the codebase:

```bash
./scripts/koyeb-deploy-direct.sh
```

This creates a new deployment. The old one is automatically removed.

### Delete Service

When you're done:

```bash
koyeb services delete vscode
```

---

## Optional: Deploy with tt-metal Pre-installed

By default, the IDE deploys quickly (~5 minutes) without tt-metal. To include a fully-built tt-metal installation:

**Edit `Dockerfile.koyeb`** and uncomment the tt-metal section:

```dockerfile
# Clone and install tt-metal with submodules
RUN git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git /home/coder/tt-metal

# Switch to root to install dependencies
USER root
RUN cd /home/coder/tt-metal && ./install_dependencies.sh

# Switch back to coder to build tt-metal
USER coder
RUN cd /home/coder/tt-metal && ./build_metal.sh
```

Then deploy:

```bash
./scripts/koyeb-deploy-direct.sh
```

⏱️ **Build time:** 15-25 minutes (one-time cost)

**Benefits:**
- tt-metal ready at `~/tt-metal`
- Python environment auto-activated
- All metal tools available immediately
- No need to build on first use

---

## Troubleshooting

### Deployment Stuck or Fails

**Check build logs:**
```bash
koyeb services get vscode --full -o json | jq
```

**Common issues:**
- ✅ Network timeouts: Redeploy, usually succeeds on retry
- ✅ Archive too large: Ensure old `.vsix` files are cleaned up
- ✅ CLI bug errors: Ignore `DEPLOYMENT_STRATEGY_TYPE_DEFAULT` messages, check actual service status

### Can't Access Hardware

If `tt-smi` shows "Permission denied":

1. Verify privileged mode: Check `koyeb services get vscode` shows `privileged: true`
2. Check device files: `ls -la /dev/tenstorrent/`
3. Verify groups: `groups` should show `sudo`, `video`, `render`

The entrypoint script automatically fixes permissions on startup via:
```bash
sudo chmod -R 666 /dev/tenstorrent/*
```

### Extension Not Visible

If the Tenstorrent extension doesn't appear:

1. Check Extensions sidebar (Cmd/Ctrl+Shift+X)
2. Search for "Tenstorrent"
3. Verify it's installed and enabled

The extension should auto-install during build. If missing, the container needs rebuilding with proper user permissions.

---

## Cost Considerations

**Koyeb N300 pricing:**
- Charges while running (compute + hardware)
- No charge when stopped/paused
- Pay-as-you-go or reserved instances

**Recommendations:**
- Delete service when not in use: `koyeb services delete vscode`
- Redeploy in ~5 minutes when needed
- Archive size is small (84MB) so redeployment is fast

---

## Advanced: Custom Configuration

### Change Password

Set `PASSWORD` environment variable before deploying:

```bash
export PASSWORD=my-secure-password
./scripts/koyeb-deploy-direct.sh
```

### Change Region

Edit `scripts/koyeb-deploy-direct.sh` and modify:

```bash
--regions na  # North America (default)
# Options: na (US), fra (Europe), sin (Asia)
```

### Use Different Instance Type

For testing without hardware:

```bash
koyeb deploy . tt-vscode-toolkit/vscode \
  --archive-builder docker \
  --archive-docker-dockerfile Dockerfile.koyeb \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=yourpass \
  --regions na \
  --instance-type small  # No hardware, cheaper for testing
```

---

## Next Steps

✅ **You now have a cloud VSCode IDE with Tenstorrent hardware!**

**Continue your journey:**
1. 🎯 [Deploy Your Work to Koyeb](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22deploy-to-koyeb%22%7D) - Deploy your own apps with N300
2. 🚀 [Your First Inference](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22first-inference%22%7D) - Run your first model
3. 🏭 [vLLM Production](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D) - Production model serving

**Share your deployment:**
- Show teammates: Share the URL + password
- Create tutorials: Your environment is reproducible
- Test ideas: Spin up/down as needed

---

## Summary

**What you learned:**
- ✅ Deploy VSCode to cloud with one command
- ✅ Access real Tenstorrent N300 hardware remotely
- ✅ Manage cloud deployments with Koyeb CLI
- ✅ Configure hardware access in containers

**Key files:**
- `scripts/koyeb-deploy-direct.sh` - One-command deployment
- `Dockerfile.koyeb` - Container configuration
- `docker-entrypoint.sh` - Startup configuration

**Time invested:** ~15 minutes
**Result:** Professional cloud IDE with hardware access 🎉
