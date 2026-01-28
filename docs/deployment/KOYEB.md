# Koyeb Deployment Guide

Deploy the Tenstorrent VSCode Toolkit to Koyeb cloud platform with optional N300 hardware acceleration.

## 🚀 Quick Deploy Buttons

### Option 1: Git-Based Deploy (Recommended for Development)

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&builder=dockerfile&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)

Builds from source using Koyeb-optimized Dockerfile.

**After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s** for hardware, or **small** for CPU-only
2. Set `PASSWORD` environment variable
3. Click Deploy

**Build time:** ~5-10 minutes

### Option 2: Docker Image Deploy (Recommended for Production)

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

Uses pre-built image from GitHub Container Registry.

**After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s** or **small**
2. Update `PASSWORD` environment variable
3. Click Deploy

**Deploy time:** ~2 minutes

> **Note:** Both methods require manual selection of instance type. Hardware selection is optional.

## Deployment Methods

There are **3 ways** to deploy to Koyeb:

| Method | Registry Login? | Build Location | Speed | Use Case |
|--------|----------------|----------------|-------|----------|
| **Direct Deploy** | ❌ No | Koyeb cloud | Fast | Testing/Dev |
| **Registry Deploy** | ✅ Yes | Local | Moderate | Production, CI/CD |
| **UI Deploy** | ✅ Yes | Local | Slowest | Manual, one-off |

### Method 1: Direct Deploy ⭐ (Best for Testing)

**No registry login needed!** Koyeb builds the image for you.

```bash
./koyeb-deploy-direct.sh
```

**How it works:**
1. Builds extension locally
2. Uploads directory to Koyeb
3. Koyeb builds Docker image remotely
4. Koyeb deploys automatically

**Perfect for:**
- Quick testing and development iterations
- When you don't want to set up registry access

### Method 2: Registry Deploy (Production)

**Registry login required.** You build and push, Koyeb pulls.

```bash
# One-time setup
./koyeb-registry-login.sh

# Deploy
./quick-deploy-koyeb.sh
```

**How it works:**
1. Builds extension locally
2. Builds Docker image locally
3. Pushes to registry.koyeb.com
4. Koyeb pulls and deploys

**Perfect for:**
- Production deployments
- CI/CD pipelines
- When you want image versioning

### Method 3: UI Deploy

**Manual deployment through Koyeb dashboard.**

1. Build and push image to registry
2. Go to https://app.koyeb.com
3. Click "Create Service"
4. Enter image URL
5. Configure options (see below)
6. Deploy

**Perfect for:**
- First-time users
- Visual learners
- One-off deployments

## Detailed Setup

### Prerequisites

```bash
# Install Koyeb CLI (one-time)
curl -fsSL https://cli.koyeb.com/install.sh | sh

# Login to Koyeb
koyeb login

# Set your organization (optional)
export KOYEB_ORG=your-org-name
```

### Direct Deploy Workflow

```bash
# Deploy directly without registry
./koyeb-deploy-direct.sh

# That's it! Watch the logs:
koyeb services logs vscode -f
```

**Output example:**
```
🚀 Koyeb Direct Deploy

Service: vscode
Password: abc123xyz456

1/2 Building extension...
2/2 Deploying to Koyeb...

✅ Deployed!
🌐 https://vscode-yourorg.koyeb.app
🔑 abc123xyz456
```

### Registry Deploy Workflow

```bash
# One-time: Login to Koyeb registry
./koyeb-registry-login.sh
# Or manually: podman login registry.koyeb.com

# Deploy
./quick-deploy-koyeb.sh

# Watch logs
koyeb services logs vscode -f
```

**Output example:**
```
🚀 Quick Deploy to Koyeb

Service: vscode
Image: registry.koyeb.com/yourorg/tt-vscode-toolkit:latest
Password: abc123xyz456

1/4 Building extension...
2/4 Building container...
3/4 Pushing to registry...
4/4 Deploying to Koyeb...

✅ Deployed!
🌐 https://vscode-yourorg.koyeb.app
🔑 abc123xyz456
```

### Manual CLI Deploy

```bash
# Build and push image
podman build -t registry.koyeb.com/myorg/tt-vscode:latest .
podman login registry.koyeb.com
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Deploy service
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=your-secure-password \
  --regions was \
  --instance-type small
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PASSWORD` | Yes | `changeme` | Password for accessing IDE |
| `SUDO_PASSWORD` | No | Same as PASSWORD | Sudo password inside container |
| `MESH_DEVICE` | No | - | Hardware type (N300, T3K, etc.) |
| `TT_METAL_HOME` | No | `/home/coder/tt-metal` | Path to tt-metal |

**⚠️ IMPORTANT:** Always set a custom password for production!

### Instance Types

| Type | CPU | RAM | Use Case |
|------|-----|-----|----------|
| nano | 0.1 vCPU | 512MB | Testing only |
| micro | 0.25 vCPU | 1GB | Light development |
| small | 1 vCPU | 2GB | **Recommended** |
| medium | 2 vCPU | 4GB | Heavy workloads |
| large | 4 vCPU | 8GB | Large models |

**Recommendation:** Start with `small` instance type.

### Regions

Choose a region close to you:
- `was` - Washington DC (US East)
- `fra` - Frankfurt (Europe)
- `sin` - Singapore (Asia)

## Hardware Acceleration (N300)

**🎉 Koyeb provides Tenstorrent N300 accelerators!**

### Requesting Hardware

#### Using koyeb.yaml

The `koyeb.yaml` file is pre-configured for N300:

```yaml
accelerator:
  type: tenstorrent-n300
  count: 1

env:
  - key: MESH_DEVICE
    value: N300
```

Deploy:
```bash
koyeb service create --yaml koyeb.yaml
```

#### Using CLI

```bash
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=mypassword \
  --env MESH_DEVICE=N300 \
  --regions was \
  --instance-type small \
  --accelerator tenstorrent-n300:1
```

#### Using Interactive Script

```bash
./deploy-to-koyeb.sh
```

You'll be asked:
```
Add Tenstorrent hardware accelerator?
  1) Yes - Single N300 (2 chips, recommended)
  2) Yes - 3x N300 (6 chips total)
  3) No - Software only (learning mode)
Choice [1]:
```

### Hardware Options

| Hardware | Chips | Use Case |
|----------|-------|----------|
| **Single N300** | 2 chips (Wormhole) | Development, testing, single models |
| **3x N300** | 6 chips total | Production, multi-model serving |

### What Works with Hardware

✅ **All 16 lessons fully functional**
✅ **Hardware detection** (sees real N300)
✅ **Model inference** (Llama, Qwen, etc.)
✅ **vLLM production serving**
✅ **Image generation** (Stable Diffusion)
✅ **Real performance testing**

### Verifying Hardware Access

After deployment:

1. Open the IDE URL
2. Open VSCode terminal (Ctrl+`)
3. Run:
   ```bash
   tt-smi
   ```
4. You should see:
   ```
   Device 0: N300
   Device 1: N300
   Status: Active
   ```

## Startup Logs

When your container starts, you'll see:

```
╔════════════════════════════════════════════════════════════╗
║  Tenstorrent VSCode Toolkit                                ║
║  Browser-based VSCode with TT Extension Pre-installed      ║
╚════════════════════════════════════════════════════════════╝

📍 Environment: Koyeb

🚀 YOUR IDE IS STARTING...

═══════════════════════════════════════════════════════════
📝 IMPORTANT - SAVE THESE DETAILS:
═══════════════════════════════════════════════════════════

  🌐 Access URL:  https://your-app.koyeb.app
  🔑 Password:    your-password-here

═══════════════════════════════════════════════════════════

🎯 WHAT'S INCLUDED:

  ✅ VSCode in your browser (code-server)
  ✅ Tenstorrent extension pre-installed
  ✅ 16 interactive hardware lessons
  ✅ Production deployment guides
  ✅ Template scripts and examples

✅ Service is ready!
```

## Accessing Your IDE

### From Koyeb Dashboard

1. Go to https://app.koyeb.com
2. Click your service
3. Click the **public URL** at the top
4. Enter your password
5. Extension is already loaded!

### Direct URL

```
https://your-service-name-your-org.koyeb.app
```

### Verifying Extension is Installed

1. Look for the **Tenstorrent icon** in the left sidebar (orange/purple TT logo)
2. Click the icon to see lessons
3. Or open Command Palette (Cmd/Ctrl+Shift+P) and type "Tenstorrent"

## Managing Your Deployment

### View Logs

```bash
# Real-time logs
koyeb service logs tt-vscode -f

# Last 100 lines
koyeb service logs tt-vscode --tail 100
```

### Check Status

```bash
koyeb service get tt-vscode
```

### Health Check

```bash
curl https://your-app.koyeb.app/healthz
```

### Update Deployment

```bash
# Rebuild and push new image
podman build -t registry.koyeb.com/myorg/tt-vscode:latest .
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Redeploy service
koyeb service redeploy tt-vscode
```

### Delete Service

```bash
koyeb service delete tt-vscode
```

## Troubleshooting

### Service Won't Start

**Check logs:**
```bash
koyeb service logs tt-vscode -f
```

**Common issues:**
- Wrong image URL
- Image not public (or not logged in)
- Port not set to 8080
- Incorrect environment variables

### Extension Not Showing

**Verify in logs:**
```
[info] Extension host agent started
```

**Check in IDE:**
- Cmd/Ctrl+Shift+X (Extensions)
- Search for "Tenstorrent"
- Should show as installed

### Can't Connect / 502 Error

- Wait 30-60 seconds for container to start
- Check health endpoint: `https://your-app.koyeb.app/healthz`
- Review logs for errors

### Password Doesn't Work

- Check environment variable in Koyeb dashboard
- Make sure `PASSWORD` is set correctly
- Try redeploying with correct password

### Hardware Not Detected

```bash
# In IDE terminal
lspci | grep Tenstorrent

# Reset devices
tt-smi -r

# Check environment
echo $MESH_DEVICE
```

## Security Best Practices

### 1. Strong Password

```bash
# Generate strong password
PASSWORD=$(openssl rand -base64 32)

# Use in deployment
koyeb service create ... --env PASSWORD=$PASSWORD
```

### 2. Use Koyeb Secrets

```bash
# Create secret
koyeb secret create vscode-password --value "your-strong-password"

# Reference in deployment
koyeb service create ... --env PASSWORD=@vscode-password
```

### 3. Keep Image Updated

```bash
# Rebuild and redeploy regularly
npm run container:build
podman push registry.koyeb.com/myorg/tt-vscode:latest
koyeb service redeploy tt-vscode
```

## Cost Optimization

### Free Tier
- Good for testing and light development
- May sleep after inactivity

### Paid Plans
- For production use
- Better performance
- No sleep

### Tips
1. Use `nano` or `micro` for testing
2. Delete service when not in use
3. Use hardware only when needed
4. Start with single N300, scale to 3x only if needed

## Persistent Storage

⚠️ **Important:** Koyeb containers are ephemeral. Work is lost when container restarts!

**Solutions:**

### Option 1: Use Git
```bash
git clone https://github.com/your-username/your-repo.git
# Make changes
git add . && git commit -m "Changes" && git push
```

### Option 2: Download Files
- Use VSCode download feature (right-click → Download)
- Download before shutting down

### Option 3: External Storage
- Mount external volume (Koyeb paid plans)
- Use S3/object storage for large files

## Quick Reference Commands

```bash
# Direct deploy (testing)
./koyeb-deploy-direct.sh

# Registry deploy (production)
./koyeb-registry-login.sh  # one-time
./quick-deploy-koyeb.sh    # every deployment

# Interactive deploy
./deploy-to-koyeb.sh

# Check status
koyeb services get vscode

# Watch logs
koyeb services logs vscode -f

# Delete service
koyeb services delete vscode
```

## Resources

- **Koyeb Documentation:** https://www.koyeb.com/docs
- **Koyeb Hardware:** https://www.koyeb.com/docs/accelerators
- **General Deployment:** [DEPLOYMENT.md](./DEPLOYMENT.md)
- **Extension Issues:** https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Tenstorrent Discord:** https://discord.gg/tenstorrent
- **Koyeb Support:** support@koyeb.com

## Summary

✅ **Quick deploy buttons** - One-click deployment
✅ **Hardware acceleration** - Real N300 chips available
✅ **Multiple methods** - Direct, registry, or UI deploy
✅ **Clear logs** - Helpful startup messages guide you
✅ **Pre-installed extension** - Ready to use immediately

**Start with direct deploy for testing, move to registry deploy for production! 🚀**
