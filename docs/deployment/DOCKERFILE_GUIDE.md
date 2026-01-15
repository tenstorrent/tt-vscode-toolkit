# Dockerfile Guide - Which One to Use?

We provide **4 different Dockerfiles** for different use cases:

## Quick Reference

| Dockerfile | Size | tt-metal | Use Case |
|------------|------|----------|----------|
| `Dockerfile` | ~500MB | ❌ Not included | **Local testing, no hardware** |
| `Dockerfile.full` | ~2GB | ⚠️ Deps only | Development, some dependencies |
| `Dockerfile.koyeb` | ~600MB | ⚠️ Uses host | **Koyeb with N300** (recommended) |
| `Dockerfile.production` | ~8GB | ✅ Fully built | **Self-hosted with hardware** |

## Detailed Comparison

### 1. Dockerfile (Basic)

**Size:** ~500MB
**Contents:** code-server + extension only
**tt-metal:** Not included

```bash
podman build -t tt-vscode:basic .
npm run container:build
```

**Perfect for:**
- ✅ Local testing without hardware
- ✅ Quick builds (2-3 minutes)
- ✅ Learning mode (read lessons, no execution)
- ✅ Minimal resource usage

**Not good for:**
- ❌ Running actual tt-metal code
- ❌ Hardware testing
- ❌ Production deployments

---

### 2. Dockerfile.full (Dependencies)

**Size:** ~2GB
**Contents:** code-server + extension + system dependencies
**tt-metal:** System dependencies installed, but not tt-metal itself

```bash
podman build -f Dockerfile.full -t tt-vscode:full .
npm run container:build-full
```

**Perfect for:**
- ✅ Installing tt-metal inside running container
- ✅ Building tt-metal from source at runtime
- ✅ Development environments

**Not good for:**
- ❌ Production (tt-metal not pre-built)
- ❌ Quick starts (still need to build tt-metal)

---

### 3. Dockerfile.koyeb (Koyeb Optimized) ⭐

**Size:** ~600MB
**Contents:** code-server + extension + minimal tools
**tt-metal:** Expects to use host installation at `/opt/tt-metal`

```bash
podman build -f Dockerfile.koyeb -t tt-vscode:koyeb .
npm run container:build-koyeb
```

**Perfect for:**
- ✅ **Koyeb deployment with N300 hardware** (RECOMMENDED)
- ✅ Cloud platforms with pre-installed tt-metal
- ✅ Fast builds
- ✅ Small image size
- ✅ Uses system tt-metal (better performance)

**How it works:**
- Container checks for tt-metal at `/opt/tt-metal` (Koyeb standard location)
- Falls back to `~/tt-metal` if available
- Auto-configures environment variables
- Activates Python venv if found

**Best for Koyeb because:**
- Koyeb provides tt-metal on N300 instances
- No need to build tt-metal in container
- Faster deployment
- Smaller transfer size

---

### 4. Dockerfile.production (Everything Included) 🏭

**Size:** ~8GB
**Contents:** code-server + extension + tt-metal fully built + vLLM
**tt-metal:** Fully installed and built at `~/tt-metal`

```bash
podman build -f Dockerfile.production -t tt-vscode:production .
npm run container:build-production
```

**Perfect for:**
- ✅ **Self-hosted deployments with hardware**
- ✅ Complete standalone environment
- ✅ Air-gapped deployments
- ✅ Guaranteed working environment

**Includes:**
- ✅ tt-metal cloned and built
- ✅ All submodules initialized
- ✅ Python environment with tt-metal packages
- ✅ vLLM installed
- ✅ PyTorch, transformers, etc.
- ✅ Ready to run any lesson

**Takes longer to build:**
- Initial build: 30-60 minutes
- But everything is ready to go!

---

## Recommendation by Use Case

### For Koyeb with N300 Hardware

```bash
# Use Dockerfile.koyeb (recommended)
npm run container:build-koyeb
podman push registry.koyeb.com/myorg/tt-vscode:latest
```

**Why:** Koyeb provides tt-metal on N300 instances, so use their system installation.

### For Self-Hosted with Hardware

```bash
# Use Dockerfile.production
npm run container:build-production
podman run -it --device=/dev/tenstorrent/0 -p 8080:8080 tt-vscode:production
```

**Why:** Complete environment, everything works out of the box.

### For Local Testing (No Hardware)

```bash
# Use Dockerfile (basic)
npm run container:build
podman run -it -p 8080:8080 tt-vscode:basic
```

**Why:** Fast builds, small size, perfect for testing the extension UI.

### For Development

```bash
# Use Dockerfile.full
npm run container:build-full
podman run -it -p 8080:8080 tt-vscode:full

# Then inside container:
cd ~/tt-metal
git clone https://github.com/tenstorrent/tt-metal.git .
./build_metal.sh
```

**Why:** Dependencies installed, but you control tt-metal version.

---

## Build Times

| Dockerfile | First Build | Rebuild | Deploy Time |
|------------|-------------|---------|-------------|
| Basic | 2-3 min | 1 min | Fast |
| Full | 5-10 min | 2-3 min | Fast |
| Koyeb | 3-5 min | 1-2 min | Fast |
| Production | 30-60 min | 5-10 min | Medium |

---

## Environment Variables

All Dockerfiles support these:

```bash
PASSWORD=mypassword          # IDE password
MESH_DEVICE=N300             # Hardware type
TT_METAL_HOME=/path/to/metal # tt-metal location
PYTHONPATH=/path/to/metal    # Python path
```

### Dockerfile.koyeb Auto-Detection

The Koyeb image automatically checks for:
1. `/opt/tt-metal` (Koyeb standard)
2. `~/tt-metal` (fallback)
3. Activates Python venv if found
4. Sets all environment variables

---

## Quick Comparison

**Need everything to just work?**
→ Use `Dockerfile.production`

**Deploying to Koyeb with N300?**
→ Use `Dockerfile.koyeb`

**Just testing the extension UI?**
→ Use `Dockerfile`

**Want flexibility in development?**
→ Use `Dockerfile.full`

---

## Building for Koyeb

```bash
# Recommended for Koyeb
npm run build
npm run package
podman build -f Dockerfile.koyeb -t registry.koyeb.com/myorg/tt-vscode:latest .
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Deploy with N300
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --accelerator tenstorrent-n300:1 \
  --ports 8080:http \
  --env PASSWORD=mypassword \
  --env MESH_DEVICE=N300
```

The container will automatically find and use Koyeb's tt-metal installation!

---

## What Happens at Container Start

### Basic/Full/Koyeb:
1. Shows welcome banner
2. Displays URL and password
3. Checks for tt-metal (Koyeb only)
4. Starts code-server
5. Extension auto-loads

### Production:
1. Shows welcome banner
2. Confirms tt-metal is ready
3. Activates Python environment
4. Shows tt-metal location
5. Starts code-server
6. Extension auto-loads with full functionality

---

## Summary

**For Koyeb:** Use `Dockerfile.koyeb` - it's optimized for their environment!
**For Self-Hosting:** Use `Dockerfile.production` - complete and ready!
**For Testing:** Use `Dockerfile` - fast and simple!

All containers provide the same extension experience, but production/koyeb containers can actually run the hardware lessons. 🚀
