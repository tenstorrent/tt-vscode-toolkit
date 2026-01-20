# Deploy to Koyeb Button Guide

This document explains how the "Deploy to Koyeb" button works and how we publish images on every release.

## Overview

We support **two deployment methods** via buttons:

1. **Git-based (Recommended)** - Koyeb builds from source
2. **Docker-based** - Uses pre-built images from releases

---

## Method 1: Git-Based Deployment (Recommended)

### The Button

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&builder=dockerfile&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)

### How It Works

1. User clicks the button
2. Redirected to Koyeb with pre-filled configuration
3. Koyeb clones the repository
4. Builds using `Dockerfile.koyeb`
5. Deploys to N300 hardware

### Configuration

The button URL includes these parameters:

```
https://app.koyeb.com/deploy?
  type=git
  &builder=dockerfile
  &repository=github.com/tenstorrent/tt-vscode-toolkit
  &branch=main
  &name=tt-vscode-toolkit
```

**After clicking, users must:**
1. Select **Instance Type**: `gpu-tenstorrent-n300s` (Tenstorrent N300)
2. Add environment variable: `PASSWORD=<strong-password>`
3. Verify port: `8080:http`
4. Choose region (e.g., `na` for North America)
5. Click "Deploy"

### Advantages

✅ Always builds from latest code
✅ Uses Koyeb-optimized `Dockerfile.koyeb`
✅ Users can choose branch/tag
✅ Transparent build process
✅ Automatic N300 hardware provisioning

### Disadvantages

⚠️ Requires manual configuration of instance type
⚠️ Build takes 5-10 minutes
⚠️ Requires Koyeb account

---

## Method 2: Docker-Based Deployment

### The Button

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

### How It Works

1. User clicks the button
2. Koyeb pulls pre-built image from GitHub Container Registry
3. Deploys immediately (no build step)
4. User configures N300 hardware

### Configuration

The button URL includes:

```
https://app.koyeb.com/deploy?
  type=docker
  &image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest
  &name=tt-vscode-toolkit
  &ports=8080;http;/
  &env[PASSWORD]=changeme
```

**After clicking, users must:**
1. Select **Instance Type**: `gpu-tenstorrent-n300s`
2. Update `PASSWORD` environment variable
3. Click "Deploy"

### Available Images

We publish Docker images on **every release**:

```bash
# Latest stable release
ghcr.io/tenstorrent/tt-vscode-toolkit:latest

# Specific version
ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.254

# Development branch
ghcr.io/tenstorrent/tt-vscode-toolkit:main
```

### Advantages

✅ Fast deployment (no build)
✅ Pre-tested images
✅ Specific version control
✅ Works with private registries

### Disadvantages

⚠️ Uses generic Dockerfile (not Koyeb-optimized)
⚠️ Still requires manual N300 configuration
⚠️ Larger image size (~3GB with tt-metal)

---

## Release Process - Automated Image Publishing

When you push a version tag (e.g., `v0.0.254`), GitHub Actions automatically:

### 1. Test & Build
- Runs 315+ mocha tests
- Builds extension
- Packages VSIX

### 2. Publish Docker Images
- Tags images with version number
- Updates `latest` tag
- Pushes to GitHub Container Registry

```bash
# After release v0.0.254, these images are available:
ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.254
ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.254-full
ghcr.io/tenstorrent/tt-vscode-toolkit:latest
ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full
```

### 3. Create GitHub Release
- Attaches VSIX file
- Includes installation instructions
- Auto-generates release notes
- Links to Docker images

### 4. Deploy Button Updates
- `latest` tag automatically points to new version
- Users get newest release when clicking button

---

## Advanced: Instance Type in URL

⚠️ **Current Limitation:** Koyeb's deploy button doesn't support `instance_type` parameter in URLs.

**Workaround:** We provide `koyeb.yaml` configuration file that users can modify:

```yaml
# koyeb.yaml
accelerator:
  type: tenstorrent-n300
  count: 1
```

**Alternative:** Users must manually select instance type after clicking deploy button.

---

## Implementation in README.md

### For Git-Based Deployment (Recommended)

```markdown
## Quick Deploy to Koyeb

Deploy your own cloud VSCode IDE with Tenstorrent N300 hardware:

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&builder=dockerfile&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)

**After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s**
2. Set environment variable: `PASSWORD=<your-password>`
3. Click Deploy

**Build time:** 5-10 minutes | **Cost:** ~$0.XX/hour with N300
```

### For Docker-Based Deployment

```markdown
## Quick Deploy with Docker Image

Faster deployment using pre-built images:

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

**After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s**
2. Update `PASSWORD` environment variable
3. Click Deploy

**Deploy time:** ~2 minutes | **Cost:** ~$0.XX/hour with N300
```

---

## Comparison

| Aspect | Git-Based | Docker-Based |
|--------|-----------|--------------|
| **Build Time** | 5-10 min | No build (~2 min deploy) |
| **Dockerfile** | Dockerfile.koyeb (optimized) | Dockerfile (generic) |
| **Image Source** | Built by Koyeb | ghcr.io pre-built |
| **Version Control** | Branch/tag selection | Specific image tags |
| **Customization** | Full source access | Limited to pre-built |
| **Best For** | Development, latest features | Production, stability |

---

## FAQ

### Q: Why two deployment methods?

**Git-based** is better for development and testing (always builds from source with Koyeb optimizations). **Docker-based** is better for production (faster, tested releases).

### Q: Can we auto-configure N300 hardware?

Not currently. Koyeb's deploy button doesn't support `instance_type` in URL parameters. Users must manually select it after clicking.

**Future:** We're exploring Koyeb's API to create a custom deploy flow with N300 pre-configured.

### Q: What about private repositories?

For private repos, use Docker-based deployment or deploy via Koyeb CLI:

```bash
git clone https://github.com/your-org/tt-vscode-toolkit
cd tt-vscode-toolkit
./scripts/koyeb-deploy-direct.sh
```

### Q: How often are images published?

- **On every release** (version tags)
- **On every push to main** (main tag)
- **On every push to docker-image** (docker-image tag)

---

## Resources

- **Koyeb Deploy Button Docs:** https://www.koyeb.com/docs/build-and-deploy/deploy-to-koyeb-button
- **Tenstorrent N300 on Koyeb:** https://www.koyeb.com/docs/hardware/tenstorrent-n300
- **GitHub Container Registry:** https://github.com/tenstorrent/tt-vscode-toolkit/pkgs/container/tt-vscode-toolkit

---

## For Maintainers

### Testing the Deploy Button

```bash
# 1. Create test branch
git checkout -b test-deploy-button

# 2. Update button URL in README.md
# Change branch=main to branch=test-deploy-button

# 3. Push and test
git push origin test-deploy-button

# 4. Click button and verify
```

### Updating Button Parameters

Edit `docs/deployment/README.md` and update the deploy button URL:

```markdown
[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)
```

### Publishing to Different Registry

To use a different container registry (e.g., Docker Hub, AWS ECR):

1. Update `.github/workflows/release.yml`
2. Change `REGISTRY` environment variable
3. Add registry credentials to GitHub Secrets
4. Update deploy button URL with new image path
