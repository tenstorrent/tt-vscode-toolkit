# Container Registry Guide

This document explains the different Docker images, where they're built, and how they relate to deployments.

## Overview

The project uses **three different Docker image sources**:

1. **GitHub Container Registry (ghcr.io)** - Pre-built images from GitHub Actions
2. **Koyeb Private Registry** - Images built during Koyeb deployment
3. **Local Images** - Images built on your machine for testing

**Important:** These are **separate systems** - they don't share images or registries.

---

## 1. GitHub Container Registry (ghcr.io)

**Purpose:** Public Docker images anyone can pull and run

**Built by:** GitHub Actions workflow (`.github/workflows/docker-build.yml`)

**Triggers:**
- Push to `main` or `docker-image` branch
- Push version tag (e.g., `v0.0.254`)
- Pull request to `main`

**Images created:**
```bash
# Basic image (small, fast build ~5 min)
ghcr.io/tenstorrent/tt-vscode-toolkit:main
ghcr.io/tenstorrent/tt-vscode-toolkit:docker-image
ghcr.io/tenstorrent/tt-vscode-toolkit:v0.0.254      # On release
ghcr.io/tenstorrent/tt-vscode-toolkit:latest        # On release

# Full image (includes tt-metal pre-built, ~20 min build)
ghcr.io/tenstorrent/tt-vscode-toolkit:main-full
ghcr.io/tenstorrent/tt-vscode-toolkit:docker-image-full
ghcr.io/tenstorrent/tt-vscode-toolkit:v0.0.254-full # On release
ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full   # On release
```

**How to use:**
```bash
# Pull and run basic image
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest
docker run -p 8080:8080 -e PASSWORD=test ghcr.io/tenstorrent/tt-vscode-toolkit:latest

# Pull and run full image (with tt-metal)
docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full
docker run -p 8080:8080 -e PASSWORD=test ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full
```

**Built from:**
- Basic: `Dockerfile`
- Full: `Dockerfile.full`

---

## 2. Koyeb Private Registry

**Purpose:** Private images for your Koyeb deployments

**Built by:** Koyeb's build system during deployment

**Triggers:**
- When you run `./scripts/koyeb-deploy-direct.sh`
- Or manually: `koyeb deploy . app/service --archive-builder docker`

**Images created:**
```bash
# Internal Koyeb registry (not publicly accessible)
registry01.prod.koyeb.com/k-{org-id}/{service-id}:{deployment-id}

# Example from our deployment:
registry01.prod.koyeb.com/k-558a542e-213f-45b5-b23b-e9391ae2a31d/104372ff-4e2c-4d56-b669-ae24a11fe56a:08b42ad1-6ade-4920-80e2-5877550e5e30
```

**How Koyeb deployment works:**
1. You run `./scripts/koyeb-deploy-direct.sh`
2. Script runs `npm run build && npm run package` locally
3. Koyeb CLI creates archive of entire directory (except .dockerignore)
4. Archive uploaded to Koyeb (includes source, Dockerfile.koyeb, .vsix)
5. **Koyeb builds Docker image in the cloud** using `Dockerfile.koyeb`
6. Image stored in Koyeb's private registry
7. Container deployed to N300 hardware

**Built from:**
- `Dockerfile.koyeb`

**Key differences from GitHub images:**
- Built on Koyeb infrastructure (not GitHub Actions)
- Uses `Dockerfile.koyeb` (optimized for Koyeb/N300)
- Stored in private registry (only your Koyeb org can access)
- Archives include your latest local changes (no git commit needed)

---

## 3. Local Images

**Purpose:** Testing on your development machine

**Built by:** You, manually

**Commands:**
```bash
# Build basic image
npm run build && npm run package
docker build -t tt-vscode-toolkit:local .

# Build full image
npm run build && npm run package
docker build -f Dockerfile.full -t tt-vscode-toolkit:local-full .

# Build Koyeb image (for testing)
npm run build && npm run package
docker build -f Dockerfile.koyeb -t tt-vscode-toolkit:local-koyeb .

# Run local image
docker run -p 8080:8080 -e PASSWORD=test tt-vscode-toolkit:local
```

---

## Quick Reference: Which Image Should I Use?

### For Development/Testing
- **Local build:** `docker build -t tt-vscode:test .`
- **Pre-built:** `docker pull ghcr.io/tenstorrent/tt-vscode-toolkit:main`

### For Production (Self-Hosted)
- **Basic (fast):** `ghcr.io/tenstorrent/tt-vscode-toolkit:latest`
- **Full (with tt-metal):** `ghcr.io/tenstorrent/tt-vscode-toolkit:latest-full`

### For Koyeb Cloud (with N300 hardware)
- **Deploy script:** `./scripts/koyeb-deploy-direct.sh`
- Koyeb builds from `Dockerfile.koyeb` automatically

---

## Dockerfile Comparison

| Dockerfile | Purpose | Build Time | Image Size | tt-metal | Registry |
|------------|---------|------------|------------|----------|----------|
| `Dockerfile` | Basic cloud-ready | ~5 min | ~500MB | ❌ No | ghcr.io |
| `Dockerfile.full` | Full with tt-metal | ~20 min | ~3GB | ✅ Pre-built | ghcr.io |
| `Dockerfile.koyeb` | Koyeb + N300 optimized | ~5 min build<br>~10 min startup | ~500MB | ✅ Startup install | Koyeb private |
| `Dockerfile.production` | Self-hosted production | ~20 min | ~3GB | ✅ Pre-built | Your registry |

---

## Answering Your Questions

### Q: "Is docker-image and docker-image-full the same one we're sending to Koyeb?"

**No.** They're different:

**GitHub images** (`docker-image`, `docker-image-full`):
- Built by GitHub Actions
- Stored in `ghcr.io` (public registry)
- Anyone can pull and use them
- Used for: Public downloads, self-hosted deployments

**Koyeb images**:
- Built by Koyeb during deployment
- Stored in Koyeb's private registry
- Only your Koyeb org can access them
- Used for: Your specific Koyeb deployment

**Why separate?**
- Different build infrastructure (GitHub vs Koyeb)
- Different Dockerfiles (`Dockerfile.full` vs `Dockerfile.koyeb`)
- Different optimization goals (general purpose vs N300-specific)
- Different deployment models (pull pre-built vs build on deploy)

### Q: "If we were doing a release, could we promote it easily?"

**Yes!** The new `.github/workflows/release.yml` workflow shows how:

**Automated release process:**
1. Push a version tag: `git tag v0.0.254 && git push --tags`
2. GitHub Actions automatically:
   - ✅ Runs full test suite (mocha)
   - ✅ Builds and packages VSIX
   - ✅ Creates GitHub Release with VSIX attached
   - ✅ Tags Docker images with version number
   - ✅ Updates `latest` tags
   - ✅ (Optional) Publishes to VS Code Marketplace

**Manual promotion:**
```bash
# 1. Update version
npm version patch  # or minor, major
# Creates: v0.0.255

# 2. Push tag
git push --tags

# 3. GitHub Actions does the rest!
```

**What users get:**
- **VSIX download:** From GitHub Releases page
- **Docker images:**
  - `ghcr.io/tenstorrent/tt-vscode-toolkit:0.0.254`
  - `ghcr.io/tenstorrent/tt-vscode-toolkit:latest`
- **Koyeb deployment:**
  ```bash
  git checkout v0.0.254
  ./scripts/koyeb-deploy-direct.sh
  ```

---

## Best Practices

### For Development
```bash
# Test locally first
npm run build && npm run package
docker build -t test .
docker run -p 8080:8080 -e PASSWORD=test test

# Then commit and push
git commit -am "New feature"
git push origin docker-image

# GitHub Actions will test and build automatically
```

### For Releases
```bash
# 1. Test everything
npm test  # Must pass

# 2. Update version
npm version patch

# 3. Tag and push
git push && git push --tags

# 4. Check GitHub Actions builds
# 5. Verify GitHub Release created
# 6. Test deployed version
```

### For Koyeb Deployments
```bash
# Always test locally first
npm test
npm run build && npm run package

# Then deploy to Koyeb
./scripts/koyeb-deploy-direct.sh

# Koyeb builds fresh from your local changes
# No need to push to git first (but recommended)
```

---

## Summary

**Three separate image ecosystems:**

1. **GitHub Container Registry** - Public pre-built images
   - For general use, self-hosted deployments
   - Built by GitHub Actions
   - Uses `Dockerfile` and `Dockerfile.full`

2. **Koyeb Private Registry** - Your deployment-specific images
   - Built during each Koyeb deployment
   - Uses `Dockerfile.koyeb`
   - Optimized for N300 hardware

3. **Local Images** - Development and testing
   - Built on your machine
   - All Dockerfiles available

**They don't overlap** - each serves a different purpose in the deployment pipeline.
