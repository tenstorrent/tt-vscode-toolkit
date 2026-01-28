# Deployment Guide

Deploy the Tenstorrent VSCode Toolkit as a browser-based IDE using code-server.

## Quick Start

### Option 1: Automated Script (Recommended)

```bash
./build-docker.sh
```

The script will:
1. Build the extension (.vsix)
2. Let you choose basic or full image
3. Build the Docker/Podman image
4. Provide usage instructions

### Option 2: NPM Scripts

```bash
# Build and run with Podman (recommended)
npm run container:build
npm run container:run

# Or with Docker
npm run docker:build
npm run docker:run
```

### Option 3: Compose (Best for Development)

```bash
# Copy example environment file
cp .env.example .env

# Edit .env to set your password
vim .env

# Start with compose
npm run container:compose   # podman-compose
# or
npm run docker:compose      # docker-compose
```

Access at: **http://localhost:8080**

## Manual Build

```bash
# 1. Build the extension
npm install
npm run build
npm run package

# 2. Build container image (works with docker or podman)
docker build -t tt-vscode-toolkit:basic .

# 3. Run the container
docker run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic
```

> **Note:** All commands work with both Docker and Podman. Just replace `docker` with `podman`.

## Image Variants

### Basic Image (`Dockerfile`)
- **Size:** ~500MB
- **Contents:** code-server + Tenstorrent extension
- **Use case:** Lightweight deployment, remote development
- **Build:** `docker build -t tt-vscode-toolkit:basic .`

### Full Image (`Dockerfile.full`)
- **Size:** ~2GB
- **Contents:** code-server + extension + tt-metal dependencies
- **Use case:** Complete development environment with all tools
- **Build:** `docker build -f Dockerfile.full -t tt-vscode-toolkit:full .`

### Koyeb Image (`Dockerfile.koyeb`)
- **Size:** ~800MB
- **Contents:** Optimized for Koyeb cloud deployment
- **Use case:** Koyeb cloud platform
- **Build:** `docker build -f Dockerfile.koyeb -t tt-vscode-toolkit:koyeb .`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PASSWORD` | `tenstorrent` | Password for accessing code-server |
| `SUDO_PASSWORD` | Same as PASSWORD | Sudo password inside container |
| `TT_METAL_HOME` | `/home/coder/tt-metal` | Path to tt-metal (full image only) |
| `MESH_DEVICE` | - | Hardware type (N300, T3K, etc.) |

**⚠️ IMPORTANT:** Always set a custom password for production deployments!

## Common Commands

```bash
# Build basic image
npm run container:build

# Build full image with tt-metal
npm run container:build-full

# Run container
npm run container:run

# Start with compose (persistent storage)
npm run container:compose

# Stop compose
npm run container:compose-down

# View logs
docker logs <container-id>

# Shell access
docker exec -it <container-id> bash

# List running containers
docker ps
```

## Persistent Storage (Compose)

The `docker-compose.yml` / `podman-compose.yml` set up volumes for:

- **vscode-data:** Extension settings and configurations
- **models-data:** Downloaded AI models (can be large!)
- **scratchpad-data:** Your experimental code (`~/tt-scratchpad`)
- **workspace:** Bind mount to `./workspace` directory (optional)

## Cloud Deployment

### Koyeb (Recommended)

See [KOYEB.md](./KOYEB.md) for complete Koyeb deployment guide.

Quick deploy buttons available for:
- Git-based deployment (builds from source)
- Docker image deployment (uses pre-built image)
- Hardware-accelerated instances (N300 support)

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch --image tt-vscode-toolkit:basic
fly deploy
```

## Accessing the IDE

After deployment:
1. Open your browser to the deployed URL (e.g., `http://localhost:8080`)
2. Enter the password you set
3. The Tenstorrent extension is pre-installed and ready to use!

The welcome page should automatically open with walkthrough lessons.

## Development Workflow

### With Hardware Access

If you have Tenstorrent hardware:

```bash
docker run -it \
  -p 8080:8080 \
  -e PASSWORD=mypassword \
  --device=/dev/tenstorrent/0 \
  --device=/dev/tenstorrent/1 \
  tt-vscode-toolkit:full
```

### Without Hardware (Learning Mode)

```bash
docker run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic
```

Most lessons can be read and studied without hardware. Hardware-dependent commands will show appropriate guidance.

## Troubleshooting

### Extension Not Showing Up

```bash
# Enter the container
docker exec -it <container-id> bash

# Check installed extensions
code-server --list-extensions

# Should see: tenstorrent.tt-vscode-toolkit
```

### Port Already in Use

```bash
# Use a different port
docker run -it -p 3000:8080 tt-vscode-toolkit:basic
# Access at http://localhost:3000
```

### Permission Issues

```bash
# Fix ownership inside container
docker exec -it <container-id> sudo chown -R coder:coder /home/coder
```

### Cannot Access from Browser

1. Check container is running: `docker ps`
2. Check firewall rules allow port 8080
3. Try accessing via IP instead of localhost: `http://192.168.x.x:8080`
4. Check container logs: `docker logs <container-id>`

### Compose Command Not Found

```bash
# Install podman-compose
pip3 install podman-compose
# or: brew install podman-compose

# Install docker-compose
pip3 install docker-compose
# or: brew install docker-compose
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Build Container Image

on:
  push:
    branches: [ main ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build extension
        run: |
          npm install
          npm run build
          npm run package

      - name: Build container image
        run: docker build -t ghcr.io/${{ github.repository }}:latest .

      - name: Push to registry
        run: docker push ghcr.io/${{ github.repository }}:latest
```

## Security Best Practices

### 1. Strong Password

```bash
# Generate a strong password
PASSWORD=$(openssl rand -base64 32)
docker run -it -p 8080:8080 -e PASSWORD=$PASSWORD tt-vscode-toolkit:basic
echo "Save this password: $PASSWORD"
```

### 2. Use Secrets for Production

```bash
# Docker secrets
docker secret create vscode_password password.txt
docker service create --secret vscode_password tt-vscode-toolkit:basic

# Environment file (compose)
echo "PASSWORD=$(openssl rand -base64 32)" > .env
docker-compose up -d
```

### 3. Keep Image Updated

```bash
# Rebuild regularly
npm run container:build
docker-compose up -d --build
```

### 4. Run as Non-Root

The Dockerfiles already use non-root user `coder` (UID 1000). No additional configuration needed.

## File Locations in Container

- **Extension:** `/home/coder/.local/share/code-server/extensions/`
- **Workspace:** `/home/coder/`
- **Scratchpad:** `/home/coder/tt-scratchpad/`
- **Models:** `/home/coder/models/`
- **Settings:** `/home/coder/.local/share/code-server/User/settings.json`

## Building for Production

### Multi-Architecture Build

```bash
# Build for multiple architectures
docker buildx create --use
docker buildx build --platform linux/amd64,linux/arm64 \
  -t tt-vscode-toolkit:latest \
  --push .
```

### Optimize Image Size

```bash
# Use basic image for smaller size
docker build -t tt-vscode-toolkit:basic .

# Clean up build cache
docker builder prune
```

## Resources

- **Koyeb Deployment:** [KOYEB.md](./KOYEB.md)
- **code-server Documentation:** https://coder.com/docs/code-server
- **Docker Documentation:** https://docs.docker.com/
- **Podman Documentation:** https://docs.podman.io/
- **Extension Issues:** https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Tenstorrent Discord:** https://discord.gg/tenstorrent

## Support

Issues with deployment?

1. Check this guide first
2. Review [GitHub Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues)
3. Join [Tenstorrent Discord](https://discord.gg/tenstorrent)
4. Check code-server docs for VSCode-specific issues
