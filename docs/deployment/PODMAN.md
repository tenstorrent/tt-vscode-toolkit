# Podman Deployment Guide

This guide explains how to deploy the Tenstorrent VSCode Toolkit using **Podman** - a daemonless, rootless container engine that's fully compatible with Docker.

## Why Podman?

- ✅ **Daemonless** - No background service required
- ✅ **Rootless** - Better security, runs as your user
- ✅ **Docker-compatible** - Same commands, can even alias `docker=podman`
- ✅ **Pod support** - Native Kubernetes-style pod management
- ✅ **No licensing** - Fully open source

## Quick Start

### Prerequisites

```bash
# Check if podman is installed
podman --version

# Install podman if needed (macOS)
brew install podman

# Install podman if needed (Linux - Fedora/RHEL)
sudo dnf install podman

# Install podman if needed (Linux - Ubuntu/Debian)
sudo apt-get install podman

# Initialize podman machine (macOS only)
podman machine init
podman machine start
```

### Install podman-compose (Optional)

```bash
# Using pip
pip3 install podman-compose

# Or using brew (macOS)
brew install podman-compose
```

### Fastest Build & Run

```bash
# Option 1: Automated script (detects podman automatically)
./build-docker.sh

# Option 2: NPM scripts
npm run container:build
npm run container:run

# Option 3: Direct podman commands
npm run build && npm run package
podman build -t tt-vscode-toolkit:basic .
podman run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic
```

Access at: **http://localhost:8080**

## NPM Scripts (Podman)

All container operations work with npm:

```bash
# Build images
npm run container:build           # Basic image
npm run container:build-full      # Full image with dependencies

# Run container
npm run container:run             # Run basic image

# Using compose
npm run container:compose         # Start services
npm run container:compose-down    # Stop services

# Backwards compatibility (these call container:* scripts)
npm run docker:build              # Same as container:build
npm run docker:run                # Same as container:run
```

## Using podman-compose

```bash
# Copy environment file
cp .env.example .env
# Edit .env to set your password

# Start services
podman-compose -f podman-compose.yml up -d

# View logs
podman-compose logs -f

# Stop services
podman-compose down
```

## Direct Podman Commands

### Build Image

```bash
# Basic image (~500MB)
podman build -t tt-vscode-toolkit:basic .

# Full image with tt-metal deps (~2GB)
podman build -f Dockerfile.full -t tt-vscode-toolkit:full .
```

### Run Container

```bash
# Basic run
podman run -it -p 8080:8080 tt-vscode-toolkit:basic

# With custom password
podman run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic

# With persistent volume
podman run -it -p 8080:8080 \
  -v vscode-data:/home/coder/.local/share/code-server \
  -e PASSWORD=mypassword \
  tt-vscode-toolkit:basic

# Detached mode
podman run -d -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic
```

### Manage Containers

```bash
# List running containers
podman ps

# List all containers
podman ps -a

# View logs
podman logs <container-id>

# Follow logs
podman logs -f <container-id>

# Stop container
podman stop <container-id>

# Remove container
podman rm <container-id>

# Shell access
podman exec -it <container-id> bash
```

### Manage Images

```bash
# List images
podman images

# Remove image
podman rmi tt-vscode-toolkit:basic

# Prune unused images
podman image prune
```

## Rootless Advantages

Podman runs rootless by default (as your user):

```bash
# Check container user
podman run tt-vscode-toolkit:basic whoami
# Output: coder (not root!)

# No sudo needed for any podman commands
podman ps
podman build -t myimage .
```

## Podman-Specific Features

### Pods (Kubernetes-style)

```bash
# Create a pod
podman pod create --name tt-vscode-pod -p 8080:8080

# Run container in pod
podman run -d --pod tt-vscode-pod tt-vscode-toolkit:basic

# Manage pod
podman pod ps
podman pod stop tt-vscode-pod
podman pod rm tt-vscode-pod
```

### Generate Kubernetes YAML

```bash
# Run container
podman run -d --name tt-vscode -p 8080:8080 tt-vscode-toolkit:basic

# Generate Kubernetes deployment YAML
podman generate kube tt-vscode > tt-vscode-k8s.yaml

# Deploy to Kubernetes
kubectl apply -f tt-vscode-k8s.yaml
```

### Generate Systemd Service

```bash
# Create systemd service file
podman generate systemd --new --name tt-vscode > ~/.config/systemd/user/tt-vscode.service

# Enable and start service
systemctl --user enable tt-vscode.service
systemctl --user start tt-vscode.service

# Auto-start on boot
loginctl enable-linger $USER
```

## Podman Machine (macOS)

On macOS, Podman runs containers in a lightweight VM:

```bash
# Initialize machine (first time)
podman machine init

# Start machine
podman machine start

# Check status
podman machine list

# SSH into machine
podman machine ssh

# Stop machine
podman machine stop

# Remove machine
podman machine rm
```

## Docker Compatibility

Podman is fully Docker-compatible. You can alias:

```bash
# Add to ~/.bashrc or ~/.zshrc
alias docker=podman
alias docker-compose=podman-compose

# Now use familiar docker commands
docker build -t myimage .
docker run -it myimage
docker ps
```

## Differences from Docker

### What's the Same
- ✅ All `docker` commands work with `podman`
- ✅ Dockerfile syntax identical
- ✅ Image format identical (OCI standard)
- ✅ Registry push/pull works the same

### What's Different
- ⚠️ No daemon running in background
- ⚠️ Runs rootless by default (more secure!)
- ⚠️ `podman-compose` is separate install
- ⚠️ Some compose features might differ slightly

## Pushing to Registry

```bash
# Tag image
podman tag tt-vscode-toolkit:basic ghcr.io/tenstorrent/tt-vscode:latest

# Login to registry
podman login ghcr.io

# Push image
podman push ghcr.io/tenstorrent/tt-vscode:latest
```

## Cloud Deployment

Most cloud platforms support OCI images built with Podman:

### Build and Push

```bash
# Build
podman build -t myregistry.io/tt-vscode:latest .

# Push
podman push myregistry.io/tt-vscode:latest
```

### Deploy to Cloud

Then use platform CLI to deploy (Koyeb, Railway, Fly.io, etc.)

See [DOCKER.md](DOCKER.md) for platform-specific deployment instructions - they work the same with Podman-built images!

## Troubleshooting

### podman-compose not found

```bash
pip3 install podman-compose
# or
brew install podman-compose
```

### Permission denied

Podman is rootless - you shouldn't need sudo!

```bash
# Check if podman machine is running (macOS)
podman machine list
podman machine start

# Check if socket is accessible
podman info
```

### Port already in use

```bash
# Use different port
podman run -it -p 3000:8080 tt-vscode-toolkit:basic
# Access at http://localhost:3000
```

### Image build fails

```bash
# Make sure extension is built first
npm run build
npm run package

# Check .vsix file exists
ls -lh tt-vscode-toolkit-*.vsix
```

### Container won't start

```bash
# Check logs
podman logs <container-id>

# Check running containers
podman ps -a

# Remove and recreate
podman rm -f <container-id>
podman run -it -p 8080:8080 tt-vscode-toolkit:basic
```

## Performance Tips

### Use Volume Mounts

Bind mounts are slower on macOS. Use named volumes:

```bash
# Create volume
podman volume create vscode-data

# Use volume
podman run -it -p 8080:8080 \
  -v vscode-data:/home/coder/.local/share/code-server \
  tt-vscode-toolkit:basic
```

### Adjust Machine Resources (macOS)

```bash
# Stop machine
podman machine stop

# Adjust CPU and memory
podman machine set --cpus 4 --memory 8192

# Restart machine
podman machine start
```

## Migration from Docker

If you're switching from Docker:

```bash
# 1. Stop Docker containers
docker stop $(docker ps -q)

# 2. Export Docker image
docker save tt-vscode-toolkit:basic > tt-vscode.tar

# 3. Import to Podman
podman load < tt-vscode.tar

# 4. Run with Podman
podman run -it -p 8080:8080 tt-vscode-toolkit:basic
```

Or simply rebuild with Podman (recommended):

```bash
npm run container:build
```

## Resources

- **Podman Documentation**: https://docs.podman.io
- **Podman Desktop**: https://podman-desktop.io (GUI alternative)
- **Migration Guide**: https://docs.podman.io/en/latest/markdown/podman-docker.1.html
- **Troubleshooting**: https://github.com/containers/podman/blob/main/troubleshooting.md

## Next Steps

1. **Build and test locally**
   ```bash
   ./build-docker.sh
   ```

2. **Set up systemd service** (Linux)
   ```bash
   podman generate systemd --new --name tt-vscode > ~/.config/systemd/user/tt-vscode.service
   systemctl --user enable --now tt-vscode.service
   ```

3. **Deploy to cloud**
   - Build and push image
   - Use cloud platform CLI/UI to deploy
   - See [DOCKER.md](DOCKER.md) for platform guides

## Support

- **Issues**: https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord**: https://discord.gg/tenstorrent
- **Podman Help**: https://docs.podman.io

---

**Note**: All `docker-compose.yml` files in this repo also work with `podman-compose`. The `podman-compose.yml` file includes Podman-specific optimizations.
