# Podman Quick Start

**The fastest way to run Tenstorrent VSCode Toolkit with Podman!**

## Prerequisites

```bash
# Install Podman
# macOS:
brew install podman

# Linux (Fedora/RHEL):
sudo dnf install podman

# Linux (Ubuntu/Debian):
sudo apt-get install podman

# Initialize Podman machine (macOS only)
podman machine init
podman machine start
```

## Three Ways to Run

### 1. Automated Script (Easiest)

```bash
./build-docker.sh
```

The script auto-detects Podman and builds everything for you!

### 2. NPM Scripts (Recommended)

```bash
# Build extension and container
npm run container:build

# Run the container
npm run container:run

# Access at http://localhost:8080
```

### 3. Direct Podman Commands

```bash
# Build extension
npm install
npm run build
npm run package

# Build container image
podman build -t tt-vscode-toolkit:basic .

# Run container
podman run -it -p 8080:8080 -e PASSWORD=tenstorrent tt-vscode-toolkit:basic

# Access at http://localhost:8080
```

## Using podman-compose

```bash
# Install podman-compose first
pip3 install podman-compose
# or: brew install podman-compose

# Copy environment file
cp .env.example .env
# Edit .env to set your password

# Start services
podman-compose -f podman-compose.yml up -d

# Stop services
podman-compose down
```

## Common Tasks

```bash
# List running containers
podman ps

# View logs
podman logs <container-id>
podman logs -f <container-id>  # Follow mode

# Shell access
podman exec -it <container-id> bash

# Stop container
podman stop <container-id>

# Remove container
podman rm <container-id>

# List images
podman images

# Remove image
podman rmi tt-vscode-toolkit:basic
```

## Custom Password

```bash
# Set password when running
podman run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic

# Or edit .env file for compose
echo "PASSWORD=mypassword" > .env
podman-compose up -d
```

## Persistent Data

```bash
# Create volume
podman volume create vscode-data

# Run with volume
podman run -it -p 8080:8080 \
  -v vscode-data:/home/coder/.local/share/code-server \
  -e PASSWORD=mypassword \
  tt-vscode-toolkit:basic

# List volumes
podman volume ls

# Inspect volume
podman volume inspect vscode-data
```

## Troubleshooting

**Podman not found?**
```bash
# Install it first!
brew install podman  # macOS
sudo dnf install podman  # Linux
```

**Machine not running (macOS)?**
```bash
podman machine start
```

**Port already in use?**
```bash
# Use different port
podman run -it -p 3000:8080 tt-vscode-toolkit:basic
# Access at http://localhost:3000
```

**Extension not showing up?**
```bash
# Check extension is installed
podman exec -it <container-id> code-server --list-extensions

# Should show: tenstorrent.tt-vscode-toolkit
```

## Advanced: Systemd Service (Linux)

Run as a system service:

```bash
# Generate systemd service file
podman generate systemd --new --name tt-vscode \
  > ~/.config/systemd/user/tt-vscode.service

# Enable and start
systemctl --user enable tt-vscode.service
systemctl --user start tt-vscode.service

# Auto-start on boot
loginctl enable-linger $USER

# Check status
systemctl --user status tt-vscode
```

## Why Podman?

- ✅ **No daemon** - Lightweight, no background process
- ✅ **Rootless** - Runs as your user, more secure
- ✅ **Docker-compatible** - Same commands work
- ✅ **Free** - Fully open source, no licensing

## Full Documentation

- **[PODMAN.md](PODMAN.md)** - Complete Podman guide
- **[DOCKER_QUICKSTART.md](DOCKER_QUICKSTART.md)** - General container guide
- **[DOCKER.md](DOCKER.md)** - Docker-specific details

## Support

- **Issues**: https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord**: https://discord.gg/tenstorrent
- **Podman Docs**: https://docs.podman.io

---

**That's it! You're running VSCode with the Tenstorrent extension in your browser! 🚀**
