# Container Quick Start - TL;DR Version

Works with **Podman** (recommended) or Docker!

## Fastest Path to Running

```bash
# Option 1: Using the automated script (detects podman/docker automatically)
./build-docker.sh

# Option 2: Using npm scripts (podman)
npm run container:build
npm run container:run

# Option 3: Using compose (recommended for development)
cp .env.example .env
# Edit .env to set your password
npm run container:compose

# Option 4: Direct commands
podman build -t tt-vscode-toolkit:basic .
podman run -it -p 8080:8080 -e PASSWORD=tenstorrent tt-vscode-toolkit:basic
```

Access at: **http://localhost:8080**

> **Using Podman?** All commands work! Just use `podman` instead of `docker`.
> The build script auto-detects which you have installed.

## One-Command Build & Run

```bash
# Build extension, create Docker image, and run
npm run docker:build && npm run docker:run
```

## What You Get

- ✅ VSCode in your browser (code-server)
- ✅ Tenstorrent extension pre-installed
- ✅ All walkthroughs ready to use
- ✅ Persistent storage for your work

## Common Commands

```bash
# Build basic image (~500MB) - Podman
npm run container:build

# Build full image with tt-metal deps (~2GB) - Podman
npm run container:build-full

# Run the container - Podman
npm run container:run

# Start with compose (recommended) - Podman
npm run container:compose

# Stop compose - Podman
npm run container:compose-down

# Or use docker:* aliases (same commands)
npm run docker:build
npm run docker:run
npm run docker:compose

# View logs
podman logs <container-id>    # or docker logs

# Shell access
podman exec -it <container-id> bash    # or docker exec
```

## Set Custom Password

```bash
# Method 1: Environment variable (Podman)
podman run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic

# Method 2: Edit .env file (for compose)
echo "PASSWORD=mypassword" > .env
podman-compose up -d

# Method 3: compose with env var
PASSWORD=mypassword podman-compose up -d

# Works the same with docker/docker-compose!
```

## Deploy to Cloud

### Koyeb
```bash
podman build -t your-registry/tt-vscode:latest .
podman push your-registry/tt-vscode:latest
# Then use koyeb.yaml configuration
```

### Railway
```bash
railway login
railway init
railway up
```

### Fly.io
```bash
fly launch --image tt-vscode-toolkit:basic
fly deploy
```

## Troubleshooting

**Extension not showing up?**
```bash
podman exec -it <container-id> code-server --list-extensions
```

**Port already in use?**
```bash
podman run -it -p 3000:8080 tt-vscode-toolkit:basic
# Access at http://localhost:3000
```

**Permission errors?**
```bash
podman exec -it <container-id> sudo chown -R coder:coder /home/coder
```

**podman-compose not found?**
```bash
pip3 install podman-compose
# or: brew install podman-compose
```

## File Locations in Container

- Extension: `/home/coder/.local/share/code-server/extensions/`
- Workspace: `/home/coder/`
- Scratchpad: `/home/coder/tt-scratchpad/`
- Models: `/home/coder/models/`

## For Full Documentation

- **[PODMAN.md](PODMAN.md)** - Complete Podman guide (rootless, pods, systemd)
- **[DOCKER.md](DOCKER.md)** - Complete Docker guide (cloud deployment, CI/CD)
- Both guides cover:
  - Multi-stage builds
  - Security hardening
  - CI/CD integration
  - Cloud deployment details
  - Advanced configuration
