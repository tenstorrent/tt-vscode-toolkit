# Docker Deployment Guide

This guide explains how to deploy the Tenstorrent VSCode Toolkit in a Docker container using code-server (browser-based VSCode).

## Quick Start

### Option 1: Automated Build Script (Recommended)

```bash
./build-docker.sh
```

This interactive script will:
1. Build the extension package (.vsix)
2. Let you choose between basic or full image
3. Build the Docker image
4. Provide usage instructions

### Option 2: Manual Build

```bash
# 1. Build the extension
npm install
npm run build
npm run package

# 2. Build Docker image
docker build -t tt-vscode-toolkit:basic .

# 3. Run the container
docker run -it -p 8080:8080 -e PASSWORD=mypassword tt-vscode-toolkit:basic
```

### Option 3: Docker Compose

```bash
# Edit docker-compose.yml to set your password
export PASSWORD=mypassword

# Start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

## Image Variants

### Basic Image (`Dockerfile`)
- **Size:** ~500MB
- **Contents:** code-server + Tenstorrent extension
- **Use case:** Lightweight deployment, remote extension development
- **Build:** `docker build -t tt-vscode-toolkit:basic .`

### Full Image (`Dockerfile.full`)
- **Size:** ~2GB
- **Contents:** code-server + extension + tt-metal dependencies
- **Use case:** Complete development environment with all tools
- **Build:** `docker build -f Dockerfile.full -t tt-vscode-toolkit:full .`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PASSWORD` | `tenstorrent` | Password for accessing code-server |
| `SUDO_PASSWORD` | `tenstorrent` | Sudo password inside container |
| `TT_METAL_HOME` | `/home/coder/tt-metal` | Path to tt-metal installation (full image only) |

## Volume Mounts

The docker-compose.yml sets up persistent volumes for:

- **vscode-data:** Extension settings and configurations
- **models-data:** Downloaded AI models (can be large!)
- **scratchpad-data:** Your experimental code (`~/tt-scratchpad`)
- **workspace:** Bind mount to `./workspace` directory (optional)

## Cloud Deployment

### Koyeb

Create a Koyeb service using the Docker image:

```bash
# 1. Build and push to a registry
docker build -t your-registry/tt-vscode-toolkit:latest .
docker push your-registry/tt-vscode-toolkit:latest

# 2. Deploy to Koyeb via CLI
koyeb service create tt-vscode \
  --docker your-registry/tt-vscode-toolkit:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=your-secure-password \
  --regions was \
  --instance-type nano
```

Or use the Koyeb web interface:
1. Go to https://app.koyeb.com
2. Create new service → Docker
3. Enter your image URL
4. Set port to 8080
5. Add environment variable: `PASSWORD=your-password`
6. Deploy!

### Railway

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login and deploy
railway login
railway init
railway up
```

Configure in `railway.toml`:
```toml
[deploy]
startCommand = "code-server --bind-addr 0.0.0.0:8080 /home/coder"
healthcheckPath = "/healthz"
restartPolicyType = "always"

[[ports]]
port = 8080
```

### Fly.io

```bash
# Install flyctl
curl -L https://fly.io/install.sh | sh

# Deploy
fly launch --image tt-vscode-toolkit:basic
fly deploy
```

Configure in `fly.toml`:
```toml
[env]
  PASSWORD = "your-password"

[[services]]
  internal_port = 8080
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]
```

## Accessing the IDE

After deployment:
1. Open your browser to the deployed URL (e.g., `http://localhost:8080`)
2. Enter the password you set (default: `tenstorrent`)
3. The Tenstorrent extension is pre-installed and ready to use!

The welcome page should automatically open showing the walkthrough lessons.

## Development Workflow

### With Hardware Access

If you have Tenstorrent hardware and want to use it with the container:

```bash
docker run -it \
  -p 8080:8080 \
  -e PASSWORD=mypassword \
  --device=/dev/tenstorrent/0 \
  --device=/dev/tenstorrent/1 \
  tt-vscode-toolkit:full
```

### Without Hardware (Simulator)

The extension works without hardware for learning the content:

```bash
docker run -it -p 8080:8080 tt-vscode-toolkit:basic
```

Most lessons can be read and studied without hardware. Commands that require hardware will show appropriate error messages.

## Troubleshooting

### Extension Not Showing Up

```bash
# Enter the container
docker exec -it <container-id> bash

# Check installed extensions
code-server --list-extensions

# Manually install if needed
code-server --install-extension /path/to/extension.vsix
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
3. Try accessing via IP instead of localhost
4. Check container logs: `docker logs <container-id>`

## Building for Production

### Multi-stage Build (Smaller Image)

```dockerfile
# Stage 1: Build extension
FROM node:18 AS builder
WORKDIR /build
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build && npm run package

# Stage 2: Runtime
FROM codercom/code-server:latest
COPY --from=builder /build/tt-vscode-toolkit-*.vsix /tmp/extension.vsix
RUN code-server --install-extension /tmp/extension.vsix && rm /tmp/extension.vsix
# ... rest of Dockerfile
```

### Security Hardening

```dockerfile
# Run as non-root user (already done in our Dockerfile)
USER coder

# Use secrets for passwords
# Pass via docker secrets or environment variables at runtime
# Never hardcode in Dockerfile!

# Scan for vulnerabilities
# docker scan tt-vscode-toolkit:latest
```

## Advanced Configuration

### Custom Settings

Create a `settings.json` to bake in custom VSCode settings:

```json
{
  "workbench.colorTheme": "Solarized Dark",
  "terminal.integrated.defaultProfile.linux": "bash",
  "tenstorrent.showUnvalidatedLessons": true
}
```

Copy into image:
```dockerfile
COPY settings.json /home/coder/.local/share/code-server/User/settings.json
```

### Preloaded Models

To include models in the image (WARNING: very large!):

```dockerfile
# Download models during build
RUN mkdir -p /home/coder/models && \
    cd /home/coder/models && \
    huggingface-cli download Qwen/Qwen3-0.6B
```

Better approach: Use volume mounts to share models between containers.

## CI/CD Integration

### GitHub Actions

```yaml
name: Build Docker Image

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

      - name: Build Docker image
        run: docker build -t ghcr.io/${{ github.repository }}:latest .

      - name: Push to registry
        run: docker push ghcr.io/${{ github.repository }}:latest
```

## Resources

- [code-server Documentation](https://coder.com/docs/code-server/latest)
- [Docker Documentation](https://docs.docker.com/)
- [Koyeb Documentation](https://www.koyeb.com/docs)
- [Tenstorrent Discord](https://discord.gg/tenstorrent)

## Support

Issues with Docker deployment? Check:
1. This guide first
2. [FAQ.md](FAQ.md) for extension-specific issues
3. [GitHub Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues)
4. [Tenstorrent Discord](https://discord.gg/tenstorrent)
