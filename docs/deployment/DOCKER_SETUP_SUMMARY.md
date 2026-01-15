# Docker Setup Summary

This document lists all Docker-related files created for deploying the Tenstorrent VSCode Toolkit.

## Files Created

### Core Docker Files

1. **`Dockerfile`** - Basic image (~500MB)
   - code-server + extension only
   - Lightweight deployment
   - Best for: Remote development, extension testing

2. **`Dockerfile.full`** - Full development image (~2GB)
   - Includes tt-metal dependencies
   - Python packages pre-installed
   - Best for: Complete development environment

3. **`docker-compose.yml`** - Orchestration configuration
   - Persistent volumes for data
   - Easy service management
   - Environment variable configuration

4. **`.dockerignore`** - Build optimization
   - Excludes unnecessary files from build context
   - Reduces image size
   - Speeds up build process

### Configuration Files

5. **`.env.example`** - Environment variable template
   - Copy to `.env` and customize
   - Sets passwords and paths
   - Used by docker-compose

6. **`koyeb.yaml`** - Koyeb deployment config
   - Ready-to-use cloud deployment
   - Just update registry URL
   - Includes resource limits and health checks

### Scripts and Automation

7. **`build-docker.sh`** - Interactive build script
   - Builds extension
   - Creates Docker image
   - Provides usage instructions
   - Offers registry tagging

8. **`.github/workflows/docker-build.yml`** - CI/CD automation
   - Automated builds on push
   - Publishes to GitHub Container Registry
   - Runs tests on pull requests
   - Multi-image builds (basic + full)

### Documentation

9. **`DOCKER.md`** - Comprehensive guide
   - Detailed deployment instructions
   - Cloud platform examples (Koyeb, Railway, Fly.io)
   - Troubleshooting section
   - Advanced configuration

10. **`DOCKER_QUICKSTART.md`** - TL;DR version
    - Quick commands
    - Common use cases
    - Fast reference

11. **`DOCKER_SETUP_SUMMARY.md`** - This file
    - Overview of all Docker files
    - Architecture explanation

## NPM Scripts Added

Added to `package.json`:

```json
{
  "scripts": {
    "docker:build": "Build basic Docker image",
    "docker:build-full": "Build full Docker image",
    "docker:run": "Run basic image",
    "docker:compose": "Start with docker-compose",
    "docker:compose-down": "Stop docker-compose"
  }
}
```

## Architecture Overview

```
┌─────────────────────────────────────────────────┐
│          Browser (http://localhost:8080)         │
└────────────────────┬────────────────────────────┘
                     │
                     │ HTTP/WebSocket
                     │
┌────────────────────▼────────────────────────────┐
│         code-server (VSCode in browser)          │
│  ┌─────────────────────────────────────────┐   │
│  │  Tenstorrent VSCode Extension           │   │
│  │  (pre-installed from .vsix)             │   │
│  │                                          │   │
│  │  - 16 Interactive Lessons               │   │
│  │  - Hardware Detection                    │   │
│  │  - Command Integration                   │   │
│  │  - Template Scripts                      │   │
│  └─────────────────────────────────────────┘   │
└────────────────────┬────────────────────────────┘
                     │
                     │ File System
                     │
┌────────────────────▼────────────────────────────┐
│              Docker Volumes                      │
│  - vscode-data:  Extension settings             │
│  - models-data:  AI models                      │
│  - scratchpad:   User experiments               │
└─────────────────────────────────────────────────┘
```

## Image Variants Comparison

| Feature | Basic Image | Full Image |
|---------|------------|-----------|
| Size | ~500MB | ~2GB |
| code-server | ✅ | ✅ |
| TT Extension | ✅ | ✅ |
| tt-metal deps | ❌ | ✅ |
| Python packages | Basic | Full |
| Build time | 2-3 min | 5-10 min |
| Use case | Learning, remote dev | Full development |

## Quick Start Commands

### Local Development
```bash
# Easiest way
./build-docker.sh

# Or with npm
npm run docker:build
npm run docker:run
```

### Production Deployment
```bash
# Build and tag
docker build -t myregistry/tt-vscode:latest .
docker push myregistry/tt-vscode:latest

# Deploy to cloud (example: Koyeb)
koyeb service create --yaml koyeb.yaml
```

### Docker Compose (Recommended)
```bash
cp .env.example .env
# Edit .env to set PASSWORD
docker-compose up -d
```

## Cloud Platform Support

### ✅ Tested/Documented
- **Koyeb** - Configuration provided (`koyeb.yaml`)
- **Railway** - Instructions in DOCKER.md
- **Fly.io** - Instructions in DOCKER.md
- **Local Docker** - Full support

### ⚙️ Should Work (Untested)
- **DigitalOcean App Platform**
- **Google Cloud Run**
- **AWS ECS/Fargate**
- **Azure Container Instances**

## Security Considerations

1. **Password Protection**
   - Default password: `tenstorrent`
   - ALWAYS change in production!
   - Use environment variables

2. **Non-root User**
   - Runs as `coder` user
   - Sudo available if needed
   - Better security posture

3. **Network Exposure**
   - Port 8080 exposed
   - Use reverse proxy with TLS in production
   - Consider VPN or authentication proxy

## Resource Requirements

### Basic Image
- **CPU:** 0.5-1 vCPU
- **RAM:** 512MB-1GB
- **Disk:** 1GB

### Full Image
- **CPU:** 1-2 vCPU
- **RAM:** 2-4GB
- **Disk:** 5GB

### With Models
- **Disk:** +20-100GB depending on models
- **RAM:** +8-16GB for model inference

## Testing the Setup

```bash
# 1. Build the extension
npm install
npm run build
npm run package

# 2. Build Docker image
docker build -t tt-vscode-test .

# 3. Run container
docker run -it -p 8080:8080 -e PASSWORD=test tt-vscode-test

# 4. Access in browser
# http://localhost:8080

# 5. Verify extension is loaded
# Look for "Tenstorrent" in the sidebar
```

## Next Steps

1. **Test locally**
   ```bash
   ./build-docker.sh
   ```

2. **Customize for your needs**
   - Edit Dockerfile to add dependencies
   - Modify docker-compose.yml for volumes
   - Update koyeb.yaml for cloud deployment

3. **Deploy to cloud**
   - Choose platform (Koyeb, Railway, Fly.io)
   - Build and push image to registry
   - Deploy using platform's CLI or UI

4. **Set up CI/CD**
   - Enable GitHub Actions workflow
   - Configure secrets (registry credentials)
   - Automated builds on every push

## Troubleshooting

See [DOCKER.md](DOCKER.md) for detailed troubleshooting, including:
- Extension not showing up
- Permission errors
- Port conflicts
- Container startup issues
- Performance problems

## Support

- **Issues:** https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord:** https://discord.gg/tenstorrent
- **Documentation:** [DOCKER.md](DOCKER.md), [README.md](README.md)

## Version History

- **v1.0** (2026-01-12) - Initial Docker setup
  - Basic and full Dockerfiles
  - docker-compose configuration
  - Build automation script
  - GitHub Actions workflow
  - Comprehensive documentation
