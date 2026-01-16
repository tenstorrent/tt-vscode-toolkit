# Deployment Documentation

This directory contains comprehensive documentation for Docker and Koyeb deployment workflows.

---

## 🚀 Quick Deploy to Koyeb

Deploy your own cloud VSCode IDE with Tenstorrent N300 hardware in minutes:

### Option 1: Git-Based (Recommended for Development)

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=git&builder=dockerfile&repository=github.com/tenstorrent/tt-vscode-toolkit&branch=main&name=tt-vscode-toolkit)

Builds from source using Koyeb-optimized Dockerfile. **After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s** (Tenstorrent N300)
2. Set `PASSWORD` environment variable
3. Click Deploy

**Build time:** ~5-10 minutes

### Option 2: Docker Image (Recommended for Production)

[![Deploy to Koyeb](https://www.koyeb.com/static/images/deploy/button.svg)](https://app.koyeb.com/deploy?type=docker&image=ghcr.io/tenstorrent/tt-vscode-toolkit:latest&name=tt-vscode-toolkit&ports=8080;http;/&env[PASSWORD]=changeme)

Uses pre-built image from GitHub Container Registry. **After clicking:**
1. Select instance type: **gpu-tenstorrent-n300s**
2. Update `PASSWORD` environment variable
3. Click Deploy

**Deploy time:** ~2 minutes

> **Note:** Both methods require manual selection of the N300 instance type. See [DEPLOY_BUTTON_GUIDE.md](./DEPLOY_BUTTON_GUIDE.md) for details.

---

## Quick Start Guides

- **[DEPLOYMENT_METHODS.md](./DEPLOYMENT_METHODS.md)** - Overview of all deployment options
- **[DOCKER_QUICKSTART.md](./DOCKER_QUICKSTART.md)** - Get started with Docker in 5 minutes
- **[KOYEB_DEPLOYMENT.md](./KOYEB_DEPLOYMENT.md)** - Deploy to Koyeb cloud platform

## Docker Documentation

- **[DOCKER.md](./DOCKER.md)** - Complete Docker guide for the extension
- **[DOCKER_SETUP_SUMMARY.md](./DOCKER_SETUP_SUMMARY.md)** - Docker setup summary
- **[DOCKERFILE_GUIDE.md](./DOCKERFILE_GUIDE.md)** - Guide to the various Dockerfiles

## Podman Documentation

- **[PODMAN.md](./PODMAN.md)** - Using Podman instead of Docker
- **[PODMAN_QUICKSTART.md](./PODMAN_QUICKSTART.md)** - Podman quick start
- **[PODMAN_MIGRATION_SUMMARY.md](./PODMAN_MIGRATION_SUMMARY.md)** - Migrating from Docker to Podman

## Koyeb Cloud Deployment

- **[KOYEB_CLI_GUIDE.md](./KOYEB_CLI_GUIDE.md)** - Complete Koyeb CLI reference
- **[KOYEB_WITH_HARDWARE.md](./KOYEB_WITH_HARDWARE.md)** - Deploying with N300 hardware access
- **[KOYEB_LOGS_EXAMPLE.txt](./KOYEB_LOGS_EXAMPLE.txt)** - Example deployment logs

## Container Registries & Releases

- **[CONTAINER_REGISTRY_GUIDE.md](./CONTAINER_REGISTRY_GUIDE.md)** - Understanding GitHub vs Koyeb images, release workflow
- **[DEPLOY_BUTTON_GUIDE.md](./DEPLOY_BUTTON_GUIDE.md)** - Deploy to Koyeb button implementation and automated releases
- **[MAKING_IMAGES_PUBLIC.md](./MAKING_IMAGES_PUBLIC.md)** - How to make images publicly accessible for anyone to deploy

## Interactive Lessons

For step-by-step deployment tutorials, see:
- [Deploy tt-vscode-toolkit to Koyeb](../../content/lessons/deploy-vscode-to-koyeb.md)
- [Deploy Your Work to Koyeb](../../content/lessons/deploy-to-koyeb.md)

## File Locations

- **Dockerfiles:** Project root (`Dockerfile`, `Dockerfile.koyeb`, etc.)
- **Configuration:** Project root (`koyeb.yaml`, `.dockerignore`, etc.)
- **Scripts:** `../../scripts/` directory
- **Compose files:** Project root (`docker-compose.yml`, `podman-compose.yml`)

## Getting Help

- Check the [main README](../../README.md) for project overview
- Browse [interactive lessons](../../content/lessons/) in the VSCode extension
- Report issues at https://github.com/tenstorrent/tt-vscode-toolkit/issues
