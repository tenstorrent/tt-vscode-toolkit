# Deployment Scripts

This directory contains scripts for building and deploying the tt-vscode-toolkit extension.

## Docker Build Scripts

- **build-docker.sh** - Builds Docker images locally with different profiles
- **docker-entrypoint.sh** - Container startup script used by all Dockerfiles

## Koyeb Deployment Scripts

- **koyeb-deploy-direct.sh** - One-command deployment to Koyeb (recommended)
  - Builds extension locally
  - Uploads archive to Koyeb
  - Deploys with N300 hardware support
  - Used in: [Deploy tt-vscode-toolkit to Koyeb](../content/lessons/deploy-vscode-to-koyeb.md) lesson

- **deploy-to-koyeb.sh** - Alternative deployment script with registry push
- **quick-deploy-koyeb.sh** - Quick deployment helper
- **koyeb-registry-login.sh** - Helper for Koyeb registry authentication
- **test-koyeb-auth.sh** - Test Koyeb CLI authentication

## Usage

Most scripts should be run from the project root:

```bash
# Build extension and deploy to Koyeb
./scripts/koyeb-deploy-direct.sh

# Build Docker image locally
./scripts/build-docker.sh
```

## Related Documentation

See [docs/deployment/](../docs/deployment/) for detailed deployment guides.
