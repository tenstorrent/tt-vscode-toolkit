# Deployment Scripts

This directory contains scripts for building and deploying the TT-VSCode-Toolkit extension.

## Docker Build Scripts

- **build-docker.sh** - Builds Docker images locally with different profiles
- **docker-entrypoint.sh** - Container startup script used by all Dockerfiles

## Usage

Most scripts should be run from the project root:

```bash
# Build Docker image locally
./scripts/build-docker.sh
```

## Related Documentation

See [docs/deployment/](../docs/deployment/) for detailed deployment guides.
