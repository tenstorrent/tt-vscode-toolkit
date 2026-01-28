# Deployment Documentation

Simple deployment guides for the Tenstorrent VSCode Toolkit.

## 📚 Documentation Files

### [DEPLOYMENT.md](./DEPLOYMENT.md)
General deployment guide covering:
- Quick start with Docker/Podman
- Building and running locally
- Environment configuration
- Cloud deployment basics
- Troubleshooting

### [KOYEB.md](./KOYEB.md)
Koyeb-specific deployment guide covering:
- One-click deploy buttons
- Hardware acceleration (N300)
- Multiple deployment methods
- Koyeb configuration
- Cloud-specific troubleshooting

## 🚀 Quick Start

### Local Development
```bash
npm run container:build
npm run container:run
# Access at http://localhost:8080
```

### Deploy to Koyeb
Click the deploy button in [KOYEB.md](./KOYEB.md) or use:
```bash
./koyeb-deploy-direct.sh
```

## Getting Help

- Check [DEPLOYMENT.md](./DEPLOYMENT.md) for local/general deployment issues
- Check [KOYEB.md](./KOYEB.md) for Koyeb-specific issues
- [GitHub Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues)
- [Tenstorrent Discord](https://discord.gg/tenstorrent)
