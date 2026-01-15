# Koyeb CLI Deployment Guide

Deploy to Koyeb without using the web UI - pure command line!

## Quick Start

### Install Koyeb CLI

```bash
# Install
curl -fsSL https://cli.koyeb.com/install.sh | sh

# Login
koyeb login

# Verify
koyeb profile show
```

## Three Ways to Deploy

### 1. Interactive Deployment (Recommended for First Time)

**The `deploy-to-koyeb.sh` script walks you through everything:**

```bash
./deploy-to-koyeb.sh
```

**What it does:**
- ✅ Builds extension and container
- ✅ Asks for configuration (registry, region, instance type)
- ✅ Pushes to registry
- ✅ Deploys to Koyeb
- ✅ Shows you the URL and password
- ✅ Offers to watch logs

**Perfect for:** First deployment, testing different configs

---

### 2. Quick Deploy (Recommended for Testing)

**The `quick-deploy-koyeb.sh` script uses defaults for speed:**

```bash
# Set your Koyeb org once
export KOYEB_ORG=your-org-name

# Deploy with one command!
./quick-deploy-koyeb.sh
```

**What it does:**
- ✅ Uses smart defaults (nano instance, was region)
- ✅ Generates random password
- ✅ Builds, pushes, deploys - all silent
- ✅ Shows URL and password at the end

**Perfect for:** Quick testing, iteration, development

**Customize:**
```bash
# Custom service name
SERVICE_NAME=my-test ./quick-deploy-koyeb.sh

# Custom password
PASSWORD=mypass123 ./quick-deploy-koyeb.sh

# Both
SERVICE_NAME=my-ide PASSWORD=secretpass ./quick-deploy-koyeb.sh
```

---

### 3. Manual CLI Commands (Full Control)

**For when you want complete control:**

```bash
# 1. Build and push image
npm run container:build
podman tag tt-vscode-toolkit:basic registry.koyeb.com/myorg/tt-vscode:latest
koyeb registry login
podman push registry.koyeb.com/myorg/tt-vscode:latest

# 2. Create service
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=mypassword \
  --regions was \
  --instance-type small

# 3. Watch logs
koyeb service logs tt-vscode -f
```

**Perfect for:** CI/CD, automation, scripts

---

## Koyeb CLI Commands Reference

### Service Management

```bash
# Create service
koyeb service create <name> --docker <image> [options]

# Update service (e.g., new image)
koyeb service update <name> --docker-image <image>

# Get service info
koyeb service get <name>

# List all services
koyeb service list

# Delete service
koyeb service delete <name>

# Redeploy (restart with same image)
koyeb service redeploy <name>
```

### Logs

```bash
# Follow logs (live)
koyeb service logs <name> -f

# Last 100 lines
koyeb service logs <name> --tail 100

# Logs since timestamp
koyeb service logs <name> --since 2026-01-12T10:00:00Z
```

### Registry

```bash
# Login to Koyeb registry
koyeb registry login

# Logout
koyeb registry logout
```

### Secrets (for sensitive data)

```bash
# Create secret
koyeb secret create <name> --value <value>

# List secrets
koyeb secret list

# Use in service
koyeb service create <name> ... --env PASSWORD=@my-password-secret
```

## Common Workflows

### Initial Deployment

```bash
# 1. Set your org
export KOYEB_ORG=myorg

# 2. Deploy
./deploy-to-koyeb.sh
# Follow prompts, choose options

# 3. Wait for "Service is ready!" in logs
# 4. Visit URL shown
```

### Quick Iteration During Testing

```bash
# Make changes to extension...

# Redeploy with one command
./quick-deploy-koyeb.sh

# Watch it start
koyeb service logs tt-vscode-test -f
```

### Update Existing Deployment

```bash
# Rebuild
npm run build && npm run package
podman build -t registry.koyeb.com/myorg/tt-vscode:latest .
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Update service with new image
koyeb service update tt-vscode --docker-image registry.koyeb.com/myorg/tt-vscode:latest

# Watch it redeploy
koyeb service logs tt-vscode -f
```

### Change Password

```bash
koyeb service update tt-vscode --env PASSWORD=newpassword
```

### Change Instance Size

```bash
koyeb service update tt-vscode --instance-type small
```

### Move to Different Region

```bash
koyeb service update tt-vscode --regions fra
```

## Comparison: CLI vs Web UI

| Feature | CLI | Web UI |
|---------|-----|--------|
| Speed | ⚡ Fast | 🐌 Slower |
| Automation | ✅ Easy | ❌ Manual |
| Image URL | 🤖 Automatic | 📝 Copy/paste |
| Configuration | 💾 Reusable | 🔄 Re-enter each time |
| Logs | 📊 Real-time | 📄 Dashboard |
| Perfect for | Development, Testing | First-time setup, Visual |

## Tips & Tricks

### 1. Use Environment Variables

```bash
# In ~/.bashrc or ~/.zshrc
export KOYEB_ORG=myorg
export KOYEB_REGION=was
export KOYEB_INSTANCE=small

# Now scripts use these by default
```

### 2. Create Aliases

```bash
# Add to ~/.bashrc or ~/.zshrc
alias kd='./deploy-to-koyeb.sh'
alias kq='./quick-deploy-koyeb.sh'
alias kl='koyeb service logs tt-vscode -f'
alias ks='koyeb service get tt-vscode'
```

### 3. Watch Logs in Separate Terminal

```bash
# Terminal 1: Deploy
./quick-deploy-koyeb.sh

# Terminal 2: Watch logs
koyeb service logs tt-vscode-test -f
```

### 4. Use Registry Credentials Helper

```bash
# Store registry credentials
podman login registry.koyeb.com
# or
docker login registry.koyeb.com

# Now pushes are automatic
```

### 5. Quick Delete for Testing

```bash
# Delete service when done testing
koyeb service delete tt-vscode-test

# Or keep it but scale down to nano
koyeb service update tt-vscode-test --instance-type nano
```

## Troubleshooting

### "koyeb: command not found"

```bash
# Install Koyeb CLI
curl -fsSL https://cli.koyeb.com/install.sh | sh

# Add to PATH (if needed)
export PATH="$HOME/.koyeb/bin:$PATH"
```

### "Authentication required"

```bash
koyeb login
```

### "Image not found" or "Pull error"

```bash
# Make sure you pushed the image
podman images | grep tt-vscode

# Push again
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Verify in registry
koyeb registry list  # (if this command exists)
```

### Service won't start

```bash
# Check logs for errors
koyeb service logs <name>

# Common issues:
# - Wrong port (should be 8080)
# - Image not public (use Koyeb registry)
# - Out of memory (increase instance type)
```

### Can't see deployment URL

```bash
# Get service details
koyeb service get <name>

# Or list all services
koyeb service list

# Or check in web dashboard
open https://app.koyeb.com
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Deploy to Koyeb

on:
  push:
    branches: [ main ]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Koyeb CLI
        run: curl -fsSL https://cli.koyeb.com/install.sh | sh

      - name: Build extension
        run: |
          npm ci
          npm run build
          npm run package

      - name: Build and push image
        run: |
          podman build -t registry.koyeb.com/${{ secrets.KOYEB_ORG }}/tt-vscode:latest .
          echo "${{ secrets.KOYEB_TOKEN }}" | koyeb login --token
          koyeb registry login
          podman push registry.koyeb.com/${{ secrets.KOYEB_ORG }}/tt-vscode:latest

      - name: Deploy
        run: |
          koyeb service update tt-vscode \
            --docker-image registry.koyeb.com/${{ secrets.KOYEB_ORG }}/tt-vscode:latest \
            --env PASSWORD=${{ secrets.VSCODE_PASSWORD }}
```

## Advanced: Using koyeb.yaml with CLI

```bash
# Deploy from config file
koyeb service create --yaml koyeb.yaml

# Update from config file
koyeb service update tt-vscode --yaml koyeb.yaml
```

## Comparison of Deployment Methods

| Method | Commands | Time | Use Case |
|--------|----------|------|----------|
| `deploy-to-koyeb.sh` | 1 | ~5 min | First time, production |
| `quick-deploy-koyeb.sh` | 1 | ~2 min | Testing, iteration |
| Manual CLI | 5-10 | ~3 min | CI/CD, automation |
| Web UI | N/A | ~10 min | Visual setup |

## Summary

✅ **For first deployment**: Use `./deploy-to-koyeb.sh` (interactive)
✅ **For quick testing**: Use `./quick-deploy-koyeb.sh` (fast)
✅ **For automation**: Use manual CLI commands
✅ **For CI/CD**: Use GitHub Actions example above

**No more manual URL entry in the web UI! 🎉**

The CLI makes Koyeb deployment fast, repeatable, and perfect for testing. Your container's helpful logs will guide users once deployed.

## Resources

- **Koyeb CLI Docs**: https://www.koyeb.com/docs/cli
- **Koyeb API**: https://www.koyeb.com/docs/api
- **Extension Issues**: https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord**: https://discord.gg/tenstorrent
