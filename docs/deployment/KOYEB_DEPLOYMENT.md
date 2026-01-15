# Koyeb Deployment Guide

Deploy the Tenstorrent VSCode Toolkit to Koyeb cloud platform.

## What You'll See in Koyeb Logs

When your container starts on Koyeb, you'll see helpful logs like this:

```
╔════════════════════════════════════════════════════════════╗
║  Tenstorrent VSCode Toolkit                                ║
║  Browser-based VSCode with TT Extension Pre-installed      ║
╚════════════════════════════════════════════════════════════╝

📍 Environment: Koyeb

🚀 YOUR IDE IS STARTING...

═══════════════════════════════════════════════════════════
📝 IMPORTANT - SAVE THESE DETAILS:
═══════════════════════════════════════════════════════════

  🌐 Access URL:  https://your-app.koyeb.app
  🔑 Password:    your-password-here

═══════════════════════════════════════════════════════════

📋 NEXT STEPS (Koyeb):

  1. Wait for 'Service is ready' message below
  2. Click the URL in Koyeb dashboard or visit:
     https://your-app.koyeb.app
  3. Enter password: your-password-here
  4. Look for 'Tenstorrent' in the VSCode sidebar (TT icon)
  5. Click 'Welcome Page' to start learning!

🎯 WHAT'S INCLUDED:

  ✅ VSCode in your browser (code-server)
  ✅ Tenstorrent extension pre-installed
  ✅ 16 interactive hardware lessons
  ✅ Production deployment guides
  ✅ Template scripts and examples

🔍 HEALTH CHECK:
  Endpoint: https://your-app.koyeb.app/healthz

═══════════════════════════════════════════════════════════
📡 CODE-SERVER LOGS:
═══════════════════════════════════════════════════════════

[info] code-server 4.x.x started
[info] Using password from $PASSWORD
[info] HTTP server listening on http://0.0.0.0:8080

✅ Service is ready!

🎉 YOUR IDE IS NOW AVAILABLE AT:
   https://your-app.koyeb.app
```

## Quick Deployment (Method 1: Using koyeb.yaml)

### 1. Build and Push Image

```bash
# Build the image
npm run container:build

# Tag for your registry
podman tag tt-vscode-toolkit:basic registry.koyeb.com/your-org/tt-vscode:latest

# Login to Koyeb registry
podman login registry.koyeb.com

# Push image
podman push registry.koyeb.com/your-org/tt-vscode:latest
```

### 2. Edit koyeb.yaml

```bash
# Edit the config file
vim koyeb.yaml
```

Update these fields:
```yaml
# Change this line:
image: registry.koyeb.com/your-org/tt-vscode:latest  # Your registry!

# Update password:
env:
  - key: PASSWORD
    value: your-secure-password-here  # CHANGE THIS!
```

### 3. Deploy

```bash
# Install Koyeb CLI
curl -fsSL https://cli.koyeb.com/install.sh | sh

# Login
koyeb login

# Deploy
koyeb service create --yaml koyeb.yaml

# Watch logs
koyeb service logs tt-vscode-toolkit -f
```

## Quick Deployment (Method 2: Koyeb Dashboard)

### 1. Build and Push Image (same as above)

### 2. Deploy via Web UI

1. Go to https://app.koyeb.com
2. Click **"Create Service"**
3. Choose **"Docker"**
4. Enter your image URL: `registry.koyeb.com/your-org/tt-vscode:latest`
5. Configure:
   - **Name:** `tt-vscode-toolkit`
   - **Port:** `8080`
   - **Protocol:** `http`
   - **Environment:**
     - Key: `PASSWORD`, Value: `your-secure-password`
6. Click **"Deploy"**

### 3. Watch the Logs!

In the Koyeb dashboard, go to your service and click **"Logs"** tab. You'll see the helpful startup messages showing:
- ✅ Your access URL
- ✅ Your password
- ✅ Next steps
- ✅ What's included

## Quick Deployment (Method 3: GitHub Container Registry)

### 1. Enable GitHub Actions

The repository includes `.github/workflows/docker-build.yml` which will:
- Build on every push to main
- Publish to `ghcr.io/your-org/tt-vscode-toolkit`

### 2. Get Image URL

After GitHub Actions runs:
```
ghcr.io/your-github-username/tt-vscode-toolkit:latest
```

### 3. Deploy to Koyeb

Use this image URL in Koyeb dashboard or update `koyeb.yaml`:

```yaml
deployment:
  docker:
    image: ghcr.io/your-github-username/tt-vscode-toolkit:latest
```

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `PASSWORD` | Yes | `tenstorrent` | Password for accessing IDE |
| `SUDO_PASSWORD` | No | Same as PASSWORD | Sudo password inside container |
| `TT_METAL_HOME` | No | `/home/coder/tt-metal` | Path to tt-metal (full image only) |

**⚠️ IMPORTANT:** Always set a custom password for production!

### Instance Types

| Type | CPU | RAM | Use Case |
|------|-----|-----|----------|
| nano | 0.1 vCPU | 512MB | Testing only |
| micro | 0.25 vCPU | 1GB | Light development |
| small | 1 vCPU | 2GB | **Recommended** - Full development |
| medium | 2 vCPU | 4GB | Heavy workloads |
| large | 4 vCPU | 8GB | Large models |

**Recommendation:** Start with `small` instance type.

### Regions

Choose a region close to you:
- `was` - Washington DC (US East)
- `fra` - Frankfurt (Europe)
- `sin` - Singapore (Asia)

## Accessing Your IDE

### From Koyeb Dashboard

1. Go to https://app.koyeb.com
2. Click your service
3. Click the **public URL** at the top
4. Enter your password
5. Done! Extension is already loaded.

### Direct URL

```
https://your-service-name-your-org.koyeb.app
```

### From Logs

The startup logs will show you the exact URL to visit!

## Verifying Extension is Installed

Once you access the IDE:

1. **Look for the Tenstorrent icon** in the left sidebar (orange/purple TT logo)
2. Click the icon to see lessons
3. Or go to Command Palette (Cmd/Ctrl+Shift+P) and type "Tenstorrent"
4. You should see all extension commands available

## Persistent Storage

⚠️ **Important:** Koyeb containers are ephemeral. Your work will be lost when the container restarts!

**Solutions:**

### Option 1: Use Git
```bash
# In the IDE terminal
git clone https://github.com/your-username/your-repo.git
# Make changes
git add .
git commit -m "My changes"
git push
```

### Option 2: Download Files
- Use VSCode's download feature (right-click file → Download)
- Download before shutting down

### Option 3: External Storage
- Mount external volume (Koyeb paid plans)
- Use S3/object storage for large files

## Troubleshooting

### "Service won't start"

**Check logs in Koyeb dashboard:**
```bash
# Or via CLI
koyeb service logs tt-vscode-toolkit -f
```

Common issues:
- Wrong image URL
- Image not public (or not logged in)
- Port not set to 8080

### "Extension not showing"

**Verify in logs:**
```
[info] Extension host agent started
```

**Check in IDE:**
- Cmd/Ctrl+Shift+X (Extensions)
- Search for "Tenstorrent"
- Should show as installed

### "Can't connect / 502 error"

- Wait 30-60 seconds for container to start
- Check health endpoint: `https://your-app.koyeb.app/healthz`
- Review logs for errors

### "Password doesn't work"

- Check environment variable in Koyeb dashboard
- Make sure `PASSWORD` is set correctly
- Try redeploying with correct password

## Cost Optimization

### Free Tier
- Koyeb offers free tier with limitations
- Good for testing and light development
- May sleep after inactivity

### Paid Plans
- For production use
- Persistent volumes available
- Better performance
- No sleep

### Cost Tips
1. Use `nano` or `micro` for testing
2. Delete service when not in use
3. Consider spot instances for development
4. Use `small` only when actively developing

## Security Best Practices

### 1. Strong Password
```yaml
env:
  - key: PASSWORD
    value: $(openssl rand -base64 32)  # Generate strong password
```

### 2. Use Koyeb Secrets
Instead of hardcoding password in `koyeb.yaml`:

```bash
# Create secret
koyeb secret create tt-vscode-password --value "your-strong-password"

# Reference in deployment
koyeb service create ... --env PASSWORD=@tt-vscode-password
```

### 3. Restrict Access
- Use VPN if possible
- Consider adding IP whitelist
- Use strong, unique password
- Change password regularly

### 4. Keep Image Updated
```bash
# Rebuild and redeploy regularly
podman build -t registry.koyeb.com/your-org/tt-vscode:latest .
podman push registry.koyeb.com/your-org/tt-vscode:latest
koyeb service redeploy tt-vscode-toolkit
```

## Updating the Deployment

### Rebuild and Redeploy

```bash
# 1. Build new extension version
npm install
npm run build
npm run package

# 2. Build new image
podman build -t registry.koyeb.com/your-org/tt-vscode:latest .

# 3. Push to registry
podman push registry.koyeb.com/your-org/tt-vscode:latest

# 4. Redeploy on Koyeb
koyeb service redeploy tt-vscode-toolkit
```

### Zero-Downtime Updates

```bash
# Deploy with version tags
podman tag tt-vscode-toolkit:basic registry.koyeb.com/your-org/tt-vscode:v1.0.1
podman push registry.koyeb.com/your-org/tt-vscode:v1.0.1

# Update service to new version
koyeb service update tt-vscode-toolkit --docker-image registry.koyeb.com/your-org/tt-vscode:v1.0.1
```

## Monitoring

### View Logs
```bash
# Real-time logs
koyeb service logs tt-vscode-toolkit -f

# Last 100 lines
koyeb service logs tt-vscode-toolkit --tail 100
```

### Check Status
```bash
koyeb service get tt-vscode-toolkit
```

### Health Check
```bash
curl https://your-app.koyeb.app/healthz
```

## Getting Help

- **Koyeb Documentation:** https://www.koyeb.com/docs
- **Extension Issues:** https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord:** https://discord.gg/tenstorrent
- **Koyeb Support:** support@koyeb.com

## Example: Complete Deployment

```bash
# 1. Build extension
npm install && npm run build && npm run package

# 2. Build image
podman build -t tt-vscode:latest .

# 3. Tag for Koyeb registry
podman tag tt-vscode:latest registry.koyeb.com/myorg/tt-vscode:latest

# 4. Login to Koyeb
podman login registry.koyeb.com

# 5. Push image
podman push registry.koyeb.com/myorg/tt-vscode:latest

# 6. Deploy
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=my-secure-password-123 \
  --regions was \
  --instance-type small

# 7. Watch it start
koyeb service logs tt-vscode -f

# 8. Open the URL shown in logs!
```

## Summary

✅ **Helpful startup logs** guide you through access
✅ **Auto-detects Koyeb** environment and shows correct URL
✅ **Clear instructions** in logs for next steps
✅ **Extension pre-installed** and ready to use
✅ **Easy deployment** via CLI, dashboard, or GitHub Actions

**Your Koyeb experience will be smooth with clear logs telling you exactly what to do! 🚀**
