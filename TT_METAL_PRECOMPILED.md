# Pre-compiled TT-Metalium and MOTD System

**Version:** 0.0.276
**Date:** 2026-01-27

## Overview

The Docker images now include:
1. **Pre-compiled TT-Metalium** - Ready to use immediately, no build time on first startup
2. **MOTD (Message of the Day) system** - Helpful terminal welcome messages with system information

## Changes Made

### 1. Pre-compiled TT-Metalium

**Dockerfile updates:**
- Added TT-Metalium clone and build step during image build (lines 36-60)
- Pre-configures Python virtual environment at `~/tt-metal/python_env`
- Sets up environment variables in `.bashrc` (TT_METAL_HOME, PYTHONPATH, PATH)
- Build time: ~15-20 minutes (one-time cost during image build)
- Users save: ~10 minutes on every container startup

**Benefits:**
- ✅ Instant access to TT-Metalium on first terminal open
- ✅ Python environment auto-activated
- ✅ No waiting for compilation on first use
- ✅ Consistent build across all deployments

### 2. MOTD System

**New files:**
- `content/motd.txt` - Editable MOTD content shown to all users
- `scripts/show-motd.sh` - Display script with dynamic system info

**How it works:**
1. `motd.txt` is copied to `/home/coder/.motd` during image build
2. `show-motd.sh` reads the MOTD and adds dynamic system information:
   - RAM and CPU cores
   - Tenstorrent hardware detection (via tt-smi)
   - TT-Metalium version and status
   - Python version
   - Environment variable status
3. Configured in `.bashrc` to run on terminal open
4. Only displays once per session (via `TENSTORRENT_MOTD_SHOWN` flag)

**What users see:**
```
╔════════════════════════════════════════════════════════════
║  Welcome to your Tenstorrent development environment!
╚════════════════════════════════════════════════════════════

🚀 Quick Start:
   • Open Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
   • Search: "Tenstorrent: Show Welcome Page"
   [...]

💻 System Information:
   RAM: 32GB  |  CPU Cores: 16
   Tenstorrent: n300 (x2)
   tt-metal: main@abc1234 (pre-compiled)
   Python env: activated (/home/coder/tt-metal/python_env)
   Python: 3.12.0

✅ Environment configured for tt-metal
```

### 3. docker-entrypoint.sh Simplification

**Removed:**
- Complex TT-Metalium installation logic (100+ lines)
- Old MOTD creation code (80+ lines)

**Kept:**
- Environment detection (Koyeb, Railway, Fly.io, Local)
- Startup banner with access information
- Device permission fixing
- CLI tool verification

**Result:**
- Simpler, more maintainable entrypoint script
- Faster container startup (no TT-Metalium build)
- Cleaner logs

### 4. Documentation Updates

**Updated lessons:**
- `deploy-vscode-to-koyeb.md` - Added TT-Metalium + MOTD to "What's Included"
- `deploy-to-koyeb.md` - Added to base image benefits

**Key messaging:**
- "TT-Metalium pre-compiled at ~/tt-metal (Python environment ready)"
- "MOTD (Message of the Day) displays on terminal open with system info"

## Customizing the MOTD

To customize the MOTD for your deployment:

1. **Edit the content file:**
   ```bash
   vim content/motd.txt
   ```

2. **Rebuild the image:**
   ```bash
   docker build -t myimage:latest .
   ```

3. **Deploy your custom image:**
   ```bash
   koyeb services create vscode \
     --docker myimage:latest \
     [... other flags ...]
   ```

The MOTD text is static and editable, while system information (RAM, CPU, hardware) is dynamically added by `show-motd.sh`.

## File Locations

**In the repository:**
- `content/motd.txt` - MOTD content (version controlled)
- `scripts/show-motd.sh` - Display script with dynamic info
- `Dockerfile` - Builds TT-Metalium and copies MOTD files
- `scripts/docker-entrypoint.sh` - Simplified startup script

**In the container:**
- `/home/coder/.motd` - Copy of MOTD content
- `/usr/local/bin/show-motd.sh` - Display script
- `/home/coder/tt-metal/` - Pre-compiled TT-Metalium
- `/home/coder/tt-metal/python_env/` - Python virtual environment
- `/home/coder/.bashrc` - Configured to show MOTD and activate TT-Metalium env

## Build Notes

**Image size impact:**
- tt-metal source: ~500MB
- TT-Metalium build artifacts: ~1-2GB
- Total image size: ~3-4GB (acceptable for pre-built convenience)

**Build time:**
- Without TT-Metalium: ~2-3 minutes
- With TT-Metalium: ~20-25 minutes (one-time cost)

**User startup time:**
- Old: ~10 minutes (TT-Metalium build on first startup)
- New: ~30 seconds (just image pull and start)

## Testing

To test the MOTD system locally:

```bash
# Build the image
docker build -t tt-vscode-toolkit:test .

# Run container
docker run -d -p 8080:8080 -e PASSWORD=test tt-vscode-toolkit:test

# Open terminal in the browser IDE
# You should see the MOTD immediately
```

To test with Koyeb hardware:

```bash
koyeb services create vscode-test \
  --docker ghcr.io/tenstorrent/tt-vscode-toolkit:latest \
  --instance-type gpu-tenstorrent-n300s \
  --privileged \
  [...]
```

Open a terminal and verify:
1. MOTD displays with system info
2. TT-Metalium is available at `~/tt-metal`
3. Python environment is activated
4. `import ttnn` works immediately

## Troubleshooting

**MOTD doesn't show:**
- Check if `.bashrc` includes the show-motd.sh source line
- Verify `show-motd.sh` has execute permissions
- Make sure terminal is using login shell (VSCode setting: `"args": ["-l"]`)

**TT-Metalium not found:**
- Verify image was built correctly
- Check `/home/coder/tt-metal` exists
- Look at build logs for compilation errors

**Python environment not activated:**
- Check `.bashrc` for activation commands
- Manually activate: `source ~/tt-metal/python_env/bin/activate`
- Verify `$VIRTUAL_ENV` is set

## Future Enhancements

Potential improvements:
- Add hardware-specific MOTD variants (n150, n300, T3000, etc.)
- Include recent lesson completions in MOTD
- Show available models in `~/models`
- Display resource usage (disk space, etc.)
- Add tips based on detected configuration

## Related Issues

- #TBD - Pre-compile TT-Metalium in Docker images
- #TBD - Add MOTD system for better UX

## Version History

- **0.0.276** (2026-01-27) - Added pre-compiled TT-Metalium and MOTD system
- **0.0.275** (2026-01-27) - Deployment lessons updated for published images
- **0.0.274** (2026-01-27) - Docker images published to ghcr.io
