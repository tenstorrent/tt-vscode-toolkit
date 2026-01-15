# Deploying with Tenstorrent Hardware on Koyeb

**🎉 Koyeb provides actual Tenstorrent N300 accelerators!**

This guide shows how to deploy with real hardware access.

## What Koyeb Offers

| Hardware Option | Chips | Use Case |
|----------------|-------|----------|
| **Single N300** | 2 chips (Wormhole) | Development, testing, single models |
| **3x N300** | 6 chips total | Production, multi-model serving |

## Quick Deploy with N300

### Option 1: Using koyeb.yaml (Recommended)

The `koyeb.yaml` file is pre-configured to request a single N300:

```yaml
accelerator:
  type: tenstorrent-n300
  count: 1
```

**Deploy:**
```bash
# Build and push
podman build -t registry.koyeb.com/myorg/tt-vscode:latest .
koyeb registry login
podman push registry.koyeb.com/myorg/tt-vscode:latest

# Deploy with hardware
koyeb service create --yaml koyeb.yaml
```

### Option 2: Interactive Script

```bash
./deploy-to-koyeb.sh
```

**You'll be asked:**
```
Add Tenstorrent hardware accelerator?
  1) Yes - Single N300 (2 chips, recommended)
  2) Yes - 3x N300 (6 chips total)
  3) No - Software only (learning mode)
Choice [1]:
```

Just press Enter to use the default (single N300)!

### Option 3: Quick Deploy Script

```bash
export KOYEB_ORG=myorg
./quick-deploy-koyeb.sh
```

This automatically requests a single N300 accelerator.

### Option 4: Manual CLI

```bash
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:latest \
  --ports 8080:http \
  --routes /:8080 \
  --env PASSWORD=mypassword \
  --env MESH_DEVICE=N300 \
  --regions was \
  --instance-type small \
  --accelerator tenstorrent-n300:1
```

## Hardware Configuration

### Environment Variables Set Automatically

When you request hardware, these are configured:

```yaml
env:
  - MESH_DEVICE=N300        # Hardware type
  - TT_METAL_HOME=/opt/tt-metal  # If pre-installed in image
```

### Accessing Hardware in Container

Once deployed with hardware:

```bash
# In the VSCode terminal
tt-smi                    # See your N300!
tt-smi -s                # JSON status

# Your extension detects it automatically
# Hardware lessons become fully functional
```

## What Works with Hardware

✅ **All 16 lessons fully functional**
✅ **Hardware detection** (sees real N300)
✅ **Model inference** (Llama, Qwen, etc.)
✅ **vLLM production serving**
✅ **Image generation** (Stable Diffusion)
✅ **Real performance testing**

## Container Image Considerations

### Basic Image (Current)
- ✅ Works with hardware
- ⚠️ Need to install tt-metal inside container
- ⚠️ Or connect to pre-installed system tt-metal

### Full Image (Recommended for Hardware)
Use `Dockerfile.full` which includes tt-metal dependencies:

```bash
# Build full image
podman build -f Dockerfile.full -t registry.koyeb.com/myorg/tt-vscode:full .
podman push registry.koyeb.com/myorg/tt-vscode:full

# Deploy with hardware
koyeb service create tt-vscode \
  --docker registry.koyeb.com/myorg/tt-vscode:full \
  --accelerator tenstorrent-n300:1 \
  ...
```

### With Pre-installed tt-metal

If Koyeb provides images with tt-metal pre-installed:

```bash
# Use base image with tt-metal
FROM koyeb/tt-metal:latest  # (if available)

# Add code-server and extension
...
```

## Verification After Deployment

Once deployed, check hardware access:

1. **Open the URL** from deployment logs
2. **Enter password**
3. **Open VSCode terminal** (Ctrl+`)
4. **Run:**
   ```bash
   tt-smi
   ```
5. **You should see:**
   ```
   Device 0: N300
   Device 1: N300
   Status: Active
   ```

## Running Lessons with Hardware

All hardware-dependent lessons now work:

### Hardware Detection (Lesson 1)
- Click "Run Hardware Detection"
- See your actual N300 appear!

### Model Inference (Lesson 7)
```bash
# In VSCode terminal
cd ~/tt-metal
# Run vLLM with your N300
```

### Production Serving
- Deploy vLLM server on N300
- Get OpenAI-compatible API
- Test with curl/Python

## Cost Considerations

### Single N300
- **Use case:** Development, testing, learning
- **Cost:** Check Koyeb pricing page
- **Perfect for:** Extension tutorials, model testing

### 3x N300
- **Use case:** Production, multi-model serving
- **Cost:** Higher (check Koyeb pricing)
- **Perfect for:** Production deployments, scaling

### Tips to Reduce Cost
1. Delete service when not using:
   ```bash
   koyeb service delete tt-vscode
   ```

2. Use smaller instance for basic testing

3. Schedule deployments (deploy when working, delete after)

## Hardware-Specific Configuration

### For N300 (2 chips)

```yaml
accelerator:
  type: tenstorrent-n300
  count: 1

env:
  - MESH_DEVICE=N300
  - WH_ARCH_YAML=wormhole_b0_80_arch_eth_dispatch.yaml
```

### For 3x N300 (6 chips)

```yaml
accelerator:
  type: tenstorrent-n300x3
  count: 1

env:
  - MESH_DEVICE=N300
  - TT_METAL_MESH_SHAPE=1x3  # or 3x1 depending on configuration
```

## Troubleshooting

### "Hardware not detected"

```bash
# Check hardware is allocated
lspci | grep Tenstorrent

# Reset devices
tt-smi -r

# Check environment
echo $MESH_DEVICE
```

### "tt-smi not found"

```bash
# Install tt-metal in container
# Or use Dockerfile.full image
# Or mount system tt-metal

export PATH=$PATH:/opt/tt-metal/bin
```

### "Device busy or in use"

```bash
# Reset all devices
tt-smi -r

# Check processes
ps aux | grep tt

# Restart container if needed
```

### "Out of memory errors"

```bash
# Increase instance size
koyeb service update tt-vscode --instance-type medium

# Or optimize model size/parameters
```

## Advanced: Device Passthrough

If the container needs direct device access:

```yaml
# In koyeb.yaml (may need special permissions)
privileged: true
devices:
  - /dev/tenstorrent
```

## Monitoring Hardware Usage

### In VSCode Terminal

```bash
# Watch device status
watch -n 1 tt-smi

# Monitor temperature/power
tt-smi -s | jq '.devices[0].telemetry'

# Check utilization
# (if Koyeb provides monitoring tools)
```

### From Koyeb Dashboard

- Check metrics tab
- View accelerator utilization
- Monitor performance

## Example Workflows

### Quick Model Test on N300

```bash
# 1. Deploy with hardware
./quick-deploy-koyeb.sh

# 2. Access IDE

# 3. In VSCode terminal:
cd ~/tt-metal
git clone https://github.com/tenstorrent/vllm
cd vllm
pip install -e .

# 4. Run model
python examples/llm_engine_example.py

# 5. See it run on your N300!
```

### Production vLLM Server

```bash
# 1. Deploy with 3x N300
./deploy-to-koyeb.sh
# Choose: 3x N300 configuration

# 2. Start vLLM server with mesh
python start-vllm-server.py \
  --model ~/models/Llama-3.1-8B-Instruct \
  --mesh 1x3

# 3. Get OpenAI-compatible endpoint
# 4. Use in production!
```

## Summary

✅ **koyeb.yaml** - Pre-configured for single N300
✅ **deploy-to-koyeb.sh** - Asks for hardware choice
✅ **quick-deploy-koyeb.sh** - Auto-requests N300
✅ **Manual CLI** - Full control with `--accelerator` flag

**With Koyeb's N300 hardware, all extension lessons work with real hardware! 🚀**

No more simulation or learning mode - you get actual Tenstorrent acceleration in the cloud!

## Resources

- **Koyeb Hardware**: https://www.koyeb.com/docs/accelerators
- **Tenstorrent Docs**: https://docs.tenstorrent.com
- **Extension Issues**: https://github.com/tenstorrent/tt-vscode-toolkit/issues
- **Discord**: https://discord.gg/tenstorrent
