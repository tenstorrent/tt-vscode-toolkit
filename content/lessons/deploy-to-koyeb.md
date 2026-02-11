---
id: deploy-to-koyeb
title: "Deploy Your Work to Koyeb"
description: >-
  Deploy any Python application to Koyeb with Tenstorrent N300 hardware access. Learn production deployment patterns with vLLM and adapt for any application.
category: deployment
tags:
  - deployment
  - koyeb
  - production
  - vllm
  - n300
supportedHardware:
  - n300
  - p100
  - p150
  - galaxy
status: validated
validatedOn:
  - n150
estimatedMinutes: 10
---

# Deploy Your Work to Koyeb

Deploy any Python application to Koyeb with Tenstorrent N300 hardware access. We'll use vLLM as the primary example, then show how to adapt for any application.

## What You'll Learn

- Deploy vLLM to production with N300 hardware
- Containerize Python applications for Tenstorrent
- Configure hardware access and permissions
- Production deployment best practices
- Adapt the pattern for any application

## Prerequisites

- Completed [Deploy tt-vscode-toolkit to Koyeb](command:tenstorrent.showLesson?["deploy-vscode-to-koyeb"]) (recommended)
- Koyeb CLI installed and authenticated
- Docker or Podman installed locally
- Completed [vLLM Production](command:tenstorrent.showLesson?["vllm-production"]) lesson (for Part 1)

---

## Part 1: Deploy vLLM to Koyeb

### Step 1: Review Your vLLM Setup

From Lesson 7 (vLLM Production), you learned to run:

```bash
python -m vllm.entrypoints.openai.api_server \
  --model ~/models/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --port 8000
```

Now we'll deploy this to Koyeb with N300 hardware.

---

### Step 2: Create vLLM Dockerfile

Instead of building everything from scratch, extend our published image:

**Dockerfile.vllm:**

```dockerfile
# Base on Tenstorrent's published image
FROM ghcr.io/tenstorrent/tt-vscode-toolkit:latest

# Switch to root to install additional packages (if needed)
USER root
# RUN apt-get update && apt-get install -y \
#     your-dependencies-here \
#     && rm -rf /var/lib/apt/lists/*

# Switch back to coder user
USER coder
WORKDIR /home/coder

# Install vLLM
RUN git clone https://github.com/tenstorrent/vllm.git && \
    cd vllm && \
    python3 -m venv vllm-env && \
    . vllm-env/bin/activate && \
    pip install -e .

# Download your model using pre-installed HuggingFace CLI
RUN mkdir -p models && \
    hf download Qwen/Qwen3-0.6B --local-dir models/Qwen3-0.6B

# Environment variables for your app
ENV MODEL_PATH=/home/coder/models/Qwen3-0.6B

# Expose your app's port
EXPOSE 8000

# Run your app
CMD ["/bin/bash", "-c", "source vllm/vllm-env/bin/activate && python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name Qwen/Qwen3-0.6B --port 8000 --host 0.0.0.0"]
```

**Benefits:**
- ✅ 50% fewer lines (was ~60, now ~30)
- ✅ No need to set up base system (Ubuntu, apt repos, users, permissions)
- ✅ HuggingFace CLI (`hf`) pre-installed
- ✅ tt-smi pre-installed
- ✅ All hardware permissions configured
- ✅ Just add your app!

---

### Step 3: Deploy to Koyeb

```bash
koyeb deploy . my-app/vllm \
  --archive-builder docker \
  --archive-docker-dockerfile Dockerfile.vllm \
  --ports 8000:http \
  --routes /:8000 \
  --env MESH_DEVICE=N300 \
  --regions na \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

**Deployment time:** 10-15 minutes (builds vLLM + downloads model)

**Cost optimization tip:** Pre-build the Docker image locally and push to a registry to reduce deployment time:

```bash
# Build locally
docker build -t registry.koyeb.com/yourorg/vllm:latest -f Dockerfile.vllm .

# Push to registry
docker push registry.koyeb.com/yourorg/vllm:latest

# Deploy from registry (much faster!)
koyeb services create vllm \
  --app my-app \
  --docker registry.koyeb.com/yourorg/vllm:latest \
  --ports 8000:http \
  --routes /:8000 \
  --env MESH_DEVICE=N300 \
  --regions na \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

---

### Step 4: Test Your vLLM Deployment

Get your service URL:

```bash
koyeb services get vllm
```

Test with curl:

```bash
curl https://vllm-<your-hash>.koyeb.app/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "prompt": "Explain Tenstorrent hardware in one sentence:",
    "max_tokens": 50
  }'
```

Or use the OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://vllm-<your-hash>.koyeb.app/v1",
    api_key="not-needed"  # vLLM doesn't require auth by default
)

response = client.completions.create(
    model="Qwen/Qwen3-0.6B",
    prompt="What is Tenstorrent?",
    max_tokens=50
)

print(response.choices[0].text)
```

---

## Part 2: Deploy Any Python Application

### General Pattern

The Dockerfile pattern works for any Python application:

1. **Base image:** `ghcr.io/tenstorrent/tt-vscode-toolkit:latest` (includes tt-smi, permissions, CLIs)
2. **Install your app:** Clone repo, install dependencies
3. **Set environment:** Any additional environment variables your app needs
4. **Expose ports:** Your app's port
5. **Deploy with:** `--privileged` and `gpu-tenstorrent-n300s`

**Benefits of using our base image:**
- ✅ tt-smi pre-installed
- ✅ HuggingFace CLI (`hf`) and Claude CLI (`claude`) ready to use
- ✅ Hardware permissions configured (video, render groups)
- ✅ MOTD system for helpful terminal messages
- ✅ Faster builds (only your app layer)
- ✅ Clean base for your applications

---

### Example: Custom Inference Server

Let's say you built a custom Flask API in Lesson 10:

**Dockerfile.custom**:

```dockerfile
# Base on Tenstorrent's published image
FROM ghcr.io/tenstorrent/tt-vscode-toolkit:latest

USER coder
WORKDIR /home/coder

# Copy your application
COPY --chown=coder:coder . /home/coder/app

# Install your dependencies
RUN cd app && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

# Expose your port
EXPOSE 5000

# Run your app
CMD ["/bin/bash", "-c", "cd app && source venv/bin/activate && python server.py"]
```

**Much simpler!** From 80 lines to 15 lines.

**Deploy:**

```bash
koyeb deploy . my-app/inference \
  --archive-builder docker \
  --archive-docker-dockerfile Dockerfile.custom \
  --ports 5000:http \
  --routes /:5000 \
  --env MESH_DEVICE=N300 \
  --regions na \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

---

### Example: Data Processing Pipeline

For batch processing (not a server):

**Dockerfile.batch**:

```dockerfile
# Base on Tenstorrent's published image
FROM ghcr.io/tenstorrent/tt-vscode-toolkit:latest

USER coder
WORKDIR /home/coder

# Your processing script
COPY --chown=coder:coder process.py /home/coder/

# Install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install torch ttnn numpy

# Run processing script
CMD ["/bin/bash", "-c", "source venv/bin/activate && python process.py"]
```

**Even simpler!** Just 13 lines total.

This runs once per deployment. For scheduled tasks, combine with Koyeb's job scheduling.

---

## Part 3: Production Considerations

### Scaling

**Auto-scaling configuration:**

```bash
koyeb services create vllm \
  --app my-app \
  --docker registry.koyeb.com/yourorg/vllm:latest \
  --min-scale 1 \
  --max-scale 3 \
  --autoscaling-average-cpu 70 \
  --ports 8000:http \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

**Multiple regions:**

```bash
--regions na,fra  # Deploy to US and Europe
```

**Load balancing:** Automatic across all instances

---

### Monitoring

**Check service health:**

```bash
koyeb services get vllm
```

**View logs:**

```bash
koyeb services logs vllm -f
```

**Metrics:** Available in Koyeb dashboard
- Request rate
- Response time
- CPU/Memory usage
- Hardware utilization

---

### Health Checks

Add health check endpoints to your application:

```python
# In your Flask/FastAPI app
@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/readiness")
def readiness():
    # Check if model is loaded, hardware accessible, etc.
    if model_loaded and hardware_ok:
        return {"status": "ready"}
    return {"status": "not ready"}, 503
```

Configure in Koyeb:

```bash
koyeb services create vllm \
  --app my-app \
  --docker registry.koyeb.com/yourorg/vllm:latest \
  --checks "8000:http:/health" \
  --ports 8000:http \
  --instance-type gpu-tenstorrent-n300s \
  --privileged
```

---

### Security

**API Authentication:**

Add authentication to your application:

```python
from fastapi import FastAPI, Header, HTTPException

app = FastAPI()

@app.get("/v1/completions")
def completions(authorization: str = Header(None)):
    if authorization != f"Bearer {os.getenv('API_KEY')}":
        raise HTTPException(status_code=401, detail="Unauthorized")
    # ... your logic ...
```

Set API key via environment variable:

```bash
koyeb services update vllm \
  --env API_KEY=your-secret-key
```

**Network isolation:** Koyeb services are isolated by default. Use private networking for service-to-service communication.

---

## Part 4: CI/CD Integration

### GitHub Actions Example

**.github/workflows/deploy-koyeb.yml**:

```yaml
name: Deploy to Koyeb

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install Koyeb CLI
        run: curl -fsSL https://cli.koyeb.com/install.sh | sh

      - name: Build and push Docker image
        env:
          KOYEB_TOKEN: ${{ secrets.KOYEB_TOKEN }}
        run: |
          docker build -t registry.koyeb.com/${{ github.repository_owner }}/vllm:${{ github.sha }} -f Dockerfile.vllm .
          echo "$KOYEB_TOKEN" | docker login registry.koyeb.com -u ${{ github.repository_owner }} --password-stdin
          docker push registry.koyeb.com/${{ github.repository_owner }}/vllm:${{ github.sha }}

      - name: Deploy to Koyeb
        env:
          KOYEB_TOKEN: ${{ secrets.KOYEB_TOKEN }}
        run: |
          koyeb services update vllm \
            --docker-image registry.koyeb.com/${{ github.repository_owner }}/vllm:${{ github.sha }}
```

**Setup:**
1. Get API token: https://app.koyeb.com/account/api
2. Add to GitHub Secrets: `KOYEB_TOKEN`
3. Push to main branch → automatic deployment

---

## Troubleshooting

### Hardware Not Accessible

**Error:** `Permission denied: /dev/tenstorrent/0`

**Solution:** Ensure `--privileged` flag is set:

```bash
koyeb services update vllm --privileged
```

And verify user is in correct groups in Dockerfile:

```dockerfile
RUN usermod -aG video appuser && \
    groupadd -f render && \
    usermod -aG render appuser
```

---

### Build Timeouts

**Error:** Build exceeds time limit

**Solutions:**
1. **Pre-build images:** Build locally, push to registry
2. **Reduce dependencies:** Only install what you need
3. **Use build cache:** Koyeb caches Docker layers
4. **Split into stages:** Multi-stage Docker builds

---

### Model Download Fails

**Error:** HuggingFace download timeout

**Solutions:**
1. **Pre-download in image:** Include model in Docker image
2. **Use registry:** Push image with model pre-downloaded
3. **Increase timeout:** Use `--health-checks-grace-period 300`

---

## Cost Optimization

**Tips:**
- Use smaller instance types for testing (`small` instead of `gpu-tenstorrent-n300s`)
- Delete services when not in use
- Use registry-based deployment to avoid rebuilds
- Set up auto-scaling to scale down during low usage

**Example cost-effective setup:**

```bash
# Development (no hardware)
koyeb services create vllm-dev \
  --docker registry.koyeb.com/yourorg/vllm:latest \
  --instance-type small

# Production (with hardware, auto-scales)
koyeb services create vllm-prod \
  --docker registry.koyeb.com/yourorg/vllm:latest \
  --instance-type gpu-tenstorrent-n300s \
  --min-scale 1 \
  --max-scale 3 \
  --autoscaling-average-cpu 70 \
  --privileged
```

---

## Summary

**What you learned:**
- ✅ Deploy vLLM to production with N300 hardware
- ✅ Containerize any Python app for Tenstorrent by extending our base image
- ✅ Simplify Dockerfiles from 80 lines to 15 lines
- ✅ Set up monitoring and health checks
- ✅ Integrate with CI/CD pipelines
- ✅ Optimize for production and cost

**Key pattern:**
1. Base image: `ghcr.io/tenstorrent/tt-vscode-toolkit:latest`
2. Add your application layer (clone, install, configure)
3. Deploy with `--privileged` and `gpu-tenstorrent-n300s`

**Why use the base image:**
- tt-smi, HuggingFace CLI, Claude CLI pre-installed
- Hardware permissions pre-configured
- MOTD system for better UX
- Much simpler Dockerfiles (15 lines vs 80 lines)
- Faster builds (only your app layer)

**Resources:**
- [Koyeb Documentation](https://www.koyeb.com/docs)
- [Tenstorrent GitHub](https://github.com/tenstorrent)
- [vLLM Documentation](https://docs.vllm.ai)

---

## Next Steps

✅ **You can now deploy any application with Tenstorrent hardware!**

**Continue your journey:**
1. 🎯 [Interactive Chat](command:tenstorrent.showLesson?["interactive-chat"]) - Integrate with VSCode Chat
2. 🖼️ [Image Generation](command:tenstorrent.showLesson?["image-generation"]) - Deploy image generation services
3. 🧠 [CS Fundamentals](command:tenstorrent.showLesson?["cs-fundamentals-01-computer"]) - Deep dive into hardware

**Share your deployment:**
- Production APIs running on Tenstorrent hardware
- Scalable inference services
- Custom AI applications

Your applications now have access to cutting-edge AI acceleration! 🚀
