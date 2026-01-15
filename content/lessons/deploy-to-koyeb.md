# Deploy Your Work to Koyeb

Deploy any Python application to Koyeb with Tenstorrent N300 hardware access. We'll use vLLM as the primary example, then show how to adapt for any application.

## What You'll Learn

- Deploy vLLM to production with N300 hardware
- Containerize Python applications for Tenstorrent
- Configure hardware access and permissions
- Production deployment best practices
- Adapt the pattern for any application

## Prerequisites

- Completed [Deploy tt-vscode-toolkit to Koyeb](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22deploy-vscode-to-koyeb%22%7D) (recommended)
- Koyeb CLI installed and authenticated
- Docker or Podman installed locally
- Completed [vLLM Production](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22vllm-production%22%7D) lesson (for Part 1)

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

Create `Dockerfile.vllm` in your project:

```dockerfile
# Use Ubuntu 24.04 for Tenstorrent APT repository compatibility
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    build-essential \
    python3 \
    python3-pip \
    python3-venv \
    pciutils \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Add Tenstorrent APT repository for tt-smi
RUN mkdir -p /etc/apt/keyrings && chmod 755 /etc/apt/keyrings && \
    curl -fsSL https://ppa.tenstorrent.com/tt-pkg-key.asc -o /etc/apt/keyrings/tt-pkg-key.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/tt-pkg-key.asc] https://ppa.tenstorrent.com/ubuntu/ noble main" > /etc/apt/sources.list.d/tenstorrent.list && \
    apt-get update && \
    apt-get install -y tt-smi && \
    rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -m -s /bin/bash appuser && \
    usermod -aG video appuser && \
    groupadd -f render && \
    usermod -aG render appuser

# Set up vLLM environment
USER appuser
WORKDIR /home/appuser

# Clone and install vLLM (Tenstorrent fork)
RUN git clone https://github.com/tenstorrent/vllm.git && \
    cd vllm && \
    python3 -m venv vllm-env && \
    . vllm-env/bin/activate && \
    pip install -e .

# Download model (example: Qwen3-0.6B)
RUN pip install --break-system-packages huggingface-hub && \
    mkdir -p /home/appuser/models && \
    huggingface-cli download Qwen/Qwen3-0.6B --local-dir /home/appuser/models/Qwen3-0.6B

# Environment variables
ENV TT_METAL_HOME=/opt/tt-metal
ENV PYTHONPATH=/opt/tt-metal
ENV PATH="/opt/tt-metal:${PATH}"
ENV MODEL_PATH=/home/appuser/models/Qwen3-0.6B
ENV MESH_DEVICE=N300

# Expose vLLM port
EXPOSE 8000

# Start vLLM server
CMD ["/bin/bash", "-c", "source vllm/vllm-env/bin/activate && python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name Qwen/Qwen3-0.6B --port 8000 --host 0.0.0.0"]
```

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

1. **Base image:** Ubuntu 24.04 (for Tenstorrent APT repo)
2. **Add Tenstorrent repo:** For tt-smi and dependencies
3. **Install your app:** Clone repo, install dependencies
4. **Configure permissions:** video/render groups for hardware
5. **Set environment:** TT_METAL_HOME, PYTHONPATH, MESH_DEVICE
6. **Expose ports:** Your app's port
7. **Deploy with:** `--privileged` and `gpu-tenstorrent-n300s`

---

### Example: Custom Inference Server

Let's say you built a custom Flask API in Lesson 10:

**Dockerfile.custom**:

```dockerfile
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies (same as vLLM example)
RUN apt-get update && apt-get install -y \
    curl git build-essential python3 python3-pip \
    pciutils ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Add Tenstorrent APT repo (same as vLLM example)
RUN mkdir -p /etc/apt/keyrings && chmod 755 /etc/apt/keyrings && \
    curl -fsSL https://ppa.tenstorrent.com/tt-pkg-key.asc -o /etc/apt/keyrings/tt-pkg-key.asc && \
    echo "deb [signed-by=/etc/apt/keyrings/tt-pkg-key.asc] https://ppa.tenstorrent.com/ubuntu/ noble main" > /etc/apt/sources.list.d/tenstorrent.list && \
    apt-get update && apt-get install -y tt-smi && \
    rm -rf /var/lib/apt/lists/*

# Create app user with hardware access
RUN useradd -m -s /bin/bash appuser && \
    usermod -aG video appuser && \
    groupadd -f render && \
    usermod -aG render appuser

USER appuser
WORKDIR /home/appuser

# Copy your application
COPY --chown=appuser:appuser . /home/appuser/app

# Install your dependencies
RUN cd app && \
    python3 -m venv venv && \
    . venv/bin/activate && \
    pip install -r requirements.txt

# Environment for Tenstorrent
ENV TT_METAL_HOME=/opt/tt-metal
ENV PYTHONPATH=/opt/tt-metal
ENV PATH="/opt/tt-metal:${PATH}"
ENV MESH_DEVICE=N300

# Expose your port
EXPOSE 5000

# Run your app
CMD ["/bin/bash", "-c", "cd app && source venv/bin/activate && python server.py"]
```

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
FROM ubuntu:24.04

# ... (same setup as above) ...

USER appuser
WORKDIR /home/appuser

# Your processing script
COPY --chown=appuser:appuser process.py /home/appuser/

# Install dependencies
RUN python3 -m venv venv && \
    . venv/bin/activate && \
    pip install torch ttnn numpy

ENV TT_METAL_HOME=/opt/tt-metal
ENV PYTHONPATH=/opt/tt-metal
ENV MESH_DEVICE=N300

# Run processing script
CMD ["/bin/bash", "-c", "source venv/bin/activate && python process.py"]
```

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
- ✅ Containerize any Python app for Tenstorrent
- ✅ Configure hardware access and permissions
- ✅ Set up monitoring and health checks
- ✅ Integrate with CI/CD pipelines
- ✅ Optimize for production and cost

**Key pattern:**
1. Ubuntu 24.04 base
2. Tenstorrent APT repository
3. User permissions (video, render groups)
4. Deploy with `--privileged` and `gpu-tenstorrent-n300s`

**Resources:**
- [Koyeb Documentation](https://www.koyeb.com/docs)
- [Tenstorrent GitHub](https://github.com/tenstorrent)
- [vLLM Documentation](https://docs.vllm.ai)

---

## Next Steps

✅ **You can now deploy any application with Tenstorrent hardware!**

**Continue your journey:**
1. 🎯 [Interactive Chat](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22interactive-chat%22%7D) - Integrate with VSCode Chat
2. 🖼️ [Image Generation](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22image-generation%22%7D) - Deploy image generation services
3. 🧠 [CS Fundamentals](command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22cs-fundamentals-01-computer%22%7D) - Deep dive into hardware

**Share your deployment:**
- Production APIs running on Tenstorrent hardware
- Scalable inference services
- Custom AI applications

Your applications now have access to cutting-edge AI acceleration! 🚀
