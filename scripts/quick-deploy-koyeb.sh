#!/bin/bash
# SUPER QUICK Koyeb deployment - uses defaults for testing
# For production, use deploy-to-koyeb.sh with full configuration

set -e

# Quick config (edit these!)
SERVICE_NAME="${SERVICE_NAME:-tt-vscode-test}"
KOYEB_ORG="${KOYEB_ORG:-}"  # Set this to your Koyeb org
PASSWORD="${PASSWORD:-$(openssl rand -base64 12 | tr -d '=+/' | cut -c1-12)}"

# Detect container engine
CONTAINER_CMD="podman"
command -v podman &> /dev/null || CONTAINER_CMD="docker"

echo "🚀 Quick Deploy to Koyeb"
echo ""

# Check Koyeb org is set
if [ -z "$KOYEB_ORG" ]; then
    echo "⚠️  Set your Koyeb org first:"
    echo "  export KOYEB_ORG=your-org-name"
    echo "  ./quick-deploy-koyeb.sh"
    echo ""
    exit 1
fi

# Check if logged in to Koyeb (try multiple methods)
LOGIN_CHECK=false
if koyeb profile show &> /dev/null; then
    LOGIN_CHECK=true
elif koyeb service list &> /dev/null; then
    LOGIN_CHECK=true
elif koyeb org list &> /dev/null; then
    LOGIN_CHECK=true
fi

if [ "$LOGIN_CHECK" = "false" ]; then
    echo "⚠️  Cannot verify Koyeb login. Try: koyeb login"
    echo ""
    read -p "Continue anyway? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

IMAGE_URL="registry.koyeb.com/${KOYEB_ORG}/${SERVICE_NAME}:latest"

echo "Service: ${SERVICE_NAME}"
echo "Image:   ${IMAGE_URL}"
echo "Pass:    ${PASSWORD}"
echo ""

# Build
echo "1/4 Building extension..."
if ! npm run build > /tmp/build.log 2>&1; then
    echo "❌ Extension build failed. Check /tmp/build.log"
    exit 1
fi
if ! npm run package > /tmp/package.log 2>&1; then
    echo "❌ Extension package failed. Check /tmp/package.log"
    exit 1
fi

# Build container (using Koyeb-optimized Dockerfile)
echo "2/4 Building container..."
if ! $CONTAINER_CMD build -f Dockerfile.koyeb -t ${SERVICE_NAME}:latest . > /tmp/container-build.log 2>&1; then
    echo "❌ Container build failed. Check /tmp/container-build.log"
    tail -20 /tmp/container-build.log
    exit 1
fi

# Push
echo "3/4 Pushing to registry..."
$CONTAINER_CMD tag ${SERVICE_NAME}:latest ${IMAGE_URL}

# Login to registry using podman/docker directly
# Koyeb registry uses your Koyeb API token
echo "   Logging in to Koyeb registry..."
echo "   (Using Koyeb credentials from your login)"

# Try to push directly - Koyeb CLI should have set up auth
echo "   Pushing image (this may take a minute)..."
if ! $CONTAINER_CMD push ${IMAGE_URL} 2>&1 | tee /tmp/push.log; then
    echo ""
    echo "❌ Image push failed"
    echo ""
    echo "You may need to authenticate manually:"
    echo "  1. Get your Koyeb token from: https://app.koyeb.com/account/api"
    echo "  2. Login to registry:"
    echo "     $CONTAINER_CMD login registry.koyeb.com"
    echo "     Username: [your koyeb username or token]"
    echo "     Password: [your koyeb token]"
    echo ""
    tail -20 /tmp/push.log
    exit 1
fi

# Deploy
echo "4/4 Deploying to Koyeb..."
if koyeb service get ${SERVICE_NAME} &> /dev/null; then
    echo "   Updating existing service..."
    if ! koyeb service update ${SERVICE_NAME} \
        --docker-image ${IMAGE_URL} \
        --env PASSWORD=${PASSWORD} \
        --env MESH_DEVICE=N300 > /tmp/deploy.log 2>&1; then
        echo "❌ Service update failed"
        cat /tmp/deploy.log
        exit 1
    fi
else
    echo "   Creating new service..."
    if ! koyeb service create ${SERVICE_NAME} \
        --docker ${IMAGE_URL} \
        --ports 8080:http \
        --routes /:8080 \
        --env PASSWORD=${PASSWORD} \
        --env MESH_DEVICE=N300 \
        --regions was \
        --instance-type nano \
        --accelerator tenstorrent-n300:1 > /tmp/deploy.log 2>&1; then
        echo "❌ Service creation failed"
        cat /tmp/deploy.log
        exit 1
    fi
fi

echo ""
echo "✅ Deployed!"
echo ""

# Get URL
sleep 2
SERVICE_INFO=$(koyeb service get ${SERVICE_NAME} 2>/dev/null || echo "")
SERVICE_URL=$(echo "$SERVICE_INFO" | grep -oE "https://[a-zA-Z0-9.-]+\.koyeb\.app" | head -1)

if [ -n "$SERVICE_URL" ]; then
    echo "🌐 ${SERVICE_URL}"
else
    echo "🌐 Check Koyeb dashboard for URL"
fi
echo "🔑 ${PASSWORD}"
echo ""
echo "Watch logs: koyeb service logs ${SERVICE_NAME} -f"
echo ""
