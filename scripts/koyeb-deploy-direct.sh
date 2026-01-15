#!/bin/bash
# Direct deploy to Koyeb without registry
# Uses "koyeb deploy" to build and deploy in one step
# Perfect for testing!

set -e

# Only use colors if running in a terminal (not in CI/logs)
if [ -t 1 ]; then
    GREEN='\033[0;32m'
    BLUE='\033[0;34m'
    YELLOW='\033[1;33m'
    NC='\033[0m'
else
    GREEN=''
    BLUE=''
    YELLOW=''
    NC=''
fi

echo -e "${BLUE}🚀 Koyeb Direct Deploy (No Registry!)${NC}"
echo ""

# Config
APP_NAME="${APP_NAME:-tt-vscode-toolkit}"
SERVICE_NAME="${SERVICE_NAME:-vscode}"
PASSWORD="${PASSWORD:-$(openssl rand -base64 12 | tr -d '=+/' | cut -c1-12)}"

# Check Koyeb CLI
if ! command -v koyeb &> /dev/null; then
    echo "❌ Koyeb CLI not installed"
    echo "   Install: curl -fsSL https://cli.koyeb.com/install.sh | sh"
    exit 1
fi

# Check login
if ! koyeb services list &> /dev/null && ! koyeb organizations list &> /dev/null; then
    echo "⚠️  Cannot verify Koyeb login"
    echo "   Try: koyeb login"
    read -p "Continue anyway? [y/N]: " CONTINUE
    if [[ ! "$CONTINUE" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "App:     ${APP_NAME}"
echo "Service: ${SERVICE_NAME}"
echo "Pass:    ${PASSWORD}"
echo ""

# Build extension first
echo "1/2 Building extension..."
if ! npm run build > /tmp/build.log 2>&1; then
    echo "❌ Build failed. Check /tmp/build.log"
    exit 1
fi
if ! npm run package > /tmp/package.log 2>&1; then
    echo "❌ Package failed. Check /tmp/package.log"
    exit 1
fi

# Deploy directly with koyeb deploy
echo "2/2 Deploying to Koyeb..."
echo "   (Building and deploying in one step - this takes a few minutes)"
echo ""

# Use koyeb deploy (builds remotely)
# This uploads your directory and builds the Docker image on Koyeb's servers
if koyeb deploy . ${APP_NAME}/${SERVICE_NAME} \
    --archive-builder docker \
    --archive-docker-dockerfile Dockerfile.koyeb \
    --ports 8080:http \
    --routes /:8080 \
    --env PASSWORD=${PASSWORD} \
    --env MESH_DEVICE=N300 \
    --regions na \
    --instance-type gpu-tenstorrent-n300s \
    --privileged \
    2>&1 | tee /tmp/koyeb-deploy.log; then

    echo ""
    echo "✅ Deployed!"
    echo ""

    # Try to get service URL
    sleep 3
    SERVICE_INFO=$(koyeb services get ${SERVICE_NAME} 2>/dev/null || echo "")
    SERVICE_URL=$(echo "$SERVICE_INFO" | grep -oE "https://[a-zA-Z0-9.-]+\.koyeb\.app" | head -1)

    if [ -n "$SERVICE_URL" ]; then
        echo "🌐 ${SERVICE_URL}?password=${PASSWORD}"
        echo "   (URL includes password for easy access)"
    else
        echo "🌐 Check Koyeb dashboard for URL"
        echo "   Add ?password=${PASSWORD} to the URL"
    fi
    echo ""
    echo "🔑 Password: ${PASSWORD}"
    echo ""
    echo "Watch logs: koyeb services logs ${SERVICE_NAME} -f"
else
    echo ""
    echo "❌ Deployment failed"
    echo ""
    echo "Check the logs above or:"
    echo "  cat /tmp/koyeb-deploy.log"
    exit 1
fi
