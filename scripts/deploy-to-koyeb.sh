#!/bin/bash
# Streamlined Koyeb deployment script for Tenstorrent VSCode Toolkit
# Builds, pushes, and deploys in one command

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${GREEN}Koyeb Deployment - Tenstorrent VSCode Toolkit${NC}        ${BLUE}║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if Koyeb CLI is installed
if ! command -v koyeb &> /dev/null; then
    echo -e "${RED}❌ Koyeb CLI not found!${NC}"
    echo ""
    echo "Install with:"
    echo "  curl -fsSL https://cli.koyeb.com/install.sh | sh"
    echo ""
    exit 1
fi

# Check if logged in
if ! koyeb profile show &> /dev/null; then
    echo -e "${YELLOW}⚠️  Not logged in to Koyeb${NC}"
    echo ""
    echo -e "${BLUE}Logging in...${NC}"
    koyeb login
    echo ""
fi

# Detect container engine
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo -e "${RED}❌ Neither podman nor docker found${NC}"
    exit 1
fi

echo -e "${GREEN}Using container engine: $CONTAINER_CMD${NC}"
echo ""

# Get configuration
echo -e "${BLUE}━━━ Configuration ━━━${NC}"
echo ""

# Service name
read -p "Service name [tt-vscode-toolkit]: " SERVICE_NAME
SERVICE_NAME=${SERVICE_NAME:-tt-vscode-toolkit}

# Registry choice
echo ""
echo "Where should we push the image?"
echo "  1) Koyeb Registry (registry.koyeb.com) - Recommended"
echo "  2) GitHub Container Registry (ghcr.io)"
echo "  3) Docker Hub (docker.io)"
echo "  4) Custom registry"
read -p "Choice [1]: " REGISTRY_CHOICE
REGISTRY_CHOICE=${REGISTRY_CHOICE:-1}

case $REGISTRY_CHOICE in
    1)
        REGISTRY="registry.koyeb.com"
        read -p "Your Koyeb organization: " KOYEB_ORG
        IMAGE_URL="${REGISTRY}/${KOYEB_ORG}/${SERVICE_NAME}:latest"
        ;;
    2)
        REGISTRY="ghcr.io"
        read -p "Your GitHub username/org: " GITHUB_USER
        IMAGE_URL="${REGISTRY}/${GITHUB_USER}/${SERVICE_NAME}:latest"
        ;;
    3)
        REGISTRY="docker.io"
        read -p "Your Docker Hub username: " DOCKER_USER
        IMAGE_URL="${REGISTRY}/${DOCKER_USER}/${SERVICE_NAME}:latest"
        ;;
    4)
        read -p "Full image URL: " IMAGE_URL
        ;;
esac

# Password
read -p "Password for IDE [randomly generated]: " PASSWORD
if [ -z "$PASSWORD" ]; then
    PASSWORD=$(openssl rand -base64 16 | tr -d '=+/' | cut -c1-16)
    echo -e "${YELLOW}Generated password: ${PASSWORD}${NC}"
fi

# Region
echo ""
echo "Select region:"
echo "  1) was - Washington DC (US East)"
echo "  2) fra - Frankfurt (Europe)"
echo "  3) sin - Singapore (Asia)"
read -p "Choice [1]: " REGION_CHOICE
REGION_CHOICE=${REGION_CHOICE:-1}

case $REGION_CHOICE in
    1) REGION="was" ;;
    2) REGION="fra" ;;
    3) REGION="sin" ;;
esac

# Instance type
echo ""
echo "Select instance type:"
echo "  1) nano - 0.1 vCPU, 512MB (testing only)"
echo "  2) micro - 0.25 vCPU, 1GB (light use)"
echo "  3) small - 1 vCPU, 2GB (recommended)"
echo "  4) medium - 2 vCPU, 4GB (heavy use)"
read -p "Choice [3]: " INSTANCE_CHOICE
INSTANCE_CHOICE=${INSTANCE_CHOICE:-3}

case $INSTANCE_CHOICE in
    1) INSTANCE_TYPE="nano" ;;
    2) INSTANCE_TYPE="micro" ;;
    3) INSTANCE_TYPE="small" ;;
    4) INSTANCE_TYPE="medium" ;;
esac

# Tenstorrent Hardware
echo ""
echo "Add Tenstorrent hardware accelerator?"
echo "  1) Yes - Single N300 (2 chips, recommended)"
echo "  2) Yes - 3x N300 (6 chips total)"
echo "  3) No - Software only (learning mode)"
read -p "Choice [1]: " HARDWARE_CHOICE
HARDWARE_CHOICE=${HARDWARE_CHOICE:-1}

ACCELERATOR_FLAG=""
case $HARDWARE_CHOICE in
    1)
        ACCELERATOR_FLAG="--accelerator tenstorrent-n300:1"
        MESH_DEVICE="N300"
        ;;
    2)
        ACCELERATOR_FLAG="--accelerator tenstorrent-n300x3:1"
        MESH_DEVICE="N300"
        ;;
    3)
        MESH_DEVICE=""
        ;;
esac

echo ""
echo -e "${BLUE}━━━ Summary ━━━${NC}"
echo ""
echo "  Service:    ${SERVICE_NAME}"
echo "  Image:      ${IMAGE_URL}"
echo "  Password:   ${PASSWORD}"
echo "  Region:     ${REGION}"
echo "  Instance:   ${INSTANCE_TYPE}"
if [ -n "$ACCELERATOR_FLAG" ]; then
    echo "  Hardware:   Tenstorrent $(echo $ACCELERATOR_FLAG | cut -d: -f1 | cut -d' ' -f2)"
else
    echo "  Hardware:   None (software only)"
fi
echo ""
read -p "Continue? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-Y}

if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 1: Build extension
echo ""
echo -e "${BLUE}━━━ Step 1/5: Building Extension ━━━${NC}"
npm install
npm run build
npm run package

VSIX_FILE=$(ls -t tt-vscode-toolkit-*.vsix 2>/dev/null | head -1)
if [ -z "$VSIX_FILE" ]; then
    echo -e "${RED}❌ No .vsix file found${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Built: ${VSIX_FILE}${NC}"

# Step 2: Build container image
echo ""
echo -e "${BLUE}━━━ Step 2/5: Building Container Image ━━━${NC}"
$CONTAINER_CMD build -t ${SERVICE_NAME}:latest .
echo -e "${GREEN}✓ Image built${NC}"

# Step 3: Tag for registry
echo ""
echo -e "${BLUE}━━━ Step 3/5: Tagging Image ━━━${NC}"
$CONTAINER_CMD tag ${SERVICE_NAME}:latest ${IMAGE_URL}
echo -e "${GREEN}✓ Tagged: ${IMAGE_URL}${NC}"

# Step 4: Push to registry
echo ""
echo -e "${BLUE}━━━ Step 4/5: Pushing to Registry ━━━${NC}"
echo "Logging in to ${REGISTRY}..."

# Login to registry
case $REGISTRY_CHOICE in
    1)
        # Koyeb registry - use podman/docker login
        echo "Logging in to Koyeb registry..."
        echo "If this fails, get your token from: https://app.koyeb.com/account/api"
        $CONTAINER_CMD login registry.koyeb.com || {
            echo ""
            echo "Manual login required. Please run:"
            echo "  $CONTAINER_CMD login registry.koyeb.com"
            echo "  Username: your-koyeb-username"
            echo "  Password: your-api-token"
            exit 1
        }
        ;;
    2)
        # GitHub Container Registry
        echo "Login to GitHub Container Registry:"
        $CONTAINER_CMD login ghcr.io
        ;;
    3)
        # Docker Hub
        echo "Login to Docker Hub:"
        $CONTAINER_CMD login docker.io
        ;;
    4)
        # Custom - let user handle login
        read -p "Press enter after logging in to your registry..."
        ;;
esac

echo "Pushing image..."
$CONTAINER_CMD push ${IMAGE_URL}
echo -e "${GREEN}✓ Image pushed${NC}"

# Step 5: Deploy to Koyeb
echo ""
echo -e "${BLUE}━━━ Step 5/5: Deploying to Koyeb ━━━${NC}"

# Check if service exists
if koyeb service get ${SERVICE_NAME} &> /dev/null; then
    echo -e "${YELLOW}Service ${SERVICE_NAME} exists. Updating...${NC}"
    UPDATE_CMD="koyeb service update ${SERVICE_NAME} --docker-image ${IMAGE_URL} --env PASSWORD=${PASSWORD}"
    if [ -n "$MESH_DEVICE" ]; then
        UPDATE_CMD="$UPDATE_CMD --env MESH_DEVICE=${MESH_DEVICE}"
    fi
    eval $UPDATE_CMD
else
    echo -e "${GREEN}Creating new service: ${SERVICE_NAME}${NC}"
    CREATE_CMD="koyeb service create ${SERVICE_NAME} --docker ${IMAGE_URL} --ports 8080:http --routes /:8080 --env PASSWORD=${PASSWORD} --regions ${REGION} --instance-type ${INSTANCE_TYPE}"
    if [ -n "$ACCELERATOR_FLAG" ]; then
        CREATE_CMD="$CREATE_CMD $ACCELERATOR_FLAG"
    fi
    if [ -n "$MESH_DEVICE" ]; then
        CREATE_CMD="$CREATE_CMD --env MESH_DEVICE=${MESH_DEVICE}"
    fi
    eval $CREATE_CMD
fi

# Wait a moment for deployment to start
sleep 3

# Get service info
echo ""
echo -e "${BLUE}━━━ Deployment Complete! ━━━${NC}"
echo ""

SERVICE_INFO=$(koyeb service get ${SERVICE_NAME} 2>/dev/null || echo "")
if [ -n "$SERVICE_INFO" ]; then
    # Try to extract URL (Koyeb CLI output format may vary)
    SERVICE_URL=$(echo "$SERVICE_INFO" | grep -oE "https://[a-zA-Z0-9.-]+\.koyeb\.app" | head -1)

    if [ -n "$SERVICE_URL" ]; then
        echo -e "${GREEN}🌐 URL:${NC}      ${SERVICE_URL}"
    else
        echo -e "${GREEN}🌐 URL:${NC}      Check Koyeb dashboard"
    fi
fi

echo -e "${GREEN}🔑 Password:${NC} ${PASSWORD}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "  1. Wait for deployment (check logs below)"
echo "  2. Visit the URL above"
echo "  3. Enter password: ${PASSWORD}"
echo "  4. Look for 'Tenstorrent' icon in VSCode sidebar"
echo ""

# Offer to watch logs
read -p "Watch deployment logs? [Y/n]: " WATCH_LOGS
WATCH_LOGS=${WATCH_LOGS:-Y}

if [[ "$WATCH_LOGS" =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${BLUE}━━━ Live Deployment Logs ━━━${NC}"
    echo -e "${YELLOW}(Press Ctrl+C to stop watching)${NC}"
    echo ""
    sleep 2
    koyeb service logs ${SERVICE_NAME} -f
fi

echo ""
echo -e "${GREEN}✅ Deployment complete!${NC}"
echo ""
echo "Useful commands:"
echo "  koyeb service get ${SERVICE_NAME}       # Check status"
echo "  koyeb service logs ${SERVICE_NAME} -f   # Watch logs"
echo "  koyeb service delete ${SERVICE_NAME}    # Delete service"
echo ""
