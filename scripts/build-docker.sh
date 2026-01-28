#!/bin/bash
# Build script for Tenstorrent VSCode Podman/Docker image

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Detect container engine (podman or docker)
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
    COMPOSE_CMD="podman-compose"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
    COMPOSE_CMD="docker-compose"
else
    echo -e "${RED}Error: Neither podman nor docker found. Please install one of them.${NC}"
    exit 1
fi

echo -e "${BLUE}=== Tenstorrent VSCode Toolkit - Container Build ===${NC}"
echo -e "${GREEN}Using: $CONTAINER_CMD${NC}\n"

# Step 1: Build the extension
echo -e "${GREEN}Step 1: Building extension package...${NC}"
npm install
npm run build
npm run package

# Find the latest .vsix file
VSIX_FILE=$(ls -t tt-vscode-toolkit-*.vsix | head -1)

if [ -z "$VSIX_FILE" ]; then
    echo -e "${RED}Error: No .vsix file found. Build failed.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Extension built: $VSIX_FILE${NC}\n"

# Step 2: Choose Dockerfile
echo -e "${BLUE}Choose Docker image type:${NC}"
echo "1) Basic (code-server + extension only) - ~500MB"
echo "2) Full (includes tt-metal dependencies) - ~2GB"
read -p "Enter choice [1-2]: " CHOICE

case $CHOICE in
    1)
        DOCKERFILE="Dockerfile"
        IMAGE_NAME="tt-vscode-toolkit:basic"
        echo -e "${GREEN}Building basic image...${NC}"
        ;;
    2)
        DOCKERFILE="Dockerfile.full"
        IMAGE_NAME="tt-vscode-toolkit:full"
        echo -e "${GREEN}Building full image...${NC}"
        ;;
    *)
        echo -e "${RED}Invalid choice. Using basic image.${NC}"
        DOCKERFILE="Dockerfile"
        IMAGE_NAME="tt-vscode-toolkit:basic"
        ;;
esac

# Step 3: Build container image
echo -e "\n${GREEN}Step 2: Building container image...${NC}"
$CONTAINER_CMD build -f $DOCKERFILE -t $IMAGE_NAME .

echo -e "\n${GREEN}✓ Container image built successfully: $IMAGE_NAME${NC}"

# Step 4: Provide usage instructions
echo -e "\n${BLUE}=== Usage Instructions ===${NC}"
echo -e "
${GREEN}Run with $CONTAINER_CMD:${NC}
  $CONTAINER_CMD run -it -p 8080:8080 -e PASSWORD=yourpassword $IMAGE_NAME

${GREEN}Or use compose:${NC}
  $COMPOSE_CMD up -d

${GREEN}Access the IDE:${NC}
  Open your browser to: http://localhost:8080
  Default password: tenstorrent

${GREEN}Stop the container:${NC}
  $COMPOSE_CMD down
"

# Optional: Push to registry
read -p "Do you want to tag this image for pushing to a registry? (y/n): " PUSH_CHOICE
if [ "$PUSH_CHOICE" = "y" ]; then
    read -p "Enter registry URL (e.g., ghcr.io/tenstorrent/tt-vscode): " REGISTRY_URL
    $CONTAINER_CMD tag $IMAGE_NAME $REGISTRY_URL
    echo -e "${GREEN}✓ Tagged as: $REGISTRY_URL${NC}"
    echo -e "To push: ${BLUE}$CONTAINER_CMD push $REGISTRY_URL${NC}"
fi

echo -e "\n${GREEN}=== Build Complete! ===${NC}"
