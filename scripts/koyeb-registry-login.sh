#!/bin/bash
# Helper script to login to Koyeb registry
# The Koyeb CLI doesn't have a "registry login" command
# You need to use podman/docker directly

set -e

# Detect container engine
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "❌ Neither podman nor docker found"
    exit 1
fi

echo "🔐 Koyeb Registry Login"
echo ""
echo "You need:"
echo "  1. Your Koyeb username or organization name"
echo "  2. A Koyeb API token"
echo ""
echo "Get your API token from: https://app.koyeb.com/account/api"
echo ""
echo "Then run:"
echo "  $CONTAINER_CMD login registry.koyeb.com"
echo ""
echo "When prompted:"
echo "  Username: your-koyeb-username (or org name)"
echo "  Password: paste-your-api-token"
echo ""

read -p "Ready to login? [Y/n]: " READY
READY=${READY:-Y}

if [[ "$READY" =~ ^[Yy]$ ]]; then
    echo ""
    echo "Running: $CONTAINER_CMD login registry.koyeb.com"
    echo ""
    $CONTAINER_CMD login registry.koyeb.com

    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Successfully logged in to Koyeb registry!"
        echo ""
        echo "You can now push images:"
        echo "  $CONTAINER_CMD push registry.koyeb.com/yourorg/image:tag"
    else
        echo ""
        echo "❌ Login failed. Make sure:"
        echo "  - Username is correct (try your org name)"
        echo "  - Token has registry permissions"
        echo "  - Token is copied correctly (no extra spaces)"
    fi
else
    echo "Cancelled."
fi
