#!/bin/bash
# Test Koyeb authentication and registry access

set -e

echo "🔍 Testing Koyeb Setup"
echo ""

# Check CLI installed
if ! command -v koyeb &> /dev/null; then
    echo "❌ Koyeb CLI not installed"
    echo "   Install: curl -fsSL https://cli.koyeb.com/install.sh | sh"
    exit 1
fi
echo "✅ Koyeb CLI installed"

# Check logged in (try multiple methods)
echo "Checking authentication..."
LOGIN_OK=false

if koyeb profile show &> /dev/null; then
    echo "✅ Logged in to Koyeb (profile check)"
    LOGIN_OK=true
elif koyeb org list &> /dev/null; then
    echo "✅ Logged in to Koyeb (org list check)"
    LOGIN_OK=true
elif koyeb service list &> /dev/null; then
    echo "✅ Logged in to Koyeb (service list check)"
    LOGIN_OK=true
fi

if [ "$LOGIN_OK" = "false" ]; then
    echo "❌ Not logged in to Koyeb"
    echo ""
    echo "Try:"
    echo "  koyeb login"
    echo ""
    echo "Or if you're already logged in, check:"
    echo "  koyeb --version"
    echo "  which koyeb"
    echo "  koyeb org list"
    exit 1
fi

# Show profile info
echo ""
echo "📋 Account Info:"
koyeb profile show 2>/dev/null || koyeb org list | head -5

# Check org
if [ -z "$KOYEB_ORG" ]; then
    echo ""
    echo "⚠️  KOYEB_ORG not set"
    echo "   Set it with: export KOYEB_ORG=your-org-name"
    echo ""
    echo "   Available orgs:"
    koyeb org list
    exit 1
fi
echo ""
echo "✅ KOYEB_ORG set: $KOYEB_ORG"

# Test container engine
echo ""
if command -v podman &> /dev/null; then
    CONTAINER_CMD="podman"
elif command -v docker &> /dev/null; then
    CONTAINER_CMD="docker"
else
    echo "❌ Neither podman nor docker found"
    exit 1
fi
echo "✅ Container engine: $CONTAINER_CMD"

# Test registry access
echo ""
echo "🔐 Testing registry access..."
echo "   Koyeb CLI doesn't have 'registry login' command"
echo "   Registry auth is handled through container engine"
echo ""
echo "   To manually login to registry:"
echo "   $CONTAINER_CMD login registry.koyeb.com"
echo "   Username: your-koyeb-username"
echo "   Password: your-koyeb-api-token (from https://app.koyeb.com/account/api)"
echo ""

# Test if we can push
echo ""
echo "🧪 Testing image push capability..."
TEST_IMAGE="registry.koyeb.com/${KOYEB_ORG}/test:latest"

# Create a minimal test image
cat > /tmp/Dockerfile.test << 'EOF'
FROM alpine:latest
RUN echo "test"
EOF

echo "   Building test image..."
if ! $CONTAINER_CMD build -t test:latest -f /tmp/Dockerfile.test /tmp > /dev/null 2>&1; then
    echo "❌ Test image build failed"
    exit 1
fi

echo "   Tagging test image..."
$CONTAINER_CMD tag test:latest ${TEST_IMAGE}

echo "   Pushing test image..."
if $CONTAINER_CMD push ${TEST_IMAGE} 2>&1 | grep -q "Writing manifest\|digest:"; then
    echo "✅ Test push successful"
    echo ""
    echo "🎉 Everything is configured correctly!"
    echo ""
    echo "You can now run:"
    echo "  ./quick-deploy-koyeb.sh"

    # Cleanup
    $CONTAINER_CMD rmi test:latest ${TEST_IMAGE} > /dev/null 2>&1 || true
    rm /tmp/Dockerfile.test
else
    echo "❌ Test push failed"
    echo ""
    echo "This might be a permissions issue. Check:"
    echo "  - Your organization has registry access"
    echo "  - Your account has push permissions"
    exit 1
fi
