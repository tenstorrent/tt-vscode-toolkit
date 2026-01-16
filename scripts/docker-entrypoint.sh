#!/bin/bash
# Entrypoint script for Tenstorrent VSCode Toolkit container
# Provides helpful logging for cloud environments like Koyeb

set -e

# Colors for terminal output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Detect environment
if [ -n "$KOYEB_PUBLIC_DOMAIN" ]; then
    ENVIRONMENT="Koyeb"
    ACCESS_URL="https://${KOYEB_PUBLIC_DOMAIN}"
elif [ -n "$RAILWAY_PUBLIC_DOMAIN" ]; then
    ENVIRONMENT="Railway"
    ACCESS_URL="https://${RAILWAY_PUBLIC_DOMAIN}"
elif [ -n "$FLY_APP_NAME" ]; then
    ENVIRONMENT="Fly.io"
    ACCESS_URL="https://${FLY_APP_NAME}.fly.dev"
else
    ENVIRONMENT="Local/Other"
    ACCESS_URL="http://localhost:8080"
fi

# Display banner
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}Tenstorrent VSCode Toolkit${NC}                             ${BLUE}║${NC}"
echo -e "${BLUE}║${NC}  Browser-based VSCode with TT Extension Pre-installed    ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Show environment info
echo -e "${CYAN}📍 Environment:${NC} ${ENVIRONMENT}"
echo ""

# Show access information
echo -e "${GREEN}${BOLD}🚀 YOUR IDE IS STARTING...${NC}"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}📝 IMPORTANT - SAVE THESE DETAILS:${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${CYAN}🌐 Access URL:${NC}  ${BOLD}${ACCESS_URL}${NC}"
echo -e "  ${CYAN}🔑 Password:${NC}    ${BOLD}${PASSWORD:-tenstorrent}${NC}"
echo ""
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Instructions based on environment
if [ "$ENVIRONMENT" = "Koyeb" ]; then
    echo -e "${GREEN}📋 NEXT STEPS (Koyeb):${NC}"
    echo ""
    echo "  1. Wait for 'Service is ready' message below"
    echo "  2. Click the URL in Koyeb dashboard or visit:"
    echo -e "     ${BOLD}${ACCESS_URL}${NC}"
    echo "  3. Enter password: ${BOLD}${PASSWORD:-tenstorrent}${NC}"
    echo "  4. Look for 'Tenstorrent' in the VSCode sidebar (TT icon)"
    echo "  5. Click 'Welcome Page' to start learning!"
    echo ""
elif [ "$ENVIRONMENT" = "Railway" ]; then
    echo -e "${GREEN}📋 NEXT STEPS (Railway):${NC}"
    echo ""
    echo "  1. Wait for 'Deployed' status in Railway dashboard"
    echo "  2. Visit: ${BOLD}${ACCESS_URL}${NC}"
    echo "  3. Enter password: ${BOLD}${PASSWORD:-tenstorrent}${NC}"
    echo "  4. Extension is pre-installed and ready!"
    echo ""
elif [ "$ENVIRONMENT" = "Fly.io" ]; then
    echo -e "${GREEN}📋 NEXT STEPS (Fly.io):${NC}"
    echo ""
    echo "  1. Wait for deployment to complete"
    echo "  2. Visit: ${BOLD}${ACCESS_URL}${NC}"
    echo "  3. Enter password: ${BOLD}${PASSWORD:-tenstorrent}${NC}"
    echo ""
else
    echo -e "${GREEN}📋 NEXT STEPS (Local):${NC}"
    echo ""
    echo "  1. Wait for 'HTTP server listening' message below"
    echo "  2. Open your browser to: ${BOLD}${ACCESS_URL}${NC}"
    echo "  3. Enter password: ${BOLD}${PASSWORD:-tenstorrent}${NC}"
    echo ""
fi

# Show extension info
echo -e "${CYAN}🎯 WHAT'S INCLUDED:${NC}"
echo ""
echo "  ✅ VSCode in your browser (code-server)"
echo "  ✅ Tenstorrent extension pre-installed"
echo "  ✅ 16 interactive hardware lessons"
echo "  ✅ Production deployment guides"
echo "  ✅ Template scripts and examples"

# Check for tt-metal installation - install if missing
if [ -d "$HOME/tt-metal" ] && [ -f "$HOME/tt-metal/python_env/bin/activate" ]; then
    echo "  ✅ tt-metal ready at: ~/tt-metal"
else
    echo ""
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}🔧 Installing tt-metal (first startup - this takes ~10 minutes)${NC}"
    echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
    echo ""

    # Clone tt-metal with submodules
    if [ ! -d "$HOME/tt-metal" ]; then
        echo -e "${CYAN}📥 Cloning tt-metal repository...${NC}"
        git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git "$HOME/tt-metal" || {
            echo -e "${YELLOW}⚠️  Failed to clone tt-metal - will be available via extension lessons${NC}"
            echo "  ⚠️  tt-metal not installed (lessons work in learning mode)"
        }
    fi

    # Install dependencies and build
    if [ -d "$HOME/tt-metal" ]; then
        cd "$HOME/tt-metal"

        echo -e "${CYAN}📦 Installing system dependencies...${NC}"
        # Run install script without PPA additions (avoids timeouts)
        # Install only essential packages that are available in Ubuntu repos
        sudo apt-get update -qq
        sudo apt-get install -y -qq \
            build-essential \
            cmake \
            python3-dev \
            libboost-all-dev \
            libyaml-cpp-dev \
            libhwloc-dev \
            libgtest-dev \
            libgmock-dev \
            ninja-build || {
            echo -e "${YELLOW}⚠️  Some dependencies may be missing${NC}"
        }

        echo -e "${CYAN}🔨 Building tt-metal...${NC}"
        ./build_metal.sh || {
            echo -e "${YELLOW}⚠️  Build failed - will retry via extension lessons${NC}"
        }

        echo -e "${CYAN}🐍 Setting up Python environment...${NC}"
        python3 -m venv python_env
        source python_env/bin/activate
        pip install --upgrade pip -q
        pip install -e . -q || {
            echo -e "${YELLOW}⚠️  Python package installation incomplete${NC}"
        }

        # Configure environment
        echo "" >> "$HOME/.bashrc"
        echo "# Tenstorrent tt-metal environment" >> "$HOME/.bashrc"
        echo "export TT_METAL_HOME=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo "export PYTHONPATH=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/tt-metal:${PATH}"' >> "$HOME/.bashrc"
        echo 'if [ -f "$HOME/tt-metal/python_env/bin/activate" ]; then' >> "$HOME/.bashrc"
        echo '    source $HOME/tt-metal/python_env/bin/activate' >> "$HOME/.bashrc"
        echo 'fi' >> "$HOME/.bashrc"

        # Set for current session
        export TT_METAL_HOME="$HOME/tt-metal"
        export PYTHONPATH="$HOME/tt-metal"
        export PATH="$HOME/tt-metal:${PATH}"

        echo ""
        echo -e "${GREEN}✅ tt-metal installation complete!${NC}"
        echo ""
    fi

    cd "$HOME"
fi

# Check for tt-smi installation
if command -v tt-smi &> /dev/null; then
    echo -e "  ${GREEN}✅ tt-smi installed${NC}"
else
    echo -e "  ${YELLOW}⚠️  tt-smi not found${NC}"
fi

# Check and fix Tenstorrent device permissions
if [ -d "/dev/tenstorrent" ]; then
    echo -e "  ${CYAN}🔧 Configuring Tenstorrent hardware access...${NC}"
    # Change device permissions to allow access
    sudo chmod -R 666 /dev/tenstorrent/* 2>/dev/null || true
    # Verify access with tt-smi
    if command -v tt-smi &> /dev/null && tt-smi -v &> /dev/null; then
        DEVICE_INFO=$(tt-smi -v 2>/dev/null | grep -E "Board type|Num devices" | head -2 || echo "")
        if [ -n "$DEVICE_INFO" ]; then
            echo -e "  ${GREEN}✅ Hardware detected and accessible:${NC}"
            echo "$DEVICE_INFO" | sed 's/^/     /'
        else
            DEVICE_COUNT=$(ls -1 /dev/tenstorrent/ 2>/dev/null | wc -l)
            echo -e "  ${GREEN}✅ Device files configured (${DEVICE_COUNT} device(s))${NC}"
        fi
    elif [ -r "/dev/tenstorrent/0" ]; then
        DEVICE_COUNT=$(ls -1 /dev/tenstorrent/ 2>/dev/null | wc -l)
        echo -e "  ${GREEN}✅ Device permissions configured (${DEVICE_COUNT} device(s) found)${NC}"
    else
        echo -e "  ${YELLOW}⚠️  Device permissions may need adjustment${NC}"
    fi
fi
echo ""

# Health check info
echo -e "${CYAN}🔍 HEALTH CHECK:${NC}"
echo "  Endpoint: ${ACCESS_URL}/healthz"
echo ""

# Separator before code-server logs
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}📡 CODE-SERVER LOGS:${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Start code-server with logging
exec code-server \
    --bind-addr 0.0.0.0:8080 \
    --auth password \
    --disable-telemetry \
    --disable-update-check \
    /home/coder 2>&1 | while IFS= read -r line; do
    # Enhance code-server logs
    if [[ "$line" =~ "HTTP server listening" ]]; then
        echo -e "${GREEN}✅ Service is ready!${NC}"
        echo ""
        echo -e "${BOLD}🎉 YOUR IDE IS NOW AVAILABLE AT:${NC}"
        echo -e "   ${CYAN}${ACCESS_URL}${NC}"
        echo ""
        echo "$line"
    else
        echo "$line"
    fi
done
