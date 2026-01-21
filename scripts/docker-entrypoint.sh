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
echo -e "${BLUE}╔════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}║${NC}  ${BOLD}Welcome to this Tenstorrent dev environment${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════${NC}"
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
echo "  ✅ Interactive hardware lessons"
echo "  ✅ Production deployment guides"
echo "  ✅ Template scripts and examples"

# Check for tt-metal installation
# If using pre-built image (like tt-metalium), skip installation entirely
if [ "$TT_METAL_PREBUILT" = "true" ]; then
    echo "  ✅ tt-metal pre-built in base image (cloning source for examples)"

    # Clone tt-metal for source code and examples (but don't build)
    if [ ! -d "$HOME/tt-metal" ]; then
        echo -e "${CYAN}📥 Cloning tt-metal repository (source code only)...${NC}"
        git clone --recurse-submodules https://github.com/tenstorrent/tt-metal.git "$HOME/tt-metal" || {
            echo -e "${YELLOW}⚠️  Failed to clone tt-metal - will be available via extension lessons${NC}"
        }
    fi

    # Configure environment variables in .bashrc if not already set
    if ! grep -q "TT_METAL_HOME" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Tenstorrent tt-metal environment" >> "$HOME/.bashrc"
        echo "export TT_METAL_HOME=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo "export PYTHONPATH=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/tt-metal:${PATH}"' >> "$HOME/.bashrc"
        echo "" >> "$HOME/.bashrc"
        echo "# Activate Python virtual environment" >> "$HOME/.bashrc"
        echo 'if [ -f "/opt/venv/bin/activate" ]; then' >> "$HOME/.bashrc"
        echo '    source /opt/venv/bin/activate' >> "$HOME/.bashrc"
        echo 'fi' >> "$HOME/.bashrc"
    fi

    # Set for current session
    export TT_METAL_HOME="$HOME/tt-metal"
    export PYTHONPATH="$HOME/tt-metal"
    export PATH="$HOME/tt-metal:${PATH}"

    # Activate venv for current session
    if [ -f "/opt/venv/bin/activate" ]; then
        source /opt/venv/bin/activate
    fi

    echo -e "  ${GREEN}✅ Source code available at: ~/tt-metal${NC}"
    echo -e "  ${CYAN}   Using pre-built binaries from base image${NC}"
elif [ -d "$HOME/tt-metal" ] && [ -f "$HOME/tt-metal/python_env/bin/activate" ]; then
    echo "  ✅ tt-metal pre-built and ready at: ~/tt-metal"

    # Configure environment variables in .bashrc if not already set
    if ! grep -q "TT_METAL_HOME" "$HOME/.bashrc" 2>/dev/null; then
        echo "" >> "$HOME/.bashrc"
        echo "# Tenstorrent tt-metal environment" >> "$HOME/.bashrc"
        echo "export TT_METAL_HOME=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo "export PYTHONPATH=\$HOME/tt-metal" >> "$HOME/.bashrc"
        echo 'export PATH="$HOME/tt-metal:${PATH}"' >> "$HOME/.bashrc"
        echo 'if [ -f "$HOME/tt-metal/python_env/bin/activate" ]; then' >> "$HOME/.bashrc"
        echo '    source $HOME/tt-metal/python_env/bin/activate' >> "$HOME/.bashrc"
        echo 'fi' >> "$HOME/.bashrc"
    fi

    # Set for current session
    export TT_METAL_HOME="$HOME/tt-metal"
    export PYTHONPATH="$HOME/tt-metal"
    export PATH="$HOME/tt-metal:${PATH}"
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

    # Reset devices to ensure clean state (prevents communication errors)
    if command -v tt-smi &> /dev/null; then
        echo -e "  ${CYAN}   Initializing devices...${NC}"
        # Try a quick detection first
        if ! timeout 3s sudo tt-smi -s &> /dev/null; then
            # Detection failed or timed out - reset devices
            echo -e "  ${CYAN}   Resetting devices for clean initialization...${NC}"
            sudo tt-smi -r &> /dev/null || true
            sleep 2  # Give devices time to reset
        fi
    fi

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

# Reset extension state to ensure welcome page shows on first container startup
EXTENSION_STORAGE="$HOME/.local/share/code-server/User/globalStorage/tenstorrent.tt-vscode-toolkit"
if [ -d "$EXTENSION_STORAGE" ]; then
    echo -e "${CYAN}🔄 Resetting extension state for fresh start...${NC}"
    rm -rf "$EXTENSION_STORAGE"
fi

# Health check info
echo -e "${CYAN}🔍 HEALTH CHECK:${NC}"
echo "  Endpoint: ${ACCESS_URL}/healthz"
echo ""

# Create a nice MOTD for terminal sessions
cat > "$HOME/.bashrc_tenstorrent" << 'EOF'
# Tenstorrent VSCode Toolkit MOTD
if [ -z "$TENSTORRENT_MOTD_SHOWN" ]; then
    export TENSTORRENT_MOTD_SHOWN=1

    echo ""
    echo "╔════════════════════════════════════════════════════════════"
    echo "║  Welcome to your Tenstorrent development environment!"
    echo "╚════════════════════════════════════════════════════════════"
    echo ""

    # System info
    echo "💻 System:"
    TOTAL_RAM=$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "Unknown")
    CPU_CORES=$(nproc 2>/dev/null || echo "Unknown")
    echo "   RAM: ${TOTAL_RAM}  |  CPU Cores: ${CPU_CORES}"

    # Tenstorrent hardware detection (with timeout)
    if command -v tt-smi &> /dev/null; then
        TT_INFO=$(timeout 2s tt-smi -s 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    devices = data.get('devices', [])
    if devices:
        board_type = devices[0].get('board_type', 'Unknown')
        count = len(devices)
        print(f'{board_type} (x{count})')
    else:
        print('No devices detected')
except:
    print('Detection failed')
" 2>/dev/null || echo "Not detected")
        echo "   Tenstorrent: ${TT_INFO}"
    else
        echo "   Tenstorrent: tt-smi not available"
    fi

    # tt-metal version
    if [ -d "$HOME/tt-metal" ]; then
        if [ -f "$HOME/tt-metal/.git/HEAD" ]; then
            TT_METAL_BRANCH=$(cd "$HOME/tt-metal" && git branch --show-current 2>/dev/null || echo "unknown")
            TT_METAL_COMMIT=$(cd "$HOME/tt-metal" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
            echo "   tt-metal: ${TT_METAL_BRANCH}@${TT_METAL_COMMIT}"
        else
            echo "   tt-metal: installed (no git info)"
        fi
    else
        echo "   tt-metal: not installed"
    fi

    # Python version
    PYTHON_VER=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "Not found")
    echo "   Python: ${PYTHON_VER}"

    echo ""
    echo "📚 To get started:"
    echo "   • Open Command Palette (Ctrl+Shift+P)"
    echo "   • Search: 'Tenstorrent: Show Welcome Page'"
    echo "   • Or click the Tenstorrent icon in the left sidebar"
    echo ""
    echo "🔧 Quick commands:"
    echo "   tt-smi              - Check hardware status"
    echo "   tt-smi -r           - Reset devices (if needed)"
    echo ""
    echo "💡 Tip: We have several lessons in the Tenstorrent sidebar!"
    echo ""
fi
EOF

# Append to .bashrc
echo "" >> "$HOME/.bashrc"
echo "# Tenstorrent MOTD" >> "$HOME/.bashrc"
echo "source ~/.bashrc_tenstorrent" >> "$HOME/.bashrc"

# Separator before code-server logs
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BOLD}📡 CODE-SERVER LOGS:${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Start code-server without opening a file
exec code-server \
    --bind-addr 0.0.0.0:8080 \
    --auth password \
    --disable-telemetry \
    --disable-update-check \
    2>&1 | while IFS= read -r line; do
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
