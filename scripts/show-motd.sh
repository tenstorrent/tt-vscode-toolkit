#!/bin/bash
# Display MOTD for Tenstorrent VSCode Toolkit
# Shows common message from motd.txt plus dynamic system information

# Only show once per session
if [ -n "$TENSTORRENT_MOTD_SHOWN" ]; then
    return 0
fi
export TENSTORRENT_MOTD_SHOWN=1

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Display the main MOTD content
if [ -f "/home/coder/.motd" ]; then
    cat /home/coder/.motd
    echo ""
fi

# Add dynamic system information
echo -e "${CYAN}💻 System Information:${NC}"
echo ""

# RAM and CPU
TOTAL_RAM=$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo "Unknown")
CPU_CORES=$(nproc 2>/dev/null || echo "Unknown")
echo -e "   ${BLUE}RAM:${NC} ${TOTAL_RAM}  |  ${BLUE}CPU Cores:${NC} ${CPU_CORES}"

# Tenstorrent hardware detection (with timeout to avoid hangs)
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
    print('Detection in progress...')
" 2>/dev/null || echo "Not detected")
    echo -e "   ${BLUE}Tenstorrent:${NC} ${TT_INFO}"
else
    echo -e "   ${BLUE}Tenstorrent:${NC} tt-smi not available"
fi

# tt-metal version and status
if [ -d "$HOME/tt-metal" ]; then
    if [ -f "$HOME/tt-metal/.git/HEAD" ]; then
        TT_METAL_BRANCH=$(cd "$HOME/tt-metal" && git branch --show-current 2>/dev/null || echo "unknown")
        TT_METAL_COMMIT=$(cd "$HOME/tt-metal" && git rev-parse --short HEAD 2>/dev/null || echo "unknown")
        echo -e "   ${BLUE}tt-metal:${NC} ${TT_METAL_BRANCH}@${TT_METAL_COMMIT} ${GREEN}(pre-compiled)${NC}"
    else
        echo -e "   ${BLUE}tt-metal:${NC} installed ${GREEN}(pre-compiled)${NC}"
    fi

    # Check if Python environment is activated
    if [ -n "$VIRTUAL_ENV" ]; then
        echo -e "   ${BLUE}Python env:${NC} ${GREEN}activated${NC} ($VIRTUAL_ENV)"
    elif [ -f "$HOME/tt-metal/python_env/bin/activate" ]; then
        echo -e "   ${BLUE}Python env:${NC} available at ~/tt-metal/python_env"
    fi
else
    echo -e "   ${BLUE}tt-metal:${NC} ${YELLOW}not installed${NC}"
fi

# Python version
PYTHON_VER=$(python3 --version 2>/dev/null | cut -d' ' -f2 || echo "Not found")
echo -e "   ${BLUE}Python:${NC} ${PYTHON_VER}"

echo ""

# Environment variables check
if [ -n "$TT_METAL_HOME" ]; then
    echo -e "${GREEN}✅ Environment configured for tt-metal${NC}"
else
    echo -e "${YELLOW}⚠️  TT_METAL_HOME not set - run 'source ~/tt-metal/python_env/bin/activate'${NC}"
fi

echo ""
