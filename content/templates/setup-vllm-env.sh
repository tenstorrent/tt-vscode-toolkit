#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Setup vLLM environment for Tenstorrent hardware
#
# This script creates the correct Python environment for vLLM following
# the official installation process from tt-vllm/tt_metal/README.md
#
# What this script does:
# 1. Validates prerequisites (tt-metal installed, TT_METAL_HOME set)
# 2. Creates Python venv at ${TT_METAL_HOME}/build/python_env_vllm
# 3. Installs exact PyTorch version (2.5.0+cpu) for TT hardware compatibility
# 4. Builds and installs vLLM from source with TT hardware support
# 5. Installs all required dependencies
# 6. Validates the installation
#
# Usage:
#   bash ~/tt-scratchpad/setup-vllm-env.sh
#
# After running, activate with:
#   cd ~/tt-vllm
#   source tt_metal/setup-metal.sh
#   source $PYTHON_ENV_DIR/bin/activate

set -e  # Exit on error

echo "============================================================"
echo "🚀 Setting up vLLM Environment for Tenstorrent Hardware"
echo "============================================================"
echo ""

# Step 1: Validate prerequisites
echo "📋 Step 1/6: Validating prerequisites..."

# Check if TT_METAL_HOME is set
if [ -z "$TT_METAL_HOME" ]; then
    TT_METAL_HOME="$HOME/tt-metal"
    echo "   TT_METAL_HOME not set, using default: $TT_METAL_HOME"
fi

# Check if tt-metal exists
if [ ! -d "$TT_METAL_HOME" ]; then
    echo "   ❌ ERROR: tt-metal not found at $TT_METAL_HOME"
    echo "   Please install tt-metal first or set TT_METAL_HOME correctly"
    exit 1
fi
echo "   ✓ tt-metal found at $TT_METAL_HOME"

# Check if tt-vllm exists
if [ ! -d "$HOME/tt-vllm" ]; then
    echo "   ❌ ERROR: tt-vllm not found at $HOME/tt-vllm"
    echo "   Please clone tt-vllm first:"
    echo "   git clone --branch dev https://github.com/tenstorrent/vllm.git ~/tt-vllm"
    exit 1
fi
echo "   ✓ tt-vllm found at $HOME/tt-vllm"

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "   ✓ Python version: $PYTHON_VERSION"
echo ""

# Step 2: Source setup-metal.sh to get correct environment variables
echo "⚙️  Step 2/6: Sourcing tt-metal environment..."
cd "$HOME/tt-vllm"
export vllm_dir=$(pwd)
source "$vllm_dir/tt_metal/setup-metal.sh"

if [ -z "$PYTHON_ENV_DIR" ]; then
    echo "   ❌ ERROR: PYTHON_ENV_DIR not set after sourcing setup-metal.sh"
    exit 1
fi
echo "   ✓ PYTHON_ENV_DIR set to: $PYTHON_ENV_DIR"
echo "   ✓ TT_METAL_HOME: $TT_METAL_HOME"
echo ""

# Step 3: Create/recreate Python virtual environment
echo "🔨 Step 3/6: Creating Python virtual environment..."

if [ -d "$PYTHON_ENV_DIR" ]; then
    echo "   Virtual environment already exists at $PYTHON_ENV_DIR"
    echo "   Remove it and recreate? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        echo "   Removing existing environment..."
        rm -rf "$PYTHON_ENV_DIR"
        python3 -m venv "$PYTHON_ENV_DIR"
        echo "   ✓ Virtual environment recreated"
    else
        echo "   ✓ Using existing virtual environment"
    fi
else
    python3 -m venv "$PYTHON_ENV_DIR"
    echo "   ✓ Virtual environment created"
fi
echo ""

# Step 4: Install PyTorch with exact version for TT hardware
echo "🔥 Step 4/6: Installing PyTorch 2.5.0+cpu (TT hardware requirement)..."
source "$PYTHON_ENV_DIR/bin/activate"

pip install --quiet --upgrade pip

# Install exact torch version from requirements/tt.txt
echo "   Installing torch==2.5.0+cpu torchvision==0.20.0 torchaudio==2.5.0..."
pip install --quiet --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.0+cpu \
    torchvision==0.20.0 \
    torchaudio==2.5.0

# Verify torch version
TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
echo "   ✓ PyTorch installed: $TORCH_VERSION"

if [[ ! "$TORCH_VERSION" == 2.5.0* ]]; then
    echo "   ⚠️  WARNING: Expected torch 2.5.0+cpu, got $TORCH_VERSION"
    echo "   This may cause compatibility issues with TT hardware"
fi
echo ""

# Step 5: Install vLLM and dependencies
echo "📦 Step 5/6: Installing vLLM and dependencies..."

# Install ttnn and pytest (required for TT hardware)
echo "   Installing ttnn and pytest..."
pip install --quiet --upgrade ttnn pytest

# Install additional dependencies from requirements/tt.txt
echo "   Installing additional dependencies (fairscale, termcolor, loguru, etc.)..."
pip install --quiet fairscale termcolor loguru blobfile fire pytz llama-models==0.0.48

# Install vLLM in editable mode
echo "   Installing vLLM from source (this may take 5-10 minutes)..."
cd "$vllm_dir"
pip install --quiet -e . --extra-index-url https://download.pytorch.org/whl/cpu

echo "   ✓ vLLM installed successfully"
echo ""

# Step 6: Validate installation
echo "🧪 Step 6/6: Validating installation..."

# Test torch import and version
TORCH_CHECK=$(python3 -c "import torch; print('✓ torch', torch.__version__)" 2>&1)
if [ $? -eq 0 ]; then
    echo "   $TORCH_CHECK"
else
    echo "   ❌ torch import failed"
    exit 1
fi

# Test vLLM import
VLLM_CHECK=$(python3 -c "import vllm; print('✓ vllm import successful')" 2>&1)
if [ $? -eq 0 ]; then
    echo "   $VLLM_CHECK"
else
    echo "   ❌ vllm import failed"
    exit 1
fi

# Test ttnn import
TTNN_CHECK=$(python3 -c "import ttnn; print('✓ ttnn import successful')" 2>&1)
if [ $? -eq 0 ]; then
    echo "   $TTNN_CHECK"
else
    echo "   ⚠️  Warning: ttnn import failed (may need to source setup-metal.sh)"
fi

echo ""

# Create convenient activation script
echo "📝 Creating activation script..."
cat > "$HOME/activate-vllm-env.sh" << 'EOF'
#!/bin/bash
# Activate vLLM environment for Tenstorrent hardware
cd ~/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate
export LD_LIBRARY_PATH=/opt/openmpi-v5.0.7-ulfm/lib:$LD_LIBRARY_PATH
echo "✓ vLLM environment activated"
echo "  PYTHON_ENV_DIR: $PYTHON_ENV_DIR"
echo "  TT_METAL_HOME: $TT_METAL_HOME"
echo ""
echo "Ready to start vLLM server!"
EOF

chmod +x "$HOME/activate-vllm-env.sh"
echo "   ✓ Created ~/activate-vllm-env.sh"
echo ""

# Print success message
echo "============================================================"
echo "✅ vLLM Environment Setup Complete!"
echo "============================================================"
echo ""
echo "📚 Quick Start:"
echo ""
echo "1. Activate the environment:"
echo "   source ~/activate-vllm-env.sh"
echo ""
echo "2. Start vLLM server with Qwen2.5-Coder:"
echo "   python ~/tt-scratchpad/start-vllm-server.py --model ~/models/Qwen2.5-Coder-1.5B-Instruct"
echo ""
echo "3. In a new terminal, test the server:"
echo "   curl http://localhost:8000/health"
echo ""
echo "📖 Environment Details:"
echo "   Python: $PYTHON_VERSION"
echo "   PyTorch: $TORCH_VERSION"
echo "   Location: $PYTHON_ENV_DIR"
echo "   TT-Metal: $TT_METAL_HOME"
echo ""
echo "🔧 Manual Activation (if needed):"
echo "   cd ~/tt-vllm"
echo "   source tt_metal/setup-metal.sh"
echo "   source \$PYTHON_ENV_DIR/bin/activate"
echo ""
echo "Happy coding! 🚀"
