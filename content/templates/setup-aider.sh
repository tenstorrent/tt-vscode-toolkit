#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Setup Aider CLI coding assistant for Tenstorrent vLLM server
#
# This script automates the installation and configuration of Aider,
# a powerful AI coding assistant that works with your local vLLM server.
#
# What this script does:
# 1. Creates a dedicated Python virtual environment for Aider
# 2. Installs aider-chat package
# 3. Creates Aider configuration file (~/.aider/aider.conf.yml)
# 4. Creates a convenient wrapper script (~/bin/aider-tt)
# 5. Tests the connection to your vLLM server
#
# Usage:
#   bash ~/tt-scratchpad/setup-aider.sh
#
# After running, you can start Aider with: aider-tt

set -e  # Exit on error

echo "============================================================"
echo "🚀 Setting up Aider for Tenstorrent vLLM"
echo "============================================================"
echo ""

# Check if vLLM server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  WARNING: vLLM server is not running at http://localhost:8000"
    echo "   You'll need to start the vLLM server before using Aider."
    echo "   Continue with installation? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi
fi

# Step 1: Create virtual environment
echo "📦 Step 1/5: Creating Python virtual environment..."
if [ -d ~/aider-venv ]; then
    echo "   Virtual environment already exists at ~/aider-venv"
    echo "   Remove it and recreate? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        rm -rf ~/aider-venv
        python3 -m venv ~/aider-venv
        echo "   ✓ Virtual environment recreated"
    else
        echo "   ✓ Using existing virtual environment"
    fi
else
    python3 -m venv ~/aider-venv
    echo "   ✓ Virtual environment created"
fi
echo ""

# Step 2: Install Aider
echo "📥 Step 2/5: Installing aider-chat..."
source ~/aider-venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet aider-chat
echo "   ✓ Aider installed (version: $(aider --version 2>/dev/null || echo 'unknown'))"
deactivate
echo ""

# Step 3: Create Aider configuration
echo "⚙️  Step 3/5: Creating Aider configuration..."
mkdir -p ~/.aider

cat > ~/.aider/aider.conf.yml << 'EOF'
# Aider configuration for local vLLM server on Tenstorrent hardware

# Use OpenAI-compatible API format with Qwen2.5-Coder (code-specialized model!)
model: openai/Qwen/Qwen2.5-Coder-1.5B-Instruct

# Point to your local vLLM server
openai-api-base: http://localhost:8000/v1

# No API key needed for local server
openai-api-key: sk-no-key-required

# Model settings optimized for Qwen2.5-Coder
max-tokens: 2048
temperature: 0.6

# Git settings
auto-commits: false
dirty-commits: true
EOF

echo "   ✓ Configuration created at ~/.aider/aider.conf.yml"
echo ""

# Step 4: Create wrapper script
echo "🔧 Step 4/5: Creating aider-tt wrapper script..."
mkdir -p ~/bin

cat > ~/bin/aider-tt << 'EOF'
#!/bin/bash
# Aider wrapper for Tenstorrent local models

source ~/aider-venv/bin/activate

# Check if server is running
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ ERROR: vLLM server is not running at http://localhost:8000"
    echo ""
    echo "Start the server first:"
    echo "  source ~/tt-vllm-venv/bin/activate"
    echo "  export MESH_DEVICE=N150   # match your hardware"
    echo "  vllm serve ~/models/Qwen2.5-Coder-1.5B-Instruct --block-size 64"
    echo ""
    exit 1
fi

# Run Aider with local code-specialized model
exec aider \
    --model openai/Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --openai-api-base http://localhost:8000/v1 \
    --openai-api-key sk-no-key-required \
    "$@"
EOF

chmod +x ~/bin/aider-tt
echo "   ✓ Wrapper script created at ~/bin/aider-tt"
echo ""

# Add ~/bin to PATH if not already there
if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
    echo "   Adding ~/bin to PATH..."
    echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/bin:$PATH"
    echo "   ✓ Added to ~/.bashrc"
fi
echo ""

# Step 5: Test connection
echo "🧪 Step 5/5: Testing connection to vLLM server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✓ vLLM server is running!"
    echo ""
    echo "   Testing Aider connection..."
    source ~/aider-venv/bin/activate
    if timeout 10 aider --model openai/Qwen/Qwen2.5-Coder-1.5B-Instruct \
                        --openai-api-base http://localhost:8000/v1 \
                        --openai-api-key sk-no-key-required \
                        --yes \
                        --message "/exit" > /dev/null 2>&1; then
        echo "   ✓ Aider successfully connected to vLLM!"
    else
        echo "   ⚠️  Connection test timed out or failed"
        echo "   This might be normal if the model is still loading"
    fi
    deactivate
else
    echo "   ⚠️  vLLM server is not running - skipping connection test"
    echo "   Start your vLLM server to test the connection"
fi
echo ""

# Print success message
echo "============================================================"
echo "✅ Aider setup complete!"
echo "============================================================"
echo ""
echo "📚 Quick Start:"
echo ""
echo "1. Make sure vLLM server is running with Qwen2.5-Coder:"
echo "   source ~/tt-vllm-venv/bin/activate"
echo "   export MESH_DEVICE=N150   # match your hardware"
echo "   vllm serve ~/models/Qwen2.5-Coder-1.5B-Instruct --block-size 64"
echo ""
echo "2. In a new terminal, start Aider:"
echo "   aider-tt"
echo ""
echo "3. Or navigate to a project and start coding:"
echo "   cd ~/my-project"
echo "   aider-tt"
echo ""
echo "📖 Useful Aider commands (inside Aider prompt):"
echo "   /help     - Show all commands"
echo "   /add FILE - Add file to context"
echo "   /diff     - Show pending changes"
echo "   /commit   - Commit with AI-generated message"
echo "   /exit     - Exit Aider"
echo ""
echo "🎯 Example prompts to try:"
echo "   'Create a Python function to calculate fibonacci numbers'"
echo "   'Add error handling to this function'"
echo "   'Refactor this code to use list comprehension'"
echo "   'Add docstrings to all functions'"
echo ""
echo "Happy coding! 🚀"
