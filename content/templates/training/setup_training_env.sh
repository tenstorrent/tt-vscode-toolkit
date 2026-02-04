#!/bin/bash
#
# Custom Training Environment Setup
# Use for CT4-CT8 lessons
#
# This script sets up the environment for custom training on Tenstorrent hardware.
# Run: source setup_training_env.sh
#

echo "🎭 Setting up Custom Training environment..."

# Detect tt-metal installation
if [ -z "$TT_METAL_HOME" ]; then
    # Try common locations
    if [ -d "$HOME/tt-metal" ]; then
        export TT_METAL_HOME="$HOME/tt-metal"
    elif [ -d "$HOME/tt-metal-v0.64.5" ]; then
        export TT_METAL_HOME="$HOME/tt-metal-v0.64.5"
    else
        echo "❌ Error: TT_METAL_HOME not set and tt-metal not found in common locations"
        echo "   Please set TT_METAL_HOME to your tt-metal installation directory"
        return 1
    fi
fi

# Set library paths
export LD_LIBRARY_PATH="$TT_METAL_HOME/build/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$TT_METAL_HOME:$PYTHONPATH"

# Detect hardware (default to N150 if not set)
if [ -z "$MESH_DEVICE" ]; then
    export MESH_DEVICE=N150
    echo "ℹ️  MESH_DEVICE not set, defaulting to N150"
fi

# Activate Python environment if it exists
if [ -f "$TT_METAL_HOME/python_env/bin/activate" ]; then
    source "$TT_METAL_HOME/python_env/bin/activate"
else
    echo "⚠️  Warning: python_env not found at $TT_METAL_HOME/python_env"
fi

echo ""
echo "✅ Environment configured:"
echo "   TT_METAL_HOME: $TT_METAL_HOME"
echo "   MESH_DEVICE: $MESH_DEVICE"
echo "   Python: $(which python)"
echo ""

# Verify critical imports
echo "Verifying imports..."
python -c "import ttml; print('   ✅ ttml')" 2>/dev/null || echo "   ❌ ttml (pip install may be needed)"
python -c "import torch; print('   ✅ PyTorch')" 2>/dev/null || echo "   ❌ PyTorch"
python -c "import transformers; print('   ✅ transformers')" 2>/dev/null || echo "   ⚠️  transformers (needed for CT4: pip install transformers)"

echo ""
echo "🚀 Ready for Custom Training lessons (CT4-CT8)!"
echo "   Run: python test_training_startup.py (for full validation)"
