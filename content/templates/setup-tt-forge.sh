#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# TT-Forge Isolated Environment Setup
#
# This script sets up an isolated environment for TT-Forge (experimental MLIR compiler)
# that won't conflict with tt-metal direct API environment.
#
# Usage:
#   source ~/tt-scratchpad/setup-tt-forge.sh
#   # Now run TT-Forge builds and demos...
#
# Why this is needed:
# - TT-Forge requires unsetting tt-metal variables
# - Needs absolute paths (CMake doesn't expand ~)
# - Requires clang-17 compiler
# - Uses Python 3.11 (separate from tt-metal's 3.10)

echo "🔧 Setting up TT-Forge environment..."

# CRITICAL: Unset tt-metal variables to avoid build conflicts
if [ -n "$TT_METAL_HOME" ]; then
    echo "   ℹ️  Unsetting TT_METAL_HOME (TT-Forge build requirement)"
    unset TT_METAL_HOME
fi

if [ -n "$TT_METAL_VERSION" ]; then
    echo "   ℹ️  Unsetting TT_METAL_VERSION"
    unset TT_METAL_VERSION
fi

# Set toolchain directories (MUST be absolute paths - CMake requirement!)
export TTFORGE_TOOLCHAIN_DIR="/home/$USER/ttforge-toolchain"
export TTMLIR_TOOLCHAIN_DIR="/home/$USER/ttmlir-toolchain"
export TTFORGE_PYTHON_VERSION="python3.11"

echo "   ✓ TTFORGE_TOOLCHAIN_DIR=$TTFORGE_TOOLCHAIN_DIR"
echo "   ✓ TTMLIR_TOOLCHAIN_DIR=$TTMLIR_TOOLCHAIN_DIR"

# Create toolchain directories if they don't exist
mkdir -p "$TTFORGE_TOOLCHAIN_DIR"
mkdir -p "$TTMLIR_TOOLCHAIN_DIR"

# Set compiler paths (clang-17 required)
if [ -f "/usr/bin/clang-17" ]; then
    export CC="/usr/bin/clang-17"
    export CXX="/usr/bin/clang++-17"
    echo "   ✓ Using clang-17 compiler"
else
    echo "   ⚠️  clang-17 not found!"
    echo "   → Install with: sudo apt install clang-17"
    echo "   → Or create symlink: sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100"
fi

# Activate TT-Forge Python environment (Python 3.11)
if [ -f "$HOME/tt-forge-fe/env/activate" ]; then
    source "$HOME/tt-forge-fe/env/activate"
    echo "   ✓ Activated TT-Forge Python environment (Python 3.11)"

    # Verify installation
    if python -c "import forge" 2>/dev/null; then
        echo "   ✓ TT-Forge installed and importable"
    else
        echo "   ⚠️  TT-Forge not found - run build steps first"
    fi
else
    echo "   ⚠️  TT-Forge environment not found at ~/tt-forge-fe/env/"
    echo "   → Run Lesson 11 build steps first"
fi

echo ""
echo "✅ TT-Forge environment ready!"
echo ""
echo "Environment status:"
echo "  Python: $(python --version 2>&1)"
echo "  CC: ${CC:-<not set>}"
echo "  CXX: ${CXX:-<not set>}"
echo "  TT_METAL_HOME: ${TT_METAL_HOME:-<unset> ✓}"
echo "  TT_METAL_VERSION: ${TT_METAL_VERSION:-<unset> ✓}"
echo ""
echo "You can now:"
echo "  1. Build TT-Forge:"
echo "     cd ~/tt-forge-fe"
echo "     cmake -B env/build env"
echo "     cmake --build env/build  # Takes 45-60 minutes!"
echo ""
echo "  2. Run TT-Forge demos:"
echo "     python ~/tt-scratchpad/tt-forge-classifier.py"
echo ""
echo "To return to tt-metal environment:"
echo "  deactivate  # Exit this venv"
echo "  source ~/tt-metal/python_env/bin/activate  # Re-enter tt-metal"
echo "  export TT_METAL_HOME=~/tt-metal  # Restore variable"
