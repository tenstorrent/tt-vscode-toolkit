#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# TT-XLA Isolated Environment Setup
#
# This script sets up an isolated environment for TT-XLA (JAX compiler)
# that won't conflict with tt-metal direct API environment.
#
# Usage:
#   source ~/tt-scratchpad/setup-tt-xla.sh
#   # Now run TT-XLA demos...
#
# Why this is needed:
# - TT-XLA uses its own bundled runtime (doesn't need tt-metal's)
# - Setting TT_METAL_HOME or LD_LIBRARY_PATH causes conflicts
# - Must use separate Python environment (Python 3.11)

echo "🔧 Setting up TT-XLA environment..."

# CRITICAL: Unset tt-metal variables to avoid conflicts
if [ -n "$TT_METAL_HOME" ]; then
    echo "   ℹ️  Unsetting TT_METAL_HOME (TT-XLA uses bundled runtime)"
    unset TT_METAL_HOME
fi

if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "   ℹ️  Unsetting LD_LIBRARY_PATH (TT-XLA has its own libraries)"
    unset LD_LIBRARY_PATH
fi

# Set PYTHONPATH for tt-forge imports (needed for demos)
if [ -d "$HOME/tt-forge" ]; then
    export PYTHONPATH="$HOME/tt-forge:$PYTHONPATH"
    echo "   ✓ PYTHONPATH includes: ~/tt-forge"
else
    echo "   ⚠️  ~/tt-forge not found (will be created during Lesson 12)"
fi

# Activate TT-XLA Python environment (Python 3.11)
if [ -f "$HOME/tt-xla-venv/bin/activate" ]; then
    source "$HOME/tt-xla-venv/bin/activate"
    echo "   ✓ Activated TT-XLA Python environment (Python 3.11)"

    # Verify installation
    if python -c "import jax" 2>/dev/null; then
        echo "   ✓ JAX installed and importable"
    else
        echo "   ⚠️  JAX not found - run Lesson 12 installation steps"
    fi
else
    echo "   ⚠️  TT-XLA venv not found at ~/tt-xla-venv"
    echo "   → Run Lesson 12 installation steps first"
fi

echo ""
echo "✅ TT-XLA environment ready!"
echo ""
echo "Environment status:"
echo "  Python: $(python --version 2>&1)"
echo "  TT_METAL_HOME: ${TT_METAL_HOME:-<unset> ✓}"
echo "  LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-<unset> ✓}"
echo "  PYTHONPATH: $PYTHONPATH"
echo ""
echo "You can now run TT-XLA demos:"
echo "  python ~/tt-forge/demos/gpt2_demo.py"
echo ""
echo "To return to tt-metal environment:"
echo "  deactivate  # Exit this venv"
echo "  source ~/tt-metal/python_env/bin/activate  # Re-enter tt-metal"
