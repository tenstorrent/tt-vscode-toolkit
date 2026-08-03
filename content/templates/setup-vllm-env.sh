#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Setup vLLM environment for Tenstorrent hardware
#
# This script creates a Python environment for serving on Tenstorrent hardware
# with upstream vLLM plus the Tenstorrent vLLM platform plugin
# (https://github.com/tenstorrent/vllm-tt-plugin).
#
# What this script does:
# 1. Validates prerequisites (tt-metal present, plugin checkout present)
# 2. Activates a Python env that has an importable `ttnn`
# 3. Installs upstream vLLM + the plugin via the plugin's own installer
# 4. Installs the extra deps ttnn and the tt-metal model tree need
# 5. Verifies the plugin is discovered (imports AND entry points)
#
# Usage:
#   bash ~/tt-scratchpad/setup-vllm-env.sh
#
# Override paths if your layout differs:
#   TT_METAL_HOME=~/tt-metal VLLM_TT_PLUGIN_SRC=~/vllm-tt-plugin \
#   PYTHON_ENV_DIR=~/.tenstorrent-venv bash ~/tt-scratchpad/setup-vllm-env.sh

set -e  # Exit on error

echo "============================================================"
echo "🚀 Setting up vLLM + TT plugin for Tenstorrent Hardware"
echo "============================================================"
echo ""

# Step 1: Validate prerequisites
echo "📋 Step 1/5: Validating prerequisites..."

if [ -z "$TT_METAL_HOME" ]; then
    TT_METAL_HOME="$HOME/tt-metal"
    echo "   TT_METAL_HOME not set, using default: $TT_METAL_HOME"
fi

if [ ! -d "$TT_METAL_HOME" ]; then
    echo "   ❌ ERROR: tt-metal not found at $TT_METAL_HOME"
    echo "   Please install tt-metal first or set TT_METAL_HOME correctly"
    exit 1
fi
echo "   ✓ tt-metal found at $TT_METAL_HOME"

PLUGIN_SRC="${VLLM_TT_PLUGIN_SRC:-$HOME/vllm-tt-plugin}"
if [ ! -d "$PLUGIN_SRC" ]; then
    echo "   ❌ ERROR: TT vLLM plugin not found at $PLUGIN_SRC"
    echo "   Clone it first:"
    echo "      git clone https://github.com/tenstorrent/vllm-tt-plugin.git $PLUGIN_SRC"
    exit 1
fi
echo "   ✓ TT vLLM plugin found at $PLUGIN_SRC"

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "   ✓ Python version: $PYTHON_VERSION"
echo ""

# Step 2: Pick and activate the Python environment.
#
# Upstream expects the install to land in an environment that already has
# tt-metal, because the plugin activates only when `ttnn` is importable and most
# of its remaining dependencies come from there. We therefore prefer an existing
# tt-metal env over creating an empty venv.
echo "⚙️  Step 2/5: Selecting Python environment..."

if [ -z "$PYTHON_ENV_DIR" ]; then
    for candidate in "$HOME/.tenstorrent-venv" \
                     "$TT_METAL_HOME/python_env" \
                     "$TT_METAL_HOME/build/python_env_vllm"; do
        if [ -d "$candidate" ]; then
            PYTHON_ENV_DIR="$candidate"
            break
        fi
    done
fi

if [ -z "$PYTHON_ENV_DIR" ]; then
    echo "   ❌ ERROR: could not find a Python env with tt-metal installed."
    echo "   Set PYTHON_ENV_DIR explicitly, e.g.:"
    echo "      PYTHON_ENV_DIR=~/.tenstorrent-venv bash $0"
    exit 1
fi

echo "   Using: $PYTHON_ENV_DIR"
# shellcheck disable=SC1091
source "$PYTHON_ENV_DIR/bin/activate"
echo ""

# Step 3: Confirm ttnn is importable — this is the plugin's activation gate.
#
# If ttnn cannot import, the plugin will install cleanly and then never activate,
# and vLLM will start up reporting no TT hardware. Fail here instead, where the
# cause is obvious.
echo "🔎 Step 3/5: Checking that ttnn is importable..."
if python3 -c "import ttnn" >/dev/null 2>&1; then
    echo "   ✓ ttnn imports"
else
    echo "   ❌ ERROR: ttnn is not importable in $PYTHON_ENV_DIR."
    echo "      The TT platform plugin activates only when ttnn imports, so vLLM"
    echo "      would start without TT hardware."
    echo ""
    echo "      A real QB2 wires this with a .pth file listing:"
    echo "         $TT_METAL_HOME"
    echo "         $TT_METAL_HOME/ttnn"
    echo "         $TT_METAL_HOME/tools"
    echo "      Build tt-metal (with Python bindings) or point PYTHON_ENV_DIR at an"
    echo "      env that already has ttnn."
    exit 1
fi
echo ""

# Note on torch: it is deliberately NOT pinned here. The old torch==2.5.0+cpu pin
# predates the plugin and conflicts with what upstream vLLM 0.24.0 resolves.

# Step 4: Install upstream vLLM and the Tenstorrent vLLM platform plugin
#
# Tenstorrent support is an out-of-tree vLLM *platform plugin*, and its official
# home is the standalone repo:
#
#   https://github.com/tenstorrent/vllm-tt-plugin
#
# It works against *upstream* vLLM — no Tenstorrent fork. Its own installer is
# the supported entry point and pins the vLLM version it is tested against:
#
#   VLLM_TARGET_DEVICE=empty uv pip install --no-binary vllm \
#       --override docs/vllm-overrides.txt vllm==0.24.0
#   uv pip uninstall torchaudio    # CUDA wheel; transformers>=5.12 imports it if present
#   uv pip install -e .
#
# VLLM_TARGET_DEVICE=empty is correct: the `tt` platform is contributed at
# runtime by the plugin rather than compiled into vLLM. Never export it as `tt`.
#
# docs/vllm-overrides.txt is load-bearing — it pins numpy<2 (which ttnn requires)
# against vLLM's opencv floor. Without it the install appears to succeed and then
# `import ttnn` fails, and because the plugin activates only when ttnn imports,
# vLLM would start up seeing no TT hardware.
#
# No manual model registration and no wrapper script are involved — plain
# `vllm serve` works once the plugin is discovered.
echo "📦 Step 4/5: Installing upstream vLLM and the Tenstorrent vLLM plugin..."

# PLUGIN_SRC was resolved in Step 1; re-check the installer itself, because a
# directory can exist without being a usable checkout (wrong repo, shallow copy,
# or an old in-fork clone that has no docs/ at the root).
if [ ! -f "$PLUGIN_SRC/docs/install-vllm-tt.sh" ]; then
    echo "   ❌ ERROR: no installer at $PLUGIN_SRC/docs/install-vllm-tt.sh"
    echo "      Clone the plugin first:"
    echo "         git clone https://github.com/tenstorrent/vllm-tt-plugin.git $PLUGIN_SRC"
    echo ""
    echo "      Note: the older in-fork plugin (tenstorrent/vllm, dev branch,"
    echo "      plugins/vllm-tt-plugin) is being retired — use the standalone repo."
    exit 1
fi

# uv is what the upstream installer uses, and it resolves this dependency set far
# more reliably than pip does.
pip install --quiet --upgrade pip setuptools wheel uv

cd "$PLUGIN_SRC"
echo "   Using upstream installer: docs/install-vllm-tt.sh"
# shellcheck disable=SC1091
source docs/install-vllm-tt.sh

# Dependencies that neither upstream vLLM nor the plugin declares, but that ttnn
# and the tt-metal model tree need. Upstream sidesteps this by installing into the
# tt-metal env, which already has them. All three were real observed failures:
#   pandas/seaborn/ml_dtypes/graphviz  ttnn's tracy tooling imports them; a miss
#                                      shows as an opaque "error while
#                                      initializing the extension" from ttnn._ttnn
#   torchvision                        transformers' pixtral image processor
#                                      imports it while vLLM inspects the TT model
#                                      class -> "Model architectures [...] failed
#                                      to be inspected"
#   pytest                             tt-metal models/common/utility_functions.py
#                                      imports it at module scope
# The override keeps these from dragging numpy back above 2.
echo "   Installing ttnn + tt-metal model-tree dependencies..."
uv pip install --quiet --override docs/vllm-overrides.txt \
    pandas seaborn ml_dtypes graphviz networkx pytest
uv pip install --quiet --override docs/vllm-overrides.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match torchvision

# Confirm the plugin is importable AND that both entry points are registered.
# Import alone is not enough: vLLM selects the TT platform through the entry
# points, so a registration miss resurfaces much later as "no TT platform found".
python3 - <<'PYEOF' || exit 1
import sys
from importlib.metadata import entry_points, version

import vllm_tt_plugin  # noqa: F401
import ttnn  # noqa: F401

# ttnn exposes no __version__; read it from distribution metadata.
try:
    print("   ttnn:", version("ttnn"))
except Exception:
    print("   ttnn: (version unknown)")

expected = {
    ("vllm.platform_plugins", "tt"),
    ("vllm.general_plugins", "tt_model_registry"),
}
found = {
    (g, e.name)
    for g in ("vllm.platform_plugins", "vllm.general_plugins")
    for e in entry_points(group=g)
}
missing = expected - found
if missing:
    print("   ERROR: missing vLLM entry points:", sorted(missing), file=sys.stderr)
    sys.exit(1)
print("   vllm-tt-plugin entry points registered")
PYEOF

echo "   ✓ vLLM installed successfully"
echo ""

# Step 5: Validate installation
echo "🧪 Step 5/5: Validating installation..."

for probe in \
    "import torch; print('torch', torch.__version__)" \
    "import vllm; print('vllm', vllm.__version__)" \
    "import ttnn; print('ttnn import OK')" \
    "import vllm_tt_plugin; print('vllm_tt_plugin import OK')" \
    "import numpy; print('numpy', numpy.__version__)"; do
    if OUT=$(python3 -c "$probe" 2>&1); then
        echo "   ✓ $OUT"
    else
        echo "   ❌ failed: $probe"
        echo "      $OUT"
        exit 1
    fi
done

# ttnn requires numpy<2. A 2.x here means the override did not apply and ttnn will
# break on some paths even if the import above happened to succeed.
if python3 -c "import numpy,sys; sys.exit(0 if int(numpy.__version__.split('.')[0]) < 2 else 1)"; then
    echo "   ✓ numpy is <2 as ttnn requires"
else
    echo "   ❌ numpy is >=2 but ttnn requires <2 — the override did not apply"
    exit 1
fi

echo ""

# Create a convenient activation script.
echo "📝 Creating activation script..."
cat > "$HOME/activate-vllm-env.sh" <<EOF
#!/bin/bash
# Activate the vLLM + TT plugin environment for Tenstorrent hardware.
export TT_METAL_HOME="${TT_METAL_HOME}"
source "${PYTHON_ENV_DIR}/bin/activate"

# Set the mesh shape for your hardware before serving. This — not
# --tensor-parallel-size, which the TT platform rejects — is the multi-chip control.
#   N150 N300 N150x4 T3K TG | P100 P150 P150x2 P300 P150x4 P150x8 P300x2
# A TT-QuietBox 2 is P300x2 (a 1x4 mesh over all four Blackhole chips).
: "\${MESH_DEVICE:=N150}"
export MESH_DEVICE

echo "✓ vLLM environment activated"
echo "  PYTHON_ENV_DIR: ${PYTHON_ENV_DIR}"
echo "  TT_METAL_HOME:  ${TT_METAL_HOME}"
echo "  MESH_DEVICE:    \$MESH_DEVICE"
echo ""
echo "Ready to start vLLM. Remember: if --model is a local path, HF_MODEL must be"
echo "set to that same path (tt_transformers uses it as the checkpoint directory)."
EOF

chmod +x "$HOME/activate-vllm-env.sh"
echo "   ✓ Created ~/activate-vllm-env.sh"
echo ""

echo "============================================================"
echo "✅ vLLM + TT plugin setup complete!"
echo "============================================================"
echo ""
echo "📚 Quick Start:"
echo ""
echo "1. Activate the environment:"
echo "   source ~/activate-vllm-env.sh"
echo ""
echo "2. Set the mesh shape for your hardware and serve:"
echo "   export MESH_DEVICE=N150     # N300 / T3K / P100 / P150 / P300 / P300x2"
echo "   export HF_MODEL=~/models/Qwen2.5-Coder-1.5B-Instruct   # required for local paths"
echo "   vllm serve ~/models/Qwen2.5-Coder-1.5B-Instruct --block-size 64"
echo ""
echo "   Startup logs 'Platform plugin tt is activated' when discovery worked."
echo ""
echo "3. In a new terminal, test the server:"
echo "   curl http://localhost:8000/health"
echo ""
echo "📖 Environment Details:"
echo "   Python:   $PYTHON_VERSION"
echo "   Location: $PYTHON_ENV_DIR"
echo "   TT-Metal: $TT_METAL_HOME"
echo "   Plugin:   $PLUGIN_SRC"
echo ""
echo "Happy coding! 🚀"
