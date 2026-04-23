#!/bin/bash
# post-create.sh — runs once after the dev container is created.
# Sets up the ttlang toolchain marker and optionally downloads ttsim binaries.
set -e

echo "=== Tenstorrent Simulator Dev Container Setup ==="

# Create the toolchain marker directory that detectExecutionContext() checks.
# This signals to the VSCode extension that we're in simulator-only mode.
sudo mkdir -p /opt/ttlang-toolchain
echo "tt-lang-dist" | sudo tee /opt/ttlang-toolchain/version > /dev/null
echo "[OK] toolchain marker created at /opt/ttlang-toolchain"

# Install Python dependencies for ttlang-sim-lite
echo "Installing Python dependencies..."
pip install --quiet greenlet numpy pydantic
echo "[OK] numpy, greenlet, pydantic installed"

# Install tt-lang Python package if present in the container
if python3 -c "import ttl" 2>/dev/null; then
  echo "[OK] ttl package already available"
elif [ -f "/opt/tt-lang/setup.py" ] || [ -f "/opt/tt-lang/pyproject.toml" ]; then
  pip install --quiet -e /opt/tt-lang
  echo "[OK] tt-lang installed from /opt/tt-lang"
else
  echo "[INFO] tt-lang not found; ttlang-sim-lite (numpy backend) will be used for browser lessons"
fi

# Optional: download ttsim hardware-emulation binaries
# Uncomment and update SIM_VERSION to download prebuilt .so files:
#
# SIM_VERSION="v0.5"
# mkdir -p ~/sim
# if command -v wget &>/dev/null; then
#   wget -q "https://github.com/tenstorrent/ttsim/releases/download/${SIM_VERSION}/libttsim_wh.so" \
#        -O ~/sim/libttsim_wh.so && echo "[OK] libttsim_wh.so downloaded"
# fi
# echo "ttsim available at ~/sim/"

echo ""
echo "=== Setup complete ==="
echo "  Simulator mode: TTLANG_SIM_ONLY=1"
echo "  Hardware-only commands will show an info message instead of running."
echo "  Open a lesson in VSCode to get started."
