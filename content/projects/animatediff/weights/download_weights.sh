#!/bin/bash
# Download AnimateDiff motion module weights

set -e

echo "================================================"
echo "AnimateDiff Weight Download"
echo "================================================"
echo

# Check if hf CLI is installed
if ! command -v hf &> /dev/null; then
    echo "❌ hf CLI not found"
    echo
    echo "Install it with:"
    echo "  pip install huggingface_hub"
    echo
    exit 1
fi

# Set download location
WEIGHTS_DIR="${HOME}/models/animatediff"
mkdir -p "$WEIGHTS_DIR"

echo "Download location: $WEIGHTS_DIR"
echo

# Download AnimateDiff motion module
echo "Downloading mm_sd_v15_v2.ckpt (1.7GB)..."
echo "This may take a few minutes..."
echo

hf download \
    guoyww/animatediff \
    mm_sd_v15_v2.ckpt \
    --local-dir "$WEIGHTS_DIR"

echo
echo "================================================"
echo "✓ Download Complete!"
echo "================================================"
echo
echo "Weights saved to: $WEIGHTS_DIR/mm_sd_v15_v2.ckpt"
echo "Size: $(du -h "$WEIGHTS_DIR/mm_sd_v15_v2.ckpt" | cut -f1)"
echo
echo "You can now run the examples:"
echo "  python examples/generate_2frame_video.py"
echo "  python examples/generate_16frame_video.py"
echo
