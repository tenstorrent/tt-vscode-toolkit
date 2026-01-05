#!/bin/bash
# Cleanup script for tt-vscode-toolkit
# Removes build artifacts, cache files, and old packages

echo "🧹 Cleaning up tt-vscode-toolkit..."

# Remove Python cache files from source
echo "Removing Python cache files..."
find content -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null
find content -type f -name "*.pyc" -delete 2>/dev/null
find content -type f -name "*.pyo" -delete 2>/dev/null

# Remove old .vsix packages
echo "Removing old .vsix packages..."
rm -f *.vsix

# Remove dist directory
echo "Removing dist/ directory..."
rm -rf dist/

# Remove .DS_Store files (macOS)
echo "Removing .DS_Store files..."
find . -name ".DS_Store" -delete 2>/dev/null

echo "✅ Cleanup complete!"
echo ""
echo "Next steps:"
echo "  npm run build    - Rebuild the extension"
echo "  npm run package  - Create a fresh .vsix"
