#!/bin/bash
# zen_cleanup.sh - Remove zen artifacts from tt-vscode-toolkit
# This script removes all "zen master" themed files in favor of trickster theme

set -e

echo "🧹 Zen Artifact Cleanup Script"
echo "=============================="
echo ""
echo "This will remove 6 zen artifact files from tt-vscode-toolkit."
echo "Trickster replacements already exist - this just cleans up old files."
echo ""

# Check if we're in the right directory
if [ ! -d "content/templates/training" ]; then
    echo "❌ Error: Not in tt-vscode-toolkit directory"
    echo "   Please cd to tt-vscode-toolkit first, then run this script"
    exit 1
fi

echo "📂 Current directory: $(pwd)"
echo ""

# Show what will be deleted
echo "Files to be deleted:"
echo "  - content/templates/training/finetune_zen.py"
echo "  - content/templates/training/test_zen_model.py"
echo "  - content/templates/training/zen_dataset_starter.jsonl"
echo "  - content/templates/training/configs/finetune_zen_n150.yaml"
echo "  - content/templates/training/configs/finetune_zen_n300.yaml"
echo "  - content/templates/training/configs/finetune_zen_blackhole.yaml"
echo ""

read -p "Continue with deletion? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

echo ""
echo "🗑️  Removing zen artifacts..."

# Remove zen Python scripts
if [ -f "content/templates/training/finetune_zen.py" ]; then
    rm -f content/templates/training/finetune_zen.py
    echo "  ✅ Removed finetune_zen.py"
else
    echo "  ⏭️  finetune_zen.py not found (already removed)"
fi

if [ -f "content/templates/training/test_zen_model.py" ]; then
    rm -f content/templates/training/test_zen_model.py
    echo "  ✅ Removed test_zen_model.py"
else
    echo "  ⏭️  test_zen_model.py not found (already removed)"
fi

# Remove zen dataset
if [ -f "content/templates/training/zen_dataset_starter.jsonl" ]; then
    rm -f content/templates/training/zen_dataset_starter.jsonl
    echo "  ✅ Removed zen_dataset_starter.jsonl"
else
    echo "  ⏭️  zen_dataset_starter.jsonl not found (already removed)"
fi

# Remove zen configs
if [ -f "content/templates/training/configs/finetune_zen_n150.yaml" ]; then
    rm -f content/templates/training/configs/finetune_zen_n150.yaml
    echo "  ✅ Removed finetune_zen_n150.yaml"
else
    echo "  ⏭️  finetune_zen_n150.yaml not found (already removed)"
fi

if [ -f "content/templates/training/configs/finetune_zen_n300.yaml" ]; then
    rm -f content/templates/training/configs/finetune_zen_n300.yaml
    echo "  ✅ Removed finetune_zen_n300.yaml"
else
    echo "  ⏭️  finetune_zen_n300.yaml not found (already removed)"
fi

if [ -f "content/templates/training/configs/finetune_zen_blackhole.yaml" ]; then
    rm -f content/templates/training/configs/finetune_zen_blackhole.yaml
    echo "  ✅ Removed finetune_zen_blackhole.yaml"
else
    echo "  ⏭️  finetune_zen_blackhole.yaml not found (already removed)"
fi

echo ""
echo "🔍 Verifying cleanup..."

# Verify no zen references remain in filenames
ZEN_FILES=$(find content/templates/training -name "*zen*" 2>/dev/null | wc -l)

if [ "$ZEN_FILES" -eq 0 ]; then
    echo "  ✅ No zen artifact files found!"
else
    echo "  ⚠️  Warning: Found $ZEN_FILES files with 'zen' in name:"
    find content/templates/training -name "*zen*" 2>/dev/null
fi

echo ""
echo "✨ Zen cleanup complete!"
echo ""
echo "Next steps:"
echo "  1. Review changes: git status"
echo "  2. Commit deletions: git add -A && git commit -m 'Remove zen artifacts, use trickster theme'"
echo "  3. Apply version patch: git apply ../patches/version_requirement_update.patch"
echo ""
