# Zen→Trickster Cleanup Instructions

**Purpose**: Remove all remaining "zen master" references from Custom Training lessons in favor of the trickster theme.

**Status**: Trickster files already exist and lessons reference them correctly, but old zen artifacts remain and must be deleted.

**Target repo**: `tt-vscode-toolkit`

---

## File Migration History (Already Done)

The following files were previously created to replace zen versions:

| Zen File (OLD) | Trickster File (NEW) | Status |
|----------------|----------------------|--------|
| `finetune_zen.py` | `finetune_trickster.py` | ✅ Exists |
| `test_zen_model.py` | `test_trickster.py` | ✅ Exists |
| `zen_dataset_starter.jsonl` | `trickster_dataset_starter.jsonl` | ✅ Exists |
| `finetune_zen_n150.yaml` | `trickster_n150.yaml` | ✅ Exists |
| `finetune_zen_n300.yaml` | `trickster_n300.yaml` | ✅ Exists |
| `finetune_zen_blackhole.yaml` | *N/A (not needed)* | — |
| *N/A* | `nano_trickster.py` | ✅ Exists (CT8) |
| *N/A* | `nano_trickster.yaml` | ✅ Exists (CT8) |

**Naming pattern used**:
- `finetune_zen.py` → `finetune_trickster.py` (replace "zen" with "trickster")
- `test_zen_model.py` → `test_trickster.py` (replace "zen_model" with "trickster")
- `zen_dataset_starter.jsonl` → `trickster_dataset_starter.jsonl` (replace "zen" with "trickster")
- `finetune_zen_*.yaml` → `trickster_*.yaml` (remove "finetune_" prefix, replace "zen" with "trickster")

---

## Files to DELETE (Action Required)

### Python Scripts (2 files)
```bash
rm content/templates/training/finetune_zen.py
rm content/templates/training/test_zen_model.py
```

### Config Files (3 files)
```bash
rm content/templates/training/configs/finetune_zen_n150.yaml
rm content/templates/training/configs/finetune_zen_n300.yaml
rm content/templates/training/configs/finetune_zen_blackhole.yaml
```

### Dataset Files (1 file)
```bash
rm content/templates/training/zen_dataset_starter.jsonl
```

**Total**: 6 zen artifact files to remove

---

## Files That Are CORRECT (Keep These!)

### Python Scripts ✅
- `content/templates/training/finetune_trickster.py` (22KB) - CT4 fine-tuning
- `content/templates/training/test_trickster.py` (11KB) - CT4 testing
- `content/templates/training/nano_trickster.py` (13KB) - CT8 architecture

### Config Files ✅
- `content/templates/training/configs/trickster_n150.yaml` (2.0KB) - Single-chip
- `content/templates/training/configs/trickster_n300.yaml` (2.1KB) - Dual-chip
- `content/templates/training/configs/nano_trickster.yaml` (2.4KB) - From-scratch

### Dataset Files ✅
- `content/templates/training/trickster_dataset_starter.jsonl` (11KB) - 50 examples

---

## Verification Steps

After deletion, verify no zen references remain:

```bash
cd /path/to/tt-vscode-toolkit

# Search for any remaining zen references
grep -r "zen" content/lessons/ct*.md content/templates/training/ 2>/dev/null | grep -i "zen"

# Expected: Should only find lowercase "zen" in unrelated contexts (if any)
# Should NOT find: finetune_zen, test_zen_model, zen_dataset, etc.
```

---

## Lessons Status

### CT4 (Fine-tuning Basics) ✅
**Already correct** - References trickster files:
- Line 238: `trickster_dataset_starter.jsonl`
- Line 340: `finetune_trickster.py`
- Line 354: `trickster_dataset_starter.jsonl`
- Line 536: `test_trickster.py`
- Line 760: `finetune_trickster.py`
- Lines 821-822: trickster file documentation

**No changes needed to CT4**

### CT8 (Training from Scratch) ✅
**Already correct** - References:
- `nano_trickster.py` - Model architecture
- `nano_trickster.yaml` - Config for from-scratch training

**No changes needed to CT8**

---

## Complete File Renaming Guide (For Reference)

**Note**: The trickster files already exist. This section documents how they were created from zen files for future reference.

### How Zen Files Were Renamed to Trickster

```bash
# Python scripts
finetune_zen.py → finetune_trickster.py
# Pattern: Replace "zen" with "trickster"

test_zen_model.py → test_trickster.py
# Pattern: Replace "zen_model" with "trickster"

# Dataset
zen_dataset_starter.jsonl → trickster_dataset_starter.jsonl
# Pattern: Replace "zen" with "trickster"

# Config files
finetune_zen_n150.yaml → trickster_n150.yaml
# Pattern: Remove "finetune_" prefix, replace "zen" with "trickster"

finetune_zen_n300.yaml → trickster_n300.yaml
# Pattern: Remove "finetune_" prefix, replace "zen" with "trickster"

finetune_zen_blackhole.yaml → (deleted, no trickster equivalent needed)
```

### Within File Content Changes

When files were renamed, internal references were also updated:

1. **Python imports and paths**:
   ```python
   # OLD
   from zen_model import ZenModel
   config_path = "configs/finetune_zen_n150.yaml"

   # NEW
   from trickster_model import TricksterModel
   config_path = "configs/trickster_n150.yaml"
   ```

2. **Model class names**:
   ```python
   # OLD
   class ZenMaster(nn.Module):

   # NEW
   class Trickster(nn.Module):
   ```

3. **Dataset file references**:
   ```python
   # OLD
   data_file = "zen_dataset_starter.jsonl"

   # NEW
   data_file = "trickster_dataset_starter.jsonl"
   ```

4. **Comments and docstrings**:
   ```python
   # OLD
   """Fine-tune a zen master model"""
   # Train the zen master

   # NEW
   """Fine-tune a trickster model"""
   # Train the trickster
   ```

---

## Step-by-Step Cleanup Instructions

### Step 1: Navigate to Repository
```bash
cd ~/tt-vscode-toolkit
```

### Step 2: Delete Zen Python Scripts
```bash
# Delete finetune script (replaced by finetune_trickster.py)
rm content/templates/training/finetune_zen.py

# Delete test script (replaced by test_trickster.py)
rm content/templates/training/test_zen_model.py
```

### Step 3: Delete Zen Config Files
```bash
# Delete N150 config (replaced by trickster_n150.yaml)
rm content/templates/training/configs/finetune_zen_n150.yaml

# Delete N300 config (replaced by trickster_n300.yaml)
rm content/templates/training/configs/finetune_zen_n300.yaml

# Delete Blackhole config (no trickster equivalent needed)
rm content/templates/training/configs/finetune_zen_blackhole.yaml
```

### Step 4: Delete Zen Dataset
```bash
# Delete zen dataset (replaced by trickster_dataset_starter.jsonl)
rm content/templates/training/zen_dataset_starter.jsonl
```

### Step 5: Verify Cleanup
```bash
# Search for any remaining zen references in files
grep -r "finetune_zen\|test_zen_model\|zen_dataset" content/lessons/ content/templates/training/ 2>/dev/null

# Expected: Should return no results (exit code 1 means no matches found)
# If references found, review them to ensure they're not false positives
```

### Step 6: Check Git Status
```bash
git status
```

**Expected output**:
```
deleted:    content/templates/training/finetune_zen.py
deleted:    content/templates/training/test_zen_model.py
deleted:    content/templates/training/configs/finetune_zen_n150.yaml
deleted:    content/templates/training/configs/finetune_zen_n300.yaml
deleted:    content/templates/training/configs/finetune_zen_blackhole.yaml
deleted:    content/templates/training/zen_dataset_starter.jsonl
```

## Automated Cleanup Script (Optional)

If you prefer a single script to run all commands:

```bash
#!/bin/bash
# zen_cleanup.sh - Remove zen artifacts from tt-vscode-toolkit

set -e

cd ~/tt-vscode-toolkit

echo "🧹 Cleaning up zen artifacts..."

# Remove zen Python scripts
rm -f content/templates/training/finetune_zen.py
rm -f content/templates/training/test_zen_model.py
echo "  ✅ Removed zen Python scripts"

# Remove zen configs
rm -f content/templates/training/configs/finetune_zen_n150.yaml
rm -f content/templates/training/configs/finetune_zen_n300.yaml
rm -f content/templates/training/configs/finetune_zen_blackhole.yaml
echo "  ✅ Removed zen config files"

# Remove zen dataset
rm -f content/templates/training/zen_dataset_starter.jsonl
echo "  ✅ Removed zen dataset"

# Verify no zen references remain
echo ""
echo "🔍 Verifying cleanup..."
ZEN_REFS=$(grep -r "finetune_zen\|test_zen_model\|zen_dataset" content/ 2>/dev/null | wc -l)

if [ "$ZEN_REFS" -eq 0 ]; then
    echo "  ✅ No zen artifact references found!"
else
    echo "  ⚠️  Warning: Found $ZEN_REFS zen references:"
    grep -r "finetune_zen\|test_zen_model\|zen_dataset" content/ 2>/dev/null
    exit 1
fi

echo ""
echo "✨ Zen cleanup complete! Trickster reigns supreme."
```

**Usage**:
```bash
# Make script executable
chmod +x zen_cleanup.sh

# Run cleanup
./zen_cleanup.sh
```

---

## Git Commit

After running cleanup:

```bash
git status  # Should show 6 deletions

git add -A  # Stage all deletions

git commit -m "Remove zen artifacts in favor of trickster theme

- Deleted finetune_zen.py, test_zen_model.py (replaced by trickster versions)
- Deleted zen config files (finetune_zen_*.yaml)
- Deleted zen_dataset_starter.jsonl (replaced by trickster_dataset_starter.jsonl)
- Lessons CT4/CT8 already reference trickster files correctly
- Trickster theme now complete across all Custom Training content"
```

---

## Why This Cleanup Matters

### User Feedback
> "we still have zen master refs everywhere and it's distasteful to people, our selves, and the trickster gods we serve"

### The Issue
- Lessons were updated to reference trickster files (CT4, CT8)
- But old zen files were left in the repository
- Creates confusion: "Do I use zen or trickster files?"
- Zen theme doesn't match the witty, irreverent trickster persona

### The Solution
- Remove all zen artifacts
- Keep only trickster files
- Single clear path for users
- Theme consistency across all Custom Training

---

## Trickster Theme Philosophy

**Trickster**: Creative, flexible, witty AI for explaining concepts

**Example responses** (from trickster_dataset_starter.jsonl):
- "Training is teaching the model (slow, uses lots of memory, updates weights). Inference is using the model (fast, less memory, weights frozen). One builds, one deploys."
- "Start by breaking things. Then learn why they broke. Then break them again, but differently. Repeat until you're hired."

**Contrast with zen master**: Too serious, meditative, not playful

**Trickster matches the codebase vibe**: Practical, no-nonsense, with personality

---

## File Size Comparison

**Zen artifacts (being deleted)**:
- `zen_dataset_starter.jsonl`: 23KB (larger dataset)
- `finetune_zen.py`: 22KB
- `test_zen_model.py`: 11KB
- 3 config files: ~5KB total
- **Total**: ~61KB of zen artifacts

**Trickster files (keeping)**:
- `trickster_dataset_starter.jsonl`: 11KB (50 concise examples)
- `finetune_trickster.py`: 22KB
- `test_trickster.py`: 11KB
- 3 config files: ~6KB total
- **Total**: ~50KB of trickster files

**Benefit**: Cleaner, more focused, single theme

---

## Validation

After cleanup, these commands should work:

```bash
# CT4 Fine-tuning (with trickster dataset)
cd ~/tt-scratchpad/training
python finetune_trickster.py --config configs/trickster_n150.yaml

# CT8 From-scratch (with nano-trickster)
python train_from_scratch.py --config configs/nano_trickster.yaml
```

Both commands use **trickster** theme exclusively - no zen confusion!

---

## Summary

**Action**: Delete 6 zen artifact files

**Impact**:
- ✅ Removes theme confusion
- ✅ Single clear path for users
- ✅ Matches trickster personality
- ✅ Lessons already correct (no changes needed)

**Time**: 2 minutes (run cleanup script + commit)

**Benefit**: Respect the trickster gods we serve 🎭

---

**🎭 "The trickster doesn't meditate - they iterate, break things, and learn by doing!"**
