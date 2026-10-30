# Quick Start Guide

Apply all Custom Training validation fixes in 5 minutes.

---

## Prerequisites

✅ Have tt-vscode-toolkit repository cloned with git access
✅ Have this validation package extracted

---

## Step 1: Navigate to Repository

```bash
cd /path/to/tt-vscode-toolkit
```

---

## Step 2: Apply Patches (2 minutes)

### Apply API Compatibility Fix

```bash
git apply /path/to/custom-training-validation-package/patches/finetune_trickster_api_v066_fix.patch
```

**What this does**: Updates `finetune_trickster.py` for v0.66.0-rc7 API (7 fixes)

### Apply Version Requirement Update

```bash
git apply /path/to/custom-training-validation-package/patches/version_requirement_update.patch
```

**What this does**: Updates CT4/CT8 lessons to specify minimum tt-metal v0.66.0-rc5+

---

## Step 3: Remove Zen Artifacts (1 minute)

```bash
/path/to/custom-training-validation-package/scripts/zen_cleanup.sh
```

**What this does**: Removes 6 zen theme files (will prompt for confirmation)

---

## Step 4: Verify Changes

```bash
git status
# Should show:
#   modified: content/templates/training/finetune_trickster.py
#   modified: content/lessons/ct4-finetuning-basics.md
#   modified: content/lessons/ct8-training-from-scratch.md
#   deleted: 6 zen artifact files

git diff --stat
# Review changes
```

---

## Step 5: Update Version & Commit

### Update package.json

Change version: `0.0.297` → `0.0.298`

### Commit Changes

```bash
git add -A
git commit -m "Fix: Update Custom Training for tt-metal v0.66.0-rc5+ compatibility

- Update finetune_trickster.py API for v0.66.0-rc7 (7 fixes)
- Update CT4/CT8 minimum version requirement to v0.66.0-rc5+
- Remove zen artifacts in favor of trickster theme (6 files)
- Add hardware requirement notes (N150 vs N300+ for different models)

Validated on N150 hardware with both from-scratch (CT8) and fine-tuning
(trickster) workflows. All 8 Custom Training lessons now production-ready."
```

---

## ✅ Done!

**What changed**:
- ✅ finetune_trickster.py: v0.66.0-rc7 API compatible
- ✅ CT4/CT8 lessons: Minimum version specified
- ✅ Zen artifacts: Removed (trickster theme complete)
- ✅ Documentation: Hardware requirements clarified

**Next steps**:
1. Build extension: `npm run build`
2. Package extension: `npm run package`
3. Test on hardware (if available)

---

## 🐛 Quick Troubleshooting

### Patch fails to apply

**Check**: Are you in tt-vscode-toolkit root directory?
```bash
pwd  # Should end with /tt-vscode-toolkit
```

**Check**: Has it already been applied?
```bash
git log --oneline | head -3
```

### Zen cleanup script fails

**Fix**: Run from tt-vscode-toolkit directory
```bash
cd /path/to/tt-vscode-toolkit
/path/to/scripts/zen_cleanup.sh
```

---

## 📚 Need More Detail?

See **README.md** in this package for:
- Detailed explanations of each change
- Training results and metrics
- Complete troubleshooting guide
- Testing recommendations

See **documentation/** directory for:
- Complete validation results
- API fix details
- Training performance data
- Full session documentation

---

**Package version**: 1.0
**Date**: 2026-02-03
**Status**: Ready to apply ✅
