# vLLM Installation Drift Analysis

**Date:** 2026-03-04
**vLLM Repository:** https://github.com/tenstorrent/vllm (dev branch)
**Extension Version:** 0.0.338

## Executive Summary

The official Tenstorrent vLLM repository has made significant changes to the installation process that are not reflected in our Lesson 7 (vllm-production.md) and associated setup scripts. The key change is the adoption of **`uv`** - a fast Python package installer that replaces traditional `pip` commands.

## Key Changes

### 1. Installation Command - Using `uv` Instead of `pip`

**Current Repository (tt-vllm/tt_metal/install-vllm-tt.sh):**
```bash
export VLLM_TARGET_DEVICE="tt"
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

**Our Current Lesson (vllm-production.md:369):**
```bash
cd $vllm_dir
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

**Our Setup Script (setup-vllm-env.sh:134):**
```bash
pip install --quiet -e . --extra-index-url https://download.pytorch.org/whl/cpu
```

**Changes:**
1. **`uv pip install` instead of `pip install`** - Uses uv's pip interface for faster installation
2. **New flag: `--index-strategy unsafe-best-match`** - Allows uv to select the best matching package versions

### 2. Branch References

**Repository README (vllm/README.md:2):**
```markdown
To use vLLM with Tenstorrent hardware, please follow the instructions in
tt_metal/README.md (Note: `dev` is the main development branch).
```

**Our Lesson:**
- ✅ Correctly references `dev` branch
- ✅ Clone command uses `--branch dev`

### 3. Installation Script Location

**Repository:**
- Installation script: `tt_metal/install-vllm-tt.sh`
- Comprehensive docs: `tt_metal/README.md`

**Our Lesson:**
- ✅ References correct paths
- ❌ Does not use official installation script
- ❌ Manually replicates steps instead of running official script

## What is `uv`?

**uv** is a fast Python package installer and resolver written in Rust, created by Astral (makers of Ruff).

**Key Features:**
- 🚀 **10-100x faster** than pip for dependency resolution and installation
- ⚡ **Better conflict resolution** - smarter dependency solving
- 🔧 **pip-compatible** - drop-in replacement with `uv pip` commands
- 📦 **Built-in caching** - reuses downloaded packages across projects
- 🦀 **Rust-powered** - compiled binary, no Python overhead

**Installation:**
```bash
# Standalone installer (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via pip (if uv not available)
pip install uv

# Or via Homebrew (macOS)
brew install uv
```

**Usage Examples:**
```bash
# Instead of: pip install package
uv pip install package

# Instead of: pip install -e .
uv pip install -e .

# Instead of: pip install -r requirements.txt
uv pip install -r requirements.txt
```

**Why vLLM uses it:**
- Faster dependency resolution for complex dependency trees
- More reliable package conflict resolution
- Reduced installation time (important for large projects like vLLM)
- Better handling of PyTorch CPU/GPU index URLs

## Detailed Comparison

### Official Install Process (tt_metal/README.md)

**Step 1: Install tt-metal**
```bash
# Follow INSTALLING.md instructions
# Build tt-metal and create virtual environment
```

**Step 2: Enter tt-metal environment and install vLLM**
```bash
# From root vLLM directory (dev branch)
source tt_metal/install-vllm-tt.sh
```

**What the official script does:**
```bash
export VLLM_TARGET_DEVICE="tt"
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
```

### Our Current Process

**Step 1: Clone vLLM**
```bash
git clone --branch dev https://github.com/tenstorrent/vllm.git ~/tt-vllm
```

**Step 2: Run our setup script**
```bash
bash ~/tt-scratchpad/setup-vllm-env.sh
```

**What our script does:**
1. Source `tt_metal/setup-metal.sh`
2. Create venv at `$PYTHON_ENV_DIR`
3. Install PyTorch 2.5.0+cpu manually
4. Install ttnn, pytest, dependencies manually
5. Install vLLM with `pip install -e .`

## Recommendations

### Option 1: Align with Official Process (Recommended)

**Pros:**
- Always up-to-date with official installation method
- Less maintenance burden
- User learns official workflow
- Inherits future improvements automatically

**Cons:**
- Requires uv installation
- Less control over installation steps
- Harder to troubleshoot if issues occur

**Implementation:**
1. Document uv installation
2. Update lesson to use official `install-vllm-tt.sh` script
3. Provide fallback instructions if uv not available

### Option 2: Keep Custom Process with uv

**Pros:**
- Detailed step-by-step visibility
- Easier to troubleshoot
- More educational for users

**Cons:**
- Must manually track upstream changes
- Higher maintenance burden
- May drift again in future

**Implementation:**
1. Update `setup-vllm-env.sh` to use uv
2. Add uv installation check/fallback
3. Add `--index-strategy unsafe-best-match` flag

### Option 3: Hybrid Approach

**Pros:**
- Official method for production
- Custom method for learning/troubleshooting
- Best of both worlds

**Cons:**
- More documentation to maintain
- Two paths to test and validate

**Implementation:**
1. Add "Quick Install" section using official script
2. Keep "Manual Install" section with uv
3. Document when to use each approach

## Proposed Changes

### 1. Update vllm-production.md Lesson

**Add uv installation section:**
```markdown
## Prerequisites

- tt-metal installed and working
- Model downloaded
- Python 3.10+ recommended
- **uv installed** (Python package installer)

### Installing uv

uv is a fast Python package installer used by vLLM for reliable dependency management.

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Verify installation:**
```bash
uv --version
```

**If uv is not available,** you can still use regular pip, but installation will be slower.
```

**Update installation instructions:**
```markdown
## Step 2: Install vLLM (Official Method)

**Recommended approach using official installation script:**

```bash
cd ~/tt-vllm
source tt_metal/install-vllm-tt.sh
```

This script:
- Sets `VLLM_TARGET_DEVICE="tt"`
- Installs vLLM using uv for faster, more reliable installation
- Handles all dependencies and PyTorch versions automatically

**Alternative: Manual installation (if troubleshooting needed):**

```bash
cd ~/tt-vllm
export vllm_dir=$(pwd)
source $vllm_dir/tt_metal/setup-metal.sh
source $PYTHON_ENV_DIR/bin/activate

# Using uv (faster)
uv pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

# OR using regular pip (slower, but works without uv)
pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu
```
```

### 2. Update setup-vllm-env.sh Script

**Add uv detection and usage:**
```bash
# Step 5: Install vLLM and dependencies
echo "📦 Step 5/6: Installing vLLM and dependencies..."

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "   ✓ Using uv for fast installation"
    INSTALL_CMD="uv pip install"
    INDEX_STRATEGY="--index-strategy unsafe-best-match"
else
    echo "   ℹ️  uv not found, using standard pip (slower)"
    echo "   Tip: Install uv for faster installs: curl -LsSf https://astral.sh/uv/install.sh | sh"
    INSTALL_CMD="pip install"
    INDEX_STRATEGY=""
fi

# Install ttnn and pytest (required for TT hardware)
echo "   Installing ttnn and pytest..."
$INSTALL_CMD --quiet --upgrade ttnn pytest

# Install additional dependencies
echo "   Installing additional dependencies..."
$INSTALL_CMD --quiet fairscale termcolor loguru blobfile fire pytz llama-models==0.0.48

# Install vLLM in editable mode
echo "   Installing vLLM from source (this may take 5-10 minutes)..."
cd "$vllm_dir"
$INSTALL_CMD --quiet -e . --extra-index-url https://download.pytorch.org/whl/cpu $INDEX_STRATEGY
```

### 3. Add uv Installation Command

**Create convenience command in extension:**
```typescript
vscode.commands.registerCommand('tenstorrent.installUv', async () => {
  const terminal = getOrCreateTerminal('main');
  runInTerminal(terminal, 'curl -LsSf https://astral.sh/uv/install.sh | sh');
  vscode.window.showInformationMessage('Installing uv... Check terminal for progress.');
});
```

**Add to lesson:**
```markdown
[📦 Install uv (Fast Python Package Installer)](command:tenstorrent.installUv)
```

## Testing Plan

1. **Test uv installation:**
   - Install uv on clean system
   - Verify `uv --version` works
   - Test `uv pip install` commands

2. **Test official script:**
   - Clone fresh vllm repo
   - Run `source tt_metal/install-vllm-tt.sh`
   - Verify vLLM installation
   - Test server startup

3. **Test updated setup script:**
   - With uv installed
   - Without uv installed (fallback to pip)
   - Verify both paths work

4. **Test on different hardware:**
   - N150
   - N300
   - T3K
   - P100

## Migration Path

**Phase 1: Documentation (Immediate)**
1. Add VLLM_INSTALLATION_DRIFT.md (this file)
2. Update CLAUDE.md to reference vendor/vllm checkout
3. Add note to vllm-production.md about upcoming changes

**Phase 2: Implementation (Next Release)**
1. Update vllm-production.md with uv instructions
2. Update setup-vllm-env.sh with uv support
3. Add uv installation command
4. Test on all hardware types

**Phase 3: Validation (Before Merge)**
1. Test full installation flow with uv
2. Test fallback flow without uv
3. Update FAQ with uv troubleshooting
4. Get user feedback

## Additional Findings

### Requirements Files

**Repository has multiple requirement files:**
- `requirements/tt.txt` - TT-specific dependencies
- `requirements/build.txt` - Build dependencies
- `requirements/common.txt` - Common dependencies
- `requirements/cuda.txt` - CUDA dependencies

**Our script manually installs packages instead of using requirements files.**

**Recommendation:** Consider using requirements files for more maintainable dependency management:
```bash
pip install -r requirements/tt.txt
```

### Environment Variables

**Official script sets:**
```bash
export VLLM_TARGET_DEVICE="tt"
```

**Our lesson/scripts:**
- ❌ Do not consistently set `VLLM_TARGET_DEVICE`
- ✅ Set `TT_METAL_HOME`, `MESH_DEVICE` correctly

**Recommendation:** Add `VLLM_TARGET_DEVICE="tt"` to all vLLM-related commands and scripts.

### Activation Flow

**Official approach:**
```bash
# After installation
export VLLM_TARGET_DEVICE="tt" && source $PYTHON_ENV_DIR/bin/activate
```

**Our approach:**
```bash
# Using activation script
source ~/activate-vllm-env.sh
```

**Both work, ours is more convenient for users.**

## References

- **vLLM Repository:** https://github.com/tenstorrent/vllm
- **vLLM TT Docs:** https://github.com/tenstorrent/vllm/blob/dev/tt_metal/README.md
- **uv Documentation:** https://github.com/astral-sh/uv
- **uv Installation:** https://docs.astral.sh/uv/getting-started/installation/
- **Our Lesson:** content/lessons/vllm-production.md
- **Our Setup Script:** content/templates/setup-vllm-env.sh

## Action Items

- [ ] Install uv on development machine
- [ ] Test official installation script
- [ ] Update lesson documentation
- [ ] Update setup script with uv support
- [ ] Add uv installation command to extension
- [ ] Test on multiple hardware types
- [ ] Update CLAUDE.md with uv context
- [ ] Create PR with changes
- [ ] Get user testing feedback

---

**Status:** Draft - Pending Implementation
**Author:** Claude Code Analysis
**Last Updated:** 2026-03-04
