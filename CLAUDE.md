# CLAUDE.md (Condensed)

Guidance for Claude Code working with this Tenstorrent VSCode extension.

## Project Overview

VSCode extension for Tenstorrent hardware development:

1. **Walkthroughs** - Step-by-step guides via VSCode Walkthroughs API
2. **Device Monitoring** - Statusbar with tt-smi integration
3. **Chat Integration** - @tenstorrent participant via vLLM
4. **Templates** - Production-ready Python scripts
5. **Auto-config** - Solarized Dark + terminal on activation
6. **Lesson Metadata** - Hardware compatibility and validation tracking (see LESSON_METADATA.md)

## 🔧 Recent Multi-Device API Update (Jan 2026)

**IMPORTANT:** Multi-device TTNN code must now use `CreateDevices`/`CloseDevices` API.

**Problem:** Opening/closing devices individually causes dispatch core errors:
```python
# ❌ OLD (Broken)
for id in range(4):
    device = ttnn.open_device(device_id=id)
    devices.append(device)
for device in devices:
    ttnn.close_device(device)  # Crashes with dispatch core error
```

**Solution:** Use coordinated device management:
```python
# ✅ NEW (Required)
num_devices = ttnn.GetNumAvailableDevices()
devices = ttnn.CreateDevices(list(range(num_devices)))
try:
    # Use devices...
finally:
    ttnn.CloseDevices(devices)  # Proper cleanup
```

**Updated templates:**
- `content/templates/cookbook/particle_life/particle_life_multi_device.py`
- `content/templates/cookbook/particle_life/test_multi_device.py`

See `MULTI_DEVICE_FIX.md` for full details.

## Hardware Configuration Formatting

**v0.0.98+ (Current)**: Lesson 7 uses clean markdown headers for better walkthrough rendering:

```markdown
### N150 (Wormhole - Single Chip) - Most common for development

**✅ Recommended: Qwen3-0.6B** - Tiny, fast, reasoning-capable!

```bash
command here...
```

---
```

- Pure markdown (no HTML)
- Better multi-line code block rendering
- Cleaner, more maintainable

**v0.0.85-0.0.97 (Legacy)**: Some lessons still use CSS-styled `<details>` sections:

```html
<details open style="border: 1px solid var(--vscode-panel-border); ...">
<summary><b>🔧 Hardware Name</b></summary>
Content...
</details>
```

- Used in Lessons 6, 9, 12 (not yet migrated)
- See `HARDWARE_CONFIG_TEMPLATE.md` + `STYLING_GUIDE.md`

## Build Commands

```bash
npm install           # Install dependencies
npm run build         # Compile TS → dist/
npm run watch         # Auto-recompile on changes
npm run package       # Create .vsix
```

## ⚠️ NO BUNDLING (esbuild/webpack)

**CRITICAL:** This extension CANNOT be bundled with esbuild or webpack.

**Attempts made:**
- v0.0.125-126 (2025-01-06): First bundling attempt with esbuild
- v0.0.226 (2025-01-07): Second bundling attempt with esbuild + jsdom external

**Why bundling fails:**
1. **Tree data providers break** - "no data provider registered" errors
2. **View registration fails** - Sidebar and toolbar additions disappear
3. **Module resolution issues** - Dynamic imports and class exports break
4. **jsdom file dependencies** - Even when marked external, causes runtime errors

**Result:** Both attempts rolled back. Extension works correctly with full node_modules (~60MB, 2031 files).

**DO NOT attempt bundling again.** The extension architecture is incompatible with bundlers.

## Version Management

**⚠️ CRITICAL: ALWAYS increment version in `package.json` after ANY changes:**
- **Bug fixes:** increment PATCH (0.0.224 → 0.0.225)
- **New features:** increment MINOR (0.0.36 → 0.0.37)
- **Breaking changes:** increment MAJOR
- **Content changes:** increment PATCH (markdown edits, templates, etc.)

**Why this matters:**
- VSCode extension caching causes issues without version changes
- Users may see stale content/functionality if version doesn't increment
- Even small changes (single line fixes) require version bump
- **Rule:** After completing ANY bugfix, content change, or series of alterations → increment version → rebuild → repackage

## Testing

Press `F5` to launch Extension Development Host.

**Manually open walkthrough:**
1. Cmd+Shift+P → "Welcome: Open Walkthrough"
2. Select "Get Started with Tenstorrent"

Or run: "Tenstorrent: Show Welcome Page"

## File Structure

```
content/
  lessons/        # Markdown content (editable by writers)
  templates/      # Python script templates
  welcome/        # Welcome page HTML
src/
  extension.ts    # Main extension code
  commands/terminalCommands.ts  # Command definitions
vendor/           # Reference repos (NOT deployed with extension)
  tt-metal/       # Main tt-metal repo - demos, examples, APIs
  vllm/           # TT vLLM fork - production inference
  tt-xla/         # TT-XLA/JAX - compiler, examples
  tt-forge-fe/    # TT-Forge frontend - experimental compiler
  tt-inference-server/  # Production deployment automation
  tt-installer/   # Installation automation reference
  ttsim/          # Simulator reference
package.json      # Extension manifest + walkthrough definitions
```

**Generated:** `~/tt-scratchpad/` - Extension-created scripts

**⚠️ Important:** The `vendor/` directory contains reference repositories for lesson authoring:
- **tt-metal** - Primary reference: demos, APIs, examples, model implementations
- **vllm** - Production inference patterns, server examples
- **tt-xla** - JAX/TT-XLA examples, demos, compiler documentation
- **tt-forge-fe** - TT-Forge examples, experimental compiler reference
- **tt-inference-server** - Production deployment automation, MODEL_SPECS
- **tt-installer** - Installation workflows, setup patterns
- **ttsim** - Simulator reference for testing without hardware

These repos are **NOT deployed** with the extension - they're local references only for development and lesson authoring. Always verify commands, paths, and API examples against these repos before publishing lessons.

**🚀 CRITICAL FOR CLAUDE CODE:** When working on lessons or features, **liberally clone/checkout packages to `vendor/` as needed**. Don't work blind - get the actual reference implementation first:

```bash
# Clone new reference repo when needed
cd vendor/
git clone https://github.com/tenstorrent/[repo-name].git

# Update existing repos
cd vendor/[repo-name]
git pull origin main
```

**Examples when you should clone/update vendor repos:**
- Authoring a new lesson about a feature
- Updating commands or API examples in existing lessons
- Verifying hardware configurations or flags
- Checking model paths, formats, or implementations
- Confirming environment variable names
- Finding correct import paths or function signatures

**Don't guess - check the source!** The vendor directory exists specifically so you can reference actual implementations.

**Note:** `vendor/` is in `.gitignore` - these reference repos are NOT committed to the extension's git repository. They're local-only for development. Each developer/AI should clone what they need.

## Lesson Metadata System (v0.0.86+)

**Every lesson now has metadata for hardware compatibility and validation tracking.**

See `LESSON_METADATA.md` for complete documentation.

**Quick reference:**
```json
"metadata": {
  "supportedHardware": ["n150", "n300", "t3k", "p100", "p150", "galaxy"],
  "status": "validated" | "draft" | "blocked",
  "validatedOn": ["n150", "n300"],
  "blockReason": "Optional reason if blocked",
  "minTTMetalVersion": "v0.51.0"
}
```

**Hardware values:** `n150`, `n300`, `t3k`, `p100`, `p150`, `galaxy`, `simulator`

**Status values:**
- `validated` - Tested and ready for production release
- `draft` - In development, hide in production builds
- `blocked` - Known issue, show with warning

**Use cases:**
1. **Release gating** - Filter lessons by status before packaging
2. **Hardware filtering** - Show only relevant lessons for detected hardware
3. **Quality tracking** - Know which configs have been tested
4. **Development workflow** - Clear status for each lesson

**All 16 lessons have metadata as of v0.0.86.**

## Architecture

**Content-First Design:** Content in markdown, code handles execution only.

**Walkthrough Structure:** Defined in `package.json` → `contributes.walkthroughs`
- Steps auto-complete via `completionEvents`
- Markdown rendered natively by VSCode
- Command links as buttons (on own line)
- Each step now includes `metadata` field (v0.0.86+)

**Terminal Management (v0.0.66+):**
- **2 terminals only:** `main` (setup/testing) and `server` (long-running)
- Reuse existing terminals (no terminal clutter)
- Environment persists across lessons

**Device Detection:** `updateDeviceStatus()` parses tt-smi, caches device info

## Adding New Lessons

1. **Research:** Check `vendor/` directory for reference implementations:
   - `vendor/tt-metal/` - Demos, examples, API patterns, model implementations
   - `vendor/vllm/` - Production inference, server configurations
   - `vendor/tt-xla/` - JAX examples, PJRT integration, demos
   - `vendor/tt-forge-fe/` - TT-Forge examples, experimental models
   - `vendor/tt-inference-server/` - MODEL_SPECS, validated configs, workflows
   - `vendor/tt-installer/` - Installation workflows, setup automation
   - `vendor/ttsim/` - Simulator for testing without hardware

   **If repo missing or outdated:** Clone/update it! Don't work without references:
   ```bash
   cd vendor/
   git clone https://github.com/tenstorrent/[repo-name].git
   # or update existing:
   cd vendor/[repo-name] && git pull origin main
   ```

2. Create `content/lessons/XX-your-lesson.md`
3. Add to `package.json` → `contributes.walkthroughs[0].steps`
4. Define commands needed
5. Implement handlers in `src/extension.ts`
6. Register commands in `activate()`

**For hardware-specific lessons:** Use styled `<details>` pattern from template.

**Best practice:** Always verify commands, paths, and examples against the vendor repos before publishing. They're cloned specifically for this purpose. **Clone liberally - don't guess!**

## Critical Patterns

**tt-metal builds:**
```bash
./install_dependencies.sh  # ALWAYS run first
./build_metal.sh --clean   # Troubleshooting
./build_metal.sh --enable-ccache  # Fast rebuilds
```

**vLLM commands (Lesson 7):**
- Hardware-specific Llama: `startVllmServerN150/N300/T3K/P100()`
- Hardware-specific Qwen: `startVllmServerN150Qwen/N300Qwen/T3KQwen/P100Qwen()` (v0.0.89+)
- Helper: `startVllmServerForHardware(hardware, config)` - accepts optional `modelPath` parameter
- All use `'server'` terminal type

**Model Support (updated v0.0.97):**
- **Qwen3-0.6B** - Ultra-lightweight (0.6B params), dual thinking modes, reasoning excellence ✅ **PRIMARY RECOMMENDATION for N150**
  - MMLU-Redux: 55.6, MATH-500: 77.6 (impressive for 0.6B!)
  - Sub-millisecond inference, 10,000+ QPS capable
  - Multilingual, 32K context
  - **Perfect for development and many production use cases**
- **Gemma 3-1B-IT** - Small (1B params), multilingual (140+ langs), 32K context ✅ **Good for N150**
- **Llama-3.1-8B-Instruct** - General-purpose chat (8B params, gated) ⚠️ **Requires N300/T3K/P100**
- **Qwen3-8B** - Multilingual coding/math (8B params) ⚠️ **Requires N300+ for reliable operation**

**🔑 HF_MODEL Auto-Detection (v0.0.97):**
- `start-vllm-server.py` now auto-detects and sets `HF_MODEL` from `--model` path
- Qwen models: `HF_MODEL=Qwen/{model_name}` (e.g., `Qwen/Qwen3-0.6B`)
- Gemma models: `HF_MODEL=google/{model_name}` (e.g., `google/gemma-3-1b-it`)
- Llama models: No HF_MODEL needed (auto-detects correctly)
- **Users no longer need to manually export HF_MODEL** - script handles it automatically!

**⚠️ N150 DRAM Reality:**
- Llama-3.1-8B-Instruct consistently exhausts DRAM on N150
- **Solution**: Start with Qwen3-0.6B (13x smaller, reasoning-capable, production-ready)
- Lesson 7 completely rewritten around Qwen3-0.6B as the hero model (v0.0.97)

**Symlink Workaround Technical Details (v0.0.92):**
```typescript
// Helper function creates symlink if needed
async function createQwenSymlink(qwenPath: string): Promise<string> {
  // Target: ~/models/Llama-3.1-8B-Instruct-qwen -> ~/models/Qwen3-8B
  // Path contains expected string, points to actual Qwen model
  // Checks if symlink already exists and points to correct location
  // Returns symlink path to use with vLLM
}

// All 4 Qwen handlers now:
// 1. Show informational dialog about symlink
// 2. Call createQwenSymlink() to create/verify symlink
// 3. Pass symlink path to startVllmServerForHardware()
// 4. vLLM's path check passes, model loads successfully
```

**User Experience:**
- Click Qwen command → See explanation dialog → Click "Start Server"
- Extension creates symlink (shows confirmation)
- vLLM starts with symlink path
- Everything works transparently
- Symlink persists for future use

**Model Registry:** `MODEL_REGISTRY` in `src/extension.ts`
- Current default: Llama-3.1-8B-Instruct
- Add models here to make available throughout extension

## Key Implementation Notes

- **No custom UI** - All UI from VSCode native
- **Markdown deployed** - `content/` copied to `dist/`
- **Terminal persistence** - Survives between invocations
- **Password input** - `password: true` in `showInputBox()`
- **Completion tracking** - VSCode auto-tracks via `completionEvents`

## Lessons Summary

| Lesson | Focus | Hardware Variants |
|--------|-------|-------------------|
| 1-5 | Setup, Direct API | Generic |
| 6-7 | Production (tt-inference-server, vLLM) | ✅ N150/N300/T3K/P100 |
| 8 | VSCode Chat | Generic |
| 9 | Image Generation (SD 3.5) | ✅ N150/N300/T3K/P100 |
| 10 | Coding Assistant | Generic |
| 11 | TT-Forge (experimental) | N150 only |
| 12 | TT-XLA JAX | ✅ N150/N300/T3K/Galaxy |

## Troubleshooting

**Environment variables matter:**
- vLLM: `TT_METAL_HOME`, `MESH_DEVICE`, `PYTHONPATH`
- Blackhole (P100): Also needs `TT_METAL_ARCH_NAME=blackhole`
- TT-Forge: `unset TT_METAL_HOME TT_METAL_VERSION`

**Model paths:**
- HuggingFace format: `~/models/Llama-3.1-8B-Instruct`
- Meta format: `~/models/Llama-3.1-8B-Instruct/original`

## Documentation Files

- `CLAUDE.md` - Full details (this file)
- `HARDWARE_CONFIG_TEMPLATE.md` - Pattern for hardware configs
- `STYLING_GUIDE.md` - CSS styling reference
- `FAQ.md` - User-facing troubleshooting
- `README.md` - Public-facing documentation

## Vendor Directory Reference Guide

**When authoring lessons, check these repos:**

| Lesson Type | Primary Reference | Secondary References |
|-------------|-------------------|---------------------|
| Setup/Installation | `tt-installer/` | `tt-metal/` |
| Direct API (tt-metal) | `tt-metal/models/` | `tt-metal/demos/` |
| vLLM Production | `vllm/tt_metal/` | `tt-inference-server/` |
| tt-inference-server | `tt-inference-server/` | `vllm/` |
| Image Generation | `tt-metal/models/experimental/` | - |
| TT-Forge | `tt-forge-fe/` | `tt-metal/` |
| TT-XLA/JAX | `tt-xla/demos/` | `tt-xla/` |
| Simulator Testing | `ttsim/` | `tt-metal/` |

**Always verify:**
- Command syntax and flags
- Model paths and formats
- Environment variables
- Hardware configurations
- API examples and patterns

## Changelog Policy

**⚠️ IMPORTANT: As of v0.0.268, changelog management has been standardized:**

### Where to Find Version History

1. **CHANGELOG.md** - Complete version history in Keep a Changelog format
   - **All releases** documented here with full details
   - Follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) specification
   - Adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
   - **This is the single source of truth for version history**

2. **README.md** - User-facing highlights only
   - **Most recent 2 releases** with brief highlights
   - Links to CHANGELOG.md for complete history
   - Designed for quick overview of latest features

3. **CLAUDE.md** - No version history
   - This file (CLAUDE.md) contains **no changelog entries**
   - Documents project structure, workflows, and development guidance
   - References CHANGELOG.md for version history

### Version Management Workflow

**⚠️ CRITICAL: ALWAYS increment version in `package.json` after ANY changes:**
- **Bug fixes:** increment PATCH (0.0.268 → 0.0.269)
- **New features:** increment MINOR (0.0.268 → 0.1.0)
- **Breaking changes:** increment MAJOR (0.0.268 → 1.0.0)
- **Content changes:** increment PATCH (markdown edits, templates, etc.)

**After making changes:**
1. Increment version in `package.json`
2. Add entry to `CHANGELOG.md` with proper categorization (Added/Changed/Fixed/Removed)
3. If this is one of the 2 most recent releases, update `README.md` highlights
4. Rebuild and repackage extension
5. Test installation to verify no caching issues

**Why this matters:**
- VSCode extension caching causes issues without version changes
- Users may see stale content/functionality if version doesn't increment
- Even small changes (single line fixes) require version bump
- **Rule:** After completing ANY bugfix, content change, or series of alterations → increment version → rebuild → repackage

### Historical Note

Prior to v0.0.268, this file contained extensive changelog entries spanning 500+ lines. This has been consolidated to improve maintainability and reduce duplication. All historical version information remains available in `CHANGELOG.md`.

**See [CHANGELOG.md](CHANGELOG.md) for complete version history.**
