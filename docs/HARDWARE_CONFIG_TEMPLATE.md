# Hardware Configuration Template

This template provides a compassionate, learner-friendly format for hardware-specific instructions.

## CSS Styling

All `<details>` and `<summary>` tags include inline CSS using VSCode theme variables for consistent, theme-aware appearance:

**Details styling:**
```css
border: 1px solid var(--vscode-panel-border);
border-radius: 6px;
padding: 12px;
margin: 8px 0;
background: var(--vscode-editor-background);
```

**Summary styling:**
```css
cursor: pointer;
font-weight: bold;
padding: 4px;
margin: -12px -12px 12px -12px;
background: var(--vscode-sideBar-background);
border-radius: 4px 4px 0 0;
border-bottom: 1px solid var(--vscode-panel-border);
```

**VSCode Theme Variables Used:**
- `--vscode-panel-border` - Subtle border that adapts to theme
- `--vscode-editor-background` - Main content background
- `--vscode-sideBar-background` - Slightly different shade for headers
- Auto-adapts to light/dark themes!

---

## Standard Format

```markdown
## Configure for Your Hardware

**Quick Check:** Not sure which hardware you have? Run this command to detect your device:

[🔍 Detect Hardware](command:tenstorrent.runHardwareDetection)

Look for the "Board Type" field in the output (e.g., n150, n300, t3k, p100).

---

**Choose your hardware configuration below:**

<details open style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N150 (Wormhole - Single Chip)</b> - Most common for development</summary>

**Specifications:**
- Chips: 1
- Context Length: 64K tokens
- Best for: Development, single-user deployments, learning

**Environment Variables:**
```bash
export MESH_DEVICE=N150
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**Command:**
```bash
# [Lesson-specific command here]
cd ~/tt-vllm && \
  source ~/tt-vllm-venv/bin/activate && \
  export MESH_DEVICE=N150 && \
  [... rest of command ...]
```

[🚀 Run Command for N150](command:tenstorrent.lessonCommandN150)

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N300 (Wormhole - Dual Chip)</b></summary>

**Specifications:**
- Chips: 2
- Context Length: 128K tokens
- Best for: Higher throughput, production deployments

**Environment Variables:**
```bash
export MESH_DEVICE=N300
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**Command:**
```bash
# [Lesson-specific command here]
cd ~/tt-vllm && \
  source ~/tt-vllm-venv/bin/activate && \
  export MESH_DEVICE=N300 && \
  [... rest of command ...]
```

[🚀 Run Command for N300](command:tenstorrent.lessonCommandN300)

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 T3K (Wormhole - 8 Chips)</b></summary>

**Specifications:**
- Chips: 8
- Context Length: 128K+ tokens
- Best for: Large models (70B+), multi-user production

**Environment Variables:**
```bash
export MESH_DEVICE=T3K
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**Command:**
```bash
# [Lesson-specific command here]
cd ~/tt-vllm && \
  source ~/tt-vllm-venv/bin/activate && \
  export MESH_DEVICE=T3K && \
  [... rest of command ...]
```

[🚀 Run Command for T3K](command:tenstorrent.lessonCommandT3K)

</details>

<details style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 P100 (Blackhole - Single Chip)</b></summary>

**Specifications:**
- Chips: 1 (newer architecture)
- Context Length: 64K tokens
- Best for: Development with latest hardware

**Environment Variables:**
```bash
export MESH_DEVICE=P100
export TT_METAL_ARCH_NAME=blackhole  # Required for Blackhole
export TT_METAL_HOME=~/tt-metal
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH
```

**Command:**
```bash
# [Lesson-specific command here]
cd ~/tt-vllm && \
  source ~/tt-vllm-venv/bin/activate && \
  export MESH_DEVICE=P100 && \
  export TT_METAL_ARCH_NAME=blackhole && \
  [... rest of command ...]
```

[🚀 Run Command for P100](command:tenstorrent.lessonCommandP100)

</details>

---

**💡 Tip:** If you're unsure, start with N150 configuration - it works on most hardware, just with potentially different performance characteristics.

```

## Design Principles

1. **First section open by default** (`<details open>`) - N150 is most common
2. **Consistent emoji** - 🔧 for all hardware configs
3. **Clear hierarchy** - Summary → Specs → Env Vars → Command → Button
4. **Hardware detection at top** - Always provide a way to check
5. **Separate commands per hardware** - Each gets its own registered command
6. **Compassionate language** - "Not sure?" "Most common" "Best for"
7. **Progressive disclosure** - Collapsed by default except N150
8. **Tip at bottom** - Reassure beginners they can start with N150

## Implementation Notes

### For each lesson:
1. Identify the hardware-variant step
2. Replace existing hardware instructions with this template
3. Customize the "Command" section for that lesson's specific needs
4. Create hardware-specific commands in `terminalCommands.ts`
5. Register commands in `extension.ts`
6. Add to `package.json`

### Command naming convention:
- `tenstorrent.[lessonName][HardwareType]`
- Examples:
  - `tenstorrent.startVllmServerN150`
  - `tenstorrent.startVllmServerN300`
  - `tenstorrent.generateImageN150`
  - `tenstorrent.runTtXlaTestT3K`

### Terminal reuse:
- All commands should use existing terminal types (`main` or `server`)
- Do NOT create new terminals for hardware variants
- Example: All vLLM commands use `server` terminal type
