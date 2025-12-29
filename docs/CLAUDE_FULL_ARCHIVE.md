# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VS Code extension for Tenstorrent hardware setup and development:

1. **Interactive Walkthroughs** - Step-by-step guides using VSCode's native Walkthroughs API
2. **Device Status Monitoring** - Real-time statusbar with tt-smi integration
3. **VSCode Chat Integration** - @tenstorrent chat participant via local vLLM
4. **Template Scripts** - Production-ready Python scripts
5. **Auto-configured UX** - Solarized Dark theme + terminal on first activation

## Styled Hardware Configuration Sections (v0.0.85+)

All hardware-specific instructions use CSS-styled `<details>` sections for clean, collapsible UI:

```html
<details open style="border: 1px solid var(--vscode-panel-border); border-radius: 6px; padding: 12px; margin: 8px 0; background: var(--vscode-editor-background);">
<summary style="cursor: pointer; font-weight: bold; padding: 4px; margin: -12px -12px 12px -12px; background: var(--vscode-sideBar-background); border-radius: 4px 4px 0 0; border-bottom: 1px solid var(--vscode-panel-border);"><b>🔧 N150 (Wormhole - Single Chip)</b></summary>

**Specifications:** ...
**Environment Variables:** ...
**Command:** ...

</details>
```

**Key points:**
- Uses VSCode CSS variables (auto-adapts to light/dark themes)
- N150 open by default (most common hardware)
- Consistent across Lessons 6, 7, 9, 12
- See `HARDWARE_CONFIG_TEMPLATE.md` for full pattern
- See `STYLING_GUIDE.md` for visual preview and CSS details

## Setup Guide: tt-installer 2.0 (Optional/Informational)

**Status:** Implemented (v0.0.81) - Informational guide, not a required walkthrough step

### Overview

tt-installer 2.0 is Tenstorrent's official one-command installation tool. **Note:** Many users (Tenstorrent Cloud, Quietbox with preinstalled images, managed systems) already have this setup complete and can skip directly to the Learning Path.

For users who need to set up from scratch, tt-installer:

- ✅ **One command setup** - Complete stack installation in 5-15 minutes
- ✅ **Container-based** - tt-metalium runs in Podman containers (no complex builds)
- ✅ **Full automation** - Kernel drivers, HugePages, firmware, tt-smi, tools
- ✅ **Production-ready** - Tested across Ubuntu, Debian, Fedora
- ✅ **Two container options:**
  - Standard (1GB) - TTNN library for inference
  - Model Demos (10GB) - Full tt-metal with demos and examples

### Quick Start

```bash
# One-command installation (interactive)
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"

# Or download and customize
curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh -O
chmod +x install.sh
./install.sh
```

### What Gets Installed

1. **System packages** - Build tools, dependencies
2. **Python environment** - Virtual environment (default: `~/.tenstorrent-venv`)
3. **Kernel-Mode Driver (KMD)** - Tenstorrent hardware driver
4. **tt-flash** - Firmware updater (auto-updates device firmware)
5. **HugePages** - Kernel memory configuration
6. **tt-smi** - System management interface
7. **Podman** - Container runtime
8. **tt-metalium containers** - Standard (1GB) and/or Model Demos (10GB)
9. **tt-inference-server** - Production inference serving
10. **SFPI** - Scalar floating point interface

### Using tt-metalium Containers

```bash
# Interactive shell
tt-metalium

# Run commands directly
tt-metalium "python3 -c 'import ttnn; print(ttnn.__version__)'"

# Run demos (if Model Demos container installed)
tt-metalium-models
cd tt-metal/models/demos
pytest wormhole/llama31_8b/demo/demo.py
```

**Key benefit:** Your home directory is mounted inside containers - all your files are accessible!

### Non-Interactive Installation

For automated deployments or cloud environments:

```bash
./install.sh --mode-non-interactive \
  --python-choice=new-venv \
  --install-metalium-models-container=off \
  --reboot-option=never
```

### Container Mode

When running inside a container (e.g., Docker):

```bash
./install.sh --mode-container
```

This automatically skips KMD, HugePages, Podman installation (must live on host).

### Comparison: tt-installer vs Manual Setup

| Feature | tt-installer 2.0 | Manual Setup (Old Lessons 1-2) |
|---------|------------------|--------------------------------|
| **Time** | 5-15 minutes | 1-2 hours |
| **Complexity** | One command | 15+ manual steps |
| **Kernel driver** | ✅ Automatic | ❌ Manual DKMS configuration |
| **Firmware** | ✅ Auto-updated | ❌ Manual tt-flash usage |
| **HugePages** | ✅ Auto-configured | ❌ Manual sysctl.conf editing |
| **tt-metalium** | ✅ Container (1GB) | ❌ Build from source (20+ min) |
| **Updates** | ✅ Re-run installer | ❌ Rebuild everything |
| **Production** | ✅ Ready | ❌ Requires hardening |

### Who Should Use tt-installer?

**Use tt-installer if you have:**
- ✅ Fresh Ubuntu/Debian/Fedora system without Tenstorrent software
- ✅ Personal workstation or server you're setting up yourself
- ✅ Development environment that needs tt-metalium containers
- ✅ System where you have sudo access and want one-command setup

**Skip tt-installer if you have:**
- ⏭️ **Tenstorrent Cloud environment** (pre-configured)
- ⏭️ **Quietbox with preinstalled image** (drivers/tools already installed)
- ⏭️ **Managed system** (sysadmin already set up)
- ⏭️ **Container environment** (host already configured)

### When to Use Manual Setup Instead

Use manual setup only if:

- You need bleeding-edge unreleased features (build from main)
- You're developing tt-metal itself (need source access)
- Your OS is unsupported by tt-installer
- You need custom compiler flags or build options

### Implementation Details

**Files created:**
- `content/lessons/00-tt-installer.md` - Comprehensive lesson (600+ lines)
- Commands in `src/commands/terminalCommands.ts`:
  - `QUICK_INSTALL` - One-command installation
  - `DOWNLOAD_INSTALLER` - Download script for inspection
  - `RUN_INTERACTIVE_INSTALL` - Interactive mode with prompts
  - `RUN_NON_INTERACTIVE_INSTALL` - Automated mode
  - `TEST_METALIUM_CONTAINER` - Verify tt-metalium works

**Command handlers in `src/extension.ts`:**
- `runQuickInstall()` - Runs quick install (with warning)
- `downloadInstaller()` - Downloads install.sh to ~/
- `runInteractiveInstall()` - Runs with prompts
- `runNonInteractiveInstall()` - Runs with defaults
- `testMetaliumContainer()` - Tests TTNN import

**Positioning in Extension:**
- **Not a walkthrough step** - Informational guide only
- Referenced in Lesson 1 (Hardware Detection) for users who need setup
- Displayed in Welcome page under "Setup Information (Optional)"
- Commands available via Command Palette for users who need them
- Cloud/Quietbox users can skip directly to Learning Path

**Resources:**
- GitHub: https://github.com/tenstorrent/tt-installer
- Wiki: https://github.com/tenstorrent/tt-installer/wiki
- Included in extension: `vendor/tt-installer/` (for reference)

### Support

**Supported Operating Systems:**
- Ubuntu 24.04 LTS ✅ (Recommended)
- Ubuntu 22.04 LTS ✅ (Recommended, most tested)
- Ubuntu 20.04 LTS ⚠️ (Deprecated, Metalium cannot be installed)
- Debian 12.10.0 ✅
- Fedora 41-42 ✅
- Other DEB/RPM distros ⚠️ (May work, unsupported)

**Recommended:** Ubuntu 22.04.5 LTS for best compatibility.

## Critical Best Practice: install_dependencies.sh (Manual Setup)

**⚠️ Always run `install_dependencies.sh` when manually installing or updating tt-metal.**

**Note:** If you used tt-installer 2.0, this is already done for you!

```bash
cd ~/tt-metal
./install_dependencies.sh  # Install system dependencies
./build_metal.sh           # Then build tt-metal
```

**Why this matters:**
- Installs required system libraries (build tools, kernel modules, device drivers)
- Prevents common build failures due to missing dependencies
- Only takes 2-5 minutes but saves hours of debugging
- Should be run after system updates or fresh installations

**When to run it:**
- First time installing tt-metal
- After `git pull` on tt-metal (before rebuilding)
- After system updates
- When encountering build errors
- When setting up vLLM or other tt-metal-dependent projects

## Advanced Build Options

### Essential Troubleshooting Flag

**`--clean` - Remove all build artifacts**
```bash
./build_metal.sh --clean
```
- Removes all cached builds and intermediate files
- Forces complete rebuild from scratch
- **Critical for troubleshooting build issues**
- Run this before rebuilding if you encounter strange errors

### Development Optimization Flags

**`--enable-ccache` - Speed up rebuilds**
```bash
./build_metal.sh --enable-ccache
```
- Caches compilation results
- Makes subsequent builds **much faster** (often 10x or more)
- Essential for active development and iteration
- Recommended for users working on Lesson 11 (Forge) or custom kernel development

**Build type flags:**
```bash
./build_metal.sh --release         # Production build (default, optimized)
./build_metal.sh --development     # Debug + optimizations (RelWithDebInfo)
./build_metal.sh --debug           # Debug symbols, no optimizations
```
- `--release`: Best performance, minimal debug info (default)
- `--development`: Balanced - some debug info with optimizations
- `--debug`: Full debug symbols, easier debugging, slower execution

### Educational/Example Flags

**`--build-programming-examples` - Build tt-metal examples**
```bash
./build_metal.sh --build-programming-examples
```
- Builds the programming examples from tt-metal repository
- Useful for "Exploring TT-Metalium" lesson
- Provides hands-on code examples for learning

### Combining Flags

You can combine multiple flags:
```bash
# Fast development builds with examples
./build_metal.sh --enable-ccache --development --build-programming-examples

# Clean rebuild with ccache
./build_metal.sh --clean
./build_metal.sh --enable-ccache
```

### When to Use Each Flag

| Flag | When to Use |
|------|-------------|
| `--clean` | Troubleshooting build errors, after major updates |
| `--enable-ccache` | Active development, frequent rebuilds |
| `--release` | Production deployment, benchmarking |
| `--development` | Debugging while keeping reasonable performance |
| `--debug` | Deep debugging of tt-metal itself |
| `--build-programming-examples` | Learning tt-metal programming patterns |

## Build and Development Commands

```bash
# Install dependencies
npm install

# Compile TypeScript to dist/
npm run build

# Watch mode for development (auto-recompile on changes)
npm run watch

# Package extension as .vsix file
npm run package
```

## Version Management

**IMPORTANT:** Always increment the version number in `package.json` with every package of changes.

- Use semantic versioning: `MAJOR.MINOR.PATCH`
- For new features/lessons: increment MINOR (e.g., `0.0.36` → `0.0.37`)
- For bug fixes: increment PATCH (e.g., `0.0.36` → `0.0.37`)
- For breaking changes: increment MAJOR (when ready for 1.0.0)

Example workflow:
1. Make changes (add lesson, fix bug, etc.)
2. Update version in `package.json`
3. Build and test
4. Commit with version number in message

## Running and Testing

Press `F5` in VS Code to launch the Extension Development Host with the extension loaded. The walkthrough will appear in the "Get Started" section of VS Code.

To manually open the walkthrough:
1. Open Command Palette (`Cmd+Shift+P` or `Ctrl+Shift+P`)
2. Search for "Welcome: Open Walkthrough"
3. Select "Get Started with Tenstorrent"

Alternatively, open the welcome page:
1. Open Command Palette
2. Run "Tenstorrent: Show Welcome Page"
3. The welcome page provides an overview and links to all walkthroughs

## Working Directory Structure

The extension creates a dedicated directory for all generated scripts and files:

**Scratchpad Directory:** `~/tt-scratchpad/`
- All Python scripts created from templates are saved here
- Examples: `tt-chat-direct.py`, `tt-api-server-direct.py`
- Keeps user workspace organized and scripts easy to find
- Directory is automatically created when first script is generated

**Benefits:**
- ✅ Clear separation between extension-generated code and user projects
- ✅ All scripts in one predictable location
- ✅ Easy to customize, backup, or delete generated files
- ✅ No clutter in home directory root

## First-Time User Experience (v0.0.65+)

The extension automatically configures an optimal development environment on first activation:

### Auto-Configured Settings

**1. Terminal Auto-Open**
- Creates and opens a "Tenstorrent" terminal immediately
- Terminal is ready for command execution
- Uses `preserveFocus=true` to keep editor focus
- Terminal persists across the session

**2. Solarized Dark Theme**
- Automatically sets `workbench.colorTheme` to "Solarized Dark"
- Only applies if user is on default theme (respects existing preferences)
- Optimal contrast for terminal visibility
- Professional, eye-friendly color scheme

**3. Recommended Extensions Prompt (v0.0.67+)**
- **Non-intrusive notification** prompts user to install recommended extensions
- User choices: "Install All", "Not Now", or "Show Details"
- Only shows extensions that aren't already installed
- Installs in background without blocking workflow
- Recommended extensions:
  - **Python** - Python language support
  - **Pylance** - Fast Python IntelliSense
  - **Jupyter** - Notebook support
  - **C/C++** - For tt-metal C++ development
  - **CMake Tools** - Build system integration

**4. Welcome Page**
- Opens automatically on first activation
- Provides overview of all walkthrough lessons
- Links to documentation and resources

### Implementation Details

**Location:** `src/extension.ts` (lines 3177-3206)

The auto-configuration happens during the `activate()` function:

```typescript
// Create and show terminal
const defaultTerminal = vscode.window.createTerminal({
  name: 'Tenstorrent',
  cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
});
defaultTerminal.show(true); // preserveFocus=true

// Set theme on first run only
const hasSeenWelcome = context.globalState.get<boolean>('hasSeenWelcome', false);
if (!hasSeenWelcome) {
  context.globalState.update('hasSeenWelcome', true);

  const config = vscode.workspace.getConfiguration();
  const currentTheme = config.get<string>('workbench.colorTheme');

  // Only set if still on default theme
  if (currentTheme === 'Default Dark Modern' || currentTheme === 'Default Light Modern' || !currentTheme) {
    config.update('workbench.colorTheme', 'Solarized Dark', vscode.ConfigurationTarget.Global);
  }

  // Show welcome page
  setTimeout(() => showWelcome(context), 1000);
}
```

**Key Design Decisions:**
- ✅ **Respects user preferences:** Only sets theme if user is on a default theme
- ✅ **One-time configuration:** Uses `hasSeenWelcome` flag to avoid repeated changes
- ✅ **Non-intrusive terminal:** Opens with `preserveFocus=true` to not disrupt workflow
- ✅ **Global settings:** Uses `ConfigurationTarget.Global` so theme persists across workspaces
- ✅ **User choice for extensions (v0.0.67+):** Prompts instead of auto-installing, respects user control

**Extension Installation Strategy (v0.0.67+):**

The extension uses a **prompt-based approach** instead of `extensionPack` to give users control:

```typescript
async function promptRecommendedExtensions(): Promise<void> {
  const recommendedExtensions = [
    { id: 'ms-python.python', name: 'Python' },
    { id: 'ms-python.vscode-pylance', name: 'Pylance' },
    { id: 'ms-toolsai.jupyter', name: 'Jupyter' },
    { id: 'ms-vscode.cpptools', name: 'C/C++' },
    { id: 'ms-vscode.cmake-tools', name: 'CMake Tools' },
  ];

  // Check which are missing
  const missingExtensions = recommendedExtensions.filter(
    (ext) => !vscode.extensions.getExtension(ext.id)
  );

  if (missingExtensions.length === 0) return;

  // Show non-intrusive notification
  const choice = await vscode.window.showInformationMessage(
    `Install recommended extensions? (${extensionNames})`,
    'Install All',
    'Not Now',
    'Show Details'
  );

  if (choice === 'Install All') {
    // Install in background
    for (const ext of missingExtensions) {
      await vscode.commands.executeCommand('workbench.extensions.installExtension', ext.id);
    }
  }
}
```

**Why this approach:**
- ❌ **Old:** `extensionPack` auto-installs all extensions (no user control)
- ✅ **New:** User prompt with clear options (Install All / Not Now / Show Details)
- ✅ **Background installation:** Non-blocking, doesn't interrupt workflow
- ✅ **Smart detection:** Only prompts for missing extensions
- ✅ **One-time:** Only shows on first activation

**For code-server/cloud deployments:**
This configuration works automatically when the extension is installed. No Docker image modifications needed - the extension handles everything at runtime.

## Terminal Management (v0.0.66+)

### Two-Terminal Strategy

The extension uses a **unified two-terminal approach** to prevent terminal clutter and preserve environment state across lessons:

**1. "Tenstorrent" (main terminal)**
- Used for: All setup, testing, one-off commands
- Examples: Hardware detection, downloads, installs, tests, script creation
- **Environment persists:** Python venvs, exported variables stay active across lessons
- Users can incrementally build their environment

**2. "Tenstorrent Server" (server terminal)**
- Used for: Long-running interactive processes
- Examples: vLLM servers, API servers, chat sessions, image generation, coding assistant
- **Keeps main terminal free:** Users can test/explore while servers run
- Can be stopped with Ctrl+C without affecting main environment

### Benefits

✅ **Environment persistence** - No need to re-export vars or re-activate venvs when switching lessons
✅ **Clean UI** - Maximum 2 terminal tabs (vs 9 with old approach)
✅ **Realistic workflow** - Matches how developers actually work
✅ **Non-blocking** - Servers don't block the main terminal
✅ **Simple mental model** - Easy to predict which terminal will be used

### Implementation

**Location:** `src/extension.ts` (lines 512-564)

```typescript
/**
 * Stores references to terminals used by the extension.
 */
const terminals = {
  main: undefined as vscode.Terminal | undefined,      // "Tenstorrent"
  server: undefined as vscode.Terminal | undefined,    // "Tenstorrent Server"
};

type TerminalType = 'main' | 'server';

function getOrCreateTerminal(type: TerminalType = 'main'): vscode.Terminal {
  // Check if terminal still exists
  if (terminals[type] && vscode.window.terminals.includes(terminals[type]!)) {
    return terminals[type]!;
  }

  // Create new terminal with appropriate name
  const name = type === 'server' ? 'Tenstorrent Server' : 'Tenstorrent';
  const terminal = vscode.window.createTerminal({
    name,
    cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
  });

  terminals[type] = terminal;
  return terminal;
}
```

### Routing Rules

Commands are automatically routed to the appropriate terminal:

**Main Terminal (`'main'`):**
- Hardware detection (`tt-smi`)
- Installation verification
- Model downloads
- Package installations
- Environment setup
- Testing commands (curl, pytest one-offs)
- Script creation/copying
- Jukebox operations
- Forge builds and tests

**Server Terminal (`'server'`):**
- Interactive chat sessions (`startChatSession*`)
- API servers (`startApiServer*`)
- vLLM servers (`startVllmServer`, `startVllmForChat`)
- Image generation (`startInteractiveImageGen`)
- Coding assistant (`startCodingAssistant`)
- Forge classifier (interactive mode)

### Migration from v0.0.65

**Old approach:** 9 different terminal types
- `hardwareDetection`, `verifyInstallation`, `modelDownload`, `interactiveChat`, `apiServer`, `vllmServer`, `imageGeneration`, `forgeInstall`, `forgeClassifier`

**New approach:** 2 terminal types
- `main`, `server`

**User impact:**
- Environment variables and Python venvs now persist across lessons
- Less terminal clutter (2 tabs max vs 9)
- Easier to follow lesson progression without losing context

## Architecture

### Content-First Design

**Key Principle:** Content is completely separated from code. Technical writers can edit lesson content without touching TypeScript files.

**Content Location:** `content/lessons/*.md`
- `01-hardware-detection.md` - Hardware detection lesson content
- `02-verify-installation.md` - Installation verification lesson content
- `03-download-model.md` - Model download lesson content

**Content Format:** Standard markdown with support for:
- Formatting: `**bold**`, `*italic*`, `` `code` ``
- Links to external URLs: `[text](https://url)`
- Command links: `[Run Command](command:tenstorrent.commandId)`
- Lists, headings, and other standard markdown features

**Command Links as Buttons:** By VS Code convention, command links on their own line are automatically rendered as buttons. No custom CSS or HTML is needed:

```markdown
[🔍 Detect Tenstorrent Hardware](command:tenstorrent.runHardwareDetection)
```

This native approach:
- Automatically matches the user's VS Code theme
- Requires no custom styling or CSS
- Follows VS Code extension best practices
- Keeps markdown files clean and simple for technical writers

### Walkthrough Structure

**Definition:** `package.json` → `contributes.walkthroughs`

This declarative structure defines:
- Walkthrough ID, title, and description
- Ordered list of steps
- Each step's title, description, and markdown content file
- Completion events (what triggers step completion)

**Scroll Behavior Workaround (v0.0.68):**

VS Code's native walkthrough API doesn't provide control over scroll position. When navigating between walkthrough steps, the scroll position can persist from the previous step, creating a confusing UX.

**Solution:** Force-reopen the walkthrough when navigating from the Welcome page:

**Location:** `src/extension.ts` (lines 2529-2549)

```typescript
case 'openWalkthrough':
  // Workaround: Close active editor if it's a walkthrough, then reopen
  // This forces scroll-to-top behavior
  const activeEditor = vscode.window.activeTextEditor;
  if (!activeEditor || vscode.window.tabGroups.activeTabGroup.activeTab?.label.includes('Tenstorrent')) {
    await vscode.commands.executeCommand('workbench.action.closeActiveEditor');
  }

  // Small delay to ensure close completes
  setTimeout(() => {
    vscode.commands.executeCommand(
      'workbench.action.openWalkthrough',
      {
        category: 'tenstorrent.tenstorrent-developer-extension#tenstorrent.setup',
        step: message.stepId
      }
    );
  }, 100);
  break;
```

**Why this works:**
- Detects if a Tenstorrent walkthrough/welcome tab is currently active
- Closes it before opening the new step
- 100ms delay ensures close operation completes
- Fresh walkthrough view always starts at top
- Only closes Tenstorrent tabs (preserves user's code files)

**Limitation:** This workaround only applies when navigating from the Welcome page. Direct navigation within the walkthrough UI still has the scroll issue (VS Code limitation).

**Adding a New Lesson:**
1. Create a new markdown file in `content/lessons/`
2. Add a new step to `package.json` → `contributes.walkthroughs[0].steps`
3. Define the command(s) needed for that step
4. Implement the command handler(s) in `src/extension.ts`

### Extension Code Structure

**src/extension.ts** - Single file containing all extension logic (~217 lines)

**Sections:**
1. **Terminal Management** (lines 20-71)
   - `terminals` object: Stores references to named terminals
   - `getOrCreateTerminal()`: Gets or creates a terminal, reuses if still alive
   - `runInTerminal()`: Executes commands and shows terminal to user

2. **Command Handlers** (lines 73-167)
   - `runHardwareDetection()`: Executes `tt-smi` command
   - `verifyInstallation()`: Runs tt-metal test program
   - `setHuggingFaceToken()`: Prompts for HF token, sets as env var
   - `loginHuggingFace()`: Authenticates with HF CLI
   - `downloadModel()`: Downloads Llama model from HF

3. **Extension Lifecycle** (lines 169-217)
   - `activate()`: Registers all commands
   - `deactivate()`: Cleans up terminal references

**Key Design Decisions:**
- No state management needed (VSCode tracks completion)
- No webview/HTML generation (uses native walkthrough UI)
- No custom UI code (leverages VSCode theming)
- Commands are simple: create terminal → send command → show feedback

### Device Status Monitoring (Statusbar Integration)

**Overview:**
The extension includes a non-obtrusive statusbar item that monitors Tenstorrent device status in real-time using tt-smi. This provides quick access to device information and actions without cluttering the UI.

**Statusbar Display States:**
- `$(check) TT: N150` - Device healthy (green checkmark)
- `$(warning) TT: N150` - Device has issues (yellow warning)
- `$(sync~spin) TT: Checking...` - Currently checking status
- `$(x) TT: No device` - No device detected (red X)
- `$(question) TT: Unknown` - Status unknown

**Architecture:**

1. **Cached Status Updates:**
   - Device status is cached to minimize tt-smi calls
   - Default update interval: 60 seconds (configurable: 30s, 1m, 2m, 5m, 10m)
   - Updates run in background using `child_process.exec()`
   - Timeout: 5 seconds to prevent blocking
   - Can be enabled/disabled by user

2. **Parsing tt-smi Output:**
   - Extracts device type (N150, N300, T3K, etc.) from "Board Type:" line
   - Extracts firmware version from "FW Version:" or "Firmware Version:" line
   - Detects errors by searching for keywords: "error", "failed", "timeout"
   - Status determination:
     - `healthy`: Device found, no errors
     - `warning`: Device found but has issues
     - `error`: tt-smi failed or errors detected
     - `unknown`: Initial state or parsing failed

3. **Quick Actions Menu (on click):**
   Opens QuickPick menu with contextual actions:
   - **Refresh Status** - Manually trigger tt-smi update (shows time since last check)
   - **Check Device Status** - Open terminal and run tt-smi for full output
   - **Firmware Version** - Display firmware version in info message (if available)
   - **Reset Device** - Run `tt-smi -r` with confirmation
   - **Clear Device State** - Kill processes, clear /dev/shm, reset device (requires sudo)
   - **Configure Update Interval** - Change auto-update frequency
   - **Enable/Disable Auto-Update** - Toggle periodic status checks

4. **Device Management Commands:**

   **tenstorrent.resetDevice:**
   ```typescript
   async function resetDevice(): Promise<void> {
     // Confirms with user before resetting
     // Runs tt-smi -r in terminal
     // Refreshes statusbar after 3 seconds
   }
   ```

   **tenstorrent.clearDeviceState:**
   ```typescript
   async function clearDeviceState(): Promise<void> {
     // Confirms with user (warns about sudo requirement)
     // Multi-step cleanup:
     //   1. Kill tt-metal and vllm processes
     //   2. Clear /dev/shm/tenstorrent* and /dev/shm/tt_*
     //   3. Reset device with tt-smi -r
     // Refreshes statusbar after 5 seconds
   }
   ```

**Implementation Details:**

**File:** `src/extension.ts` (lines 41-374)

**Key Functions:**
- `parseDeviceInfo(output: string)` - Parses tt-smi output into DeviceInfo object
- `updateDeviceStatus()` - Runs tt-smi async and updates cache
- `updateStatusBarItem()` - Updates statusbar text/icon based on cached status
- `showDeviceActionsMenu()` - Displays QuickPick menu with actions
- `configureUpdateInterval()` - Allows user to set update frequency
- `toggleAutoUpdate()` - Enable/disable periodic updates
- `startStatusUpdateTimer()` - Starts setInterval timer for updates
- `stopStatusUpdateTimer()` - Clears update timer
- `resetDevice()` - Soft reset via tt-smi -r
- `clearDeviceState()` - Full cleanup (processes, /dev/shm, reset)

**Persistent State:**
Stored in VSCode globalState (survives extension restarts):
- `STATUSBAR_UPDATE_INTERVAL` - Update interval in seconds (default: 60)
- `STATUSBAR_ENABLED` - Whether auto-update is enabled (default: true)

**Lifecycle:**
- Initialized in `activate()` - Creates statusbar item, registers commands, starts timer
- Cleaned up in `deactivate()` - Stops timer, clears references

**Error Handling:**
- tt-smi not found: Shows "No device" status
- tt-smi timeout (>5s): Shows "error" status
- Parse errors: Falls back to "unknown" status
- All errors are handled gracefully without blocking UI

**Benefits:**
- ✅ Non-obtrusive - Single statusbar item, doesn't take much space
- ✅ Efficient - Cached updates, configurable intervals
- ✅ Contextual - Only shows relevant actions based on device state
- ✅ Safe - Confirms before destructive operations
- ✅ Informative - Shows device type, firmware, last check time

**Use Cases:**
1. **Quick health check** - Glance at statusbar to see if device is healthy
2. **Firmware debugging** - View firmware version without running tt-smi
3. **Recovery from errors** - Reset device or clear state when initialization fails
4. **Development workflow** - Monitor device during development without manual checks

### VSCode Walkthroughs API Features Used

**Automatic Step Completion:**
- `completionEvents: ["onCommand:commandId"]` in package.json
- Steps auto-check when the specified command executes
- No manual state tracking required

**Native UI Benefits:**
- Automatically themed to match user's VS Code theme
- Appears in VS Code's "Get Started" page
- Can auto-open on extension first install
- Progress tracking built-in
- Markdown rendering built-in

**Terminal Integration:**
- Each lesson uses a dedicated named terminal
- Terminals persist across command invocations
- Terminal reuse: if terminal exists and is alive, reuse it
- Workspace root used as cwd for all terminals

## Adding New Lessons

To add a new lesson to the walkthrough:

1. **Create Content File:** `content/lessons/04-your-lesson.md`
   ```markdown
   # Your Lesson Title

   Description of what this lesson does...

   Show the command that will be executed:

   ```bash
   your-shell-command
   ```

   [🚀 Run Your Command](command:tenstorrent.yourCommand)
   ```

   **Note:** Command links on their own line are automatically rendered as buttons by VS Code. Use emojis for visual appeal.

2. **Add to package.json:**
   ```json
   {
     "id": "your-lesson-id",
     "title": "Your Lesson Title",
     "description": "Brief description...",
     "media": {
       "markdown": "content/lessons/04-your-lesson.md"
     },
     "completionEvents": ["onCommand:tenstorrent.yourCommand"]
   }
   ```

3. **Register Command in package.json:**
   ```json
   {
     "command": "tenstorrent.yourCommand",
     "title": "Your Command Title",
     "category": "Tenstorrent"
   }
   ```

4. **Implement Handler in extension.ts:**
   ```typescript
   function yourCommandHandler(): void {
     const terminal = getOrCreateTerminal('Your Terminal', 'yourTerminal');
     runInTerminal(terminal, 'your-shell-command');
     vscode.window.showInformationMessage('Your feedback message');
   }
   ```

5. **Register in activate():**
   ```typescript
   vscode.commands.registerCommand('tenstorrent.yourCommand', yourCommandHandler)
   ```

## Important Implementation Notes

- **No Custom UI:** All UI is provided by VSCode's native walkthrough system
- **Markdown Files Are Deployed:** The `content/` directory is copied to `dist/` during build
- **Terminal Persistence:** Terminals survive between command invocations but are cleaned up on deactivation
- **Password Input:** Use `password: true` in `showInputBox()` for sensitive inputs like tokens
- **Command Discovery:** Users find commands via walkthrough links, not the Command Palette
- **Completion Tracking:** VSCode automatically tracks which steps are completed based on `completionEvents`

## Lessons 4-6: Direct API and Production Deployment

### Evolution of Approach

**Initial Attempt (Deprecated):**
- Wrapped pytest demo with temporary JSON files
- Model reloaded for each query (2-5 minutes each time)
- Limited customization, black-box approach

**Current Approach (Implemented):**
- Use Generator API directly - model stays in memory
- 2-5 min initial load, then 1-3 sec per query (100x faster!)
- Full source code visible and customizable
- Educational and production-ready

### Challenge

User feedback: "A developer wants to do more than just force test code to work. They want the first thing they build to be something they can customize too."

Requirements:
- Use Generator API directly (not pytest wrapper)
- Open created files in editor automatically
- Progress from basic chat → HTTP API → production (vLLM)
- Maintain educational value while being production-ready

### Solution: Three-Tier Approach

#### Lesson 4: Direct API Chat
**File:** `content/templates/tt-chat-direct.py`

Uses Generator API to load model once and reuse it:

```python
# Load model once (slow)
from models.tt_transformers.tt.generator import Generator
from models.tt_transformers.tt.common import create_tt_model

model_args, model, tt_kv_cache, _ = create_tt_model(mesh_device, ...)
generator = Generator([model], [model_args], mesh_device, ...)

# Chat loop (fast - model already loaded!)
while True:
    prompt = input("> ")

    # Preprocess
    tokens, encoded, pos, lens = preprocess_inputs_prefill([prompt], ...)

    # Prefill (process prompt)
    logits = generator.prefill_forward_text(tokens, ...)

    # Decode (generate response)
    for _ in range(max_tokens):
        logits = generator.decode_forward_text(out_tok, current_pos, ...)
        next_token = sample(logits)
        if is_end_token(next_token):
            break

    response = tokenizer.decode(all_tokens)
    print(response)
```

**Key Benefits:**
- ✅ Model loads once, stays in memory
- ✅ ~100x faster for subsequent queries
- ✅ Full control over sampling, temperature, max_tokens
- ✅ Educational - see exactly how inference works
- ✅ Foundation for custom applications

#### Lesson 5: Direct API Server
**File:** `content/templates/tt-api-server-direct.py`

Flask HTTP server that loads model once on startup:

```python
# Global model loaded at startup
GENERATOR = None
MODEL_ARGS = None

def initialize_model():
    """Load model once when server starts"""
    global GENERATOR, MODEL_ARGS
    # ... load model using Generator API ...

@app.route('/chat', methods=['POST'])
def chat():
    """Use the loaded model for fast inference"""
    data = request.get_json()
    prompt = data['prompt']

    # Fast inference - model already loaded!
    response = generate_response(GENERATOR, prompt, ...)

    return jsonify({
        "response": response,
        "tokens_per_second": ...,
    })

# Load model once at startup
initialize_model()
app.run(...)
```

**Key Benefits:**
- ✅ Model loaded once on startup
- ✅ 1-3 second response time per request
- ✅ Full Flask customization available
- ✅ Production-grade patterns (globals, error handling)
- ✅ Foundation for microservices

#### Lesson 6: vLLM Production
**File:** `content/lessons/06-vllm-production.md`

Comprehensive guide to Tenstorrent's vLLM fork:

**Features:**
- OpenAI-compatible API (drop-in replacement)
- Continuous batching (efficient multi-user serving)
- Request queuing, priority scheduling
- Streaming responses
- Production-tested at scale

**Why vLLM:**
- Enterprise features vs. basic Flask server
- Standardized API vs. custom endpoints
- Battle-tested vs. prototype
- Horizontal scaling support

### Implementation Details

#### Editor Opening Feature

Commands now open created files automatically:

```typescript
async function createChatScriptDirect(): Promise<void> {
    // ... copy template ...

    // Open in editor automatically
    const doc = await vscode.workspace.openTextDocument(destPath);
    await vscode.window.showTextDocument(doc);

    vscode.window.showInformationMessage(
        '✅ Created script. The file is now open - review the code!'
    );
}
```

This allows developers to immediately see and customize the code.

#### Command Structure

New commands follow this pattern:
- `createXXXDirect()` - Copy template + open in editor
- `startXXXDirect()` - Start with proper environment setup
- `testXXXDirect()` - Test functionality with curl/SDK

### Performance Comparison

| Approach | Load Time | Query Time | Use Case |
|----------|-----------|------------|----------|
| Lesson 3 (pytest) | 2-5 min | 2-5 min | Testing |
| Old Lesson 4-5 (pytest wrapper) | 2-5 min | 2-5 min | Learning (deprecated) |
| New Lesson 4 (Direct API) | 2-5 min | 1-3 sec | Custom apps |
| New Lesson 5 (Flask Direct) | 2-5 min | 1-3 sec | API services |
| Lesson 6 (vLLM) | 2-5 min | 1-3 sec | Production |

**Impact:** 100x faster for subsequent queries!

### Why This Progression?

1. **Lesson 4:** Understand the Generator API
   - Load model, run inference
   - Full control and visibility
   - Foundation for everything else

2. **Lesson 5:** Wrap in HTTP API
   - Same Generator API, different interface
   - Learn server patterns
   - Prepare for deployment

3. **Lesson 6:** Scale to production
   - vLLM handles complexity
   - OpenAI compatibility
   - Enterprise features

Each lesson builds on the previous, maintaining the Generator API understanding while adding capabilities.

### Model Format Requirements (IMPORTANT)

**Two model formats needed:**
- **Meta native format** (in `original/` subdirectory): Used by pytest demos (Lesson 3)
  - Files: `params.json`, `consolidated.00.pth`, `tokenizer.model`
  - Environment variable: `LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original`
- **HuggingFace format** (in root directory): Used by Direct API and vLLM (Lessons 4-7)
  - Files: `config.json`, `model.safetensors`, etc.
  - Environment variable: `HF_MODEL=~/models/Llama-3.1-8B-Instruct`

**Current implementation:**
- Lesson 3 download command downloads BOTH formats (no `--include` filter)
- This ensures all subsequent lessons work correctly
- Total download size: ~16GB (both formats included)

**Model paths by lesson:**
- **Lesson 3** (pytest demos): `LLAMA_DIR=~/models/Llama-3.1-8B-Instruct/original`
- **Lessons 4-5** (Direct API): `HF_MODEL=~/models/Llama-3.1-8B-Instruct`
- **Lessons 6-7** (vLLM): `HF_MODEL=~/models/Llama-3.1-8B-Instruct`

### Lessons 6-7: vLLM and VSCode Chat (Implemented)

**Status:** Fully implemented and working

**Key features:**
1. **Lesson 6:** vLLM production server
   - Dedicated venv at `~/tt-vllm-venv` to avoid dependency conflicts
   - Requires dependencies: `fairscale`, `termcolor`, `loguru`, `blobfile`, `fire`, and `llama-models==0.0.48`
     - These are optional dependencies of `llama-models` that must be installed separately
     - `fairscale`: Required for model parallelism in llama3 reference implementation
     - `termcolor`: Required for colored terminal output in llama3/generation.py
     - `loguru`: Required by Tenstorrent's tt-metal vision demo code
     - `blobfile`, `fire`: Additional llama3 requirements
   - Uses HuggingFace model format
   - OpenAI-compatible API

2. **Lesson 7:** VSCode Chat integration
   - Chat participant: `@tenstorrent`
   - Connects to local vLLM server on port 8000
   - Streaming responses via Server-Sent Events
   - Full chat history context support

**Implementation files:**
- `content/lessons/06-vllm-production.md` (621 lines)
- `content/lessons/07-vscode-chat.md` (586 lines)
- Added chat participant and commands to `src/extension.ts`
- Updated `package.json` with chat participant definition

### Lesson 8: Image Generation with Stable Diffusion 3.5 Large (Implemented)

**Status:** Fully implemented with NATIVE TT HARDWARE ACCELERATION

**Key features:**
- ✅ **Native TT Acceleration** - Runs on tt-metal using TT-NN operators (NOT CPU!)
- ✅ **Stable Diffusion 3.5 Large** - State-of-the-art MMDiT architecture
- ✅ **High Resolution** - Generates 1024x1024 images (vs 512x512)
- ✅ **Fast** - ~12-15 seconds per image on N150 with hardware acceleration
- ✅ **Interactive Mode** - Built-in prompt input for multiple generations

**Architecture:**
- **3 Text Encoders:** CLIP-L, CLIP-G, T5-XXL for rich text understanding
- **MMDiT Transformer:** 38 blocks running on TT hardware
- **28 Inference Steps:** Optimized for quality/speed
- **VAE Decoder:** Converts latents to 1024x1024 pixels

**Implementation:**

1. **Model:** Stable Diffusion 3.5 Large (~10 GB)
   - Location: `models/experimental/stable_diffusion_35_large/`
   - From: `stabilityai/stable-diffusion-3.5-large`
   - Native TT-NN implementation in tt-metal

2. **Hardware Support:**
   - ✅ N150 (1x1 mesh) - Single chip
   - ✅ N300 (1x2 mesh) - Dual chip
   - ✅ T3K (1x8 mesh) - 8-chip system
   - ✅ TG (8x4 mesh) - Galaxy 32-chip

3. **Performance on N150:**
   - First run: Downloads model (~10 GB), loads (2-5 min)
   - Generation: ~12-15 seconds per 1024x1024 image
   - **Native hardware acceleration** - runs on TT cores!

4. **Commands:**
   ```bash
   # Set environment
   export MESH_DEVICE=N150

   # Generate with default prompt (sample)
   pytest models/experimental/stable_diffusion_35_large/demo.py

   # Interactive mode (custom prompts)
   export NO_PROMPT=0
   pytest models/experimental/stable_diffusion_35_large/demo.py
   ```

**Implementation files:**
- `content/lessons/08-image-generation.md` - Updated for SD 3.5 Large
- `src/commands/terminalCommands.ts` - 2 commands (generate, interactive)
- `src/extension.ts` - 2 command handlers
- `package.json` - Updated walkthrough step and commands
- Welcome page updated

**Extension Commands:**
- `tenstorrent.generateRetroImage` - Generate sample 1024x1024 image with default prompt
- `tenstorrent.startInteractiveImageGen` - Start interactive mode for custom prompts

**Key Advantages over SD 1.4:**
- ✅ **Native TT acceleration** (not CPU fallback)
- ✅ **4x higher resolution** (1024x1024 vs 512x512)
- ✅ **Better quality** - MMDiT architecture
- ✅ **Built into tt-metal** - No external dependencies
- ✅ **Production ready** - Optimized parallelization

The lesson teaches:
- How MMDiT transformers work
- Using experimental models in tt-metal
- Mesh device configuration for N150
- Native hardware-accelerated image generation
- Interactive prompt-based workflows

### Key Learnings

**From user feedback:**
- Developers want to customize, not just run black boxes
- Opening files in editor is crucial for learning
- Performance matters - 2-5 min per query is unacceptable for iteration
- Need clear progression: learn → prototype → production

## N150 Hardware Golden Path (Lesson 6-7)

**Status:** Configured for N150 single-chip cloud deployment

**Hardware Target:**
- N150 (Wormhole) single chip
- Cloud environment
- tt-metal: **latest main branch** (must be rebuilt after updates)
- vLLM branch: **dev (HEAD)**

**⚠️ Version Strategy Change:**
- Initially tried pinning to tt-metal v0.62.0-rc9 (August 2025)
- Found that vLLM dev requires latest tt-metal APIs
- **Solution:** Use latest on both repos (simpler, better tested)
- **Critical:** Must run `./install_dependencies.sh` then `./build_metal.sh` after git pull

**Model Configuration:**
- Model: Llama-3.1-8B-Instruct
- Size: 8B parameters (perfect for N150)
- Tensor Parallelism: Not needed (single chip)
- Context Length: 64K tokens (N150 limit)

**Required Environment Variables:**
```bash
export TT_METAL_HOME=~/tt-metal                # Point to tt-metal (required by setup-metal.sh)
export MESH_DEVICE=N150                        # Target N150 hardware
export HF_MODEL=~/models/Llama-3.1-8B-Instruct # Model path
export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH   # Critical: Find TTLlamaForCausalLM
```

**Required vLLM Flags:**
```bash
--max-model-len 65536                # 64K context limit for N150
```

**Why TT_METAL_HOME + PYTHONPATH matter:**
- `TT_METAL_HOME`: setup-metal.sh uses this to set PYTHON_ENV_DIR correctly
- `PYTHONPATH`: vLLM needs to import `TTLlamaForCausalLM` from tt-metal
- Without these, you get: "Cannot find model module 'TTLlamaForCausalLM'"
- The tt-metal directory contains the model implementation in `models/tt_transformers/tt/generator_vllm.py`

**Why This Configuration:**
1. Llama-3.1-8B officially supported on N150 (per vLLM tt_metal/README.md)
2. Single chip = simpler deployment (no multi-chip tensor parallelism)
3. Already downloaded in Lesson 3 (no additional downloads)
4. Compatible with tt-metal v0.62.0 family
5. Best balance of model capability vs hardware constraints

**Alternative Models (NOT recommended for N150):**
- Qwen 2.5 7B: Requires N300 (2 chips, TP=2)
- Llama 3.3 70B: Requires QuietBox (8 chips, TP=8)
- Larger models exceed N150 memory capacity

**Commands Updated:**
- `tenstorrent.runVllmOffline`: Includes MESH_DEVICE=N150 and --max_model_len 65536 (underscores)
- `tenstorrent.startVllmServer`: Includes MESH_DEVICE=N150 and --max-model-len 65536 (hyphens)
- `tenstorrent.startVllmForChat`: Includes MESH_DEVICE=N150 and --max-model-len 65536 (hyphens)
- `tenstorrent.installVllm`: Includes all discovered dependencies

**Note:** offline_inference_tt.py uses underscores `--max_model_len`, API server uses hyphens `--max-model-len`

**Complete Dependency List (discovered through testing):**
```bash
pip install --upgrade ttnn pytest  # Critical: must upgrade ttnn for v0.62.0-rc9
pip install fairscale termcolor loguru blobfile fire pytz llama-models==0.0.48
```

**Documentation Updated:**
- Lesson 6: Added "Hardware Configuration: N150 Golden Path" section
- Lesson 6: Updated all vLLM commands with N150 configuration
- Lesson 6: Added ttnn, pytest, and pytz to dependency list
- All commands now explicitly set MESH_DEVICE=N150 and context limit
- Troubleshooting section updated with ttnn upgrade instructions

### ✅ WORKING: N150 vLLM Golden Path (2025-11-05)

**Status:** Successfully tested and working on N150 hardware in cloud environment.

**Complete Working Command:**
```bash
cd ~/tt-vllm && \
  source ~/tt-vllm-venv/bin/activate && \
  export TT_METAL_HOME=~/tt-metal && \
  export MESH_DEVICE=N150 && \
  export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH && \
  source ~/tt-vllm/tt_metal/setup-metal.sh && \
  python ~/tt-scratchpad/start-vllm-server.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 65536 \
    --max-num-seqs 16 \
    --block-size 64
```

**Key Success Factors:**
1. ✅ Custom starter script with `if __name__ == '__main__':` guard (prevents multiprocessing errors)
2. ✅ Local model path (avoids HuggingFace gated repo authentication)
3. ✅ TT model registration via `register_tt_models()`
4. ✅ N150-specific configuration (64K context, appropriate batch size)
5. ✅ All environment variables set correctly (TT_METAL_HOME, MESH_DEVICE, PYTHONPATH)

**Version:** Extension v0.0.20 includes all fixes and tested configuration.

### CRITICAL: vLLM TT Model Registration (2025-11-05)

**Problem Discovered:**
When users ran `python -m vllm.entrypoints.openai.api_server` directly, they got:
```
ValidationError: Cannot find model module. 'TTLlamaForCausalLM' is not a registered model
WARNING: TTLlamaForCausalLM has no vLLM implementation, falling back to Transformers
```

**Root Cause:**
vLLM doesn't automatically know about TT-specific model implementations (TTLlamaForCausalLM, etc.) in the tt-metal repository. These models must be explicitly registered using vLLM's `ModelRegistry.register_model()` API before starting the server.

**Solution:**
Use `examples/server_example_tt.py` instead of calling the API server directly. This script:

1. Calls `register_tt_models()` which registers all TT models:
   ```python
   ModelRegistry.register_model("TTLlamaForCausalLM",
       "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM")
   ```

2. Then starts the API server with the correct parameters

**Updated Command (Step 4 in Lesson 6):**
```bash
python ~/tt-scratchpad/start-vllm-server.py \
    --model ~/models/Llama-3.1-8B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 16 \
    --block-size 64
```

**Why a custom starter script? (v0.0.69+)**

vLLM requires explicit registration of Tenstorrent model implementations before starting. Our production-ready starter script (`start-vllm-server.py`):

1. **Self-contained registration** - Inlines the model registration code (no dependency on fragile `examples/` directory)
2. **Production-ready** - Can be version controlled, deployed, and maintained
3. **Accepts local model paths** - No validation against hardcoded lists
4. **Passes all args through** - Full control over vLLM server flags

**What it does:**
```python
from vllm import ModelRegistry

def register_tt_models():
    # Register TT Llama implementation with vLLM
    ModelRegistry.register_model(
        "TTLlamaForCausalLM",
        "models.tt_transformers.tt.generator_vllm:LlamaForCausalLM"
    )
    print("✓ Registered Tenstorrent models with vLLM")

if __name__ == '__main__':
    register_tt_models()  # Must happen before server starts
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
```

The extension automatically creates this script in `~/tt-scratchpad/` from the template at `content/templates/start-vllm-server.py`.

**Why not import from examples/?**
- ❌ `examples/` is not production code (may change/break)
- ❌ Creates fragile dependency on repository structure
- ❌ Not suitable for deployment

**Why not use `python -m vllm.entrypoints.openai.api_server` directly?**
- ❌ TT models not registered → ValidationError
- ❌ No CLI flags or environment variables for registration
- ❌ Falls back to CPU-based HuggingFace Transformers (slow)

**Key Insight:**
Model registration via `ModelRegistry.register_model()` is NOT optional. Without it, vLLM cannot find TT model implementations even with correct PYTHONPATH and TT_METAL_HOME.

**Environment variables still required:**
- `TT_METAL_HOME=~/tt-metal` - Required by setup-metal.sh
- `PYTHONPATH=$TT_METAL_HOME` - Allows Python to import from tt-metal
- `MESH_DEVICE=N150` - Hardware target
- Must source `~/tt-vllm/tt_metal/setup-metal.sh`

**Model path:** Use the local path `~/models/Llama-3.1-8B-Instruct` (not HF model name). The model has HuggingFace format files in the root directory.

## File Structure

```
tt-vscode-ext-clean/
├── content/
│   ├── lessons/           # Markdown content files (editable by technical writers)
│   │   ├── 01-hardware-detection.md
│   │   ├── 02-verify-installation.md
│   │   ├── 03-download-model.md
│   │   ├── 04-interactive-chat.md
│   │   ├── 05-api-server.md
│   │   ├── 06-vllm-production.md
│   │   └── 07-vscode-chat.md
│   ├── templates/         # Python script templates deployed with extension
│   │   ├── tt-chat.py              # Interactive REPL wrapper
│   │   ├── tt-api-server.py        # Flask HTTP API wrapper
│   │   ├── tt-chat-direct.py       # Direct Generator API chat
│   │   └── tt-api-server-direct.py # Direct Generator API server
│   └── welcome/           # Welcome page content
│       └── welcome.html   # Welcome page HTML with walkthrough links
├── src/
│   ├── extension.ts       # Main extension code (command handlers + webview)
│   ├── commands/
│   │   └── terminalCommands.ts  # Terminal command definitions
│   └── types/
│       └── lesson.ts      # Legacy type definitions (not currently used)
├── vendor/
│   └── tt-metal/          # Cloned for reference (not deployed)
├── package.json           # Extension manifest with walkthrough definitions
├── tsconfig.json          # TypeScript configuration
└── CLAUDE.md             # This file
```

**Generated Files (User System):**
```
~/tt-scratchpad/           # Extension-generated scripts (auto-created)
├── tt-chat-direct.py      # Direct API chat script
└── tt-api-server-direct.py # Direct API server script
```

## Migration Notes

This extension was refactored from a custom webview-based approach to use VSCode's native Walkthroughs API. Benefits:

- **72% code reduction** (775 lines → 217 lines)
- **Zero custom UI code** (no HTML/CSS/webview management)
- **Native markdown support** (technical writers edit .md files)
- **Automatic completion tracking** (no manual state management)
- **Better UX** (native VS Code theming and integration)

The old approach required custom state management, HTML generation, and webview message passing. The new approach is declarative and leverages VSCode's built-in capabilities.

## Lesson 9: Coding Assistant with Prompt Engineering (2025-11-06)

**Final Implementation:** Uses Llama 3.1 8B with coding-focused system prompt

### Evolution of Lesson 9

**Iteration 1: Qwen 2.5 Coder 7B (Abandoned)**
- Attempted to use Qwen 2.5 Coder 7B for N150
- **Blocker:** Qwen requires N300 minimum (TP=2, dual-chip)
- No Qwen model supports N150 single-chip

**Iteration 2: Llama 3.2 6B AlgoCode (Abandoned)**
- Attempted community fine-tune "prithivMLmods/Llama-3.2-6B-AlgoCode"
- **Blocker 1:** Model only has HuggingFace format (no Meta `original/` checkpoint)
- **Blocker 2:** Weight dimensions not tile-aligned for tt-metal (32x32 requirement)
- Error: `Physical shard shape (10036, 352) must be tile {32, 32} sized!`
- **Root cause:** Direct API requires Meta checkpoint format with tt-metal-compatible weights

**Iteration 3: Llama 3.1 8B + Prompt Engineering (FINAL)**
- ✅ Uses proven tt-metal compatible model (already downloaded in Lesson 3)
- ✅ No compatibility issues - Meta format with proper tile alignment
- ✅ Teaches prompt engineering - critical real-world skill
- ✅ 80%+ of specialized model quality through system prompts
- ✅ Transferable knowledge to all LLMs

### Key Technical Insights

**Why Specialized Models Failed:**
1. **Model format mismatch:** Community HuggingFace models lack Meta checkpoint format
2. **Tile alignment:** tt-metal requires weights divisible by 32x32 tiles
3. **Hardware constraints:** Some models require multi-chip (TP > 1)
4. **Direct API requirements:** Needs specifically adapted model architectures

**Why Prompt Engineering Succeeds:**
- Works with any compatible model
- No weight conversion needed
- No additional downloads
- Industry-standard technique
- Easy to customize and iterate

### Implementation Details

**Files Changed:**
1. `content/lessons/09-algocode-assistant.md` - Complete rewrite with prompt engineering focus
   - Added "Future Model Options" preamble explaining blockers
   - Emphasis on prompt engineering as real-world technique
   - Examples of system prompt architecture
   - Comparison tables: prompt engineering vs model specialization
   - Advanced techniques: few-shot, chain-of-thought, constrained generation
   - Customization ideas for different domains

2. `content/templates/tt-coding-assistant.py` - New template (replaces tt-algocode-chat.py)
   - Uses Llama 3.1 8B with LLAMA_DIR (Meta format)
   - Coding-focused system prompt embedded
   - Temperature 0.7 for balanced determinism
   - Educational comments explaining prompt engineering

3. `src/extension.ts` - Updated command handlers
   - Removed AlgoCode model from MODEL_REGISTRY
   - `createAlgoCodeChatScript` now creates tt-coding-assistant.py
   - `startAlgoCodeChat` uses Llama 3.1 8B original path
   - Updated user-facing messages

4. `src/commands/terminalCommands.ts` - Updated command templates
   - DOWNLOAD_ALGOCODE now verifies Llama 3.1 8B (not downloads AlgoCode)
   - START_ALGOCODE_CHAT uses correct model path
   - Updated descriptions

5. `package.json` - Updated walkthrough metadata
   - Title: "Coding Assistant with Prompt Engineering"
   - Description emphasizes prompt engineering skill
   - Command titles updated

**System Prompt Example:**
```python
SYSTEM_PROMPT = """You are an expert coding assistant specializing in:
- Algorithm design and analysis
- Data structures (trees, graphs, heaps, hash tables)
- Code debugging and optimization
...

When answering:
- Provide clear, concise explanations
- Include code examples when relevant
- Explain your reasoning
- Consider edge cases
..."""
```

### User Experience

**What Users Learn:**
1. Prompt engineering fundamentals
2. System prompt architecture
3. Shaping model behavior through instructions
4. Few-shot learning, chain-of-thought, constrained generation
5. Domain-specific customization

**Practical Applications:**
- Interactive code review
- Algorithm learning
- Debugging assistance
- Code translation between languages
- Test generation
- Documentation generation

**Extensibility:**
- Customize system prompt for different domains (web dev, systems programming)
- Add context management for multi-turn conversations
- Integrate code execution
- Add file I/O for codebase analysis
- Build IDE integrations

### Future Path Forward

**When Model Support Expands:**
The lesson includes a forward-looking "Future Model Options" section listing:
- Llama 3.2 6B AlgoCode - Needs weight conversion for tile alignment
- Qwen 2.5 Coder 7B - Needs N300 or single-chip optimization
- CodeLlama - Needs architecture compatibility work
- StarCoder2 - Needs custom tt-metal implementation

**Migration Path:**
Once models become compatible, users can swap them in using the same Direct API pattern they learned. The prompt engineering skills remain valuable across all models.

### Key Takeaway

**Prompt engineering is not a workaround - it's a feature.**

Real production systems rely heavily on prompt engineering because:
- It works across all models
- Easy to iterate and customize
- No model training required
- Immediate results
- Often delivers 80%+ of specialized model quality

This lesson teaches a critical skill that applies to GPT, Claude, Gemini, and all future LLMs.

## TT-Jukebox: Moved to Standalone Repository

**Note:** TT-Jukebox has been extracted into its own open-source project!

**Repository Location:** `~/tt-jukebox/`

TT-Jukebox is an intelligent environment manager for Tenstorrent hardware that:
- Auto-detects your hardware (N150, N300, T3K, P100, etc.)
- Fetches official model specifications from GitHub
- Matches models to your hardware with intelligent filtering
- Executes automated setup (checkouts, builds, downloads)
- Shows transparent, copy-pasteable vLLM commands

### Quick Start

```bash
# Navigate to the repository
cd ~/tt-jukebox

# List compatible models
python3 tt-jukebox.py --list

# Find chat models
python3 tt-jukebox.py chat

# Search for specific model
python3 tt-jukebox.py --model llama

# Get ready-to-run vLLM command
python3 tt-jukebox.py --model llama-3.1-8b

# Full automated setup
python3 tt-jukebox.py --model llama-3.1-8b --setup --force
```

### Documentation

For complete documentation, see:
- `~/tt-jukebox/README.md` - Full project documentation
- `~/tt-jukebox/USAGE_EXAMPLES.md` - Detailed usage examples
- `~/tt-jukebox/tt-jukebox.py` - Source code with inline documentation

### Why a Separate Repository?

TT-Jukebox is now a standalone tool that can be:
- ✅ Used independently of the VSCode extension
- ✅ Version controlled separately
- ✅ Shared across teams
- ✅ Integrated into CI/CD pipelines
- ✅ Contributed to by the community

The project includes:
- Full source code
- Comprehensive README
- Usage examples for all hardware types
- MIT License for open-source use

## Lesson 11: Image Classification with TT-Forge (2025-11-14)

**Status:** Implemented with critical environment variable fixes (v0.0.50)

### Overview

TT-Forge is Tenstorrent's MLIR-based compiler that aims to run PyTorch models on TT hardware with less manual kernel programming than TT-Metal. However, it's under active development and not all models work yet.

**Key lesson approach:**
- Realistic about limitations (many models fail to compile)
- Start with validated models (MobileNetV2, ResNet family)
- Two installation options: build from source (recommended) vs wheels (quick but may fail)
- Visual feedback through image classification

### Critical Discovery: Environment Variable Pollution (v0.0.50)

**Problem:** The #1 cause of `ImportError: undefined symbol` errors is environment variable pollution from TT-Metal installations.

**Root Cause:** From [GitHub issue #529](https://github.com/tenstorrent/tt-forge/issues/529):
- `TT_METAL_HOME` and `TT_METAL_VERSION` environment variables cause TT-Forge to load TT-Metal from outdated system paths
- Even with correct installation, these variables override the build and load the wrong version
- Results in symbol resolution failures

**Solution:** Must unset these variables BEFORE building or running TT-Forge:

```bash
# Critical first step
unset TT_METAL_HOME
unset TT_METAL_VERSION
```

**Permanent fix in `~/.bashrc`:**
```bash
# Prevent TT-Metal environment pollution for forge
unset TT_METAL_HOME
unset TT_METAL_VERSION
```

### Implementation Changes (v0.0.50)

**1. Updated Lesson Content:** `content/lessons/11-forge-image-classification.md`
- Added prominent warning about environment variables at top of Step 1
- Updated both build-from-source and wheel installation commands to include `unset` statements
- Completely rewrote troubleshooting section to prioritize environment variables as ROOT CAUSE #1 (90% of cases)
- Added links to GitHub issue #529 for reference

**2. Updated Terminal Commands:** `src/commands/terminalCommands.ts`
```typescript
BUILD_FORGE_FROM_SOURCE: {
  template: 'unset TT_METAL_HOME && unset TT_METAL_VERSION && sudo mkdir -p ...',
  description: 'Builds TT-Forge from source... Clears environment variables first to prevent conflicts.',
}

INSTALL_FORGE: {
  template: 'unset TT_METAL_HOME && unset TT_METAL_VERSION && python3 -m venv ...',
  description: 'Creates venv and installs TT-Forge-FE wheels... Clears environment variables first to prevent conflicts.',
}
```

**3. Version Bump:** `package.json` → v0.0.50

### Installation Options

**Option A: Build from Source (Recommended)**
- Official CMake-based build process
- Guarantees compatibility with your exact TT-Metal version
- Builds against `/opt/ttforge-toolchain` and `/opt/ttmlir-toolchain`
- Requires clang-17, cmake, ninja-build
- Takes 10-20 minutes (one-time cost)

**Option B: Wheel Installation**
- Quick installation from Tenstorrent PyPI
- May have version mismatches (wheels built against specific TT-Metal versions)
- Useful for quick prototyping if willing to troubleshoot

### Model Selection: MobileNetV2

**Why this model:**
- ✅ Validated in tt-forge-models repository (confirmed working)
- ✅ Lightweight (3.5M parameters)
- ✅ Standard CNN architecture (no exotic operators)
- ✅ Fast compilation (simpler graph than larger models)
- ✅ Visual feedback (1000 ImageNet classes)

**Realistic expectations:**
- Not all PyTorch models will compile
- Start with validated examples from tt-forge-models (169 models)
- Operator coverage expanding but incomplete
- Compilation failures are normal for unvalidated models

### Workflow

1. **Clear environment variables** (CRITICAL)
2. **Install TT-Forge** (build from source or wheels)
3. **Test installation** (verify forge module loads)
4. **Create classifier script** (MobileNetV2 with forge.compile())
5. **Run classification** (first compilation takes 2-5 minutes)
6. **Classify custom images** (subsequent runs faster with caching)

### Key Technical Details

**forge.compile() API:**
```python
import forge

# Create sample input for shape inference
sample_input = torch.randn(1, 3, 224, 224)  # Batch, Channels, Height, Width

# Compile for TT hardware
compiled_model = forge.compile(model, sample_inputs=[sample_input])

# Run inference
output = compiled_model(input_tensor)
```

**What happens during compilation:**
1. Graph capture: Traces PyTorch operations
2. Operator validation: Checks if all ops are supported
3. Optimization: Applies fusion, layout transforms
4. Lowering: Converts to TTNN operations (TT-Metal layer)
5. Device mapping: Allocates tensors, schedules execution
6. JIT compilation: Generates device kernels

**Can fail if:**
- Unsupported operators encountered
- Dynamic shapes or control flow
- Memory constraints exceeded
- Operator combinations not validated

### Troubleshooting Priority

**90% of symbol errors:** Environment variable pollution
- Check `echo $TT_METAL_HOME` and `echo $TT_METAL_VERSION`
- Unset both before running forge
- Add to `~/.bashrc` for permanent fix

**10% of symbol errors:** True version mismatch
- Wheels built against different TT-Metal version
- Solution: Build from source with both repos on main

### Files Modified

**content/lessons/11-forge-image-classification.md:**
- Added environment variable warning section
- Updated installation commands with `unset` statements
- Rewrote troubleshooting to prioritize environment variables
- Added GitHub issue #529 references

**content/templates/tt-forge-classifier.py:**
- Complete MobileNetV2 classification script
- forge.compile() usage example
- ImageNet preprocessing
- Top-5 prediction display

**src/commands/terminalCommands.ts:**
- BUILD_FORGE_FROM_SOURCE: Prepended with `unset TT_METAL_HOME && unset TT_METAL_VERSION`
- INSTALL_FORGE: Prepended with `unset TT_METAL_HOME && unset TT_METAL_VERSION`
- TEST_FORGE_INSTALL: Tests forge import and device detection
- CREATE_FORGE_CLASSIFIER: Copies template to ~/tt-scratchpad
- RUN_FORGE_CLASSIFIER: Activates forge env and runs classifier

**src/extension.ts:**
- Added forge terminal types
- Registered 5 forge commands
- Command handlers with proper user feedback

**package.json:**
- Version bumped to 0.0.50
- Added walkthrough step between Jukebox and Bounty Program
- Registered forge commands

**content/welcome/welcome.html:**
- Added Lesson 11 card with 🔨 icon

### Key Takeaways

**1. Environment variables are critical**
- Must unset `TT_METAL_HOME` and `TT_METAL_VERSION` before using forge
- This is the #1 cause of ImportError issues (90% of cases)
- Make permanent by adding to `~/.bashrc`

**2. Start with validated models**
- 169 models in tt-forge-models are tested/confirmed working
- Compilation failures are normal for unvalidated models
- Operator coverage expanding but incomplete

**3. Build from source is most reliable**
- Guarantees compatibility with your TT-Metal version
- Avoids wheel version mismatch issues
- Better for development and experimentation

**4. Realistic expectations**
- TT-Forge is under active development
- Not all PyTorch models will compile
- Start with validated examples
- File issues to help the community

### Resources

- TT-Forge Overview: https://github.com/tenstorrent/tt-forge
- TT-Forge-FE: https://github.com/tenstorrent/tt-forge-fe
- TT-Forge-Models (169 validated): https://github.com/tenstorrent/tt-forge-models
- Environment variable issue: https://github.com/tenstorrent/tt-forge/issues/529
- Discord: https://discord.gg/tenstorrent

## Lesson 12: JAX Inference with TT-XLA (2025-11-27)

**Status:** Implemented (v0.0.77) - Production-ready alternative to TT-Forge

### Overview

TT-XLA is Tenstorrent's production-ready XLA-based compiler that provides JAX and PyTorch/XLA support with multi-chip capabilities. Unlike TT-Forge (experimental, single-chip), TT-XLA is mature and supports N300/T3K/Galaxy systems.

**Key advantages:**
- ✅ **Simple installation:** Wheel-based via pip (no building)
- ✅ **Python 3.10+ compatible:** Works with standard Python versions
- ✅ **No tt-metal rebuild:** Works with existing installation
- ✅ **Multi-chip ready:** Tensor parallelism (TP) and data parallelism (DP)
- ✅ **Production-tested:** Most mature compiler in the stack

### User Problem Solved

**Context:** User's cloud environment constraints:
- tt-metal locked at SHA 9b67e09
- Python 3.10.12 (cannot change)
- TT-Forge requires Python 3.11+ and complex build
- User reported: "I still can't get the forge lesson to work. Let's try another tact."

**Solution:** TT-XLA installation via PJRT plugin:
```bash
python3 -m venv tt-xla-venv
source tt-xla-venv/bin/activate
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install jax flax transformers
```

### Architecture

**PJRT (Portable JAX Runtime):**
- Google's standard interface for hardware backends
- Used by TPUs, GPUs, now Tenstorrent
- Enables automatic device selection
- No custom integration code needed

**Stack:**
```
Your Code (JAX)
    ↓
TT-XLA PJRT Plugin (pjrt-plugin-tt wheel)
    ↓
TT-MLIR Compiler (bundled in wheel)
    ↓
TT-Metal (existing ~/tt-metal)
    ↓
Hardware (N150/N300/T3K/Galaxy)
```

**Why this works without rebuilding tt-metal:**
The `pjrt-plugin-tt` wheel includes:
1. TT-MLIR compiler (bundled)
2. PJRT interface (connects JAX to hardware)
3. Runtime compatibility layer (interfaces with tt-metal)

### Implementation

**Files created:**

1. **`content/lessons/12-tt-xla-jax.md`** - Complete lesson with:
   - Installation instructions
   - JAX test script
   - Official GPT-2 demo
   - Multi-chip configuration
   - Comparison table: TT-XLA vs TT-Forge vs TT-Metal
   - Troubleshooting section

2. **`TT_XLA_INSTALL_GUIDE.md`** - Standalone reference guide

**Files modified:**

1. **`src/commands/terminalCommands.ts`** - Added 5 commands:
   - `INSTALL_TT_XLA` - Creates venv and installs PJRT plugin
   - `CREATE_TT_XLA_TEST` - Creates JAX test script
   - `TEST_TT_XLA_INSTALL` - Runs test to verify TtDevice
   - `DOWNLOAD_TT_XLA_DEMO` - Downloads GPT-2 demo
   - `RUN_TT_XLA_DEMO` - Runs GPT-2 inference

2. **`src/extension.ts`** - Added 3 command handlers:
   - `installTtXla()` - Installs TT-XLA PJRT plugin
   - `testTtXlaInstall()` - Creates and runs test script
   - `runTtXlaDemo()` - Downloads and runs GPT-2 demo

3. **`package.json`** - Version 0.0.76 → 0.0.77
   - Added 3 commands to contributes.commands
   - Added walkthrough step between Forge and Bounty Program
   - Completion event: `onCommand:tenstorrent.testTtXlaInstall`

### Lesson Structure

**Step 1: Install TT-XLA**
- One pip command installs PJRT plugin
- Works with Python 3.10+
- No compilation required

**Step 2: Test Installation**
- Creates JAX test script at `~/tt-scratchpad/test-tt-xla.py`
- Verifies TtDevice detection
- Simple dot product example

**Step 3: Run Official GPT-2 Demo**
- Downloads from tt-forge repository
- Uses Flax (JAX neural network library)
- Runs inference on TT hardware

**Step 4: Multi-Chip Configuration**
- N300 (2 chips): `jax.config.update('jax_tt_mesh', '1x2')`
- T3K (8 chips): `jax.config.update('jax_tt_mesh', '1x8')`
- Galaxy (32 chips): `jax.config.update('jax_tt_mesh', '8x4')`

**Step 5: Next Steps**
- Image classification with JAX/Flax
- More demos from tt-forge repository
- PyTorch/XLA support (still maturing)

### Comparison: TT-XLA vs TT-Forge vs TT-Metal

| Feature | TT-XLA | TT-Forge | TT-Metal |
|---------|--------|----------|----------|
| **Maturity** | Production | Beta/Experimental | Stable |
| **Installation** | Wheel (easy) | Build from source | Already installed |
| **Python version** | 3.10+ | 3.11+ | Any |
| **Multi-chip** | ✅ Yes (TP/DP) | ❌ Single-chip | ✅ Yes |
| **Frameworks** | JAX, PyTorch/XLA | PyTorch, ONNX | Direct API |
| **Use case** | Production inference | Experimental models | Low-level kernels |

### Key Benefits

**For constrained cloud environments:**
- No Python version change required
- No tt-metal rebuild needed
- Simple one-command installation
- Production-ready maturity

**For general users:**
- Learn JAX framework (increasingly popular)
- Production-grade inference
- Multi-chip scaling path
- PJRT standard interface (portable knowledge)

### Resources

**Official Documentation:**
- TT-XLA Docs: https://docs.tenstorrent.com/tt-xla/
- TT-XLA Repo: https://github.com/tenstorrent/tt-xla
- TT-Forge Demos: https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla
- PJRT Overview: https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html

**JAX Resources:**
- JAX Documentation: https://jax.readthedocs.io/
- Flax (Neural Networks): https://flax.readthedocs.io/
- JAX Tutorials: https://jax.readthedocs.io/en/latest/notebooks/quickstart.html

### Walkthrough Hierarchy

Current lesson progression:
- Lesson 10: Environment Management with TT-Jukebox
- Lesson 11: Image Classification with TT-Forge (experimental)
- **Lesson 12: JAX Inference with TT-XLA** (production) ← NEW
- Bounty Program: Model Bring-Up
- Exploring TT-Metalium
- TT-Metalium Cookbook

This completes the compiler stack coverage:
- **Low-level:** TT-Metal (Lessons 1-10)
- **Experimental:** TT-Forge (Lesson 11)
- **Production:** TT-XLA (Lesson 12) ← NEW
- **Specialized:** vLLM for LLMs (Lessons 6-7)

---

## User-Facing Documentation

**For user-facing FAQs and troubleshooting**, see `FAQ.md` in the project root. That file is deployed with the extension and contains comprehensive user guidance.

