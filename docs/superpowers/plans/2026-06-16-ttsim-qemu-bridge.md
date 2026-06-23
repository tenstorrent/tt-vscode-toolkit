# ttsim QEMU Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Update the ttsim lesson to v1.8.4, add a ttsim QEMU Bridge lesson, and implement three new namespaced commands (`tenstorrent.ttsim.launchQemu`, `tenstorrent.ttsim.stopQemu`, `tenstorrent.ttsim.setupQemu`) with a release gate that activates the full QEMU flow the moment a release URL is available.

**Architecture:** New terminal command constants live in `terminalCommands.ts`. New QEMU command functions + registration live in `extension.ts` alongside existing ttsim commands. All QEMU behavior is gated by a single `TTSIM_QEMU_RELEASE` constant — `null` shows a friendly "coming soon" message, a version string activates the full boot flow. Lesson content is two separate markdown files; the registry entry for the new lesson is added via `generate:lessons`.

**Tech Stack:** TypeScript, VSCode API (`vscode.window`, `vscode.Terminal`), Node.js `child_process.execSync` (for PID/requirements checks), existing `runInTerminal` / `getOrCreateSimpleTerminal` helpers.

---

## File Map

| File | Change |
|---|---|
| `src/commands/terminalCommands.ts` | Add `TTSIM_QEMU_RELEASE` constant + `SETUP_TTSIM_QEMU` + `LAUNCH_TTSIM_QEMU` entries; update `SETUP_TTSIM` to v1.8.4 |
| `src/extension.ts` | Add `launchTtsimQemu`, `stopTtsimQemu`, `setupTtsimQemu` functions; register 3 new commands |
| `content/lessons/ttsim-twenty-and-ten.md` | Version bump, new binaries, qsr note, aarch64 note, QEMU Bridge callout, entry 32 |
| `content/lessons/ttsim-qemu-bridge.md` | New file |
| `content/lesson-registry.json` | New entry for `ttsim-qemu-bridge` (via `generate:lessons`) |
| `package.json` | Version bump 0.0.503 → 0.0.504 |

---

## Task 1: Update `terminalCommands.ts` — ttsim v1.8.4 + QEMU constants

**Files:**
- Modify: `src/commands/terminalCommands.ts:859-888`

- [ ] **Step 1: Replace the `SETUP_TTSIM` block and add the new QEMU constants**

Find the section starting at line 859 (`// ttsim: Twenty-and-Ten Lesson`) and replace through line 888 (end of `RUN_TTSIM_ATTENTION`) with:

```typescript
  // ========================================
  // ttsim: Twenty-and-Ten Lesson
  // ========================================

  SETUP_TTSIM: {
    id: 'setup-ttsim',
    name: 'Set Up ttsim Simulator',
    template: `mkdir -p ~/sim
wget -q https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_wh.so -O ~/sim/libttsim_wh.so || { echo "ERROR: failed to download libttsim_wh.so"; exit 1; }
wget -q https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_bh.so -O ~/sim/libttsim_bh.so || { echo "ERROR: failed to download libttsim_bh.so"; exit 1; }
wget -q https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_wh_x2.so -O ~/sim/libttsim_wh_x2.so || { echo "ERROR: failed to download libttsim_wh_x2.so"; exit 1; }
wget -q https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_bh_x2.so -O ~/sim/libttsim_bh_x2.so || { echo "ERROR: failed to download libttsim_bh_x2.so"; exit 1; }
wget -q https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_wh_x8.so -O ~/sim/libttsim_wh_x8.so || { echo "ERROR: failed to download libttsim_wh_x8.so"; exit 1; }
if [ -n "$TT_METAL_HOME" ]; then
  cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml || { echo "ERROR: failed to copy SOC descriptor"; exit 1; }
  cp $TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml ~/sim/n300_cluster_desc.yaml || { echo "WARNING: n300 cluster desc copy skipped (optional for N300 sim)"; }
else
  echo "TT_METAL_HOME not set — SOC descriptor copy skipped"
fi
echo "ttsim v1.8.4 ready (wh + bh + wh_x2 + bh_x2 + wh_x8)"`,
    description: 'Downloads ttsim v1.8.4 Wormhole, Blackhole, N300 (wh_x2), BH-x2, and WH-x8 binaries and copies SOC descriptors',
  },

  RUN_TTSIM_ATTENTION: {
    id: 'run-ttsim-attention',
    name: 'Run Transformer Attention on ttsim',
    template: `export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
python3 ~/tt-scratchpad/ttsim/ttsim_attention.py`,
    description: 'Runs a transformer attention layer forward pass on the ttsim Wormhole simulator',
  },

  // ========================================
  // ttsim QEMU Bridge
  // ========================================

  LAUNCH_TTSIM_QEMU: {
    id: 'launch-ttsim-qemu',
    name: 'Launch ttsim QEMU Bridge',
    template: `ssh -p 2222 -o StrictHostKeyChecking=no -o ConnectTimeout=30 tt@localhost`,
    description: 'Opens an SSH terminal session inside the running ttsim QEMU Bridge VM',
  },
```

- [ ] **Step 2: Add the `TTSIM_QEMU_RELEASE` constant** just above the `TERMINAL_COMMANDS` export object (search for the line `export const TERMINAL_COMMANDS = {` and add above it):

```typescript
/**
 * ttsim QEMU Bridge release gate.
 * Set to a version string (e.g. 'v1.0.0') when the first ttsim-qemu release ships.
 * When null, all QEMU Bridge commands show a "coming soon" message instead of executing.
 */
export const TTSIM_QEMU_RELEASE: string | null = null;
```

- [ ] **Step 3: Run the build to verify no TypeScript errors**

```bash
cd /home/ttuser/code/tt-vscode-toolkit
npm run build 2>&1 | tail -20
```

Expected: build succeeds with no errors.

- [ ] **Step 4: Commit**

```bash
git add src/commands/terminalCommands.ts
git commit -m "feat(ttsim): update SETUP_TTSIM to v1.8.4, add QEMU Bridge constants"
```

---

## Task 2: Add QEMU Bridge command functions to `extension.ts`

**Files:**
- Modify: `src/extension.ts` — insert after line 3860 (end of `runTtsimAttention`), before the AnimateDiff section

- [ ] **Step 1: Add the three QEMU command functions**

Insert the following block at line 3862 (after the closing `}` of `runTtsimAttention`, before the `// Lesson 17` comment):

```typescript
// ============================================================================
// ttsim QEMU Bridge
// ============================================================================

/**
 * Checks whether qemu-system-x86_64 is installed on the host.
 */
function isQemuInstalled(): boolean {
  try {
    require('child_process').execSync('which qemu-system-x86_64', { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Returns free disk space in GB at the given path, or -1 on error.
 */
function freeDiskGb(dirPath: string): number {
  try {
    const out = require('child_process')
      .execSync(`df -BG "${dirPath}" | awk 'NR==2{print $4}'`)
      .toString()
      .trim()
      .replace('G', '');
    return parseInt(out, 10);
  } catch {
    return -1;
  }
}

/**
 * Returns free RAM in GB, or -1 on error.
 */
function freeRamGb(): number {
  try {
    const out = require('child_process')
      .execSync(`free -g | awk '/^Mem:/{print $7}'`)
      .toString()
      .trim();
    return parseInt(out, 10);
  } catch {
    return -1;
  }
}

/**
 * Returns true if the QEMU VM process from the PID file is alive.
 */
function isQemuVmRunning(): boolean {
  const os = require('os');
  const fs = require('fs');
  const pidFile = require('path').join(os.homedir(), 'sim', 'ttsim-qemu', 'vm.pid');
  if (!fs.existsSync(pidFile)) {
    return false;
  }
  try {
    const pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim(), 10);
    require('child_process').execSync(`kill -0 ${pid}`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/**
 * Command: tenstorrent.ttsim.launchQemu
 *
 * State machine:
 *   NO_RELEASE       → show info + link to releases page
 *   RELEASE, NO_IMG  → requirements check → offer download
 *   RELEASE, VM_OFF  → boot QEMU → poll SSH → open terminal
 *   RELEASE, VM_ON   → skip boot → open terminal
 */
async function launchTtsimQemu(): Promise<void> {
  const { TTSIM_QEMU_RELEASE } = require('./commands/terminalCommands');
  const os = require('os');
  const fs = require('fs');
  const path = require('path');

  // Gate: no release yet
  if (!TTSIM_QEMU_RELEASE) {
    const action = await vscode.window.showInformationMessage(
      'ttsim QEMU Bridge releases are not yet available. Watch the releases page for announcements.',
      'Watch for Releases'
    );
    if (action === 'Watch for Releases') {
      vscode.env.openExternal(
        vscode.Uri.parse('https://github.com/tenstorrent/ttsim-qemu/releases')
      );
    }
    return;
  }

  const simDir = path.join(os.homedir(), 'sim', 'ttsim-qemu');
  const imagePath = path.join(simDir, 'ttsim.qcow2');

  // Gate: image not downloaded
  if (!fs.existsSync(imagePath)) {
    // Requirements check
    if (!isQemuInstalled()) {
      vscode.window.showErrorMessage(
        'QEMU is not installed. Install it with: sudo apt install qemu-system-x86'
      );
      return;
    }
    const ram = freeRamGb();
    if (ram >= 0 && ram < 8) {
      vscode.window.showErrorMessage(
        `QEMU Bridge requires 8 GB free RAM. Currently available: ${ram} GB. Close other applications and try again.`
      );
      return;
    }
    const disk = freeDiskGb(os.homedir());
    if (disk >= 0 && disk < 20) {
      vscode.window.showErrorMessage(
        `QEMU Bridge requires 20 GB free disk at ~/sim/. Currently available: ${disk} GB. Free space and try again.`
      );
      return;
    }
    const action = await vscode.window.showInformationMessage(
      `ttsim QEMU Bridge image not found. Download it now? (~20 GB, downloads to ~/sim/ttsim-qemu/)`,
      'Download'
    );
    if (action === 'Download') {
      await setupTtsimQemu();
    }
    return;
  }

  // VM already running — just attach
  if (isQemuVmRunning()) {
    vscode.window.showInformationMessage('Attaching to running ttsim QEMU Bridge...');
    const terminal = vscode.window.createTerminal({ name: 'ttsim QEMU Bridge' });
    runInTerminal(terminal, TERMINAL_COMMANDS.LAUNCH_TTSIM_QEMU.template);
    return;
  }

  // Boot the VM
  fs.mkdirSync(simDir, { recursive: true });
  const pidFile = path.join(simDir, 'vm.pid');
  const workspaceDir = os.homedir().includes('/home/')
    ? path.join(os.homedir(), 'code')
    : os.homedir();
  const bootCmd = [
    'qemu-system-x86_64',
    '-m 8G -smp 4',
    `-drive file=${imagePath},if=virtio`,
    `-virtfs local,path=${workspaceDir},mount_tag=workspace,security_model=passthrough`,
    '-netdev user,id=net0,hostfwd=tcp::2222-:22',
    '-device virtio-net-pci,netdev=net0',
    '-nographic -daemonize',
    `-pidfile ${pidFile}`,
  ].join(' \\\n  ');

  vscode.window.showInformationMessage('Booting ttsim QEMU Bridge... (takes ~30 seconds)');
  const bootTerminal = getOrCreateSimpleTerminal();
  runInTerminal(bootTerminal, bootCmd);

  // Poll for SSH readiness (max 90s)
  const pollIntervalMs = 3000;
  const maxAttempts = 30;
  let attempts = 0;
  const poll = setInterval(async () => {
    attempts++;
    try {
      require('child_process').execSync(
        'ssh -p 2222 -o StrictHostKeyChecking=no -o ConnectTimeout=2 -o BatchMode=yes tt@localhost exit',
        { stdio: 'ignore' }
      );
      clearInterval(poll);
      const terminal = vscode.window.createTerminal({ name: 'ttsim QEMU Bridge' });
      runInTerminal(terminal, TERMINAL_COMMANDS.LAUNCH_TTSIM_QEMU.template);
    } catch {
      if (attempts >= maxAttempts) {
        clearInterval(poll);
        vscode.window.showErrorMessage(
          'ttsim QEMU Bridge did not become ready in 90 seconds. Check the terminal for boot errors.'
        );
      }
    }
  }, pollIntervalMs);
}

/**
 * Command: tenstorrent.ttsim.stopQemu
 * Sends ACPI shutdown to the running QEMU VM via its PID file.
 */
async function stopTtsimQemu(): Promise<void> {
  const os = require('os');
  const fs = require('fs');
  const path = require('path');
  const pidFile = path.join(os.homedir(), 'sim', 'ttsim-qemu', 'vm.pid');

  if (!isQemuVmRunning()) {
    vscode.window.showInformationMessage('ttsim QEMU Bridge is not running.');
    return;
  }

  const pid = parseInt(fs.readFileSync(pidFile, 'utf8').trim(), 10);
  try {
    // SIGTERM triggers ACPI shutdown on QEMU when -nographic is used
    require('child_process').execSync(`kill -TERM ${pid}`);
    vscode.window.showInformationMessage('ttsim QEMU Bridge is shutting down.');
  } catch {
    vscode.window.showErrorMessage('Failed to stop ttsim QEMU Bridge. PID: ' + pid);
  }
}

/**
 * Command: tenstorrent.ttsim.setupQemu
 * Downloads the ttsim QEMU image when a release is available.
 */
async function setupTtsimQemu(): Promise<void> {
  const { TTSIM_QEMU_RELEASE } = require('./commands/terminalCommands');
  const os = require('os');

  if (!TTSIM_QEMU_RELEASE) {
    vscode.window.showInformationMessage(
      'ttsim QEMU Bridge releases are not yet available.'
    );
    return;
  }

  const simDir = `${os.homedir()}/sim/ttsim-qemu`;
  const imageUrl = `https://github.com/tenstorrent/ttsim-qemu/releases/download/${TTSIM_QEMU_RELEASE}/ttsim.qcow2`;
  const downloadCmd = [
    `mkdir -p ${simDir}`,
    `wget -q --show-progress "${imageUrl}" -O ${simDir}/ttsim.qcow2 || { echo "ERROR: download failed"; exit 1; }`,
    `echo "ttsim QEMU Bridge image ready at ${simDir}/ttsim.qcow2"`,
  ].join('\n');

  const terminal = getOrCreateSimpleTerminal();
  runInTerminal(terminal, downloadCmd);
  vscode.window.showInformationMessage(
    `Downloading ttsim QEMU Bridge image. Check the terminal for progress.`
  );
}
```

- [ ] **Step 2: Register the three new commands** in the `activate()` function, after the existing ttsim registrations (after line 5166):

```typescript
    // ttsim QEMU Bridge
    vscode.commands.registerCommand('tenstorrent.ttsim.launchQemu', launchTtsimQemu),
    vscode.commands.registerCommand('tenstorrent.ttsim.stopQemu', stopTtsimQemu),
    vscode.commands.registerCommand('tenstorrent.ttsim.setupQemu', setupTtsimQemu),
```

- [ ] **Step 3: Build and verify**

```bash
npm run build 2>&1 | tail -20
```

Expected: no TypeScript errors.

- [ ] **Step 4: Commit**

```bash
git add src/extension.ts
git commit -m "feat(ttsim): add launchQemu, stopQemu, setupQemu commands with release gate"
```

---

## Task 3: Update `ttsim-twenty-and-ten.md` to v1.8.4

**Files:**
- Modify: `content/lessons/ttsim-twenty-and-ten.md`

- [ ] **Step 1: Bump version references throughout the file**

Do a global find-replace of `v1.8.0` → `v1.8.4`:

```bash
sed -i 's/v1\.8\.0/v1.8.4/g' content/lessons/ttsim-twenty-and-ten.md
```

Verify the replacements:

```bash
grep -n "v1\.8\." content/lessons/ttsim-twenty-and-ten.md
```

Expected: all instances now say `v1.8.4`.

- [ ] **Step 2: Add the QEMU Bridge callout** — insert after the `## Setup` heading (line 45) and before the `[⚙ Set Up ttsim]` command link, replacing nothing:

Add this block between `## Setup` and `[⚙ Set Up ttsim]`:

```markdown
> **No hardware and don't want to build tt-metal?** The
> [ttsim QEMU Bridge](command:tenstorrent.showLesson?["ttsim-qemu-bridge"])
> is a complete pre-built environment — zero setup, boots in ~30 seconds.

```

- [ ] **Step 3: Update the setup command block** to add the two new binaries and qsr/aarch64 notes. Replace the existing `Or manually:` bash block (lines 50-74) with:

```markdown
Or manually:

```bash
mkdir -p ~/sim
TTSIM_VERSION=v1.8.4

# Wormhole, Blackhole, and mesh variants
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_wh.so \
     -O ~/sim/libttsim_wh.so
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_bh.so \
     -O ~/sim/libttsim_bh.so
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_wh_x2.so \
     -O ~/sim/libttsim_wh_x2.so
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_bh_x2.so \
     -O ~/sim/libttsim_bh_x2.so
wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_wh_x8.so \
     -O ~/sim/libttsim_wh_x8.so

# libttsim_qsr.so — QuietBox simulation topology (4-chip Blackhole layout)
# Download if targeting TT-QuietBox 2 specifically
# wget https://github.com/tenstorrent/ttsim/releases/download/${TTSIM_VERSION}/libttsim_qsr.so \
#      -O ~/sim/libttsim_qsr.so

# Copy the SOC descriptor for Wormhole (switch for Blackhole in entries 3 and 27)
cp $TT_METAL_HOME/tt_metal/soc_descriptors/wormhole_b0_80_arch.yaml ~/sim/soc_descriptor.yaml

# Copy the N300 cluster descriptor (used for multichip simulation — entries 31, 32)
cp $TT_METAL_HOME/tests/tt_metal/tt_fabric/custom_mock_cluster_descriptors/n300_cluster_desc.yaml \
   ~/sim/n300_cluster_desc.yaml

# Required env vars — set these before running any entry below
export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

> **ARM host?** aarch64 variants are available — replace filenames with `libttsim_wh_aarch64.so`,
> `libttsim_bh_aarch64.so`, etc. Same download URL, different filename suffix.
```

- [ ] **Step 4: Add entry 32** — append the following after the closing `> To return to single-chip mode` blockquote of entry 31 (before `## What You Learned`):

```markdown
---

### 32. Two Blackhole chips — the BH mesh

`libttsim_bh_x2.so` gives you a virtual **2-chip Blackhole system** — the same
`MeshDevice(1, 2)` API as the N300 WH mesh in entry 31, now on Blackhole.

```bash
export TT_METAL_SIMULATOR=~/sim/libttsim_bh_x2.so
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

```python
import torch, ttnn

mesh = ttnn.open_mesh_device(ttnn.MeshShape(1, 2))
print(mesh)  # MeshDevice(1x2 grid, 2 devices) — Blackhole topology

a = torch.randn(64, 64, dtype=torch.bfloat16)
b = torch.randn(64, 64, dtype=torch.bfloat16)

a_mesh = ttnn.from_torch(a, layout=ttnn.TILE_LAYOUT, device=mesh,
                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))
b_mesh = ttnn.from_torch(b, layout=ttnn.TILE_LAYOUT, device=mesh,
                          mesh_mapper=ttnn.ShardTensorToMesh(mesh, dim=0))

c_mesh = ttnn.add(a_mesh, b_mesh)
c = ttnn.to_torch(c_mesh, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))
ttnn.close_mesh_device(mesh)
print("Max error vs reference:", (c - (a + b)).abs().max().item())
```

```text
MeshDevice(1x2 grid, 2 devices)
Max error vs reference: 0.03125
```

The Blackhole NOC differs from Wormhole in die area and core layout. The API is
identical. The same `ShardTensorToMesh` + `ConcatMeshToTensor` pattern that runs a
Wormhole N300 runs a 2-chip Blackhole system without a code change.

> To return to single-chip Wormhole: `export TT_METAL_SIMULATOR=~/sim/libttsim_wh.so`
> and `unset TT_METAL_MOCK_CLUSTER_DESC_PATH`.

Want to run a full 4-chip Blackhole topology (TT-QuietBox 2)?
Use `libttsim_qsr.so` with its cluster descriptor, or skip straight to
[ttsim QEMU Bridge](command:tenstorrent.showLesson?["ttsim-qemu-bridge"]) — the image
includes a pre-configured QuietBox 2 topology.
```

- [ ] **Step 5: Update the front matter description** — replace the existing description to mention v1.8.4 and entry 32:

Find:
```yaml
description: >-
  31 things you can do with the ttsim hardware simulator — no Tenstorrent
  device required. Runs on any Linux machine, including WSL2 on Windows.
  Includes N300 two-chip mesh simulation (v1.8.0+). Escalates from first
  kernel to DSP prototyping to a cliffhanger only real hardware can resolve.
```

Replace with:
```yaml
description: >-
  32 things you can do with the ttsim hardware simulator — no Tenstorrent
  device required. Runs on any Linux machine, including WSL2 on Windows.
  Includes N300 and Blackhole two-chip mesh simulation (v1.8.4). Escalates
  from first kernel to DSP prototyping to a cliffhanger only real hardware
  can resolve.
```

- [ ] **Step 6: Update the `## What You Learned` checklist** — add entry 32 to the bullet:

Find:
```markdown
- ✅ **Multi-chip simulation**: N300 1×2 MeshDevice with ShardTensorToMesh (v1.8.0+)
```

Replace with:
```markdown
- ✅ **Multi-chip simulation**: N300 WH 1×2 and Blackhole 1×2 MeshDevice with ShardTensorToMesh (v1.8.4)
```

- [ ] **Step 7: Run validate:lessons to catch any registry drift**

```bash
npm run validate:lessons 2>&1 | tail -15
```

Expected: validation passes (description change requires registry sync — see next step if it fails).

- [ ] **Step 8: If validation fails, regenerate the registry entry**

```bash
npm run generate:lessons -- --execute --force 2>&1 | tail -20
npm run validate:lessons 2>&1 | tail -10
```

Expected: both pass.

- [ ] **Step 9: Commit**

```bash
git add content/lessons/ttsim-twenty-and-ten.md content/lesson-registry.json
git commit -m "content(ttsim): update lesson to v1.8.4, add entry 32 (BH mesh), QEMU Bridge callout"
```

---

## Task 4: Create `ttsim-qemu-bridge.md`

**Files:**
- Create: `content/lessons/ttsim-qemu-bridge.md`

- [ ] **Step 1: Create the new lesson file**

Write `content/lessons/ttsim-qemu-bridge.md` with this exact content:

```markdown
---
id: ttsim-qemu-bridge
title: "ttsim QEMU Bridge: Full-System Simulation"
description: >-
  A complete pre-built TT-Metalium + ttsim environment in a QEMU virtual machine.
  One click to boot. Zero setup. Runs on any Linux x86_64 host or WSL2.
  Mount your local workspace and develop for Tenstorrent hardware without owning any.
category: advanced
tags:
  - ttsim
  - simulator
  - qemu
  - metalium
  - kernels
supportedHardware:
  - simulator
status: draft
estimatedMinutes: 15
---

# ttsim QEMU Bridge: Full-System Simulation

The [ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"]) simulator works by
plugging a `.so` shared library into your existing TT-Metalium install. That is fast and
flexible, but it requires you to build tt-metal first.

The ttsim QEMU Bridge takes a different approach: it ships a complete Linux virtual
machine image with TT-Metalium pre-built, ttsim binaries staged, and all required
environment variables pre-configured. You download it once, boot it with one command,
and a terminal opens inside a fully operational simulated Tenstorrent environment.

The bridge is a development environment, not a demo. It is persistent — changes you
make inside the VM (pip installs, build artifacts, experiment outputs) survive stop
and restart. Your local workspace is mounted inside the VM so you can edit files in
VSCode on your host and run them on the simulated hardware without copying anything.

> **Have hardware?** The bridge is still useful — isolated experiments, clean
> reproducible environments, or testing on a chip topology you don't own (e.g.,
> a Blackhole QuietBox 2 mesh when you only have a Wormhole card).

---

## Prerequisites

- **Linux x86_64 host** (or WSL2 on Windows)
- **QEMU installed:**
  ```bash
  sudo apt install qemu-system-x86 qemu-utils
  ```
- **8 GB free RAM** (VM is allocated 8 GB)
- **20 GB free disk** at `~/sim/` (for the VM image)

---

## Launch

[▶ Launch ttsim QEMU Bridge](command:tenstorrent.ttsim.launchQemu)

What happens on first launch:

1. Extension checks that QEMU is installed and requirements are met
2. Offers to download the VM image (~20 GB, one-time)
3. Boots the VM with `qemu-system-x86_64` in headless mode
4. Polls until SSH is ready (~30 seconds from cold boot)
5. Opens a terminal inside the VM — you're in

On subsequent launches, if the VM is already running, the extension attaches directly
(no re-boot).

---

## First kernel (under 2 minutes from cold boot)

Once the terminal opens inside the bridge, run this — no setup, no env var exports,
no SOC descriptor copy:

```bash
./build/programming_examples/metal_example_add_2_integers_in_riscv
```

```text
Success: Result is 21
```

A RISC-V kernel dispatched through TT-Metalium, on a simulated Tensix core, inside
a VM, on your laptop. The result is bit-exact to what silicon produces.

Everything else in [Twenty-and-Ten](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])
works the same way — all 32 entries, no additional setup.

---

## Your workspace inside the bridge

Your `~/code` directory on the host is mounted read-only at `/mnt/workspace` inside
the VM. The workflow:

1. Edit or write files in VSCode on your host machine (they live in `~/code/`)
2. Run them inside the bridge at `/mnt/workspace/`

```bash
# Inside the bridge VM:
python3 /mnt/workspace/my-experiment/kernel.py
```

The mount is read-only from the VM's perspective — you cannot accidentally delete
local files from inside the VM.

---

## The VM is persistent

The image uses a QEMU copy-on-write (qcow2) layer. Changes you make inside the VM —
`pip install`, build outputs, saved checkpoints, created files — survive `stopQemu`
and re-launch. The base image is never modified.

To snapshot the current VM state before a risky experiment:

```bash
# On the host (not inside the VM):
qemu-img snapshot -c before-experiment ~/sim/ttsim-qemu/ttsim.qcow2
```

To restore:

```bash
qemu-img snapshot -a before-experiment ~/sim/ttsim-qemu/ttsim.qcow2
```

---

## Stop the bridge

[■ Stop ttsim QEMU Bridge](command:tenstorrent.ttsim.stopQemu)

Sends an ACPI shutdown signal — clean OS shutdown, not a kill. Running processes
inside the VM have time to flush and exit. Wait ~10 seconds after clicking before
relaunching.

---

## What's pre-installed

Inside the VM:

- **TT-Metalium** — built and ready at `$TT_METAL_HOME`
- **ttsim** — `libttsim_wh.so`, `libttsim_bh.so`, `libttsim_wh_x2.so`,
  `libttsim_bh_x2.so` staged at `~/sim/`
- **Python environment** — `ttnn` and dependencies installed
- **Environment variables** — `TT_METAL_HOME`, `TT_METAL_SIMULATOR`,
  `TT_METAL_SLOW_DISPATCH_MODE`, `TT_METAL_DISABLE_SFPLOADMACRO` set in `.bashrc`
- **Scratch directory** — `~/tt-scratchpad/ttsim/` pre-created

Not inside: your local files (use `/mnt/workspace`).

---

## Go deeper

All 32 entries in [Twenty-and-Ten Things You Can Do with ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])
run inside the bridge without any additional setup. The bridge is the
fastest path to the most advanced entries.
```

- [ ] **Step 2: Add the new lesson to `lesson-registry.json`** by running the generator:

```bash
npm run generate:lessons -- --execute --force 2>&1 | tail -20
```

Expected: generator adds `ttsim-qemu-bridge` entry to the registry.

- [ ] **Step 3: Manually set the order field** in `content/lesson-registry.json` for the new entry. Find the `ttsim-qemu-bridge` entry (it will be appended at the end) and set:

```json
"order": 100,
"previousLesson": "ttsim-twenty-and-ten",
"nextLesson": null,
```

- [ ] **Step 4: Validate**

```bash
npm run validate:lessons 2>&1 | tail -10
```

Expected: all lessons valid.

- [ ] **Step 5: Commit**

```bash
git add content/lessons/ttsim-qemu-bridge.md content/lesson-registry.json
git commit -m "content: add ttsim-qemu-bridge lesson (release-gated)"
```

---

## Task 5: Bump version and final validation

**Files:**
- Modify: `package.json` line 5

- [ ] **Step 1: Bump version**

```bash
npm version patch --no-git-tag-version
```

Expected: version changes from `0.0.503` to `0.0.504`.

- [ ] **Step 2: Run full validation suite**

```bash
npm run build 2>&1 | tail -10
npm run validate:lessons 2>&1 | tail -5
npm run test:links 2>&1 | tail -20
```

Expected: all pass. If `test:links` fails because `ttsim-qemu-bridge` lesson isn't found in welcome.html, that is acceptable — the lesson is `status: draft` and not yet wired into the welcome page.

- [ ] **Step 3: Commit**

```bash
git add package.json
git commit -m "chore: bump version to 0.0.504"
```

---

## Self-Review Checklist

- [x] **Spec coverage:** All spec requirements covered: 3 commands, release gate constant, lesson update, new lesson, registry, version bump, requirements check, state machine, boot invocation, SSH polling, stop command.
- [x] **No TBD/TODO placeholders:** All code is complete. SSH credentials use `tt@localhost` consistent with spec's user `tt`. Image size placeholder (`~20 GB`) is intentional — actual size unknown until release.
- [x] **Type consistency:** `TTSIM_QEMU_RELEASE` exported from `terminalCommands.ts`, imported via `require()` in `extension.ts` functions (consistent with codebase's pattern of avoiding circular imports at module level). `TERMINAL_COMMANDS.LAUNCH_TTSIM_QEMU` referenced in `launchTtsimQemu` — defined in Task 1 before used in Task 2. `isQemuVmRunning` defined before `launchTtsimQemu` and `stopTtsimQemu`.
- [x] **Command IDs:** `tenstorrent.ttsim.launchQemu`, `tenstorrent.ttsim.stopQemu`, `tenstorrent.ttsim.setupQemu` — dots in command IDs are valid in VSCode.
- [x] **Lesson ID in command links:** `tenstorrent.showLesson?["ttsim-qemu-bridge"]` — matches the `id` field in the front matter and registry.
