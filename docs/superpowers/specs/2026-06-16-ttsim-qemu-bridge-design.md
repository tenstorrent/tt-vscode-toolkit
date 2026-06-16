# ttsim QEMU Bridge — Design Spec
_2026-06-16_

## Summary

Add "ttsim QEMU Bridge" to the tt-vscode-toolkit: a one-click VSCode command that boots a
pre-configured QEMU VM containing TT-Metalium and ttsim, opens an integrated terminal inside it,
and mounts the user's local workspace. Simultaneously update the existing ttsim lesson to v1.8.4
and add a new `ttsim-qemu-bridge.md` walkthrough lesson.

---

## Scope

### In scope
- New lesson: `content/lessons/ttsim-qemu-bridge.md`
- Updated lesson: `content/lessons/ttsim-twenty-and-ten.md` (v1.8.0 → v1.8.4, new binaries, qsr note, aarch64 note, QEMU Bridge callout, entry 32)
- New commands: `tenstorrent.ttsim.launchQemu`, `tenstorrent.ttsim.stopQemu`, `tenstorrent.ttsim.setupQemu`
- New terminal command constants: `SETUP_TTSIM_QEMU`, `LAUNCH_TTSIM_QEMU` in `terminalCommands.ts`
- New QEMU launch/state logic in `extension.ts`
- `package.json`: new walkthrough step, version bump

### Out of scope
- Renaming existing `tenstorrent.setupTtsim` / `tenstorrent.runTtsimAttention` (deferred cleanup)
- Windows-native QEMU support (WSL2 path only for now)
- x32 mesh variants (`libttsim_wh_x32.so`, `libttsim_bh_x32.so`) — niche, deferred
- macOS support

---

## Command Namespace

New commands follow `tenstorrent.ttsim.*` pattern:

| Command | Description |
|---|---|
| `tenstorrent.ttsim.launchQemu` | Boot VM or attach to running VM, open integrated terminal |
| `tenstorrent.ttsim.stopQemu` | Graceful VM shutdown |
| `tenstorrent.ttsim.setupQemu` | Download QEMU image (future: once release exists) |

Existing commands (`tenstorrent.setupTtsim`, `tenstorrent.runTtsimAttention`) unchanged.

---

## Release Gate

A single constant in `terminalCommands.ts` controls whether QEMU features are active:

```typescript
const TTSIM_QEMU_RELEASE: string | null = null; // set to 'v1.0.0' when first release ships
```

When `null`: all three commands show an informational message with a link to the
`tenstorrent/ttsim-qemu` releases page. No download, no boot attempt. The lesson renders
fully but its launch button produces the friendly gate message.

When set to a version string: full flow activates. Image URL constructed as:
```
https://github.com/tenstorrent/ttsim-qemu/releases/download/${TTSIM_QEMU_RELEASE}/ttsim.qcow2
```

---

## QEMU Launch State Machine

`tenstorrent.ttsim.launchQemu` walks through these states in order:

```
NO_RELEASE
  → show info message: "ttsim QEMU Bridge releases not yet available"
  → button: "Watch for Releases" → opens github.com/tenstorrent/ttsim-qemu/releases

RELEASE_EXISTS, IMAGE_NOT_DOWNLOADED
  → requirements check (qemu-system-x86_64 present, ≥8 GB RAM, ≥20 GB disk)
  → if requirements fail: show actionable error per missing requirement
  → if ok: offer download (show image size, confirm dialog)
  → on confirm: tenstorrent.ttsim.setupQemu runs download + checksum verify

IMAGE_DOWNLOADED, VM_STOPPED
  → boot VM (see Boot Invocation below)
  → poll SSH on localhost:2222 until ready (timeout 60s)
  → open integrated terminal running: ssh -p 2222 -o StrictHostKeyChecking=no tt@localhost

IMAGE_DOWNLOADED, VM_RUNNING  (pid file exists + process alive)
  → skip boot
  → show status bar message: "Attaching to running ttsim QEMU Bridge..."
  → open integrated terminal (same SSH command)
```

VM running state detected via PID file at `~/sim/ttsim-qemu/vm.pid`. Process liveness
checked with `kill -0 <pid>`.

---

## Boot Invocation

```bash
qemu-system-x86_64 \
  -m 8G -smp 4 \
  -drive file=~/sim/ttsim-qemu/ttsim.qcow2,if=virtio \
  -virtfs local,path=$HOME/code,mount_tag=workspace,security_model=passthrough \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -nographic -daemonize \
  -pidfile ~/sim/ttsim-qemu/vm.pid
```

Image location: `~/sim/ttsim-qemu/ttsim.qcow2`
SSH credentials: user `tt`, no password (key or blank — TBD from image spec)
Workspace mount: `$HOME/code` → `/mnt/workspace` inside VM (read-only passthrough)

---

## Requirements Check

Before download or boot, verify:

| Check | Failure message |
|---|---|
| `which qemu-system-x86_64` exits 0 | "Install QEMU: `sudo apt install qemu-system-x86`" |
| Free RAM ≥ 8 GB | "QEMU Bridge requires 8 GB free RAM. Close other applications and try again." |
| Free disk at `~/sim/` ≥ 20 GB | "QEMU Bridge requires 20 GB free disk at ~/sim/. Free space and try again." |

Failure shows VSCode error notification with the message. No partial execution.

---

## Updated Lesson: ttsim-twenty-and-ten.md

Changes from current (v1.8.0 base):

1. **Version bump**: every `v1.8.0` reference → `v1.8.4`

2. **Setup command** downloads updated binary list:
   - Keep: `libttsim_wh.so`, `libttsim_bh.so`, `libttsim_wh_x2.so`
   - Add: `libttsim_bh_x2.so`, `libttsim_wh_x8.so`
   - Skip (for now): `libttsim_wh_x32.so`, `libttsim_bh_x32.so` (32-chip mesh, niche)
   - `libttsim_qsr.so`: mention in setup with note (see below)

3. **Top callout** added before the Setup section:
   > **No hardware and don't want to build tt-metal?** The
   > [ttsim QEMU Bridge](command:tenstorrent.showLesson?["ttsim-qemu-bridge"])
   > is a complete pre-built environment — zero setup, boots in ~30 seconds.

4. **qsr note** in setup (inline, not a new entry):
   ```
   # libttsim_qsr.so — QuietBox simulation topology
   # Download if targeting TT-QuietBox 2 layout specifically
   wget https://github.com/tenstorrent/ttsim/releases/download/v1.8.4/libttsim_qsr.so \
        -O ~/sim/libttsim_qsr.so
   ```
   Brief explanation: QSR = QuietBox topology simulator; 4-chip Blackhole system layout.

5. **aarch64 note** (inline callout after setup, not a full new section):
   > aarch64 variants (`libttsim_wh_aarch64.so`, etc.) are available for ARM hosts.
   > Replace the filename suffix in the download URLs above.

6. **Entry 32** added at end of "The Ten" section:
   - Title: "Two Blackhole chips — the BH mesh"
   - Uses `libttsim_bh_x2.so` + `bh_cluster_desc.yaml`
   - Mirrors entry 31's N300 story: same MeshDevice API, Blackhole topology
   - Closing note: points to the QEMU Bridge for running the full QuietBox 2 (4-chip) topology

---

## New Lesson: ttsim-qemu-bridge.md

Front matter:
```yaml
id: ttsim-qemu-bridge
title: "ttsim QEMU Bridge: Full-System Simulation"
description: >-
  A complete pre-built TT-Metalium + ttsim environment in a QEMU virtual machine.
  One click to boot. Zero setup. Runs on any Linux x86_64 host or WSL2.
  Mount your local workspace and develop for Tenstorrent hardware without owning any.
category: advanced
tags: [ttsim, simulator, qemu, metalium, kernels]
supportedHardware: [sim]
status: draft
estimatedMinutes: 15
```

Sections (in order):

1. **What the QEMU Bridge is** — 3 paragraphs distinguishing it from `libttsim.so`.
   The `.so` approach requires a working tt-metal install on your host.
   The QEMU Bridge ships everything pre-installed inside a virtual machine:
   tt-metal built, ttsim binaries staged, env vars configured in `.bashrc`.
   The bridge is a dev environment — persistent, mountable, stoppable, restartable.

2. **Prerequisites** — QEMU install one-liner, RAM/disk requirements, WSL2 note.

3. **Launch** — command button + "what happens" explanation (download once ~Xgb,
   boots in ~30s, SSH terminal opens inside VM). When release not yet available:
   button shows friendly gate message.

4. **First kernel (under 2 minutes from cold boot)** — entry 2 from Twenty-and-Ten,
   copy-pasted verbatim. No env var setup. Just run it. Expected output shown.

5. **Your workspace inside the bridge** — `~/code` is at `/mnt/workspace`. Workflow:
   edit in VSCode on host, run inside the bridge. Read-only mount means you can't
   accidentally delete local files from inside the VM.

6. **The VM is persistent** — qcow2 COW layer: changes inside the VM (pip installs,
   build artifacts, saved outputs) survive stop/start. Explain how to snapshot if
   desired (one-liner `qemu-img snapshot`).

7. **Stop the bridge** — command button + explanation (ACPI shutdown, not kill).

8. **What's pre-installed** — bullet list: TT-Metalium (built), ttsim wh + bh + wh_x2
   + bh_x2, Python env, all required env vars, `~/tt-scratchpad` directory.

9. **Go deeper** — link to ttsim-twenty-and-ten. All 31 entries run inside the bridge
   without any additional setup.

---

## package.json Changes

- New walkthrough step `ttsim-qemu-bridge` after existing `ttsim-twenty-and-ten` step
- Version bump (PATCH)

---

## File Changelist

| File | Change |
|---|---|
| `content/lessons/ttsim-twenty-and-ten.md` | Update v1.8.0→v1.8.4, binaries, qsr, aarch64, callout, entry 32 |
| `content/lessons/ttsim-qemu-bridge.md` | New file |
| `src/commands/terminalCommands.ts` | Add `TTSIM_QEMU_RELEASE`, `SETUP_TTSIM_QEMU`, `LAUNCH_TTSIM_QEMU` |
| `src/extension.ts` | Add `launchTtsimQemu`, `stopTtsimQemu`, `setupTtsimQemu` functions + register |
| `package.json` | New walkthrough step, version bump |

---

## Open Questions (to resolve from image spec when release ships)

- SSH credentials inside the VM: key-based or blank password?
- Exact image size (for download confirmation dialog)
- Whether `bh_cluster_desc.yaml` ships inside the image or needs to be copied from `$TT_METAL_HOME`
- Whether the VM exposes any services beyond SSH (e.g., Jupyter, metrics endpoint)
