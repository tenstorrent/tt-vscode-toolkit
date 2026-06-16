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

Sends a SIGTERM to the QEMU process. The VM stops immediately — save your work inside
the VM before clicking. Persistent changes (files you wrote, packages you installed)
are preserved in the qcow2 layer and will be there on next launch.

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
