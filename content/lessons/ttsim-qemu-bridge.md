---
id: ttsim-qemu-bridge
title: "ttsim QEMU Bridge: Full-System Simulation"
description: >-
  Run TT-Metalium kernels on a virtual Tenstorrent PCI device inside a QEMU VM.
  The VM sees real Wormhole hardware — no TT_METAL_SIMULATOR env var needed.
  Bring your own Ubuntu image, boot with one command, develop with pip install ttnn.
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
estimatedMinutes: 20
---

# ttsim QEMU Bridge: Full-System Simulation

The [ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"]) simulator ships as
a `.so` shared library that TT-Metalium loads on the host when you set
`TT_METAL_SIMULATOR`. That requires a working TT-Metalium install on your machine.

The ttsim QEMU Bridge is different: it adds a virtual Tenstorrent PCI device to any
Linux VM. The guest OS sees vendor `0x1e52` — a real Wormhole chip from its perspective.
TT-Metalium inside the VM talks to it exactly as it would talk to silicon. No
`TT_METAL_SIMULATOR` env var. No special paths. Just `pip install ttnn` and go.

`libttsim_wh.so` runs on the host. QEMU bridges it to the guest via the PCI device at
boot time (`-device ttsim,lib=...`). The split is clean: QEMU handles the translation;
the guest is none the wiser.

> **ttsim-qemu** is a [QEMU fork](https://github.com/tenstorrent/ttsim-qemu) — a single
> patch on top of upstream QEMU `stable-11.0`. The system QEMU does not have the
> `ttsim` device. You must build or install the fork.

---

## Important constraint: slow dispatch only

`libttsim` does not yet implement fast dispatch. Inside the VM you must set:

```bash
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

This makes the VM a **kernel development and learning environment** — not a production
inference environment. LLM serving, training runs, and image generation are too slow
to be practical. What works well:

- TTNN operations and kernel authoring
- All entries in [Twenty-and-Ten Things You Can Do with ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])
- The [CS Fundamentals](command:tenstorrent.showLesson?["cs-fundamentals-01-computer"]) series
- Cookbook examples: [Game of Life](command:tenstorrent.showLesson?["cookbook-game-of-life"]),
  [Particle Life](command:tenstorrent.showLesson?["cookbook-particle-life"])

---

## Prerequisites

- **ttsim-qemu fork built from source:**
  ```bash
  git clone -b stable-11.0-ttsim --depth=1 https://github.com/tenstorrent/ttsim-qemu
  cd ttsim-qemu
  mkdir build && cd build
  ../configure --target-list=x86_64-softmmu --prefix=$HOME/.local --disable-docs
  ninja -j$(nproc)
  ninja install
  ```
  Adds `qemu-system-x86_64` with the `ttsim` device to `~/.local/bin/`.

- **libttsim_wh.so** in `~/sim/` (run Setup ttsim from the
  [ttsim lesson](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"]) first):

  [▶ Setup ttsim](command:tenstorrent.setupTtsim)

- **Ubuntu 24.04 minimal cloud image** (~600 MB) — the Launch command offers to
  download it automatically.

- **8 GB free RAM**, **2 GB free disk** at `~/sim/`

---

## Launch

[▶ Launch ttsim QEMU Bridge](command:tenstorrent.ttsim.launchQemu)

What happens:

1. Checks that the ttsim-qemu fork is on PATH and `libttsim_wh.so` exists
2. If no VM image found, offers to download Ubuntu 24.04 (~600 MB, one-time)
3. Boots the VM with `-device ttsim,lib=~/sim/libttsim_wh.so`
4. Polls until SSH is ready on port 2222 (~30–60 seconds)
5. Opens a terminal inside the VM via SSH — you're in

To boot manually (e.g. to customise RAM/CPU):

```bash
# Create a cloud-init seed ISO first (once) to inject your SSH key:
cloud-localds "$HOME/sim/ttsim-qemu/seed.iso" \
  <(echo "#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)")

qemu-system-x86_64 \
  -m 8G -smp 4 \
  -drive file="$HOME/sim/ttsim-qemu/ubuntu.qcow2",if=virtio,snapshot=on \
  -drive file="$HOME/sim/ttsim-qemu/seed.iso",if=virtio,format=raw,readonly=on \
  -device ttsim,lib="$HOME/sim/libttsim_wh.so" \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -serial file:/tmp/ttsim-qemu-serial.log \
  -chardev socket,id=mon,path=/tmp/ttsim-mon.sock,server=on,wait=off \
  -mon chardev=mon,mode=readline \
  -display none -daemonize \
  -pidfile "$HOME/sim/ttsim-qemu/vm.pid"
```

Then SSH in: `ssh -p 2222 -o StrictHostKeyChecking=no ubuntu@localhost`

---

## First steps inside the VM

> **TT-Metal version matching required.** The `pip install ttnn` pre-built wheels are
> built against specific UMD versions. If the wheel version doesn't match the `tt-kmd`
> driver ABI, `open_device()` may crash with SIGILL during topology discovery. To avoid
> this, build tt-metal from source inside the VM against the same kernel driver, or use
> the same ttnn wheel version that was validated with your `tt-kmd` build.
>
> The alternative is the host `TT_METAL_SIMULATOR` path — set
> `TT_METAL_SIMULATOR=~/sim/libttsim_wh.so` on the host and run tt-metal there. All
> ttsim-twenty-and-ten entries work this way without the QEMU VM.

```bash
# Install the Tenstorrent kernel driver (must match ttnn wheel ABI)
sudo apt-get install -y linux-headers-$(uname -r)
# Build and load tt-kmd from source (see tenstorrent/tt-kmd on GitHub)
# Or use DKMS if your distribution packages it

# Confirm the device node is present
ls /dev/tenstorrent/

# Set required env vars
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

Verify the PCI device is recognised by the driver:

```bash
lspci -k | grep -A2 Tenstorrent
# 00:03.0 Processing accelerators: Tenstorrent Inc Wormhole (rev 01)
#         Kernel driver in use: tenstorrent
```

Run a sanity check once tt-metal is installed inside the VM:

```python
import ttnn
device = ttnn.open_device(device_id=0)
print(device)
ttnn.close_device(device)
```

```text
MeshDevice(1x1 grid, 1 devices)
```

The device is the virtual Wormhole PCI card (0x1e52:0x401e).

---

## Verify the PCI device

Inside the VM:

```bash
lspci | grep -i tenstorrent
# 00:04.0 Class 4608: 1e52:401e
```

Three BARs are mapped (512 MB registers, 1 MB config space, 32 GB DRAM window) — the
same layout as physical Wormhole silicon.

---

## Your local files inside the VM

Pass `~/code` as a virtfs mount to access your host workspace read-only inside the VM:

```bash
# Add to the boot command above:
-virtfs local,path="$HOME/code",mount_tag=workspace,security_model=passthrough
```

Inside the VM:

```bash
sudo mkdir -p /mnt/workspace
sudo mount -t 9p -o trans=virtio,version=9p2000.L workspace /mnt/workspace
```

Edit in VSCode on the host, run at `/mnt/workspace/` inside the VM.

---

## The VM is persistent

Boot with `snapshot=on` (shown above) for ephemeral sessions — the image is never
modified and boots clean every time. Remove `snapshot=on` to persist changes across
reboots (pip installs, built artifacts, etc.).

To snapshot a persistent image before a risky experiment:

```bash
# On the host:
qemu-img snapshot -c before-experiment ~/sim/ttsim-qemu/ubuntu.qcow2
# Restore:
qemu-img snapshot -a before-experiment ~/sim/ttsim-qemu/ubuntu.qcow2
```

---

## Stop the bridge

[■ Stop ttsim QEMU Bridge](command:tenstorrent.ttsim.stopQemu)

Sends a clean shutdown via the QEMU monitor socket. Save work inside the VM first —
any in-progress writes may not flush if the process exits abruptly.

---

## Go deeper

All single-chip Wormhole entries in
[Twenty-and-Ten Things You Can Do with ttsim](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])
run inside this VM without any additional setup beyond `pip install ttnn`. That's
entries 1–17 and 22–30 — matrix ops, data types, convolutions, reductions, multi-core
dispatch, NoC transfers, and more.

The [CS Fundamentals](command:tenstorrent.showLesson?["cs-fundamentals-01-computer"]) series
walks through computer architecture concepts directly on simulated Tensix cores —
the QEMU VM makes a clean isolated environment for those experiments.
