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

## Current status: QEMU PCI path requires matching ttsim + tt-kmd versions

> **⚠️ Note (as of ttsim v1.9.1 + ttnn 0.72.0):** The QEMU PCI device path is
> architecture-validated but the full `ttnn.open_device()` flow is not yet working
> end-to-end. Two compatibility constraints apply:
>
> - **tt-kmd version:** Use `ttkmd-2.3.0` or earlier. Versions ≥ 2.4.0 probe the
>   `ARC_MSG_QCB_PTR` register (`RESET_UNIT + 0x1D8`) during `insmod`. ttsim v1.9.1
>   does not implement this reset unit register and aborts with
>   `UnimplementedFunctionality: arc_reset_unit_rd32: offset=0x1d8`.
>
> - **UMD/ttnn ARC tile reads:** ttnn ≥ 0.69.0 UMD reads the ARC tile register at
>   NOC address `0x10000108` (via TLB window) during topology discovery. ttsim does
>   not yet implement this register, causing a hard abort with
>   `UnimplementedFunctionality: arc_tile_rd_bytes: arc: addr=0x10000108`.
>
> Until ttsim adds these register stubs, the recommended path for development and
> learning remains the host `TT_METAL_SIMULATOR` approach — which works completely
> with ttsim v1.9.1 and ttnn 0.72.0.
>
> The QEMU PCI device itself enumerates correctly: the guest kernel detects
> `Tenstorrent Inc Wormhole (rev 01)` at `00:03.0`. The gap is in ttsim's ARC tile
> register coverage for UMD topology discovery.

For a fully working simulator path today, use the host
[TT_METAL_SIMULATOR approach](command:tenstorrent.showLesson?["ttsim-twenty-and-ten"])
and all entries in ttsim-twenty-and-ten will work without any VM setup.

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
3. Boots the VM with the correct flags (see below)
4. Polls until SSH is ready on port 2222 (~30–60 seconds in TCG mode)
5. Opens a terminal inside the VM via SSH — you're in

To boot manually:

```bash
# Create a cloud-init seed ISO first (once) to inject your SSH key:
cloud-localds "$HOME/sim/ttsim-qemu/seed.iso" \
  <(echo "#cloud-config
users:
  - name: ubuntu
    sudo: ALL=(ALL) NOPASSWD:ALL
    ssh_authorized_keys:
      - $(cat ~/.ssh/id_ed25519.pub 2>/dev/null || cat ~/.ssh/id_rsa.pub)")

# Boot in TCG (software emulation) mode — required for ttsim
# -cpu max: enable all CPU features without KVM
# bar4-size=32M: correct for Wormhole (Blackhole uses bar4-size=32G)
# No -enable-kvm: KVM intercepts MMIO at the EPT level and cannot emulate
#   16-byte SSE/AVX loads from WC-mapped TLB window pages.
qemu-system-x86_64 \
  -m 8G -smp 4 \
  -cpu max \
  -drive file="$HOME/sim/ttsim-qemu/ubuntu.qcow2",if=virtio,snapshot=on \
  -drive file="$HOME/sim/ttsim-qemu/seed.iso",if=virtio,format=raw,readonly=on \
  -device ttsim,lib="$HOME/sim/libttsim_wh.so",bar4-size=32M \
  -netdev user,id=net0,hostfwd=tcp::2222-:22 \
  -device virtio-net-pci,netdev=net0 \
  -serial file:/tmp/ttsim-qemu-serial.log \
  -chardev socket,id=mon,path=/tmp/ttsim-mon.sock,server=on,wait=off \
  -mon chardev=mon,mode=readline \
  -display none -daemonize \
  -pidfile "$HOME/sim/ttsim-qemu/vm.pid"
```

> **Why TCG (no `-enable-kvm`)?** KVM intercepts MMIO at the x86 EPT level. The
> UMD allocates TLB windows as WC (write-combining) mapped memory and reads from
> them using SSE/AVX instructions (`vmovups` — 16 bytes). KVM's MMIO emulator only
> handles 1–8 byte accesses; a 16-byte MMIO load causes a SIGILL. QEMU's TCG mode
> emulates every instruction in software, routes MMIO through its `MemoryRegionOps`
> (which respects `max_access_size=8` and splits naturally), and avoids this entirely.
>
> Boot time in TCG mode is ~30–60 seconds. Python execution inside the VM is slower
> than native — typically 3–5× — but adequate for kernel development and short
> experiments.

Then SSH in: `ssh -p 2222 -o StrictHostKeyChecking=no ubuntu@localhost`

---

## First steps inside the VM

### Kernel driver (tt-kmd)

The ttsim PCI device shows up as a standard PCIe device. You need the Tenstorrent
kernel driver to surface it as `/dev/tenstorrent/0`.

> **Version constraint:** Use tt-kmd ≤ v2.3.0. Newer versions probe `ARC_MSG_QCB_PTR`
> during `insmod`, which ttsim does not yet implement. tt-kmd v2.3.0 only reads
> `ARC_TELEMETRY_PTR` and `ARC_TELEMETRY_DATA` from the reset unit; ttsim handles
> those (returning 0, logging a telemetry timeout warning that is harmless).

```bash
# Install build dependencies
sudo apt-get install -y linux-headers-$(uname -r) build-essential git

# Build and load tt-kmd v2.3.0
git clone --depth=1 --branch ttkmd-2.3.0 https://github.com/tenstorrent/tt-kmd.git
cd tt-kmd
make -j$(nproc)
sudo insmod tenstorrent.ko

# Grant access
sudo chmod a+rw /dev/tenstorrent/0
```

Verify the PCI device is recognised by the driver:

```bash
lspci -k | grep -A2 Tenstorrent
# 00:03.0 Processing accelerators: Tenstorrent Inc Wormhole (rev 01)
#         Kernel driver in use: tenstorrent
```

### Python environment and ttnn

```bash
python3 -m venv ~/venv
~/venv/bin/pip install ttnn

# Required env vars
export TT_METAL_SLOW_DISPATCH_MODE=1
export TT_METAL_DISABLE_SFPLOADMACRO=1
```

### Sanity check

> **Note (as of ttnn 0.72.0 / ttsim v1.9.1):** `open_device()` will fail during
> topology discovery because UMD reads ARC tile register `0x10000108` (NOC_NODE_ID
> via TLB window) — a register not yet implemented in ttsim. Track
> [tenstorrent/ttsim](https://github.com/tenstorrent/ttsim) for updates.
>
> The PCI device enumerates correctly (`lspci` shows it, `insmod` succeeds) — the
> gap is in ARC tile register coverage inside ttsim for UMD topology discovery.

Once ttsim implements the missing ARC tile register stubs, this will work:

```python
import ttnn
device = ttnn.open_device(device_id=0)
print(device)
ttnn.close_device(device)
```

```text
MeshDevice(1x1 grid, 1 devices)
```

---

## Verify the PCI device

Inside the VM (works today, no ttsim update needed):

```bash
lspci | grep -i tenstorrent
# 00:03.0 Class 1200: 1e52:401e
```

Three BARs are mapped (512 MB registers, 1 MB config space, 32 MB DRAM window for
Wormhole) — the same layout as physical Wormhole silicon.

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
run on the **host simulator path** (`TT_METAL_SIMULATOR=~/sim/libttsim_wh.so`)
without any QEMU VM — entries 1–17 and 22–30. That path works completely today
with ttsim v1.9.1 and ttnn 0.72.0.

The QEMU VM path will unlock when ttsim implements the remaining ARC tile registers
needed by UMD topology discovery. Watch
[tenstorrent/ttsim releases](https://github.com/tenstorrent/ttsim/releases) for updates.

The [CS Fundamentals](command:tenstorrent.showLesson?["cs-fundamentals-01-computer"]) series
walks through computer architecture concepts directly on simulated Tensix cores.
