---
id: hardware-detection
title: Hardware Detection
description: >-
  Scan for connected Tenstorrent devices and verify they're properly recognized
  by the system.
category: first-inference
tags:
  - hardware
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - galaxy
status: validated
validatedOn:
  - n150
estimatedMinutes: 5
---

# Hardware Detection

Detect and verify your Tenstorrent hardware using the `tt-smi` command-line tool.

## Before You Begin: Is Your System Already Set Up?

**If you're using:**
- ✅ **Tenstorrent Cloud** - Your environment is pre-configured
- ✅ **Quietbox with preinstalled image** - tt-smi and drivers are already installed
- ✅ **Managed system** - Your sysadmin likely ran tt-installer already

**You can skip directly to running tt-smi below!**

### Need to Install? Use tt-installer 2.0

If `tt-smi` is not found on your system, the fastest way to set up is with **tt-installer 2.0** (recommended for 95% of users):

```bash
# One-command installation (5-15 minutes)
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

**What it installs:**
- ✅ tt-smi (this tool)
- ✅ Kernel drivers
- ✅ Firmware updates
- ✅ HugePages configuration
- ✅ tt-metalium containers (1GB standard, optional 10GB with demos)
- ✅ Python environment


Consult the [tt-installer](https://github.com/tenstorrent/tt-installer) repo for more information.

---

## What This Does

The `tt-smi` command scans your system for connected Tenstorrent devices and displays their status, including:
- Device model and ID
- PCIe information
- Temperature and power status
- Driver version

## Run the Command

This command will scan for connected Tenstorrent devices:

```bash
tt-smi
```

[🔍 Detect Tenstorrent Hardware](command:tenstorrent.runHardwareDetection)

## Understanding Your Hardware

After running `tt-smi`, you'll see information about your Tenstorrent hardware. To get structured output, use:

```bash
tt-smi -s
```

**Example output (JSON format):**
```json
{
  "board_info": {
    "board_type": "n150",
    "coords": "0,0"
  },
  "telemetry": {
    "voltage": 0.8,
    "current": 25.3,
    "power": 20.24,
    "temperature": 45.0
  }
}
```

### Hardware Types

Look for the `board_type` field to identify your hardware:

**Wormhole Family (2nd Generation):**
- **n150** - Single chip, 72 Tensix cores
  - Best for: Development, prototyping, single-user workloads
  - Context limit: 64K tokens for most models
  - Common in: Cloud instances, development workstations

- **n300** - Dual chip (2 chips), 144 Tensix cores
  - Best for: Higher throughput, longer context, multi-user serving
  - Context limit: 128K tokens
  - Tensor parallelism: TP=2 (uses both chips)

- **t3k** - Eight chips (8 chips), 576 Tensix cores
  - Best for: Large models (70B+), production serving
  - Context limit: 128K tokens
  - Tensor parallelism: TP=8 (uses all chips)

**Blackhole Family (Latest Generation):**
- **p100** - Single chip (cloud/standalone deployments)
  - Best for: Similar to N150 but with newer architecture
  - Context limit: 64K tokens
  - Status: Some models validated, others experimental

- **p150** - Dual chip (higher performance)
  - Best for: Similar to N300 but with improvements
  - Context limit: 128K tokens
  - Status: Check official documentation for validated configurations

- **p300/p300c** - Single chip (QuietBox variant)
  - Architecture: Blackhole (identical to P100)
  - Common in: Multi-device QuietBox Tower systems
  - MESH_DEVICE: Use P100 for single-chip lessons
  - Example: 4x P300c = 4 separate single-chip devices

**Blackhole Architecture Equivalence:**
All Blackhole cards (P100, P150, P300/P300c) share the same instruction set and capabilities. Lessons supporting P100 will work on P300/P300c without modification.

**QuietBox Multi-Device Detection:**
If you have a QuietBox Tower (4x P300c), `tt-smi` will show 4 devices:
```
Device 0: 0000:01:00.0 | P300c | FW 19.4.0.0
Device 1: 0000:02:00.0 | P300c | FW 19.4.0.0
Device 2: 0000:03:00.0 | P300c | FW 19.4.0.0
Device 3: 0000:04:00.0 | P300c | FW 19.4.0.0
```

Each device is a **separate single-chip Blackhole card**. For single-chip lessons, use device 0. For multi-chip lessons, all 4 devices are available for workload distribution.

### Quick Hardware Check

Extract just your hardware type:

```bash
tt-smi -s | grep -o '"board_type": "[^"]*"'
```

**Output:** `"board_type": "n150"`

**Save this for later lessons!** You'll need to know your hardware type when configuring models and servers.

---

## Expected Output

When you run `tt-smi`, you should see:

**For Wormhole hardware (N150, N300, T3K):**
```bash
Device 0: Wormhole
Board Type: n150
PCIe: Bus 0x01, Device 0x00
Firmware Version: 1.2.3
Temperature: 45°C
Power: 20W
```

**For Blackhole hardware (P100, P150):**
```text
Device 0: Blackhole
Board Type: p100
PCIe: Bus 0x01, Device 0x00
Firmware Version: 2.0.1
Temperature: 42°C
Power: 18W
```

**Multiple devices:**
```text
Device 0: Wormhole (n150)
Device 1: Wormhole (n150)
Device 2: Wormhole (n150)
Device 3: Wormhole (n150)
```

---

## Troubleshooting: No Hardware Detected

Don't worry if `tt-smi` doesn't detect your hardware immediately. You can usually fix this easily. Try these steps:

### Check 1: Hardware Connection

**Verify card is detected by PCIe:**
```bash
lspci | grep -i tenstorrent
```

**Expected output:**
```yaml
01:00.0 Processing accelerators: Tenstorrent Inc. Device [model]
```

**If nothing appears:**
- Card isn't properly seated
- Power cable not connected
- System needs to be restarted
- BIOS/UEFI settings may need adjustment

### Check 2: Driver Installation

**Check if tt-smi is installed:**
```bash
which tt-smi
```

**Expected:** `/usr/local/bin/tt-smi` or similar path

**If not found:**
- Driver not installed
- Install from: [tt-smi installation guide](https://github.com/tenstorrent/tt-smi)

### Check 3: Permissions

**Try with sudo:**
```bash
sudo tt-smi
```

**If this works but `tt-smi` alone doesn't:**
- Permissions issue
- Add your user to the appropriate group
- Or use sudo for now

**Fix permissions (Linux):**
```bash
sudo usermod -a -G tenstorrent $USER
# Log out and back in for group changes to take effect
```

### Check 4: Device Reset

**If device appears but shows errors:**
```bash
tt-smi -r
```

**What this does:**
- Resets the Tenstorrent device
- Clears any error states
- Takes ~5-10 seconds

**If reset fails:**
```bash
# Full state cleanup (requires sudo)
sudo pkill -9 -f tt-metal
sudo pkill -9 -f vllm
sudo rm -rf /dev/shm/tenstorrent* /dev/shm/tt_*
tt-smi -r
```

### Still Having Issues?

**Check system logs:**
```bash
dmesg | grep -i tenstorrent
```

**Get help:**
- [tt-smi troubleshooting guide](https://github.com/tenstorrent/tt-smi/blob/main/TROUBLESHOOTING.md)
- [Tenstorrent Discord community](https://discord.gg/tenstorrent)
- GitHub issues: [tt-smi issues](https://github.com/tenstorrent/tt-smi/issues)

**Common issues:**
- **"No kernel driver loaded"** → Reinstall tt-smi/drivers
- **"Device initialization timeout"** → Try device reset (`tt-smi -r`)
- **"Permission denied"** → Add user to tenstorrent group or use sudo

---

## QuietBox Multi-Device Systems

**What is QuietBox?**
QuietBox is a Tenstorrent multi-chip development system. The QuietBox Blackhole Tower contains **4x P300c cards** (4 separate single-chip Blackhole devices).

**Key Concepts:**

**4x P300c ≠ 4-chip System**
- **4x P300c** = 4 separate cards, each with 1 Blackhole chip
- Total: 4 devices, each independently addressable
- Each device runs in P100 mode (single Blackhole chip)

**Device Enumeration:**
```bash
tt-smi -s  # Shows all 4 devices
{
  "device_0": {"board_type": "p300c", "pci_bus": "0000:01:00.0"},
  "device_1": {"board_type": "p300c", "pci_bus": "0000:02:00.0"},
  "device_2": {"board_type": "p300c", "pci_bus": "0000:03:00.0"},
  "device_3": {"board_type": "p300c", "pci_bus": "0000:04:00.0"}
}
```

**Configuration for Lessons:**

**Single-Chip Lessons** (Most lessons):
- Use device 0 only: `export TT_METAL_DEVICE_ID=0`
- Set MESH_DEVICE: `export MESH_DEVICE=P100`
- Architecture: `export TT_METAL_ARCH_NAME=blackhole`

**Multi-Device Lessons** (Advanced):
- Use all devices: `TT_METAL_NUM_DEVICES=4`
- See Lesson 15 (Metalium Cookbook - Particle Life) for multi-device example
- Achieves 2x speedup on 4x P300c through workload parallelization

**Troubleshooting:**
- If script says "Unknown board type 'P300C'": Treat as P100 (single Blackhole)
- Multi-chip mesh initialization: All 4 devices will initialize fabric
- Device reset: Use `tt-smi -r` carefully (close all processes first)

---

## What You Learned

- ✅ How to detect Tenstorrent hardware with `tt-smi`
- ✅ Understanding different hardware types (N150, N300, T3K, P100, P150, P300/P300c)
- ✅ Using `tt-smi -s` for structured JSON output
- ✅ Identifying your specific hardware for later lessons
- ✅ QuietBox multi-device system configuration
- ✅ Troubleshooting hardware detection issues

**Next step:** Now that you know your hardware, verify your tt-metal installation works correctly.

Continue to Lesson 2: Verify Installation!

---

## Learn More

- **tt-smi documentation:** [github.com/tenstorrent/tt-smi](https://github.com/tenstorrent/tt-smi)
- **Hardware specs:** [tenstorrent.com/hardware](https://tenstorrent.com/hardware)
- **Community support:** [discord.gg/tenstorrent](https://discord.gg/tenstorrent)
