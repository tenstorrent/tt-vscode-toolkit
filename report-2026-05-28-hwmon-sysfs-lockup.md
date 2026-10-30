# Tenstorrent Kernel Driver v2.8.0 — hwmon sysfs reads block indefinitely under compile load, causing full system lockup

**Filed:** 2026-05-28  
**Machine:** tt-quietbox  
**Severity:** Critical — full system lockup, no input/SSH/recovery path, requires hard AC power cycle  
**Relation to prior report:** See `bug-report-arc-hang-chip3.md`. This is a distinct failure mode but shares the same driver code path. In the prior report, the ARC firmware hangs and the driver loops retrying `arc_msg()`. In this report, the driver's hwmon `.read()` callback blocks indefinitely while a compile is in progress — likely on the same lock or MMIO path. Both result in hard lockups.

---

## System Configuration

| Field | Value |
|-------|-------|
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 6.17.0-29-generic (SMP PREEMPT_DYNAMIC) |
| TT driver | tenstorrent.ko v2.8.0 (DKMS, out-of-tree) |
| Firmware bundle | 19.8.0.0 |
| Ethernet firmware | 1.10.1 |

### Chips

| Chip | PCIe address | Board ID | PCIe link | hwmon sysfs path |
|------|-------------|----------|-----------|-----------------|
| 0 | 0000:01:00.0 | 0000046131924062 | Gen4 x4 | /sys/class/hwmon/hwmon1 |
| 1 | 0000:02:00.0 | 0000046131924062 | Gen5 x4 | /sys/class/hwmon/hwmon3 |
| 2 | 0000:03:00.0 | 0000046131924055 | Gen5 x4 | /sys/class/hwmon/hwmon4 |
| 3 | 0000:04:00.0 | 0000046131924055 | Gen5 x4 | /sys/class/hwmon/hwmon5 |

Two P300C cards, two chips per card. Cards share board IDs within each pair.

### Normal hwmon values at idle

```
chip0  temp=36626 mC (36.6°C)   power=32000000 uW (32W)
chip1  temp=38741 mC (38.7°C)   power=12000000 uW (12W)
chip2  temp=35409 mC (35.4°C)   power=30000000 uW (30W)
chip3  temp=37524 mC (37.5°C)   power=13000000 uW (13W)
```

---

## The Bug

When 4 concurrent forge/ttnn compile operations are in progress (one per chip), reads from
the TT hwmon sysfs interface block indefinitely in uninterruptible kernel sleep (process
state D). Any process that reads these files hangs. The blocking duration escalates as
compile intensity increases:

```
Idle:                         < 1ms per read
Under 4-chip compile load:    10s → 97s → 479s per read
```

Because sysfs attribute reads run on kernel work queue threads that cannot be killed
with SIGKILL while blocked in D state, the blocked reads accumulate and starve the entire
system. SSH becomes unresponsive. Input stops working. The machine must be hard power cycled.

---

## How to Reproduce

These steps use only standard Linux commands and mainline Tenstorrent tools. No
third-party software required.

### Step 1 — Verify hwmon interface works at idle

Run these before starting any compile workload. Each should return immediately (< 1ms):

```bash
# Find TT hwmon paths (will be hwmon1 through hwmon4 or hwmon5 depending on system)
for h in /sys/class/hwmon/hwmon*; do
    name=$(cat $h/name 2>/dev/null)
    dev=$(readlink -f $h/device 2>/dev/null | sed 's|.*/||')
    [ "$name" = "blackhole" ] && echo "$h -> $dev"
done

# Read temperature and power from each chip (should return instantly)
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = "blackhole" ] || continue
    dev=$(readlink -f $h/device | sed 's|.*/||')
    echo "$dev: temp=$(cat $h/temp1_input) mC  power=$(cat $h/power1_input) uW"
done
```

Expected output: all reads complete immediately with normal temperature values
(roughly 35000–50000 mC at idle).

### Step 2 — Start concurrent compile load

Start a compile workload on all chips simultaneously. Use any of the following:

**Option A — forge/ttnn (simplest)**

Run a short compile on each chip in parallel. From a Python environment with forge:

```python
import torch, forge, os, subprocess, sys

# Launch one compiler subprocess per chip — each opens /dev/tenstorrent/N
for chip_id in range(4):
    cmd = [sys.executable, '-c', f"""
import os, torch, forge
os.environ['TT_VISIBLE_DEVICES'] = '{chip_id}'
model = torch.nn.Linear(256, 256)
x = torch.randn(1, 256)
compiled = forge.compile(model, sample_inputs=[x])
print('chip {chip_id} done')
"""]
    subprocess.Popen(cmd)
```

**Option B — tt-smi stress (if available)**

```bash
tt-smi --stress  # if your version supports it
```

**Option C — any sustained TTNN workload**

Any workload that runs heavy inference or compilation on all 4 chips concurrently
will reproduce this. The key is sustained load across all chips for 10–30 minutes.

### Step 3 — Poll sysfs while compile runs (the trigger)

In a separate terminal, run this loop while the compile is active:

```bash
# Simple blocking-read test — time how long each read takes
while true; do
    for h in /sys/class/hwmon/hwmon*; do
        [ "$(cat $h/name 2>/dev/null)" = "blackhole" ] || continue
        start=$(date +%s%3N)
        val=$(cat $h/temp1_input 2>/dev/null)
        elapsed=$(( $(date +%s%3N) - start ))
        echo "$(date +%H:%M:%S)  $h  temp=${val} mC  read_time=${elapsed}ms"
    done
    sleep 3
done
```

Under normal conditions every read completes in < 1ms. When the bug triggers,
individual reads will take 10s, then 60s, then several minutes. The shell running
this loop will become unresponsive and eventually the whole system will freeze.

### Step 4 — Observe the progression

| Phase | What you see |
|-------|-------------|
| Normal | `read_time=0ms` on every iteration |
| Early contention | `read_time=10000ms` — one read blocks for 10 seconds |
| Escalating | `read_time=90000ms` — reads now blocking for ~90 seconds |
| Terminal | Shell frozen, no new output. SSH becomes unresponsive. |
| Lockup | Machine stops responding entirely. Requires hard power cycle. |

### Step 5 — Confirm with a single blocking read (non-destructive test)

To verify the contention exists without risking a full lockup, run this in one terminal
during active compile and measure the duration:

```bash
# Time a single sysfs read — safe to Ctrl-C if it blocks
time cat /sys/class/hwmon/hwmon3/temp1_input
```

If this returns in < 1ms, compile load is insufficient or the bug is not present.  
If it hangs for seconds or minutes, the bug is reproducible on your system.

---

## Observed Crash Sequence (2026-05-28, tt-quietbox)

```
10:16  Machine booted
11:07  4-chip forge compile run started
       Chips: 44–52°C, 55–75W — normal operating range

11:35:08  sysfs poll loop blocked for 10.6 seconds (first sign of contention)
11:38:17  sysfs poll loop blocked for 96.9 seconds
11:39:06  snapd systemd watchdog timeout (5 min limit) — snapd cannot respond
          because its threads are blocked waiting on kernel work queues
11:43:06  snapd SIGKILL'd — still cannot die (uninterruptible D state)
11:44:53  sysfs poll loop blocked for 479.4 seconds (8 minutes on one read)
          — last successful log entry from any userspace process

11:44 – 13:11  System alive but completely unresponsive:
               - SSH: connection refused / timeout
               - Keyboard/mouse: no input processed
               - dbus: service activation timeouts
               - NetworkManager dispatcher: timed out at 25000ms

13:11  Last journal entry (cron). No further kernel or userspace messages.
13:15  Hard AC power cycle. Machine recovers.
```

Post-reboot indicators confirming hard lockup (not clean shutdown):
- `systemd-fsck`: "Dirty bit is set. Fs was not properly unmounted"
- `systemd-journald`: "File corrupted or uncleanly shut down"
- No kernel panic, no MCE, no PCIe AER errors recorded

---

## What Can Be Ruled Out

| Hypothesis | Ruled out by |
|-----------|-------------|
| Thermal event | Chips at 44–52°C; thermal trip is 90°C, shutdown is 110°C |
| OOM / memory pressure | No OOM events in journal; RAM auto-cap was active |
| ARC firmware hang (prior bug) | No ARC timeout messages; no sentinel values (65536°C) in any hwmon reading; different symptom profile |
| tt-qb-lights / observer causation | ARC hangs predate tt-qb-lights installation (Dec 2025 vs Feb 2026); 78 minutes of clean polling before first block; any process reading the same sysfs files would exhibit identical blocking |
| Hardware fault | forge/ttnn access to all 4 chips working before and after; chips healthy on next boot |
| PCIe errors | No AER events in kernel log |

---

## Root Cause Analysis

The Linux hwmon subsystem contract requires that `.read()` callbacks return promptly
and never block on in-progress hardware operations. The tenstorrent.ko driver violates
this contract.

The driver's hwmon read path appears to share a lock or MMIO register access path with
the active compile pipeline. When a compile holds this resource, sysfs reads queue behind
it. The compile may run for many minutes; during this time all sysfs readers are stuck in
uninterruptible sleep (D state). Processes in D state cannot be killed — not even with
SIGKILL — because the signal is not delivered until the process returns from kernel space.
With enough blocked kernel threads, the work queue starves and the system becomes
permanently unresponsive.

**This is a kernel driver bug, not a firmware bug and not a user software bug.**

The firmware correctly exposes telemetry via SMBus/I2C (visible in `tt-smi -s` output
as `smbus_telem`). If the hwmon path read from the SMBus channel — which is independent
of the PCIe compute path — this bug would not exist. Instead it appears to read via
the same PCIe MMIO path used by the compute pipeline.

Note: `tt-smi -s` is documented as safe for concurrent use. If `tt-smi` reads from the
same hwmon sysfs interface, it is equally susceptible to this hang. The safety guarantee
likely holds for firmware-level operations but does not carry through to the kernel
driver's sysfs implementation.

---

## Suggested Fix

The hwmon `.read()` implementation in `tenstorrent.ko` should not block on compute
pipeline resources. Three viable approaches:

1. **Cached telemetry thread:** Maintain a background kernel thread that polls chip
   telemetry on a short interval (e.g. 1 second) and stores results in driver memory.
   The hwmon `.read()` handler returns the cached value immediately with no hardware
   access. This is the pattern used by most GPU drivers.

2. **SMBus side-channel:** Route hwmon reads through the board's SMBus/I2C interface
   (already providing data in `smbus_telem`) rather than the PCIe MMIO path. SMBus is
   electrically independent of PCIe and cannot be blocked by compute activity.

3. **Non-blocking read with timeout:** If the existing path cannot be restructured,
   at minimum add a short timeout (e.g. 100ms) to the hwmon read path and return
   a cached/stale value rather than blocking indefinitely.

---

## Relation to Prior ARC Hang Report

See `bug-report-arc-hang-chip3.md` for a distinct but related bug: the ARC firmware
processor on chip 3 hangs under heavy load, producing `Timeout waiting for ARC response`
in the kernel log and sentinel values (65536°C / 4294W) in hwmon output.

The two bugs are separable:

| | ARC hang (prior report) | hwmon sysfs lockup (this report) |
|--|------------------------|----------------------------------|
| Trigger | Sustained TTNN inference, especially on chip 3 | Concurrent reads of hwmon sysfs during any 4-chip compile |
| Kernel indicator | `Timeout waiting for ARC response` | No kernel messages — silent until total freeze |
| hwmon values | 65536°C / 4294W (UINT32_MAX sentinels) | Normal values, then reads block indefinitely |
| Chip affected | Chip 3 exclusively (all events) | All chips (sysfs reads across all 4 block) |
| Recovery | AC power cycle | AC power cycle |
| Observer causation | Not applicable | Ruled out (see above) |

However they may share a common code path: the ARC hang causes `arc_msg()` to loop
indefinitely in the driver (Phase 6 of prior report). The hwmon sysfs lockup may also
route through `arc_msg()` or a shared lock. If so, a fix to the hwmon path may also
improve the ARC hang recovery behavior.

---

## Appendix: Live Data (captured 2026-05-28 at idle)

```
$ uname -r
6.17.0-29-generic

$ modinfo tenstorrent | grep version
version: 2.8.0

$ tt-smi -s (relevant fields)
chip0  0000:01:00.0  fw=19.8.0.0  eth_fw=1.10.1  temp=35.6°C  power=15W  aiclk=800MHz
chip1  0000:02:00.0  fw=19.8.0.0  eth_fw=1.10.1  temp=38.1°C  power=12W  aiclk=800MHz
chip2  0000:03:00.0  fw=19.8.0.0  eth_fw=1.10.1  temp=35.4°C  power=12W  aiclk=800MHz
chip3  0000:04:00.0  fw=19.8.0.0  eth_fw=1.10.1  temp=37.3°C  power=13W  aiclk=800MHz

$ for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat $h/name 2>/dev/null)" = "blackhole" ] || continue
    dev=$(readlink -f $h/device | sed 's|.*/||')
    echo "$dev: temp=$(cat $h/temp1_input) mC  power=$(cat $h/power1_input) uW"
  done
0000:01:00.0: temp=36626 mC  power=32000000 uW
0000:02:00.0: temp=38741 mC  power=12000000 uW
0000:03:00.0: temp=35409 mC  power=30000000 uW
0000:04:00.0: temp=37524 mC  power=13000000 uW
```

All reads returned in < 1ms at idle. Under 4-chip concurrent compile, these same reads
block for 10 seconds → 97 seconds → 479 seconds, ultimately producing a full system lockup.
