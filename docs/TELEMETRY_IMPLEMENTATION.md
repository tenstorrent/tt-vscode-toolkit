# Telemetry Monitor Implementation (v0.0.211)

## Overview

Live hardware monitoring for VSCode statusbar using **completely non-invasive** sysfs reading.

## Architecture

### Data Source: Linux sysfs (hwmon + Tenstorrent driver)

**Path structure:**
```
/sys/class/tenstorrent/tenstorrent!N/
├── tt_aiclk              # AI clock speed (MHz)
├── tt_card_type          # Board type (n150, n300, p100, etc.)
├── device/
│   └── hwmon/hwmonN/     # Linux hardware monitoring interface
│       ├── temp1_input   # ASIC temperature (millidegrees C)
│       ├── power1_input  # Power consumption (microwatts)
│       ├── curr1_input   # Current (milliamps)
│       └── in0_input     # Voltage (millivolts)
```

### Why sysfs?

1. **Truly non-invasive** - Just reads kernel-exposed attributes
2. **No device initialization** - Doesn't touch hardware at all
3. **Standard Linux interface** - Uses hwmon (hardware monitoring) subsystem
4. **Works during workloads** - Completely safe to read while models run
5. **No special permissions** - User-readable attributes
6. **No tt-smi dependency** - Direct kernel access

### What About tt-smi?

tt-smi was avoided because:
- Known to be "somewhat invasive despite signs it shouldn't be"
- Makes ioctl calls to kernel driver
- May interfere with running workloads
- Adds unnecessary process overhead

## Implementation Details

### Python Script (`telemetryReader.py`)

**What it does:**
1. Finds all Tenstorrent devices in `/sys/class/tenstorrent/`
2. Reads hwmon data from `device/hwmon/hwmon*/`
3. Reads clock speeds from `tt_aiclk` sysfs attribute
4. Reads board type from `tt_card_type`
5. Returns JSON with scaled values

**Data conversions:**
```python
# Temperature: millidegrees Celsius → degrees Celsius
temp = float(temp_raw) / 1000.0

# Power: microwatts → watts
power = float(power_raw) / 1000000.0

# Current: milliamps → amperes
current = float(current_raw) / 1000.0

# Voltage: millivolts → volts
voltage = float(voltage_raw) / 1000.0
```

**Example output:**
```json
{
    "asic_temp": 45.3,
    "board_temp": 45.3,
    "aiclk": 1000,
    "power": 23.0,
    "voltage": 0.91,
    "current": 26.0,
    "board_type": "n150",
    "pci_bus": "0000:01:00.0"
}
```

### TypeScript Monitor (`TelemetryMonitor.ts`)

**What it does:**
1. Creates statusbar item (right side, priority 100)
2. Calls Python script every 5 seconds
3. Parses JSON output
4. Updates statusbar display
5. Shows detailed tooltip on hover

**Display format:**
```
🌡️ 45.3°C | ⚡ 23.0W | 🔊 1000MHz
```

**Temperature indicators:**
- 🌡️ Cool (<50°C)
- 🔥 Warm (50-70°C)
- 🚨 Hot (70-85°C)
- ⚠️ Critical (>85°C)

**Tooltip content:**
```
Tenstorrent Hardware Status

Board: n150
ASIC Temperature: 45.3°C
Board Temperature: 45.3°C
AI Clock: 1000 MHz
Power: 23.0 W
Current: 26.0 A
Voltage: 0.91 V
PCI Bus: 0000:01:00.0

Source: Linux sysfs (hwmon)
Click for details
```

## Multi-Device Support

**Current:** Reads first device only

**Future enhancement:**
```python
# TODO: Support multi-device by returning array
devices = find_tenstorrent_devices()
telemetry = [read_device_telemetry(d) for d in devices]
```

**Display options:**
1. Show aggregate (total power, max temp)
2. Show per-device in tooltip
3. Allow user to select which device to monitor
4. Cycle through devices automatically

## Verification

**Test script works:**
```bash
python3 src/telemetry/telemetryReader.py
# Output: {"asic_temp": 45.3, "board_temp": 45.3, ...}
```

**Verify sysfs attributes exist:**
```bash
ls -la /sys/class/tenstorrent/tenstorrent\!3/device/hwmon/hwmon*/
# Should show: temp1_input, power1_input, curr1_input, in0_input
```

**Read raw values:**
```bash
cat /sys/class/tenstorrent/tenstorrent\!3/device/hwmon/hwmon6/temp1_input
# Output: 45312 (45.312°C in millidegrees)
```

## Troubleshooting

**No devices found:**
```json
{"error": "No devices found"}
```
→ Check: `ls /sys/class/tenstorrent/`

**Permission denied:**
```json
{"error": "Permission denied"}
```
→ hwmon attributes should be world-readable. Check permissions:
```bash
ls -la /sys/class/tenstorrent/tenstorrent\!*/device/hwmon/hwmon*/
```

**hwmon directory not found:**
→ Kernel driver may not be exposing hwmon. Check dmesg for errors.

**Zero values:**
```json
{"asic_temp": 0.0, "power": 0.0, ...}
```
→ hwmon files may be returning zero. Check raw values:
```bash
cat /sys/class/tenstorrent/tenstorrent\!*/device/hwmon/hwmon*/temp1_input
```

## Performance Impact

**Overhead:** Minimal
- Python script execution: ~10ms
- sysfs reads: <1ms each
- Update interval: 5 seconds
- No ioctl calls
- No device initialization
- No process forking (uses same Python interpreter)

**Memory:** ~1KB for JSON output per device

## Comparison with Other Approaches

| Approach | Invasive? | Dependencies | Speed | Multi-device |
|----------|-----------|--------------|-------|--------------|
| **sysfs (current)** | ❌ No | None | Fast | ✅ Easy |
| tt-smi | ⚠️ Somewhat | tt-smi binary | Medium | ✅ Yes |
| tt_umd Python | ⚠️ Yes (init) | tt_umd module | Fast | ✅ Yes |
| tt-telemetry server | ⚠️ Yes (server) | Build required | Fast | ✅ Yes |
| lm-sensors | ❌ No | sensors binary | Fast | ✅ Yes |

**Winner: sysfs** - Most direct, least invasive, no dependencies

## Future Enhancements

1. **Historical graphs** - Store last N samples, plot trends
2. **Alert thresholds** - Notify when temp >80°C or power >100W
3. **Multi-device aggregation** - Show total power, max temp across all devices
4. **Fan speed control** - If exposed via sysfs (read-only for now)
5. **Clock frequency adjustment** - If exposed and writable
6. **Detailed webview** - Click statusbar to show charts and history
7. **Export metrics** - Save to CSV for analysis
8. **Prometheus endpoint** - Expose metrics for monitoring stack

## References

- Linux hwmon documentation: https://www.kernel.org/doc/html/latest/hwmon/
- sysfs documentation: https://www.kernel.org/doc/html/latest/filesystems/sysfs.html
- Tenstorrent kernel driver: tt-kmd (exposes sysfs attributes)

## Version History

- **v0.0.211** - Initial sysfs-based implementation
  - Replaced tt-smi with direct sysfs reading
  - Added hwmon temperature, power, current, voltage
  - Added multi-device discovery (reads first device)
  - Completely non-invasive approach
