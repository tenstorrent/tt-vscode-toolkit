#!/usr/bin/env python3
"""
Lightweight telemetry reader for Tenstorrent hardware.
Reads directly from sysfs (hwmon + Tenstorrent driver attributes).
Completely non-invasive - just reads kernel-exposed attributes.

Usage:
    python telemetryReader.py

Output:
    JSON with telemetry metrics or error message

Example output:
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

Error output:
    {
        "error": "No devices found"
    }
"""

import sys
import json
import os
import glob

def read_sysfs_value(path):
    """Read a single value from sysfs, return None if unavailable."""
    try:
        with open(path, 'r') as f:
            return f.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None

def find_tenstorrent_devices():
    """Find all Tenstorrent devices in /sys/class/tenstorrent/."""
    try:
        devices = glob.glob('/sys/class/tenstorrent/tenstorrent*')
        # Filter out subdirectories, only keep actual device nodes
        devices = [d for d in devices if os.path.isdir(d) and '!' in d]
        return sorted(devices)
    except Exception:
        return []

def read_device_telemetry(device_path):
    """Read telemetry from a single Tenstorrent device."""

    # Read basic device info
    card_type = read_sysfs_value(os.path.join(device_path, 'tt_card_type'))
    aiclk = read_sysfs_value(os.path.join(device_path, 'tt_aiclk'))
    arcclk = read_sysfs_value(os.path.join(device_path, 'tt_arcclk'))
    axiclk = read_sysfs_value(os.path.join(device_path, 'tt_axiclk'))

    # Get PCI bus ID from device symlink
    device_link = os.path.join(device_path, 'device')
    pci_bus = None
    if os.path.islink(device_link):
        pci_bus = os.path.basename(os.readlink(device_link))

    # Find hwmon directory (may be hwmon0, hwmon1, etc.)
    hwmon_base = os.path.join(device_path, 'device', 'hwmon')
    hwmon_dirs = glob.glob(os.path.join(hwmon_base, 'hwmon*'))

    # Read hwmon telemetry
    temp = None
    power = None
    current = None
    voltage = None

    if hwmon_dirs:
        hwmon_dir = hwmon_dirs[0]  # Use first hwmon device

        # Temperature (millidegrees Celsius)
        temp_raw = read_sysfs_value(os.path.join(hwmon_dir, 'temp1_input'))
        if temp_raw:
            temp = float(temp_raw) / 1000.0  # Convert to degrees C

        # Power (microwatts)
        power_raw = read_sysfs_value(os.path.join(hwmon_dir, 'power1_input'))
        if power_raw:
            power = float(power_raw) / 1000000.0  # Convert to watts

        # Current (milliamps)
        current_raw = read_sysfs_value(os.path.join(hwmon_dir, 'curr1_input'))
        if current_raw:
            current = float(current_raw) / 1000.0  # Convert to amps

        # Voltage (millivolts)
        voltage_raw = read_sysfs_value(os.path.join(hwmon_dir, 'in0_input'))
        if voltage_raw:
            voltage = float(voltage_raw) / 1000.0  # Convert to volts

    return {
        "asic_temp": temp if temp is not None else 0.0,
        "board_temp": temp if temp is not None else 0.0,  # Use same temp for both
        "aiclk": int(aiclk) if aiclk else 0,
        "arcclk": int(arcclk) if arcclk else 0,
        "axiclk": int(axiclk) if axiclk else 0,
        "power": power if power is not None else 0.0,
        "voltage": voltage if voltage is not None else 0.0,
        "current": current if current is not None else 0.0,
        "board_type": card_type if card_type else "unknown",
        "pci_bus": pci_bus if pci_bus else "unknown",
    }

try:
    # Find all Tenstorrent devices
    devices = find_tenstorrent_devices()

    if not devices:
        print(json.dumps({"error": "No devices found"}))
        sys.exit(1)

    # Read telemetry from all devices
    telemetry_data = []
    for idx, device_path in enumerate(devices):
        device_telemetry = read_device_telemetry(device_path)
        device_telemetry['device_index'] = idx  # Add device index for identification
        telemetry_data.append(device_telemetry)

    # Output array of telemetry data (one per device)
    print(json.dumps({"devices": telemetry_data, "count": len(telemetry_data)}))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
