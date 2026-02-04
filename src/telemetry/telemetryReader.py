#!/usr/bin/env python3
"""
Lightweight telemetry reader for Tenstorrent hardware.
Reads directly from sysfs (hwmon + Tenstorrent driver attributes).
Completely non-invasive - just reads kernel-exposed attributes.

Multi-tenant isolation:
    In shared environments, /sys/class/tenstorrent/ may show all devices
    on the host, but /dev/tenstorrent/ only exposes allocated devices.
    This script automatically filters to show only accessible devices.

Usage:
    python telemetryReader.py

Output:
    JSON with telemetry metrics or error message

Example output (single device):
    {
        "devices": [{
            "asic_temp": 45.3,
            "board_temp": 45.3,
            "aiclk": 1000,
            "power": 23.0,
            "voltage": 0.91,
            "current": 26.0,
            "board_type": "n150",
            "pci_bus": "0000:01:00.0",
            "device_index": 0
        }],
        "count": 1
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

def get_accessible_pci_addresses():
    """
    Get PCI addresses of devices accessible via /dev/tenstorrent/.
    This handles multi-tenant environments where sysfs shows all devices
    but /dev only exposes allocated devices.
    """
    accessible_pci = set()
    dev_dir = '/dev/tenstorrent'

    if not os.path.exists(dev_dir):
        return accessible_pci

    try:
        # List all device nodes in /dev/tenstorrent/
        dev_nodes = [os.path.join(dev_dir, f) for f in os.listdir(dev_dir)]

        for dev_node in dev_nodes:
            if not os.path.exists(dev_node):
                continue

            # Get major:minor for this device node
            stat_info = os.stat(dev_node)
            if not stat_info:
                continue

            major = os.major(stat_info.st_rdev)
            minor = os.minor(stat_info.st_rdev)

            # Look up PCI device via /sys/dev/char/<major>:<minor>
            sysfs_path = f'/sys/dev/char/{major}:{minor}'
            if not os.path.exists(sysfs_path):
                continue

            # Follow device symlink to find PCI address
            device_link = os.path.join(sysfs_path, 'device')
            if os.path.islink(device_link):
                pci_path = os.readlink(device_link)
                # Extract PCI address from path like ../../../0000:61:00.0
                pci_addr = os.path.basename(pci_path)
                accessible_pci.add(pci_addr)

    except Exception:
        # Silently handle missing /dev/tenstorrent directory or permission errors
        # Returns empty set if device enumeration fails (e.g., no hardware present)
        pass

    return accessible_pci

def find_tenstorrent_devices():
    """
    Find Tenstorrent devices in /sys/class/tenstorrent/ that are accessible.
    Filters to only show devices available in /dev/tenstorrent/.
    """
    try:
        devices = glob.glob('/sys/class/tenstorrent/tenstorrent*')
        # Filter out subdirectories, only keep actual device nodes
        devices = [d for d in devices if os.path.isdir(d) and '!' in d]

        # Get PCI addresses of accessible devices
        accessible_pci = get_accessible_pci_addresses()

        # If we found accessible devices, filter to only those
        if accessible_pci:
            filtered_devices = []
            for device_path in devices:
                # Get PCI address for this sysfs device
                device_link = os.path.join(device_path, 'device')
                if os.path.islink(device_link):
                    pci_path = os.readlink(device_link)
                    pci_addr = os.path.basename(pci_path)
                    if pci_addr in accessible_pci:
                        filtered_devices.append(device_path)

            return sorted(filtered_devices)

        # If no /dev/tenstorrent/ devices found, return all (fallback to old behavior)
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
