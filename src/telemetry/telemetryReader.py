#!/usr/bin/env python3
"""
Lightweight telemetry reader for Tenstorrent hardware.
Outputs JSON for VSCode extension consumption.

Usage:
    python telemetryReader.py

Output:
    JSON with telemetry metrics or error message

Example output:
    {
        "asic_temp": 12345678,
        "board_temp": 12345678,
        "aiclk": 1200,
        "tdp": 150,
        "tdc": 25,
        "vcore": 850,
        "fan_speed": 2400
    }

Error output:
    {
        "error": "No devices found"
    }
"""

import sys
import json

try:
    import tt_umd

    # Enumerate devices
    dev_ids = tt_umd.PCIDevice.enumerate_devices()

    if len(dev_ids) == 0:
        print(json.dumps({"error": "No devices found"}))
        sys.exit(1)

    # Create device (reads telemetry without disrupting workloads)
    dev = tt_umd.TTDevice.create(dev_ids[0])
    dev.init_tt_device()

    # Get telemetry reader
    tel_reader = dev.get_arc_telemetry_reader()

    # Read key metrics
    telemetry = {
        "asic_temp": tel_reader.read_entry(int(tt_umd.TelemetryTag.ASIC_TEMPERATURE)),
        "board_temp": tel_reader.read_entry(int(tt_umd.TelemetryTag.BOARD_TEMPERATURE)),
        "aiclk": tel_reader.read_entry(int(tt_umd.TelemetryTag.AICLK)),
        "tdp": tel_reader.read_entry(int(tt_umd.TelemetryTag.TDP)),
        "tdc": tel_reader.read_entry(int(tt_umd.TelemetryTag.TDC)),
        "vcore": tel_reader.read_entry(int(tt_umd.TelemetryTag.VCORE)),
        "fan_speed": tel_reader.read_entry(int(tt_umd.TelemetryTag.FAN_SPEED)),
    }

    print(json.dumps(telemetry))

except Exception as e:
    print(json.dumps({"error": str(e)}))
    sys.exit(1)
