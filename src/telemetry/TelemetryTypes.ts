/**
 * Telemetry Types for Tenstorrent Hardware Monitoring
 *
 * Data read from Linux sysfs (hwmon + Tenstorrent driver attributes).
 * Completely non-invasive - just reading kernel-exposed values.
 */

/**
 * Telemetry data from sysfs hwmon interface (already scaled to human-readable values)
 */
export interface TelemetryData {
    asic_temp: number;    // °C (from hwmon temp1_input)
    board_temp: number;   // °C (same as asic_temp)
    aiclk: number;        // MHz - Tensix AI cores clock (from tt_aiclk sysfs attribute)
    arcclk: number;       // MHz - ARC processor clock (from tt_arcclk)
    axiclk: number;       // MHz - AXI bus clock (from tt_axiclk)
    power: number;        // Watts (from hwmon power1_input)
    voltage: number;      // Volts (from hwmon in0_input)
    current: number;      // Amperes (from hwmon curr1_input)
    board_type: string;   // e.g., "n150", "n300", "p100" (from tt_card_type)
    pci_bus: string;      // PCI bus ID (e.g., "0000:01:00.0")
    device_index?: number; // Optional device index for multi-device systems
}

/**
 * Multi-device telemetry response (v0.0.230+)
 */
export interface MultiDeviceTelemetry {
    devices: TelemetryData[];
    count: number;
}

/**
 * Error response from telemetry reader
 */
export interface TelemetryError {
    error: string;
}

/**
 * Type guard to check if response is multi-device format
 */
export function isMultiDeviceTelemetry(data: any): data is MultiDeviceTelemetry {
    return data && typeof data === 'object' && 'devices' in data && 'count' in data && Array.isArray(data.devices);
}

/**
 * Type guard to check if response is single device format (legacy)
 */
export function isSingleDeviceTelemetry(data: any): data is TelemetryData {
    return data && typeof data === 'object' && 'asic_temp' in data && !('devices' in data);
}
