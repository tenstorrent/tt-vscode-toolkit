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
    aiclk: number;        // MHz (from tt_aiclk sysfs attribute)
    power: number;        // Watts (from hwmon power1_input)
    voltage: number;      // Volts (from hwmon in0_input)
    current: number;      // Amperes (from hwmon curr1_input)
    board_type: string;   // e.g., "n150", "n300", "p100" (from tt_card_type)
    pci_bus: string;      // PCI bus ID (e.g., "0000:01:00.0")
}

/**
 * Error response from telemetry reader
 */
export interface TelemetryError {
    error: string;
}
