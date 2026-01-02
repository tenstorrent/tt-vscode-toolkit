/**
 * Telemetry Types for Tenstorrent Hardware Monitoring
 *
 * Raw telemetry data from Python script needs scaling to human-readable units.
 */

/**
 * Raw telemetry values from tt_umd library
 * These need scaling before display
 */
export interface TelemetryData {
    asic_temp: number;    // Raw value (needs masking and scaling)
    board_temp: number;   // Raw value (needs masking and scaling)
    aiclk: number;        // Raw value (needs masking to get MHz)
    tdp: number;          // Raw value (needs masking to get Watts)
    tdc: number;          // Raw value (needs masking to get Amperes)
    vcore: number;        // Raw value in millivolts
    fan_speed: number;    // Raw value (RPM)
}

/**
 * Scaled telemetry values ready for display
 */
export interface ScaledTelemetry {
    asicTemp: number;     // °C
    boardTemp: number;    // °C
    aiClock: number;      // MHz
    power: number;        // Watts
    current: number;      // Amperes
    voltage: number;      // Volts
    fanSpeed: number;     // RPM
}

/**
 * Error response from telemetry reader
 */
export interface TelemetryError {
    error: string;
}
