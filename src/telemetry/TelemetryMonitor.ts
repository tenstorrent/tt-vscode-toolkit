/**
 * TelemetryMonitor - Live Hardware Monitoring for VSCode Statusbar
 *
 * Displays real-time hardware metrics from Tenstorrent devices:
 * - Temperature (ASIC and board)
 * - Power consumption (TDP)
 * - Clock speeds (AICLK)
 * - Status indicators (color-coded by temperature)
 *
 * Non-invasive: reads telemetry without disrupting running workloads.
 */

import * as vscode from 'vscode';
import * as child_process from 'child_process';
import * as path from 'path';
import { TelemetryData, TelemetryError, MultiDeviceTelemetry, isMultiDeviceTelemetry, isSingleDeviceTelemetry } from './TelemetryTypes';

export class TelemetryMonitor {
    private statusBarItem: vscode.StatusBarItem;
    private updateInterval: NodeJS.Timeout | undefined;
    private pythonPath: string;
    private scriptPath: string;
    private onTelemetryUpdate?: (telemetry: TelemetryData) => void;
    private currentTelemetry?: TelemetryData;
    private currentMultiDeviceTelemetry?: MultiDeviceTelemetry;
    private lastError?: string;

    constructor(context: vscode.ExtensionContext, onTelemetryUpdate?: (telemetry: TelemetryData) => void) {
        // Create statusbar item (right side, high priority)
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = 'tenstorrent.showDeviceActions';
        context.subscriptions.push(this.statusBarItem);

        // Paths (use system Python - script only needs subprocess and json)
        this.pythonPath = 'python3';
        // Use extension path for reliable script location
        this.scriptPath = path.join(context.extensionPath, 'dist', 'src', 'telemetry', 'telemetryReader.py');

        // Store callback
        this.onTelemetryUpdate = onTelemetryUpdate;

        // Start monitoring
        this.startMonitoring();
    }

    private startMonitoring() {
        // Update immediately
        this.updateTelemetry();

        // Update every 5 seconds
        this.updateInterval = setInterval(() => {
            this.updateTelemetry();
        }, 5000);
    }

    private async updateTelemetry() {
        try {
            const rawTelemetry = await this.readTelemetry();

            if ('error' in rawTelemetry) {
                this.lastError = rawTelemetry.error;
                this.currentTelemetry = undefined;
                this.currentMultiDeviceTelemetry = undefined;
                this.showError(rawTelemetry.error);
                return;
            }

            // Store both aggregated and multi-device formats
            if (isMultiDeviceTelemetry(rawTelemetry)) {
                this.currentMultiDeviceTelemetry = rawTelemetry;
                this.currentTelemetry = this.aggregateTelemetry(rawTelemetry);
            } else {
                this.currentTelemetry = rawTelemetry;
                this.currentMultiDeviceTelemetry = undefined;
            }

            this.lastError = undefined;
            this.updateStatusBar(this.currentTelemetry);

            // Notify callback (e.g., for logo animation)
            if (this.onTelemetryUpdate && this.currentTelemetry) {
                this.onTelemetryUpdate(this.currentTelemetry);
            }

        } catch (error) {
            const errorMsg = `Telemetry error: ${error}`;
            this.lastError = errorMsg;
            this.currentTelemetry = undefined;
            this.currentMultiDeviceTelemetry = undefined;
            this.showError(errorMsg);
        }
    }

    private readTelemetry(): Promise<TelemetryData | MultiDeviceTelemetry | TelemetryError> {
        return new Promise((resolve, reject) => {
            child_process.exec(
                `${this.pythonPath} ${this.scriptPath}`,
                { timeout: 3000 },
                (error, stdout, _stderr) => {
                    if (error) {
                        reject(error);
                        return;
                    }

                    try {
                        const data = JSON.parse(stdout);

                        // Handle multi-device format (v0.0.230+)
                        if (isMultiDeviceTelemetry(data)) {
                            // Return raw multi-device data (aggregation happens in updateTelemetry)
                            resolve(data);
                        } else if (isSingleDeviceTelemetry(data)) {
                            // Legacy single device format
                            resolve(data);
                        } else if ('error' in data) {
                            // Error response
                            resolve(data as TelemetryError);
                        } else {
                            reject(new Error('Unknown telemetry format'));
                        }
                    } catch (e) {
                        reject(e);
                    }
                }
            );
        });
    }

    /**
     * Aggregate multi-device telemetry into single view for status bar display
     */
    private aggregateTelemetry(multiDevice: MultiDeviceTelemetry): TelemetryData {
        const devices = multiDevice.devices;

        if (devices.length === 0) {
            throw new Error('No devices in telemetry data');
        }

        // For single device, just return it
        if (devices.length === 1) {
            return devices[0];
        }

        // For multiple devices, aggregate metrics
        const temps = devices.map(d => d.asic_temp);
        const powers = devices.map(d => d.power);
        const voltages = devices.map(d => d.voltage);
        const currents = devices.map(d => d.current);

        return {
            asic_temp: Math.max(...temps), // Show hottest device
            board_temp: Math.max(...temps),
            aiclk: devices[0].aiclk, // All devices should have same clock
            arcclk: devices[0].arcclk,
            axiclk: devices[0].axiclk,
            power: powers.reduce((sum, p) => sum + p, 0), // Total power
            voltage: voltages.reduce((sum, v) => sum + v, 0) / devices.length, // Avg voltage
            current: currents.reduce((sum, c) => sum + c, 0), // Total current
            board_type: `${devices[0].board_type} (${devices.length}x)`,
            pci_bus: `${devices.length} devices`
        };
    }

    private updateStatusBar(telemetry: TelemetryData) {
        // Color-coded status based on temperature
        const tempIcon = this.getTempIcon(telemetry.asic_temp);

        // Determine device config string
        let deviceConfig = '';
        if (this.currentMultiDeviceTelemetry && this.currentMultiDeviceTelemetry.count > 0) {
            const count = this.currentMultiDeviceTelemetry.count;
            const boardType = this.currentMultiDeviceTelemetry.devices[0].board_type.toUpperCase();
            if (count === 1) {
                deviceConfig = boardType;
            } else {
                deviceConfig = `${count}x ${boardType}`;
            }
        } else {
            // Single device format (legacy)
            deviceConfig = telemetry.board_type.toUpperCase();
        }

        // Format: 🌡️ 45°C | ⚡ 12.5W | 🔊 1200MHz | 2x P300
        const text = `${tempIcon} ${telemetry.asic_temp.toFixed(1)}°C | ` +
                    `⚡ ${telemetry.power.toFixed(1)}W | ` +
                    `🔊 ${telemetry.aiclk}MHz | ` +
                    `${deviceConfig}`;

        this.statusBarItem.text = text;
        this.statusBarItem.tooltip = this.buildTooltip();
        this.statusBarItem.show();
    }

    private getTempIcon(temp: number): string {
        if (temp < 50) return '🌡️';     // Cool
        if (temp < 70) return '🔥';     // Warm
        if (temp < 85) return '🚨';     // Hot
        return '⚠️';                      // Critical
    }

    private buildTooltip(): string {
        if (!this.currentTelemetry) {
            return 'No telemetry data available';
        }

        let tooltip = 'Tenstorrent Hardware Status\n\n';

        // Show individual device snapshots if multi-device
        if (this.currentMultiDeviceTelemetry && this.currentMultiDeviceTelemetry.count > 1) {
            tooltip += `${this.currentMultiDeviceTelemetry.count} devices detected:\n\n`;

            for (let i = 0; i < this.currentMultiDeviceTelemetry.devices.length; i++) {
                const device = this.currentMultiDeviceTelemetry.devices[i];
                tooltip += `Device ${i}: ${device.board_type.toUpperCase()}\n`;
                tooltip += `  Temp: ${device.asic_temp.toFixed(1)}°C | `;
                tooltip += `Power: ${device.power.toFixed(1)}W | `;
                tooltip += `Clock: ${device.aiclk}MHz\n`;
                tooltip += `  PCI: ${device.pci_bus}\n`;
                if (i < this.currentMultiDeviceTelemetry.devices.length - 1) {
                    tooltip += '\n';
                }
            }

            tooltip += '\nAggregated metrics (all devices):\n';
            tooltip += `Total Power: ${this.currentTelemetry.power.toFixed(1)} W\n`;
            tooltip += `Hottest Temp: ${this.currentTelemetry.asic_temp.toFixed(1)}°C\n`;
        } else {
            // Single device
            const telemetry = this.currentTelemetry;
            tooltip += `Board: ${telemetry.board_type.toUpperCase()}\n`;
            tooltip += `ASIC Temperature: ${telemetry.asic_temp.toFixed(1)}°C\n`;
            tooltip += `Board Temperature: ${telemetry.board_temp.toFixed(1)}°C\n`;
            tooltip += `\n`;
            tooltip += `Clocks:\n`;
            tooltip += `  AI (Tensix): ${telemetry.aiclk} MHz\n`;
            tooltip += `  ARC: ${telemetry.arcclk} MHz\n`;
            tooltip += `  AXI Bus: ${telemetry.axiclk} MHz\n`;
            tooltip += `\n`;
            tooltip += `Power: ${telemetry.power.toFixed(1)} W\n`;
            tooltip += `Current: ${telemetry.current.toFixed(1)} A\n`;
            tooltip += `Voltage: ${telemetry.voltage.toFixed(2)} V\n`;
            tooltip += `PCI Bus: ${telemetry.pci_bus}\n`;
        }

        tooltip += `\nSource: Linux sysfs (hwmon)\n`;
        tooltip += `Click for device actions`;

        return tooltip;
    }

    private showError(_message: string) {
        // Hide status bar when hardware isn't available (no sysfs, no tt-smi, etc.)
        // This is cleaner than showing an error icon - telemetry is optional
        this.statusBarItem.hide();

        // Stop polling after first error to avoid wasting resources
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = undefined;
        }
    }

    /**
     * Get the current telemetry details (same as hover tooltip)
     * @returns The tooltip content, or undefined if no telemetry available
     */
    public getCurrentDetails(): string | undefined {
        if (this.currentTelemetry) {
            return this.buildTooltip();
        } else if (this.lastError) {
            return `Telemetry unavailable: ${this.lastError}`;
        }
        return undefined;
    }

    /**
     * Get raw multi-device telemetry for external use (e.g., device actions menu)
     */
    public getMultiDeviceTelemetry(): MultiDeviceTelemetry | undefined {
        return this.currentMultiDeviceTelemetry;
    }

    public dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.statusBarItem.dispose();
    }
}
