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
import { TelemetryData, TelemetryError } from './TelemetryTypes';

export class TelemetryMonitor {
    private statusBarItem: vscode.StatusBarItem;
    private updateInterval: NodeJS.Timeout | undefined;
    private pythonPath: string;
    private scriptPath: string;
    private onTelemetryUpdate?: (telemetry: TelemetryData) => void;
    private currentTelemetry?: TelemetryData;
    private lastError?: string;

    constructor(context: vscode.ExtensionContext, onTelemetryUpdate?: (telemetry: TelemetryData) => void) {
        // Create statusbar item (right side, high priority)
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = 'tenstorrent.showTelemetryDetails';
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
            const telemetry = await this.readTelemetry();

            if ('error' in telemetry) {
                this.lastError = telemetry.error;
                this.currentTelemetry = undefined;
                this.showError(telemetry.error);
                return;
            }

            this.currentTelemetry = telemetry;
            this.lastError = undefined;
            this.updateStatusBar(telemetry);

            // Notify callback (e.g., for logo animation)
            if (this.onTelemetryUpdate) {
                this.onTelemetryUpdate(telemetry);
            }

        } catch (error) {
            const errorMsg = `Telemetry error: ${error}`;
            this.lastError = errorMsg;
            this.currentTelemetry = undefined;
            this.showError(errorMsg);
        }
    }

    private readTelemetry(): Promise<TelemetryData | TelemetryError> {
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
                        resolve(data);
                    } catch (e) {
                        reject(e);
                    }
                }
            );
        });
    }

    private updateStatusBar(telemetry: TelemetryData) {
        // Color-coded status based on temperature
        const tempIcon = this.getTempIcon(telemetry.asic_temp);

        // Format: 🌡️ 45°C | ⚡ 12.5W | 🔊 1200MHz
        const text = `${tempIcon} ${telemetry.asic_temp.toFixed(1)}°C | ` +
                    `⚡ ${telemetry.power.toFixed(1)}W | ` +
                    `🔊 ${telemetry.aiclk}MHz`;

        this.statusBarItem.text = text;
        this.statusBarItem.tooltip = this.buildTooltip(telemetry);
        this.statusBarItem.show();
    }

    private getTempIcon(temp: number): string {
        if (temp < 50) return '🌡️';     // Cool
        if (temp < 70) return '🔥';     // Warm
        if (temp < 85) return '🚨';     // Hot
        return '⚠️';                      // Critical
    }

    private buildTooltip(telemetry: TelemetryData): string {
        return `Tenstorrent Hardware Status\n\n` +
               `Board: ${telemetry.board_type.toUpperCase()}\n` +
               `ASIC Temperature: ${telemetry.asic_temp.toFixed(1)}°C\n` +
               `Board Temperature: ${telemetry.board_temp.toFixed(1)}°C\n` +
               `\n` +
               `Clocks:\n` +
               `  AI (Tensix): ${telemetry.aiclk} MHz\n` +
               `  ARC: ${telemetry.arcclk} MHz\n` +
               `  AXI Bus: ${telemetry.axiclk} MHz\n` +
               `\n` +
               `Power: ${telemetry.power.toFixed(1)} W\n` +
               `Current: ${telemetry.current.toFixed(1)} A\n` +
               `Voltage: ${telemetry.voltage.toFixed(2)} V\n` +
               `PCI Bus: ${telemetry.pci_bus}\n\n` +
               `Source: Linux sysfs (hwmon)\n` +
               `Click for details`;
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
            return this.buildTooltip(this.currentTelemetry);
        } else if (this.lastError) {
            return `Telemetry unavailable: ${this.lastError}`;
        }
        return undefined;
    }

    public dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.statusBarItem.dispose();
    }
}
