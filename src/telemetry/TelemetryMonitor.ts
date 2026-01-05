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

    constructor(context: vscode.ExtensionContext, onTelemetryUpdate?: (telemetry: TelemetryData) => void) {
        console.log('[TelemetryMonitor] Initializing...');

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

        console.log(`[TelemetryMonitor] Script path: ${this.scriptPath}`);
        console.log(`[TelemetryMonitor] Callback registered: ${onTelemetryUpdate ? 'YES' : 'NO'}`);

        // Store callback
        this.onTelemetryUpdate = onTelemetryUpdate;

        // Start monitoring
        this.startMonitoring();
    }

    private startMonitoring() {
        console.log('[TelemetryMonitor] Starting monitoring (5s interval)');

        // Update immediately
        this.updateTelemetry();

        // Update every 5 seconds
        this.updateInterval = setInterval(() => {
            this.updateTelemetry();
        }, 5000);
    }

    private async updateTelemetry() {
        try {
            console.log('[TelemetryMonitor] Reading telemetry...');
            const telemetry = await this.readTelemetry();

            if ('error' in telemetry) {
                console.error('[TelemetryMonitor] Telemetry error:', telemetry.error);
                this.showError(telemetry.error);
                return;
            }

            console.log('[TelemetryMonitor] Telemetry received:', telemetry);
            this.updateStatusBar(telemetry);

            // Notify callback (e.g., for logo animation)
            if (this.onTelemetryUpdate) {
                console.log('[TelemetryMonitor] Invoking callback with telemetry');
                this.onTelemetryUpdate(telemetry);
            } else {
                console.warn('[TelemetryMonitor] No callback registered - animation will not update');
            }

        } catch (error) {
            console.error('[TelemetryMonitor] Exception:', error);
            this.showError(`Telemetry error: ${error}`);
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

    private showError(message: string) {
        this.statusBarItem.text = '⚠️ TT Hardware';
        this.statusBarItem.tooltip = `Telemetry unavailable: ${message}`;
        this.statusBarItem.show();
    }

    public dispose() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.statusBarItem.dispose();
    }
}
