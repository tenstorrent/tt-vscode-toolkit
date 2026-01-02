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
import { TelemetryData, ScaledTelemetry, TelemetryError } from './TelemetryTypes';

export class TelemetryMonitor {
    private statusBarItem: vscode.StatusBarItem;
    private updateInterval: NodeJS.Timeout | undefined;
    private pythonPath: string;
    private scriptPath: string;

    constructor(context: vscode.ExtensionContext) {
        // Create statusbar item (right side, high priority)
        this.statusBarItem = vscode.window.createStatusBarItem(
            vscode.StatusBarAlignment.Right,
            100
        );
        this.statusBarItem.command = 'tenstorrent.showTelemetryDetails';
        context.subscriptions.push(this.statusBarItem);

        // Paths (use tt-metal Python environment)
        this.pythonPath = '/home/user/tt-metal/python_env/bin/python3';
        this.scriptPath = path.join(__dirname, 'telemetryReader.py');

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
                this.showError(telemetry.error);
                return;
            }

            const scaled = this.scaleTelemetry(telemetry);
            this.updateStatusBar(scaled);

        } catch (error) {
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

    private scaleTelemetry(raw: TelemetryData): ScaledTelemetry {
        // Apply scaling factors from tt-metal telemetry_provider.cpp
        return {
            asicTemp: this.scaleTemperature(raw.asic_temp),
            boardTemp: this.scaleTemperature(raw.board_temp),
            aiClock: (raw.aiclk & 0xffff),  // Mask to get MHz
            power: (raw.tdp & 0xffff),      // Already in Watts
            current: (raw.tdc & 0xffff),    // Already in Amperes
            voltage: (raw.vcore & 0xffffffff) / 1000,  // millivolts to volts
            fanSpeed: (raw.fan_speed & 0xffffffff),
        };
    }

    private scaleTemperature(raw: number): number {
        // From telemetry_provider.cpp: mask and scale
        const masked = (raw & 0xffffffff);
        return masked * (1.0 / 65536.0);
    }

    private updateStatusBar(telemetry: ScaledTelemetry) {
        // Color-coded status based on temperature
        const tempIcon = this.getTempIcon(telemetry.asicTemp);

        // Format: 🌡️ 45°C | ⚡ 12.5W | 🔊 1200MHz
        const text = `${tempIcon} ${telemetry.asicTemp.toFixed(1)}°C | ` +
                    `⚡ ${telemetry.power.toFixed(1)}W | ` +
                    `🔊 ${telemetry.aiClock}MHz`;

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

    private buildTooltip(telemetry: ScaledTelemetry): string {
        return `Tenstorrent Hardware Status\n\n` +
               `ASIC Temperature: ${telemetry.asicTemp.toFixed(1)}°C\n` +
               `Board Temperature: ${telemetry.boardTemp.toFixed(1)}°C\n` +
               `AI Clock: ${telemetry.aiClock} MHz\n` +
               `Power: ${telemetry.power.toFixed(1)} W\n` +
               `Current: ${telemetry.current.toFixed(1)} A\n` +
               `Voltage: ${telemetry.voltage.toFixed(2)} V\n` +
               `Fan Speed: ${telemetry.fanSpeed} RPM\n\n` +
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
