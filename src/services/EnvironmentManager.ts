/**
 * EnvironmentManager - Manages Python environments for each terminal
 *
 * Features:
 * - Tracks which environment is active in each terminal
 * - Auto-activates correct environment when terminal is created
 * - Allows manual switching via command palette
 *
 * This solves the "environment drift" problem where users lose track
 * of which Python venv is active.
 *
 * Note: Status bar display has been removed to reduce clutter. Environment
 * switching is still available via command palette.
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as os from 'os';
import { EnvironmentConfig, TerminalContext, ENVIRONMENT_REGISTRY } from '../types/EnvironmentConfig';

export class EnvironmentManager {
  /** Map of terminals to their active environments */
  private activeEnvironments: Map<vscode.Terminal, EnvironmentConfig> = new Map();

  constructor(context: vscode.ExtensionContext) {
    // Listen for terminal lifecycle events
    context.subscriptions.push(
      vscode.window.onDidOpenTerminal(this.onTerminalOpened, this),
      vscode.window.onDidCloseTerminal(this.onTerminalClosed, this)
    );
  }

  /**
   * Check if venv path exists
   */
  private async checkVenvExists(venvPath: string): Promise<boolean> {
    const expandedPath = venvPath.replace('~', os.homedir());
    try {
      await fs.access(expandedPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Track terminal for environment status (does NOT auto-activate)
   *
   * @param terminal - Terminal to track
   * @param terminalContext - Terminal context (determines environment)
   */
  trackTerminal(
    terminal: vscode.Terminal,
    terminalContext: TerminalContext
  ): void {
    const envConfig = ENVIRONMENT_REGISTRY[terminalContext];

    // Store environment info
    this.activeEnvironments.set(terminal, envConfig);
  }

  /**
   * Set environment for a terminal (manually activates)
   *
   * @param terminal - Terminal to set environment for
   * @param terminalContext - Terminal context (determines environment)
   */
  async setEnvironment(
    terminal: vscode.Terminal,
    terminalContext: TerminalContext
  ): Promise<void> {
    const envConfig = ENVIRONMENT_REGISTRY[terminalContext];

    // Check if venv exists (for venv-based environments)
    if (envConfig.venvPath) {
      const exists = await this.checkVenvExists(envConfig.venvPath);
      if (!exists) {
        vscode.window.showWarningMessage(
          `Python environment not found: ${envConfig.displayName} (${envConfig.venvPath}). ` +
          `Run the installation command for this lesson first.`
        );
        // Still set the environment in tracking (user might create it later)
      }
    }

    // Store active environment
    this.activeEnvironments.set(terminal, envConfig);

    // Manually activate environment (user triggered)
    await this.activateEnvironment(terminal, envConfig);
  }

  /**
   * Activate environment in terminal
   */
  private async activateEnvironment(
    terminal: vscode.Terminal,
    envConfig: EnvironmentConfig
  ): Promise<void> {
    // Unset TT_METAL_HOME if needed (for TT-Forge)
    if (envConfig.unsetTTMetalHome) {
      terminal.sendText('unset TT_METAL_HOME && unset TT_METAL_VERSION');
    }

    // Send activation command
    if (envConfig.activationCommand) {
      terminal.sendText(envConfig.activationCommand);
    }
  }

  /**
   * Switch environment (with quick pick selection)
   *
   * @param terminal - Terminal to switch environment for
   */
  async switchEnvironment(terminal: vscode.Terminal): Promise<void> {
    // Build quick pick items
    const items: (vscode.QuickPickItem & { context: TerminalContext })[] = Object.entries(
      ENVIRONMENT_REGISTRY
    ).map(([context, config]) => ({
      label: `${config.icon || '$(python)'} ${config.displayName}`,
      description: config.description,
      detail: config.venvPath || config.activationCommand || 'System default',
      context: context as TerminalContext,
    }));

    // Show quick pick
    const selected = await vscode.window.showQuickPick(items, {
      placeHolder: 'Select Python environment for this terminal',
      title: `Switch Environment (${terminal.name})`,
      matchOnDescription: true,
      matchOnDetail: true,
    });

    if (selected) {
      await this.setEnvironment(terminal, selected.context);
    }
  }

  /**
   * Detect active environment by checking terminal name
   *
   * This is used for terminals that were created outside our control.
   * We match terminal names to known patterns.
   *
   * @param terminal - Terminal to detect environment for
   */
  async detectActiveEnvironment(terminal: vscode.Terminal): Promise<void> {
    const terminalName = terminal.name;

    // Try to match terminal name to known environments
    for (const [_context, config] of Object.entries(ENVIRONMENT_REGISTRY)) {
      if (terminalName.includes(config.displayName)) {
        this.activeEnvironments.set(terminal, config);
        return;
      }
    }
  }

  /**
   * Get current environment for terminal
   *
   * @param terminal - Terminal to get environment for
   * @returns Environment config, or undefined if not tracked
   */
  getActiveEnvironment(terminal: vscode.Terminal): EnvironmentConfig | undefined {
    return this.activeEnvironments.get(terminal);
  }

  /**
   * Terminal opened event handler
   */
  private onTerminalOpened(terminal: vscode.Terminal): void {
    // Detect environment for new terminals
    // (terminals created by our extension will have setEnvironment called explicitly)
    this.detectActiveEnvironment(terminal);
  }

  /**
   * Terminal closed event handler
   */
  private onTerminalClosed(terminal: vscode.Terminal): void {
    // Remove from active environments
    this.activeEnvironments.delete(terminal);
  }

  /**
   * Dispose all resources
   */
  dispose(): void {
    // Clear environment tracking
    this.activeEnvironments.clear();
  }
}
