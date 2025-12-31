/**
 * EnvironmentManager - Manages Python environments for each terminal
 *
 * Features:
 * - Tracks which environment is active in each terminal
 * - Shows status bar indicator with environment name and icon
 * - Auto-activates correct environment when terminal is created
 * - Allows manual switching via status bar click or command palette
 *
 * This solves the "environment drift" problem where users lose track
 * of which Python venv is active.
 */

import * as vscode from 'vscode';
import * as fs from 'fs/promises';
import * as os from 'os';
import { EnvironmentConfig, TerminalContext, ENVIRONMENT_REGISTRY } from '../types/EnvironmentConfig';

export class EnvironmentManager {
  /** Map of terminals to their active environments */
  private activeEnvironments: Map<vscode.Terminal, EnvironmentConfig> = new Map();

  /** Map of terminals to their status bar items */
  private statusBarItems: Map<vscode.Terminal, vscode.StatusBarItem> = new Map();

  /** Extension context for subscriptions */
  private context: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;

    // Listen for terminal lifecycle events
    context.subscriptions.push(
      vscode.window.onDidOpenTerminal(this.onTerminalOpened, this),
      vscode.window.onDidCloseTerminal(this.onTerminalClosed, this),
      vscode.window.onDidChangeActiveTerminal(this.onActiveTerminalChanged, this)
    );
  }

  /**
   * Get or create status bar item for a terminal
   */
  private getOrCreateStatusBarItem(terminal: vscode.Terminal): vscode.StatusBarItem {
    let statusBarItem = this.statusBarItems.get(terminal);

    if (!statusBarItem) {
      statusBarItem = vscode.window.createStatusBarItem(
        vscode.StatusBarAlignment.Left,
        100 // Priority (higher = more left)
      );

      statusBarItem.command = 'tenstorrent.selectPythonEnvironment';
      statusBarItem.tooltip = 'Click to change Python environment';

      this.statusBarItems.set(terminal, statusBarItem);
      this.context.subscriptions.push(statusBarItem);
    }

    return statusBarItem;
  }

  /**
   * Update status bar to show current environment
   */
  private updateStatusBar(terminal: vscode.Terminal): void {
    const statusBarItem = this.getOrCreateStatusBarItem(terminal);
    const activeEnv = this.activeEnvironments.get(terminal);

    if (activeEnv) {
      const icon = activeEnv.icon || '$(python)';
      statusBarItem.text = `${icon} ${activeEnv.displayName}`;
      statusBarItem.show();
    } else {
      statusBarItem.text = '$(python) Unknown Env';
      statusBarItem.show();
    }
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
   * Set environment for a terminal (auto-activates)
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
    this.updateStatusBar(terminal);

    // Auto-activate environment
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

    // Show confirmation (only for non-system environments)
    if (envConfig.id !== 'system') {
      vscode.window.showInformationMessage(
        `✓ Activated ${envConfig.displayName} in terminal "${terminal.name}"`
      );
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
        this.updateStatusBar(terminal);
        return;
      }
    }

    // Fallback: mark as unknown environment
    this.updateStatusBar(terminal);
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
    // Clean up status bar item
    const statusBarItem = this.statusBarItems.get(terminal);
    if (statusBarItem) {
      statusBarItem.dispose();
      this.statusBarItems.delete(terminal);
    }

    // Remove from active environments
    this.activeEnvironments.delete(terminal);
  }

  /**
   * Active terminal changed event handler
   */
  private onActiveTerminalChanged(terminal: vscode.Terminal | undefined): void {
    // Hide all status bars except for active terminal
    for (const [term, statusBarItem] of this.statusBarItems.entries()) {
      if (term === terminal) {
        statusBarItem.show();
      } else {
        statusBarItem.hide();
      }
    }
  }

  /**
   * Dispose all resources
   */
  dispose(): void {
    // Dispose all status bar items
    for (const statusBarItem of this.statusBarItems.values()) {
      statusBarItem.dispose();
    }
    this.statusBarItems.clear();
    this.activeEnvironments.clear();
  }
}
