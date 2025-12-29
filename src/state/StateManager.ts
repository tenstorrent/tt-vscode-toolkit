/**
 * State Manager
 *
 * Central state management for the extension.
 * Handles:
 * - Current lesson selection
 * - Filter state
 * - UI preferences
 * - Extension settings
 */

import * as vscode from 'vscode';

/**
 * Extension state keys for globalState storage
 */
const STATE_KEYS = {
  CURRENT_LESSON: 'tenstorrent.currentLesson',
  FILTER_STATE: 'tenstorrent.filterState',
  TREE_EXPANDED: 'tenstorrent.treeExpanded',
  LAST_HARDWARE: 'tenstorrent.lastHardware',
  UI_PREFERENCES: 'tenstorrent.uiPreferences',
} as const;

/**
 * UI preferences
 */
export interface UIPreferences {
  /** Show progress badges in tree */
  showProgress: boolean;

  /** Show category groupings */
  showCategories: boolean;

  /** Theme preference (light/dark/auto) */
  theme: 'light' | 'dark' | 'auto';

  /** Font size for lessons */
  fontSize: number;

  /** Whether to auto-open next lesson on completion */
  autoAdvance: boolean;
}

/**
 * Central state manager
 */
export class StateManager {
  private context: vscode.ExtensionContext;
  private _onDidChangeState = new vscode.EventEmitter<string>();
  public readonly onDidChangeState = this._onDidChangeState.event;

  constructor(context: vscode.ExtensionContext) {
    this.context = context;
  }

  /**
   * Get current selected lesson ID
   */
  getCurrentLesson(): string | undefined {
    return this.context.globalState.get<string>(STATE_KEYS.CURRENT_LESSON);
  }

  /**
   * Set current selected lesson ID
   */
  async setCurrentLesson(lessonId: string): Promise<void> {
    await this.context.globalState.update(STATE_KEYS.CURRENT_LESSON, lessonId);
    this._onDidChangeState.fire('currentLesson');
  }

  /**
   * Get filter state
   */
  getFilterState(): any {
    return this.context.globalState.get(STATE_KEYS.FILTER_STATE, {});
  }

  /**
   * Set filter state
   */
  async setFilterState(filterState: any): Promise<void> {
    await this.context.globalState.update(STATE_KEYS.FILTER_STATE, filterState);
    this._onDidChangeState.fire('filterState');
  }

  /**
   * Get tree expanded state
   */
  getTreeExpanded(): Record<string, boolean> {
    return this.context.globalState.get(STATE_KEYS.TREE_EXPANDED, {});
  }

  /**
   * Set tree expanded state for a category
   */
  async setTreeExpanded(categoryId: string, expanded: boolean): Promise<void> {
    const state = this.getTreeExpanded();
    state[categoryId] = expanded;
    await this.context.globalState.update(STATE_KEYS.TREE_EXPANDED, state);
    this._onDidChangeState.fire('treeExpanded');
  }

  /**
   * Get last detected hardware
   */
  getLastHardware(): string | undefined {
    return this.context.globalState.get<string>(STATE_KEYS.LAST_HARDWARE);
  }

  /**
   * Set last detected hardware
   */
  async setLastHardware(hardware: string): Promise<void> {
    await this.context.globalState.update(STATE_KEYS.LAST_HARDWARE, hardware);
    this._onDidChangeState.fire('lastHardware');
  }

  /**
   * Get UI preferences
   */
  getUIPreferences(): UIPreferences {
    return this.context.globalState.get<UIPreferences>(
      STATE_KEYS.UI_PREFERENCES,
      {
        showProgress: true,
        showCategories: true,
        theme: 'auto',
        fontSize: 14,
        autoAdvance: false,
      }
    );
  }

  /**
   * Set UI preferences
   */
  async setUIPreferences(preferences: Partial<UIPreferences>): Promise<void> {
    const current = this.getUIPreferences();
    const updated = { ...current, ...preferences };
    await this.context.globalState.update(STATE_KEYS.UI_PREFERENCES, updated);
    this._onDidChangeState.fire('uiPreferences');
  }

  /**
   * Clear all state (useful for debugging/reset)
   */
  async clearAll(): Promise<void> {
    for (const key of Object.values(STATE_KEYS)) {
      await this.context.globalState.update(key, undefined);
    }
    this._onDidChangeState.fire('all');
  }

  /**
   * Export all state for debugging
   */
  exportState(): Record<string, any> {
    return {
      currentLesson: this.getCurrentLesson(),
      filterState: this.getFilterState(),
      treeExpanded: this.getTreeExpanded(),
      lastHardware: this.getLastHardware(),
      uiPreferences: this.getUIPreferences(),
    };
  }
}
