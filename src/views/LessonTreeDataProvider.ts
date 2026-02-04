/**
 * Lesson Tree Data Provider
 *
 * Provides hierarchical lesson structure to VSCode TreeView.
 * Features:
 * - Category groupings
 * - Progress badges
 * - Hardware filtering
 * - Search/filter support
 */

import * as vscode from 'vscode';
import { LessonRegistry } from '../utils';
import { ProgressTracker } from '../state';
import {
  LessonMetadata,
  CategoryDefinition,
  LessonCategory,
  FilterOptions,
} from '../types';

/**
 * Tree item type
 */
export type LessonTreeItemType = 'category' | 'lesson' | 'special';

/**
 * Lesson tree item
 */
export class LessonTreeItem extends vscode.TreeItem {
  public readonly specialCommand?: string;

  constructor(
    public readonly type: LessonTreeItemType,
    public readonly lesson?: LessonMetadata,
    public readonly category?: CategoryDefinition,
    public readonly label?: string,
    public readonly collapsibleState?: vscode.TreeItemCollapsibleState,
    specialCommand?: string
  ) {
    super(
      label || lesson?.title || category?.title || 'Unknown',
      collapsibleState || vscode.TreeItemCollapsibleState.None
    );

    this.specialCommand = specialCommand;

    if (type === 'lesson' && lesson) {
      this.setupLessonItem(lesson);
    } else if (type === 'category' && category) {
      this.setupCategoryItem(category);
    } else if (type === 'special' && label && specialCommand) {
      this.setupSpecialItem(label, specialCommand);
    }
  }

  /**
   * Setup lesson tree item
   */
  private setupLessonItem(lesson: LessonMetadata): void {
    this.id = lesson.id;
    this.description = ''; // Keep description empty, only in tooltip
    this.tooltip = new vscode.MarkdownString(
      `**${lesson.title}**\n\n${lesson.description}\n\n` +
      `**Category:** ${lesson.category}\n` +
      `**Hardware:** ${lesson.supportedHardware.join(', ')}\n` +
      `**Status:** ${lesson.status}\n` +
      (lesson.estimatedMinutes ? `**Duration:** ~${lesson.estimatedMinutes} min\n` : '') +
      (lesson.tags.length > 0 ? `**Tags:** ${lesson.tags.join(', ')}` : '')
    );

    // Set icon based on lesson properties
    if (lesson.icon) {
      this.iconPath = new vscode.ThemeIcon(lesson.icon);
    } else {
      // Default icon based on status
      switch (lesson.status) {
        case 'validated':
          this.iconPath = new vscode.ThemeIcon('check');
          break;
        case 'draft':
          this.iconPath = new vscode.ThemeIcon('edit');
          break;
        case 'blocked':
          this.iconPath = new vscode.ThemeIcon('warning');
          break;
      }
    }

    // Make it clickable
    this.command = {
      command: 'tenstorrent.showLesson',
      title: 'Show Lesson',
      arguments: [lesson.id],
    };

    // Context value for menus
    this.contextValue = 'lesson';
  }

  /**
   * Setup category tree item
   */
  private setupCategoryItem(category: CategoryDefinition): void {
    this.id = `category-${category.id}`;
    this.description = ''; // Keep description empty
    this.tooltip = category.description;

    // Set icon
    if (category.icon) {
      this.iconPath = new vscode.ThemeIcon(category.icon);
    } else {
      this.iconPath = new vscode.ThemeIcon('folder');
    }

    // Context value for menus
    this.contextValue = 'category';
  }

  /**
   * Setup special tree item (Welcome, FAQ, etc.)
   */
  private setupSpecialItem(label: string, commandId: string): void {
    this.id = `special-${commandId}`;
    this.tooltip = label;

    // Set icon based on label
    if (label.includes('Welcome')) {
      this.iconPath = new vscode.ThemeIcon('home');
    } else if (label.includes('FAQ')) {
      this.iconPath = new vscode.ThemeIcon('question');
    } else {
      this.iconPath = new vscode.ThemeIcon('book');
    }

    // Make it clickable
    this.command = {
      command: commandId,
      title: label,
      arguments: [],
    };

    // Context value for menus
    this.contextValue = 'special';
  }

  /**
   * Add progress badge to label
   */
  addProgressBadge(_status: 'not-started' | 'in-progress' | 'completed'): void {
    // No visual badges - keep clean UI
    this.description = '';
  }
}

/**
 * Lesson Tree Data Provider
 */
export class LessonTreeDataProvider implements vscode.TreeDataProvider<LessonTreeItem> {
  private _onDidChangeTreeData = new vscode.EventEmitter<LessonTreeItem | undefined | null | void>();
  readonly onDidChangeTreeData = this._onDidChangeTreeData.event;

  private lessonRegistry: LessonRegistry;
  private progressTracker: ProgressTracker;
  private filterOptions: FilterOptions = {};

  constructor(lessonRegistry: LessonRegistry, progressTracker: ProgressTracker) {
    this.lessonRegistry = lessonRegistry;
    this.progressTracker = progressTracker;

    // Apply default filter based on configuration
    this.applyConfigurationFilter();

    // Listen to progress changes
    progressTracker.onDidChangeProgress(() => {
      this.refresh();
    });

    // Listen to theme changes to update logo
    vscode.window.onDidChangeActiveColorTheme(() => {
      this.refresh();
    });

    // Listen to configuration changes
    vscode.workspace.onDidChangeConfiguration(e => {
      if (e.affectsConfiguration('tenstorrent.showUnvalidatedLessons')) {
        this.applyConfigurationFilter();
        this.refresh();
      }
    });
  }

  /**
   * Apply filter based on configuration setting
   */
  private applyConfigurationFilter(): void {
    const config = vscode.workspace.getConfiguration('tenstorrent');
    const showUnvalidated = config.get<boolean>('showUnvalidatedLessons', true);

    // If showUnvalidatedLessons is false, filter to validated only
    if (!showUnvalidated) {
      this.filterOptions.validatedOnly = true;
    } else {
      // If explicitly enabled, remove the validated-only filter
      this.filterOptions.validatedOnly = false;
    }
  }

  /**
   * Refresh tree view
   */
  refresh(): void {
    this._onDidChangeTreeData.fire();
  }

  /**
   * Get tree item
   */
  getTreeItem(element: LessonTreeItem): vscode.TreeItem {
    return element;
  }

  /**
   * Get children for tree node
   */
  async getChildren(element?: LessonTreeItem): Promise<LessonTreeItem[]> {
    if (!element) {
      // Root level - add logo header, then categories
      const items: LessonTreeItem[] = [];

      // Add logo header at the top
      const logoItem = this.createLogoHeaderItem();
      items.push(logoItem);

      // Add categories
      const categoryItems = this.getCategoryItems();
      items.push(...categoryItems);

      return items;
    }

    if (element.type === 'category' && element.category) {
      // Category level - return lessons in that category
      return this.getLessonItems(element.category.id);
    }

    return [];
  }

  /**
   * Get category tree items
   */
  private getCategoryItems(): LessonTreeItem[] {
    const categories = this.lessonRegistry.getCategories();
    const items: LessonTreeItem[] = [];

    for (const category of categories) {
      const lessons = this.filterLessons(this.lessonRegistry.getByCategory(category.id));

      // Always show welcome category (has special items), or categories with lessons
      if (category.id === 'welcome' || lessons.length > 0) {
        const item = new LessonTreeItem(
          'category',
          undefined,
          category,
          category.title,
          vscode.TreeItemCollapsibleState.Expanded
        );
        items.push(item);
      }
    }

    return items;
  }

  /**
   * Create logo header item for the top of the tree
   */
  private createLogoHeaderItem(): LessonTreeItem {
    const item = new LessonTreeItem(
      'special',
      undefined,
      undefined,
      '━━━  T E N S T O R R E N T  ━━━', // Visual separator with branding
      vscode.TreeItemCollapsibleState.None
    );

    // Select logo based on theme
    const themeKind = vscode.window.activeColorTheme.kind;
    const isDark = themeKind === vscode.ColorThemeKind.Dark || themeKind === vscode.ColorThemeKind.HighContrast;

    // Use symbol logo (smaller, cleaner)
    const logoFileName = isDark ? 'tt_symbol_purple.svg' : 'tt_symbol_black.svg';

    // Set icon path to the logo symbol
    const extensionPath = this.lessonRegistry.getExtensionPath();
    item.iconPath = vscode.Uri.file(`${extensionPath}/assets/img/${logoFileName}`);

    // Style as a separator/header
    item.description = '';
    item.tooltip = 'Tenstorrent Developer Extension';
    item.contextValue = 'logo-header'; // For styling purposes

    // Make it clickable to open welcome page
    item.command = {
      command: 'tenstorrent.showWelcome',
      title: 'Open Welcome Page',
      arguments: [],
    };

    return item;
  }

  /**
   * Get lesson tree items for a category
   */
  private getLessonItems(categoryId: LessonCategory): LessonTreeItem[] {
    const items: LessonTreeItem[] = [];

    // Add special items for welcome category
    if (categoryId === 'welcome') {
      // Welcome page
      items.push(new LessonTreeItem(
        'special',
        undefined,
        undefined,
        '🏠 Welcome Page',
        vscode.TreeItemCollapsibleState.None,
        'tenstorrent.showWelcome'
      ));

      // FAQ
      items.push(new LessonTreeItem(
        'special',
        undefined,
        undefined,
        '❓ Frequently Asked Questions',
        vscode.TreeItemCollapsibleState.None,
        'tenstorrent.showFaq'
      ));

      return items;
    }

    // Regular lessons
    const lessons = this.filterLessons(this.lessonRegistry.getByCategory(categoryId));

    for (const lesson of lessons) {
      const item = new LessonTreeItem(
        'lesson',
        lesson,
        undefined,
        lesson.title,
        vscode.TreeItemCollapsibleState.None
      );

      // Add progress badge
      const progress = this.progressTracker.getProgress(lesson.id);
      item.addProgressBadge(progress.status);

      items.push(item);
    }

    return items;
  }

  /**
   * Filter lessons based on current filter options
   */
  private filterLessons(lessons: LessonMetadata[]): LessonMetadata[] {
    let filtered = [...lessons];

    // Text search
    if (this.filterOptions.query) {
      const query = this.filterOptions.query.toLowerCase();
      filtered = filtered.filter(
        lesson =>
          lesson.title.toLowerCase().includes(query) ||
          lesson.description.toLowerCase().includes(query) ||
          lesson.tags.some(tag => tag.toLowerCase().includes(query))
      );
    }

    // Hardware filter
    if (this.filterOptions.hardware && this.filterOptions.hardware.length > 0) {
      filtered = filtered.filter(lesson =>
        this.filterOptions.hardware!.some(hw => lesson.supportedHardware.includes(hw))
      );
    }

    // Status filter
    if (this.filterOptions.status && this.filterOptions.status.length > 0) {
      filtered = filtered.filter(lesson =>
        this.filterOptions.status!.includes(lesson.status)
      );
    }

    // Category filter
    if (this.filterOptions.categories && this.filterOptions.categories.length > 0) {
      filtered = filtered.filter(lesson =>
        this.filterOptions.categories!.includes(lesson.category)
      );
    }

    // Tags filter
    if (this.filterOptions.tags && this.filterOptions.tags.length > 0) {
      filtered = filtered.filter(lesson =>
        this.filterOptions.tags!.some(tag => lesson.tags.includes(tag))
      );
    }

    // Progress filter
    if (this.filterOptions.progress && this.filterOptions.progress.length > 0) {
      filtered = filtered.filter(lesson => {
        const progress = this.progressTracker.getProgress(lesson.id);
        return this.filterOptions.progress!.includes(progress.status);
      });
    }

    // Validated only
    if (this.filterOptions.validatedOnly) {
      filtered = filtered.filter(lesson => lesson.status === 'validated');
    }

    return filtered;
  }

  /**
   * Apply filter options
   */
  applyFilter(options: FilterOptions): void {
    this.filterOptions = options;
    this.refresh();
  }

  /**
   * Clear all filters
   */
  clearFilters(): void {
    this.filterOptions = {};
    this.refresh();
  }

  /**
   * Get current filter options
   */
  getFilterOptions(): FilterOptions {
    return { ...this.filterOptions };
  }
}
