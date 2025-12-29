/**
 * Lesson Registry
 *
 * Manages loading, caching, and querying lesson metadata from lesson-registry.json.
 * Provides centralized access to all lesson information for TreeView and Webview.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';
import {
  LessonRegistry as LessonRegistryType,
  LessonMetadata,
  CategoryDefinition,
  HardwareType,
  LessonCategory,
} from '../types';

/**
 * Manages lesson registry loading and querying
 */
export class LessonRegistry {
  private registry: LessonRegistryType | null = null;
  private lessonMap: Map<string, LessonMetadata> = new Map();
  private categoryMap: Map<LessonCategory, CategoryDefinition> = new Map();
  private extensionContext: vscode.ExtensionContext;

  constructor(context: vscode.ExtensionContext) {
    this.extensionContext = context;
  }

  /**
   * Load lesson registry from JSON file
   */
  async load(): Promise<void> {
    try {
      const registryPath = path.join(
        this.extensionContext.extensionPath,
        'content',
        'lesson-registry.json'
      );

      if (!fs.existsSync(registryPath)) {
        throw new Error(`Lesson registry not found at: ${registryPath}`);
      }

      const content = fs.readFileSync(registryPath, 'utf-8');
      this.registry = JSON.parse(content) as LessonRegistryType;

      // Build maps for fast lookup
      this.buildMaps();

      console.log(`Loaded ${this.registry.lessons.length} lessons from registry`);
    } catch (error) {
      vscode.window.showErrorMessage(
        `Failed to load lesson registry: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
      throw error;
    }
  }

  /**
   * Build internal maps for fast lookup
   */
  private buildMaps(): void {
    if (!this.registry) {
      return;
    }

    // Build lesson map
    this.lessonMap.clear();
    for (const lesson of this.registry.lessons) {
      this.lessonMap.set(lesson.id, lesson);
    }

    // Build category map
    this.categoryMap.clear();
    for (const category of this.registry.categories) {
      this.categoryMap.set(category.id, category);
    }
  }

  /**
   * Get a lesson by ID
   */
  get(lessonId: string): LessonMetadata | undefined {
    return this.lessonMap.get(lessonId);
  }

  /**
   * Get all lessons
   */
  getAll(): LessonMetadata[] {
    return this.registry?.lessons ?? [];
  }

  /**
   * Get lessons by category
   */
  getByCategory(category: LessonCategory): LessonMetadata[] {
    return this.getAll()
      .filter(lesson => lesson.category === category)
      .sort((a, b) => a.order - b.order);
  }

  /**
   * Get all categories, sorted by order
   */
  getCategories(): CategoryDefinition[] {
    return this.registry?.categories.sort((a, b) => a.order - b.order) ?? [];
  }

  /**
   * Get a category definition
   */
  getCategory(categoryId: LessonCategory): CategoryDefinition | undefined {
    return this.categoryMap.get(categoryId);
  }

  /**
   * Get lessons by hardware type
   */
  getByHardware(hardware: HardwareType): LessonMetadata[] {
    return this.getAll().filter(lesson =>
      lesson.supportedHardware.includes(hardware)
    );
  }

  /**
   * Get lessons by tag
   */
  getByTag(tag: string): LessonMetadata[] {
    return this.getAll().filter(lesson =>
      lesson.tags.includes(tag)
    );
  }

  /**
   * Get lessons by tags (must have all tags)
   */
  getByTags(tags: string[]): LessonMetadata[] {
    return this.getAll().filter(lesson =>
      tags.every(tag => lesson.tags.includes(tag))
    );
  }

  /**
   * Get validated lessons only
   */
  getValidated(): LessonMetadata[] {
    return this.getAll().filter(lesson => lesson.status === 'validated');
  }

  /**
   * Get lessons by status
   */
  getByStatus(status: 'validated' | 'draft' | 'blocked'): LessonMetadata[] {
    return this.getAll().filter(lesson => lesson.status === status);
  }

  /**
   * Search lessons by query (fuzzy search on title, description, tags)
   */
  search(query: string): LessonMetadata[] {
    if (!query.trim()) {
      return this.getAll();
    }

    const lowerQuery = query.toLowerCase();

    return this.getAll().filter(lesson => {
      return (
        lesson.title.toLowerCase().includes(lowerQuery) ||
        lesson.description.toLowerCase().includes(lowerQuery) ||
        lesson.tags.some(tag => tag.toLowerCase().includes(lowerQuery)) ||
        lesson.id.toLowerCase().includes(lowerQuery)
      );
    });
  }

  /**
   * Get lessons organized by category
   */
  getOrganized(): Map<LessonCategory, LessonMetadata[]> {
    const organized = new Map<LessonCategory, LessonMetadata[]>();

    for (const category of this.getCategories()) {
      organized.set(category.id, this.getByCategory(category.id));
    }

    return organized;
  }

  /**
   * Get previous lesson in sequence
   */
  getPrevious(lessonId: string): LessonMetadata | undefined {
    const lesson = this.get(lessonId);
    if (!lesson?.previousLesson) {
      return undefined;
    }
    return this.get(lesson.previousLesson);
  }

  /**
   * Get next lesson in sequence
   */
  getNext(lessonId: string): LessonMetadata | undefined {
    const lesson = this.get(lessonId);
    if (!lesson?.nextLesson) {
      return undefined;
    }
    return this.get(lesson.nextLesson);
  }

  /**
   * Check if prerequisites are met for a lesson
   */
  arePrerequisitesMet(
    lessonId: string,
    completedLessons: Set<string>
  ): boolean {
    const lesson = this.get(lessonId);
    if (!lesson?.prerequisites || lesson.prerequisites.length === 0) {
      return true;
    }

    // Check required prerequisites
    const requiredPrereqs = lesson.prerequisites.filter(p => p.required);
    return requiredPrereqs.every(prereq =>
      completedLessons.has(prereq.lessonId)
    );
  }

  /**
   * Get unmet prerequisites for a lesson
   */
  getUnmetPrerequisites(
    lessonId: string,
    completedLessons: Set<string>
  ): LessonMetadata[] {
    const lesson = this.get(lessonId);
    if (!lesson?.prerequisites) {
      return [];
    }

    return lesson.prerequisites
      .filter(prereq => prereq.required && !completedLessons.has(prereq.lessonId))
      .map(prereq => this.get(prereq.lessonId))
      .filter((l): l is LessonMetadata => l !== undefined);
  }

  /**
   * Get total lesson count
   */
  getTotalCount(): number {
    return this.getAll().length;
  }

  /**
   * Get registry version
   */
  getVersion(): string {
    return this.registry?.version ?? 'unknown';
  }

  /**
   * Reload registry from disk
   */
  async reload(): Promise<void> {
    await this.load();
  }

  /**
   * Get extension path
   */
  getExtensionPath(): string {
    return this.extensionContext.extensionPath;
  }
}
