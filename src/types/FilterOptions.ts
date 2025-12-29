/**
 * Filter and Search Types
 *
 * Defines filtering and search options for lesson discovery and organization.
 */

import { HardwareType, LessonStatus, LessonCategory } from './LessonMetadata';
import { ProgressStatus } from './ProgressState';

/**
 * Filter options for lesson discovery
 */
export interface FilterOptions {
  /** Text search query (searches title, description, tags) */
  query?: string;

  /** Filter by hardware compatibility */
  hardware?: HardwareType[];

  /** Filter by validation status */
  status?: LessonStatus[];

  /** Filter by category */
  categories?: LessonCategory[];

  /** Filter by tags */
  tags?: string[];

  /** Filter by progress status */
  progress?: ProgressStatus[];

  /** Show only validated lessons */
  validatedOnly?: boolean;

  /** Show only lessons with prerequisites met */
  prerequisitesMet?: boolean;

  /** Minimum tt-metal version (if user's version is known) */
  minTTMetalVersion?: string;
}

/**
 * Sort options for lesson display
 */
export type SortOption =
  | 'order'         // Default order (category, then order field)
  | 'alphabetical'  // A-Z by title
  | 'recent'        // Most recently accessed
  | 'progress'      // By completion status
  | 'duration';     // By estimated duration

/**
 * Sort direction
 */
export type SortDirection = 'asc' | 'desc';

/**
 * Complete filter and sort configuration
 */
export interface FilterConfig {
  /** Filter options */
  filters: FilterOptions;

  /** Sort option */
  sortBy: SortOption;

  /** Sort direction */
  sortDirection: SortDirection;

  /** Whether to show category groupings */
  showCategories: boolean;

  /** Whether to show progress badges */
  showProgress: boolean;
}

/**
 * Filter preset for quick access
 */
export interface FilterPreset {
  /** Preset ID */
  id: string;

  /** Display name */
  name: string;

  /** Icon (codicon name) */
  icon?: string;

  /** Filter configuration */
  config: FilterConfig;

  /** Whether this is a built-in preset */
  builtin: boolean;
}

/**
 * Search result with relevance scoring
 */
export interface SearchResult {
  /** Lesson ID */
  lessonId: string;

  /** Relevance score (0-1, higher = more relevant) */
  relevance: number;

  /** Matching fields */
  matchedFields: string[];

  /** Text snippets showing matches */
  highlights?: string[];
}
