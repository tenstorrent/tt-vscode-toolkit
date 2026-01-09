/**
 * Lesson Metadata Types
 *
 * Defines the structure for lesson content, organization, and metadata.
 * Used by TreeView, Webview, and content management systems.
 */

/**
 * Hardware types supported by Tenstorrent
 */
export type HardwareType =
  | 'n150'    // Wormhole - Single Chip
  | 'n300'    // Wormhole - Dual Chip
  | 't3k'     // Wormhole - 8 Chips
  | 'p100'    // Blackhole - Single Chip
  | 'p150'    // Blackhole - Dual Chip
  | 'p300'    // Blackhole - Single Chip (QuietBox)
  | 'galaxy'  // Galaxy configuration
  | 'simulator'; // Simulator mode

/**
 * Lesson validation status
 */
export type LessonStatus =
  | 'validated'  // Tested and ready for production
  | 'draft'      // In development
  | 'blocked';   // Known issues preventing use

/**
 * Lesson category for organization
 */
export type LessonCategory =
  | 'welcome'           // Welcome page, FAQ, getting started resources
  | 'first-inference'   // Setup, first model, basic inference
  | 'serving'           // Production deployment, serving infrastructure
  | 'ecosystem';        // Advanced topics, tools, community

/**
 * Prerequisite lesson requirement
 */
export interface PrerequisiteInfo {
  /** ID of the prerequisite lesson */
  lessonId: string;

  /** Whether this prerequisite is required (true) or optional (false) */
  required: boolean;

  /** Optional description of why this prerequisite matters */
  description?: string;
}

/**
 * Complete lesson metadata
 */
export interface LessonMetadata {
  // Identification
  /** Unique lesson identifier (e.g., "01-hardware-detection") */
  id: string;

  /** Display title for the lesson */
  title: string;

  /** Short description shown in tree view */
  description: string;

  // Organization
  /** Category for hierarchical organization */
  category: LessonCategory;

  /** Optional subcategory for nested organization */
  subcategory?: string;

  /** Display order within category (lower = earlier) */
  order: number;

  // Content
  /** Path to markdown content file (relative to content/ directory) */
  markdownFile: string;

  // Hardware compatibility
  /** List of hardware types this lesson supports */
  supportedHardware: HardwareType[];

  /** Validation status and testing */
  status: LessonStatus;

  /** Hardware types where this lesson has been validated */
  validatedOn: HardwareType[];

  /** If blocked, reason for blocking */
  blockReason?: string;

  /** Minimum tt-metal version required (e.g., "v0.51.0") */
  minTTMetalVersion?: string;

  // Progress tracking
  /** Command IDs that mark lesson completion */
  completionEvents: string[];

  // Discovery and filtering
  /** Tags for search and filtering (e.g., ["vllm", "production", "chat"]) */
  tags: string[];

  // Navigation and prerequisites
  /** Lessons that should be completed before this one */
  prerequisites?: PrerequisiteInfo[];

  /** ID of previous lesson in sequence */
  previousLesson?: string;

  /** ID of next lesson in sequence */
  nextLesson?: string;

  /** Estimated time to complete (in minutes) */
  estimatedMinutes?: number;

  // UI hints
  /** Icon name for tree view (codicon name) */
  icon?: string;

  /** Whether to expand this lesson's content by default */
  expandByDefault?: boolean;
}

/**
 * Category definition for lesson organization
 */
export interface CategoryDefinition {
  /** Unique category ID */
  id: LessonCategory;

  /** Display title */
  title: string;

  /** Short description */
  description: string;

  /** Display order (lower = earlier) */
  order: number;

  /** Icon for tree view (codicon name) */
  icon?: string;
}

/**
 * Root structure for lesson-registry.json
 */
export interface LessonRegistry {
  /** Registry format version */
  version: string;

  /** Category definitions */
  categories: CategoryDefinition[];

  /** All lessons */
  lessons: LessonMetadata[];

  /** Last update timestamp */
  lastUpdated?: string;
}
