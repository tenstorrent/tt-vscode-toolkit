/**
 * Progress Tracking Types
 *
 * Tracks user progress through lessons, completed commands, and lesson state.
 * Stored in VSCode ExtensionContext globalState.
 */

/**
 * Lesson progress status
 */
export type ProgressStatus =
  | 'not-started'   // User hasn't opened this lesson yet
  | 'in-progress'   // User has started but not completed
  | 'completed';    // User has completed all required commands

/**
 * Progress state for a single lesson
 */
export interface ProgressState {
  /** Lesson ID this progress is for */
  lessonId: string;

  /** Current completion status */
  status: ProgressStatus;

  /** List of command IDs that have been executed */
  completedCommands: string[];

  /** Timestamp of when lesson was first opened (ms since epoch) */
  firstAccessed?: number;

  /** Timestamp of last access (ms since epoch) */
  lastAccessed: number;

  /** Total time spent on this lesson (seconds) */
  timeSpentSeconds: number;

  /** Number of times user has opened this lesson */
  viewCount: number;

  /** Optional user notes */
  notes?: string;
}

/**
 * Global progress tracking for all lessons
 */
export interface GlobalProgressState {
  /** Map of lessonId → ProgressState */
  lessons: Record<string, ProgressState>;

  /** Last time progress was synced (ms since epoch) */
  lastSync: number;

  /** Total lessons completed */
  totalCompleted: number;

  /** Total lessons in progress */
  totalInProgress: number;

  /** Total time spent across all lessons (seconds) */
  totalTimeSpent: number;
}

/**
 * Progress change event
 */
export interface ProgressChangeEvent {
  /** Lesson ID that changed */
  lessonId: string;

  /** Previous status */
  oldStatus: ProgressStatus;

  /** New status */
  newStatus: ProgressStatus;

  /** Command that triggered the change (if any) */
  triggeringCommand?: string;

  /** Timestamp of change */
  timestamp: number;
}

/**
 * Progress statistics
 */
export interface ProgressStatistics {
  /** Total lessons */
  totalLessons: number;

  /** Completed lessons */
  completedLessons: number;

  /** In-progress lessons */
  inProgressLessons: number;

  /** Not-started lessons */
  notStartedLessons: number;

  /** Completion percentage (0-100) */
  completionPercentage: number;

  /** Total time spent (seconds) */
  totalTimeSpent: number;

  /** Average time per lesson (seconds) */
  averageTimePerLesson: number;

  /** Most recently accessed lesson */
  mostRecentLesson?: string;

  /** Longest lesson by time spent */
  longestLesson?: string;
}
