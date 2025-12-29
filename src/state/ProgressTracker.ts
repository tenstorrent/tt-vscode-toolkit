/**
 * Progress Tracker
 *
 * Tracks user progress through lessons:
 * - Command execution
 * - Lesson completion
 * - Time spent
 * - Statistics
 *
 * Integrates with existing command system to auto-update progress.
 */

import * as vscode from 'vscode';
import {
  ProgressState,
  GlobalProgressState,
  ProgressChangeEvent,
  ProgressStatistics,
} from '../types';
import { LessonMetadata } from '../types';
import { StateManager } from './StateManager';

const PROGRESS_STATE_KEY = 'tenstorrent.progress';

/**
 * Tracks lesson progress and statistics
 */
export class ProgressTracker {
  private context: vscode.ExtensionContext;
  private _onDidChangeProgress = new vscode.EventEmitter<ProgressChangeEvent>();
  public readonly onDidChangeProgress = this._onDidChangeProgress.event;

  // Session tracking
  private currentSession: {
    lessonId: string;
    startTime: number;
  } | null = null;

  constructor(context: vscode.ExtensionContext, _stateManager: StateManager) {
    this.context = context;
  }

  /**
   * Get global progress state
   */
  private getGlobalState(): GlobalProgressState {
    return this.context.globalState.get<GlobalProgressState>(
      PROGRESS_STATE_KEY,
      {
        lessons: {},
        lastSync: Date.now(),
        totalCompleted: 0,
        totalInProgress: 0,
        totalTimeSpent: 0,
      }
    );
  }

  /**
   * Save global progress state
   */
  private async saveGlobalState(state: GlobalProgressState): Promise<void> {
    state.lastSync = Date.now();
    await this.context.globalState.update(PROGRESS_STATE_KEY, state);
  }

  /**
   * Get progress for a specific lesson
   */
  getProgress(lessonId: string): ProgressState {
    const globalState = this.getGlobalState();
    return (
      globalState.lessons[lessonId] ?? {
        lessonId,
        status: 'not-started',
        completedCommands: [],
        lastAccessed: 0,
        timeSpentSeconds: 0,
        viewCount: 0,
      }
    );
  }

  /**
   * Start tracking a lesson session
   */
  startSession(lessonId: string): void {
    // End previous session if any
    if (this.currentSession) {
      this.endSession();
    }

    this.currentSession = {
      lessonId,
      startTime: Date.now(),
    };

    // Update view count and last accessed
    this.updateLessonAccess(lessonId);
  }

  /**
   * End current lesson session
   */
  endSession(): void {
    if (!this.currentSession) {
      return;
    }

    const duration = Math.floor((Date.now() - this.currentSession.startTime) / 1000);
    const lessonId = this.currentSession.lessonId;

    // Update time spent
    const globalState = this.getGlobalState();
    const progress = this.getProgress(lessonId);
    progress.timeSpentSeconds += duration;
    globalState.lessons[lessonId] = progress;
    globalState.totalTimeSpent += duration;

    this.saveGlobalState(globalState);
    this.currentSession = null;
  }

  /**
   * Update lesson access (view count, last accessed)
   */
  private async updateLessonAccess(lessonId: string): Promise<void> {
    const globalState = this.getGlobalState();
    const progress = this.getProgress(lessonId);

    if (!progress.firstAccessed) {
      progress.firstAccessed = Date.now();
    }
    progress.lastAccessed = Date.now();
    progress.viewCount = (progress.viewCount || 0) + 1;

    // Update status to in-progress if not started
    if (progress.status === 'not-started') {
      progress.status = 'in-progress';
      globalState.totalInProgress++;
    }

    globalState.lessons[lessonId] = progress;
    await this.saveGlobalState(globalState);
  }

  /**
   * Record a command execution
   */
  async recordCommandExecution(
    lessonId: string,
    commandId: string,
    lesson: LessonMetadata
  ): Promise<void> {
    const globalState = this.getGlobalState();
    const progress = this.getProgress(lessonId);
    const oldStatus = progress.status;

    // Add command if not already recorded
    if (!progress.completedCommands.includes(commandId)) {
      progress.completedCommands.push(commandId);
    }

    // Check if lesson is now complete
    const isComplete = this.isLessonComplete(lesson, progress);
    if (isComplete && progress.status !== 'completed') {
      progress.status = 'completed';

      // Update global counts
      if (oldStatus === 'in-progress') {
        globalState.totalInProgress--;
      } else if (oldStatus === 'not-started') {
        // Shouldn't happen, but handle it
      }
      globalState.totalCompleted++;

      // Show completion notification
      this.showCompletionNotification(lesson);
    } else if (progress.status === 'not-started') {
      progress.status = 'in-progress';
      globalState.totalInProgress++;
    }

    globalState.lessons[lessonId] = progress;
    await this.saveGlobalState(globalState);

    // Fire change event
    this._onDidChangeProgress.fire({
      lessonId,
      oldStatus,
      newStatus: progress.status,
      triggeringCommand: commandId,
      timestamp: Date.now(),
    });
  }

  /**
   * Check if lesson is complete
   */
  private isLessonComplete(
    lesson: LessonMetadata,
    progress: ProgressState
  ): boolean {
    if (!lesson.completionEvents || lesson.completionEvents.length === 0) {
      return false;
    }

    // Check if all completion events have been executed
    return lesson.completionEvents.every(eventCommand =>
      progress.completedCommands.includes(eventCommand)
    );
  }

  /**
   * Show lesson completion notification
   */
  private showCompletionNotification(lesson: LessonMetadata): void {
    vscode.window
      .showInformationMessage(
        `🎉 Lesson completed: ${lesson.title}`,
        'Next Lesson',
        'View Progress'
      )
      .then(selection => {
        if (selection === 'Next Lesson' && lesson.nextLesson) {
          vscode.commands.executeCommand('tenstorrent.showLesson', lesson.nextLesson);
        } else if (selection === 'View Progress') {
          vscode.commands.executeCommand('tenstorrent.showProgress');
        }
      });
  }

  /**
   * Get progress statistics
   */
  getStatistics(totalLessons: number): ProgressStatistics {
    const globalState = this.getGlobalState();
    const completedLessons = globalState.totalCompleted;
    const inProgressLessons = globalState.totalInProgress;
    const notStartedLessons = totalLessons - completedLessons - inProgressLessons;

    // Find most recent and longest lessons
    let mostRecentLesson: string | undefined;
    let mostRecentTime = 0;
    let longestLesson: string | undefined;
    let longestTime = 0;

    for (const [lessonId, progress] of Object.entries(globalState.lessons)) {
      if (progress.lastAccessed > mostRecentTime) {
        mostRecentTime = progress.lastAccessed;
        mostRecentLesson = lessonId;
      }
      if (progress.timeSpentSeconds > longestTime) {
        longestTime = progress.timeSpentSeconds;
        longestLesson = lessonId;
      }
    }

    return {
      totalLessons,
      completedLessons,
      inProgressLessons,
      notStartedLessons,
      completionPercentage: (completedLessons / totalLessons) * 100,
      totalTimeSpent: globalState.totalTimeSpent,
      averageTimePerLesson:
        completedLessons > 0
          ? globalState.totalTimeSpent / completedLessons
          : 0,
      mostRecentLesson,
      longestLesson,
    };
  }

  /**
   * Get set of completed lesson IDs
   */
  getCompletedLessons(): Set<string> {
    const globalState = this.getGlobalState();
    const completed = new Set<string>();

    for (const [lessonId, progress] of Object.entries(globalState.lessons)) {
      if (progress.status === 'completed') {
        completed.add(lessonId);
      }
    }

    return completed;
  }

  /**
   * Reset progress for a lesson
   */
  async resetProgress(lessonId: string): Promise<void> {
    const globalState = this.getGlobalState();
    const progress = globalState.lessons[lessonId];

    if (progress) {
      // Update global counts
      if (progress.status === 'completed') {
        globalState.totalCompleted--;
      } else if (progress.status === 'in-progress') {
        globalState.totalInProgress--;
      }

      // Remove lesson progress
      delete globalState.lessons[lessonId];
      await this.saveGlobalState(globalState);

      this._onDidChangeProgress.fire({
        lessonId,
        oldStatus: progress.status,
        newStatus: 'not-started',
        timestamp: Date.now(),
      });
    }
  }

  /**
   * Reset all progress
   */
  async resetAllProgress(): Promise<void> {
    await this.context.globalState.update(PROGRESS_STATE_KEY, {
      lessons: {},
      lastSync: Date.now(),
      totalCompleted: 0,
      totalInProgress: 0,
      totalTimeSpent: 0,
    });

    this._onDidChangeProgress.fire({
      lessonId: 'all',
      oldStatus: 'in-progress',
      newStatus: 'not-started',
      timestamp: Date.now(),
    });
  }

  /**
   * Export progress for debugging/backup
   */
  exportProgress(): GlobalProgressState {
    return this.getGlobalState();
  }

  /**
   * Import progress from backup
   */
  async importProgress(state: GlobalProgressState): Promise<void> {
    await this.saveGlobalState(state);
    this._onDidChangeProgress.fire({
      lessonId: 'all',
      oldStatus: 'not-started',
      newStatus: 'in-progress',
      timestamp: Date.now(),
    });
  }
}
