/**
 * Lesson Webview Manager
 *
 * Manages webview lifecycle and content rendering for lessons.
 * Handles:
 * - Webview creation and disposal
 * - Markdown rendering
 * - Command execution
 * - Message passing
 * - Theme changes
 */

import * as vscode from 'vscode';
import * as path from 'path';
import { LessonRegistry } from '../utils';
import { ProgressTracker } from '../state';
import { MarkdownRenderer } from '../renderers';
import { LessonMetadata } from '../types';

/**
 * Message types for webview communication
 */
interface WebviewMessage {
  type: 'executeCommand' | 'copyCode' | 'ready';
  command?: string;
  code?: string;
  lessonId?: string;
}

/**
 * Manages lesson webview display
 */
export class LessonWebviewManager {
  private panel: vscode.WebviewPanel | undefined;
  private context: vscode.ExtensionContext;
  private lessonRegistry: LessonRegistry;
  private progressTracker: ProgressTracker;
  private markdownRenderer: MarkdownRenderer;
  private currentLesson: LessonMetadata | undefined;
  private disposables: vscode.Disposable[] = [];

  constructor(
    context: vscode.ExtensionContext,
    lessonRegistry: LessonRegistry,
    progressTracker: ProgressTracker
  ) {
    this.context = context;
    this.lessonRegistry = lessonRegistry;
    this.progressTracker = progressTracker;
    this.markdownRenderer = new MarkdownRenderer();

    // Listen for theme changes
    vscode.window.onDidChangeActiveColorTheme(() => {
      if (this.panel && this.currentLesson) {
        this.refresh();
      }
    }, null, this.disposables);
  }

  /**
   * Show a lesson in the webview
   */
  async showLesson(lessonIdOrMetadata: string | LessonMetadata): Promise<void> {
    // Get lesson metadata
    const lesson =
      typeof lessonIdOrMetadata === 'string'
        ? this.lessonRegistry.get(lessonIdOrMetadata)
        : lessonIdOrMetadata;

    if (!lesson) {
      vscode.window.showErrorMessage('Lesson not found');
      return;
    }

    this.currentLesson = lesson;

    // Start tracking session
    this.progressTracker.startSession(lesson.id);

    // Create or reveal panel
    if (this.panel) {
      this.panel.reveal(vscode.ViewColumn.One);
    } else {
      this.panel = vscode.window.createWebviewPanel(
        'tenstorrentLesson',
        lesson.title,
        vscode.ViewColumn.One,
        {
          enableScripts: true,
          retainContextWhenHidden: true,
          localResourceRoots: [
            vscode.Uri.file(path.join(this.context.extensionPath, 'dist', 'src', 'webview')),
            vscode.Uri.file(path.join(this.context.extensionPath, 'dist', 'content')),
          ],
        }
      );

      // Handle panel disposal
      this.panel.onDidDispose(() => {
        this.progressTracker.endSession();
        this.panel = undefined;
      }, null, this.disposables);

      // Handle messages from webview
      this.panel.webview.onDidReceiveMessage(
        (message: WebviewMessage) => this.handleMessage(message),
        null,
        this.disposables
      );
    }

    // Update panel title
    this.panel.title = lesson.title;

    // Render content
    await this.renderLesson(lesson);
  }

  /**
   * Render lesson content
   */
  private async renderLesson(lesson: LessonMetadata): Promise<void> {
    if (!this.panel) {
      return;
    }

    try {
      // Render markdown
      const contentPath = path.join(
        this.context.extensionPath,
        lesson.markdownFile
      );
      const rendered = await this.markdownRenderer.renderFile(contentPath);

      // Get webview URIs
      const cssUri = this.panel.webview.asWebviewUri(
        vscode.Uri.file(
          path.join(this.context.extensionPath, 'dist', 'src', 'webview', 'styles', 'lesson-theme.css')
        )
      );

      const jsUri = this.panel.webview.asWebviewUri(
        vscode.Uri.file(
          path.join(this.context.extensionPath, 'dist', 'src', 'webview', 'scripts', 'lesson-viewer.js')
        )
      );

      // Generate HTML
      const html = this.generateHTML(lesson, rendered.html, cssUri, jsUri);

      // Set webview content
      this.panel.webview.html = html;
    } catch (error) {
      vscode.window.showErrorMessage(
        `Failed to render lesson: ${error instanceof Error ? error.message : 'Unknown error'}`
      );
    }
  }

  /**
   * Generate full HTML for webview
   */
  private generateHTML(
    lesson: LessonMetadata,
    content: string,
    cssUri: vscode.Uri,
    jsUri: vscode.Uri
  ): string {
    const nonce = this.getNonce();

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${this.panel!.webview.cspSource} 'unsafe-inline' https://cdnjs.cloudflare.com; script-src 'nonce-${nonce}' https://cdnjs.cloudflare.com; img-src ${this.panel!.webview.cspSource} https: data:;">
  <title>${this.escapeHtml(lesson.title)}</title>
  <link rel="stylesheet" href="${cssUri}">
  <!-- Prism.js for syntax highlighting (VSCode-like theme) -->
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" nonce="${nonce}">
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.css" nonce="${nonce}">
</head>
<body data-lesson-id="${this.escapeHtml(lesson.id)}">
  <div class="lesson-header" style="border-bottom: 1px solid var(--tt-border); padding-bottom: 16px; margin-bottom: 24px;">
    <h1 style="margin: 0;">${this.escapeHtml(lesson.title)}</h1>
    <div style="display: flex; gap: 8px; margin-top: 12px; flex-wrap: wrap;">
      ${this.generateStatusBadge(lesson)}
      ${this.generateHardwareBadges(lesson)}
      ${lesson.estimatedMinutes ? `<span class="status-badge" style="background: var(--tt-blue); color: white;">~${lesson.estimatedMinutes} min</span>` : ''}
    </div>
  </div>

  <div class="lesson-content">
    ${content}
  </div>

  ${this.generateLessonNav(lesson)}

  <!-- Prism.js for syntax highlighting -->
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/plugins/line-numbers/prism-line-numbers.min.js"></script>
  <!-- Language components for common languages -->
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-bash.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-javascript.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-typescript.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-json.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-yaml.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-markdown.min.js"></script>
  <script nonce="${nonce}" src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-cpp.min.js"></script>
  <script nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
  }

  /**
   * Generate status badge HTML
   */
  private generateStatusBadge(lesson: LessonMetadata): string {
    const statusMap = {
      validated: '✅ Validated',
      draft: '📝 Draft',
      blocked: '⚠️ Blocked',
    };

    return `<span class="status-badge status-${lesson.status}">${statusMap[lesson.status]}</span>`;
  }

  /**
   * Generate hardware badges HTML
   */
  private generateHardwareBadges(lesson: LessonMetadata): string {
    return lesson.supportedHardware
      .map(hw => `<span class="hardware-chip">${hw.toUpperCase()}</span>`)
      .join('');
  }

  /**
   * Generate lesson navigation HTML
   */
  private generateLessonNav(lesson: LessonMetadata): string {
    const prevLesson = lesson.previousLesson
      ? this.lessonRegistry.get(lesson.previousLesson)
      : undefined;
    const nextLesson = lesson.nextLesson
      ? this.lessonRegistry.get(lesson.nextLesson)
      : undefined;

    if (!prevLesson && !nextLesson) {
      return '';
    }

    return `
      <div class="lesson-nav">
        ${prevLesson ? `<button class="nav-prev tt-command-button" data-command="tenstorrent.showLesson" data-lesson="${prevLesson.id}">← ${this.escapeHtml(prevLesson.title)}</button>` : '<div></div>'}
        ${nextLesson ? `<button class="nav-next tt-command-button" data-command="tenstorrent.showLesson" data-lesson="${nextLesson.id}">${this.escapeHtml(nextLesson.title)} →</button>` : '<div></div>'}
      </div>
    `;
  }

  /**
   * Handle messages from webview
   */
  private async handleMessage(message: WebviewMessage): Promise<void> {
    switch (message.type) {
      case 'executeCommand':
        if (message.command) {
          // Execute the command with optional lesson ID argument
          if (message.lessonId) {
            await vscode.commands.executeCommand(message.command, message.lessonId);
          } else {
            await vscode.commands.executeCommand(message.command);
          }

          // Record progress if we have a current lesson
          if (this.currentLesson) {
            await this.progressTracker.recordCommandExecution(
              this.currentLesson.id,
              message.command,
              this.currentLesson
            );
          }
        }
        break;

      case 'copyCode':
        if (message.code) {
          await vscode.env.clipboard.writeText(message.code);
          vscode.window.showInformationMessage('Code copied to clipboard');
        }
        break;

      case 'ready':
        // Webview is ready
        break;
    }
  }

  /**
   * Refresh current lesson
   */
  async refresh(): Promise<void> {
    if (this.currentLesson) {
      await this.renderLesson(this.currentLesson);
    }
  }

  /**
   * Dispose webview manager
   */
  dispose(): void {
    this.progressTracker.endSession();

    if (this.panel) {
      this.panel.dispose();
    }

    while (this.disposables.length) {
      const disposable = this.disposables.pop();
      if (disposable) {
        disposable.dispose();
      }
    }
  }

  /**
   * Generate nonce for CSP
   */
  private getNonce(): string {
    let text = '';
    const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    for (let i = 0; i < 32; i++) {
      text += possible.charAt(Math.floor(Math.random() * possible.length));
    }
    return text;
  }

  /**
   * Escape HTML special characters
   */
  private escapeHtml(text: string): string {
    const map: Record<string, string> = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;',
    };
    return text.replace(/[&<>"']/g, char => map[char]);
  }
}
