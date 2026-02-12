/**
 * Lesson Viewer Script
 *
 * Runs in the webview context to handle:
 * - Command button clicks
 * - Code copying
 * - Progress tracking
 * - Message passing to extension
 */

(function() {
  // Get VS Code API
  const vscode = acquireVsCodeApi();

  /**
   * Initialize lesson viewer
   */
  function initialize() {
    setupCommandButtons();
    setupCodeBlocks();
    restoreScrollPosition();
  }

  /**
   * Setup command button click handlers
   */
  function setupCommandButtons() {
    const buttons = document.querySelectorAll('.tt-command-button');

    buttons.forEach(button => {
      button.addEventListener('click', function() {
        const commandId = this.getAttribute('data-command');
        const argsJson = this.getAttribute('data-args');

        if (commandId) {
          // Send message to extension to execute command
          const message = {
            type: 'executeCommand',
            command: commandId
          };

          // If there are arguments, parse and include them
          if (argsJson) {
            try {
              message.args = JSON.parse(argsJson);
            } catch (e) {
              console.warn('Failed to parse command args:', argsJson, e);
            }
          }

          vscode.postMessage(message);

          // Visual feedback
          this.textContent = '✓ ' + this.textContent;
          this.style.background = '#27AE60';

          setTimeout(() => {
            this.textContent = this.textContent.replace('✓ ', '');
            this.style.background = '';
          }, 2000);
        }
      });
    });
  }

  /**
   * Setup code block copy functionality
   */
  function setupCodeBlocks() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach((block, index) => {
      // Create copy button
      const copyButton = document.createElement('button');
      copyButton.className = 'copy-button';
      copyButton.textContent = '📋 Copy';
      copyButton.style.cssText = `
        position: absolute;
        top: 8px;
        right: 8px;
        padding: 4px 8px;
        font-size: 0.85em;
        background: var(--vscode-button-background);
        color: var(--vscode-button-foreground);
        border: none;
        border-radius: 3px;
        cursor: pointer;
        opacity: 0;
        transition: opacity 0.2s;
      `;

      // Make pre relative for absolute positioning
      const pre = block.parentElement;
      pre.style.position = 'relative';

      // Show button on hover
      pre.addEventListener('mouseenter', () => {
        copyButton.style.opacity = '1';
      });

      pre.addEventListener('mouseleave', () => {
        copyButton.style.opacity = '0';
      });

      // Copy functionality
      copyButton.addEventListener('click', () => {
        const code = block.textContent;

        vscode.postMessage({
          type: 'copyCode',
          code: code
        });

        // Visual feedback
        copyButton.textContent = '✓ Copied!';
        copyButton.style.background = '#27AE60';

        setTimeout(() => {
          copyButton.textContent = '📋 Copy';
          copyButton.style.background = '';
        }, 2000);
      });

      pre.appendChild(copyButton);
    });
  }

  /**
   * Get current lesson ID from body data attribute
   */
  function getCurrentLessonId() {
    return document.body.getAttribute('data-lesson-id');
  }

  /**
   * Restore scroll position from state (only if same lesson)
   */
  function restoreScrollPosition() {
    const state = vscode.getState();
    const currentLessonId = getCurrentLessonId();

    // Only restore scroll position if we're viewing the same lesson
    if (state && state.lessonId === currentLessonId && state.scrollPosition) {
      window.scrollTo(0, state.scrollPosition);
    } else {
      // New lesson - scroll to top
      window.scrollTo(0, 0);
    }
  }

  /**
   * Save scroll position to state with lesson ID
   */
  function saveScrollPosition() {
    const currentLessonId = getCurrentLessonId();
    vscode.setState({
      ...vscode.getState(),
      lessonId: currentLessonId,
      scrollPosition: window.scrollY
    });
  }

  /**
   * Handle messages from extension
   */
  window.addEventListener('message', event => {
    const message = event.data;

    switch (message.type) {
      case 'clearState':
        // Clear saved scroll state when switching lessons
        vscode.setState({ lessonId: null, scrollPosition: 0 });
        break;

      case 'scrollToTop':
        // Force scroll to top (used when switching lessons)
        window.scrollTo(0, 0);
        // Also clear state to ensure it doesn't restore old position
        const currentLessonId = getCurrentLessonId();
        vscode.setState({ lessonId: currentLessonId, scrollPosition: 0 });
        break;

      case 'refresh':
        // Refresh content
        location.reload();
        break;

      case 'scrollTo':
        // Scroll to specific element
        const element = document.getElementById(message.elementId);
        if (element) {
          element.scrollIntoView({ behavior: 'smooth' });
        }
        break;

      case 'highlight':
        // Highlight specific section
        const section = document.getElementById(message.sectionId);
        if (section) {
          section.style.background = 'rgba(79, 209, 197, 0.2)';
          section.scrollIntoView({ behavior: 'smooth' });
          setTimeout(() => {
            section.style.background = '';
          }, 2000);
        }
        break;
    }
  });

  // Save scroll position before unload
  window.addEventListener('beforeunload', saveScrollPosition);

  // Save scroll position periodically
  let scrollTimeout;
  window.addEventListener('scroll', () => {
    clearTimeout(scrollTimeout);
    scrollTimeout = setTimeout(saveScrollPosition, 500);
  });

  // Initialize when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }
})();
