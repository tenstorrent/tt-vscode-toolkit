/**
 * Command Button Renderer
 *
 * Transforms markdown command links into clickable buttons.
 * Converts: [Button Text](command:tenstorrent.commandName)
 * Into: <button data-command="..." class="tt-command-button">Button Text</button>
 */

import { marked } from 'marked';

/**
 * Custom renderer for command buttons
 */
export class CommandButtonRenderer extends marked.Renderer {
  /**
   * Override link rendering to detect command: links
   */
  link(token: any): string {
    const { href, title, tokens } = token;
    const text = this.parser?.parseInline(tokens) || '';

    // Check if this is a command link
    if (href && href.startsWith('command:')) {
      const commandId = href.replace('command:', '');
      const titleAttr = title ? ` title="${this.escapeHtml(title)}"` : '';

      return `<button class="tt-command-button"
                      data-command="${this.escapeHtml(commandId)}"
                      ${titleAttr}>
                ${text}
              </button>`;
    }

    // For regular links, use default rendering
    return super.link(token);
  }

  /**
   * Escape HTML to prevent XSS
   */
  private escapeHtml(text: string): string {
    const map: Record<string, string> = {
      '&': '&amp;',
      '<': '&lt;',
      '>': '&gt;',
      '"': '&quot;',
      "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }
}
