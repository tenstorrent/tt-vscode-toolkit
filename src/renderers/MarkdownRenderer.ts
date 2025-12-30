/**
 * Markdown Renderer
 *
 * Converts markdown content to HTML with:
 * - Command button support
 * - Syntax highlighting
 * - Theme-aware styling
 * - XSS protection
 */

import { marked } from 'marked';
import { markedHighlight } from 'marked-highlight';
import DOMPurify from 'isomorphic-dompurify';
import * as fs from 'fs';
import * as path from 'path';
import matter from 'gray-matter';

/**
 * Markdown rendering configuration
 */
export interface MarkdownRenderOptions {
  /** Base path for resolving relative links and images */
  basePath?: string;

  /** Whether to enable syntax highlighting */
  enableHighlight?: boolean;

  /** Whether to sanitize HTML (XSS protection) */
  sanitize?: boolean;

  /** Custom CSS classes to add to rendered HTML */
  customClasses?: Record<string, string>;
}

/**
 * Rendered markdown result
 */
export interface RenderedMarkdown {
  /** Rendered HTML content */
  html: string;

  /** Frontmatter data (if any) */
  frontmatter: Record<string, any>;

  /** List of command IDs found in content */
  commands: string[];
}

/**
 * Markdown to HTML renderer with command button support
 */
export class MarkdownRenderer {
  private options: MarkdownRenderOptions;

  constructor(options: MarkdownRenderOptions = {}) {
    this.options = {
      enableHighlight: true,
      sanitize: true,
      ...options,
    };

    this.configureMarked();
  }

  /**
   * Configure marked with custom renderer and extensions
   */
  private configureMarked(): void {
    // Setup marked with Prism.js-compatible code blocks
    marked.use(
      markedHighlight({
        highlight: (code, lang) => {
          if (!this.options.enableHighlight) {
            return code;
          }

          // Output Prism.js-compatible structure
          // Don't escape HTML - Prism needs the raw code and will handle it
          const language = lang || 'plaintext';
          return `<code class="language-${language}">${code}</code>`;
        }
      })
    );

    // Configure marked v17 API with custom link renderer for command buttons
    marked.use({
      renderer: {
        link: (token: any) => {
          const { href, title, tokens } = token;
          const text = this.parseInlineTokens(tokens);

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
          const escapedHref = this.escapeHtml(href || '');
          const titleStr = title ? ` title="${this.escapeHtml(title)}"` : '';
          return `<a href="${escapedHref}"${titleStr}>${text}</a>`;
        }
      }
    });

    marked.setOptions({
      gfm: true,  // GitHub Flavored Markdown
      breaks: true,  // Convert \n to <br>
    });
  }

  /**
   * Render markdown file to HTML
   */
  async renderFile(filePath: string): Promise<RenderedMarkdown> {
    const content = fs.readFileSync(filePath, 'utf-8');
    return this.render(content, path.dirname(filePath));
  }

  /**
   * Render markdown string to HTML
   */
  async render(
    markdown: string,
    _basePath?: string
  ): Promise<RenderedMarkdown> {
    // Parse frontmatter
    const parsed = matter(markdown);
    const frontmatter = parsed.data;
    const content = parsed.content;

    // Convert markdown to HTML
    let html = await marked.parse(content);

    // Sanitize HTML if enabled
    if (this.options.sanitize) {
      html = DOMPurify.sanitize(html, {
        ADD_TAGS: ['button'],  // Allow our command buttons
        ADD_ATTR: ['data-command', 'class'],  // Allow our data attributes
      });
    }

    // Extract command IDs from HTML
    const commands = this.extractCommands(html);

    return {
      html,
      frontmatter,
      commands,
    };
  }

  /**
   * Extract command IDs from rendered HTML
   */
  private extractCommands(html: string): string[] {
    const commands: string[] = [];
    const regex = /data-command="([^"]+)"/g;
    let match;

    while ((match = regex.exec(html)) !== null) {
      commands.push(match[1]);
    }

    return commands;
  }

  /**
   * Parse inline tokens to text
   */
  private parseInlineTokens(tokens: any[]): string {
    if (!tokens || tokens.length === 0) {
      return '';
    }
    return tokens.map(token => {
      if (token.type === 'text') {
        return token.text || '';
      }
      if (token.type === 'strong') {
        return `<strong>${this.parseInlineTokens(token.tokens)}</strong>`;
      }
      if (token.type === 'em') {
        return `<em>${this.parseInlineTokens(token.tokens)}</em>`;
      }
      if (token.type === 'codespan') {
        return `<code>${this.escapeHtml(token.text || '')}</code>`;
      }
      return token.raw || '';
    }).join('');
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
      "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, (char) => map[char]);
  }

  /**
   * Wrap rendered HTML in a template
   */
  wrapInTemplate(
    html: string,
    title: string,
    cssUri?: string,
    jsUri?: string
  ): string {
    return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${this.escapeHtml(title)}</title>
  ${cssUri ? `<link rel="stylesheet" href="${cssUri}">` : ''}
</head>
<body>
  <div class="lesson-content">
    ${html}
  </div>
  ${jsUri ? `<script src="${jsUri}"></script>` : ''}
</body>
</html>
    `.trim();
  }
}
