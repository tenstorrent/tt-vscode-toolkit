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

  /** Optional function to transform image URLs (e.g., to webview URIs) */
  transformImageUrl?: (url: string) => string;
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
    // Capture 'this' for use in renderer callbacks
    const self = this;

    // Setup marked with Prism.js-compatible code blocks and mermaid.js support
    marked.use(
      markedHighlight({
        highlight: (code, lang) => {
          if (!self.options.enableHighlight) {
            return code;
          }

          // Skip mermaid blocks - they need special handling in custom renderer
          if (lang === 'mermaid') {
            return code;
          }

          // Output Prism.js-compatible structure for code
          // Don't escape HTML - Prism needs the raw code and will handle it
          const language = lang || 'plaintext';
          return `<code class="language-${language}">${code}</code>`;
        }
      })
    );

    // Configure marked v17 API with custom renderers for command buttons and images
    marked.use({
      renderer: {
        // Custom code block renderer to handle mermaid diagrams
        code(token: any): string | false {
          const code = token.text;
          const lang = token.lang || '';

          // Handle mermaid diagrams - use <pre> to preserve whitespace
          if (lang === 'mermaid') {
            // Escape HTML entities so browser doesn't try to parse mermaid syntax as HTML
            return `<pre class="mermaid">${self.escapeHtml(code)}</pre>\n`;
          }

          // Default behavior for other code blocks - let marked handle it
          return false;
        },
        link: (token: any) => {
          const { href, title, tokens } = token;
          const text = self.parseInlineTokens(tokens);

          // Check if this is a command link
          if (href && href.startsWith('command:')) {
            const commandUrl = href.replace('command:', '');
            const titleAttr = title ? ` title="${self.escapeHtml(title)}"` : '';

            // Parse command ID and arguments
            // Format: commandId?encodedJsonArgs or just commandId
            let commandId = commandUrl;
            let argsAttr = '';

            if (commandUrl.includes('?')) {
              const [cmd, args] = commandUrl.split('?', 2);
              commandId = cmd;

              // Try to parse arguments (typically URI-encoded JSON like %7B%22lessonId%22%3A%22value%22%7D)
              try {
                const decodedArgs = decodeURIComponent(args);
                const argsObj = JSON.parse(decodedArgs);

                // Store the full arguments object as JSON in data-args attribute
                // This handles lessonId, hardware, and any future argument types
                argsAttr = ` data-args='${JSON.stringify(argsObj)}'`;
              } catch (e) {
                // If parsing fails, ignore - just use the command ID
                console.warn('Failed to parse command arguments:', args, e);
              }
            }

            return `<button class="tt-command-button"
                            data-command="${self.escapeHtml(commandId)}"${argsAttr}
                            ${titleAttr}>
                      ${text}
                    </button>`;
          }

          // For regular links, use default rendering
          const escapedHref = self.escapeHtml(href || '');
          const titleStr = title ? ` title="${self.escapeHtml(title)}"` : '';
          return `<a href="${escapedHref}"${titleStr}>${text}</a>`;
        },
        image: (token: any) => {
          const { href, title, text } = token;

          // Transform image URL if transformer is provided
          const imageUrl = self.options.transformImageUrl
            ? self.options.transformImageUrl(href)
            : self.escapeHtml(href || '');

          const titleAttr = title ? ` title="${self.escapeHtml(title)}"` : '';
          const altText = self.escapeHtml(text || '');

          return `<img src="${imageUrl}" alt="${altText}"${titleAttr}>`;
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
      // Extract mermaid blocks before sanitization (they contain special syntax)
      const mermaidBlocks: string[] = [];
      const mermaidPlaceholder = '___MERMAID_BLOCK___';

      // Replace mermaid pre blocks with placeholders
      html = html.replace(/<pre class="mermaid">([\s\S]*?)<\/pre>/g, (_match, content) => {
        mermaidBlocks.push(content);
        return `<pre class="mermaid">${mermaidPlaceholder}${mermaidBlocks.length - 1}</pre>`;
      });

      // Sanitize everything else
      html = DOMPurify.sanitize(html, {
        ADD_TAGS: ['button', 'div', 'pre'],  // Allow our command buttons, divs, and pre blocks
        ADD_ATTR: ['data-command', 'class', 'data-args'],  // Allow our data attributes
      });

      // Restore mermaid blocks with original unsanitized content
      mermaidBlocks.forEach((content, index) => {
        html = html.replace(`${mermaidPlaceholder}${index}`, content);
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
