/**
 * Markdown Validation Tests
 *
 * Validates markdown formatting across all lesson files to catch common errors:
 * 1. Malformed code block fencing (e.g., ```bash followed by ```text)
 * 2. Unclosed code blocks
 * 3. Mismatched opening/closing fences
 * 4. Empty code blocks
 */

import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

describe('Markdown Validation Tests', () => {
  const lessonsDir = path.join(__dirname, '../../content/lessons');
  const lessonFiles = fs.readdirSync(lessonsDir).filter(file => file.endsWith('.md'));

  describe('Code Block Fencing', () => {
    lessonFiles.forEach(file => {
      it(`${file} should have properly matched code block fences`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        let inCodeBlock = false;
        let openingFence = '';
        let lineNumber = 0;

        for (const line of lines) {
          lineNumber++;
          const trimmedLine = line.trim();

          // Check for opening fence (``` or ~~~)
          if (trimmedLine.startsWith('```') || trimmedLine.startsWith('~~~')) {
            if (!inCodeBlock) {
              // Opening fence
              inCodeBlock = true;
              openingFence = trimmedLine.substring(0, 3); // ``` or ~~~
            } else {
              // Closing fence
              const closingFence = trimmedLine.substring(0, 3);

              // Verify closing fence matches opening fence
              expect(closingFence, `${file}:${lineNumber} - Closing fence ${closingFence} doesn't match opening fence ${openingFence}`).to.equal(openingFence);

              // Verify closing fence is just ``` or ~~~ (no language specifier)
              const closingFenceContent = trimmedLine.substring(3);
              expect(closingFenceContent, `${file}:${lineNumber} - Closing fence should be just '${openingFence}' without additional text: '${trimmedLine}'`).to.equal('');

              inCodeBlock = false;
              openingFence = '';
            }
          }
        }

        // Verify all code blocks are closed
        expect(inCodeBlock, `${file} - Unclosed code block at end of file (opened with ${openingFence})`).to.be.false;
      });
    });
  });

  describe('Code Block Language Specifiers', () => {
    lessonFiles.forEach(file => {
      it(`${file} should have language specifiers on opening code fences`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        let lineNumber = 0;
        let inCodeBlock = false;

        for (const line of lines) {
          lineNumber++;
          const trimmedLine = line.trim();

          if (trimmedLine.startsWith('```') || trimmedLine.startsWith('~~~')) {
            if (!inCodeBlock) {
              // Opening fence - should have language specifier
              const remainder = trimmedLine.substring(3);

              // Allow empty specifier for generic code blocks, but warn about it
              if (remainder.length === 0) {
                // This is acceptable but not ideal - could be improved
                // Not failing the test, just noting it
              } else {
                // Verify it's a valid language specifier (alphanumeric, dash, underscore)
                expect(remainder, `${file}:${lineNumber} - Invalid language specifier: '${remainder}'`).to.match(/^[a-zA-Z0-9_-]+$/);
              }

              inCodeBlock = true;
            } else {
              // Closing fence
              inCodeBlock = false;
            }
          }
        }
      });
    });
  });

  describe('Empty Code Blocks', () => {
    lessonFiles.forEach(file => {
      it(`${file} should not have empty code blocks`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        let lineNumber = 0;
        let inCodeBlock = false;
        let codeBlockStartLine = 0;
        let codeBlockContent: string[] = [];

        for (const line of lines) {
          lineNumber++;
          const trimmedLine = line.trim();

          if (trimmedLine.startsWith('```') || trimmedLine.startsWith('~~~')) {
            if (!inCodeBlock) {
              // Opening fence
              inCodeBlock = true;
              codeBlockStartLine = lineNumber;
              codeBlockContent = [];
            } else {
              // Closing fence - check if block has content
              const hasContent = codeBlockContent.some(line => line.trim().length > 0);
              expect(hasContent, `${file}:${codeBlockStartLine}-${lineNumber} - Empty code block`).to.be.true;

              inCodeBlock = false;
              codeBlockContent = [];
            }
          } else if (inCodeBlock) {
            codeBlockContent.push(line);
          }
        }
      });
    });
  });

  describe('Frontmatter Validation', () => {
    lessonFiles.forEach(file => {
      it(`${file} should have valid YAML frontmatter`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        // Check for opening ---
        expect(lines[0], `${file} - Missing opening frontmatter delimiter (---)`).to.equal('---');

        // Find closing ---
        let closingIndex = -1;
        for (let i = 1; i < lines.length; i++) {
          if (lines[i].trim() === '---') {
            closingIndex = i;
            break;
          }
        }

        expect(closingIndex, `${file} - Missing closing frontmatter delimiter (---)`).to.be.greaterThan(0);
        expect(closingIndex, `${file} - Frontmatter should be at top of file`).to.be.lessThan(50);
      });
    });
  });

  describe('Common Markdown Issues', () => {
    lessonFiles.forEach(file => {
      it(`${file} should not have trailing spaces on code fence lines`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        let lineNumber = 0;
        for (const line of lines) {
          lineNumber++;
          const trimmed = line.trimEnd();

          // Check if this is a code fence line
          if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
            // Verify no trailing spaces after the fence
            expect(line, `${file}:${lineNumber} - Code fence line has trailing spaces`).to.equal(trimmed);
          }
        }
      });

      it(`${file} should not have tabs in code fences`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const lines = content.split('\n');

        let lineNumber = 0;
        for (const line of lines) {
          lineNumber++;
          const trimmed = line.trim();

          // Check if this is a code fence line
          if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
            // Verify no tabs in the fence line
            expect(line, `${file}:${lineNumber} - Code fence line contains tabs`).to.not.include('\t');
          }
        }
      });
    });
  });

  describe('Lesson Structure', () => {
    lessonFiles.forEach(file => {
      it(`${file} should have at least one heading after frontmatter`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');

        // Skip frontmatter
        const contentAfterFrontmatter = content.split('---').slice(2).join('---');

        // Check for at least one # heading
        const hasHeading = contentAfterFrontmatter.includes('\n#');
        expect(hasHeading, `${file} - No headings found after frontmatter`).to.be.true;
      });
    });
  });

  describe('Mermaid Diagram Validation', () => {
    /**
     * Extract all mermaid code blocks from markdown content
     * Returns array of {code, startLine, endLine} objects
     */
    function extractMermaidBlocks(content: string): Array<{code: string, startLine: number, endLine: number}> {
      const lines = content.split('\n');
      const mermaidBlocks: Array<{code: string, startLine: number, endLine: number}> = [];
      let inMermaidBlock = false;
      let currentBlock: string[] = [];
      let blockStartLine = 0;

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        const trimmed = line.trim();

        if (trimmed.startsWith('```mermaid')) {
          inMermaidBlock = true;
          blockStartLine = i + 1; // Line number (1-indexed)
          currentBlock = [];
        } else if (inMermaidBlock && trimmed === '```') {
          // End of mermaid block
          mermaidBlocks.push({
            code: currentBlock.join('\n'),
            startLine: blockStartLine + 1, // 1-indexed for user-friendly errors
            endLine: i + 1
          });
          inMermaidBlock = false;
          currentBlock = [];
        } else if (inMermaidBlock) {
          currentBlock.push(line);
        }
      }

      return mermaidBlocks;
    }

    /**
     * Validate mermaid syntax using mermaid.js parser
     * This catches syntax errors like missing graph type, invalid syntax, etc.
     *
     * NOTE: Disabled because DOMPurify gives false positives in Node.js test environment.
     * The diagrams render correctly in the extension (browser webview).
     * The stroke property test below is sufficient for our needs.
     */
    lessonFiles.forEach(file => {
      it.skip(`${file} should have syntactically valid mermaid diagrams`, async () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const mermaidBlocks = extractMermaidBlocks(content);

        // Skip files with no mermaid blocks
        if (mermaidBlocks.length === 0) {
          return;
        }

        // Dynamic import of mermaid (ESM module)
        const mermaid = await import('mermaid');

        // Test each mermaid block
        for (const block of mermaidBlocks) {
          try {
            // Use mermaid's parse function to validate syntax
            // Note: mermaid v11.x changed API - now uses mermaid.parse()
            await mermaid.default.parse(block.code);
          } catch (error: any) {
            // Syntax error found - provide detailed error message
            const errorMsg = error.message || error.toString();
            expect.fail(
              `${file}:${block.startLine}-${block.endLine} - Mermaid syntax error:\n${errorMsg}\n\nDiagram code:\n${block.code}`
            );
          }
        }
      });
    });

    /**
     * Validate that style statements include required properties
     * Mermaid v10+ requires explicit stroke property in style statements
     */
    lessonFiles.forEach(file => {
      it(`${file} mermaid style statements should include stroke property`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const mermaidBlocks = extractMermaidBlocks(content);

        // Skip files with no mermaid blocks
        if (mermaidBlocks.length === 0) {
          return;
        }

        // Check each mermaid block for style statements
        for (const block of mermaidBlocks) {
          const lines = block.code.split('\n');

          lines.forEach((line, index) => {
            const trimmed = line.trim();

            // Check for style statements
            if (trimmed.startsWith('style ')) {
              // Extract the style properties (everything after "style NodeName ")
              const match = trimmed.match(/^style\s+\S+\s+(.+)$/);
              if (match) {
                const properties = match[1];

                // Check if stroke property is present
                // Valid formats: stroke:#fff, stroke: #fff, stroke:#000, etc.
                const hasStroke = /stroke\s*:\s*#[0-9a-fA-F]{3,6}/.test(properties);

                expect(hasStroke,
                  `${file}:${block.startLine + index} - Style statement missing 'stroke' property:\n` +
                  `  Line: ${trimmed}\n` +
                  `  Note: Mermaid v10+ requires explicit stroke (border) color in style statements.\n` +
                  `  Example: style NodeName fill:#5347a4,stroke:#fff,color:#fff`
                ).to.be.true;
              }
            }
          });
        }
      });
    });

    /**
     * Validate that styled nodes are actually defined in the diagram
     * This catches typos in node names in style statements
     */
    lessonFiles.forEach(file => {
      it(`${file} mermaid style statements should reference defined nodes`, () => {
        const filePath = path.join(lessonsDir, file);
        const content = fs.readFileSync(filePath, 'utf8');
        const mermaidBlocks = extractMermaidBlocks(content);

        // Skip files with no mermaid blocks
        if (mermaidBlocks.length === 0) {
          return;
        }

        // Check each mermaid block
        for (const block of mermaidBlocks) {
          const lines = block.code.split('\n');
          const definedNodes = new Set<string>();
          const styledNodes: Array<{name: string, line: string, lineNum: number}> = [];

          // First pass: collect all defined nodes
          lines.forEach((line) => {
            const trimmed = line.trim();

            // Skip empty lines, comments, and graph type declarations
            if (!trimmed || trimmed.startsWith('%%') ||
                trimmed.startsWith('graph ') || trimmed.startsWith('sequenceDiagram') ||
                trimmed.startsWith('style ')) {
              return;
            }

            // Extract node names from various mermaid syntax patterns
            // Patterns: NodeName[Label], NodeName(Label), NodeName{Label}, NodeName>Label]
            // Also: A --> B, A --- B, etc.
            const nodeMatches = trimmed.matchAll(/([A-Z][a-zA-Z0-9]*)[(\[{>]/g);
            for (const match of nodeMatches) {
              definedNodes.add(match[1]);
            }

            // Also capture simple node references: A --> B
            const arrowMatches = trimmed.matchAll(/([A-Z][a-zA-Z0-9]*)\s*[-=]+>/g);
            for (const match of arrowMatches) {
              definedNodes.add(match[1]);
            }
          });

          // Second pass: collect all styled nodes
          lines.forEach((line, index) => {
            const trimmed = line.trim();

            if (trimmed.startsWith('style ')) {
              const match = trimmed.match(/^style\s+(\S+)\s+/);
              if (match) {
                styledNodes.push({
                  name: match[1],
                  line: trimmed,
                  lineNum: block.startLine + index
                });
              }
            }
          });

          // Verify all styled nodes are defined
          for (const styled of styledNodes) {
            expect(definedNodes.has(styled.name),
              `${file}:${styled.lineNum} - Style references undefined node '${styled.name}':\n` +
              `  Line: ${styled.line}\n` +
              `  Defined nodes in this diagram: ${Array.from(definedNodes).join(', ')}`
            ).to.be.true;
          }
        }
      });
    });
  });
});
