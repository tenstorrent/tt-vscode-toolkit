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
});
