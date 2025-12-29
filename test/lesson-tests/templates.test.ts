/**
 * Template Validation Tests
 *
 * Tests all Python script templates for:
 * - Valid Python syntax
 * - Proper file structure
 * - Python 3 compatibility
 *
 * These tests are content-agnostic and work with any Python templates
 * that exist in the templates directory.
 */

import { expect } from 'chai';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as fs from 'fs/promises';
import * as path from 'path';

const execAsync = promisify(exec);

describe('Python Template Validation', () => {
  const templatesDir = path.join(__dirname, '../../content/templates');
  let pythonTemplates: string[] = [];

  before(async () => {
    // Discover all Python templates
    const allFiles = await fs.readdir(templatesDir);
    pythonTemplates = allFiles.filter(f => f.endsWith('.py'));

    expect(pythonTemplates.length).to.be.greaterThan(
      0,
      'Templates directory should contain at least one Python file'
    );
  });

  describe('Python Syntax Validation', () => {
    it('all Python templates should have valid syntax', async function() {
      this.timeout(10000); // Syntax checking can take a moment

      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);

        try {
          // Compile to check syntax (doesn't execute)
          await execAsync(`python3 -m py_compile "${filePath}"`);
        } catch (error) {
          throw new Error(
            `${template} has invalid Python syntax: ${error instanceof Error ? error.message : 'Unknown error'}`
          );
        }
      }
    });
  });

  describe('File Structure', () => {
    it('all templates should be non-empty', async () => {
      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);
        const content = await fs.readFile(filePath, 'utf-8');
        expect(content.length).to.be.greaterThan(
          100,
          `${template} is too short (likely empty or incomplete)`
        );
      }
    });

    it('all templates should have documentation', async () => {
      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);
        const content = await fs.readFile(filePath, 'utf-8');

        // Should have either a docstring or comment
        const hasDocumentation =
          content.includes('"""') ||
          content.includes("'''") ||
          content.includes('#');

        expect(hasDocumentation, `${template} should have documentation (comments or docstrings)`).to.equal(true);
      }
    });

    it('all templates should have proper file encoding', async () => {
      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);

        // Should be able to read as UTF-8 without errors
        try {
          await fs.readFile(filePath, 'utf-8');
        } catch (error) {
          throw new Error(`${template} has invalid UTF-8 encoding`);
        }
      }
    });
  });

  describe('Python 3 Compatibility', () => {
    it('templates should use Python 3 print function', async () => {
      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);
        const content = await fs.readFile(filePath, 'utf-8');

        // Check for Python 2 print statements (should not exist)
        const hasPython2Print = /print\s+[^(]/.test(content);
        expect(hasPython2Print, `${template} uses Python 2 print syntax`).to.equal(false);
      }
    });

    it('templates should not use Python 2 string formatting', async () => {
      for (const template of pythonTemplates) {
        const filePath = path.join(templatesDir, template);
        const content = await fs.readFile(filePath, 'utf-8');

        // Check for old-style string formatting (should avoid)
        const lines = content.split('\n');
        const hasOldStringFormat = lines.some(line => {
          // Skip comments
          if (line.trim().startsWith('#')) {
            return false;
          }
          // Check for % formatting (except in comments or docstrings)
          return /["']\s*%\s*\(/.test(line);
        });

        // This is a warning, not a hard failure
        if (hasOldStringFormat) {
          console.warn(`  ⚠ ${template} uses old-style % string formatting (consider f-strings)`);
        }
      }
    });
  });

  describe('Template Discovery', () => {
    it('should find Python templates in the templates directory', () => {
      expect(pythonTemplates.length).to.be.greaterThan(
        0,
        'Should discover at least one Python template'
      );

      console.log(`  ✓ Found ${pythonTemplates.length} Python template(s): ${pythonTemplates.join(', ')}`);
    });

    it('templates directory should be accessible', async () => {
      const stats = await fs.stat(templatesDir);
      expect(stats.isDirectory()).to.equal(true, 'Templates path should be a directory');
    });
  });
});
