#!/usr/bin/env node

/**
 * Validation script for VSCode command URIs in markdown files
 *
 * Checks that command URIs use the correct format:
 * ✅ CORRECT: command:tenstorrent.showLesson?["lesson-id"]
 * ✅ CORRECT: command:tenstorrent.startServer?[{"hardware":"N150"}]
 * ❌ WRONG: command:tenstorrent.showLesson?%7B%22lessonId%22%3A%22lesson-id%22%7D
 *
 * VSCode command URI format:
 * - No arguments: command:commandName
 * - With arguments: command:commandName?[arg1, arg2, ...]
 * - Object arguments: command:commandName?[{"key":"value"}]
 */

const fs = require('fs');
const path = require('path');
const { glob } = require('glob');

// ANSI color codes
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  cyan: '\x1b[36m',
  gray: '\x1b[90m',
};

// Patterns to detect
const URL_ENCODED_PATTERN = /command:[a-zA-Z0-9._-]+\?%[0-9A-F]{2}/g;
const ALL_COMMAND_PATTERN = /command:([a-zA-Z0-9._-]+)(?:\?(.+?))?(?=[\s\)]|$)/g;

/**
 * Extract registered commands from extension.ts
 */
function extractKnownCommands() {
  const extensionPath = path.join(__dirname, '..', 'src', 'extension.ts');
  if (!fs.existsSync(extensionPath)) {
    console.warn(`${colors.yellow}⚠️  Could not find extension.ts for command validation${colors.reset}`);
    return [];
  }

  const content = fs.readFileSync(extensionPath, 'utf8');
  const commands = new Set();

  // Pattern to match: vscode.commands.registerCommand('tenstorrent.commandName', ...)
  const registerPattern = /vscode\.commands\.registerCommand\s*\(\s*['"]([^'"]+)['"]/g;

  let match;
  while ((match = registerPattern.exec(content)) !== null) {
    commands.add(match[1]);
  }

  return Array.from(commands).sort();
}

const KNOWN_COMMANDS = extractKnownCommands();

let totalErrors = 0;
let totalWarnings = 0;
let filesChecked = 0;

/**
 * Check if a command URI uses URL encoding (old format)
 */
function hasUrlEncoding(uri) {
  return /%[0-9A-F]{2}/.test(uri);
}

/**
 * Check if a command URI uses correct array format
 */
function hasCorrectFormat(commandName, args) {
  if (!args) return true; // No args is fine

  // Should start with [ and end with ]
  if (!args.startsWith('[') || !args.endsWith(']')) {
    return false;
  }

  return true;
}

/**
 * Validate a single markdown file
 */
function validateFile(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const lines = content.split('\n');
  const errors = [];
  const warnings = [];

  // Check for URL-encoded commands (old format)
  lines.forEach((line, lineNum) => {
    const matches = line.matchAll(URL_ENCODED_PATTERN);
    for (const match of matches) {
      errors.push({
        line: lineNum + 1,
        text: match[0],
        message: 'URL-encoded command URI (old format)',
        suggestion: 'Use array format: command:name?["arg"] or command:name?[{"key":"value"}]'
      });
    }
  });

  // Check all command URIs for correct format
  lines.forEach((line, lineNum) => {
    const matches = line.matchAll(ALL_COMMAND_PATTERN);
    for (const match of matches) {
      const [fullMatch, commandName, args] = match;

      // Skip if already flagged as URL-encoded
      if (hasUrlEncoding(fullMatch)) continue;

      // Check if command is known
      if (!KNOWN_COMMANDS.includes(commandName)) {
        warnings.push({
          line: lineNum + 1,
          text: fullMatch,
          message: `Unknown command: ${commandName}`,
          suggestion: 'Verify this command exists in extension.ts'
        });
      }

      // Check argument format
      if (args && !hasCorrectFormat(commandName, args)) {
        errors.push({
          line: lineNum + 1,
          text: fullMatch,
          message: 'Invalid argument format',
          suggestion: 'Arguments must be in array format: ?["arg"] or ?[{"key":"value"}]'
        });
      }
    }
  });

  return { errors, warnings };
}

/**
 * Main validation logic
 */
async function validateCommandUris() {
  console.log(`${colors.cyan}🔍 Validating command URIs...${colors.reset}\n`);

  // Find all markdown files in content/
  const contentDir = path.join(__dirname, '..', 'content');
  const mdFiles = await glob('**/*.md', { cwd: contentDir });

  if (mdFiles.length === 0) {
    console.log(`${colors.yellow}⚠️  No markdown files found in content/${colors.reset}`);
    return 0;
  }

  console.log(`${colors.gray}📚 Found ${mdFiles.length} markdown files${colors.reset}\n`);
  filesChecked = mdFiles.length;

  const fileResults = [];

  // Validate each file
  for (const relPath of mdFiles) {
    const filePath = path.join(contentDir, relPath);
    const { errors, warnings } = validateFile(filePath);

    if (errors.length > 0 || warnings.length > 0) {
      fileResults.push({ filePath: relPath, errors, warnings });
      totalErrors += errors.length;
      totalWarnings += warnings.length;
    }
  }

  // Report results
  if (fileResults.length === 0) {
    console.log(`${colors.green}✅ All command URIs are valid!${colors.reset}\n`);
    console.log(`${colors.gray}Checked ${filesChecked} markdown files.${colors.reset}`);
    return 0;
  }

  // Show errors and warnings
  console.log('━'.repeat(70));
  console.log(`${colors.red}Found ${totalErrors} error(s) and ${totalWarnings} warning(s)${colors.reset}\n`);

  fileResults.forEach(({ filePath, errors, warnings }) => {
    console.log(`${colors.cyan}📄 ${filePath}${colors.reset}`);

    errors.forEach(({ line, text, message, suggestion }) => {
      console.log(`  ${colors.red}❌ Line ${line}: ${message}${colors.reset}`);
      console.log(`     ${colors.gray}Found: ${text}${colors.reset}`);
      console.log(`     ${colors.yellow}→ ${suggestion}${colors.reset}`);
    });

    warnings.forEach(({ line, text, message, suggestion }) => {
      console.log(`  ${colors.yellow}⚠️  Line ${line}: ${message}${colors.reset}`);
      console.log(`     ${colors.gray}Found: ${text}${colors.reset}`);
      console.log(`     ${colors.yellow}→ ${suggestion}${colors.reset}`);
    });

    console.log();
  });

  console.log('━'.repeat(70));
  console.log(`${colors.red}Summary: ${totalErrors} error(s), ${totalWarnings} warning(s) in ${fileResults.length} file(s)${colors.reset}\n`);

  if (totalErrors > 0) {
    console.log(`${colors.red}❌ Validation failed. Fix errors before committing.${colors.reset}`);
    return 1;
  } else {
    console.log(`${colors.yellow}⚠️  Warnings found. Review before committing.${colors.reset}`);
    return 0; // Don't fail on warnings
  }
}

// Run validation
validateCommandUris()
  .then(exitCode => {
    process.exit(exitCode);
  })
  .catch(error => {
    console.error(`${colors.red}Validation script error:${colors.reset}`, error);
    process.exit(1);
  });
