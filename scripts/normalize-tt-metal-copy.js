#!/usr/bin/env node
/**
 * Normalize prose/sample output: TT Metal → TT-Metalium
 * Skips executable code fences (python, bash, etc.) and inline `code`.
 *
 * Usage:
 *   node scripts/normalize-tt-metal-copy.js
 *   node scripts/normalize-tt-metal-copy.js --check
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const checkOnly = process.argv.includes('--check');
const FROM_RE = /\bTT Metal\b/g;
const TO = 'TT-Metalium';

const EXEC_FENCE_LANGS = new Set([
  'bash', 'sh', 'shell', 'python', 'py', 'javascript', 'js', 'typescript', 'ts',
  'json', 'yaml', 'yml', 'dockerfile', 'cmake', 'rust', 'go', 'sql', 'html', 'css',
  'ruby', 'java', 'cpp', 'c',
]);

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates',
  'docs',
  'plans',
];

const COPY_FILES = [
  'README.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CLAUDE.md',
];

function replaceSegment(segment) {
  if (!FROM_RE.test(segment)) {
    FROM_RE.lastIndex = 0;
    return segment;
  }
  FROM_RE.lastIndex = 0;
  return segment.replace(FROM_RE, TO);
}

function transformMarkdownLine(line) {
  const urlSlots = [];
  const s = line.replace(/\]\([^)]*\)/g, (m) => {
    urlSlots.push(m);
    return `\x00U${urlSlots.length - 1}\x00`;
  });

  const parts = s.split(/(`[^`]*`)/);
  const out = parts.map((part, i) => (i % 2 === 1 ? part : replaceSegment(part)));
  return out.join('').replace(/\x00U(\d+)\x00/g, (_, idx) => urlSlots[Number(idx)]);
}

function transformContent(text) {
  const lines = text.split('\n');
  const hasYamlFrontmatter = lines.length > 0 && lines[0] === '---';
  let inFrontmatter = false;
  let frontmatterDone = !hasYamlFrontmatter;
  let inFence = false;
  let fenceLang = '';
  let changed = false;

  const out = lines.map((line) => {
    if (!frontmatterDone) {
      if (line === '---') {
        if (!inFrontmatter) {
          inFrontmatter = true;
          return line;
        }
        inFrontmatter = false;
        frontmatterDone = true;
        return line;
      }
      if (inFrontmatter) {
        return line;
      }
    }

    const trimmed = line.trimStart();
    const fence = trimmed.startsWith('```') || trimmed.startsWith('~~~');
    if (fence) {
      if (!inFence) {
        fenceLang = trimmed.slice(3).replace(/~+$/, '').trim().toLowerCase();
      } else {
        fenceLang = '';
      }
      inFence = !inFence;
      return line;
    }

    if (inFence && EXEC_FENCE_LANGS.has(fenceLang)) {
      return line;
    }

    const next = transformMarkdownLine(line);
    if (next !== line) {
      changed = true;
    }
    return next;
  });

  return { text: out.join('\n'), changed };
}

function collectFiles(dir, acc) {
  if (!fs.existsSync(dir)) return;
  const stat = fs.statSync(dir);
  if (stat.isFile()) {
    const ext = path.extname(dir);
    if (['.md', '.html'].includes(ext)) acc.push(dir);
    return;
  }
  for (const entry of fs.readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'vendor' || entry === 'site') continue;
    collectFiles(path.join(dir, entry), acc);
  }
}

function main() {
  const files = [];
  for (const rel of COPY_ROOTS) {
    collectFiles(path.join(ROOT, rel), files);
  }
  for (const rel of COPY_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) files.push(abs);
  }

  let touched = 0;
  for (const abs of files) {
    const raw = fs.readFileSync(abs, 'utf8');
    const before = (raw.match(FROM_RE) || []).length;
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const after = (text.match(FROM_RE) || []).length;
    touched++;
    const rel = path.relative(ROOT, abs);
    if (checkOnly) {
      console.log(`  ${rel} (${before - after} would change)`);
    } else {
      fs.writeFileSync(abs, text, 'utf8');
      console.log(`  ${rel} (${before - after} replacements)`);
    }
  }

  if (checkOnly) {
    if (touched > 0) {
      console.error(`\n${touched} file(s) still contain prose "TT Metal".`);
      process.exit(1);
    }
    console.log('OK: no prose TT Metal in copy paths.');
    process.exit(0);
  }

  console.log(touched ? `\nDone: ${touched} file(s) updated.` : 'No files needed TT Metal updates.');
}

main();
