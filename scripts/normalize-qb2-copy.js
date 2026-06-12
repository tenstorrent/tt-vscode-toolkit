#!/usr/bin/env node
/**
 * Replace QB2 → TT-QuietBox 2 in user-facing copy only (not code, slugs, or paths).
 *
 * Usage: node scripts/normalize-qb2-copy.js
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FROM = 'QB2';
const TO = 'TT-QuietBox 2';
const MATCH_RE = /\bQB2\b/g;

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'docs',
  'plans',
  '.github/ISSUE_TEMPLATE',
  '.github/workflows',
];

const COPY_FILES = [
  'README.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CLAUDE.md',
  'TT_METAL_PRECOMPILED.md',
  'content/lesson-registry.json',
];

function isCodeLine(line) {
  const patterns = [
    /\bqb2[-_]/i,
    /\/qb2/i,
    /content\/lessons\/qb2/i,
    /command:tenstorrent\.\w*qb2/i,
    /tenstorrent\.\w*qb2/i,
    /\w+QB2\w*\(/i,
    /createQB2/i,
    /APPLY_QB2/i,
    /\bQB2_/,
    /\["qb2-/i,
    /showLesson\?\["qb2/i,
    /"id"\s*:\s*"qb2/i,
    /markdownFile.*qb2/i,
    /nextLesson.*qb2/i,
    /previousLesson.*qb2/i,
    /~\/?qb2/i,
    /qb2[-\w]*\.md/i,
    /\.md.*\bqb2/i,
    /setup_qb2/i,
    /"qb2-demos"/i,
    /category.*qb2-demos/i,
    /^\s*-\s*qb2\s*$/,
    /^\s*"qb2",?\s*$/,
  ];
  return patterns.some((p) => p.test(line));
}

function transformContent(text) {
  const lines = text.split('\n');
  let inFence = false;
  let fenceLang = '';
  let changed = false;

  const out = lines.map((line) => {
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

    const inMermaid = inFence && fenceLang === 'mermaid';
    const inHtmlComment = !inFence && /<!--.*\bQB2\b/.test(line);

    if (inFence && !inMermaid && !inHtmlComment) return line;
    if (!inMermaid && !inHtmlComment && isCodeLine(line)) return line;
    if (!MATCH_RE.test(line)) {
      MATCH_RE.lastIndex = 0;
      return line;
    }
    MATCH_RE.lastIndex = 0;
    changed = true;
    return line.replace(/\bQB2\b/g, TO);
  });

  return { text: out.join('\n'), changed };
}

function collectFiles(dir, acc) {
  if (!fs.existsSync(dir)) return;
  const stat = fs.statSync(dir);
  if (stat.isFile()) {
    const ext = path.extname(dir);
    if (['.md', '.html', '.yml', '.yaml', '.json'].includes(ext)) acc.push(dir);
    return;
  }
  for (const entry of fs.readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'vendor' || entry === 'site') continue;
    collectFiles(path.join(dir, entry), acc);
  }
}

function countMatches(text) {
  return (text.match(MATCH_RE) || []).length;
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
  let replacements = 0;

  for (const abs of files) {
    const raw = fs.readFileSync(abs, 'utf8');
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const before = countMatches(raw);
    const after = countMatches(text);
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} QB2 → TT-QuietBox 2 in prose.\n`);
}

main();
