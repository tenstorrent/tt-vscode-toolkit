#!/usr/bin/env node
/**
 * Normalize prose: TTNN → TT-NN
 * Skips code fences, inline `code`, markdown link URLs, and technical lines.
 *
 * Usage:
 *   node scripts/normalize-ttnn-copy.js
 *   node scripts/normalize-ttnn-copy.js --check
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FROM_RE = /\bTTNN\b/g;
const TO = 'TT-NN';

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates',
  'docs',
  'plans',
  '.github/ISSUE_TEMPLATE',
];

const COPY_FILES = [
  'README.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CLAUDE.md',
  'TT_METAL_PRECOMPILED.md',
  'content/lesson-registry.json',
];

function replaceInProseSegment(segment) {
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
  const out = parts.map((part, i) => {
    if (i % 2 === 1) {
      return part;
    }
    return replaceInProseSegment(part);
  });

  return out.join('').replace(/\x00U(\d+)\x00/g, (_, idx) => urlSlots[Number(idx)]);
}

function transformContent(text) {
  const lines = text.split('\n');
  let inFence = false;
  let changed = false;

  const out = lines.map((line) => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith('```') || trimmed.startsWith('~~~')) {
      inFence = !inFence;
      return line;
    }
    if (inFence) {
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
    if (['.md', '.html', '.yml', '.yaml', '.json'].includes(ext)) {
      acc.push(dir);
    }
    return;
  }
  for (const entry of fs.readdirSync(dir)) {
    if (entry === 'node_modules' || entry === 'vendor' || entry === 'site') continue;
    collectFiles(path.join(dir, entry), acc);
  }
}

function main() {
  const checkOnly = process.argv.includes('--check');
  const files = [];
  for (const rel of COPY_ROOTS) {
    collectFiles(path.join(ROOT, rel), files);
  }
  for (const rel of COPY_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) {
      files.push(abs);
    }
  }

  let touched = 0;
  let count = 0;

  for (const abs of files) {
    const raw = fs.readFileSync(abs, 'utf8');
    const before = (raw.match(FROM_RE) || []).length;
    const { text, changed } = transformContent(raw);
    if (!changed) {
      continue;
    }
    const after = (text.match(FROM_RE) || []).length;
    const n = before - after;
    count += n;
    touched++;
    if (checkOnly) {
      console.log(`  ${path.relative(ROOT, abs)} (${n} would change)`);
    } else {
      fs.writeFileSync(abs, text, 'utf8');
      console.log(`  ${path.relative(ROOT, abs)} (${n} replacements)`);
    }
  }

  if (checkOnly) {
    if (touched > 0) {
      console.error(`\n${touched} file(s) still contain prose TTNN.`);
      process.exit(1);
    }
    console.log('OK: no prose TTNN in copy paths.');
    process.exit(0);
  }

  console.log(`\nDone: ${touched} files, ${count} TTNN → TT-NN in prose.\n`);
}

main();
