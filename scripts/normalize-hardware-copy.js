#!/usr/bin/env node
/**
 * Normalize hardware model IDs in user-facing copy (e.g. N300 → n300).
 * Skips code fences (except mermaid), shell env vars, and command identifiers.
 *
 * Usage:
 *   node scripts/normalize-hardware-copy.js N300
 *   node scripts/normalize-hardware-copy.js N150
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

/** Prose replacements (uppercase ID → display form). Others default to lowercase. */
const PROSE_FORM = {
  T3K: 'T3000',
};

const hwArg = process.argv[2];
if (!hwArg || !/^(N\d{3}|P\d{3}c?|T3K)$/i.test(hwArg)) {
  console.error('Usage: node scripts/normalize-hardware-copy.js <N150|N300|T3K|P150|P300|P300c|…>');
  process.exit(1);
}

// P300c: match P300c and P300C in copy
const IS_P300C = /^P300c$/i.test(hwArg);
const UPPER = IS_P300C ? 'P300C' : hwArg.toUpperCase();
const LOWER = IS_P300C ? 'p300c' : (PROSE_FORM[UPPER] || hwArg.toLowerCase());
const MATCH_RE = IS_P300C ? /\bP300[cC]\b/g : new RegExp(`\\b${UPPER}\\b`, 'g');

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates/cookbook',
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

/** Lines that must not be altered (shell, commands, APIs). */
function isCodeLine(line) {
  const u = UPPER;
  const patterns = [
    new RegExp(`MESH_DEVICE\\s*=\\s*${u}\\b`),
    new RegExp(`MESH_DEVICE['"]?\\s*:\\s*['"]${u}['"]`),
    new RegExp(`command:tenstorrent\\.\\w*${u}`, 'i'),
    new RegExp(`tenstorrent\\.\\w*${u}`, 'i'),
    new RegExp(`\\w+${u}\\w*\\(`),
    new RegExp(`lessonCommand${u}`, 'i'),
    new RegExp(`startVllmServer${u}`, 'i'),
    new RegExp(`startTtInferenceServer${u}`, 'i'),
    new RegExp(`generateImage${u}`, 'i'),
    new RegExp(`START_[A-Z_]*${u}`),
    new RegExp(`\\{"hardware"\\s*:\\s*"${u}"\\}`),
    new RegExp(`hardware=${u}`, 'i'),
    new RegExp(`data-hw=["']${u}["']`, 'i'),
    new RegExp(`${u}Qwen`, 'i'),
  ];
  return patterns.some((p) => p.test(line));
}

function countMatches(text) {
  const re = IS_P300C ? /\bP300[cC]\b/g : new RegExp(`\\b${UPPER}\\b`, 'g');
  return (text.match(re) || []).length;
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
    const inHtmlSummary =
      !inFence &&
      (IS_P300C
        ? /<summary[^>]*>.*\bP300[cC]\b/i.test(line)
        : new RegExp(`<summary[^>]*>.*\\b${UPPER}\\b`, 'i').test(line));

    if (inFence && !inMermaid) return line;
    if (!inMermaid && !inHtmlSummary && isCodeLine(line)) return line;
    if (!MATCH_RE.test(line)) {
      MATCH_RE.lastIndex = 0;
      return line;
    }
    MATCH_RE.lastIndex = 0;
    changed = true;
    return line.replace(IS_P300C ? /\bP300[cC]\b/g : new RegExp(`\\b${UPPER}\\b`, 'g'), LOWER);
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

  const label = IS_P300C ? 'P300c/P300C' : UPPER;
  console.log(`\nDone: ${touched} files, ~${replacements} ${label} → ${LOWER} in prose.\n`);
}

main();
