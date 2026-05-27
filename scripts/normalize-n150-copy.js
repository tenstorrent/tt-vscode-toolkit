#!/usr/bin/env node
/**
 * Replace N150 → n150 in user-facing copy only (not code).
 * One-off maintenance script; safe to re-run (no-op when already normalized).
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

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
  'styles/Tenstorrent/ProductNames.yml',
];

/** Lines that must not be altered (shell, commands, APIs). */
function isCodeLine(line) {
  if (/MESH_DEVICE\s*=\s*N150\b/.test(line)) return true;
  if (/MESH_DEVICE['"]?\s*:\s*['"]N150['"]/.test(line)) return true;
  if (/command:tenstorrent\.\w*N150/i.test(line)) return true;
  if (/tenstorrent\.\w*N150/i.test(line)) return true;
  if (/\w+N150\w*\(/.test(line)) return true;
  if (/lessonCommandN150/i.test(line)) return true;
  if (/startVllmServerN150/i.test(line)) return true;
  if (/START_[A-Z_]*N150/.test(line)) return true;
  if (/\{"hardware"\s*:\s*"N150"\}/.test(line)) return true;
  if (/hardware=N150/i.test(line)) return true;
  if (/data-hw=["']N150["']/i.test(line)) return true;
  return false;
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
      !inFence && /<summary[^>]*>.*\bN150\b/i.test(line);

    if (inFence && !inMermaid) return line;
    if (!inMermaid && !inHtmlSummary && isCodeLine(line)) return line;
    if (!/\bN150\b/.test(line)) return line;

    changed = true;
    return line.replace(/\bN150\b/g, 'n150');
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
    const before = (raw.match(/\bN150\b/g) || []).length;
    const after = (text.match(/\bN150\b/g) || []).length;
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} N150 → n150 in prose.\n`);
}

main();
