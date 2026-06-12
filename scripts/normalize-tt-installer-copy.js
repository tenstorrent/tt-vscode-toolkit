#!/usr/bin/env node
/**
 * Normalize user-facing prose: tt-installer → TT-Installer.
 * Skips code fences, paths, URLs, lesson slugs, and command IDs.
 *
 * Usage: node scripts/normalize-tt-installer-copy.js
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FROM = 'tt-installer';
const TO = 'TT-Installer';
const FROM_RE = /\btt-installer\b/g;

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'docs',
  'plans',
  '.github/ISSUE_TEMPLATE',
];

const COPY_FILES = [
  'README.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CLAUDE.md',
  'content/lesson-registry.json',
];

function replaceInProse(line) {
  if (!FROM_RE.test(line)) {
    FROM_RE.lastIndex = 0;
    return line;
  }
  FROM_RE.lastIndex = 0;

  const slots = [];
  const slot = (m) => {
    const i = slots.length;
    slots.push(m);
    return `\x00S${i}\x00`;
  };

  let s = line;

  s = s.replace(/\]\(https?:\/\/[^\s)]*tt-installer[^\s)]*\)/g, slot);
  s = s.replace(/https?:\/\/[^\s)'"`<>]*tt-installer[^\s)'"`<>]*/g, slot);
  s = s.replace(/github\.com\/tenstorrent\/tt-installer[^\s)`]*/g, slot);

  s = s.replace(/showLesson\?\[["']tt-installer["']\]/g, slot);
  s = s.replace(/openWalkthrough\(\s*['"]tt-installer['"]\s*\)/g, slot);
  s = s.replace(/\/lessons\/tt-installer\/?/g, slot);
  s = s.replace(/"id"\s*:\s*"tt-installer"/g, slot);
  s = s.replace(/^id:\s*tt-installer\s*$/gm, slot);
  s = s.replace(/"(nextLesson|previousLesson)"\s*:\s*"tt-installer"/g, slot);
  s = s.replace(/tt-installer\.md/g, slot);
  s = s.replace(/00-tt-installer\.md/g, slot);

  s = s.replace(/`?vendor\/tt-installer[\w./-]*`?/g, slot);
  s = s.replace(/`?~?\/[\w./-]*tt-installer[\w./-]*`?/g, (m) =>
    m.includes('tt-installer') ? slot(m) : m
  );
  s = s.replace(/tt-installer\.git/g, slot);
  s = s.replace(/`tt-installer`/g, slot);

  s = s.replace(FROM_RE, TO);

  return s.replace(/\x00S(\d+)\x00/g, (_, i) => slots[Number(i)]);
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
    if (inFence && !inMermaid) return line;

    const next = replaceInProse(line);
    if (next !== line) changed = true;
    return next;
  });

  return { text: out.join('\n'), changed };
}

function countMatches(text) {
  return (text.match(FROM_RE) || []).length;
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
  for (const rel of COPY_ROOTS) collectFiles(path.join(ROOT, rel), files);
  for (const rel of COPY_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) files.push(abs);
  }

  let touched = 0;
  let replacements = 0;

  for (const abs of files) {
    const raw = fs.readFileSync(abs, 'utf8');
    const before = countMatches(raw);
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const after = countMatches(text);
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} ${FROM} → ${TO} in prose.\n`);
}

main();
