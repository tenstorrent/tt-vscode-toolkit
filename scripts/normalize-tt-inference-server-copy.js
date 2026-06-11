#!/usr/bin/env node
/**
 * Normalize user-facing prose: tt-inference-server → TT-Inference-Server.
 * Skips code fences, paths, URLs, Docker images, lesson slugs, and command IDs.
 *
 * Usage: node scripts/normalize-tt-inference-server-copy.js
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const FROM = 'tt-inference-server';
const TO = 'TT-Inference-Server';
const FROM_RE = /\btt-inference-server\b/g;

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

/** Protect technical spans; return restored string after replacement. */
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

  // URLs and registry paths
  s = s.replace(/https?:\/\/[^\s)'"`<>]*tt-inference-server[^\s)'"`<>]*/g, slot);
  s = s.replace(/ghcr\.io\/[^\s`]*tt-inference-server[^\s`]*/g, slot);
  s = s.replace(/github\.com\/tenstorrent\/tt-inference-server[^\s)\]\x00]*/g, slot);

  // Lesson / walkthrough identifiers
  s = s.replace(/openWalkthrough\(\s*['"]tt-inference-server['"]\s*\)/g, slot);
  s = s.replace(/\/lessons\/tt-inference-server\/?/g, slot);
  s = s.replace(/"id"\s*:\s*"tt-inference-server"/g, slot);
  s = s.replace(/^id:\s*tt-inference-server\s*$/gm, slot);
  s = s.replace(/"(nextLesson|previousLesson)"\s*:\s*"tt-inference-server"/g, slot);
  s = s.replace(/tt-inference-server\.md/g, slot);

  // Paths (with or without backticks)
  s = s.replace(
    /`?~?\/[\w./-]*tt-inference-server[\w./-]*`?/g,
    (m) => (m.includes('tt-inference-server') ? slot(m) : m)
  );
  s = s.replace(/`?vendor\/tt-inference-server[\w./-]*`?/g, slot);
  s = s.replace(/`?[\w-]*\/tt-inference-server[\w./-]*`?/g, (m) => {
    if (
      /\b(cd|ls|nano|git clone|docker|--dir|ancestor=|filter)\b/i.test(line) ||
      m.startsWith('`') ||
      m.includes('/')
    ) {
      return slot(m);
    }
    return m;
  });

  // git clone targets
  s = s.replace(/tt-inference-server\.git/g, slot);

  // Inline code that is only the repo slug (paths handled above)
  s = s.replace(/`tt-inference-server`/g, slot);

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
  return (text.match(FROM_RE) || []).length;
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
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const before = countMatches(raw);
    const after = countMatches(text);
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} ${FROM} → ${TO} in prose.\n`);
}

main();
