#!/usr/bin/env node
/**
 * Normalize user-facing prose:
 *   tt-vscode-toolkit → TT-VSCode-Toolkit
 *   TT Developer Toolkit → TT-VSCode-Toolkit
 *   Tenstorrent VSCode Toolkit → TT-VSCode-Toolkit
 *
 * Skips code fences, paths, URLs, extension IDs, and lesson slugs.
 *
 * Usage: node scripts/normalize-tt-vscode-toolkit-copy.js
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const TO = 'TT-VSCode-Toolkit';

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
  'TT_METAL_PRECOMPILED.md',
  'SECURITY.md',
  'COMMUNITY_GUIDELINES.md',
  'content/lesson-registry.json',
];

const SLUG_RE = /\btt-vscode-toolkit\b/g;
const DEV_RE = /TT Developer Toolkit/g;
const TEN_RE = /Tenstorrent VSCode Toolkit/g;

function protectTechnical(line, slots) {
  const slot = (m) => {
    const i = slots.length;
    slots.push(m);
    return `\x00S${i}\x00`;
  };

  let s = line;

  s = s.replace(/https?:\/\/[^\s)'"`<>]*tt-vscode-toolkit[^\s)'"`<>]*/g, slot);
  s = s.replace(/ghcr\.io\/[^\s`]*tt-vscode-toolkit[^\s`]*/g, slot);
  s = s.replace(/\]\(https?:\/\/[^\s)]*tt-vscode-toolkit[^\s)]*\)/g, slot);
  s = s.replace(/https?:\/\/[^\s)'"`<>]*tt-vscode-toolkit[^\s)'"`<>]*/g, slot);
  s = s.replace(/docs\.tenstorrent\.com\/tt-vscode-toolkit[^\s)`]*/g, slot);
  s = s.replace(/itemName=Tenstorrent\.tt-vscode-toolkit/g, slot);
  s = s.replace(/Tenstorrent\.tt-vscode-toolkit/g, slot);
  s = s.replace(/open-vsx\.org\/extension\/Tenstorrent\/tt-vscode-toolkit/g, slot);

  s = s.replace(/tt-vscode-toolkit-\*\.vsix/g, slot);
  s = s.replace(/tt-vscode-toolkit-[0-9][\w.-]*\.vsix/g, slot);
  s = s.replace(/tt-vscode-toolkit\.git/g, slot);

  s = s.replace(/`?~?\/[\w./-]*tt-vscode-toolkit[\w./-]*`?/g, (m) =>
    m.includes('tt-vscode-toolkit') ? slot(m) : m
  );
  s = s.replace(/`tt-vscode-toolkit`/g, slot);
  s = s.replace(/\b(cd|ls|git clone|--repo|--app|--docker|FROM|fetch\()\b[^\n]*tt-vscode-toolkit/g, slot);

  // Container / CLI tokens (not prose)
  if (/\b(--app|--docker|container_name:|SITE_BASE_PATH:|name=)tt-vscode-toolkit\b/.test(s)) {
    s = s.replace(/(--app|--docker|container_name:|SITE_BASE_PATH:|name=)tt-vscode-toolkit/g, (m) =>
      slot(m)
    );
  }
  s = s.replace(/\btt-vscode-toolkit\/vscode\b/g, slot);
  s = s.replace(/\btt-vscode-toolkit:\s*latest\b/g, slot);

  return s;
}

function replaceProse(line) {
  if (!SLUG_RE.test(line) && !DEV_RE.test(line) && !TEN_RE.test(line)) {
    return line;
  }
  SLUG_RE.lastIndex = 0;

  const slots = [];
  let s = protectTechnical(line, slots);

  s = s.replace(TEN_RE, TO);
  s = s.replace(DEV_RE, TO);
  s = s.replace(SLUG_RE, TO);

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

    const next = replaceProse(line);
    if (next !== line) changed = true;
    return next;
  });

  return { text: out.join('\n'), changed };
}

function countBefore(text) {
  return (
    (text.match(SLUG_RE) || []).length +
    (text.match(DEV_RE) || []).length +
    (text.match(TEN_RE) || []).length
  );
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
    const before = countBefore(raw);
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const after = countBefore(text);
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} → ${TO} in prose.\n`);
}

main();
