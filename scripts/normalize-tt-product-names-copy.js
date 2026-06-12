#!/usr/bin/env node
/**
 * Normalize Tenstorrent product names in user-facing prose:
 *   tt-metal, TT-Metal → TT-Metalium
 *   tt-metalium → TT-Metalium
 *   tt-forge → TT-Forge (not tt-forge-fe / tt-forge-models)
 *   tt-xla → TT-XLA (not lesson slug tt-xla-jax)
 *   tt-lang → TT-Lang (not tt-lang-intro slug)
 *
 * Skips code fences, paths, URLs, env vars, and lesson slugs.
 *
 * Usage: node scripts/normalize-tt-product-names-copy.js
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
];

const COPY_FILES = [
  'README.md',
  'CONTRIBUTING.md',
  'CHANGELOG.md',
  'CLAUDE.md',
  'TT_METAL_PRECOMPILED.md',
  'content/lesson-registry.json',
];

const REPLACEMENTS = [
  [/TT-Metal(?!ium)/g, 'TT-Metalium'],
  [/\btt-metalium\b/g, 'TT-Metalium'],
  [/\btt-metal\b/g, 'TT-Metalium'],
  [/\btt-forge\b(?!-fe|-models)/g, 'TT-Forge'],
  [/\btt-xla\b(?!-jax|-venv)/g, 'TT-XLA'],
  [/\btt-lang\b(?!-intro)/g, 'TT-Lang'],
];

const NEEDLE_RES = [
  /TT-Metal(?!ium)/,
  /\btt-metalium\b/,
  /\btt-metal\b/,
  /\btt-forge\b(?!-fe|-models)/,
  /\btt-xla\b(?!-jax|-venv)/,
  /\btt-lang\b(?!-intro)/,
];

function lineNeedsWork(line) {
  return NEEDLE_RES.some((re) => {
    re.lastIndex = 0;
    return re.test(line);
  });
}

function protectTechnical(line, slots) {
  const slot = (m) => {
    const i = slots.length;
    slots.push(m);
    return `\x00S${i}\x00`;
  };

  let s = line;

  // Markdown link destinations
  s = s.replace(/\]\(https?:\/\/[^\s)]*\)/g, (m) =>
    /tt-metal|tt-forge|tt-xla|tt-lang/i.test(m) ? slot(m) : m
  );

  // URLs
  s = s.replace(/https?:\/\/[^\s)'"`<>]+/g, (m) =>
    /tt-metal|tt-forge|tt-xla|tt-lang/i.test(m) ? slot(m) : m
  );

  // Lesson slugs / walkthroughs
  const slugPatterns = [
    /showLesson\?\[["'][^"']+["']\]/g,
    /openWalkthrough\(\s*['"][^'"]+['"]\s*\)/g,
    /\/lessons\/[a-z0-9-]+\/?/g,
    /"id"\s*:\s*"[^"]+"/g,
    /^id:\s*[a-z0-9-]+\s*$/gm,
    /"(nextLesson|previousLesson)"\s*:\s*"[^"]+"/g,
    /[a-z0-9-]+\.md/g,
  ];
  for (const re of slugPatterns) {
    s = s.replace(re, (m) =>
      /tt-metal|tt-forge|tt-xla|tt-lang|build-tt/i.test(m) ? slot(m) : m
    );
  }

  // Paths and repos
  s = s.replace(/`[^`]*`/g, (m) =>
    /tt-metal|tt-forge|tt-xla|tt-lang|TT_METAL/i.test(m) ? slot(m) : m
  );
  s = s.replace(
    /~?\/[\w./-]*(?:tt-metal|tt-forge-fe|tt-forge-models|tt-xla|tt-lang)[\w./-]*/g,
    slot
  );
  s = s.replace(/\b[\w-]*\/tt-metal[\w./-]*/g, slot);
  s = s.replace(/tt-metal\.git|tt-forge-fe|tt-forge-models|tt-xla-venv|ttlang/gi, (m) =>
    slot(m)
  );

  // Env / CLI lines
  if (
    /\b(export|source|cd|git clone|pkill|PYTHONPATH|TT_METAL_HOME|setup-tt-|build_metal|python3 -c)\b/i.test(
      line
    ) &&
    /tt-metal|tt-forge|tt-xla/i.test(line)
  ) {
    return slot(line);
  }

  return s;
}

function replaceInProse(line) {
  if (!lineNeedsWork(line)) return line;

  const slots = [];
  let s = protectTechnical(line, slots);

  for (const [re, to] of REPLACEMENTS) {
    re.lastIndex = 0;
    s = s.replace(re, to);
  }

  return s.replace(/\x00S(\d+)\x00/g, (_, i) => slots[Number(i)]);
}

function countNeedles(text) {
  let n = 0;
  for (const re of NEEDLE_RES) {
    const m = text.match(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g'));
    if (m) n += m.length;
  }
  return n;
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

    if (inFence && fenceLang !== 'mermaid') return line;

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
    const before = countNeedles(raw);
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const after = countNeedles(text);
    replacements += before - after;
    fs.writeFileSync(abs, text, 'utf8');
    touched++;
    console.log(`  ${path.relative(ROOT, abs)} (${before - after} replacements)`);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} product name updates in prose.\n`);
}

main();
