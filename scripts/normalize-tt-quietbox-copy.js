#!/usr/bin/env node
/**
 * Prefix user-facing prose: QuietBox → TT-QuietBox (skips existing TT-QuietBox).
 * Protects quietbox2 hostnames/URLs and markdown link destinations.
 *
 * Usage:
 *   node scripts/normalize-tt-quietbox-copy.js
 *   node scripts/normalize-tt-quietbox-copy.js --check
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const RE = /(?<!TT-)QuietBox/g;

const COPY_ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates',
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
  'COMMUNITY_GUIDELINES.md',
  'content/lesson-registry.json',
];

function protectTechnical(segment, slots) {
  const slot = (m) => {
    const i = slots.length;
    slots.push(m);
    return `\x00S${i}\x00`;
  };

  let s = segment;
  s = s.replace(/https?:\/\/[^\s)'"`<>]*/gi, (m) =>
    /quietbox/i.test(m) ? slot(m) : m
  );
  s = s.replace(/\bquietbox2\b/gi, slot);
  s = s.replace(/http:\/\/quietbox2[^\s)'"`]*/gi, slot);
  return s;
}

function replaceInSegment(segment) {
  if (!RE.test(segment)) {
    RE.lastIndex = 0;
    return segment;
  }
  RE.lastIndex = 0;
  const slots = [];
  let s = protectTechnical(segment, slots);
  s = s.replace(RE, 'TT-QuietBox');
  return s.replace(/\x00S(\d+)\x00/g, (_, i) => slots[Number(i)]);
}

function transformMarkdownLine(line) {
  const urlSlots = [];
  const s = line.replace(/\]\([^)]*\)/g, (m) => {
    urlSlots.push(m);
    return `\x00U${urlSlots.length - 1}\x00`;
  });

  const parts = s.split(/(`[^`]*`)/);
  const out = parts.map((part, i) => (i % 2 === 1 ? part : replaceInSegment(part)));
  return out.join('').replace(/\x00U(\d+)\x00/g, (_, idx) => urlSlots[Number(idx)]);
}

function transformContent(text) {
  const lines = text.split('\n');
  let changed = false;
  const out = lines.map((line) => {
    const next = transformMarkdownLine(line);
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

function countNeedles(text) {
  const re = /(?<!TT-)QuietBox/g;
  return (text.match(re) || []).length;
}

function main() {
  const checkOnly = process.argv.includes('--check');
  const files = [];
  for (const rel of COPY_ROOTS) collectFiles(path.join(ROOT, rel), files);
  for (const rel of COPY_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) files.push(abs);
  }

  let touched = 0;
  let count = 0;

  for (const abs of files) {
    const raw = fs.readFileSync(abs, 'utf8');
    const before = countNeedles(raw);
    if (before === 0) continue;
    const { text, changed } = transformContent(raw);
    if (!changed) continue;
    const after = countNeedles(text);
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
      console.error(`\n${touched} file(s) still have unprefixed QuietBox.`);
      process.exit(1);
    }
    console.log('OK: all copy-path QuietBox mentions use TT- prefix.');
    process.exit(0);
  }

  console.log(`\nDone: ${touched} files, ${count} QuietBox → TT-QuietBox.\n`);
}

main();
