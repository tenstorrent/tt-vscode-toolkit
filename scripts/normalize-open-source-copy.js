#!/usr/bin/env node
/**
 * Normalize user-facing prose: open-source → open source (Open-source → Open source).
 * Skips code fences, Vale swap keys, style-guide negatives, and opensource.* URLs.
 *
 * Usage: node scripts/normalize-open-source-copy.js [--check]
 *   --check  Exit 1 if any file would change (audit only).
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const HYPHEN_RE = /\bopen-source\b/gi;

const SKIP_FILES = new Set([
  path.join('styles', 'Tenstorrent', 'Terminology.yml'),
]);

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
  'COMMUNITY_GUIDELINES.md',
  'CODE_OF_CONDUCT.md',
  'SECURITY.md',
  'content/lesson-registry.json',
];

function shouldSkipLine(line, relPath) {
  if (relPath === 'docs/STYLE_GUIDE.md' && line.includes('❌')) {
    return true;
  }
  if (relPath === 'CHANGELOG.md' && line.includes('`open-source`')) {
    return true;
  }
  if (/https?:\/\/[^\s)]*opensource[^\s)]*/i.test(line)) {
    return true;
  }
  return false;
}

function replaceHyphenated(line) {
  return line.replace(HYPHEN_RE, (m) => {
    const upper = m[0] === 'O' || m[0] === 'o' && m === 'Open-source';
    if (m === 'Open-source' || m === 'OPEN-SOURCE') {
      return 'Open source';
    }
    if (m === 'open-source' || m === 'OPEN-source') {
      return 'open source';
    }
    return m[0] === 'O' ? 'Open source' : 'open source';
  });
}

function transformContent(text, relPath) {
  const lines = text.split('\n');
  let inFence = false;
  const out = [];

  for (const line of lines) {
    const fence = line.match(/^(`{3,}|~{3,})/);
    if (fence) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (inFence || shouldSkipLine(line, relPath)) {
      out.push(line);
      continue;
    }
    out.push(replaceHyphenated(line));
  }

  return out.join('\n');
}

function collectFiles() {
  const files = new Set();

  for (const rel of COPY_FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) {
      files.add(rel);
    }
  }

  for (const root of COPY_ROOTS) {
    const absRoot = path.join(ROOT, root);
    if (!fs.existsSync(absRoot)) {
      continue;
    }
    const walk = (dir, base) => {
      for (const name of fs.readdirSync(dir)) {
        const abs = path.join(dir, name);
        const rel = path.join(base, name);
        if (SKIP_FILES.has(rel)) {
          continue;
        }
        const st = fs.statSync(abs);
        if (st.isDirectory()) {
          walk(abs, rel);
        } else if (/\.(md|html|json|yml|yaml)$/i.test(name)) {
          files.add(rel);
        }
      }
    };
    walk(absRoot, root);
  }

  return [...files].sort();
}

function main() {
  const checkOnly = process.argv.includes('--check');
  let changed = 0;

  for (const rel of collectFiles()) {
    const abs = path.join(ROOT, rel);
    const before = fs.readFileSync(abs, 'utf8');
    const after = transformContent(before, rel);
    if (before === after) {
      continue;
    }
    changed += 1;
    if (!checkOnly) {
      fs.writeFileSync(abs, after, 'utf8');
      console.log(`updated: ${rel}`);
    } else {
      console.log(`would update: ${rel}`);
    }
  }

  if (checkOnly) {
    if (changed > 0) {
      console.error(`\n${changed} file(s) still contain hyphenated open-source.`);
      process.exit(1);
    }
    console.log('OK: no hyphenated open-source in copy paths.');
    process.exit(0);
  }

  console.log(changed ? `\n${changed} file(s) updated.` : 'No changes needed.');
}

main();
