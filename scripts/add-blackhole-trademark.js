#!/usr/bin/env node
/**
 * First mention of "Blackhole" per page → Blackhole<sup>®</sup>
 * Skips YAML front matter, fenced code, inline `code`, and HTML script/style blocks.
 *
 * Usage:
 *   node scripts/add-blackhole-trademark.js          # apply
 *   node scripts/add-blackhole-trademark.js --check  # exit 1 if any file needs update
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const MARK = 'Blackhole<sup>®</sup>';
const WORD_RE = /\bBlackhole\b/;

const ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates',
  'docs',
  'plans',
];

const FILES = ['README.md'];

const SKIP_FILES = new Set([
  path.join('content', 'lesson-registry.json'),
]);

function collectFiles() {
  const out = new Set();
  for (const rel of FILES) {
    const abs = path.join(ROOT, rel);
    if (fs.existsSync(abs)) {
      out.add(rel);
    }
  }
  for (const root of ROOTS) {
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
        } else if (/\.(md|html)$/i.test(name)) {
          out.add(rel);
        }
      }
    };
    walk(absRoot, root);
  }
  return [...out].sort();
}

function replaceFirstInSegment(segment) {
  if (!WORD_RE.test(segment) || segment.includes(MARK)) {
    WORD_RE.lastIndex = 0;
    return { segment, replaced: false };
  }
  WORD_RE.lastIndex = 0;
  return {
    segment: segment.replace(WORD_RE, MARK),
    replaced: true,
  };
}

function transformMarkdownLine(line) {
  const parts = line.split(/(`[^`]*`)/);
  for (let i = 0; i < parts.length; i += 2) {
    const { segment, replaced } = replaceFirstInSegment(parts[i]);
    if (replaced) {
      parts[i] = segment;
      return parts.join('');
    }
  }
  return line;
}

function transformMarkdown(text) {
  if (text.includes(MARK)) {
    return text;
  }

  const lines = text.split('\n');
  let inFrontmatter = false;
  let frontmatterDone = false;
  let replaced = false;
  const out = [];

  for (const line of lines) {
    if (!frontmatterDone && line === '---') {
      if (!inFrontmatter) {
        inFrontmatter = true;
        out.push(line);
        continue;
      }
      inFrontmatter = false;
      frontmatterDone = true;
      out.push(line);
      continue;
    }
    if (inFrontmatter || replaced) {
      out.push(line);
      continue;
    }

    const next = transformMarkdownLine(line);
    if (next !== line) {
      replaced = true;
    }
    out.push(next);
  }

  return replaced ? out.join('\n') : text;
}

function transformHtml(text) {
  if (text.includes(MARK)) {
    return text;
  }

  const lines = text.split('\n');
  let inScript = false;
  let inStyle = false;
  let replaced = false;
  const out = [];

  for (const line of lines) {
    const lower = line.toLowerCase();
    if (/<script\b/.test(lower)) {
      inScript = true;
    }
    if (/<style\b/.test(lower)) {
      inStyle = true;
    }
    if (inScript || inStyle || replaced) {
      out.push(line);
      if (/<\/script>/.test(lower)) {
        inScript = false;
      }
      if (/<\/style>/.test(lower)) {
        inStyle = false;
      }
      continue;
    }

    if (!WORD_RE.test(line)) {
      out.push(line);
      continue;
    }

    const { segment, replaced: did } = replaceFirstInSegment(line);
    if (did) {
      replaced = true;
    }
    out.push(segment);
  }

  return replaced ? out.join('\n') : text;
}

function transformFile(rel) {
  const abs = path.join(ROOT, rel);
  const before = fs.readFileSync(abs, 'utf8');
  if (!WORD_RE.test(before)) {
    return { rel, changed: false };
  }

  const after = rel.endsWith('.html')
    ? transformHtml(before)
    : transformMarkdown(before);

  return { rel, changed: after !== before, after };
}

function main() {
  const checkOnly = process.argv.includes('--check');
  let pending = 0;

  for (const rel of collectFiles()) {
    const { changed, after } = transformFile(rel);
    if (!changed) {
      continue;
    }
    pending += 1;
    if (checkOnly) {
      console.log(`needs trademark: ${rel}`);
    } else {
      fs.writeFileSync(path.join(ROOT, rel), after, 'utf8');
      console.log(`updated: ${rel}`);
    }
  }

  if (checkOnly) {
    if (pending > 0) {
      console.error(`\n${pending} file(s) missing first-mention Blackhole<sup>®</sup>.`);
      process.exit(1);
    }
    console.log('OK: all pages with Blackhole have first-mention trademark.');
    process.exit(0);
  }

  console.log(pending ? `\n${pending} file(s) updated.` : 'No Blackhole pages needed updates.');
}

main();
