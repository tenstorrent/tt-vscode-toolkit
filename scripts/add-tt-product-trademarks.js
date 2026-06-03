#!/usr/bin/env node
/**
 * First prose mention per page (each term independently):
 *   TT-Metalium → TT-Metalium<sup>®</sup>
 *   TT-NN       → TT-NN<sup>®</sup>
 *   TT-Forge    → TT-Forge<sup>®</sup>
 *
 * Skips YAML front matter, fenced code, inline `code`, markdown link URLs,
 * and HTML script/style blocks. Does not match TTNN (no hyphen).
 *
 * Usage:
 *   node scripts/add-tt-product-trademarks.js
 *   node scripts/add-tt-product-trademarks.js --check
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');

const TERMS = [
  { id: 'TT-Metalium', re: /TT-Metalium(?!<sup>)/, mark: 'TT-Metalium<sup>®</sup>' },
  { id: 'TT-Forge', re: /TT-Forge(?!<sup>)/, mark: 'TT-Forge<sup>®</sup>' },
  { id: 'TT-NN', re: /TT-NN(?!<sup>)/, mark: 'TT-NN<sup>®</sup>' },
];

const ROOTS = [
  'content/lessons',
  'content/pages',
  'content/projects',
  'content/templates',
  'docs',
  'plans',
];

const FILES = [
  'README.md',
  'CLAUDE.md',
  'CHANGELOG.md',
  'COMMUNITY_GUIDELINES.md',
  'CONTRIBUTING.md',
  'TT_METAL_PRECOMPILED.md',
];

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

function replaceFirstInSegment(segment, term) {
  if (!term.re.test(segment) || segment.includes(term.mark)) {
    term.re.lastIndex = 0;
    return { segment, replaced: false };
  }
  term.re.lastIndex = 0;
  return {
    segment: segment.replace(term.re, term.mark),
    replaced: true,
  };
}

function transformMarkdownLine(line, term, done) {
  if (done.has(term.id)) {
    return { line, replaced: false };
  }

  const urlSlots = [];
  const s = line.replace(/\]\([^)]*\)/g, (m) => {
    urlSlots.push(m);
    return `\x00U${urlSlots.length - 1}\x00`;
  });

  const parts = s.split(/(`[^`]*`)/);
  for (let i = 0; i < parts.length; i += 2) {
    const { segment, replaced } = replaceFirstInSegment(parts[i], term);
    if (replaced) {
      parts[i] = segment;
      const out = parts.join('').replace(/\x00U(\d+)\x00/g, (_, idx) => urlSlots[Number(idx)]);
      return { line: out, replaced: true };
    }
  }
  return { line, replaced: false };
}

function transformMarkdown(text) {
  const lines = text.split('\n');
  let inFrontmatter = false;
  let frontmatterDone = false;
  let inFence = false;
  const done = new Set();
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
    if (inFrontmatter) {
      out.push(line);
      continue;
    }
    if (/^(`{3,}|~{3,})/.test(line.trim())) {
      inFence = !inFence;
      out.push(line);
      continue;
    }
    if (inFence) {
      out.push(line);
      continue;
    }

    let current = line;
    for (const term of TERMS) {
      const { line: next, replaced } = transformMarkdownLine(current, term, done);
      if (replaced) {
        done.add(term.id);
        current = next;
      }
    }
    out.push(current);
  }

  return { text: out.join('\n'), changed: text !== out.join('\n') };
}

function transformHtml(text) {
  const done = new Set();
  const lines = text.split('\n');
  let inScript = false;
  let inStyle = false;
  const out = [];

  for (const line of lines) {
    const lower = line.toLowerCase();
    if (/<script\b/.test(lower)) {
      inScript = true;
    }
    if (/<style\b/.test(lower)) {
      inStyle = true;
    }
    if (inScript || inStyle) {
      out.push(line);
      if (/<\/script>/.test(lower)) {
        inScript = false;
      }
      if (/<\/style>/.test(lower)) {
        inStyle = false;
      }
      continue;
    }

    let current = line;
    for (const term of TERMS) {
      if (done.has(term.id)) {
        continue;
      }
      const { segment, replaced } = replaceFirstInSegment(current, term);
      if (replaced) {
        done.add(term.id);
        current = segment;
      }
    }
    out.push(current);
  }

  return { text: out.join('\n'), changed: text !== out.join('\n') };
}

function needsAnyTerm(text) {
  return TERMS.some((t) => t.re.test(text) && !text.includes(t.mark));
}

function transformFile(rel) {
  const abs = path.join(ROOT, rel);
  const before = fs.readFileSync(abs, 'utf8');
  if (!needsAnyTerm(before)) {
    return { rel, changed: false };
  }

  const { text: after, changed } = rel.endsWith('.html')
    ? transformHtml(before)
    : transformMarkdown(before);

  return { rel, changed, after };
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
      console.error(`\n${pending} file(s) need TT product first-mention marks.`);
      process.exit(1);
    }
    console.log('OK: all pages have first-mention ® for TT-Metalium, TT-NN, and TT-Forge.');
    process.exit(0);
  }

  console.log(
    pending
      ? `\n${pending} file(s) updated.`
      : 'No files needed TT product trademark updates.',
  );
}

main();
