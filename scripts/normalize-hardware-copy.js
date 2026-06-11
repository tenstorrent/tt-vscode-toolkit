#!/usr/bin/env node
/**
 * Normalize hardware model IDs in user-facing copy (e.g. N300 → n300).
 * Also: `node scripts/normalize-hardware-copy.js galaxy` → capitalize Galaxy in prose/sample output.
 *
 * Skips shell env vars, command identifiers, and YAML metadata list entries.
 * Transforms sample-output fences (plain ``` blocks, mermaid, comments in shell).
 *
 * Usage:
 *   node scripts/normalize-hardware-copy.js N300
 *   node scripts/normalize-hardware-copy.js galaxy
 *   node scripts/normalize-hardware-copy.js --check N150
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const checkOnly = process.argv.includes('--check');
const hwArg = process.argv.slice(2).filter((a) => a !== '--check')[0];

/** Prose replacements (uppercase ID → display form). Others default to lowercase. */
const PROSE_FORM = {
  T3K: 'T3000',
};

const EXEC_FENCE_LANGS = new Set([
  'bash', 'sh', 'shell', 'python', 'py', 'javascript', 'js', 'typescript', 'ts',
  'json', 'yaml', 'yml', 'dockerfile', 'cmake', 'rust', 'go', 'sql', 'html', 'css',
  'ruby', 'java', 'cpp', 'c',
]);

const IS_GALAXY = /^galaxy$/i.test(hwArg || '');
const IS_P300C = /^P300c$/i.test(hwArg || '');

if (!hwArg || (!IS_GALAXY && !/^(N\d{3}|P\d{3}c?|T3K)$/i.test(hwArg))) {
  console.error('Usage: node scripts/normalize-hardware-copy.js [--check] <N150|N300|T3K|P150|P300|P300c|galaxy>');
  process.exit(1);
}

const UPPER = IS_GALAXY ? 'GALAXY' : (IS_P300C ? 'P300C' : hwArg.toUpperCase());
const TARGET = IS_GALAXY ? 'Galaxy' : (IS_P300C ? 'p300c' : (PROSE_FORM[UPPER] || hwArg.toLowerCase()));
const MATCH_RE = IS_GALAXY
  ? /\b(GALAXY|galaxy)\b/g
  : (IS_P300C ? /\bP300[cC]\b/g : new RegExp(`\\b${UPPER}\\b`, 'g'));

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
];

function isYamlMetadataLine(line) {
  return /^\s*-\s*(n150|n300|t3k|p100|p150|p300c|galaxy|simulator|sim)\s*$/.test(line)
    || /supportedHardware|validatedOn/.test(line)
    || /--tt-device\s+galaxy\b/.test(line)
    || /data-hw=/.test(line);
}

function guardProtectedSpans(line) {
  const slots = [];
  let s = line;
  const guards = [
    /MESH_DEVICE\s*=\s*[A-Z0-9_]+/gi,
    /MESH_DEVICE['"]?\s*:\s*['"][A-Z0-9_]+['"]/gi,
    /\{"hardware"\s*:\s*"[A-Z0-9_]+"\}/gi,
    /hardware=[A-Z0-9_]+/gi,
    /--env MESH_DEVICE=[A-Z0-9_]+/gi,
    /command:tenstorrent\.[^\s]+/gi,
    /tenstorrent\.[a-zA-Z]+[A-Z0-9]+\(/g,
    /startVllmServer[A-Z0-9]+/gi,
    /startTtInferenceServer[A-Z0-9]+/gi,
    /generateImage[A-Z0-9]+/gi,
    /lessonCommand[A-Z0-9]+/gi,
  ];
  for (const g of guards) {
    s = s.replace(g, (m) => {
      slots.push(m);
      return `\x00${slots.length - 1}\x00`;
    });
  }
  return { s, slots };
}

function unguard(s, slots) {
  return s.replace(/\x00(\d+)\x00/g, (_, idx) => slots[Number(idx)]);
}

function isNegationExampleLine(line) {
  return /^\s*-\s*❌/.test(line);
}

function replaceInLine(line) {
  if (isNegationExampleLine(line)) {
    return line;
  }

  if (IS_GALAXY && isYamlMetadataLine(line)) {
    return line;
  }

  const { s, slots } = guardProtectedSpans(line);
  if (!MATCH_RE.test(s)) {
    MATCH_RE.lastIndex = 0;
    return line;
  }
  MATCH_RE.lastIndex = 0;

  const replaced = IS_GALAXY
    ? s.replace(/\b(GALAXY|galaxy)\b/g, TARGET)
    : s.replace(IS_P300C ? /\bP300[cC]\b/g : new RegExp(`\\b${UPPER}\\b`, 'g'), TARGET);

  if (replaced === s) {
    return line;
  }
  return unguard(replaced, slots);
}

function shouldProcessLine(line, inFence, fenceLang) {
  if (isYamlMetadataLine(line) && !inFence) {
    return false;
  }
  if (!inFence) {
    return true;
  }
  if (fenceLang === 'mermaid') {
    return true;
  }
  if (EXEC_FENCE_LANGS.has(fenceLang)) {
    return true;
  }
  return true;
}

function countMatches(text) {
  MATCH_RE.lastIndex = 0;
  return (text.match(MATCH_RE) || []).length;
}

function transformContent(text) {
  const lines = text.split('\n');
  const hasYamlFrontmatter = lines.length > 0 && lines[0] === '---';
  let inFrontmatter = false;
  let frontmatterDone = !hasYamlFrontmatter;
  let inFence = false;
  let fenceLang = '';
  let changed = false;

  const out = lines.map((line) => {
    if (!frontmatterDone) {
      if (line === '---') {
        if (!inFrontmatter) {
          inFrontmatter = true;
          return line;
        }
        inFrontmatter = false;
        frontmatterDone = true;
        return line;
      }
      if (inFrontmatter) {
        return line;
      }
    }

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

    if (!shouldProcessLine(line, inFence, fenceLang)) {
      return line;
    }

    const inExecFence = inFence && EXEC_FENCE_LANGS.has(fenceLang);
    if (inExecFence && fenceLang !== 'mermaid') {
      const next = replaceInLine(line);
      if (next !== line) {
        changed = true;
      }
      return next;
    }

    if (inFence || !inFence) {
      const next = replaceInLine(line);
      if (next !== line) {
        changed = true;
      }
      return next;
    }

    return line;
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
    touched++;
    const rel = path.relative(ROOT, abs);
    if (checkOnly) {
      console.log(`  ${rel} (${before - after} would change)`);
    } else {
      fs.writeFileSync(abs, text, 'utf8');
      console.log(`  ${rel} (${before - after} replacements)`);
    }
  }

  const label = IS_GALAXY ? 'galaxy/GALAXY' : (IS_P300C ? 'P300c/P300C' : UPPER);
  if (checkOnly) {
    if (touched > 0) {
      console.error(`\n${touched} file(s) still need ${label} → ${TARGET} updates.`);
      process.exit(1);
    }
    console.log(`OK: ${label} normalized to ${TARGET} in copy paths.`);
    process.exit(0);
  }

  console.log(`\nDone: ${touched} files, ~${replacements} ${label} → ${TARGET} in prose.\n`);
}

main();
