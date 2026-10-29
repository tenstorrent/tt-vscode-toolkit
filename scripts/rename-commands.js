#!/usr/bin/env node
/**
 * One-shot command rename (Task 2 of docs/superpowers/plans/2026-10-29-tt-cli-leads-the-way.md).
 * Reads scripts/command-rename-map.json and rewrites every whole-token occurrence of each
 * old id in package.json, src/**, content/** and the registry. Exits non-zero if any old id
 * survives. Longest ids are replaced first so a prefix can never clobber a longer id.
 */
'use strict';
const fs = require('fs');
const path = require('path');

const map = JSON.parse(fs.readFileSync(path.join(__dirname, 'command-rename-map.json'), 'utf8'));
delete map._comment;
const pairs = Object.entries(map).sort((a, b) => b[0].length - a[0].length);

function walk(dir, exts, skip = []) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (skip.some((s) => p.includes(s))) continue;
    if (e.isDirectory()) out.push(...walk(p, exts, skip));
    else if (exts.some((x) => p.endsWith(x))) out.push(p);
  }
  return out;
}

const files = [
  'package.json',
  ...walk('src', ['.ts', '.js', '.html', '.css'], ['.bak']),
  ...walk('content', ['.md', '.html', '.json']),
];

const escape = (s) => s.replace(/\./g, '\\.');
let total = 0;
for (const f of files) {
  const before = fs.readFileSync(f, 'utf8');
  let after = before;
  let n = 0;
  for (const [oldId, newId] of pairs) {
    after = after.replace(new RegExp(`(?<![A-Za-z0-9_.])${escape(oldId)}(?![A-Za-z0-9_.])`, 'g'), () => {
      n++;
      return newId;
    });
  }
  if (n) {
    fs.writeFileSync(f, after);
    console.log(`${n}\t${f}`);
    total += n;
  }
}
console.log(`replaced ${total} occurrences in ${files.length} files scanned`);

// Leftover check: any tenstorrent.<flat> id (no namespace) still referenced?
const flat = /tenstorrent\.[a-z][A-Za-z0-9]*(?![A-Za-z0-9_.])/g;
const allowed = new Set(['tenstorrent.com', 'tenstorrent.github', 'tenstorrent.ai']);
let bad = 0;
for (const f of files) {
  const text = fs.readFileSync(f, 'utf8');
  for (const m of text.matchAll(flat)) {
    if (allowed.has(m[0])) continue;
    // Only flag command ids: `registerCommand('…`, `executeCommand('…`, `command:…`,
    // `onCommand:…`, or a package.json "command": entry. Config and state keys
    // (`tenstorrent.simulatorApiUrl`, `tenstorrent.progress`) share the prefix and are not ids.
    const pre = text.slice(Math.max(0, m.index - 40), m.index);
    // executeCommand is deliberately not checked here: a call to an unregistered id is a
    // pre-existing dead call (ProgressTracker's 'View Progress'), not a rename leftover.
    if (/registerCommand\(\s*['"`]$|command:$|"command":\s*"$/.test(pre)) {
      console.error(`LEFTOVER ${f}: ${m[0]}`);
      bad++;
    }
  }
}
if (bad) process.exit(1);
