#!/usr/bin/env node
/**
 * One-shot reconciliation for Task 1 of the tt-cli-leads-the-way plan (2026-10-29).
 * Kept so the commit is reproducible. Makes package.json, extension.ts, the lesson
 * registry and TERMINAL_COMMANDS agree with each other:
 *   - declares registered-but-undeclared ids that something uses (lesson link, src ref,
 *     completion event) in package.json;
 *   - deletes registrations + handlers nothing uses (only when the function has no
 *     other reference in extension.ts);
 *   - drops package.json entries with no registration;
 *   - drops completionEvents naming commands that do not exist;
 *   - deletes TERMINAL_COMMANDS entries no handler reads.
 */
'use strict';
const fs = require('fs');
const { buildCommandInventory } = require('./lib/command-inventory');

const KEEP_UNDECLARED = new Set([
  'tenstorrent.showTelemetryDetails', // status-bar item command (object property, not a link)
  'tenstorrent.ttsim.setupQemu', // release-gated by design (docs/superpowers/plans/2026-06-16-ttsim-qemu-bridge.md)
]);

const inv = buildCommandInventory('.');
const pkgPath = 'package.json';
const pkg = JSON.parse(fs.readFileSync(pkgPath, 'utf8'));
const pkgIds = new Set(inv.packageIds);
const reg = new Set(inv.registeredIds);
const used = new Set([
  ...inv.contentLinks.map((l) => l.id),
  ...inv.srcRefs.map((r) => r.id),
  ...inv.completionEvents.map((e) => e.id),
  ...KEEP_UNDECLARED,
]);

const humanize = (id) =>
  id
    .replace(/^tenstorrent\./, '')
    .split('.')
    .pop()
    .replace(/([A-Z])/g, ' $1')
    .replace(/^./, (c) => c.toUpperCase())
    .trim();

// 1. package.json
const toDeclare = [...reg].filter((id) => !pkgIds.has(id) && used.has(id));
for (const id of toDeclare) pkg.contributes.commands.push({ command: id, title: humanize(id), category: 'Tenstorrent' });
const toUndeclare = [...pkgIds].filter((id) => !reg.has(id));
pkg.contributes.commands = pkg.contributes.commands.filter((c) => !toUndeclare.includes(c.command));
fs.writeFileSync(pkgPath, JSON.stringify(pkg, null, 2) + '\n');

// 2. extension.ts: delete unused registrations and their handlers
let ext = fs.readFileSync('src/extension.ts', 'utf8');
const toDelete = [...reg].filter((id) => !pkgIds.has(id) && !used.has(id));
const deleted = [];
const skipped = [];
for (const id of toDelete) {
  const regRe = new RegExp(`^\\s*vscode\\.commands\\.registerCommand\\('${id.replace(/\./g, '\\.')}',\\s*(\\w+)\\),?\\n`, 'm');
  const m = ext.match(regRe);
  if (!m) { skipped.push(`${id} (no literal registration)`); continue; }
  const fn = m[1];
  const refs = (ext.match(new RegExp(`\\b${fn}\\b`, 'g')) || []).length; // registration + definition = 2
  ext = ext.replace(regRe, '');
  if (refs > 2) { skipped.push(`${id} (registration removed; ${fn} kept, called elsewhere)`); continue; }
  // handler: optional leading /** ... */ then `function fn(...): T {` through the matching top-level `}`
  const fnRe = new RegExp(`(?:/\\*\\*[\\s\\S]*?\\*/\\n)?(?:async )?function ${fn}\\([^)]*\\)[^{]*\\{`, 'm');
  const fm = ext.match(fnRe);
  if (!fm) { skipped.push(`${id} (handler ${fn} not found)`); continue; }
  let i = fm.index + fm[0].length, depth = 1;
  while (depth > 0 && i < ext.length) { if (ext[i] === '{') depth++; else if (ext[i] === '}') depth--; i++; }
  ext = ext.slice(0, fm.index) + ext.slice(i).replace(/^\n+/, '\n');
  deleted.push(`${id} (${fn})`);
}
fs.writeFileSync('src/extension.ts', ext);

// 3. registry: drop completionEvents naming nonexistent commands
const regPath = 'content/lesson-registry.json';
const registry = JSON.parse(fs.readFileSync(regPath, 'utf8'));
const droppedEvents = [];
for (const l of registry.lessons) {
  if (!l.completionEvents) continue;
  l.completionEvents = l.completionEvents.filter((ev) => {
    const id = ev.replace(/^onCommand:/, '');
    const ok = reg.has(id);
    if (!ok) droppedEvents.push(`${l.id}:${ev}`);
    return ok;
  });
}
fs.writeFileSync(regPath, JSON.stringify(registry, null, 2) + '\n');

// 4. TERMINAL_COMMANDS: delete unused template blocks
let term = fs.readFileSync('src/commands/terminalCommands.ts', 'utf8');
const uses = new Set(inv.templateUses);
const unusedKeys = inv.templateKeys.filter((k) => !uses.has(k));
for (const k of unusedKeys) {
  term = term.replace(new RegExp(`\\n  ${k}: \\{[\\s\\S]*?\\n  \\},?\\n`), '\n');
}
fs.writeFileSync('src/commands/terminalCommands.ts', term);

console.log('declared:', toDeclare.join(' '));
console.log('undeclared:', toUndeclare.join(' '));
console.log('deleted handlers:', deleted.join(' '));
console.log('skipped:', skipped.join(' ') || '(none)');
console.log('dropped events:', droppedEvents.join(' '));
console.log('deleted templates:', unusedKeys.join(' '));
