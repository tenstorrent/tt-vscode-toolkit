/**
 * Command inventory — one place that knows where command ids live.
 *
 * A lesson button is `[text](command:tenstorrent.x)`; the id must be registered in
 * src/extension.ts (or src/commands/*.ts) AND declared in package.json
 * `contributes.commands`, or the button is dead / undeclared. This module reads all
 * of those sources so a test can assert they agree. Pure file reads, no VS Code.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const ID = 'tenstorrent\\.[A-Za-z0-9.]+';

function walk(dir, exts) {
  const out = [];
  for (const e of fs.readdirSync(dir, { withFileTypes: true })) {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) out.push(...walk(p, exts));
    else if (exts.some((x) => p.endsWith(x))) out.push(p);
  }
  return out;
}

function matchAll(text, re) {
  return [...text.matchAll(re)].map((m) => m[1]);
}

/**
 * @param {string} repoRoot
 * @returns {{packageIds: string[], registeredIds: string[],
 *   contentLinks: {id: string, file: string}[],
 *   completionEvents: {id: string, lessonId: string}[],
 *   srcRefs: {id: string, file: string}[],
 *   templateKeys: string[], templateUses: string[]}}
 */
function buildCommandInventory(repoRoot) {
  const pkg = JSON.parse(fs.readFileSync(path.join(repoRoot, 'package.json'), 'utf8'));
  const packageIds = pkg.contributes.commands.map((c) => c.command);

  const srcFiles = walk(path.join(repoRoot, 'src'), ['.ts', '.js', '.html']).filter(
    (f) => !/\.bak\d*$/.test(f) && !f.includes('.bak')
  );
  const registeredIds = [];
  const srcRefs = [];
  for (const f of srcFiles) {
    const text = fs.readFileSync(f, 'utf8');
    registeredIds.push(
      ...matchAll(text, new RegExp(`registerCommand\\(\\s*['"\`](${ID})['"\`]`, 'g'))
    );
    for (const id of matchAll(text, new RegExp(`executeCommand\\(\\s*['"\`](${ID})['"\`]`, 'g'))) {
      srcRefs.push({ id, file: path.relative(repoRoot, f) });
    }
    // Webview/HTML strings that name a command (status bar, menus, welcome page).
    for (const id of matchAll(text, new RegExp(`command:(${ID})`, 'g'))) {
      srcRefs.push({ id, file: path.relative(repoRoot, f) });
    }
  }

  const contentLinks = [];
  for (const f of walk(path.join(repoRoot, 'content'), ['.md', '.html'])) {
    const text = fs.readFileSync(f, 'utf8');
    for (const id of matchAll(text, new RegExp(`command:(${ID})`, 'g'))) {
      contentLinks.push({ id, file: path.relative(repoRoot, f) });
    }
  }

  const registry = JSON.parse(
    fs.readFileSync(path.join(repoRoot, 'content', 'lesson-registry.json'), 'utf8')
  );
  const completionEvents = [];
  for (const lesson of registry.lessons) {
    for (const ev of lesson.completionEvents || []) {
      const m = ev.match(new RegExp(`^onCommand:(${ID})$`));
      if (m) completionEvents.push({ id: m[1], lessonId: lesson.id });
    }
  }

  const termText = fs.readFileSync(path.join(repoRoot, 'src/commands/terminalCommands.ts'), 'utf8');
  const block = termText.slice(termText.indexOf('export const TERMINAL_COMMANDS'));
  const templateKeys = matchAll(block, /^  ([A-Z][A-Z0-9_]+): \{/gm);
  const templateUses = [];
  for (const f of srcFiles) {
    const text = fs.readFileSync(f, 'utf8');
    templateUses.push(...matchAll(text, /TERMINAL_COMMANDS\.([A-Z][A-Z0-9_]+)/g));
    templateUses.push(...matchAll(text, /TERMINAL_COMMANDS\[['"]([A-Z][A-Z0-9_]+)['"]\]/g));
    templateUses.push(...matchAll(text, /(?:runTerminalCommand|getDisplayCommand|executeCommand)\(\s*['"]([A-Z][A-Z0-9_]+)['"]/g));
  }

  return {
    packageIds,
    registeredIds: [...new Set(registeredIds)],
    contentLinks,
    completionEvents,
    srcRefs,
    templateKeys,
    templateUses: [...new Set(templateUses)],
  };
}

module.exports = { buildCommandInventory };
