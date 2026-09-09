/**
 * Command-Map Parser Tests
 *
 * Regression coverage for scripts/lib/command-map-parser.js — the docs-site
 * generator's parser that maps `command:tenstorrent.*` links to their terminal
 * command text.
 *
 * The headline case is issue #42: a function with NO `TERMINAL_COMMANDS`
 * reference (e.g. a file-opener) must not "bleed through" and steal the key of
 * the NEXT function during the body scan. Before the fix, that rendered a
 * game-of-life command under CS-Fundamentals' "Open Kernel Source" button.
 *
 * These are pure-function tests over fixture strings — no VSCode, no disk, no
 * full site build — so they run under plain `npm test`.
 */

import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

// Pure JS module (CommonJS) — required at runtime, no TS compilation needed.
const { buildCommandMap } = require('../../scripts/lib/command-map-parser');

describe('command-map parser', () => {
  describe('terminalCommands.ts template extraction', () => {
    it('extracts single-quoted, double-quoted, and backtick templates', () => {
      const term = [
        'export const TERMINAL_COMMANDS = {',
        '  SINGLE_QUOTED: {',
        "    template: 'tt-smi',",
        '  },',
        '  DOUBLE_QUOTED: {',
        '    template: "podman info",',
        '  },',
        '  BACKTICK_INLINE: {',
        '    template: `cd ~ && ./install.sh`,',
        '  },',
        '};',
      ].join('\n');

      const map = buildCommandMap(term);
      expect(map.SINGLE_QUOTED).to.equal('tt-smi');
      expect(map.DOUBLE_QUOTED).to.equal('podman info');
      expect(map.BACKTICK_INLINE).to.equal('cd ~ && ./install.sh');
    });

    it('decodes escaped quotes and escaped newlines in quoted templates', () => {
      const term = [
        'export const TERMINAL_COMMANDS = {',
        '  ESCAPED: {',
        "    template: 'tt-metalium -c \"python3 -c \\'import ttnn; print(getattr(ttnn, \\\\\"__version__\\\\\", \\\\\"import OK\\\\\"))\\'\"\\nsecond line',",
        '  },',
        '};',
      ].join('\n');

      const map = buildCommandMap(term);
      expect(map.ESCAPED).to.equal(
        'tt-metalium -c "python3 -c \'import ttnn; print(getattr(ttnn, \\"__version__\\", \\"import OK\\"))\'"\nsecond line'
      );
    });

    it('extracts multi-line backtick templates', () => {
      const term = [
        'export const TERMINAL_COMMANDS = {',
        '  MULTILINE: {',
        '    template: `line one',
        'line two',
        'line three`,',
        '  },',
        '};',
      ].join('\n');

      const map = buildCommandMap(term);
      expect(map.MULTILINE).to.equal('line one\nline two\nline three');
    });

    it('handles keys containing digits', () => {
      const term = [
        'export const TERMINAL_COMMANDS = {',
        '  START_SERVER_N150: {',
        "    template: 'start n150',",
        '  },',
        '  RUN_ANIMATEDIFF_2FRAME: {',
        "    template: 'run 2frame',",
        '  },',
        '};',
      ].join('\n');

      const map = buildCommandMap(term);
      expect(map.START_SERVER_N150).to.equal('start n150');
      expect(map.RUN_ANIMATEDIFF_2FRAME).to.equal('run 2frame');
    });
  });

  describe('extension.ts function → command resolution (issue #42)', () => {
    const term = [
      'export const TERMINAL_COMMANDS = {',
      '  GAME_OF_LIFE: {',
      "    template: 'python3 game_of_life.py',",
      '  },',
      '};',
    ].join('\n');

    it('maps a command suffix to the template used in its handler', () => {
      const ext = [
        'function runGameOfLife() {',
        '  const command = TERMINAL_COMMANDS.GAME_OF_LIFE.template;',
        '  runInTerminal(command);',
        '}',
        "registerCommand('tenstorrent.runGameOfLife', runGameOfLife);",
      ].join('\n');

      const map = buildCommandMap(term, ext);
      expect(map['__ext__runGameOfLife']).to.equal('python3 game_of_life.py');
    });

    it('does NOT let a keyless function steal the next function\'s key', () => {
      // openKernelSource has no TERMINAL_COMMANDS reference. Before the boundary
      // guard, its body scan bled into runGameOfLife and grabbed GAME_OF_LIFE.
      const ext = [
        'function openKernelSource() {',
        "  const uri = vscode.Uri.file('/path/to/kernel.cpp');",
        '  vscode.window.showTextDocument(uri);',
        '}',
        'function runGameOfLife() {',
        '  const command = TERMINAL_COMMANDS.GAME_OF_LIFE.template;',
        '  runInTerminal(command);',
        '}',
        "registerCommand('tenstorrent.openKernelSource', openKernelSource);",
        "registerCommand('tenstorrent.runGameOfLife', runGameOfLife);",
      ].join('\n');

      const map = buildCommandMap(term, ext);
      // The regression: the file-opener must NOT resolve to a command.
      expect(map['__ext__openKernelSource']).to.be.undefined;
      // ...and the real handler still resolves correctly.
      expect(map['__ext__runGameOfLife']).to.equal('python3 game_of_life.py');
    });

    it('respects `export async function` boundaries too', () => {
      const ext = [
        'export async function openKernelSource() {',
        "  await vscode.window.showTextDocument(vscode.Uri.file('/k.cpp'));",
        '}',
        'export async function runGameOfLife() {',
        '  const command = TERMINAL_COMMANDS.GAME_OF_LIFE.template;',
        '  runInTerminal(command);',
        '}',
        "registerCommand('tenstorrent.openKernelSource', openKernelSource);",
        "registerCommand('tenstorrent.runGameOfLife', runGameOfLife);",
      ].join('\n');

      const map = buildCommandMap(term, ext);
      expect(map['__ext__openKernelSource']).to.be.undefined;
      expect(map['__ext__runGameOfLife']).to.equal('python3 game_of_life.py');
    });

    it('omits the extension pass when no extension source is provided', () => {
      const map = buildCommandMap(term);
      expect(map.GAME_OF_LIFE).to.equal('python3 game_of_life.py');
      expect(Object.keys(map).some((k) => k.startsWith('__ext__'))).to.equal(false);
    });

    it('resolves a template whose value starts on the line after `template:`', () => {
      // Regression: reformatting a template onto two lines (`template:\n
      // '...'`) once silently dropped it from the map -- and therefore from
      // the published site, as an inert "unknown command" badge with no
      // build error -- because the old parser only matched `template:` and
      // its value on the same physical line.
      const term = [
        'export const TERMINAL_COMMANDS = {',
        '  TWO_LINE: {',
        '    template:',
        "      'cd ~ && echo hi',",
        '  },',
        '};',
      ].join('\n');

      const map = buildCommandMap(term);
      expect(map.TWO_LINE).to.equal('cd ~ && echo hi');
    });
  });

  describe('every real TERMINAL_COMMANDS key resolves', () => {
    // Canary against the exact class of regression above recurring silently:
    // parse the actual source files and assert every key the TS source
    // defines comes back with a real (non-undefined) command string. A
    // future template reformatted in a way this parser can't follow would
    // fail this test immediately, instead of only showing up as a broken
    // button on the published site.
    const termSrc = fs.readFileSync(
      path.join(__dirname, '../../src/commands/terminalCommands.ts'),
      'utf8'
    );
    const extSrc = fs.readFileSync(path.join(__dirname, '../../src/extension.ts'), 'utf8');
    const map = buildCommandMap(termSrc, extSrc);

    // Same key pattern the parser itself uses to detect a `KEY: {` block.
    const keyRe = /^\s{2}([A-Z][A-Z0-9_]*):\s*\{/gm;
    const keys: string[] = [];
    let m: RegExpExecArray | null;
    while ((m = keyRe.exec(termSrc)) !== null) {
      keys.push(m[1]);
    }

    it('finds at least one TERMINAL_COMMANDS key (sanity-check the regex itself)', () => {
      expect(keys.length).to.be.greaterThan(0);
    });

    // Keys whose `template` is a computed expression (e.g. an IIFE calling
    // getDefaultModel()) rather than a string literal -- this parser
    // deliberately never evaluates source-derived code (see
    // unescapeStringLiteral's docstring), so these can't resolve statically.
    // They render as an "unknown command" badge on the published site today;
    // that's a separate, known limitation, not the line-break regression
    // this test guards against. Update this list if one is ever converted
    // to a plain string template, or leave it here if a new computed one is
    // intentionally added.
    const KNOWN_UNRESOLVABLE = new Set(['DOWNLOAD_MODEL']);

    it('resolves every key to a non-empty command string', () => {
      const unresolved = keys.filter((k) => !map[k] && !KNOWN_UNRESOLVABLE.has(k));
      expect(unresolved, `unresolved TERMINAL_COMMANDS keys: ${unresolved.join(', ')}`).to.deep
        .equal([]);
    });
  });
});
