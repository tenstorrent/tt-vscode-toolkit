/**
 * Command-link resolver tests.
 *
 * A dead button is a `command:` link whose id nobody registers; an undeclared command
 * is one registered in code but missing from package.json. Both are silent in the
 * editor and only visible to a reader who presses the button. These tests make them
 * fail `npm test` instead. Pure file reads — no VSCode, no build.
 */
import { expect } from 'chai';
import * as path from 'path';

// CommonJS module shared with the docs-site scripts.
const { buildCommandInventory } = require('../../scripts/lib/command-inventory');

const REPO = path.join(__dirname, '../..');

describe('command links', () => {
  const inv = buildCommandInventory(REPO);
  const pkg = new Set<string>(inv.packageIds);
  const reg = new Set<string>(inv.registeredIds);

  it('every command: link in content/ resolves to contributes.commands', () => {
    const missing = [...new Set(inv.contentLinks.filter((l: any) => !pkg.has(l.id)).map((l: any) => `${l.id} (${l.file})`))];
    expect(missing).to.deep.equal([]);
  });

  it('every onCommand: completionEvent resolves to contributes.commands', () => {
    const missing = inv.completionEvents.filter((e: any) => !pkg.has(e.id)).map((e: any) => `${e.id} (${e.lessonId})`);
    expect(missing).to.deep.equal([]);
  });

  it('every registered id is declared in package.json', () => {
    expect([...reg].filter((id) => !pkg.has(id)).sort()).to.deep.equal([]);
  });

  it('every package.json command is registered in code', () => {
    expect([...pkg].filter((id) => !reg.has(id)).sort()).to.deep.equal([]);
  });

  it('every TERMINAL_COMMANDS key is used by a handler', () => {
    const uses = new Set<string>(inv.templateUses);
    expect(inv.templateKeys.filter((k: string) => !uses.has(k))).to.deep.equal([]);
  });
});
