'use strict';

/*
 * ────────────────────────────────────────────────────────────────────────────
 *  Command-map parser (extracted from build-web.js)
 * ────────────────────────────────────────────────────────────────────────────
 *
 *  The docs-site generator needs the actual terminal-command text behind each
 *  `command:tenstorrent.*` link so it can render a runnable snippet. Rather than
 *  importing the extension's TypeScript at build time, we parse the two source
 *  files with regexes.
 *
 *  This logic lives in its own module (as a PURE function of two source strings)
 *  so the issue #42 fix — the function-boundary "bleed-through" guard — can be
 *  regression-tested with fixtures, without executing the full site build.
 *  See test/lesson-tests/command-map-parser.test.ts.
 *
 *  @param {string} termSrc  Contents of src/commands/terminalCommands.ts
 *  @param {string} [extSrc] Contents of src/extension.ts (optional)
 *  @returns {Object<string,string>} map of KEY / "__ext__<suffix>" → command text
 */
function buildCommandMap(termSrc, extSrc) {
  const map = {};

  // terminalCommands.ts uses two quoting styles for template values:
  //
  //   Simple single-line commands:   template: 'pip install flask',
  //   Multi-line / interpolated:     template: `cd ~/code && ...`,
  //
  // We walk line by line, tracking the current KEY name, then extract
  // the template value regardless of which quote style is used.
  let currentKey = null;
  // Keys can contain digits (e.g. START_TT_INFERENCE_SERVER_N150, DOWNLOAD_WAN22_MODEL).
  const keyRe = /^\s{2}([A-Z][A-Z0-9_]*):\s*\{/;
  const lines = termSrc.split('\n');

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];

    // Detect a new top-level KEY: { block
    const keyMatch = line.match(keyRe);
    if (keyMatch) {
      currentKey = keyMatch[1];
    }

    if (!currentKey) continue;

    // ---- single-quoted single-line:  template: 'text',
    const sqMatch = line.match(/template:\s*'([^']+)'/);
    if (sqMatch) {
      map[currentKey] = sqMatch[1];
      currentKey = null;
      continue;
    }

    // ---- double-quoted single-line:  template: "text",
    const dqMatch = line.match(/template:\s*"([^"]+)"/);
    if (dqMatch) {
      map[currentKey] = dqMatch[1];
      currentKey = null;
      continue;
    }

    // ---- backtick single-line:  template: `text`,
    const btMatch = line.match(/template:\s*`([^`]+)`/);
    if (btMatch) {
      map[currentKey] = btMatch[1];
      currentKey = null;
      continue;
    }

    // ---- backtick multi-line:  template: `first line
    //                                        ...
    //                                        last line`,
    const btOpen = line.indexOf('template: `');
    if (btOpen !== -1) {
      let text = line.slice(btOpen + 'template: `'.length);
      while (!text.includes('`') && i < lines.length - 1) {
        i++;
        text += '\n' + lines[i];
      }
      map[currentKey] = text.slice(0, text.lastIndexOf('`'));
      currentKey = null;
    }
  }

  // --- Second pass: parse extension.ts to build a direct camelCase suffix →
  //     template mapping for commands that don't follow the simple
  //     camelToUpperSnake naming convention.
  //
  //     Pattern in extension.ts:
  //       async function createApiServerDirect() {
  //         ...
  //         const command = TERMINAL_COMMANDS.CREATE_API_SERVER.template;
  //         ...
  //       }
  //       ...
  //       registerCommand('tenstorrent.createApiServerDirect', createApiServerDirect)
  //
  //     We build:
  //       funcName → TERMINAL_COMMANDS.KEY  (from function bodies)
  //       commandSuffix → funcName           (from registerCommand calls)
  //     Then join: commandSuffix → template
  if (extSrc) {
    // Map: funcName → TERMINAL_COMMANDS key (first occurrence wins).
    // The body scan is tempered to stop at the next top-level function declaration —
    // `function`, `async function`, or `export [async] function` (e.g. `export async
    // function activate(...)`) — so a function with NO TERMINAL_COMMANDS reference (e.g. a
    // file-opener like openRiscvKernel) can't "steal" a later function's key. That
    // bleed-through was rendering a game-of-life command under CS-Fundamentals'
    // "Open Kernel Source" (issue #42).
    // Keys can contain digits (START_TT_INFERENCE_SERVER_N150, RUN_ANIMATEDIFF_2FRAME, …).
    const funcToKey = {};
    const funcKeyRe = /(?:export\s+)?(?:async\s+)?function\s+(\w+)[^{]*\{(?:(?!\n(?:export )?(?:async )?function )[\s\S])*?TERMINAL_COMMANDS\.([A-Z0-9_]+)\./g;
    let fkMatch;
    while ((fkMatch = funcKeyRe.exec(extSrc)) !== null) {
      const funcName = fkMatch[1];
      const tcKey    = fkMatch[2];
      if (!funcToKey[funcName] && map[tcKey]) {
        funcToKey[funcName] = tcKey;
      }
    }

    // Map: tenstorrent.commandSuffix → funcName (from registerCommand calls)
    const regRe = /registerCommand\(['"]tenstorrent\.(\w+)['"]\s*,\s*(\w+)/g;
    let regMatch;
    while ((regMatch = regRe.exec(extSrc)) !== null) {
      const suffix   = regMatch[1];
      const funcName = regMatch[2];
      if (funcToKey[funcName]) {
        // Store as suffix → template under a special prefix so commandTextForId
        // can find it directly without the camelToUpperSnake conversion.
        map['__ext__' + suffix] = map[funcToKey[funcName]];
      }
    }
  }

  return map;
}

module.exports = { buildCommandMap };
