#!/usr/bin/env node
/**
 * Local dev server for the GitHub Pages site with auto-rebuild on save.
 *
 * Usage:
 *   npm run dev:web
 *   npm run dev:web -- --port=3000
 *   WEB_DEV_PORT=3000 npm run dev:web
 *
 * Opens:
 *   http://127.0.0.1:8000/              — install landing page
 *   http://127.0.0.1:8000/lessons/      — lesson catalog
 *   http://127.0.0.1:8000/lessons/<id>/ — individual lesson
 */

'use strict';

const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SITE_DIR = path.join(ROOT, 'site');
const NODE = process.execPath;

const portArg = process.argv.find((a) => a.startsWith('--port='));
const PORT = process.env.WEB_DEV_PORT || (portArg ? portArg.split('=')[1] : '8000');

/** Paths (relative to ROOT) that should trigger a rebuild when changed. */
const REBUILD_PREFIXES = [
  'content/lessons/',
  'content/pages/',
  'content/web/',
  'content/lesson-registry.json',
  'src/webview/',
  'src/commands/terminalCommands.ts',
  'src/extension.ts',
  'assets/img/',
  'scripts/build-web.js',
];

let building = false;
let pending = false;
let debounceTimer = null;
let serverProc = null;

function shouldRebuild(relPath) {
  const rel = relPath.replace(/\\/g, '/');
  return REBUILD_PREFIXES.some(
    (p) => rel === p || rel.startsWith(p) || (p.endsWith('.ts') && rel === p),
  );
}

function runBuild() {
  return new Promise((resolve, reject) => {
    const child = spawn(NODE, ['scripts/build-web.js'], {
      cwd: ROOT,
      stdio: 'inherit',
      env: { ...process.env, SITE_BASE_PATH: '' },
    });
    child.on('error', (err) => {
      reject(new Error(`Failed to run build: ${err.message}`));
    });
    child.on('exit', (code) => {
      if (code === 0) resolve();
      else reject(new Error(`build-web.js exited with code ${code}`));
    });
  });
}

async function rebuild(label) {
  if (building) {
    pending = true;
    return;
  }
  building = true;
  console.log(`\n[dev:web] Rebuilding (${label})…`);
  try {
    await runBuild();
    console.log('[dev:web] Rebuild done — refresh your browser\n');
  } catch (err) {
    console.error(`[dev:web] Build failed: ${err.message}`);
  } finally {
    building = false;
    if (pending) {
      pending = false;
      void rebuild('queued changes');
    }
  }
}

function scheduleRebuild(label) {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => void rebuild(label), 350);
}

function watchTree(rel) {
  const abs = path.join(ROOT, rel);
  if (!fs.existsSync(abs)) return;

  const stat = fs.statSync(abs);
  const opts = stat.isDirectory() ? { recursive: true } : {};

  try {
    fs.watch(abs, opts, (_event, filename) => {
      const changed = filename
        ? path.join(rel, filename).replace(/\\/g, '/')
        : rel;
      if (shouldRebuild(changed)) {
        scheduleRebuild(changed);
      }
    });
  } catch (err) {
    // macOS can throw EPERM for some paths; fall back to parent watch
    console.warn(`[dev:web] Could not watch ${rel}: ${err.message}`);
  }
}

function startServer() {
  if (!fs.existsSync(SITE_DIR)) {
    console.error(`[dev:web] Build did not create ${SITE_DIR}`);
    process.exit(1);
  }

  serverProc = spawn(NODE, ['scripts/serve-dev.js', String(PORT)], {
    cwd: ROOT,
    stdio: 'inherit',
  });

  serverProc.on('error', (err) => {
    console.error(`[dev:web] Server failed: ${err.message}`);
    process.exit(1);
  });

  serverProc.on('exit', (code, signal) => {
    if (signal === 'SIGTERM' || signal === 'SIGINT') return;
    if (code !== 0 && code !== null) process.exit(code);
  });
}

function shutdown() {
  if (serverProc && !serverProc.killed) {
    serverProc.kill('SIGTERM');
  }
  process.exit(0);
}

async function main() {
  console.log('[dev:web] Initial build…');
  try {
    await runBuild();
  } catch (err) {
    console.error(`[dev:web] ${err.message}`);
    console.error('  Fix build errors above, or run: npm run build:web');
    process.exit(1);
  }

  startServer();

  // Watch content + webview sources (not site/ — avoids rebuild loops)
  watchTree('content');
  watchTree('src/webview');
  watchTree('src/commands/terminalCommands.ts');
  watchTree('src/extension.ts');
  watchTree('assets/img');
  watchTree('scripts/build-web.js');

  console.log('\n[dev:web] Ready — save a lesson or page, wait for rebuild, refresh browser');
  console.log(`  Install:  http://127.0.0.1:${PORT}/`);
  console.log(`  Lessons:  http://127.0.0.1:${PORT}/lessons/<lesson-id>/`);
  console.log(`  Example:  http://127.0.0.1:${PORT}/lessons/animatediff-video-generation/`);
  console.log('  Ctrl-C to stop.\n');

  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('[dev:web]', err.message);
  process.exit(1);
});
