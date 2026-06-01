#!/usr/bin/env node
/**
 * Dev HTTP server for the GH Pages site/ directory (Node — no Python required).
 *
 * Adds COOP/COEP headers for SharedArrayBuffer (Pyodide threading in sim lessons).
 *
 * Usage:
 *   node scripts/serve-dev.js [port]   # default 8000
 */

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const SITE_DIR = path.join(ROOT, 'site');
const PORT = parseInt(process.argv[2] || process.env.WEB_DEV_PORT || '8000', 10);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.gif': 'image/gif',
  '.webp': 'image/webp',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.woff': 'font/woff',
  '.woff2': 'font/woff2',
  '.ttf': 'font/ttf',
  '.mp4': 'video/mp4',
  '.webm': 'video/webm',
  '.vsix': 'application/octet-stream',
};

function safePath(urlPath) {
  const decoded = decodeURIComponent(urlPath.split('?')[0]);
  const normalized = path.normalize(decoded).replace(/^(\.\.[/\\])+/, '');
  const filePath = path.join(SITE_DIR, normalized);
  const rel = path.relative(SITE_DIR, filePath);
  if (rel.startsWith('..') || path.isAbsolute(rel)) return null;
  return filePath;
}

function resolveFile(urlPath) {
  let filePath = safePath(urlPath);
  if (!filePath) return null;

  if (urlPath.endsWith('/')) {
    filePath = path.join(filePath, 'index.html');
  }

  if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
    return filePath;
  }

  // /lessons/foo → /lessons/foo/index.html
  if (!urlPath.endsWith('/')) {
    const withIndex = path.join(filePath, 'index.html');
    if (fs.existsSync(withIndex) && fs.statSync(withIndex).isFile()) {
      return withIndex;
    }
  }

  return null;
}

function sendFile(res, filePath, extraHeaders = {}) {
  const ext = path.extname(filePath).toLowerCase();
  const type = MIME[ext] || 'application/octet-stream';
  res.writeHead(200, { 'Content-Type': type, ...extraHeaders });
  fs.createReadStream(filePath).pipe(res);
}

/** Only Pyodide sim lessons need cross-origin isolation (SharedArrayBuffer). */
function needsCrossOriginIsolation(urlPath) {
  const p = (urlPath || '').split('?')[0];
  return (
    /^\/lessons\/tt-lang-intro(\/|$)/.test(p) ||
    p.startsWith('/assets/playground/')
  );
}

const server = http.createServer((req, res) => {
  const urlPath = req.url === '/' ? '/index.html' : req.url;
  const isolated = needsCrossOriginIsolation(urlPath);
  const extraHeaders = isolated
    ? {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
        'Cross-Origin-Resource-Policy': 'cross-origin',
      }
    : {};
  const filePath = resolveFile(urlPath);

  if (!filePath) {
    const notFound = path.join(SITE_DIR, '404.html');
    if (fs.existsSync(notFound)) {
      res.writeHead(404, { 'Content-Type': 'text/html; charset=utf-8' });
      fs.createReadStream(notFound).pipe(res);
      return;
    }
    res.writeHead(404, { 'Content-Type': 'text/plain; charset=utf-8' });
    res.end('404 Not Found');
    return;
  }

  sendFile(res, filePath, extraHeaders);
});

if (!fs.existsSync(SITE_DIR)) {
  console.error(`[serve:web] Missing ${SITE_DIR}`);
  console.error('  Run: npm run build:web   (or npm run dev:web for build + serve)');
  process.exit(1);
}

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`[serve:web] Port ${PORT} is already in use.`);
    console.error(`  Try: npm run dev:web -- --port=${PORT + 1}`);
    process.exit(1);
  }
  console.error('[serve:web]', err.message);
  process.exit(1);
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[serve:web] ${SITE_DIR}`);
  console.log(`  http://127.0.0.1:${PORT}/`);
  console.log('  COOP/COEP on /lessons/tt-lang-intro/ only (YouTube embeds work on /install/)');
  console.log('  Ctrl-C to stop.');
});
