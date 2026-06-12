#!/usr/bin/env node
/**
 * Verify dependencies required by build-web.js are installed.
 * Used as predev:web / prebuild:web hook.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const NM = path.join(ROOT, 'node_modules');

const REQUIRED = [
  'marked',
  'marked-highlight',
  'gray-matter',
  'isomorphic-dompurify',
  'sanitize-html',
  'highlight.js',
  'mermaid',
];

function missing() {
  if (!fs.existsSync(NM)) return REQUIRED;
  return REQUIRED.filter((pkg) => !fs.existsSync(path.join(NM, pkg)));
}

const absent = missing();
if (absent.length) {
  console.error('\n[dev:web] Missing npm packages:', absent.join(', '));
  console.error('  From the project root, run:\n');
  console.error('    npm install\n');
  console.error('  Then start the preview server again:\n');
  console.error('    npm run dev:web\n');
  process.exit(1);
}
