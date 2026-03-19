#!/usr/bin/env node

/**
 * Package extension with appropriate filename based on branch
 * - main branch: tt-vscode-toolkit-X.Y.Z.vsix
 * - other branches: tt-vscode-toolkit-X.Y.Z-dev.vsix
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Read package.json for version
const packageJson = JSON.parse(
  fs.readFileSync(path.join(__dirname, '..', 'package.json'), 'utf8')
);

const { name, version } = packageJson;

// Get current branch
let branch = 'unknown';
try {
  branch = execSync('git branch --show-current', { encoding: 'utf8' }).trim();
} catch (error) {
  console.warn('⚠️  Could not detect git branch, assuming dev build');
  branch = 'dev';
}

// Determine if this is a dev build
const isMainBranch = branch === 'main' || branch === 'master';
const suffix = isMainBranch ? '' : '-dev';

// Build filename
const filename = `${name}-${version}${suffix}.vsix`;

console.log(`📦 Packaging extension...`);
console.log(`   Branch: ${branch}`);
console.log(`   Version: ${version}`);
console.log(`   Output: ${filename}`);
console.log('');

// Run vsce package with custom filename
try {
  execSync(`vsce package --out ${filename}`, { stdio: 'inherit' });
  console.log('');
  console.log(`✅ Package created: ${filename}`);
} catch (error) {
  console.error('❌ Packaging failed');
  process.exit(1);
}
