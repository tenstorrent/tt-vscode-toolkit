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

// Get current branch, handling CI/detached HEAD scenarios
let branch = 'unknown';
let isRelease = false;

// Check GitHub Actions environment variables first
if (process.env.GITHUB_REF_TYPE === 'tag' || process.env.GITHUB_REF_NAME === 'main' || process.env.GITHUB_REF_NAME === 'master') {
  branch = process.env.GITHUB_REF_NAME || 'main';
  isRelease = true;
} else {
  try {
    // Try to detect if we're on a tag (release build)
    execSync('git describe --exact-match --tags', { encoding: 'utf8', stdio: 'pipe' });
    isRelease = true;
    branch = 'tag';
  } catch (tagError) {
    // Not on a tag, try to get current branch
    try {
      branch = execSync('git branch --show-current', { encoding: 'utf8' }).trim();
      // Empty string means detached HEAD
      if (!branch) {
        branch = 'detached-head';
      }
    } catch (error) {
      console.warn('⚠️  Could not detect git branch, assuming dev build');
      branch = 'dev';
    }
  }
}

// Determine if this is a dev build
const isMainBranch = branch === 'main' || branch === 'master' || isRelease;
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
