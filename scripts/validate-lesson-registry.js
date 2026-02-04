#!/usr/bin/env node

/**
 * Lesson Registry Validation Script
 *
 * Validates that lesson-registry.json is in sync with markdown front matter.
 *
 * Source of Truth: Markdown front matter
 * This script ensures that the JSON registry accurately reflects the markdown files.
 *
 * Fields validated (must match between markdown and JSON):
 * - id, title, description, category, tags
 * - supportedHardware, status, validatedOn, estimatedMinutes
 *
 * Fields only in JSON (not validated, extension-specific):
 * - order, previousLesson, nextLesson, completionEvents
 *
 * Usage:
 *   node scripts/validate-lesson-registry.js
 *   npm run validate:lessons
 *
 * Exit codes:
 *   0 = All lessons valid
 *   1 = Validation errors found
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');

// Paths
const LESSONS_DIR = path.join(__dirname, '../content/lessons');
const REGISTRY_PATH = path.join(__dirname, '../content/lesson-registry.json');

// Fields that must match between markdown and JSON
const VALIDATED_FIELDS = [
  'id',
  'title',
  'description',
  'category',
  'tags',
  'supportedHardware',
  'status',
  'validatedOn',
  'estimatedMinutes'
];

/**
 * Parse YAML front matter from markdown file
 */
function parseFrontMatter(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');

  // Extract front matter between --- delimiters
  const frontMatterMatch = content.match(/^---\n([\s\S]+?)\n---/);
  if (!frontMatterMatch) {
    return null;
  }

  try {
    return yaml.load(frontMatterMatch[1]);
  } catch (error) {
    console.error(`❌ Failed to parse front matter in ${path.basename(filePath)}: ${error.message}`);
    return null;
  }
}

/**
 * Deep equality check for arrays and primitives
 */
function deepEqual(a, b) {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    const aSorted = [...a].sort();
    const bSorted = [...b].sort();
    return aSorted.every((val, idx) => deepEqual(val, bSorted[idx]));
  }
  if (typeof a === 'object' && typeof b === 'object') {
    const keysA = Object.keys(a).sort();
    const keysB = Object.keys(b).sort();
    if (!deepEqual(keysA, keysB)) return false;
    return keysA.every(key => deepEqual(a[key], b[key]));
  }
  return false;
}

/**
 * Normalize description (remove extra whitespace from multiline strings)
 */
function normalizeDescription(desc) {
  if (typeof desc !== 'string') return desc;
  return desc.replace(/\s+/g, ' ').trim();
}

/**
 * Validate a single lesson
 */
function validateLesson(markdownPath, registryEntry) {
  const lessonName = path.basename(markdownPath, '.md');
  const frontMatter = parseFrontMatter(markdownPath);

  if (!frontMatter) {
    return [`❌ ${lessonName}: Failed to parse front matter`];
  }

  if (!registryEntry) {
    return [`❌ ${lessonName}: Not found in lesson-registry.json (id: ${frontMatter.id})`];
  }

  const errors = [];

  // Validate each field
  for (const field of VALIDATED_FIELDS) {
    const mdValue = frontMatter[field];
    const jsonValue = registryEntry[field];

    // Special handling for description (multiline strings)
    if (field === 'description') {
      const mdDesc = normalizeDescription(mdValue);
      const jsonDesc = normalizeDescription(jsonValue);
      if (mdDesc !== jsonDesc) {
        errors.push(`  ❌ ${field}:\n    Markdown: "${mdDesc}"\n    JSON:     "${jsonDesc}"`);
      }
      continue;
    }

    // Check if field exists in markdown but not in JSON
    if (mdValue !== undefined && jsonValue === undefined) {
      errors.push(`  ❌ ${field}: Present in markdown but missing in JSON`);
      continue;
    }

    // Check if field exists in JSON but not in markdown
    if (mdValue === undefined && jsonValue !== undefined) {
      errors.push(`  ⚠️  ${field}: Present in JSON but missing in markdown (consider adding to markdown)`);
      continue;
    }

    // Compare values
    if (!deepEqual(mdValue, jsonValue)) {
      errors.push(`  ❌ ${field}:\n    Markdown: ${JSON.stringify(mdValue)}\n    JSON:     ${JSON.stringify(jsonValue)}`);
    }
  }

  if (errors.length > 0) {
    return [`❌ ${lessonName} (id: ${frontMatter.id}):`, ...errors];
  }

  return null; // No errors
}

/**
 * Main validation function
 */
function main() {
  console.log('🔍 Validating lesson registry...\n');

  // Load registry
  let registry;
  try {
    registry = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
  } catch (error) {
    console.error(`❌ Failed to load lesson-registry.json: ${error.message}`);
    process.exit(1);
  }

  // Build registry lookup by ID
  const registryById = {};
  for (const lesson of registry.lessons) {
    registryById[lesson.id] = lesson;
  }

  // Get all markdown files
  const markdownFiles = fs.readdirSync(LESSONS_DIR)
    .filter(file => file.endsWith('.md'))
    .map(file => path.join(LESSONS_DIR, file));

  console.log(`📚 Found ${markdownFiles.length} markdown lessons`);
  console.log(`📋 Found ${registry.lessons.length} JSON registry entries\n`);

  // Validate each markdown file
  const allErrors = [];
  let validCount = 0;

  for (const markdownPath of markdownFiles) {
    const frontMatter = parseFrontMatter(markdownPath);
    if (!frontMatter || !frontMatter.id) {
      continue; // Skip files without proper front matter
    }

    const registryEntry = registryById[frontMatter.id];
    const errors = validateLesson(markdownPath, registryEntry);

    if (errors) {
      allErrors.push(...errors, ''); // Add blank line between lessons
    } else {
      validCount++;
    }
  }

  // Report results
  console.log('━'.repeat(60));
  if (allErrors.length === 0) {
    console.log(`✅ All ${validCount} lessons are valid!\n`);
    console.log('Markdown front matter matches lesson-registry.json perfectly.');
    process.exit(0);
  } else {
    console.log(`❌ Found validation errors:\n`);
    console.log(allErrors.join('\n'));
    console.log('━'.repeat(60));
    console.log(`\n❌ ${allErrors.filter(e => e.startsWith('❌')).length} lessons with errors`);
    console.log(`✅ ${validCount} lessons valid\n`);
    console.log('💡 To fix: Update lesson-registry.json to match markdown front matter');
    console.log('   Or run: npm run generate:lessons (if generator exists)\n');
    process.exit(1);
  }
}

// Run validation
if (require.main === module) {
  main();
}

module.exports = { validateLesson, parseFrontMatter };
