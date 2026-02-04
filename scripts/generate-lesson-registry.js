#!/usr/bin/env node

/**
 * Lesson Registry Generator
 *
 * Generates lesson-registry.json from markdown front matter (source of truth).
 *
 * SAFETY FEATURES:
 * - Dry-run mode by default (shows changes without applying)
 * - Automatic backup before any modifications
 * - Preserves manual fields (order, previousLesson, nextLesson, completionEvents)
 * - Shows clear diff of what will change
 * - Requires explicit --execute flag to apply changes
 *
 * FIELDS:
 *   FROM MARKDOWN (auto-generated):
 *     id, title, description, category, tags, supportedHardware,
 *     status, validatedOn, estimatedMinutes
 *
 *   MANUAL (preserved from existing JSON):
 *     order, previousLesson, nextLesson, completionEvents,
 *     markdownFile, recommended_metal_version
 *
 * Usage:
 *   node scripts/generate-lesson-registry.js              # Dry-run (shows changes)
 *   node scripts/generate-lesson-registry.js --execute    # Apply changes (with backup)
 *   node scripts/generate-lesson-registry.js --force      # Apply without confirmation
 *   npm run generate:lessons                              # Dry-run
 *   npm run generate:lessons -- --execute                 # Apply changes
 */

const fs = require('fs');
const path = require('path');
const yaml = require('js-yaml');
const readline = require('readline');

// Paths
const LESSONS_DIR = path.join(__dirname, '../content/lessons');
const REGISTRY_PATH = path.join(__dirname, '../content/lesson-registry.json');
const BACKUP_DIR = path.join(__dirname, '../.backups');

// Fields that come from markdown (source of truth)
const MARKDOWN_FIELDS = [
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

// Fields that are manually maintained in JSON
const MANUAL_FIELDS = [
  'order',
  'previousLesson',
  'nextLesson',
  'completionEvents',
  'markdownFile',
  'recommended_metal_version',
  'minTTMetalVersion',
  'validationDate',
  'validationNotes'
];

// Colors for terminal output
const colors = {
  reset: '\x1b[0m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  bold: '\x1b[1m'
};

/**
 * Parse YAML front matter from markdown file
 */
function parseFrontMatter(filePath) {
  const content = fs.readFileSync(filePath, 'utf8');
  const frontMatterMatch = content.match(/^---\n([\s\S]+?)\n---/);
  if (!frontMatterMatch) {
    return null;
  }
  try {
    return yaml.load(frontMatterMatch[1]);
  } catch (error) {
    console.error(`${colors.red}❌ Failed to parse ${path.basename(filePath)}: ${error.message}${colors.reset}`);
    return null;
  }
}

/**
 * Create backup of lesson-registry.json
 */
function createBackup() {
  if (!fs.existsSync(REGISTRY_PATH)) {
    console.log(`${colors.yellow}⚠️  No existing registry to backup${colors.reset}`);
    return null;
  }

  // Ensure backup directory exists
  if (!fs.existsSync(BACKUP_DIR)) {
    fs.mkdirSync(BACKUP_DIR, { recursive: true });
  }

  // Create timestamped backup
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  const backupPath = path.join(BACKUP_DIR, `lesson-registry-${timestamp}.json`);

  fs.copyFileSync(REGISTRY_PATH, backupPath);
  console.log(`${colors.green}✅ Backup created: ${path.relative(process.cwd(), backupPath)}${colors.reset}`);
  return backupPath;
}

/**
 * Compare two values and return diff description
 */
function getDiff(oldVal, newVal, field) {
  if (JSON.stringify(oldVal) === JSON.stringify(newVal)) {
    return null; // No change
  }

  if (oldVal === undefined) {
    return `${colors.green}+ ADD${colors.reset} ${field}: ${JSON.stringify(newVal)}`;
  }
  if (newVal === undefined) {
    return `${colors.red}- REMOVE${colors.reset} ${field}: ${JSON.stringify(oldVal)}`;
  }
  return `${colors.yellow}~ CHANGE${colors.reset} ${field}:\n    OLD: ${JSON.stringify(oldVal)}\n    NEW: ${JSON.stringify(newVal)}`;
}

/**
 * Generate updated lesson entry
 */
function generateLessonEntry(markdownPath, existingEntry) {
  const frontMatter = parseFrontMatter(markdownPath);
  if (!frontMatter || !frontMatter.id) {
    return null;
  }

  const newEntry = {};

  // Copy markdown fields (source of truth)
  for (const field of MARKDOWN_FIELDS) {
    if (frontMatter[field] !== undefined) {
      newEntry[field] = frontMatter[field];
    }
  }

  // Preserve manual fields from existing entry
  if (existingEntry) {
    for (const field of MANUAL_FIELDS) {
      if (existingEntry[field] !== undefined) {
        newEntry[field] = existingEntry[field];
      }
    }
  } else {
    // New lesson - set defaults for manual fields
    newEntry.markdownFile = `content/lessons/${path.basename(markdownPath)}`;
  }

  return newEntry;
}

/**
 * Show diff between old and new registry
 */
function showDiff(oldRegistry, newLessonsMap) {
  console.log(`\n${colors.bold}${colors.cyan}📋 CHANGES PREVIEW${colors.reset}\n`);

  let changeCount = 0;
  const oldLessonsMap = {};
  for (const lesson of oldRegistry.lessons) {
    oldLessonsMap[lesson.id] = lesson;
  }

  // Check for changes and additions
  for (const [id, newLesson] of Object.entries(newLessonsMap)) {
    const oldLesson = oldLessonsMap[id];
    const diffs = [];

    if (!oldLesson) {
      console.log(`${colors.green}${colors.bold}➕ NEW LESSON: ${id}${colors.reset}`);
      console.log(`   ${JSON.stringify(newLesson, null, 2).split('\n').slice(1, -1).join('\n   ')}\n`);
      changeCount++;
      continue;
    }

    // Check each markdown field for changes
    for (const field of MARKDOWN_FIELDS) {
      const diff = getDiff(oldLesson[field], newLesson[field], field);
      if (diff) {
        diffs.push(diff);
      }
    }

    if (diffs.length > 0) {
      console.log(`${colors.yellow}${colors.bold}📝 MODIFY: ${id}${colors.reset}`);
      for (const diff of diffs) {
        console.log(`   ${diff}`);
      }
      console.log('');
      changeCount++;
    }
  }

  // Check for removals
  for (const [id, oldLesson] of Object.entries(oldLessonsMap)) {
    if (!newLessonsMap[id]) {
      console.log(`${colors.red}${colors.bold}➖ REMOVE: ${id}${colors.reset}`);
      console.log(`   Lesson exists in registry but no markdown file found\n`);
      changeCount++;
    }
  }

  if (changeCount === 0) {
    console.log(`${colors.green}✅ No changes needed - registry is already in sync!${colors.reset}\n`);
  } else {
    console.log(`${colors.bold}${changeCount} lesson(s) will be updated${colors.reset}\n`);
  }

  return changeCount;
}

/**
 * Ask user for confirmation
 */
async function confirm(question) {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
  });

  return new Promise((resolve) => {
    rl.question(`${colors.bold}${question}${colors.reset} `, (answer) => {
      rl.close();
      resolve(answer.toLowerCase() === 'y' || answer.toLowerCase() === 'yes');
    });
  });
}

/**
 * Main generation function
 */
async function main() {
  const args = process.argv.slice(2);
  const isDryRun = !args.includes('--execute');
  const isForce = args.includes('--force');

  console.log(`${colors.bold}${colors.magenta}🔧 Lesson Registry Generator${colors.reset}\n`);

  if (isDryRun) {
    console.log(`${colors.cyan}ℹ️  DRY-RUN MODE: Will show changes without applying them${colors.reset}`);
    console.log(`   Run with ${colors.bold}--execute${colors.reset} to apply changes\n`);
  }

  // Load existing registry
  let oldRegistry;
  try {
    oldRegistry = JSON.parse(fs.readFileSync(REGISTRY_PATH, 'utf8'));
  } catch (error) {
    console.error(`${colors.red}❌ Failed to load existing registry: ${error.message}${colors.reset}`);
    console.log(`${colors.yellow}⚠️  Will create new registry from scratch${colors.reset}\n`);
    oldRegistry = { version: '1.0.0', categories: [], lessons: [] };
  }

  // Build old lessons map
  const oldLessonsMap = {};
  for (const lesson of oldRegistry.lessons) {
    oldLessonsMap[lesson.id] = lesson;
  }

  // Generate new lessons from markdown
  const markdownFiles = fs.readdirSync(LESSONS_DIR)
    .filter(file => file.endsWith('.md'))
    .map(file => path.join(LESSONS_DIR, file));

  console.log(`${colors.blue}📚 Processing ${markdownFiles.length} markdown files...${colors.reset}\n`);

  const newLessonsMap = {};
  for (const markdownPath of markdownFiles) {
    const newEntry = generateLessonEntry(markdownPath, oldLessonsMap[path.basename(markdownPath, '.md')]);
    if (newEntry && newEntry.id) {
      // Try to find existing lesson by ID
      const existingEntry = oldLessonsMap[newEntry.id];
      newLessonsMap[newEntry.id] = generateLessonEntry(markdownPath, existingEntry);
    }
  }

  // Show diff
  const changeCount = showDiff(oldRegistry, newLessonsMap);

  if (changeCount === 0) {
    process.exit(0);
  }

  // If dry-run, stop here
  if (isDryRun) {
    console.log(`${colors.cyan}ℹ️  To apply these changes, run:${colors.reset}`);
    console.log(`   ${colors.bold}npm run generate:lessons -- --execute${colors.reset}\n`);
    process.exit(0);
  }

  // Confirm before applying
  if (!isForce) {
    console.log(`${colors.red}${colors.bold}⚠️  WARNING: This will modify lesson-registry.json${colors.reset}\n`);
    const confirmed = await confirm('Do you want to continue? (y/N): ');
    if (!confirmed) {
      console.log(`${colors.yellow}❌ Cancelled by user${colors.reset}\n`);
      process.exit(0);
    }
  }

  // Create backup
  const backupPath = createBackup();

  // Build new registry (preserve categories and other top-level fields)
  const newRegistry = {
    version: oldRegistry.version,
    _warning: oldRegistry._warning || "⚠️  PARTIALLY AUTO-GENERATED - See scripts/generate-lesson-registry.js",
    categories: oldRegistry.categories,
    lessons: []
  };

  // Preserve order from old registry where possible
  const processedIds = new Set();

  // First, add lessons that existed before (preserving order)
  for (const oldLesson of oldRegistry.lessons) {
    if (newLessonsMap[oldLesson.id]) {
      newRegistry.lessons.push(newLessonsMap[oldLesson.id]);
      processedIds.add(oldLesson.id);
    }
  }

  // Then add new lessons
  for (const [id, newLesson] of Object.entries(newLessonsMap)) {
    if (!processedIds.has(id)) {
      newRegistry.lessons.push(newLesson);
    }
  }

  // Write new registry
  try {
    fs.writeFileSync(REGISTRY_PATH, JSON.stringify(newRegistry, null, 2) + '\n', 'utf8');
    console.log(`${colors.green}${colors.bold}✅ Successfully updated lesson-registry.json${colors.reset}\n`);

    if (backupPath) {
      console.log(`${colors.cyan}ℹ️  Original backed up to: ${path.relative(process.cwd(), backupPath)}${colors.reset}`);
      console.log(`${colors.cyan}   To restore: cp "${backupPath}" "${REGISTRY_PATH}"${colors.reset}\n`);
    }

    console.log(`${colors.green}🎉 Registry generation complete!${colors.reset}`);
    console.log(`${colors.cyan}   Run ${colors.bold}npm run validate:lessons${colors.reset}${colors.cyan} to verify${colors.reset}\n`);

  } catch (error) {
    console.error(`${colors.red}❌ Failed to write registry: ${error.message}${colors.reset}\n`);
    if (backupPath) {
      console.log(`${colors.yellow}   Restore backup: cp "${backupPath}" "${REGISTRY_PATH}"${colors.reset}\n`);
    }
    process.exit(1);
  }
}

// Run generator
if (require.main === module) {
  main().catch(error => {
    console.error(`${colors.red}❌ Fatal error: ${error.message}${colors.reset}`);
    process.exit(1);
  });
}

module.exports = { generateLessonEntry, parseFrontMatter };
