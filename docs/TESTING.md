# Testing Guide

This document provides comprehensive testing information for contributors to the Tenstorrent VSCode Toolkit.

## Table of Contents

- [Test Suite Overview](#test-suite-overview)
- [Running Tests](#running-tests)
- [Test Categories](#test-categories)
- [Test Structure](#test-structure)
- [Writing New Tests](#writing-new-tests)
- [CI/CD Integration](#cicd-integration)
- [Troubleshooting](#troubleshooting)

---

## Test Suite Overview

The extension includes a comprehensive automated test suite to ensure quality across all content and code:

- **134+ automated tests** covering markdown quality, Python templates, configuration, and more
- **Mocha test framework** with TypeScript support via ts-node
- **Content-focused validation** - tests verify lesson quality and template correctness
- **Fast feedback** - most tests run in under 10 seconds

**Test categories:**
1. **Markdown Validation** - Code block syntax, frontmatter, command links
2. **Template Validation** - Python syntax, structure, compatibility
3. **Configuration Tests** - Model registry, lesson metadata
4. **Project Tests** - Cookbook project validation
5. **Refactoring Tests** - Code quality and structure

---

## Running Tests

### Run All Tests

```bash
npm test
```

This runs all tests defined in `.mocharc.json`:
- All files in `test/lesson-tests/**/*.test.ts`
- Config extraction tests in `test/refactoring/configExtraction.test.ts`

**Expected output:**
```
Markdown Validation Tests
  Code Block Fencing
    ✓ 01-welcome.md should have properly matched code block fences
    ✓ 02-hardware-detection.md should have properly matched code block fences
    ... (16 lesson files tested)
  Code Block Language Specifiers
    ✓ 01-welcome.md should have language specifiers on opening code fences
    ... (16 lesson files tested)

Python Template Validation
  Python Syntax Validation
    ✓ all Python templates should have valid syntax
  File Structure
    ✓ all templates should be non-empty
    ✓ all templates should have documentation
    ... (40+ templates tested)

134 passing (3s)
```

### Run Specific Test Suites

**Templates only:**
```bash
npm run test:templates
```

**Watch mode (auto-rerun on changes):**
```bash
npm run test:watch
```

**Single test file:**
```bash
npx mocha test/lesson-tests/markdown-validation.test.ts --require ts-node/register
```

### Test Configuration

Tests are configured in `.mocharc.json`:

```json
{
  "require": ["ts-node/register"],
  "extensions": ["ts"],
  "spec": [
    "test/lesson-tests/**/*.test.ts",
    "test/refactoring/configExtraction.test.ts"
  ],
  "timeout": 10000,
  "color": true
}
```

**Key settings:**
- **Timeout:** 10 seconds per test (configurable for slow operations)
- **Extensions:** `.ts` files supported via ts-node
- **Spec patterns:** Tests auto-discovered in specified directories

---

## Test Categories

### 1. Markdown Validation Tests

**File:** `test/lesson-tests/markdown-validation.test.ts`

**Purpose:** Validate markdown formatting across all 16 lesson files

**What's tested:**

#### Code Block Fencing
- **Matched fences:** Every ` ``` ` opening has a matching ` ``` ` closing
- **Fence type consistency:** If opened with ` ``` `, must close with ` ``` ` (not `~~~`)
- **No text on closing fences:** Closing fences must be bare (` ``` ` only, no language specifier)
- **No unclosed blocks:** All code blocks properly terminated

**Example validation:**
```typescript
// ✅ Valid
```bash
echo "Hello"
```

// ❌ Invalid - closing fence has text
```bash
echo "Hello"
```bash

// ❌ Invalid - fence mismatch
```bash
echo "Hello"
~~~
```

#### Language Specifiers
- Opening fences should have language specifiers (bash, python, typescript, etc.)
- Language specifiers must be alphanumeric (valid: `bash`, `python3`, `c++`, `objective-c`)

#### Command Link Syntax
- Validates `[Button Text](command:commandId)` format
- Ensures command IDs are properly formatted
- Checks for malformed command links

**Files tested:** All 16 lesson files in `content/lessons/`

---

### 2. Template Validation Tests

**File:** `test/lesson-tests/templates.test.ts`

**Purpose:** Validate all Python script templates for correctness

**What's tested:**

#### Python Syntax Validation
- **Compilation check:** Uses `python3 -m py_compile` to verify syntax
- **No execution:** Scripts are compiled but not run (safe)
- **All 40+ templates:** Every `.py` file in `content/templates/` tested

**Example:**
```python
# ✅ Valid syntax
def hello_world():
    print("Hello, world!")

# ❌ Invalid syntax (caught by test)
def broken_function(
    print("Missing closing parenthesis"
```

#### File Structure
- **Non-empty:** Templates must be >100 characters
- **Documentation:** Must have comments (`#`) or docstrings (`"""`)
- **UTF-8 encoding:** All templates readable as UTF-8

#### Python 3 Compatibility
- **Print function:** Uses `print()` not `print` statement
- **No Python 2 syntax:** Catches `except Exception, e:` style
- **Modern imports:** Verifies proper import structure

**Files tested:** All Python templates in `content/templates/`:
- `tt-chat-direct.py`
- `tt-api-server-direct.py`
- `tt-forge-classifier.py`
- `start-vllm-server.py`
- `tt-coding-assistant.py`
- `tt-xla-gpt2-demo.py`
- ... (40+ templates total)

---

### 3. Configuration Tests

**File:** `test/refactoring/configExtraction.test.ts`

**Purpose:** Validate model registry and configuration structure

**What's tested:**
- Model registry structure (proper format)
- Required fields present (name, path, hardware compatibility)
- Configuration consistency across the extension

---

### 4. Project Validation Tests

**File:** `test/lesson-tests/projects.test.ts`

**Purpose:** Validate cookbook project structure

**What's tested:**
- Project directories exist and are properly structured
- README files present and non-empty
- Requirements files valid
- Python files have valid syntax

**Projects tested:**
- Particle Life simulation
- Game of Life (Conway's)
- Mandelbrot fractal explorer
- Audio processor demo
- Image filters demo

---

### 5. Refactoring Tests

**Files:** `test/refactoring/*.test.ts`

**Purpose:** Ensure code quality during refactoring

**What's tested:**
- Command consolidation (no duplicate command IDs)
- Configuration extraction (centralized model registry)
- Proper TypeScript types and interfaces

---

## Test Structure

### Directory Layout

```
test/
├── lesson-tests/              # Content validation tests
│   ├── markdown-validation.test.ts   # ~96 tests for markdown quality
│   ├── templates.test.ts             # ~40 tests for Python templates
│   ├── projects.test.ts              # Cookbook project tests
│   └── ...
├── refactoring/               # Code quality tests
│   ├── configExtraction.test.ts
│   └── commandConsolidation.test.ts
├── lesson-system/             # Integration tests
│   └── lessonSystem.test.ts
└── tsconfig.test.json         # TypeScript config for tests
```

### Test File Anatomy

**Standard structure:**
```typescript
import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

describe('Category Name', () => {
  // Setup - runs once before all tests
  before(() => {
    // Load files, initialize data
  });

  // Test suite
  describe('Specific Feature', () => {
    it('should validate something specific', () => {
      // Arrange
      const input = getTestData();

      // Act
      const result = validateInput(input);

      // Assert
      expect(result).to.be.true;
    });
  });
});
```

### Test Naming Conventions

**File names:** `descriptive-name.test.ts`

**Test descriptions:**
- **Describe blocks:** Noun phrases ("Markdown Validation", "Code Block Fencing")
- **It blocks:** Complete sentences ("should have properly matched code block fences")

**Examples:**
```typescript
describe('Markdown Validation Tests', () => {
  describe('Code Block Fencing', () => {
    it('should have properly matched code block fences', () => {
      // Test implementation
    });
  });
});
```

---

## Writing New Tests

### Adding a New Test File

1. **Create test file** in appropriate directory:
   ```bash
   touch test/lesson-tests/my-new-validation.test.ts
   ```

2. **Add TypeScript imports:**
   ```typescript
   import { expect } from 'chai';
   import * as fs from 'fs';
   import * as path from 'path';
   ```

3. **Structure with describe/it blocks:**
   ```typescript
   describe('My New Validation', () => {
     describe('Specific Feature', () => {
       it('should validate the feature correctly', () => {
         // Test implementation
       });
     });
   });
   ```

4. **Run your new test:**
   ```bash
   npx mocha test/lesson-tests/my-new-validation.test.ts --require ts-node/register
   ```

### Testing Markdown Content

**Pattern:** Iterate over all lesson files

```typescript
describe('Markdown Content Tests', () => {
  const lessonsDir = path.join(__dirname, '../../content/lessons');
  const lessonFiles = fs.readdirSync(lessonsDir).filter(f => f.endsWith('.md'));

  lessonFiles.forEach(file => {
    it(`${file} should meet quality standards`, () => {
      const filePath = path.join(lessonsDir, file);
      const content = fs.readFileSync(filePath, 'utf8');

      // Your validation logic here
      expect(content.length).to.be.greaterThan(100);
    });
  });
});
```

### Testing Python Templates

**Pattern:** Use py_compile for syntax checking

```typescript
import { exec } from 'child_process';
import { promisify } from 'util';
const execAsync = promisify(exec);

describe('Template Tests', () => {
  it('should have valid Python syntax', async function() {
    this.timeout(10000); // Increase timeout for compilation

    const filePath = path.join(__dirname, '../../content/templates/my-template.py');

    try {
      await execAsync(`python3 -m py_compile "${filePath}"`);
    } catch (error) {
      throw new Error(`Invalid Python syntax: ${error.message}`);
    }
  });
});
```

### Testing Configuration

**Pattern:** Load and validate JSON structure

```typescript
describe('Configuration Tests', () => {
  it('should have valid lesson registry', () => {
    const registryPath = path.join(__dirname, '../../content/lesson-registry.json');
    const registry = JSON.parse(fs.readFileSync(registryPath, 'utf8'));

    expect(registry).to.have.property('lessons');
    expect(registry.lessons).to.be.an('array');

    registry.lessons.forEach((lesson: any) => {
      expect(lesson).to.have.property('id');
      expect(lesson).to.have.property('title');
      expect(lesson).to.have.property('markdownFile');
    });
  });
});
```

### Best Practices

**DO:**
- ✅ Test one thing per `it` block
- ✅ Use descriptive test names
- ✅ Provide helpful error messages with context
- ✅ Test edge cases and error conditions
- ✅ Keep tests fast (< 1 second when possible)
- ✅ Use `before()` for expensive setup operations

**DON'T:**
- ❌ Test implementation details (test behavior, not internals)
- ❌ Make tests dependent on each other
- ❌ Use hardcoded paths (use `path.join(__dirname, ...)`)
- ❌ Leave console.log statements in tests
- ❌ Skip tests without good reason

### Example: Complete Test File

```typescript
/**
 * Feature Validation Tests
 *
 * Tests specific feature for correctness
 */

import { expect } from 'chai';
import * as fs from 'fs';
import * as path from 'path';

describe('Feature Validation', () => {
  const dataDir = path.join(__dirname, '../../content/data');
  let testFiles: string[] = [];

  // Setup: runs once before all tests
  before(() => {
    testFiles = fs.readdirSync(dataDir).filter(f => f.endsWith('.json'));
    expect(testFiles.length).to.be.greaterThan(0, 'Should have test data');
  });

  describe('File Structure', () => {
    testFiles.forEach(file => {
      it(`${file} should be valid JSON`, () => {
        const filePath = path.join(dataDir, file);
        const content = fs.readFileSync(filePath, 'utf8');

        expect(() => JSON.parse(content), `${file} should parse as JSON`).to.not.throw();
      });
    });
  });

  describe('Data Validation', () => {
    testFiles.forEach(file => {
      it(`${file} should have required fields`, () => {
        const filePath = path.join(dataDir, file);
        const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));

        expect(data).to.have.property('version');
        expect(data).to.have.property('entries');
        expect(data.entries).to.be.an('array');
      });
    });
  });
});
```

---

## CI/CD Integration

### GitHub Actions (Recommended)

**Create `.github/workflows/test.yml`:**

```yaml
name: Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Setup Node.js
      uses: actions/setup-node@v3
      with:
        node-version: '18'

    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.10'

    - name: Install dependencies
      run: npm install

    - name: Run tests
      run: npm test

    - name: Check test coverage
      run: npm run test:coverage  # if you add coverage
```

### Pre-commit Hooks

**Using Husky (optional):**

```bash
npm install --save-dev husky
npx husky install
npx husky add .husky/pre-commit "npm test"
```

This runs all tests before every commit (fails commit if tests fail).

### VSCode Integration

**Add to `.vscode/tasks.json`:**

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Run Tests",
      "type": "npm",
      "script": "test",
      "problemMatcher": [],
      "group": {
        "kind": "test",
        "isDefault": true
      }
    },
    {
      "label": "Run Tests (Watch)",
      "type": "npm",
      "script": "test:watch",
      "isBackground": true,
      "problemMatcher": []
    }
  ]
}
```

**Usage:**
- Press `Cmd+Shift+P` → "Tasks: Run Test Task" → "Run Tests"
- Or: Terminal → Run Task → "Run Tests"

---

## Troubleshooting

### Common Issues

#### Python not found

**Error:**
```
Error: python3: command not found
```

**Fix:**
```bash
# macOS/Linux
which python3

# If missing, install Python 3.10+
# macOS: brew install python@3.10
# Ubuntu: sudo apt install python3.10
```

#### TypeScript compilation errors

**Error:**
```
TSError: ⨯ Unable to compile TypeScript
```

**Fix:**
```bash
# Ensure TypeScript and ts-node are installed
npm install --save-dev typescript ts-node @types/node @types/mocha @types/chai

# Check tsconfig.test.json exists
cat tsconfig.test.json
```

#### Timeout errors

**Error:**
```
Error: Timeout of 2000ms exceeded
```

**Fix:** Increase timeout in test file:
```typescript
it('slow test', async function() {
  this.timeout(10000); // 10 seconds
  // Test implementation
});
```

Or globally in `.mocharc.json`:
```json
{
  "timeout": 10000
}
```

#### File not found errors

**Error:**
```
ENOENT: no such file or directory
```

**Fix:** Use `__dirname` for relative paths:
```typescript
// ❌ Wrong - fragile
const filePath = '../../content/lessons/file.md';

// ✅ Correct - robust
const filePath = path.join(__dirname, '../../content/lessons/file.md');
```

#### Mocha not finding tests

**Error:**
```
0 passing (1ms)
```

**Fix:**
1. Check `.mocharc.json` spec patterns match your test files
2. Ensure test files end with `.test.ts`
3. Verify test files are in directories specified by spec patterns

### Test Performance

**Slow tests?**

1. **Profile which tests are slow:**
   ```bash
   npm test -- --reporter json > test-results.json
   ```

2. **Common causes:**
   - File I/O (reading many files)
   - Python compilation (`py_compile`)
   - Network requests (avoid in tests)

3. **Optimizations:**
   - Use `before()` to load files once
   - Cache expensive operations
   - Run file-based tests in parallel (Mocha does this by default)

**Example optimization:**
```typescript
// ❌ Slow - reads file in every test
lessonFiles.forEach(file => {
  it(`${file} should be valid`, () => {
    const content = fs.readFileSync(filePath, 'utf8'); // Repeated I/O
    // Test...
  });
});

// ✅ Fast - reads all files once
let fileContents: Map<string, string>;
before(() => {
  fileContents = new Map();
  lessonFiles.forEach(file => {
    fileContents.set(file, fs.readFileSync(filePath, 'utf8'));
  });
});

lessonFiles.forEach(file => {
  it(`${file} should be valid`, () => {
    const content = fileContents.get(file);
    // Test...
  });
});
```

---

## Contributing

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

For packaging details, see [PACKAGING.md](PACKAGING.md).
