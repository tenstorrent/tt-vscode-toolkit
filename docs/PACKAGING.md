# Packaging Guide

This document provides comprehensive packaging and distribution information for the Tenstorrent VSCode Toolkit.

## Table of Contents

- [Build Process](#build-process)
- [Package Structure](#package-structure)
- [Lesson Validation](#lesson-validation)
- [Production vs Development Builds](#production-vs-development-builds)
- [Package Size Optimization](#package-size-optimization)
- [Distribution](#distribution)
- [Version Management](#version-management)
- [Troubleshooting](#troubleshooting)

---

## Build Process

### Standard Build Workflow

The extension uses a TypeScript compilation + content copying approach:

```bash
# Full build sequence
npm run build
```

**What happens:**
1. **Clean** - Removes `dist/` directory
2. **Compile** - TypeScript → JavaScript (`tsc -p ./`)
3. **Copy Content** - Markdown, templates, assets → `dist/`
4. **Clean Pycache** - Removes Python cache files

### Build Scripts

**Defined in `package.json`:**

```json
{
  "scripts": {
    "vscode:prepublish": "npm run build",
    "build": "npm run clean && tsc -p ./ && npm run copy-content && npm run clean-pycache",
    "clean": "rm -rf dist/",
    "clean-pycache": "find dist -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true",
    "copy-content": "mkdir -p dist/content dist/src/webview dist/src/telemetry dist/assets/img dist/themes dist/vendor && cp -r content/lessons dist/content/ && cp -r content/templates dist/content/ && cp -r content/pages dist/content/ && cp -r content/projects dist/content/ && cp content/lesson-registry.json dist/content/ && cp -r src/webview/* dist/src/webview/ && cp src/telemetry/*.py dist/src/telemetry/ && cp -r assets/img/* dist/assets/img/ && cp -r themes/* dist/themes/ && cp node_modules/mermaid/dist/mermaid.min.js dist/vendor/",
    "watch": "tsc -watch -p ./",
    "package": "vsce package",
    "compile": "tsc -p ./"
  }
}
```

### Individual Build Steps

**Clean build artifacts:**
```bash
npm run clean
```

**Compile TypeScript only:**
```bash
npm run compile
# or for watch mode:
npm run watch
```

**Copy content only:**
```bash
npm run copy-content
```

**Full development build:**
```bash
npm run build
```

**Create .vsix package:**
```bash
npm run package
```

---

## Package Structure

### What Gets Packaged

**Included in .vsix:**
```
tt-vscode-toolkit-X.Y.Z.vsix
├── dist/                          # Compiled code and content
│   ├── extension.js               # Main extension entry point
│   ├── views/                     # View managers
│   ├── commands/                  # Command handlers
│   ├── renderers/                 # Markdown renderer
│   ├── content/                   # Lessons, templates, pages
│   │   ├── lessons/               # 16 markdown lesson files
│   │   ├── templates/             # 40+ Python script templates
│   │   ├── pages/                 # Welcome page, FAQ
│   │   ├── projects/              # Cookbook projects
│   │   └── lesson-registry.json   # Lesson metadata
│   ├── src/webview/               # Webview resources
│   │   ├── scripts/lesson-viewer.js
│   │   └── styles/lesson-theme.css
│   ├── assets/img/                # Images (excluding .gif files)
│   └── themes/                    # VSCode color themes
├── node_modules/                  # Dependencies (marked, dompurify, etc.)
├── package.json                   # Extension manifest
├── README.md                      # User documentation
├── CONTRIBUTING.md                # Contribution guide
├── CHANGELOG.md                   # Version history
├── FAQ.md                         # Troubleshooting
└── LICENSE                        # MIT license
```

**Typical package size:** ~5-6 MB (with all dependencies)

### What's Excluded

**Defined in `.vscodeignore`:**

```
# Development files
.vscode/
.vscode-test/
.claude/
.git/
.gitignore

# Python cache files (CRITICAL)
**/__pycache__/
**/*.pyc
**/*.pyo
**/*.pyd

# Build configs
tsconfig.json
tsconfig.test.json
.mocharc.json

# Source files (only dist/ needed)
src/
test/

# Build artifacts
*.vsix
vsix_archive/

# Documentation (development only)
docs/
saved_notes/
scripts/

# Vendor reference repos (NOT deployed)
vendor/

# Large asset files
**/*.gif

# Package manager
package-lock.json
```

**Why exclude these?**

- **`src/`** - Only compiled `dist/` needed at runtime
- **`test/`** - Tests don't ship to users
- **`docs/`** - Technical docs for contributors, not end users
- **`vendor/`** - Reference repos for development only (NOT deployed)
- **`**/*.gif`** - Large animated files (kept in repo for docs, excluded from package)
- **`**/__pycache__/`** - Python cache bloats package size significantly

---

## Lesson Validation

### Validation Metadata System

Every lesson has metadata tracking validation status and hardware compatibility.

**Metadata structure in `content/lesson-registry.json`:**

```json
{
  "lessons": [
    {
      "id": "hardware-detection",
      "title": "Hardware Detection",
      "markdownFile": "content/lessons/02-hardware-detection.md",
      "category": "first-inference",
      "order": 2,
      "metadata": {
        "supportedHardware": ["n150", "n300", "t3k", "p100", "p150", "galaxy"],
        "status": "validated",
        "validatedOn": ["n150", "n300", "t3k"],
        "minTTMetalVersion": "v0.51.0"
      }
    }
  ]
}
```

### Validation Status Values

**`status` field:**

- **`validated`** - Tested and ready for production release
- **`draft`** - In development, should be hidden in production builds
- **`blocked`** - Known issue, show with warning or hide

**`validatedOn` array:**
- Lists hardware types where lesson has been tested
- Values: `n150`, `n300`, `t3k`, `p100`, `p150`, `galaxy`, `simulator`

**`blockReason` (optional):**
- Human-readable explanation if status is `blocked`
- Example: "Requires tt-metal v0.52.0+ (not yet released)"

### Hardware Types

**Supported values:**

| Hardware | Description | Family |
|----------|-------------|--------|
| `n150` | Wormhole - Single chip | Wormhole |
| `n300` | Wormhole - Dual chip | Wormhole |
| `t3k` | TT-QuietBox - 8 chips | Wormhole |
| `p100` | Blackhole - Single chip | Blackhole |
| `p150` | Blackhole - Dual chip | Blackhole |
| `galaxy` | TT-LoudBox - 32 chips | Wormhole |
| `simulator` | Software simulator | N/A |

---

## Production vs Development Builds

### Configuration Setting

**`package.json` configuration:**

```json
{
  "contributes": {
    "configuration": {
      "properties": {
        "tenstorrent.showUnvalidatedLessons": {
          "type": "boolean",
          "default": true,
          "description": "Show lessons that haven't been validated on hardware yet."
        }
      }
    }
  }
}
```

### Development Builds (Default)

**Setting:** `showUnvalidatedLessons: true`

**Behavior:**
- Shows ALL lessons regardless of status
- Displays `validated`, `draft`, and `blocked` lessons
- Allows testing of experimental content
- User can toggle setting via VSCode Settings

**Use case:** Internal development, beta testing, experimental features

### Production Builds (Recommended for Release)

**Setting:** `showUnvalidatedLessons: false`

**Behavior:**
- Shows ONLY `validated` lessons
- Hides `draft` and `blocked` lessons
- Ensures stable user experience
- Prevents confusion from incomplete content

**Use case:** Public releases, marketplace distribution

### Changing Default for Production

**Before release, update `package.json`:**

```json
{
  "contributes": {
    "configuration": {
      "properties": {
        "tenstorrent.showUnvalidatedLessons": {
          "default": false  // ← Change to false for production
        }
      }
    }
  }
}
```

**Recommended workflow:**
1. Keep `default: true` on main branch (development)
2. Change to `default: false` for release tags/branches
3. Users can still enable via Settings if needed

---

## Package Size Optimization

### Historical Package Sizes

| Version | Size | Files | Notes |
|---------|------|-------|-------|
| 0.0.124 | 783 KB | 193 | Before dependency bundling |
| 0.0.125 | 389 KB | 114 | Removed duplicate content/ (rolled back) |
| 0.0.126 | 5.42 MB | 1942 | Restored dependencies (necessary) |
| 0.0.241 | 5.16 MB | 1942 | Excluded .gif files (60MB → 31MB reduction) |
| 0.0.243 | ~5.4 MB | 1949 | Current |

### Why Size Increased (v0.0.126)

**Problem:** Version 0.0.125 excluded dependencies but didn't bundle them
- Extension threw `Error: Cannot find module 'marked'`
- Dependencies (marked, dompurify, jsdom, etc.) were missing

**Solution:** Include `node_modules/` in package
- Extension works correctly with all dependencies
- Package size increases to ~5.4 MB
- **This is normal for extensions without bundling**

**Future optimization:** Set up webpack or esbuild to bundle dependencies
- Would reduce package size significantly
- Requires build process changes
- Not currently implemented

### Current Optimizations

**1. Exclude .gif files (v0.0.241)**
```
# .vscodeignore
**/*.gif
```
- Reduced package size by 60 MB
- GIF files kept in repository for documentation
- Not needed for extension runtime

**2. Clean Python cache (v0.0.126)**
```bash
npm run clean-pycache
```
- Removes `__pycache__/` directories
- Removes `.pyc`, `.pyo`, `.pyd` files
- Python cache can bloat package significantly

**3. Exclude development files**
```
# .vscodeignore
src/              # Source TypeScript (only dist/ needed)
test/             # Test files
docs/             # Technical docs
vendor/           # Reference repos
```

### Future Optimizations

**Webpack bundling (recommended):**

```bash
npm install --save-dev webpack webpack-cli ts-loader
```

**Benefits:**
- Bundle all dependencies into single file
- Tree-shaking removes unused code
- Minification reduces file size
- Estimated reduction: 5.4 MB → 1-2 MB

**Trade-offs:**
- More complex build process
- Harder to debug
- Requires webpack configuration

---

## Distribution

### Creating a Package

**1. Update version in `package.json`:**
```json
{
  "version": "0.0.243"
}
```

**2. Run tests:**
```bash
npm test
```

**3. Build and package:**
```bash
npm run build
npm run package
```

**Output:** `tt-vscode-toolkit-0.0.243.vsix`

### Installing Locally

**From command line:**
```bash
code --install-extension tt-vscode-toolkit-0.0.243.vsix
```

**From VSCode UI:**
1. Open Extensions view (`Cmd+Shift+X`)
2. Click `...` menu → "Install from VSIX..."
3. Select `.vsix` file

**Force reinstall (for testing):**
```bash
code --install-extension tt-vscode-toolkit-0.0.243.vsix --force
```

### Publishing to Marketplace

**Prerequisites:**
1. [Visual Studio Marketplace account](https://marketplace.visualstudio.com/)
2. Personal Access Token (PAT) from Azure DevOps
3. Publisher account (`tenstorrent`)

**Publish command:**
```bash
vsce publish
```

**Or specify version bump:**
```bash
vsce publish patch   # 0.0.243 → 0.0.244
vsce publish minor   # 0.0.243 → 0.1.0
vsce publish major   # 0.0.243 → 1.0.0
```

**Manual upload:**
1. Create package: `npm run package`
2. Go to [Visual Studio Marketplace publisher page](https://marketplace.visualstudio.com/manage)
3. Upload `.vsix` file manually

### Pre-publication Checklist

**Before publishing:**

- [ ] Update version in `package.json`
- [ ] Update `CHANGELOG.md` with new version
- [ ] Update "What's New" section in `README.md` (5 most recent)
- [ ] Run full test suite: `npm test`
- [ ] Build fresh package: `npm run build && npm run package`
- [ ] Test installation: `code --install-extension *.vsix --force`
- [ ] Manually test key features:
  - [ ] Hardware detection
  - [ ] Lesson webview rendering
  - [ ] Command execution
  - [ ] Terminal integration
- [ ] Verify package size: `ls -lh *.vsix`
- [ ] Check file count: `unzip -l *.vsix | wc -l`
- [ ] Set `showUnvalidatedLessons: false` for production (optional)
- [ ] Create git tag: `git tag v0.0.243`
- [ ] Push tag: `git push origin v0.0.243`

---

## Version Management

### Semantic Versioning

**Format:** `MAJOR.MINOR.PATCH`

**Current:** `0.0.X` (pre-1.0 development)

**Increment rules:**

- **PATCH** (`0.0.243 → 0.0.244`) - Bug fixes, typos, small improvements
- **MINOR** (`0.0.243 → 0.1.0`) - New features, new lessons, significant additions
- **MAJOR** (`0.0.243 → 1.0.0`) - Breaking changes, major rewrites

**Examples:**

| Change | Version Bump |
|--------|--------------|
| Fix typo in lesson | PATCH |
| Add new command | PATCH |
| Add new lesson | MINOR |
| Rewrite lesson system | MAJOR |
| Update dependencies | PATCH |
| Add hardware support | MINOR |

### Version Update Workflow

**1. Update `package.json`:**
```json
{
  "version": "0.0.244"
}
```

**2. Update `CHANGELOG.md`:**
```markdown
## [0.0.244] - 2024-01-15

### Added
- New feature X

### Fixed
- Bug in Y

### Changed
- Updated Z
```

**3. Update `README.md` "What's New":**
```markdown
## 📊 What's New

### Version 0.0.244 (Latest)
- 🎯 Added feature X
- 🐛 Fixed bug in Y
- ✨ Updated Z
```

**4. Commit changes:**
```bash
git add package.json CHANGELOG.md README.md
git commit -m "Bump version to 0.0.244"
```

**5. Create git tag:**
```bash
git tag v0.0.244
git push origin main --tags
```

**6. Build and test:**
```bash
npm run build
npm test
npm run package
```

**7. Publish:**
```bash
vsce publish
```

---

## Troubleshooting

### Common Issues

#### Package size too large

**Error:**
```
Error: Extension package size exceeds 50MB limit
```

**Diagnosis:**
```bash
# Check package contents
unzip -l tt-vscode-toolkit-0.0.243.vsix | sort -k4 -rn | head -20

# Check for large files
unzip -l tt-vscode-toolkit-0.0.243.vsix | awk '{print $4, $8}' | sort -rn | head -20
```

**Common causes:**
- .gif files not excluded
- Python `__pycache__/` directories included
- Vendor repos accidentally included

**Fix:**
1. Update `.vscodeignore` to exclude large files
2. Run `npm run clean-pycache`
3. Rebuild: `npm run build && npm run package`

#### Missing dependencies at runtime

**Error:**
```
Error: Cannot find module 'marked'
```

**Cause:** Dependencies excluded from package but not bundled

**Fix:** Ensure `node_modules/` is NOT in `.vscodeignore`

**Check:**
```bash
cat .vscodeignore | grep node_modules
# Should be commented out or absent
```

#### Stale files in dist/

**Symptom:** Old code executing, unexpected behavior

**Cause:** `dist/` not cleaned before build

**Fix:**
```bash
# Manual clean
rm -rf dist/
npm run build

# Or use clean script
npm run clean && npm run build
```

**Prevention:** Build script includes `clean` step:
```json
{
  "scripts": {
    "build": "npm run clean && tsc -p ./ && npm run copy-content"
  }
}
```

#### VSIX won't install

**Error:**
```
Error: Unable to install extension
```

**Diagnosis:**
```bash
# Check package integrity
unzip -t tt-vscode-toolkit-0.0.243.vsix

# Check package.json is valid
unzip -p tt-vscode-toolkit-0.0.243.vsix extension/package.json | jq .
```

**Common causes:**
- Corrupt .vsix file
- Invalid package.json
- Missing required files (extension.js, package.json)

**Fix:** Rebuild package from scratch:
```bash
npm run clean
npm install
npm run build
npm run package
```

#### Tests fail before package

**Error:**
```
96 passing
2 failing
```

**Fix:** Tests must pass before packaging:
```bash
npm test

# Fix failing tests, then:
npm run build
npm run package
```

**CI/CD enforcement:** Set up pre-publish hook:
```json
{
  "scripts": {
    "prepublish": "npm test"
  }
}
```

---

## Best Practices

### Development Workflow

**Daily development:**
```bash
# Start watch mode
npm run watch

# In another terminal: run tests
npm run test:watch

# Press F5 in VSCode to test extension
```

**Before committing:**
```bash
# Full build and test
npm run build
npm test

# Optional: create package to verify
npm run package
ls -lh *.vsix
```

### Release Workflow

**Pre-release:**
1. ✅ All tests passing
2. ✅ Version bumped
3. ✅ CHANGELOG.md updated
4. ✅ README.md "What's New" updated
5. ✅ Git tag created
6. ✅ Package created and tested

**Release:**
1. Push commits and tags
2. Create package: `npm run package`
3. Test installation locally
4. Publish to marketplace: `vsce publish`
5. Verify marketplace listing

**Post-release:**
1. Announce on Discord/community channels
2. Update documentation if needed
3. Monitor for issues

---

## Contributing

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

For architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).

For testing details, see [TESTING.md](TESTING.md).
