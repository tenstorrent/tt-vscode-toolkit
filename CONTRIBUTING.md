# Contributing to Tenstorrent VSCode Toolkit

Thank you for your interest in contributing to the Tenstorrent VSCode Toolkit! We welcome contributions from developers of all experience levels.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Development Workflow](#development-workflow)
- [Code Style and Standards](#code-style-and-standards)
- [Git Workflow](#git-workflow)
- [Testing](#testing)
- [Documentation](#documentation)
- [Adding New Lessons](#adding-new-lessons)
- [Packaging and Distribution](#packaging-and-distribution)
- [Code Review Process](#code-review-process)
- [Community Guidelines](#community-guidelines)
- [Getting Help](#getting-help)

---

## Getting Started

All contributions require:
* **An issue** - File a bug report, feature request, or lesson request under [Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues) using the appropriate template
* **A pull request (PR)** - Your PR must be approved by at least one maintainer

**Before contributing:**
1. Read this guide thoroughly
2. Review our [Code of Conduct](CODE_OF_CONDUCT.md)
3. Check [existing issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues) to avoid duplicates
4. Join our [Discord](https://discord.gg/tenstorrent) for real-time discussions

---

## Development Setup

### Prerequisites

**Required:**
- Node.js 18+ and npm 9+
- VSCode 1.93+
- Git 2.x
- Tenstorrent hardware (for lesson validation)

**Optional:**
- tt-metal installed (for testing lesson content)
- Access to Tenstorrent Discord (for discussions)

### Quick Setup

```bash
# 1. Clone repository
git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
cd tt-vscode-toolkit

# 2. Install dependencies
npm install

# 3. Build extension
npm run build

# 4. Open in VSCode
code .

# 5. Press F5 to launch Extension Development Host
# The extension will load in a new VSCode window for testing
```

### Verify Setup

**Run tests:**
```bash
npm test              # Run all tests (should see 134+ passing)
npm run test:watch    # Watch mode for development
```

**Build and package:**
```bash
npm run build         # Compile TypeScript + copy content
npm run package       # Create .vsix file
```

**Manual testing:**
1. Press `F5` in VSCode to launch Extension Development Host
2. In the new window, click Tenstorrent icon in activity bar
3. Open a lesson and test commands
4. Check console for errors (Help → Toggle Developer Tools)

---

## Project Structure

Understanding the codebase organization:

```
tt-vscode-toolkit/
├── .github/
│   ├── ISSUE_TEMPLATE/        # Issue templates (bugs, features, lessons)
│   └── PULL_REQUEST_TEMPLATE.md
├── assets/
│   └── img/                   # Icons and images
├── content/
│   ├── lessons/               # 16 lesson markdown files (editable by writers)
│   ├── templates/             # Python script templates
│   ├── pages/                 # Welcome page, FAQ, etc.
│   ├── projects/              # Cookbook projects
│   └── lesson-registry.json   # Lesson metadata registry
├── dist/                      # Build output (TypeScript → JavaScript)
├── docs/                      # Technical documentation
│   ├── ARCHITECTURE.md        # System design and architecture
│   ├── TESTING.md             # Testing guide
│   └── PACKAGING.md           # Build and distribution
├── src/
│   ├── extension.ts           # Main extension entry point
│   ├── commands/              # Command implementations
│   ├── services/              # Business logic (environment, telemetry)
│   ├── types/                 # TypeScript interfaces
│   ├── views/                 # TreeView and Webview providers
│   ├── renderers/             # Markdown rendering
│   ├── telemetry/             # Device monitoring (Python + TypeScript)
│   └── webview/               # Frontend assets (CSS, JS)
├── test/                      # Automated test suite (134+ tests)
├── themes/                    # VSCode color themes
├── vendor/                    # Reference repos (NOT deployed, local only)
├── package.json               # Extension manifest and commands
├── tsconfig.json              # TypeScript configuration
└── README.md                  # User-facing documentation
```

### Generated Directories (User's Machine)

The extension creates these directories on the user's machine:

- **`~/tt-scratchpad/`** - Scripts created by lessons
- **`~/models/`** - Downloaded LLM models
- **`~/tt-metal/`** - TT-Metal repository (if cloned via lesson)
- **`~/tt-vllm-venv/`** - vLLM Python virtual environment
- **`~/tt-forge-venv/`** - TT-Forge Python virtual environment
- **`~/tt-xla-venv/`** - TT-XLA Python virtual environment

### Key Directories Explained

**`content/lessons/`** - Pure markdown lesson files
- Editable by technical writers without code knowledge
- Uses frontmatter for metadata (title, description, difficulty, etc.)
- Command buttons link to TypeScript commands via `command:` URIs
- Supports mermaid diagrams with Tenstorrent brand colors

**`src/commands/`** - Command implementations
- Each command registered in `package.json` has a handler here
- Commands create scripts, run terminals, manage environment
- Terminal strategy: `main` (setup/testing) and `server` (long-running)

**`src/telemetry/`** - Device monitoring system
- `telemetryReader.py` - Reads tt-smi JSON and extracts telemetry
- `TelemetryMonitor.ts` - Polls Python script and updates status bar
- Multi-device support with aggregate health status

**`test/`** - Automated test suite
- `lesson-tests/` - Markdown quality, frontmatter, command links
- `template-tests/` - Python template syntax validation
- `config-tests/` - Package.json validation
- See [docs/TESTING.md](docs/TESTING.md) for details

**`vendor/`** - Reference repositories (local only, NOT deployed)
- Contains clones of tt-metal, vllm, tt-xla, etc. for reference
- Used when authoring/updating lesson content
- Listed in `.gitignore` - each developer clones what they need
- **When authoring lessons: clone liberally, don't guess!**

---

## Development Workflow

### Making Changes

1. **Create a branch** (see [Git Workflow](#git-workflow))
2. **Make your changes**
   - Edit TypeScript in `src/`
   - Edit lessons in `content/lessons/`
   - Add tests in `test/`
3. **Test locally**
   ```bash
   npm run build     # Compile and copy content
   npm test          # Run all tests
   # Press F5 to test in Extension Development Host
   ```
4. **Commit changes** (see [Git Workflow](#git-workflow))
5. **Push and open PR** (use PR template)

### Common Development Tasks

**Adding a new command:**
1. Add command definition to `package.json` → `contributes.commands`
2. Implement handler in appropriate `src/commands/*.ts` file
3. Register command in `src/extension.ts` → `activate()`
4. Test in Extension Development Host (F5)

**Editing lesson content:**
1. Edit markdown in `content/lessons/*.md`
2. Test rendering: `npm run build` → F5 → Open lesson in extension
3. Verify command buttons work correctly
4. Update lesson metadata if changing hardware requirements

**Adding a Python template:**
1. Create template in `content/templates/`
2. Add SPDX header with Apache-2.0 license
3. Add test in `test/template-tests/templates.test.ts`
4. Create command to deploy template (see `createChatScript` as example)

### Hot Reload

**TypeScript changes:**
- Run `npm run watch` in terminal
- Press `Ctrl+R` in Extension Development Host to reload

**Content changes (lessons, templates):**
- Run `npm run build` to copy content to `dist/`
- Press `Ctrl+R` in Extension Development Host to reload

---

## Project Architecture

This extension follows a **content-first architecture** where lessons are pure markdown files and code handles execution.

### Key Architectural Principles

1. **Content-First Design** - Technical writers can edit lessons without touching code
2. **No Custom UI** - Uses VSCode's native TreeView and Webview APIs exclusively
3. **Terminal Integration** - Two-terminal strategy (`main` + `server`) for persistent sessions
4. **Hardware-Aware** - Auto-detects devices via tt-smi and adapts instructions
5. **No Bundling** - Extension ships with full `node_modules/` (bundlers break tree providers)

### Design Patterns

**Command Pattern:** Every user action is a VSCode command
- Declared in `package.json` for discoverability
- Implemented in `src/commands/` modules
- Registered in `src/extension.ts` activation

**Provider Pattern:** Lessons and preview use VSCode providers
- `LessonTreeProvider` - Sidebar tree view with hierarchical lessons
- `LessonWebviewManager` - Markdown rendering with mermaid diagrams
- `PreviewWebviewProvider` - Image/output preview panel

**Service Pattern:** Business logic separated from commands
- `EnvironmentManager` - Python venv detection and activation
- `TelemetryMonitor` - Device monitoring and health checks
- `MarkdownRenderer` - Lesson rendering with custom code blocks

### Data Flow

```
User clicks button in lesson
    ↓
Command URI invoked (command:tenstorrent.*)
    ↓
Command handler in src/commands/
    ↓
Service layer (if needed)
    ↓
Terminal command execution or VSCode API call
    ↓
UI update (status bar, webview, etc.)
```

**For complete technical documentation, see:**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Detailed system design
- **[docs/TESTING.md](docs/TESTING.md)** - Testing architecture
- **[docs/PACKAGING.md](docs/PACKAGING.md)** - Build and distribution

---

## Code Style and Standards

### TypeScript Standards

**File headers:** Every TypeScript file must include SPDX header:
```typescript
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

**Naming conventions:**
- Classes: `PascalCase` (e.g., `LessonTreeProvider`)
- Functions: `camelCase` (e.g., `updateDeviceStatus`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MODEL_REGISTRY`)
- Interfaces: `PascalCase` with `I` prefix optional (e.g., `DeviceInfo`)

**Code organization:**
- Keep functions focused and single-purpose
- Extract complex logic into service classes
- Use async/await for asynchronous operations
- Handle errors gracefully with try/catch

**Comments:**
- Add JSDoc comments for public functions and classes
- Explain "why" not "what" in inline comments
- Document non-obvious behavior or workarounds

### Python Standards

**File headers:** Every Python file must include SPDX header:
```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
```

**Style guide:**
- Follow PEP 8 conventions
- Use type hints where applicable
- Keep functions focused and well-documented

### Markdown Standards

**Lesson files:**
- Use frontmatter for metadata (title, description, category, etc.)
- Use proper heading hierarchy (# for title, ## for sections, ### for subsections)
- Include blank lines around code blocks and lists
- Test command buttons before committing

**Command button syntax:**
```markdown
command:tenstorrent.commandName
Button Text
```

### Formatting Tools

**Before committing:**
```bash
npm run lint       # ESLint for TypeScript (if configured)
npm run format     # Prettier for code formatting (if configured)
```

**VSCode settings:** Enable "Format on Save" for consistency

---

## Git Workflow

### Branch Naming

Include your username, issue number, and a brief description:

```bash
# Good examples:
git checkout -b username/123-add-new-lesson
git checkout -b username/456-fix-telemetry-bug
git checkout -b username-789_update-docs

# Avoid:
git checkout -b fix          # Too vague
git checkout -b issue-123    # Missing username
```

### Commit Messages

Use **conventional commit format** for clear history:

```
type(scope): brief description

Optional longer explanation if needed.
Link to issue or provide more context.

Fixes #123
```

**Types:**
- `feat` - New feature (e.g., new command, new lesson)
- `fix` - Bug fix
- `docs` - Documentation changes
- `style` - Code style changes (formatting, no logic changes)
- `refactor` - Code restructuring (no behavior changes)
- `test` - Test additions or updates
- `chore` - Build process, tooling, dependencies

**Examples:**
```bash
git commit -m "feat(lessons): add TT-Forge lesson with MLIR examples"
git commit -m "fix(telemetry): handle multi-device temperature aggregation"
git commit -m "docs(readme): update installation instructions for Ubuntu 24.04"
git commit -m "test(templates): add validation for particle life template"
```

### Opening a Pull Request

1. **Push your branch:**
   ```bash
   git push origin username/123-add-new-feature
   ```

2. **Create PR on GitHub** with:
   - **Title:** Clear, descriptive (use conventional commit format)
   - **Description:** Use PR template (auto-populated)
   - **Link to issue:** Use "Fixes #123" or "Closes #456"
   - **Screenshots/videos:** If UI changes are involved
   - **Test results:** Confirm all tests pass

3. **Fill out PR checklist:**
   - [ ] All tests pass
   - [ ] Extension builds successfully
   - [ ] Manually tested in Extension Development Host
   - [ ] Documentation updated (if needed)
   - [ ] CHANGELOG.md updated (for notable changes)
   - [ ] Version bumped in package.json (if needed)

4. **Request review** from maintainers

5. **Respond to feedback:**
   - Address all review comments
   - Push additional commits to same branch
   - Request re-review when ready

### Version Management

**CRITICAL:** Always increment version in `package.json` after ANY changes:

- **Patch (0.0.X):** Bug fixes, content updates, small changes
- **Minor (0.X.0):** New features, new lessons, significant updates
- **Major (X.0.0):** Breaking changes

**Why this matters:** VSCode caches extensions. Without version bump, users may see stale content even after updating.

**Rule:** After completing ANY change → increment version → rebuild → repackage

---

## Testing

The extension includes **134+ automated tests** covering markdown quality, Python templates, configuration validation, and more.

### Running Tests

**All tests:**
```bash
npm test              # Run complete test suite
npm run test:watch    # Watch mode for active development
```

**Specific test suites:**
```bash
npm run test:templates  # Test Python template syntax only
```

**Manual testing:**
1. Press `F5` to launch Extension Development Host
2. Test your changes in the new VSCode window
3. Check console for errors: Help → Toggle Developer Tools
4. Verify command buttons work from lessons
5. Test on actual hardware if modifying hardware-specific features

### Test Categories

**Markdown tests** (`test/lesson-tests/markdown-validation.test.ts`):
- Frontmatter validation (required fields, valid values)
- Command link format and uniqueness
- Heading structure and hierarchy
- Code block formatting
- Link validity
- Mermaid diagram syntax

**Template tests** (`test/template-tests/templates.test.ts`):
- Python syntax validation (compiles without errors)
- SPDX header presence
- Required imports
- File structure
- Documentation strings

**Configuration tests** (`test/config-tests/*.test.ts`):
- package.json validity
- Command registration completeness
- Lesson registry consistency

### Before Submitting a PR

**Checklist:**
- [ ] All tests pass: `npm test` (no failures)
- [ ] Extension builds: `npm run build` (no TypeScript errors)
- [ ] Extension packages: `npm run package` (creates .vsix)
- [ ] Manual testing complete (F5 → test your changes)
- [ ] Hardware testing (if applicable)
- [ ] No new warnings or errors in console
- [ ] Existing functionality still works

### Writing New Tests

When adding features, add tests:

**Example test structure:**
```typescript
describe('My New Feature', () => {
  it('should do something specific', () => {
    // Arrange
    const input = 'test data';
    
    // Act
    const result = myFunction(input);
    
    // Assert
    expect(result).to.equal('expected output');
  });
});
```

**See [docs/TESTING.md](docs/TESTING.md) for:**
- Complete testing guide
- Test patterns and examples
- CI/CD integration
- Troubleshooting test failures

---

## Documentation

### When to Update Documentation

**README.md:** Update for user-facing changes
- New features or commands
- Changed workflows or prerequisites
- Updated installation instructions
- New learning paths

**CHANGELOG.md:** Always update for notable changes
- New features
- Bug fixes
- Breaking changes
- Deprecations

**Lesson content:** Update when modifying lessons
- Fix typos or errors in `content/lessons/*.md`
- Update commands if APIs change
- Refresh screenshots if UI changes

**CONTRIBUTING.md:** Update for process changes
- New development tools
- Changed testing requirements
- Modified review process

**docs/ technical documentation:** Update for architectural changes
- `docs/ARCHITECTURE.md` - System design changes
- `docs/TESTING.md` - New test patterns or tools
- `docs/PACKAGING.md` - Build process changes

### Documentation Standards

**Markdown formatting:**
- Use proper heading hierarchy
- Include code blocks with language tags
- Add blank lines around lists and code blocks
- Link to other documentation where relevant

**Code comments:**
- Add JSDoc comments for public APIs
- Explain "why" not "what"
- Document workarounds or non-obvious behavior
- Keep comments up-to-date with code changes

---

## Adding New Lessons

### Lesson Creation Workflow

1. **Research:** Check `vendor/` for reference implementations
   ```bash
   cd vendor/
   # Clone/update relevant repos (tt-metal, vllm, tt-xla, etc.)
   git clone https://github.com/tenstorrent/[repo-name].git
   ```

2. **Create lesson file:**
   ```bash
   touch content/lessons/17-my-new-lesson.md
   ```

3. **Add frontmatter:**
   ```yaml
   ---
   title: "My New Lesson"
   description: "Learn how to..."
   category: "advanced"
   difficulty: "intermediate"
   estimatedTime: "30 minutes"
   prerequisites:
     - "Completed Hardware Detection lesson"
     - "Python 3.10+ installed"
   metadata:
     supportedHardware: ["n150", "n300", "t3k"]
     status: "draft"
     validatedOn: []
     minTTMetalVersion: "v0.51.0"
   ---
   ```

4. **Write lesson content:**
   - Clear introduction explaining what students will learn
   - Prerequisites section
   - Step-by-step instructions with command buttons
   - Code examples with explanations
   - Troubleshooting tips
   - "What's Next" section with related lessons

5. **Add command buttons:**
   ```markdown
   command:tenstorrent.myNewCommand
   Click to Run This Command
   ```

6. **Update lesson registry:**
   Edit `content/lesson-registry.json` to include new lesson.

7. **Implement commands:**
   - Add command definition to `package.json`
   - Implement handler in `src/commands/`
   - Register in `src/extension.ts`

8. **Test thoroughly:**
   - `npm run build && npm test`
   - Manual testing (F5)
   - Test on actual hardware
   - Verify all commands work

9. **Update metadata:**
   - Set `status: "validated"` after successful hardware testing
   - Add your hardware to `validatedOn` array
   - Update `supportedHardware` based on compatibility

### Lesson Content Guidelines

**Structure:**
```markdown
# Lesson Title

## Overview
Brief description (2-3 sentences)

## Prerequisites
- List required knowledge
- List required setup

## Learning Objectives
By completing this lesson, you will:
- Objective 1
- Objective 2

## Step 1: Introduction
Explanation of first concept...

command:tenstorrent.step1Command
Run Step 1

## Step 2: Next Concept
...

## Troubleshooting
Common issues and solutions...

## What's Next?
- Related Lesson A
- Related Lesson B
```

**Best practices:**
- Use clear, conversational language
- Explain concepts before showing commands
- Include expected output examples
- Add troubleshooting for common issues
- Link to external documentation where appropriate
- Test commands on all supported hardware configurations

**Hardware-specific content:**
Use markdown sections for hardware variants:
```markdown
### N150 Configuration
Instructions specific to N150...

### N300 Configuration
Instructions specific to N300...
```

**For detailed lesson authoring guidance, see:**
- Existing lessons in `content/lessons/`
- `CLAUDE.md` (internal development notes)
- Vendor repos in `vendor/` for reference implementations

---

## Packaging and Distribution

### Build Process

The extension uses a **three-stage build process**:

1. **Clean:** Remove old build artifacts
   ```bash
   npm run clean    # Removes dist/ directory
   ```

2. **Compile:** TypeScript → JavaScript
   ```bash
   npm run compile  # Runs tsc -p ./
   ```

3. **Copy Content:** Copy runtime assets to dist/
   ```bash
   npm run copy-content  # Copies lessons, templates, styles, etc.
   ```

**Combined build:**
```bash
npm run build    # All three stages: clean + compile + copy-content
```

### Creating a Package

**Package the extension:**
```bash
npm run package    # Creates tt-vscode-toolkit-X.X.X.vsix
```

**What gets packaged:**
- Compiled JavaScript in `dist/`
- Lesson content (markdown files)
- Python templates
- Styles and assets
- `node_modules/` (required - no bundling!)
- Package manifest (`package.json`)

**What's excluded** (via `.vscodeignore`):
- Source TypeScript files (`src/`)
- Tests (`test/`)
- Development tools
- Git files
- Vendor repos
- Documentation source files

### Package Size Optimization

**Current size:** ~31MB (2031 files)

**Optimization history:**
- v0.0.241: Excluded large .gif files (60MB → 31MB)
- Attempted bundling with esbuild (v0.0.125-126) → FAILED → Rolled back
- **DO NOT attempt bundling** - breaks tree providers and view registration

**To check package contents:**
```bash
unzip -l tt-vscode-toolkit-*.vsix | less
```

### Lesson Validation System

Lessons have metadata for quality tracking:

```json
"metadata": {
  "supportedHardware": ["n150", "n300", "t3k"],
  "status": "validated",     // "validated", "draft", or "blocked"
  "validatedOn": ["n150"],   // Hardware configs tested
  "minTTMetalVersion": "v0.51.0"
}
```

**Production builds:** Filter to `status: "validated"` lessons only
**Development builds:** Show all lessons (controlled by user setting)

### Distribution Workflow

1. **Version bump** in `package.json` (CRITICAL - always do this!)
2. **Update CHANGELOG.md** with changes
3. **Build and test:**
   ```bash
   npm run build
   npm test
   npm run package
   ```
4. **Test the .vsix:**
   ```bash
   code --install-extension tt-vscode-toolkit-*.vsix
   # Restart VSCode and test manually
   ```
5. **Create release on GitHub** with:
   - Tag matching version (e.g., `v0.0.248`)
   - Release notes (copy from CHANGELOG.md)
   - Attach .vsix file
6. **Announce on Discord** (optional, for major releases)

**For complete packaging documentation, see:**
- **[docs/PACKAGING.md](docs/PACKAGING.md)** - Detailed packaging guide

---

## Code Review Process

### Review Requirements

- **All PRs require at least one approval** from a maintainer
- **CI checks must pass** before merging
- **All review comments must be addressed** or discussed

### What Reviewers Look For

**Code quality:**
- Follows established patterns and conventions
- Includes appropriate error handling
- Has necessary tests
- Well-documented and commented

**Functionality:**
- Solves the stated problem
- Doesn't break existing features
- Handles edge cases
- Works on all supported platforms (if applicable)

**Documentation:**
- User-facing changes documented in README.md
- Technical changes documented in code comments
- CHANGELOG.md updated for notable changes

**Testing:**
- All tests pass
- New features have tests
- Manually tested in Extension Development Host
- Hardware tested (if applicable)

### Review Etiquette

**For authors:**
- Respond to feedback constructively
- Ask questions if feedback is unclear
- Keep PRs reasonably sized (easier to review)
- Update PR description if scope changes
- Be patient - reviews take time

**For reviewers:**
- Be respectful and constructive
- Explain the "why" behind feedback
- Distinguish between blocking issues and suggestions
- Acknowledge good work
- Respond in a reasonable timeframe

### After Approval

- **Squash and merge** (preferred) - Creates clean history
- **Regular merge** (if preserving commit history is important)
- **Delete branch** after merge (keeps repository clean)

---

## Community Guidelines

### Code of Conduct

All contributors must follow our [Code of Conduct](CODE_OF_CONDUCT.md).

**In summary:**
- Be respectful and inclusive
- Welcome newcomers
- Assume good intentions
- Give and receive feedback gracefully
- No harassment, discrimination, or trolling

**Enforcement:** Violations can be reported to ospo@tenstorrent.com

### Communication Channels

**GitHub Issues:**
- Bug reports
- Feature requests
- Lesson requests
- Technical discussions

**GitHub Pull Requests:**
- Code contributions
- Documentation improvements
- Technical reviews

**Discord ([join here](https://discord.gg/tenstorrent)):**
- Real-time help and discussions
- Community building
- Informal questions
- Announcements

**Best practices:**
- Search before posting (avoid duplicates)
- Be specific and provide context
- Include version information and error logs
- Tag appropriately for visibility
- Follow up on your issues/questions

---

## Getting Help

### For Development Questions

1. **Check existing documentation:**
   - This CONTRIBUTING.md file
   - [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
   - [docs/TESTING.md](docs/TESTING.md)
   - [docs/PACKAGING.md](docs/PACKAGING.md)

2. **Search existing issues:**
   - Someone may have asked before
   - Check closed issues too

3. **Ask on Discord:**
   - `#vscode-toolkit` channel (if it exists)
   - `#general` or `#help` channels
   - Real-time responses from community

4. **Open an issue:**
   - Use "Question" template
   - Provide context and what you've tried
   - Link to relevant code or documentation

### For Bug Reports or Feature Requests

Use our [issue templates](.github/ISSUE_TEMPLATE/):
- **Bug Report - Extension Code** - For extension functionality issues
- **Bug Report - Lesson Content** - For typos, outdated commands, etc.
- **Feature Request** - For new extension features
- **New Lesson Module Request** - For new educational content
- **Question** - For general questions

### Response Times

- **Critical bugs:** 24-48 hours
- **Regular issues:** 3-7 days
- **Feature requests:** Triaged within 2 weeks
- **Discord questions:** Often answered within hours (community-driven)

**Note:** Response times are best-effort. This is an open-source project maintained by volunteers and Tenstorrent staff.

---

## License

By contributing, you agree that your contributions will be licensed under the **Apache License 2.0**.

All source files must include the appropriate SPDX header (see [Code Style and Standards](#code-style-and-standards)).

---

## Additional Resources

### Internal Documentation
* **[README.md](README.md)** - User-facing documentation
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture
* **[docs/TESTING.md](docs/TESTING.md)** - Testing guide
* **[docs/PACKAGING.md](docs/PACKAGING.md)** - Build and distribution
* **[FAQ.md](FAQ.md)** - Troubleshooting guide
* **[CHANGELOG.md](CHANGELOG.md)** - Version history
* **[SECURITY.md](SECURITY.md)** - Security policy

### External Resources
* [VSCode Extension API](https://code.visualstudio.com/api) - Official extension development docs
* [TypeScript Documentation](https://www.typescriptlang.org/docs/) - Language reference
* [Tenstorrent Documentation](https://docs.tenstorrent.com/) - Hardware and software docs
* [Tenstorrent Discord](https://discord.gg/tenstorrent) - Community chat
* [Tenstorrent GitHub](https://github.com/tenstorrent) - Source repositories

### Related Projects
* [tt-metal](https://github.com/tenstorrent/tt-metal) - Core runtime and kernels
* [vLLM](https://github.com/tenstorrent/vllm) - High-performance LLM serving
* [tt-forge](https://github.com/tenstorrent/tt-forge) - MLIR compiler
* [tt-xla](https://github.com/tenstorrent/tt-xla) - XLA compiler plugin

---

**Thank you for contributing to the Tenstorrent VSCode Toolkit!** 🎉

Your efforts help make Tenstorrent hardware more accessible to developers worldwide. Whether you're fixing typos, adding lessons, implementing features, or helping others in Discord, every contribution matters.

*Questions about contributing? Join us on [Discord](https://discord.gg/tenstorrent) or open an issue!*
