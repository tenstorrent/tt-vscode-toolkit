# Contributing to Tenstorrent VSCode Toolkit

Thank you for your interest in contributing to the Tenstorrent VSCode Toolkit! We welcome contributions from developers of all experience levels.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Making Changes](#making-changes)
  - [Code Changes](#code-changes)
  - [Adding New Lessons](#adding-new-lessons)
- [Code Review Process](#code-review-process)
- [Additional Resources](#additional-resources)

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

**Optional:**
- Tenstorrent hardware (for lesson validation)
- tt-metal installed (for testing lesson content)

### Quick Setup

**1. Fork the repository**

Visit https://github.com/tenstorrent/tt-vscode-toolkit and click "Fork" to create your own copy.

**2. Clone your fork**

```bash
git clone https://github.com/YOUR-USERNAME/tt-vscode-toolkit.git
cd tt-vscode-toolkit
```

**3. Install dependencies**

```bash
npm install
```

**4. Build the extension**

```bash
npm run build
```

**5. Open in VSCode**

```bash
code .
```

**6. Launch Extension Development Host**

Press `F5` in VSCode to launch the extension in a new window for testing.

---

## Project Structure

Understanding the codebase organization:

```
tt-vscode-toolkit/
├── .github/
│   ├── ISSUE_TEMPLATE/        # Issue templates
│   └── PULL_REQUEST_TEMPLATE.md
├── content/
│   ├── lessons/               # 16 lesson markdown files
│   ├── templates/             # Python script templates
│   ├── pages/                 # Welcome page, FAQ, etc.
│   └── lesson-registry.json   # Lesson metadata
├── src/
│   ├── extension.ts           # Main extension entry point
│   ├── commands/              # Command implementations
│   ├── services/              # Business logic
│   ├── views/                 # TreeView and Webview providers
│   └── telemetry/             # Device monitoring
├── test/                      # Automated test suite (134+ tests)
└── docs/                      # Technical documentation
```

**Key directories:**
- **`content/lessons/`** - Pure markdown lesson files (editable by writers)
- **`src/commands/`** - Command implementations (registered in package.json)
- **`test/`** - Automated tests (markdown, templates, config)
- **`vendor/`** - Reference repos (local only, NOT deployed)

---

## Making Changes

Choose the workflow that matches your contribution:

### Code Changes

Follow this workflow when modifying extension code, adding features, fixing bugs, or updating TypeScript/configuration.

#### 1. Create a Feature Branch

**External contributors (fork-based):**
```bash
git checkout -b username/123-fix-feature
```

**Internal developers (direct push):**
```bash
git checkout -b username/123-fix-feature
```

#### 2. Make Your Code Changes

**Common tasks:**

- **Adding a new command:**
  1. Add command definition to `package.json` → `contributes.commands`
  2. Implement handler in appropriate `src/commands/*.ts` file
  3. Register command in `src/extension.ts` → `activate()`

- **Editing TypeScript:**
  - Edit files in `src/`
  - Run `npm run watch` for hot reload during development
  - Follow TypeScript standards (see [Code Style](#code-style-and-standards))

- **Updating configuration:**
  - Edit `package.json` for extension manifest changes
  - Edit `tsconfig.json` for TypeScript compiler options

#### 3. Test Your Changes

**Run automated tests:**
```bash
npm test
```

You should see 134+ tests passing. Fix any failures before proceeding.

**Manual testing:**
1. Build: `npm run build`
2. Press `F5` to launch Extension Development Host
3. Test your changes in the new VSCode window
4. Check console for errors: Help → Toggle Developer Tools

**Test packaging (optional):**
```bash
npm run package
```

This verifies the extension can be packaged successfully.

#### 4. Follow Code Style and Standards

**TypeScript files must include SPDX header:**
```typescript
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

**Naming conventions:**
- Classes: `PascalCase` (e.g., `LessonTreeProvider`)
- Functions: `camelCase` (e.g., `updateDeviceStatus`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MODEL_REGISTRY`)

**Best practices:**
- Keep functions focused and single-purpose
- Add JSDoc comments for public APIs
- Use async/await for asynchronous operations
- Handle errors gracefully with try/catch

#### 5. Commit and Push

**Commit with conventional format:**
```bash
git commit -m "feat(commands): add device reset command

Adds new command to reset device state via tt-smi.
Includes error handling and user feedback.

Fixes #123"
```

**Commit types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation changes
- `test` - Test additions or updates
- `refactor` - Code restructuring
- `chore` - Build process, tooling

**Push your branch:**

External contributors:
```bash
git push origin username/123-fix-feature
```

Internal developers:
```bash
git push origin username/123-fix-feature
```

#### 6. Open a Pull Request

1. Navigate to GitHub and click "Create Pull Request"
2. Fill out the PR template completely
3. Link to related issue (e.g., "Fixes #123")
4. Request review from maintainers

**Version management:**
- Increment version in `package.json` (CRITICAL for all changes)
- Patch (0.0.X): Bug fixes, small changes
- Minor (0.X.0): New features
- Major (X.0.0): Breaking changes

#### 7. Address Review Feedback

- Respond to all comments
- Push additional commits to same branch
- Request re-review when ready

#### Testing Requirements

All code changes must include:
- [ ] All existing tests pass
- [ ] New features have tests
- [ ] Extension builds successfully
- [ ] Manually tested in Extension Development Host
- [ ] No new warnings or errors

**For detailed testing information, see [docs/TESTING.md](docs/TESTING.md)**

---

### Adding New Lessons

Follow this workflow when creating educational content, updating lesson markdown, or adding templates.

#### 1. Research and Plan

**Check reference implementations:**
```bash
cd vendor/
# Clone/update relevant repos for reference
git clone https://github.com/tenstorrent/tt-metal.git
git clone https://github.com/tenstorrent/vllm.git
```

**Plan your lesson:**
- What will students learn?
- What prerequisites are needed?
- What hardware configurations are supported?
- What's the estimated completion time?

#### 2. Create Lesson File

```bash
touch content/lessons/17-my-new-lesson.md
```

#### 3. Add Frontmatter

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

#### 4. Write Lesson Content

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
- Test commands on all supported hardware

#### 5. Add Command Buttons

Link to commands using this syntax:
```markdown
command:tenstorrent.myNewCommand
Click to Run This Command
```

#### 6. Update Lesson Registry

Edit `content/lesson-registry.json` to include your new lesson.

#### 7. Implement Commands (if needed)

If your lesson needs new commands:
1. Add command definition to `package.json`
2. Implement handler in `src/commands/`
3. Register in `src/extension.ts`

#### 8. Test on Hardware

**Build and test:**
```bash
npm run build
# Press F5 to test in Extension Development Host
```

**Test on actual hardware:**
- Verify all commands work
- Check expected outputs match
- Test on all supported hardware configurations
- Document any hardware-specific quirks

#### 9. Update Metadata

After successful hardware testing:
```yaml
metadata:
  status: "validated"
  validatedOn: ["n150", "n300"]
```

#### 10. Commit and Push

```bash
git add content/lessons/17-my-new-lesson.md content/lesson-registry.json
git commit -m "docs(lessons): add lesson on advanced topic

New lesson covers X, Y, and Z with hands-on examples.
Validated on N150 and N300 hardware.

Closes #456"
git push origin username/456-new-lesson
```

#### 11. Open Pull Request

Use the PR template and include:
- Description of what the lesson teaches
- Hardware configurations tested
- Any special prerequisites or considerations

#### Lesson Content Guidelines

**Hardware-specific content:**
Use markdown sections for variants:
```markdown
### N150 Configuration
Instructions specific to N150...

### N300 Configuration
Instructions specific to N300...
```

**Python templates:**
Must include SPDX header:
```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
```

**For detailed lesson authoring, see existing lessons in `content/lessons/`**

---

## Code Review Process

### Review Requirements

- **All PRs require at least one approval** from a maintainer
- **CI checks must pass** before merging
- **All review comments must be addressed** or discussed

### What Reviewers Look For

**Code quality:**
- Follows established patterns
- Includes appropriate error handling
- Has necessary tests
- Well-documented

**Functionality:**
- Solves the stated problem
- Doesn't break existing features
- Handles edge cases

**Documentation:**
- User-facing changes documented in README.md
- Technical changes documented in code comments
- CHANGELOG.md updated for notable changes

**Lesson content:**
- Clear and accurate
- Commands work on specified hardware
- Proper frontmatter and metadata
- Good pedagogical flow

### After Approval

- **Squash and merge** (preferred) - Creates clean history
- **Regular merge** (if preserving commit history is important)
- **Delete branch** after merge

---

## Additional Resources

### Documentation

* **[README.md](README.md)** - User-facing documentation
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture
* **[docs/TESTING.md](docs/TESTING.md)** - Testing guide
* **[docs/PACKAGING.md](docs/PACKAGING.md)** - Build and distribution
* **[COMMUNITY_GUIDELINES.md](COMMUNITY_GUIDELINES.md)** - Community standards and communication
* **[FAQ.md](FAQ.md)** - Troubleshooting guide
* **[CHANGELOG.md](CHANGELOG.md)** - Version history
* **[SECURITY.md](SECURITY.md)** - Security policy

### External Resources

* [VSCode Extension API](https://code.visualstudio.com/api) - Official extension development docs
* [TypeScript Documentation](https://www.typescriptlang.org/docs/) - Language reference
* [Tenstorrent Documentation](https://docs.tenstorrent.com/) - Hardware and software docs
* [Tenstorrent Discord](https://discord.gg/tenstorrent) - Community chat

### Related Projects

* [tt-metal](https://github.com/tenstorrent/tt-metal) - Core runtime and kernels
* [vLLM](https://github.com/tenstorrent/vllm) - High-performance LLM serving
* [tt-forge](https://github.com/tenstorrent/tt-forge) - MLIR compiler
* [tt-xla](https://github.com/tenstorrent/tt-xla) - XLA compiler plugin

---

## License

By contributing, you agree that your contributions will be licensed under the **Apache License 2.0**.

All source files must include the appropriate SPDX header.

---

**Thank you for contributing to the Tenstorrent VSCode Toolkit!** 🎉

Your efforts help make Tenstorrent hardware more accessible to developers worldwide.

*Questions? Check [COMMUNITY_GUIDELINES.md](COMMUNITY_GUIDELINES.md) or join us on [Discord](https://discord.gg/tenstorrent)!*
