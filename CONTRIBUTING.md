# Contributing to TT-VSCode-Toolkit

Thank you for your interest in contributing to the Tenstorrent VSCode Toolkit!

## Getting Started

If you are interested in making a contribution, please familiarize yourself with our technical contribution standards as set forth in this guide.

All contributions require:
* An issue
  * Please file a feature support request or bug report under [Issues](https://github.com/tenstorrent/tt-vscode-toolkit/issues) to discuss your contribution
* A pull request (PR)
  * Your PR must be approved by appropriate reviewers

## Development Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/tenstorrent/tt-vscode-toolkit.git
   cd tt-vscode-toolkit
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Open in VS Code:
   ```bash
   code .
   ```

4. Press `F5` to launch the Extension Development Host and test your changes

## Project Architecture

This extension follows a content-first architecture where lessons are pure markdown files and code handles execution.

**Key architectural principles:**
- **Content-First Design** - Technical writers can edit lessons without touching code
- **No Custom UI** - Uses VSCode's native TreeView and Webview APIs
- **Terminal Integration** - Two-terminal strategy (main + server) for persistent sessions
- **Hardware-Aware** - Auto-detects devices via tt-smi and adapts instructions

**For detailed technical documentation, see:**
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Complete architecture reference
  - Extension structure and file organization
  - Generated files (~/tt-scratchpad/, ~/models/, etc.)
  - Design principles with implementation details
  - Module breakdown with code examples
  - Data flow diagrams
  - Extension lifecycle

## Code Style and Standards

### File Structure

Every source file must have the appropriate SPDX header at the top:

**TypeScript files:**
```typescript
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
```

**Python files:**
```python
# SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
```

### Formatting

This project uses:
* ESLint for TypeScript linting
* Prettier for code formatting

Run the following before committing:
```bash
npm run lint
npm run format
```

## Git Workflow

### Creating a Branch

Include your username, the issue number, and optionally a description:

```bash
git checkout -b username/123-add-new-feature
git checkout -b username-123_fix-bug
```

### Commit Messages

Use descriptive commit messages that follow conventional commit format:
```
type(scope): brief description

Longer explanation if needed

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

### Opening a Pull Request

1. Push your branch:
   ```bash
   git push origin username/123-add-new-feature
   ```

2. Open a PR on GitHub with:
   * Clear title and description
   * Link to the related issue
   * Screenshots/videos if UI changes are involved
   * Test results if applicable

3. Ensure all CI checks pass

4. Request review from maintainers

## Testing

The extension includes 134+ automated tests covering markdown quality, Python templates, and configuration.

**Quick test commands:**
```bash
npm test              # Run all tests
npm run test:watch    # Run tests in watch mode
```

**Before submitting a PR, ensure:**
* All tests pass: `npm test`
* The extension builds without errors: `npm run compile`
* All existing functionality still works
* New features have been tested manually
* No TypeScript errors: `npm run check-types`

**For comprehensive testing documentation, see:**
- **[docs/TESTING.md](docs/TESTING.md)** - Complete testing guide
  - Test suite overview (134+ tests)
  - Running tests (all commands and options)
  - Test categories (markdown, templates, config, projects)
  - Writing new tests (patterns and examples)
  - CI/CD integration
  - Troubleshooting

## Documentation

* Update README.md if you add new features or change existing functionality
* Update lesson content in `content/lessons/` if modifying lessons
* Add JSDoc comments to new functions and classes
* Update CHANGELOG.md with notable changes

## Packaging and Distribution

The extension uses `vsce` (Visual Studio Code Extensions CLI) for packaging and distribution.

**Create a package:**
```bash
npm run build      # Build TypeScript + copy content
npm run package    # Create .vsix file
```

**Key packaging concepts:**
- **Build process** - TypeScript compilation + content copying to `dist/`
- **Package exclusions** - Development files excluded via `.vscodeignore`
- **Validation metadata** - Lessons tracked by hardware compatibility and validation status
- **Version management** - Semantic versioning (MAJOR.MINOR.PATCH)

**For detailed packaging documentation, see:**
- **[docs/PACKAGING.md](docs/PACKAGING.md)** - Complete packaging guide
  - Build process and scripts
  - Package structure (what's included/excluded)
  - Lesson validation system
  - Production vs development builds
  - Package size optimization
  - Distribution workflow
  - Version management
  - Troubleshooting

## Code Review Process

* PRs require at least one approval from a maintainer
* Address all review comments before merging
* Keep PRs focused and reasonably sized
* Be responsive to feedback

## Adding New Lessons

To add a new lesson:

1. Create a new markdown file in `content/lessons/`
2. Add appropriate frontmatter (title, description, difficulty, etc.)
3. Include command buttons using the established syntax
4. Update `content/lesson-registry.json` with the new lesson metadata
5. Test the lesson in the extension

Example lesson structure:
```markdown
# Lesson Title

## Overview
Brief description of what the lesson covers.

## Prerequisites
- Required knowledge
- Required setup

## Steps
1. Step one
   ```button:command:workbench.action.terminal.new
   Open New Terminal
   ```

2. Step two
   ...
```

## Community Guidelines

* Be respectful and inclusive
* Follow the [Code of Conduct](CODE_OF_CONDUCT.md)
* Ask questions if you're unsure
* Help others learn and grow

## Getting Help

* Open an issue for bugs or feature requests
* Join the [Tenstorrent Discord](https://discord.gg/tvhGzHQwaj) for discussions
* Check existing issues and documentation first

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## Additional Resources

### Internal Documentation
* **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** - Technical architecture and design principles
* **[docs/TESTING.md](docs/TESTING.md)** - Comprehensive testing guide
* **[docs/PACKAGING.md](docs/PACKAGING.md)** - Build and distribution workflow
* **[README.md](README.md)** - User-facing documentation
* **[FAQ.md](FAQ.md)** - Troubleshooting guide
* **[CHANGELOG.md](CHANGELOG.md)** - Version history

### External Resources
* [VS Code Extension API](https://code.visualstudio.com/api)
* [TypeScript Documentation](https://www.typescriptlang.org/docs/)
* [Tenstorrent Documentation](https://docs.tenstorrent.com/)
* [Tenstorrent Discord](https://discord.gg/tvhGzHQwaj)

Thank you for contributing to the Tenstorrent VSCode Toolkit! 🎉
