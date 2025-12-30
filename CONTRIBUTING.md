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

Before submitting a PR, ensure:
* The extension builds without errors: `npm run compile`
* All existing functionality still works
* New features have been tested manually
* No TypeScript errors: `npm run check-types`

## Documentation

* Update README.md if you add new features or change existing functionality
* Update lesson content in `content/lessons/` if modifying lessons
* Add JSDoc comments to new functions and classes
* Update CHANGELOG.md with notable changes

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

* [VS Code Extension API](https://code.visualstudio.com/api)
* [TypeScript Documentation](https://www.typescriptlang.org/docs/)
* [Tenstorrent Documentation](https://docs.tenstorrent.com/)

Thank you for contributing to the Tenstorrent VSCode Toolkit! 🎉
