# Contributing to tt-animatediff

Thank you for your interest in contributing to tt-animatediff! We welcome contributions from the community.

## How to Contribute

### Reporting Bugs

If you find a bug, please report it using [GitHub Issues](https://github.com/tenstorrent/tt-animatediff/issues). When reporting a bug, please include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected behavior vs. actual behavior
- Your environment (OS, Python version, TTNN version, hardware — Blackhole board revision if relevant)
- Any relevant logs or error messages

### Suggesting Features

We welcome feature suggestions! Please open a [GitHub Issue](https://github.com/tenstorrent/tt-animatediff/issues) with:

- A clear description of the feature
- The use case or problem it solves
- Any implementation ideas you may have

### Submitting Pull Requests

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the project's coding standards
3. **Test your changes** — ensure `pytest tests/` passes
4. **Update documentation** if you're adding new features or changing behavior
5. **Commit your changes** with clear, descriptive commit messages
6. **Push to your fork** and submit a pull request to the `main` branch

### Pull Request Review Process

- Pull requests are typically reviewed **weekly**
- Maintainers will provide feedback on your submission
- Once approved, your PR will be merged by a maintainer

## Development Setup

### Prerequisites

- Python 3.9 or later
- A Tenstorrent Blackhole board (for hardware tests; CPU-only tests run without one)
- TTNN installed and on `PYTHONPATH`

### Installing

```bash
git clone https://github.com/tenstorrent/tt-animatediff.git
cd tt-animatediff
pip install -e ".[dev]"
```

### Running Tests

```bash
# CPU / mock tests (no hardware required)
pytest tests/test_pipeline.py

# Hardware tests (requires Blackhole board)
pytest tests/test_ttnn_pipeline.py
```

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Add SPDX headers to all new source files:
  ```python
  # SPDX-License-Identifier: Apache-2.0
  # SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
  ```
- Keep functions focused and well-documented

### Downloading Model Weights

```bash
bash weights/download_weights.sh
```

## Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to ospo@tenstorrent.com.

## Questions?

If you have questions about contributing, feel free to:

- Open a [GitHub Issue](https://github.com/tenstorrent/tt-animatediff/issues)
- Contact the maintainers at ospo@tenstorrent.com

## License

By contributing to tt-animatediff, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).
