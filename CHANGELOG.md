# Changelog

All notable changes to the Tenstorrent VSCode Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.268] - 2025-01-09

### Added
- Basic Docker support with Dockerfile for local code-server deployment
- docker-compose.yml and podman-compose.yml for container orchestration
- scripts/docker-entrypoint.sh for container initialization
- Docker build and run npm scripts
- Draft deployment lessons for Koyeb (marked as unvalidated)
- GitHub issue templates (bug, feature, question, lesson content)
- Pull request template with checklist
- GitHub Actions release workflow with automated build, test, and package
- Color-coded hardware badges (Green ✓: verified, Red ☠️: blocked, Yellow ?: untested)

### Changed
- **STATUS BAR CONSOLIDATION:** Merged two status bar items into one
- Status bar now shows: `🌡️ temp | ⚡ power | 🔊 MHz | device_config`
- All device info now comes from sysfs telemetry (non-invasive monitoring)
- Removed separate device count status bar item
- Retired Python environment status bar indicators (switching still available via command palette)
- Simplified device actions menu with sysfs as default monitoring
- Updated CODE_OF_CONDUCT, CONTRIBUTING, README, SECURITY for open-source
- Theme activation now uses standard configurationDefaults in package.json
- Moved .cleanup.sh → scripts/cleanup.sh for better organization

### Fixed
- Added `stroke` property to mermaid diagrams in CS Fundamentals lessons (5 files)
- Fixed ttnn API calls to use explicit `device_id=0` parameter in cookbook lessons
- Updated cookbook template README files for clarity
- Improved multi-device tooltip with per-device breakdown (temp, power, clock, PCI bus)

### Removed
- Old device status functions: parseDeviceInfo(), updateDeviceStatus(), updateStatusBarItem()
- Status update timer functions: startStatusUpdateTimer(), stopStatusUpdateTimer()
- Auto-update configuration options: configureUpdateInterval(), toggleAutoUpdate()
- statusBarItem global variable
- Programmatic theme setting code

## [0.0.243] - 2024-01-09

### Added
- New docs/ directory structure for technical documentation
- docs/ARCHITECTURE.md - Complete technical architecture reference
- docs/TESTING.md - Comprehensive testing guide
- docs/PACKAGING.md - Build and distribution workflow

### Changed
- Updated sidebar icon to monochrome symbol (tt_symbol_mono.svg)
- README.md completely revised (526 → 261 lines, 50% reduction)
- README.md preserves human voice while removing technical repetition
- CONTRIBUTING.md enhanced with links to new technical documentation

### Fixed
- Consolidated duplicate Quick Start sections in README
- Removed outdated version information from README

## [0.0.242] - 2024-01-08

### Added
- Full validation of CS Fundamentals series (7 modules) on QuietBox P300c
- docs/QB_RISCV_follows.md - Comprehensive validation documentation

### Fixed
- Module 1 executable path: `build/` → `build_Release/`
- Module 1 executable name: Added `metal_example_` prefix
- Updated expected output to match actual format in cs-fundamentals-01-computer.md
- Added note about firmware warnings and multi-device initialization

### Changed
- All CS Fundamentals modules validated and production-ready

## [0.0.233] - 2024-01-07

### Added
- Particle Life emergent complexity simulator (Recipe 5)
- New commands: `tenstorrent.createParticleLife`, `tenstorrent.runParticleLife`
- particle_life.py template (264 lines)
- test_particle_life.py demo script with visualization
- Cookbook now has 5 complete projects (was 4)

## [0.0.231] - 2024-01-06

### Added
- Temperature now visible directly in status bar (no hover needed)
- Multi-device temperature ranges (e.g., "4x P300 32-38°C")
- Monochrome SVG icon using `currentColor` for better theme compatibility

### Fixed
- TelemetryMonitor now handles multi-device format from v0.0.230
- Aggregates telemetry from all devices for right-side status bar
- Shows hottest device temp and total power across all devices
- Output Preview logo animation uses aggregated multi-device telemetry

### Changed
- Left status bar shows device count with temperature range
- Right status bar shows live telemetry (temp, power, frequency)

## [0.0.230] - 2024-01-05

### Added
- Multi-device telemetry support (detects ALL Tenstorrent devices)
- Status bar scales elegantly from 1 to 32+ devices
- Device Actions menu with per-device details (temp, power, PCI bus)
- Python telemetry reader returns array of all devices

### Changed
- Single device display: "✓ TT: P300" with temp/power in tooltip
- Multiple devices display: "✓ TT: 4x P300" with temperature range
- Aggregate health status (worst status wins)

## [0.0.225] - 2024-01-04

### Added
- Comprehensive mermaid diagram validation tests
- Stroke property validation test
- Node reference validation test

### Fixed
- Missing stroke properties in api-server.md (4 style statements)
- Missing stroke properties in tt-xla-jax.md (5 style statements)

### Changed
- 189 tests passing (18 pending for syntax tests - DOMPurify false positives)

### Removed
- Rolled back esbuild bundling attempt (v0.0.226) - caused activation failures

## [0.0.224] - 2024-01-03

### Added
- Comprehensive command argument handling for all button types
- Unified argument passing system (data-args with JSON)

### Changed
- Now supports lessonId, hardware, and any future argument types
- Backwards compatible with old message.lessonId format

### Fixed
- All command buttons throughout lessons now work correctly

## [0.0.223] - 2024-01-03

### Fixed
- Command buttons with URI-encoded arguments now parse correctly
- Added URL parsing to extract command ID and arguments separately

### Changed
- Superseded by v0.0.224 with more comprehensive fix

## [0.0.222] - 2024-01-02

### Fixed
- Added `stroke` property to mermaid style statements
- Mermaid v10 requires explicit stroke (border) color in styling

## [0.0.221] - 2024-01-02

### Changed
- Reverted to CDN for mermaid.js (debugging rendering issues from v0.0.220)
- Kept all fixes from v0.0.220 (custom code renderer, timing fixes)

## [0.0.220] - 2024-01-01

### Added
- Local mermaid.js bundling (removed CDN dependency)
- Mermaid v11.12.2 as devDependency
- DOMContentLoaded check for reliable diagram rendering
- Better privacy and offline support

### Fixed
- Stale content issue after installation (nuclear clean before rebuild)
- List rendering issues in vllm-production.md
- Code fence indentation problems

### Changed
- Improved list CSS (padding, margins, line-height)
- Package size: 59MB → 60.2MB (+1.2MB for local mermaid)

### Removed
- jsdelivr.net CDN dependency for mermaid.js

## [0.0.219] - 2023-12-31

### Added
- Mermaid.js diagram support in all lessons
- Comprehensive Tenstorrent stack diagram in Step Zero
- Learning paths flowchart decision tree
- Sequence diagram for Interactive Chat lesson
- MERMAID_EXAMPLES.md with usage guide and templates

### Changed
- Converted all ASCII diagrams to professional mermaid.js diagrams
- All diagrams use official Tenstorrent brand colors

## [0.0.207] - 2023-12-30

### Added
- Python environment status bar indicator for each terminal
- EnvironmentManager service tracks and activates environments per terminal
- 6 supported environments (TT-Metal, TT-Forge, TT-XLA, vLLM, API Server, Explore)
- New commands: `selectPythonEnvironment`, `refreshEnvironmentStatus`

### Changed
- All lessons now visible by default (`showUnvalidatedLessons: true`)
- Better visibility into available content

## [0.0.206] - 2023-12-29

### Added
- 7 cookbook execution commands (Game of Life variants, Mandelbrot, Audio, Image Filters)
- "Quick Start" buttons to all 4 cookbook project sections

### Changed
- Cookbook lesson now has easy button-based execution throughout

## [0.0.205] - 2023-12-28

### Added
- "Copy Demo to Scratchpad" command for image generation
- Step 6: Experiment with Code (Advanced) section in Image Generation lesson
- Literary and cultural references to image prompts (Steinbeck, Kerouac, WarGames)

### Changed
- Philosophy shift from button-pressing to code experimentation
- Package size: 5.43 MB (1949 files)

## [0.0.204] - 2023-12-27

### Added
- Prism.js syntax highlighting in code blocks
- Support for Python, Bash, JavaScript, TypeScript, JSON, YAML, Markdown, C++
- prism-tomorrow theme for VSCode-like dark highlighting

### Fixed
- Restored professional syntax highlighting with line numbers

### Changed
- Modified MarkdownRenderer for Prism.js-compatible code structure
- Package size: 5.42 MB (1942 files)

## [0.0.203] - 2023-12-26

### Added
- OpenMPI FAQ entry with LD_LIBRARY_PATH fix

### Fixed
- Missing dependencies in tt-coding-assistant.py template
- Added safetensors, termcolor, pytest installation instructions
- Added tt-transformers requirements.txt instructions

## [0.0.202] - 2023-12-25

### Changed
- Removed colored progress badges from lesson tree (cleaner UI)
- Progress still tracked internally

## [0.0.201] - 2023-12-24

### Added
- Action menu for `exploreProgrammingExamples` (Open in Terminal, Show in Explorer, Open Folder)
- Auto-configuration of Jupyter to use tt-metal Python environment

### Fixed
- `launchTtnnTutorials` now creates .vscode/settings.json with correct interpreter
- Jupyter notebooks no longer prompt for pyenv selection

## [0.0.126] - 2023-12-23

### Fixed
- Critical fix: `Error: Cannot find module 'marked'`
- Restored node_modules in package (dependencies must be included)
- Restored all 83 commands (all necessary for functionality)

### Changed
- Package size: 5.42 MB (1942 files) - includes all dependencies
- All tests passing (134/134)

## [0.0.125] - 2023-12-22

### Fixed
- "no data provider registered" error by cleaning stale files
- Added `clean` script to package.json

### Changed
- Package size reduced: 783.1 KB → 389.38 KB (50% reduction)
- File count reduced: 193 → 114 files

### Removed
- Rolled back in v0.0.126 due to missing dependencies

## [0.0.124] - 2023-12-21

### Changed
- Reduced commands from 83 → 77 (consolidated hardware variants)
- Command parameterization for hardware-specific operations

### Removed
- `startVllmServerN150/N300/T3K/P100` (4 commands - now parameterized)
- `startTtInferenceServerN150/N300` (2 commands - now parameterized)

## [0.0.102] - 2023-12-20

### Added
- Comprehensive "What's Next?" section in TT-XLA lesson
- Model bring-up checklist and workflow
- Learning resources section

### Changed
- Completely rewrote TT-XLA installation instructions (Ubuntu-specific)
- Changed from curl download to full tt-forge repo clone workflow

### Fixed
- Added `unset TT_METAL_HOME` and `unset LD_LIBRARY_PATH` steps
- Added `git submodule update --init --recursive`

## [0.0.101] - 2023-12-19

### Added
- Hardware auto-detection in vLLM starter script
- `detect_and_configure_hardware()` function
- Startup banner showing detected hardware
- Auto-sets MESH_DEVICE, TT_METAL_ARCH_NAME, TT_METAL_HOME

### Changed
- Simplified vLLM commands (no manual environment variable exports)
- Users can start vLLM with minimal command

## [0.0.100] - 2023-12-18

### Fixed
- `testChat()` command now properly opens VSCode chat panel
- Lesson 7 metadata: status changed to "validated", added N150 to validatedOn

### Changed
- Updated Lesson 8 to use Qwen3-0.6B throughout
- Simplified Lesson 8 commands using smart defaults

## [0.0.99] - 2023-12-17

### Added
- Smart defaults for vLLM starter script
- `inject_defaults()` function auto-configures parameters
- Auto-sets `--served-model-name` from model path
- "Quick Start" section in Lesson 7

### Changed
- Users can now use minimal vLLM command
- All defaults can be overridden explicitly

## [0.0.98] - 2023-12-16

### Fixed
- Multi-line code block rendering in VSCode walkthrough

### Changed
- Removed `<details>` HTML wrappers from hardware configurations
- Replaced with clean markdown headers
- Added `--served-model-name` parameter to vLLM commands

## [0.0.97] - 2023-12-15

### Added
- Step 7: Reasoning Showcase in vLLM lesson
- HF_MODEL auto-detection in start-vllm-server.py

### Changed
- Complete Lesson 7 rewrite centered on Qwen3-0.6B
- Updated all examples to use Qwen3-0.6B as primary N150 model

### Fixed
- Model recommendations (removed Gemma-2-2B-IT, added Gemma 3-1B-IT)

## [0.0.86] - 2023-12-14

### Added
- Lesson metadata system (hardware support, validation status)
- LESSON_METADATA.md with complete documentation
- Infrastructure for release gating and hardware filtering

### Fixed
- Added `sudo` prefix to all `install_dependencies.sh` commands
- Fixed emoji-based lists to use proper markdown syntax (9 lessons)

## [0.0.85] - 2023-12-13

### Added
- CSS-styled hardware configurations in Lessons 6, 7, 9, 12
- 4 new hardware-specific vLLM commands
- Template and styling guide
- Vendor directory documentation

## [0.0.84] - 2023-12-12

### Changed
- Previous stable version

---

## Version History Notes

- **v0.0.243** - Current version (documentation reorganization)
- **v0.0.242** - CS Fundamentals validation complete
- **v0.0.230-0.0.233** - Multi-device telemetry, Particle Life simulator
- **v0.0.219-0.0.225** - Mermaid.js diagrams, validation tests
- **v0.0.204-0.0.207** - Syntax highlighting, Python environment selector
- **v0.0.201-0.0.203** - Command UX improvements, dependency fixes
- **v0.0.126** - Critical dependency bundling fix
- **v0.0.97-0.0.102** - Qwen3-0.6B support, TT-XLA rewrite
- **v0.0.86** - Lesson metadata system introduced
- **v0.0.84-0.0.85** - Early hardware configuration support

For detailed git history, see: `git log --oneline`

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

For technical documentation:
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture details
- [docs/TESTING.md](docs/TESTING.md) - Testing guide
- [docs/PACKAGING.md](docs/PACKAGING.md) - Packaging workflow
