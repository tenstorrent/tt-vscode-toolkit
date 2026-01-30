# Changelog

All notable changes to the Tenstorrent VSCode Toolkit will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.0.282] - 2026-01-30

### Changed
- **Welcome Page Enhancement** - Added Discord community link to Resources section
  - `content/pages/welcome.html:435` - Added Discord link (https://discord.gg/tenstorrent) with description
  - Users can now easily find and join the Discord community for live support
- **Sidebar Category Rename** - Updated Welcome category name for better engagement
  - `content/lesson-registry.json:6` - Changed from "👋 Welcome to Tenstorrent" to "👋 Your journey begins here"
  - More inviting and user-friendly category title

---

## [0.0.281] - 2026-01-30

### Added
- **Style Guide** - Created comprehensive `docs/STYLE_GUIDE.md` documenting:
  - Terminology standards (product names, hardware models, file extensions)
  - Writing style guidelines (tone, technical accuracy, instructional structure)
  - Markdown formatting conventions (headers, lists, code blocks, links)
  - Command button formatting and grouping best practices
  - Code example patterns for Python, TypeScript, and Bash
  - Mermaid diagram color scheme and patterns (Tenstorrent brand colors)
  - Screenshot requirements and guidelines
- **Visual README Enhancement** - Added "Hands-On Cookbook Projects" section showcasing:
  - Game of Life GIF demonstration
  - Particle Life physics simulation
  - Mandelbrot set fractal rendering
  - Audio mel spectrogram processing
  - Visual table layout with descriptions for all cookbook projects
- **Docker Deployment Issue** - Created `.github/ISSUE_DOCKER_LATEST_TAG.md` documenting:
  - Problem: `:latest` tag not published to GHCR (blocks Koyeb deploy button)
  - Solution: Workflow changes needed to publish `:latest` tag
  - Blocker: GHCR repository currently not public
  - Future: Koyeb deploy button configuration and testing plan

### Fixed
- **CONTRIBUTING.md Duplicates** - Removed duplicate content throughout file:
  - Removed duplicate header (line 2)
  - Removed duplicate introduction and table of contents (lines 18-30)
  - Removed duplicate "Getting Started" section (lines 46-54)
  - Removed duplicate "Development Setup" section (lines 82-239)
  - Removed duplicate "Adding New Lessons" section (430+ lines)
  - Removed duplicate "Documentation" headers and security links
  - Removed duplicate "External Resources" and closing sections
  - File reduced from 768 lines to 511 lines (33% reduction)
- **Terminology Standardization** - Fixed inconsistent hardware naming in prose:
  - `content/lessons/video-generation-ttmetal.md:83` - Changed "(n150, n300, t3k, p100)" to "(N150, N300, T3K, P100)"
  - `content/lessons/tt-inference-server.md:548-549` - Capitalized hardware names in model compatibility list
  - `content/pages/FAQ.md:365-369` - Changed hardware names from lowercase to uppercase in prose descriptions
  - Note: Lowercase hardware names in YAML metadata, code, and command output remain correct

### Changed
- **Documentation Quality** - All files now follow consistent terminology per style guide
- **README.md** - Enhanced visual presentation with existing assets (GIFs and images)

### Technical Notes
- Style guide establishes "single source of truth" for terminology and formatting
- Hardware names: Uppercase in prose (N150, N300), lowercase in code/config
- Product names: "tt-metal" in code, "TT-Metal" in prose, "vLLM" consistently
- All changes maintain backward compatibility with existing lessons and code

---

## [0.0.280] - 2026-01-29

### Fixed
- **Device Count Display** - Fixed N150 showing "x5" in statusbar when only 1 device present
  - `readTelemetry()` now returns raw multi-device data instead of pre-aggregating
  - `updateTelemetry()` properly sets `this.currentMultiDeviceTelemetry`
  - Status bar correctly displays single device name without duplicate count suffix
  - `src/telemetry/TelemetryMonitor.ts:97` - Updated return type and logic

---

## [0.0.279] - 2026-01-29

### Fixed
- **Date Corrections:** Fixed incorrect years in changelog entries
  - Corrected v0.0.243 through v0.0.220: 2024-01-XX → 2025-01-XX
  - Corrected v0.0.219 through v0.0.84: 2023-12-XX → 2024-12-XX
  - Fixed README.md v0.0.243 date: 2024-01-09 → 2025-01-09
  - All version dates now accurately reflect the correct calendar year

---

## [0.0.276] - 2026-01-27

### Added
- **MOTD (Message of the Day) System** for terminal welcome messages
  - Created `content/motd.txt` with customizable welcome content
  - Created `scripts/show-motd.sh` for dynamic system information display
  - Displays Quick Start guide, essential commands, lesson links, and tips
  - Shows real-time system info: RAM, CPU cores, hardware detection, tt-metal status
  - Configured to display once per terminal session

### Changed
- **Deployment Lessons Simplified** - Now use published Docker images from GitHub Container Registry
  - `deploy-vscode-to-koyeb.md`: Reduced deployment time from 5-10 minutes to 60 seconds
    - Removed clone and build steps (Steps 3-4)
    - Single command deployment using `ghcr.io/tenstorrent/tt-vscode-toolkit:latest`
    - Status changed from "draft" to "validated"
    - Estimated time: 20 minutes → 5 minutes
    - Added "Advanced: Custom Builds" section for users who need customization
  - `deploy-to-koyeb.md`: Simplified Dockerfile examples to extend base image
    - vLLM Dockerfile reduced from ~60 lines to ~30 lines
    - Custom Inference Server reduced from ~80 lines to ~15 lines
    - All examples now use `FROM ghcr.io/tenstorrent/tt-vscode-toolkit:latest`
    - Status changed from "draft" to "validated"
    - Estimated time: 45 minutes → 10 minutes
- **README.md**: Added "Quick Start" section showing Docker local and Koyeb cloud deployment
- **Dockerfile**:
  - Configured terminal to use login shell (`bash -l`) for proper bashrc sourcing
  - Added MOTD file copying and script installation
  - Integrated MOTD system into bashrc configuration
- **docker-entrypoint.sh**: Simplified by removing old inline MOTD creation (180+ lines removed)

### Technical Notes
- MOTD system is modular: static content in `motd.txt` + dynamic info from `show-motd.sh`
- Terminal configuration ensures `.bashrc` is sourced on every new terminal
- Session flag (TENSTORRENT_MOTD_SHOWN) prevents duplicate displays
- Docker image build time: ~3 minutes, final size: 2.1 GB
- Published images available at `ghcr.io/tenstorrent/tt-vscode-toolkit:latest`

---

## [0.0.274] - 2026-01-27

### Added
- **HuggingFace CLI** (`hf`) installed in all Docker images for model downloads
- **Claude CLI** (`claude`) installed in Dockerfile and Dockerfile.full (not available in Koyeb due to base image constraints)
- **Docker improvements:** Added nodejs/npm for CLI tool support
- **Koyeb deployment support:** Successfully tested end-to-end deployment with N300 hardware

### Changed
- **docker-entrypoint.sh:** Skip tt-metal installation when `TT_METAL_PREBUILT=true` (for tt-metalium base image)
- **Dockerfile.koyeb:** Optimized for tt-metalium base image, HuggingFace CLI only
- **CLI tool verification:** Entrypoint now checks for `hf` command (not `huggingface-cli`)
- **deploy-vscode-to-koyeb.md:** Updated to document available CLI tools and limitations

### Fixed
- **PEP 668 compliance:** Added `--break-system-packages` flag to pip3 install commands in Dockerfiles (safe for containers)
- **Koyeb deployment errors:** Fixed tt-metal installation loop by detecting pre-built environment
- **npm installation:** Added nodejs/npm to apt-get install for Claude CLI support

### Technical Notes
- Koyeb deployment uses tt-metalium base image (pre-built tt-metal dependencies)
- tt-metal Python packages still require setup via extension lessons (quick version)
- Optional: Can pre-build tt-metal in Dockerfile.koyeb for instant readiness (15-25 min build time)
- Successfully tested with N300 hardware access, tt-smi working, HuggingFace CLI operational

---
## [0.0.271] - 2026-01-27

### Changed
- verify-installation lesson now documents both installation paradigms:
  - Pre-installed (tt-installer) - production approach
  - Manual build from source - development approach
- Added critical Python version matching requirement to build documentation
- Documented OpenMPI ULFM library path requirement (LD_LIBRARY_PATH)
- Enhanced git submodule initialization instructions

### Fixed
- Clarified Python version consistency requirement (build vs runtime must match)
- Documented separation of pre-installed and built tt-metal environments
- Added explicit git submodule initialization as critical step

## [0.0.270] - 2026-01-27

### Changed
- Updated lesson-registry.json to include P300C validation status
  - hardware-detection: Added "p300" to supportedHardware and validatedOn
  - verify-installation: Added "p300" to supportedHardware and validatedOn
- Extended supported hardware coverage for P300C Blackhole devices

## [0.0.269] - 2026-01-23

### Fixed
- **CI Build Failure:** Downgraded `chai` from v6.2.1 to v4.5.0 to fix ESM import errors in GitHub Actions
  - chai v6.x is ESM-only and incompatible with CommonJS test configuration
  - Also downgraded `@types/chai` from v5.2.3 to v4.3.20
  - All 315 tests now passing in CI environment
- **Status Bar Consolidation:** Completed merge of device monitoring into single telemetry status bar
  - Removed duplicate device count display
  - Unified all hardware info under TelemetryMonitor

## [0.0.268] - 2026-01-09

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

## [0.0.243] - 2026-01-09

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

## [0.0.242] - 2026-01-08

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

## [0.0.233] - 2026-01-07

### Added
- Particle Life emergent complexity simulator (Recipe 5)
- New commands: `tenstorrent.createParticleLife`, `tenstorrent.runParticleLife`
- particle_life.py template (264 lines)
- test_particle_life.py demo script with visualization
- Cookbook now has 5 complete projects (was 4)

## [0.0.231] - 2026-01-06

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

## [0.0.230] - 2026-01-05

### Added
- Multi-device telemetry support (detects ALL Tenstorrent devices)
- Status bar scales elegantly from 1 to 32+ devices
- Device Actions menu with per-device details (temp, power, PCI bus)
- Python telemetry reader returns array of all devices

### Changed
- Single device display: "✓ TT: P300" with temp/power in tooltip
- Multiple devices display: "✓ TT: 4x P300" with temperature range
- Aggregate health status (worst status wins)

## [0.0.225] - 2026-01-04

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

## [0.0.224] - 2026-01-03

### Added
- Comprehensive command argument handling for all button types
- Unified argument passing system (data-args with JSON)

### Changed
- Now supports lessonId, hardware, and any future argument types
- Backwards compatible with old message.lessonId format

### Fixed
- All command buttons throughout lessons now work correctly

## [0.0.223] - 2026-01-03

### Fixed
- Command buttons with URI-encoded arguments now parse correctly
- Added URL parsing to extract command ID and arguments separately

### Changed
- Superseded by v0.0.224 with more comprehensive fix

## [0.0.222] - 2026-01-02

### Fixed
- Added `stroke` property to mermaid style statements
- Mermaid v10 requires explicit stroke (border) color in styling

## [0.0.221] - 2026-01-02

### Changed
- Reverted to CDN for mermaid.js (debugging rendering issues from v0.0.220)
- Kept all fixes from v0.0.220 (custom code renderer, timing fixes)

## [0.0.220] - 2026-01-01

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

## [0.0.219] - 2025-12-31

### Added
- Mermaid.js diagram support in all lessons
- Comprehensive Tenstorrent stack diagram in Step Zero
- Learning paths flowchart decision tree
- Sequence diagram for Interactive Chat lesson
- MERMAID_EXAMPLES.md with usage guide and templates

### Changed
- Converted all ASCII diagrams to professional mermaid.js diagrams
- All diagrams use official Tenstorrent brand colors

## [0.0.207] - 2025-12-30

### Added
- Python environment status bar indicator for each terminal
- EnvironmentManager service tracks and activates environments per terminal
- 6 supported environments (TT-Metal, TT-Forge, TT-XLA, vLLM, API Server, Explore)
- New commands: `selectPythonEnvironment`, `refreshEnvironmentStatus`

### Changed
- All lessons now visible by default (`showUnvalidatedLessons: true`)
- Better visibility into available content

## [0.0.206] - 2025-12-29

### Added
- 7 cookbook execution commands (Game of Life variants, Mandelbrot, Audio, Image Filters)
- "Quick Start" buttons to all 4 cookbook project sections

### Changed
- Cookbook lesson now has easy button-based execution throughout

## [0.0.205] - 2025-12-28

### Added
- "Copy Demo to Scratchpad" command for image generation
- Step 6: Experiment with Code (Advanced) section in Image Generation lesson
- Literary and cultural references to image prompts (Steinbeck, Kerouac, WarGames)

### Changed
- Philosophy shift from button-pressing to code experimentation
- Package size: 5.43 MB (1949 files)

## [0.0.204] - 2025-12-27

### Added
- Prism.js syntax highlighting in code blocks
- Support for Python, Bash, JavaScript, TypeScript, JSON, YAML, Markdown, C++
- prism-tomorrow theme for VSCode-like dark highlighting

### Fixed
- Restored professional syntax highlighting with line numbers

### Changed
- Modified MarkdownRenderer for Prism.js-compatible code structure
- Package size: 5.42 MB (1942 files)

## [0.0.203] - 2025-12-26

### Added
- OpenMPI FAQ entry with LD_LIBRARY_PATH fix

### Fixed
- Missing dependencies in tt-coding-assistant.py template
- Added safetensors, termcolor, pytest installation instructions
- Added tt-transformers requirements.txt instructions

## [0.0.202] - 2025-12-25

### Changed
- Removed colored progress badges from lesson tree (cleaner UI)
- Progress still tracked internally

## [0.0.201] - 2025-12-24

### Added
- Action menu for `exploreProgrammingExamples` (Open in Terminal, Show in Explorer, Open Folder)
- Auto-configuration of Jupyter to use tt-metal Python environment

### Fixed
- `launchTtnnTutorials` now creates .vscode/settings.json with correct interpreter
- Jupyter notebooks no longer prompt for pyenv selection

## [0.0.126] - 2025-12-23

### Fixed
- Critical fix: `Error: Cannot find module 'marked'`
- Restored node_modules in package (dependencies must be included)
- Restored all 83 commands (all necessary for functionality)

### Changed
- Package size: 5.42 MB (1942 files) - includes all dependencies
- All tests passing (134/134)

## [0.0.125] - 2025-12-22

### Fixed
- "no data provider registered" error by cleaning stale files
- Added `clean` script to package.json

### Changed
- Package size reduced: 783.1 KB → 389.38 KB (50% reduction)
- File count reduced: 193 → 114 files

### Removed
- Rolled back in v0.0.126 due to missing dependencies

## [0.0.124] - 2025-12-21

### Changed
- Reduced commands from 83 → 77 (consolidated hardware variants)
- Command parameterization for hardware-specific operations

### Removed
- `startVllmServerN150/N300/T3K/P100` (4 commands - now parameterized)
- `startTtInferenceServerN150/N300` (2 commands - now parameterized)

## [0.0.102] - 2025-12-20

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

## [0.0.101] - 2025-12-19

### Added
- Hardware auto-detection in vLLM starter script
- `detect_and_configure_hardware()` function
- Startup banner showing detected hardware
- Auto-sets MESH_DEVICE, TT_METAL_ARCH_NAME, TT_METAL_HOME

### Changed
- Simplified vLLM commands (no manual environment variable exports)
- Users can start vLLM with minimal command

## [0.0.100] - 2025-12-18

### Fixed
- `testChat()` command now properly opens VSCode chat panel
- Lesson 7 metadata: status changed to "validated", added N150 to validatedOn

### Changed
- Updated Lesson 8 to use Qwen3-0.6B throughout
- Simplified Lesson 8 commands using smart defaults

## [0.0.99] - 2025-12-17

### Added
- Smart defaults for vLLM starter script
- `inject_defaults()` function auto-configures parameters
- Auto-sets `--served-model-name` from model path
- "Quick Start" section in Lesson 7

### Changed
- Users can now use minimal vLLM command
- All defaults can be overridden explicitly

## [0.0.98] - 2025-12-16

### Fixed
- Multi-line code block rendering in VSCode walkthrough

### Changed
- Removed `<details>` HTML wrappers from hardware configurations
- Replaced with clean markdown headers
- Added `--served-model-name` parameter to vLLM commands

## [0.0.97] - 2025-12-15

### Added
- Step 7: Reasoning Showcase in vLLM lesson
- HF_MODEL auto-detection in start-vllm-server.py

### Changed
- Complete Lesson 7 rewrite centered on Qwen3-0.6B
- Updated all examples to use Qwen3-0.6B as primary N150 model

### Fixed
- Model recommendations (removed Gemma-2-2B-IT, added Gemma 3-1B-IT)

## [0.0.86] - 2025-12-14

### Added
- Lesson metadata system (hardware support, validation status)
- LESSON_METADATA.md with complete documentation
- Infrastructure for release gating and hardware filtering

### Fixed
- Added `sudo` prefix to all `install_dependencies.sh` commands
- Fixed emoji-based lists to use proper markdown syntax (9 lessons)

## [0.0.85] - 2025-12-13

### Added
- CSS-styled hardware configurations in Lessons 6, 7, 9, 12
- 4 new hardware-specific vLLM commands
- Template and styling guide
- Vendor directory documentation

## [0.0.84] - 2025-12-12

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
