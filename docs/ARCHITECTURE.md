# Extension Architecture

This document provides detailed technical architecture for contributors to the Tenstorrent VSCode Toolkit.

## Table of Contents

- [Extension Structure](#extension-structure)
- [Generated Files](#generated-files)
- [Design Principles](#design-principles)
- [Module Breakdown](#module-breakdown)
- [Data Flow](#data-flow)
- [Extension Lifecycle](#extension-lifecycle)

---

## Extension Structure

```
tt-vscode-toolkit/
├── content/
│   ├── lessons/          # 16 markdown lesson files
│   ├── templates/        # Python script templates (40+ templates)
│   ├── pages/            # Welcome page, FAQ templates
│   ├── projects/         # Cookbook projects (Game of Life, Fractals, etc.)
│   └── lesson-registry.json  # Lesson metadata and categories
├── src/
│   ├── commands/         # Terminal command definitions
│   │   └── terminalCommands.ts  # All executable commands
│   ├── config/           # Model registry and shared config
│   │   └── modelRegistry.ts     # Centralized model configurations
│   ├── renderers/        # Markdown and command button renderers
│   │   ├── MarkdownRenderer.ts  # Markdown → HTML with mermaid support
│   │   └── index.ts
│   ├── state/            # Progress tracking and state management
│   │   ├── ProgressTracker.ts   # Lesson progress and analytics
│   │   └── index.ts
│   ├── types/            # TypeScript types and interfaces
│   │   ├── index.ts             # Lesson metadata, command types
│   │   └── webviewTypes.ts      # Webview message types
│   ├── utils/            # Lesson registry utilities
│   │   ├── LessonRegistry.ts    # Lesson loading and filtering
│   │   └── index.ts
│   ├── views/            # Tree view and webview managers
│   │   ├── LessonTreeProvider.ts     # Sidebar lesson tree
│   │   ├── LessonWebviewManager.ts   # Lesson content rendering
│   │   ├── ImagePreviewProvider.ts   # Output preview panel
│   │   └── index.ts
│   ├── webview/          # Webview resources
│   │   ├── scripts/
│   │   │   └── lesson-viewer.js      # Client-side lesson interaction
│   │   └── styles/
│   │       └── lesson-theme.css      # Theme-aware lesson styling
│   ├── telemetry/        # Usage analytics (optional)
│   │   └── telemetry.py            # Privacy-focused telemetry
│   └── extension.ts      # Main extension entry point
├── test/
│   └── lesson-tests/     # Automated validation tests
│       ├── markdown-validation.test.ts  # 96 tests for markdown quality
│       ├── templates.test.ts            # Python template validation
│       ├── config-extraction.test.ts    # Model registry tests
│       └── mermaid-validation.test.ts   # Mermaid diagram syntax
├── vendor/               # Reference repos (NOT deployed)
│   ├── tt-metal/         # Main reference: demos, APIs, examples
│   ├── vllm/             # Production inference patterns
│   ├── tt-xla/           # JAX/TT-XLA examples
│   ├── tt-forge-fe/      # TT-Forge reference
│   ├── tt-inference-server/  # Production deployment
│   ├── tt-installer/     # Installation workflows
│   └── ttsim/            # Simulator reference
├── assets/               # Images and icons
│   └── img/
│       ├── tt_symbol_mono.svg       # Sidebar icon
│       ├── tt_symbol_purple.svg
│       └── *.png                    # Lesson images
├── themes/               # VSCode color themes
│   ├── tenstorrent-theme.json       # Dark theme
│   └── tenstorrent-light-theme.json # Light theme
├── dist/                 # Compiled output (gitignored)
├── docs/                 # Documentation
├── package.json          # Extension manifest + configuration
├── tsconfig.json         # TypeScript configuration
├── .vscodeignore         # Package exclusion rules
├── FAQ.md                # User troubleshooting guide
├── CONTRIBUTING.md       # Developer guide
├── CHANGELOG.md          # Version history
└── README.md             # Main documentation
```

---

## Generated Files

The extension creates files in the user's home directory:

### ~/tt-scratchpad/

All generated scripts for user customization:

```
~/tt-scratchpad/
├── tt-chat-direct.py              # Direct API chat (Generator API)
├── tt-api-server-direct.py        # Direct API Flask server
├── tt-forge-classifier.py         # TT-Forge image classification
├── start-vllm-server.py           # vLLM production server starter
├── tt-coding-assistant.py         # AI coding assistant
├── tt-xla-gpt2-demo.py            # TT-XLA GPT-2 demo
├── particle-life/                  # Particle Life project
│   ├── particle_life.py
│   ├── requirements.txt
│   └── README.md
└── ... (more generated projects)
```

### ~/models/

Downloaded models from HuggingFace:

```
~/models/
├── Llama-3.1-8B-Instruct/
│   ├── original/                  # Meta format
│   └── ...                        # HuggingFace format
├── Qwen3-0.6B/
└── ...
```

### ~/tt-vllm/

vLLM repository (cloned by lessons):

```
~/tt-vllm/
├── vllm/
├── examples/
└── ...
```

### ~/tt-metal/

TT-Metal repository (cloned by lessons):

```
~/tt-metal/
├── tt_metal/
├── models/
├── tests/
└── ...
```

---

## Design Principles

### 1. Content-First Architecture

**Principle:** Content creators should be able to edit lessons without touching code.

**Implementation:**
- Lessons are pure markdown files in `content/lessons/`
- Metadata in `content/lesson-registry.json`
- Command buttons use simple markdown link syntax: `[Text](command:commandId)`
- No JSX, no HTML templates, no custom syntax

**Benefits:**
- Technical writers can contribute without knowing TypeScript
- Easy to review changes (just markdown diffs)
- Content versioning separate from code

### 2. No Custom UI

**Principle:** Use VSCode's native APIs instead of building custom UI.

**Implementation:**
- Sidebar uses `TreeView` API
- Lesson content uses `Webview` API
- Commands use `Terminal` API
- Progress uses built-in status bar

**Benefits:**
- Consistent with VSCode look and feel
- Automatic theme support (dark/light modes)
- Less maintenance burden
- Smaller package size

### 3. Terminal Integration

**Principle:** Run commands in persistent terminals, not hidden processes.

**Implementation:**
- Two-terminal strategy: `main` (setup/testing) and `server` (long-running)
- Reuse existing terminals instead of creating new ones
- Environment variables persist across lesson steps
- Users can see all output and interact if needed

**Benefits:**
- Transparency - users see what's happening
- Easy debugging - users can inspect errors
- Educational - users learn actual commands
- No process management complexity

### 4. Stateless Commands

**Principle:** Each command should work independently when possible.

**Implementation:**
- Commands check prerequisites before running
- Error messages guide users to missing setup
- State stored in filesystem (~/tt-scratchpad), not in memory
- Progress tracking is advisory, not enforced

**Benefits:**
- Users can jump between lessons
- Resilient to extension restarts
- Easy to test commands individually
- Flexible learning paths

### 5. Hardware-Aware

**Principle:** Detect hardware and adjust instructions automatically.

**Implementation:**
- `tt-smi -s` JSON output parsed for hardware type
- Commands adapt to N150, N300, T3K, P100, P150, Galaxy
- Lessons show hardware compatibility in metadata
- Filter tree view by detected hardware (optional)

**Benefits:**
- Users see only relevant content
- No confusion about incompatible configurations
- Smooth experience across hardware variants

### 6. Validation-Aware

**Principle:** Track lesson quality and validation status.

**Implementation:**
- Lesson metadata includes `status: validated | draft | blocked`
- `validatedOn: []` array tracks tested hardware
- `supportedHardware: []` declares compatibility
- Filter tree view by validation status

**Benefits:**
- Production builds ship only validated content
- Clear visibility of what's been tested
- Easy to identify gaps in testing coverage
- Development builds can show experimental content

---

## Module Breakdown

### src/extension.ts

**Main extension entry point.**

Key responsibilities:
- Register all commands
- Initialize tree view providers
- Set up webview managers
- Configure device detection
- Handle activation events

Key code:
```typescript
export function activate(context: vscode.ExtensionContext) {
  // Initialize registries and managers
  const lessonRegistry = new LessonRegistry(...);
  const progressTracker = new ProgressTracker(...);

  // Set up tree view
  const treeProvider = new LessonTreeProvider(...);
  vscode.window.registerTreeDataProvider('tenstorrentLessons', treeProvider);

  // Register all commands
  context.subscriptions.push(
    vscode.commands.registerCommand('tenstorrent.showLesson', ...)
  );

  // Set up device monitoring
  updateDeviceStatus();
}
```

### src/renderers/MarkdownRenderer.ts

**Converts markdown to HTML with special features.**

Key features:
- GitHub Flavored Markdown (via marked.js)
- Command button rendering: `[Text](command:id)` → `<button>`
- Mermaid diagram support (v11)
- Prism.js syntax highlighting
- XSS protection via DOMPurify
- Theme-aware styling

Key code:
```typescript
export class MarkdownRenderer {
  async render(markdown: string): Promise<RenderedMarkdown> {
    // Parse markdown with custom renderers
    let html = await marked.parse(markdown);

    // Sanitize while preserving mermaid blocks
    html = this.sanitizeWithMermaidPreservation(html);

    // Extract command IDs for registration
    const commands = this.extractCommands(html);

    return { html, frontmatter, commands };
  }
}
```

### src/views/LessonWebviewManager.ts

**Manages lesson webview lifecycle.**

Key features:
- Webview creation and disposal
- HTML generation with CSP
- Message passing (webview ↔ extension)
- Command execution from buttons
- Progress tracking integration
- Theme change handling

Key code:
```typescript
export class LessonWebviewManager {
  async showLesson(lessonId: string) {
    // Render markdown to HTML
    const rendered = await renderer.renderFile(lessonPath);

    // Generate full HTML with scripts and styles
    const html = this.generateHTML(lesson, rendered.html, cssUri, jsUri);

    // Set webview content
    this.panel.webview.html = html;

    // Track progress
    this.progressTracker.startSession(lessonId);
  }
}
```

### src/views/LessonTreeProvider.ts

**Provides sidebar tree view of lessons.**

Key features:
- Hierarchical lesson organization
- Category grouping
- Hardware filtering (optional)
- Validation status filtering
- Progress indicators
- Context menu actions

Key code:
```typescript
export class LessonTreeProvider implements vscode.TreeDataProvider<TreeItem> {
  getChildren(element?: TreeItem): TreeItem[] {
    if (!element) {
      // Return top-level categories
      return this.getCategories();
    } else if (element.type === 'category') {
      // Return lessons in category
      return this.getLessonsForCategory(element.id);
    }
  }
}
```

### src/commands/terminalCommands.ts

**Defines all executable commands.**

Key features:
- 83 commands for hardware detection, setup, inference, etc.
- Terminal type selection (main vs server)
- Environment variable setup
- Hardware-specific configurations
- Error handling and validation

Key code:
```typescript
export async function runHardwareDetection(context: vscode.ExtensionContext) {
  await executeInTerminal(
    'tt-smi -s',
    'main',
    'Hardware detection'
  );

  // Update statusbar
  await updateDeviceStatus();
}
```

### src/utils/LessonRegistry.ts

**Loads and manages lesson metadata.**

Key features:
- Load lesson-registry.json
- Filter by hardware
- Filter by validation status
- Category management
- Prerequisite tracking

Key code:
```typescript
export class LessonRegistry {
  getAll(): LessonMetadata[] {
    return this.lessons.filter(lesson => {
      // Apply hardware filter
      if (this.hardwareFilter &&
          !lesson.supportedHardware.includes(this.hardwareFilter)) {
        return false;
      }

      // Apply validation filter
      if (!this.showUnvalidated &&
          lesson.status !== 'validated') {
        return false;
      }

      return true;
    });
  }
}
```

---

## Data Flow

### Lesson Loading Flow

```
1. Extension activates
2. LessonRegistry loads lesson-registry.json
3. LessonTreeProvider requests categories
4. Tree view displays in sidebar
5. User clicks lesson
6. showLesson command fires
7. LessonWebviewManager creates webview
8. MarkdownRenderer converts lesson markdown → HTML
9. Webview displays rendered lesson
10. User clicks command button
11. Webview posts message to extension
12. Extension executes command in terminal
13. ProgressTracker records action
```

### Command Execution Flow

```
1. User clicks command button in webview
2. lesson-viewer.js (client) sends message:
   { type: 'executeCommand', command: 'commandId', args: {...} }
3. LessonWebviewManager receives message
4. Extension calls vscode.commands.executeCommand(commandId, args)
5. Command handler in src/commands/terminalCommands.ts runs
6. executeInTerminal() creates or reuses terminal
7. Command runs in terminal (visible to user)
8. ProgressTracker.recordCommandExecution() called
9. Progress saved to globalState
```

### Hardware Detection Flow

```
1. Extension activates or user runs detection command
2. Execute 'tt-smi -s' command
3. Parse JSON output
4. Extract: board_type, coords, arch, telemetry_device
5. Map to user-friendly names (N150, N300, T3K, etc.)
6. Update statusbar item
7. Update device context for filtering
```

---

## Extension Lifecycle

### Activation

**Triggers:**
- `onStartupFinished` - After VSCode window opens
- `onView:tenstorrentLessons` - When sidebar first accessed
- `onWalkthrough:tenstorrent.setup` - When walkthrough opened

**Activation sequence:**
1. Load configuration
2. Initialize lesson registry
3. Set up tree providers
4. Register all commands
5. Start device monitoring
6. Show welcome page (if first activation)

### Deactivation

**Cleanup:**
- Dispose all webviews
- End progress tracking sessions
- Clear statusbar items
- Unregister commands (automatic)

---

## Contributing

For contribution guidelines, see [CONTRIBUTING.md](../CONTRIBUTING.md).

For testing details, see [TESTING.md](TESTING.md).

For packaging details, see [PACKAGING.md](PACKAGING.md).
