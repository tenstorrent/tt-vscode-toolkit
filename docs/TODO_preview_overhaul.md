# TODO: Output Preview Overhaul - Unified VSCode/code-server Visualization

**Date:** 2026-02-11
**Status:** Research Complete, Implementation Pending
**Priority:** Medium (current solution works, this is an enhancement)

---

## Current State

The tt_vscode_bridge.py visualization system uses different approaches for desktop VSCode vs code-server:

- **Desktop VSCode:** `code --command tenstorrent.showVisualization` (fully automatic)
- **code-server:** Manual file open with Cmd/Ctrl+Click (semi-automatic, requires one-time user action)

**Files:**
- `content/templates/cookbook/tt_vscode_bridge.py` - Python bridge with environment detection
- `src/extension.ts` - showVisualization command (lines 1931-1992)
- `TenstorrentImagePreviewProvider` - Webview showing images/logo (lines 4103-4400)

**Current Approach:**
- ✅ Works in both environments
- ✅ Good performance (20+ fps animations)
- ⚠️ Requires one-time manual setup in code-server
- ⚠️ Inconsistent UX between desktop and browser

---

## Goal

Implement unified automatic visualization that works identically in both desktop VSCode and code-server without any manual user steps.

---

## Research Summary

Evaluated 5 approaches for Python→VSCode communication:

| Approach | Desktop | code-server | Complexity | Performance | Verdict |
|----------|---------|-------------|------------|-------------|---------|
| FileSystemWatcher | ✅ Auto | ✅ Auto | Low | Good (20+ fps) | ✅ **Recommended** |
| HTTP/Unix Socket | ✅ Auto | ✅ Auto | Medium | Excellent (<20ms) | ⚠️ Future |
| LSP Pattern | ✅ Auto | ✅ Auto | High | Excellent | ❌ Overkill |
| URI Handler | ✅ Works | ❌ Broken | Low | Poor | ❌ No |
| Current Manual | ✅ Auto | ⚠️ Semi | Minimal | Good | ✅ Works Now |

---

## Recommended Implementation: FileSystemWatcher

### Why FileSystemWatcher Wins

1. ✅ **Unified behavior** - Works identically in both environments
2. ✅ **Low complexity** - Simplest fully-automatic approach
3. ✅ **Good performance** - Handles 20+ fps animations
4. ✅ **No external dependencies** - Pure VSCode API
5. ✅ **Well-documented** - Established pattern, many examples
6. ✅ **No code-server limitations** - FileSystemWatcher works in browser

**Sources:**
- [VSCode FileSystemWatcher Documentation](https://vshaxe.github.io/vscode-extern/vscode/FileSystemWatcher.html)
- [File Watcher Internals - VSCode Wiki](https://github.com/microsoft/vscode/wiki/File-Watcher-Internals)
- [File Watcher Issues - VSCode Wiki](https://github.com/microsoft/vscode/wiki/File-Watcher-Issues)

### How It Works

**Python side:**
1. Script saves visualization to file (e.g., `/tmp/game_of_life_frame.png`)
2. Script writes command to `/tmp/tt_vscode_commands.json`:
   ```json
   {
     "version": "1.0",
     "timestamp": 1707654321000,
     "commands": [{
       "command": "showVisualization",
       "args": {"imagePath": "/tmp/game_of_life_frame.png"}
     }]
   }
   ```

**Extension side:**
1. FileSystemWatcher detects command file change
2. Reads and parses JSON
3. Executes `showVisualization(imagePath)`
4. TenstorrentImagePreviewProvider updates webview

**Result:** Fully automatic in both desktop VSCode and code-server!

### Command File Format

**Location:** `/tmp/tt_vscode_commands.json`

**Schema:**
```json
{
  "version": "1.0",
  "timestamp": 1707654321000,
  "commands": [
    {
      "command": "showVisualization",
      "args": {
        "imagePath": "/absolute/path/to/image.png"
      }
    }
  ]
}
```

**Fields:**
- `version` - Protocol version (for future compatibility)
- `timestamp` - Unix timestamp in milliseconds (for deduplication)
- `commands` - Array of commands (allows batching in future)
- `command` - Command name (matches VSCode command registry)
- `args` - Command arguments (command-specific)

**Extensibility:** Format supports future commands:
- `showVideo` - Display video in preview
- `showHtml` - Render custom HTML visualization
- `updateTelemetry` - Send hardware metrics to logo animation
- `clearPreview` - Reset to logo

---

## Implementation Plan

### Phase 1: Extension Side (~1 hour)

**File:** `src/extension.ts`

**Add FileSystemWatcher setup:**
```typescript
/**
 * Setup command file watcher for Python→VSCode communication
 * This allows Python scripts to trigger extension commands automatically
 * Works identically in desktop VSCode and code-server
 */
function setupCommandFileWatcher(context: vscode.ExtensionContext): void {
  const commandFile = '/tmp/tt_vscode_commands.json';
  const fs = require('fs');

  // Create watcher
  const watcher = vscode.workspace.createFileSystemWatcher(commandFile);

  // Track last processed timestamp to avoid duplicates
  let lastProcessedTimestamp = 0;

  // Handle file changes
  watcher.onDidChange(async () => {
    try {
      // Read command file
      if (!fs.existsSync(commandFile)) {
        return;
      }

      const content = fs.readFileSync(commandFile, 'utf8');
      const data = JSON.parse(content);

      // Check version
      if (data.version !== '1.0') {
        console.warn(`Unknown command file version: ${data.version}`);
        return;
      }

      // Check timestamp (avoid processing same commands twice)
      if (data.timestamp <= lastProcessedTimestamp) {
        return; // Already processed
      }
      lastProcessedTimestamp = data.timestamp;

      // Execute commands
      for (const cmd of data.commands || []) {
        if (cmd.command === 'showVisualization') {
          await showVisualization(cmd.args.imagePath);
        } else {
          console.warn(`Unknown command: ${cmd.command}`);
        }
      }
    } catch (err) {
      // Log error but don't crash extension
      console.error('Error processing command file:', err);
    }
  });

  // Also watch for file creation (first time)
  watcher.onDidCreate(async () => {
    // Same handler as onDidChange
    watcher.onDidChange();
  });

  // Cleanup on extension deactivation
  context.subscriptions.push(watcher);
}
```

**Register in activate():**
```typescript
export function activate(context: vscode.ExtensionContext) {
  // ... existing activation code ...

  // Setup command file watcher for Python scripts
  setupCommandFileWatcher(context);

  // ... rest of activation ...
}
```

**Lines added:** ~50 lines
**Files modified:** 1 file (extension.ts)

---

### Phase 2: Python Side (~30 minutes)

**File:** `content/templates/cookbook/tt_vscode_bridge.py`

**Add command file communication:**
```python
def _send_via_command_file(abs_path: str, verbose: bool = False) -> bool:
    """
    Send visualization command via FileSystemWatcher.

    This works identically in both desktop VSCode and code-server.
    The extension watches /tmp/tt_vscode_commands.json for changes.

    Args:
        abs_path: Absolute path to image file
        verbose: Print status messages

    Returns:
        True if command file written successfully
    """
    import json
    import time

    command_file = '/tmp/tt_vscode_commands.json'

    # Prepare command data
    command_data = {
        "version": "1.0",
        "timestamp": int(time.time() * 1000),
        "commands": [{
            "command": "showVisualization",
            "args": {
                "imagePath": abs_path
            }
        }]
    }

    try:
        # Write command file (triggers FileSystemWatcher)
        with open(command_file, 'w') as f:
            json.dump(command_data, f)

        if verbose:
            print(f"✅ Sent to VSCode via command file")

        return True
    except Exception as e:
        if verbose:
            print(f"⚠️  Failed to write command file: {e}", file=sys.stderr)
        return False
```

**Update show_plot() to try command file first:**
```python
def show_plot(
    image_path: str,
    fallback: bool = True,
    verbose: bool = False
) -> bool:
    """
    Display an image in VSCode's Output Preview panel.

    Tries multiple approaches in order:
    1. Command file (works in both desktop and code-server)
    2. Desktop VSCode CLI (faster, desktop only)
    3. code-server manual open (fallback)

    Args:
        image_path: Absolute or relative path to the image file
        fallback: If True, print file path when VSCode unavailable
        verbose: If True, print status messages

    Returns:
        True if successfully sent to VSCode, False if fallback used
    """
    # Convert to absolute path
    abs_path = os.path.abspath(image_path)

    # Verify file exists
    if not os.path.exists(abs_path):
        if verbose:
            print(f"⚠️  Image not found: {abs_path}", file=sys.stderr)
        return False

    # Detect VSCode environment
    vscode_type = _detect_vscode_type()

    # Try command file first (works everywhere)
    if vscode_type != 'none':
        success = _send_via_command_file(abs_path, verbose)
        if success:
            return True

    # Fallback to desktop CLI if available (faster than command file)
    if vscode_type == 'desktop':
        return _send_to_desktop_vscode(abs_path, fallback, verbose)

    # Fallback to code-server instructions
    if vscode_type == 'code-server':
        return _send_to_code_server(abs_path, fallback, verbose)

    # No VSCode detected
    if verbose:
        print("ℹ️  VSCode not detected, using fallback display", file=sys.stderr)
    if fallback:
        print(f"📊 View your visualization: {abs_path}")
        print(f"   In VSCode: Cmd/Ctrl+Click the path above")
    return False
```

**Lines modified:** ~30 lines
**Files modified:** 1 file (tt_vscode_bridge.py)

---

### Phase 3: Testing (~1 hour)

**Test Matrix:**

| Environment | Test | Expected Result |
|-------------|------|-----------------|
| Desktop VSCode | Run Game of Life | Automatic animation in Output Preview |
| Desktop VSCode | Run simple sine test | Automatic animation in Output Preview |
| code-server | Run Game of Life | Automatic animation (no manual open!) |
| code-server | Run simple sine test | Automatic animation (no manual open!) |

**Test Commands:**
```bash
# Desktop VSCode
cd ~/tt-scratchpad
python3 test_vscode_bridge_simple.py
# Should see automatic animation, no terminal instructions

# code-server (same commands)
cd ~/tt-scratchpad
python3 test_vscode_bridge_simple.py
# Should ALSO see automatic animation, no manual open needed!
```

**Success Criteria:**
- ✅ Animations work automatically in both environments
- ✅ No "Cmd/Ctrl+Click" instructions in code-server
- ✅ Smooth 20+ fps performance
- ✅ No visible UI differences between desktop and browser
- ✅ Command file cleanup on extension deactivation

---

### Phase 4: Documentation (~30 minutes)

**Update Files:**

1. **`docs/VSCODE_VISUALIZATION_BRIDGE.md`**
   - Update "How It Works" section
   - Add command file format specification
   - Update troubleshooting for new approach

2. **`VISUALIZATION_STATUS.md`**
   - Update status to "✅ WORKING (unified automatic)"
   - Remove code-server manual step documentation
   - Add command file approach notes

3. **`CODE_SERVER_FIX.md`**
   - Mark as obsolete (or archive)
   - Redirect to new unified approach

4. **Lesson content** (where visualization is used)
   - Update instructions to remove code-server-specific steps
   - Unified experience across all environments

**Lines modified:** ~200 lines across 4 files

---

### Phase 5: Cleanup and Version (~15 minutes)

1. Remove or deprecate code-server-specific workarounds
2. Keep fallbacks for compatibility
3. Increment package.json version (bug fix = patch)
4. Update CHANGELOG.md
5. Git commit with comprehensive message

---

## Total Estimated Time

| Phase | Task | Time |
|-------|------|------|
| 1 | Extension FileSystemWatcher | 1 hour |
| 2 | Python command file | 30 min |
| 3 | Testing both environments | 1 hour |
| 4 | Documentation updates | 30 min |
| 5 | Cleanup and versioning | 15 min |
| **Total** | | **3 hours 15 minutes** |

---

## Alternative Future Enhancement: HTTP/Unix Socket Server

If FileSystemWatcher performance proves insufficient (unlikely), the next enhancement would be an HTTP server:

### Pattern

**Extension side:**
```typescript
// Create HTTP server on Unix socket
const server = http.createServer((req, res) => {
  if (req.method === 'POST' && req.url === '/showImage') {
    let body = '';
    req.on('data', chunk => body += chunk);
    req.on('end', async () => {
      const { imagePath } = JSON.parse(body);
      await showVisualization(imagePath);
      res.writeHead(200);
      res.end(JSON.stringify({ success: true }));
    });
  }
});

const socketPath = `/tmp/tt-vscode-${process.pid}.sock`;
server.listen(socketPath);

// Write socket path to known location
fs.writeFileSync('/tmp/tt_vscode_socket_path', socketPath);
```

**Python side:**
```python
import http.client, socket, json

# Read socket path
with open('/tmp/tt_vscode_socket_path') as f:
    socket_path = f.read().strip()

# Connect to Unix socket
conn = http.client.HTTPConnection('localhost')
conn.sock = socket.socket(socket.AF_UNIX)
conn.sock.connect(socket_path)

# Send command
conn.request('POST', '/showImage',
             json.dumps({"imagePath": "/path/to/image.png"}))
response = conn.getresponse()
```

### Benefits Over FileSystemWatcher

- **Lower latency:** 10-20ms vs 50-100ms (2-5x faster)
- **No file I/O:** Direct memory communication
- **Bidirectional:** Extension can query Python for status
- **Better for high-frequency:** 60+ fps capable
- **Proven pattern:** Used by VSCode Git extension

### When to Consider

- User reports choppy animations at 20+ fps
- Need real-time telemetry streaming (hardware→logo animation)
- Implementing interactive visualizations (click callbacks)
- Building two-way communication (extension controls Python)

### Additional Effort

- **Extension:** 2-3 hours (HTTP server setup, cross-platform paths)
- **Python:** 1-2 hours (HTTP client, error handling)
- **Testing:** 2 hours (Windows named pipes, Unix sockets, security)
- **Total:** 6-8 hours

---

## References

### Research Sources

**FileSystemWatcher API:**
- [VSCode FileSystemWatcher Documentation](https://vshaxe.github.io/vscode-extern/vscode/FileSystemWatcher.html)
- [File Watcher Internals - VSCode Wiki](https://github.com/microsoft/vscode/wiki/File-Watcher-Internals)
- [File Watcher Issues - VSCode Wiki](https://github.com/microsoft/vscode/wiki/File-Watcher-Issues)

**HTTP/IPC Patterns:**
- [VSCode Git IPC Server](https://github.com/microsoft/vscode/blob/main/extensions/git/src/ipc/ipcServer.ts)
- [VSCode Extension IPC Discussion](https://github.com/microsoft/vscode/issues/138596)
- [Language Server Protocol Transport](https://github.com/microsoft/vscode-languageserver-node/issues/662)

**Language Server Protocol:**
- [Language Server Extension Guide](https://code.visualstudio.com/api/language-extensions/language-server-extension-guide)
- [LSP in VSCode Extension Tutorial](https://symflower.com/en/company/blog/2022/lsp-in-vscode-extension/)
- [VSCode Language Server Node](https://github.com/microsoft/vscode-languageserver-node)

**URI Handler:**
- [VSCode URI Handler Sample](https://github.com/microsoft/vscode-extension-samples/blob/main/uri-handler-sample/README.md)
- [Callback to Extension from Outside VSCode](https://www.eliostruyf.com/callback-extension-vscode/)
- [registerUriHandler code-server Issue](https://github.com/coder/code-server/discussions/3891) ❌ (broken)

**code-server Compatibility:**
- [code-server Documentation](https://coder.com/docs/code-server/FAQ)
- [code-server vs VSCode Differences](https://github.com/coder/code-server/discussions/2345)

---

## Decision Record

**Date:** 2026-02-11
**Decision:** Defer implementation until after v0.65.1 validation complete
**Rationale:**
- Current manual approach works acceptably
- User said "can do better but save for later"
- Focus on lesson validation first (higher priority)
- 3-hour implementation can wait

**Next Review:** After validation phase complete
**Owner:** TBD
**Status:** Ready to implement when prioritized
