/**
 * Tenstorrent Developer Extension
 *
 * This extension provides a walkthrough-based setup experience for Tenstorrent
 * hardware development. The walkthrough guides users through:
 *   1. Hardware detection (tt-smi)
 *   2. Installation verification (tt-metal test)
 *   3. Model downloading (Hugging Face CLI)
 *
 * Architecture:
 * - Content is defined in markdown files (content/lessons/*.md)
 * - Walkthrough structure is defined in package.json (contributes.walkthroughs)
 * - This file only registers commands that execute terminal operations
 *
 * Technical writers can edit lesson content without touching this code!
 */

import * as vscode from 'vscode';
import { TERMINAL_COMMANDS, replaceVariables } from './commands/terminalCommands';

// Configuration imports
import { getModelConfig, getModelBasePath, getModelOriginalPath, DEFAULT_MODEL_KEY } from './config';

// New lesson system imports
import { LessonRegistry } from './utils';
import { StateManager, ProgressTracker } from './state';
import { LessonTreeDataProvider, LessonWebviewManager } from './views';
import { EnvironmentManager } from './services/EnvironmentManager';

// Telemetry monitoring imports
import { TelemetryMonitor } from './telemetry/TelemetryMonitor';

// ============================================================================
// Global State
// ============================================================================

/**
 * Global extension context for accessing persistent state across commands.
 * Set during activation and used by commands that need to store/retrieve data.
 */
let extensionContext: vscode.ExtensionContext;

/**
 * Storage keys for persistent state
 */
const STATE_KEYS = {
  TT_METAL_PATH: 'ttMetalPath',
  MODEL_PATH: 'modelPath',
  STATUSBAR_UPDATE_INTERVAL: 'statusbarUpdateInterval',
  STATUSBAR_ENABLED: 'statusbarEnabled',
};

// ============================================================================
// Device Status Monitoring
// ============================================================================

/**
 * Cached device information to avoid excessive tt-smi calls
 */
// Individual device information
interface SingleDeviceInfo {
  deviceType: string | null;    // e.g., "N150", "N300", "P300"
  firmwareVersion: string | null;
  status: 'healthy' | 'warning' | 'error' | 'unknown';
  temperature: number | null;   // ASIC temperature in Celsius
  power: number | null;         // Power consumption in watts
  pciBus: string | null;        // PCI bus ID (e.g., "0000:01:00.0")
  deviceIndex: number;          // Device index (0, 1, 2, ...)
}

// Multi-device system information
interface DeviceInfo {
  devices: SingleDeviceInfo[];   // Array of all devices
  deviceCount: number;           // Total number of devices
  primaryDeviceType: string | null;  // Type of first device (for display)
  overallStatus: 'healthy' | 'warning' | 'error' | 'unknown';
  lastChecked: number;
}

let cachedDeviceInfo: DeviceInfo = {
  devices: [],
  deviceCount: 0,
  primaryDeviceType: null,
  overallStatus: 'unknown',
  lastChecked: 0,
};

let statusBarItem: vscode.StatusBarItem | undefined;
let commandMenuStatusBarItem: vscode.StatusBarItem | undefined;
let statusUpdateTimer: NodeJS.Timeout | undefined;

/**
 * Global EnvironmentManager for tracking Python environments per terminal
 */
let environmentManager: EnvironmentManager;

/**
 * Parses tt-smi output to extract multi-device information.
 * Supports both JSON format (tt-smi -s) and text format.
 *
 * @param output - Raw output from tt-smi command
 * @returns DeviceInfo object with parsed data for all devices
 */
function parseDeviceInfo(output: string): DeviceInfo {
  const devices: SingleDeviceInfo[] = [];

  // Check for error indicators first
  const hasError = output.toLowerCase().includes('error') ||
                   output.toLowerCase().includes('failed') ||
                   output.toLowerCase().includes('timeout');

  // Try to parse as JSON first (tt-smi -s format)
  try {
    const jsonMatch = output.match(/\{[\s\S]*\}/);
    if (jsonMatch) {
      const data = JSON.parse(jsonMatch[0]);

      // Extract device info from JSON - now parse ALL devices
      if (data.device_info && Array.isArray(data.device_info)) {
        for (let i = 0; i < data.device_info.length; i++) {
          const device = data.device_info[i];

          let deviceType: string | null = null;
          let firmwareVersion: string | null = null;
          let status: 'healthy' | 'warning' | 'error' | 'unknown' = 'unknown';
          let temperature: number | null = null;
          let power: number | null = null;
          let pciBus: string | null = null;

          // Get board type (e.g., "n150 L" -> "N150", "p300c" -> "P300")
          if (device.board_info && device.board_info.board_type) {
            const boardType = device.board_info.board_type.toUpperCase();
            // Extract just the device model (N150, N300, P300, etc.)
            const match = boardType.match(/([NP]\d+)/);
            deviceType = match ? match[1] : boardType.split(' ')[0];
          }

          // Get firmware version
          if (device.firmwares && device.firmwares.fw_bundle_version) {
            firmwareVersion = device.firmwares.fw_bundle_version;
          }

          // Get temperature (from telemetry)
          if (device.telemetry && device.telemetry.asic_temperature) {
            const tempStr = device.telemetry.asic_temperature.trim();
            const tempValue = parseFloat(tempStr);
            if (!isNaN(tempValue)) {
              temperature = tempValue;
            }
          }

          // Get power (from telemetry)
          if (device.telemetry && device.telemetry.power) {
            const powerStr = device.telemetry.power.trim();
            const powerValue = parseFloat(powerStr);
            if (!isNaN(powerValue)) {
              power = powerValue;
            }
          }

          // Get PCI bus ID
          if (device.board_info && device.board_info.bus_id) {
            pciBus = device.board_info.bus_id;
          }

          // Check DRAM and device status
          if (device.board_info) {
            const dramStatus = device.board_info.dram_status;
            if (dramStatus === true || dramStatus === 'true') {
              status = hasError ? 'warning' : 'healthy';
            } else {
              status = 'warning';
            }
          } else {
            status = hasError ? 'error' : 'healthy';
          }

          devices.push({
            deviceType,
            firmwareVersion,
            status,
            temperature,
            power,
            pciBus,
            deviceIndex: i,
          });
        }
      }
    }
  } catch (e) {
    // JSON parsing failed - return empty result
    console.error('Failed to parse tt-smi JSON:', e);
  }

  // Calculate overall status (worst status wins)
  let overallStatus: 'healthy' | 'warning' | 'error' | 'unknown' = 'unknown';
  if (devices.length > 0) {
    const hasAnyError = devices.some(d => d.status === 'error');
    const hasAnyWarning = devices.some(d => d.status === 'warning');

    if (hasAnyError) {
      overallStatus = 'error';
    } else if (hasAnyWarning) {
      overallStatus = 'warning';
    } else if (devices.every(d => d.status === 'healthy')) {
      overallStatus = 'healthy';
    }
  }

  return {
    devices,
    deviceCount: devices.length,
    primaryDeviceType: devices.length > 0 ? devices[0].deviceType : null,
    overallStatus,
    lastChecked: Date.now(),
  };
}

/**
 * Runs tt-smi and updates cached device info.
 * Uses child_process.exec to capture output without showing terminal.
 * Uses tt-smi -s for structured JSON output.
 *
 * @returns Promise<DeviceInfo> - Updated device information
 */
async function updateDeviceStatus(): Promise<DeviceInfo> {
  const { exec } = await import('child_process');
  const { promisify } = await import('util');
  const execAsync = promisify(exec);

  try {
    // Use tt-smi -s for structured JSON output
    const { stdout, stderr } = await execAsync('tt-smi -s', { timeout: 10000 });
    const output = stdout + stderr;

    cachedDeviceInfo = parseDeviceInfo(output);
    updateStatusBarItem();

    return cachedDeviceInfo;
  } catch (error) {
    // tt-smi not found or failed
    cachedDeviceInfo = {
      devices: [],
      deviceCount: 0,
      primaryDeviceType: null,
      overallStatus: 'error',
      lastChecked: Date.now(),
    };
    updateStatusBarItem();

    return cachedDeviceInfo;
  }
}

/**
 * Updates the statusbar item text and icon based on cached device info.
 * Displays beautiful multi-device summary that scales from 1 to 32+ devices.
 */
function updateStatusBarItem(): void {
  if (!statusBarItem) return;

  const { devices, deviceCount, primaryDeviceType, overallStatus } = cachedDeviceInfo;

  // Set icon based on overall status
  let icon = '$(question)'; // unknown
  if (overallStatus === 'healthy') {
    icon = '$(check)';
  } else if (overallStatus === 'warning') {
    icon = '$(warning)';
  } else if (overallStatus === 'error') {
    icon = '$(x)';
  }

  // Set text based on device count
  if (deviceCount === 0) {
    statusBarItem.text = `${icon} TT: No device`;
    statusBarItem.tooltip = 'No Tenstorrent device detected - Click for options';
  } else if (deviceCount === 1) {
    // Single device: "✓ TT: P300 33.0°C"
    const device = devices[0];
    const tempInfo = device.temperature ? `${device.temperature.toFixed(1)}°C` : 'N/A';
    statusBarItem.text = `${icon} TT: ${primaryDeviceType} ${tempInfo}`;
    const powerInfo = device.power ? `${device.power.toFixed(1)}W` : 'N/A';
    statusBarItem.tooltip = `Tenstorrent ${primaryDeviceType}\n${tempInfo} | ${powerInfo}\nClick for device actions`;
  } else {
    // Multiple devices: "✓ TT: 4x P300 33-37°C"
    const tempRange = devices
      .filter(d => d.temperature !== null)
      .map(d => d.temperature as number);
    const powerSum = devices
      .filter(d => d.power !== null)
      .reduce((sum, d) => sum + (d.power as number), 0);

    // Add temp range to status bar text for at-a-glance visibility
    let statusText = `${icon} TT: ${deviceCount}x ${primaryDeviceType}`;
    if (tempRange.length > 0) {
      const minTemp = Math.min(...tempRange);
      const maxTemp = Math.max(...tempRange);
      statusText += ` ${Math.round(minTemp)}-${Math.round(maxTemp)}°C`;
    }
    statusBarItem.text = statusText;

    // Build detailed tooltip with all devices
    let tooltip = `${deviceCount} Tenstorrent ${primaryDeviceType} devices\n`;

    if (tempRange.length > 0) {
      const minTemp = Math.min(...tempRange);
      const maxTemp = Math.max(...tempRange);
      tooltip += `Temperature: ${minTemp.toFixed(1)}°C - ${maxTemp.toFixed(1)}°C\n`;
    }

    if (powerSum > 0) {
      tooltip += `Total Power: ${powerSum.toFixed(1)}W\n`;
    }

    tooltip += '\nClick for per-device details';
    statusBarItem.tooltip = tooltip;
  }

  statusBarItem.show();
}

/**
 * Shows quick actions menu when statusbar item is clicked.
 * Displays per-device details for multi-device systems.
 */
async function showDeviceActionsMenu(): Promise<void> {
  const { devices, deviceCount, primaryDeviceType, lastChecked } = cachedDeviceInfo;

  // Calculate time since last check
  const minutesAgo = Math.floor((Date.now() - lastChecked) / 60000);
  const timeStr = minutesAgo === 0 ? 'just now' : `${minutesAgo}m ago`;

  const items: vscode.QuickPickItem[] = [
    {
      label: '$(sync) Refresh Status',
      description: `Last checked: ${timeStr}`,
      detail: 'Run tt-smi to update device status',
    },
    {
      label: '$(terminal) Check Device Status',
      description: 'Open terminal',
      detail: 'Run tt-smi in a terminal window to see full output',
    },
  ];

  // Add per-device details if devices exist
  if (devices.length > 0) {
    items.push({
      label: '────────── Devices ──────────',
      description: '',
      detail: '',
    } as any); // Separator

    for (const device of devices) {
      const statusIcon = device.status === 'healthy' ? '$(check)' :
                        device.status === 'warning' ? '$(warning)' :
                        device.status === 'error' ? '$(x)' : '$(question)';

      const tempStr = device.temperature ? `${device.temperature.toFixed(1)}°C` : 'N/A';
      const powerStr = device.power ? `${device.power.toFixed(1)}W` : 'N/A';

      items.push({
        label: `${statusIcon} Device ${device.deviceIndex}: ${device.deviceType}`,
        description: `${tempStr} | ${powerStr}`,
        detail: `${device.pciBus || 'Unknown bus'} | FW: ${device.firmwareVersion || 'Unknown'}`,
      });
    }
  }

  // Add device-specific actions
  if (primaryDeviceType) {
    items.push(
      {
        label: '────────── Actions ──────────',
        description: '',
        detail: '',
      } as any, // Separator
      {
        label: '$(debug-restart) Reset Device',
        description: 'Run tt-smi -r',
        detail: 'Software reset of the Tenstorrent device',
      },
      {
        label: '$(clear-all) Clear Device State',
        description: 'Clear /dev/shm and device state',
        detail: 'Remove cached data and device state (requires sudo)',
      }
    );
  }

  // Configuration options
  const autoUpdateEnabled = extensionContext.globalState.get<boolean>(
    STATE_KEYS.STATUSBAR_ENABLED,
    false
  );

  items.push(
    {
      label: '────────── Settings ──────────',
      description: '',
      detail: '',
    } as any, // Separator
    {
      label: '$(settings-gear) Configure Update Interval',
      description: 'Change auto-update frequency',
      detail: 'Set how often device status is checked automatically',
    },
    {
      label: autoUpdateEnabled ? '$(debug-pause) Disable Auto-Update' : '$(debug-start) Enable Auto-Update',
      description: autoUpdateEnabled ? 'Currently: ON' : 'Currently: OFF',
      detail: 'Turn periodic device status updates on or off',
    }
  );

  const placeholderText = deviceCount === 0
    ? 'Tenstorrent Device Actions'
    : deviceCount === 1
    ? `Tenstorrent ${primaryDeviceType} Device Actions`
    : `${deviceCount}x Tenstorrent ${primaryDeviceType} Device Actions`;

  const selected = await vscode.window.showQuickPick(items, {
    placeHolder: placeholderText,
  });

  if (!selected) return;

  // Handle selection
  if (selected.label.includes('Refresh Status')) {
    statusBarItem!.text = '$(sync~spin) TT: Checking...';
    await updateDeviceStatus();
    vscode.window.showInformationMessage(
      cachedDeviceInfo.deviceCount > 0
        ? `✓ Found ${cachedDeviceInfo.deviceCount}x ${cachedDeviceInfo.primaryDeviceType} device${cachedDeviceInfo.deviceCount > 1 ? 's' : ''}`
        : '⚠️ No Tenstorrent device detected'
    );
  } else if (selected.label.includes('Check Device Status')) {
    vscode.commands.executeCommand('tenstorrent.runHardwareDetection');
  } else if (selected.label.includes('Reset Device')) {
    vscode.commands.executeCommand('tenstorrent.resetDevice');
  } else if (selected.label.includes('Clear Device State')) {
    vscode.commands.executeCommand('tenstorrent.clearDeviceState');
  } else if (selected.label.includes('Configure Update Interval')) {
    await configureUpdateInterval();
  } else if (selected.label.includes('Auto-Update')) {
    await toggleAutoUpdate();
  }
}

/**
 * Allows user to configure the auto-update interval.
 */
async function configureUpdateInterval(): Promise<void> {
  const intervals = [
    { label: '30 seconds', value: 30 },
    { label: '1 minute', value: 60 },
    { label: '2 minutes', value: 120 },
    { label: '5 minutes', value: 300 },
    { label: '10 minutes', value: 600 },
  ];

  const selected = await vscode.window.showQuickPick(intervals, {
    placeHolder: 'Select how often to check device status',
  });

  if (selected) {
    await extensionContext.globalState.update(
      STATE_KEYS.STATUSBAR_UPDATE_INTERVAL,
      selected.value
    );

    // Restart the update timer
    startStatusUpdateTimer();

    vscode.window.showInformationMessage(
      `Device status will update every ${selected.label}`
    );
  }
}

/**
 * Toggles auto-update on or off.
 */
async function toggleAutoUpdate(): Promise<void> {
  const currentEnabled = extensionContext.globalState.get<boolean>(
    STATE_KEYS.STATUSBAR_ENABLED,
    false  // Default is false (disabled)
  );

  await extensionContext.globalState.update(
    STATE_KEYS.STATUSBAR_ENABLED,
    !currentEnabled
  );

  if (!currentEnabled) {
    // Enabling
    startStatusUpdateTimer();
    vscode.window.showInformationMessage('✓ Auto-update enabled. Device status will refresh automatically.');
  } else {
    // Disabling
    stopStatusUpdateTimer();
    vscode.window.showInformationMessage('✓ Auto-update disabled. Click "Refresh Status" to check device manually.');
  }
}

/**
 * Starts the periodic status update timer.
 */
function startStatusUpdateTimer(): void {
  // Stop existing timer
  stopStatusUpdateTimer();

  // Check if auto-update is enabled (default: false)
  const enabled = extensionContext.globalState.get<boolean>(
    STATE_KEYS.STATUSBAR_ENABLED,
    false  // Changed from true to false - auto-polling disabled by default
  );

  if (!enabled) return;

  // Get update interval (default 60 seconds)
  const intervalSeconds = extensionContext.globalState.get<number>(
    STATE_KEYS.STATUSBAR_UPDATE_INTERVAL,
    60
  );

  // Run initial check
  updateDeviceStatus();

  // Start periodic updates
  statusUpdateTimer = setInterval(() => {
    updateDeviceStatus();
  }, intervalSeconds * 1000);
}

/**
 * Stops the periodic status update timer.
 */
function stopStatusUpdateTimer(): void {
  if (statusUpdateTimer) {
    clearInterval(statusUpdateTimer);
    statusUpdateTimer = undefined;
  }
}

// ============================================================================
// Terminal Management
// ============================================================================

/**
 * Terminal Management: Project-based Context Isolation
 *
 * Strategy: 6 terminals organized by project/purpose for optimal venv isolation
 *
 * Terminals:
 * 1. "Tenstorrent: TT-Metal" - tt-metal setup, demos, TTNN, cookbook
 *    - Uses PYTHONPATH + source setup-metal.sh (no dedicated venv)
 * 2. "Tenstorrent: TT-Forge" - TT-Forge build, test, image classification
 *    - Uses ~/tt-forge-venv
 * 3. "Tenstorrent: TT-XLA" - JAX demos and testing
 *    - Uses ~/tt-xla-venv
 * 4. "Tenstorrent: vLLM Server" - Production inference with vLLM
 *    - Uses ~/tt-vllm-venv
 * 5. "Tenstorrent: API Server" - Direct API, Flask servers
 *    - Uses tt-metal environment
 * 6. "Tenstorrent: Explore" - Manual exploration, curl, ad-hoc testing
 *    - Uses system default (no venv activation)
 *
 * Benefits:
 * ✅ Clear context-specific naming (know what's running at a glance)
 * ✅ Proper venv isolation (no environment pollution between projects)
 * ✅ Terminal reuse (no clutter)
 * ✅ Project consolidation (one terminal per project)
 */

/**
 * Terminal context for project-based isolation
 */
type TerminalContext =
  | 'tt-metal'     // All tt-metal: setup, demos, TTNN, cookbook
  | 'tt-forge'     // All tt-forge: build, test, classify
  | 'tt-xla'       // All tt-xla/JAX commands
  | 'vllm-server'  // vLLM long-running servers
  | 'api-server'   // Direct API, Flask servers (uses tt-metal env)
  | 'explore';     // Manual exploration, curl commands, ad-hoc testing

/**
 * Terminal storage by context
 */
const terminals: Record<TerminalContext, vscode.Terminal | undefined> = {
  'tt-metal': undefined,
  'tt-forge': undefined,
  'tt-xla': undefined,
  'vllm-server': undefined,
  'api-server': undefined,
  'explore': undefined,
};

/**
 * Terminal display names for each context
 */
const TERMINAL_NAMES: Record<TerminalContext, string> = {
  'tt-metal': 'TT: Metal',
  'tt-forge': 'TT: Forge',
  'tt-xla': 'TT: XLA',
  'vllm-server': 'TT: vLLM',
  'api-server': 'TT: API',
  'explore': 'TT: Explore',
};

/**
 * Gets or creates a terminal based on the context.
 * Reuses existing terminals if they're still alive to avoid clutter.
 *
 * Terminal routing by context:
 * - 'tt-metal': Hardware detection, setup, demos, TTNN, cookbook (PYTHONPATH + setup-metal.sh)
 * - 'tt-forge': TT-Forge build, test, image classification (~/tt-forge-venv)
 * - 'tt-xla': TT-XLA/JAX demos and testing (~/tt-xla-venv)
 * - 'vllm-server': vLLM production inference servers (~/tt-vllm-venv)
 * - 'api-server': Direct API chat and Flask servers (tt-metal env)
 * - 'explore': Manual exploration with system default venv (no activation)
 *
 * @param context - Terminal context defining project and venv
 * @returns Active terminal instance
 */
function getOrCreateTerminal(context: TerminalContext): vscode.Terminal {
  // Check if terminal still exists and reuse it
  if (terminals[context] && vscode.window.terminals.includes(terminals[context]!)) {
    return terminals[context]!;
  }

  // Create new terminal with context-specific name
  const name = TERMINAL_NAMES[context];
  const terminal = vscode.window.createTerminal({
    name,
    cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
  });

  terminals[context] = terminal;

  // Track environment for status bar (but don't auto-activate)
  // Users can manually activate environments as needed
  if (environmentManager) {
    environmentManager.trackTerminal(terminal, context);
  }

  return terminal;
}

/**
 * Executes a command in the specified terminal.
 * Shows the terminal to the user so they can see the output.
 * Uses preserveFocus: false to ensure terminal is visible and focused.
 *
 * @param terminal - The terminal to execute the command in
 * @param command - The shell command to execute
 */
function runInTerminal(terminal: vscode.Terminal, command: string): void {
  // Show terminal and give it focus (preserveFocus: false ensures terminal panel is visible)
  terminal.show(false);
  terminal.sendText(command);
}

/**
 * Prompts the user to install recommended extensions on first activation.
 * Uses a non-intrusive notification that allows user to install all at once or dismiss.
 *
 * Recommended extensions:
 * - Python, Pylance, Jupyter (for Python development)
 * - C/C++, CMake Tools (for tt-metal C++ development)
 *
 * @param context - Extension context for accessing extension API
 */
async function promptRecommendedExtensions(): Promise<void> {
  const recommendedExtensions = [
    { id: 'ms-python.python', name: 'Python' },
    { id: 'ms-python.vscode-pylance', name: 'Pylance' },
    { id: 'ms-toolsai.jupyter', name: 'Jupyter' },
    { id: 'ms-vscode.cpptools', name: 'C/C++' },
    { id: 'ms-vscode.cmake-tools', name: 'CMake Tools' },
  ];

  // Check which extensions are not installed
  const missingExtensions = recommendedExtensions.filter(
    (ext) => !vscode.extensions.getExtension(ext.id)
  );

  if (missingExtensions.length === 0) {
    // All recommended extensions already installed
    return;
  }

  // Show notification with install action
  const extensionNames = missingExtensions.map((ext) => ext.name).join(', ');
  const message = `Install recommended extensions for Tenstorrent development? (${extensionNames})`;

  const choice = await vscode.window.showInformationMessage(
    message,
    'Install All',
    'Not Now',
    'Show Details'
  );

  if (choice === 'Install All') {
    // Install extensions one by one
    for (const ext of missingExtensions) {
      try {
        await vscode.commands.executeCommand('workbench.extensions.installExtension', ext.id);
      } catch (error) {
        console.error(`Failed to install ${ext.name}:`, error);
      }
    }

    vscode.window.showInformationMessage(
      `Installing ${missingExtensions.length} extension(s) in the background...`
    );
  } else if (choice === 'Show Details') {
    // Open Extensions view filtered to recommended
    vscode.commands.executeCommand('workbench.extensions.action.showRecommendedExtensions');
  }
  // "Not Now" or dismissed = do nothing
}

// ============================================================================
// Command Handlers
// ============================================================================

// ============================================================================
// LESSON 0: Modern Setup with tt-installer 2.0
// ============================================================================

/**
 * Command: tenstorrent.runQuickInstall
 *
 * Runs the one-command tt-installer quickstart.
 * Downloads and executes the latest installer with interactive prompts.
 */
function runQuickInstall(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.QUICK_INSTALL.template;

  vscode.window.showWarningMessage(
    '⚠️ This will download and run the tt-installer script. Review the script at https://github.com/tenstorrent/tt-installer before continuing.'
  );

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Running tt-installer. Follow the prompts in the terminal. Installation may take 5-15 minutes.'
  );
}

/**
 * Command: tenstorrent.downloadInstaller
 *
 * Downloads the tt-installer script for inspection.
 */
function downloadInstaller(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.DOWNLOAD_INSTALLER.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '📥 Downloaded install.sh to your home directory. Review it with: less ~/install.sh'
  );
}

/**
 * Command: tenstorrent.runInteractiveInstall
 *
 * Runs tt-installer with interactive prompts for customization.
 */
function runInteractiveInstall(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.RUN_INTERACTIVE_INSTALL.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '⚙️ Running interactive installation. Follow the prompts in the terminal.'
  );
}

/**
 * Command: tenstorrent.runNonInteractiveInstall
 *
 * Runs tt-installer in non-interactive mode with recommended defaults.
 */
function runNonInteractiveInstall(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.RUN_NON_INTERACTIVE_INSTALL.template;

  vscode.window.showInformationMessage(
    '🤖 Running non-interactive installation with recommended defaults. Check the terminal for progress.'
  );

  runInTerminal(terminal, command);
}

/**
 * Command: tenstorrent.testMetaliumContainer
 *
 * Tests that the tt-metalium container is installed and working.
 */
function testMetaliumContainer(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.TEST_METALIUM_CONTAINER.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🧪 Testing tt-metalium container. Check the terminal for TTNN version output.'
  );
}

// ============================================================================
// LESSON 1: Hardware Detection
// ============================================================================

/**
 * Command: tenstorrent.runHardwareDetection
 *
 * Runs the tt-smi command to detect and display connected Tenstorrent devices.
 * This is Step 1 in the walkthrough.
 */
function runHardwareDetection(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.TT_SMI.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Running hardware detection. Check the terminal for results.'
  );
}

/**
 * Command: tenstorrent.verifyInstallation
 *
 * Runs a test program to verify tt-metal is properly installed and configured.
 * This is Step 2 in the walkthrough.
 */
function verifyInstallation(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.VERIFY_INSTALLATION.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Running installation verification. Check the terminal for results.'
  );
}

/**
 * Command: tenstorrent.setHuggingFaceToken
 *
 * Prompts the user for their Hugging Face token and sets it as an environment
 * variable in the model download terminal. This is Step 3a in the walkthrough.
 */
async function setHuggingFaceToken(): Promise<void> {
  // Prompt user for their HF token (password input to hide the token)
  const token = await vscode.window.showInputBox({
    prompt: 'Enter your Hugging Face access token',
    placeHolder: 'hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',
    password: true, // Hide the token input
    ignoreFocusOut: true, // Don't dismiss if user clicks elsewhere
  });

  if (!token) {
    vscode.window.showWarningMessage('Hugging Face token is required to download models.');
    return;
  }

  // Set the token as an environment variable in the terminal
  const terminal = getOrCreateTerminal('tt-metal');
  const command = replaceVariables(TERMINAL_COMMANDS.SET_HF_TOKEN.template, { token });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Hugging Face token set. You can now authenticate and download models.'
  );
}

/**
 * Command: tenstorrent.loginHuggingFace
 *
 * Authenticates with Hugging Face using the token stored in HF_TOKEN.
 * This is Step 3b in the walkthrough.
 */
function loginHuggingFace(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.LOGIN_HF.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Authenticating with Hugging Face. Check the terminal for results.'
  );
}

/**
 * Command: tenstorrent.downloadModel
 *
 * Downloads the Llama-3.1-8B-Instruct model from Hugging Face to ~/models/.
 * Uses absolute path to ensure predictable location for inference scripts.
 * This is Step 3c in the walkthrough.
 */
function downloadModel(): void {
  const terminal = getOrCreateTerminal('tt-metal');
  const command = TERMINAL_COMMANDS.DOWNLOAD_MODEL.template;

  // Create models directory and download to absolute path
  // This ensures the model is in a predictable location for the inference script
  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Downloading model to ~/models/Llama-3.1-8B-Instruct. This is ~16GB and may take several minutes. Check the terminal for progress.'
  );
}

/**
 * Command: tenstorrent.cloneTTMetal
 *
 * Checks if ~/tt-metal exists, and offers to use it or prompt for new location.
 * Always asks the user before cloning to respect their filesystem preferences.
 * This is Step 3d in the walkthrough.
 */
async function cloneTTMetal(): Promise<void> {
  const fs = await import('fs');
  const os = await import('os');
  const path = await import('path');

  // Expand ~ to home directory
  const homeDir = os.homedir();
  const defaultTTMetalPath = path.join(homeDir, 'tt-metal');

  // Check if default tt-metal directory exists
  if (fs.existsSync(defaultTTMetalPath)) {
    // Directory exists - offer choice to user
    const choice = await vscode.window.showInformationMessage(
      `Found existing tt-metal installation at ${defaultTTMetalPath}`,
      'Use Existing',
      'Clone to Different Location'
    );

    if (choice === 'Use Existing') {
      // Store the path for use in subsequent commands
      await extensionContext.globalState.update(STATE_KEYS.TT_METAL_PATH, defaultTTMetalPath);

      vscode.window.showInformationMessage(
        `✓ Using existing tt-metal at ${defaultTTMetalPath}. Proceed to setup environment.`
      );
      return;
    } else if (choice === 'Clone to Different Location') {
      // User wants to clone to a different location - ask where
      const userPath = await vscode.window.showInputBox({
        prompt: 'Enter the full path where you want to clone tt-metal',
        placeHolder: '/home/user/my-projects/tt-metal',
        value: defaultTTMetalPath,
        ignoreFocusOut: true,
      });

      if (!userPath) {
        vscode.window.showWarningMessage('Clone cancelled. No path provided.');
        return;
      }

      // Validate and clone to user-specified path
      const parentDir = path.dirname(userPath);
      if (!fs.existsSync(parentDir)) {
        vscode.window.showErrorMessage(
          `Parent directory does not exist: ${parentDir}. Please create it first.`
        );
        return;
      }

      if (fs.existsSync(userPath)) {
        vscode.window.showErrorMessage(
          `Directory already exists: ${userPath}. Please choose a different location.`
        );
        return;
      }

      // Clone to user-specified location and store the path
      await extensionContext.globalState.update(STATE_KEYS.TT_METAL_PATH, userPath);

      const terminal = getOrCreateTerminal('tt-metal');
      const command = replaceVariables(TERMINAL_COMMANDS.CLONE_TT_METAL.template, {
        path: userPath,
      });

      runInTerminal(terminal, command);

      vscode.window.showInformationMessage(
        `Cloning tt-metal to ${userPath}. This may take several minutes. Check the terminal for progress.`
      );
    }
    // If user cancels (clicks X), do nothing
  } else {
    // Directory doesn't exist - ask user where to clone
    const choice = await vscode.window.showInformationMessage(
      'TT-Metal repository not found. Would you like to clone it?',
      'Clone to ~/tt-metal',
      'Choose Different Location'
    );

    if (choice === 'Clone to ~/tt-metal') {
      // Clone to default location and store the path
      await extensionContext.globalState.update(STATE_KEYS.TT_METAL_PATH, defaultTTMetalPath);

      const terminal = getOrCreateTerminal('tt-metal');
      const command = replaceVariables(TERMINAL_COMMANDS.CLONE_TT_METAL.template, {
        path: defaultTTMetalPath,
      });

      runInTerminal(terminal, command);

      vscode.window.showInformationMessage(
        `Cloning tt-metal to ${defaultTTMetalPath}. This may take several minutes. Check the terminal for progress.`
      );
    } else if (choice === 'Choose Different Location') {
      // Ask user for custom path
      const userPath = await vscode.window.showInputBox({
        prompt: 'Enter the full path where you want to clone tt-metal',
        placeHolder: '/home/user/my-projects/tt-metal',
        value: defaultTTMetalPath,
        ignoreFocusOut: true,
      });

      if (!userPath) {
        vscode.window.showWarningMessage('Clone cancelled. No path provided.');
        return;
      }

      // Validate and clone to user-specified path
      const parentDir = path.dirname(userPath);
      if (!fs.existsSync(parentDir)) {
        vscode.window.showErrorMessage(
          `Parent directory does not exist: ${parentDir}. Please create it first.`
        );
        return;
      }

      if (fs.existsSync(userPath)) {
        vscode.window.showErrorMessage(
          `Directory already exists: ${userPath}. Please choose a different location.`
        );
        return;
      }

      // Clone to user-specified location and store the path
      await extensionContext.globalState.update(STATE_KEYS.TT_METAL_PATH, userPath);

      const terminal = getOrCreateTerminal('tt-metal');
      const command = replaceVariables(TERMINAL_COMMANDS.CLONE_TT_METAL.template, {
        path: userPath,
      });

      runInTerminal(terminal, command);

      vscode.window.showInformationMessage(
        `Cloning tt-metal to ${userPath}. This may take several minutes. Check the terminal for progress.`
      );
    }
    // If user cancels (clicks X), do nothing
  }
}

/**
 * Command: tenstorrent.setupEnvironment
 *
 * Sets up the Python environment for running inference.
 * Sets PYTHONPATH and installs dependencies from the tt-metal repository.
 * This is Step 3e in the walkthrough.
 */
async function setupEnvironment(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const terminal = getOrCreateTerminal('tt-metal');

  // Run setup commands in sequence using the stored path
  const command = replaceVariables(TERMINAL_COMMANDS.SETUP_ENVIRONMENT.template, {
    ttMetalPath,
  });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    `Setting up Python environment in ${ttMetalPath}. This will install required dependencies. Check the terminal for progress.`
  );
}

/**
 * Command: tenstorrent.runInference
 *
 * Runs the Llama inference demo using pytest.
 * Sets LLAMA_DIR environment variable to point to the downloaded model.
 * This is Step 3f in the walkthrough - the final step!
 */
async function runInference(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  // Model path using constants
  const modelPath = await getModelOriginalPath();

  const terminal = getOrCreateTerminal('tt-metal');

  // Run inference demo with LLAMA_DIR set to the model location
  // and reasonable default parameters for seq length and token generation
  const command = replaceVariables(TERMINAL_COMMANDS.RUN_INFERENCE.template, {
    ttMetalPath,
    modelPath,
  });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Running Llama inference on Tenstorrent hardware! First run may take a few minutes for kernel compilation. Check the terminal for output.'
  );
}

/**
 * Command: tenstorrent.installInferenceDeps
 *
 * Installs additional Python dependencies required for interactive inference.
 * This is Step 4-1 in the walkthrough - Interactive Chat
 */
function installInferenceDeps(): void {
  const terminal = getOrCreateTerminal('api-server');
  const command = TERMINAL_COMMANDS.INSTALL_INFERENCE_DEPS.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Installing inference dependencies (pi and llama-models). This may take 1-2 minutes. Check the terminal for progress.'
  );
}

/**
 * Command: tenstorrent.createChatScript
 *
 * Creates the interactive chat script by copying the template to ~/tt-scratchpad/tt-chat.py
 * This is Step 4-2 in the walkthrough - Interactive Chat
 */
async function createChatScript(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  // Get the template path from the extension
  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-chat.py');

  // Check if template exists
  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  // Destination path in ~/tt-scratchpad/
  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  // Create scratchpad directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-chat.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    // Copy the template to scratchpad directory
    fs.copyFileSync(templatePath, destPath);

    // Make it executable
    fs.chmodSync(destPath, 0o755);

    vscode.window.showInformationMessage(
      `✅ Created interactive chat script at ${destPath}. You can now start a chat session!`
    );
  } catch (error) {
    vscode.window.showErrorMessage(
      `Failed to create chat script: ${error}`
    );
  }
}

/**
 * Command: tenstorrent.startChatSession
 *
 * Starts an interactive chat session with the Llama model.
 * This is Step 4-3 in the walkthrough - Interactive Chat
 */
async function startChatSession(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  // Model path using constants
  const modelPath = await getModelOriginalPath();

  const terminal = getOrCreateTerminal('api-server');

  // Run the interactive chat script with proper environment setup
  const command = replaceVariables(TERMINAL_COMMANDS.START_CHAT_SESSION.template, {
    ttMetalPath,
    modelPath,
  });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '💬 Starting interactive chat session. First load may take a few minutes. Type your prompts in the terminal!'
  );
}

/**
 * Command: tenstorrent.createApiServer
 *
 * Creates the API server script by copying the template to ~/tt-scratchpad/tt-api-server.py
 * This is Step 5a in the walkthrough - HTTP API Server
 */
async function createApiServer(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  // Get the template path from the extension
  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-api-server.py');

  // Check if template exists
  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  // Destination path in ~/tt-scratchpad/
  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  // Create scratchpad directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-api-server.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    // Copy the template to scratchpad directory
    fs.copyFileSync(templatePath, destPath);

    // Make it executable
    fs.chmodSync(destPath, 0o755);

    vscode.window.showInformationMessage(
      `✅ Created API server script at ${destPath}. Next, install Flask if you haven't already!`
    );
  } catch (error) {
    vscode.window.showErrorMessage(
      `Failed to create API server script: ${error}`
    );
  }
}

/**
 * Command: tenstorrent.installFlask
 *
 * Installs Flask web framework using pip.
 * This is Step 5b in the walkthrough - HTTP API Server
 */
function installFlask(): void {
  const terminal = getOrCreateTerminal('api-server');
  const command = TERMINAL_COMMANDS.INSTALL_FLASK.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Installing Flask. This should only take a few seconds. Check the terminal for progress.'
  );
}

/**
 * Command: tenstorrent.startApiServer
 *
 * Starts the Flask API server with the Llama model.
 * This is Step 5c in the walkthrough - HTTP API Server
 */
async function startApiServer(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  // Model path using constants
  const modelPath = await getModelOriginalPath();

  const terminal = getOrCreateTerminal('api-server');

  // Run the API server with proper environment setup
  const command = replaceVariables(TERMINAL_COMMANDS.START_API_SERVER.template, {
    ttMetalPath,
    modelPath,
  });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Starting API server on port 8080. First load may take a few minutes. Open a second terminal to test with curl!'
  );
}

/**
 * Command: tenstorrent.testApiBasic
 *
 * Tests the API server with a basic curl query.
 * This is Step 5d in the walkthrough - HTTP API Server
 */
function testApiBasic(): void {
  // Use a different terminal for testing so we don't interfere with the server
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.TEST_API_BASIC.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🧪 Testing API with basic query. Check the terminal for the response!'
  );
}

/**
 * Command: tenstorrent.testApiMultiple
 *
 * Tests the API server with multiple curl queries.
 * This is Step 5e in the walkthrough - HTTP API Server
 */
function testApiMultiple(): void {
  // Use a different terminal for testing so we don't interfere with the server
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.TEST_API_MULTIPLE.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🧪 Testing API with multiple queries. Check the terminal for the responses!'
  );
}

// ============================================================================
// Direct API Commands (Lessons 4-6)
// ============================================================================

/**
 * Command: tenstorrent.createChatScriptDirect
 * Creates the direct API chat script and opens it in the editor
 */
async function createChatScriptDirect(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-chat-direct.py');

  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  // Create scratchpad directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-chat-direct.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    fs.copyFileSync(templatePath, destPath);
    fs.chmodSync(destPath, 0o755);

    // Open the file in the editor (to the side, preserving focus on terminal)
    const doc = await vscode.workspace.openTextDocument(destPath);
    await vscode.window.showTextDocument(doc, {
      viewColumn: vscode.ViewColumn.Beside,
      preserveFocus: true,
      preview: false
    });

    vscode.window.showInformationMessage(
      `✅ Created direct API chat script at ${destPath}. The file is now open - review the code!`
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to create chat script: ${error}`);
  }
}

/**
 * Command: tenstorrent.startChatSessionDirect
 * Starts the direct API chat session
 */
async function startChatSessionDirect(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const ttMetalPath = path.join(homeDir, 'tt-metal');
  const modelPath = await getModelBasePath();

  const terminal = getOrCreateTerminal('api-server');

  const command = `cd ${ttMetalPath} && export HF_MODEL=${modelPath} && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-chat-direct.py`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '💬 Starting direct API chat. Model loads once (2-5 min), then queries are fast (1-3 sec)!'
  );
}

/**
 * Command: tenstorrent.createApiServerDirect
 * Creates the direct API server script and opens it in the editor
 */
async function createApiServerDirect(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-api-server-direct.py');

  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  // Create scratchpad directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-api-server-direct.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    fs.copyFileSync(templatePath, destPath);
    fs.chmodSync(destPath, 0o755);

    // Open the file in the editor (to the side, preserving focus on terminal)
    const doc = await vscode.workspace.openTextDocument(destPath);
    await vscode.window.showTextDocument(doc, {
      viewColumn: vscode.ViewColumn.Beside,
      preserveFocus: true,
      preview: false
    });

    vscode.window.showInformationMessage(
      `✅ Created direct API server at ${destPath}. The file is now open - review the code!`
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to create API server: ${error}`);
  }
}

/**
 * Command: tenstorrent.startApiServerDirect
 * Starts the direct API server
 */
async function startApiServerDirect(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const ttMetalPath = path.join(homeDir, 'tt-metal');
  const modelPath = await getModelBasePath();

  const terminal = getOrCreateTerminal('api-server');

  const command = `cd ${ttMetalPath} && export HF_MODEL=${modelPath} && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-api-server-direct.py --port 8080`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Starting direct API server. Model loads once (2-5 min), then handles requests fast!'
  );
}

/**
 * Command: tenstorrent.testApiBasicDirect
 * Tests the direct API server with a basic query
 */
function testApiBasicDirect(): void {
  const terminal = getOrCreateTerminal('explore');

  const command = `curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"prompt": "What is machine learning?"}'`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Testing direct API server. Check terminal for response!'
  );
}

/**
 * Command: tenstorrent.testApiMultipleDirect
 * Tests the direct API with multiple queries
 */
function testApiMultipleDirect(): void {
  const terminal = getOrCreateTerminal('explore');

  const commands = [
    `curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"prompt": "Explain neural networks"}'`,
    `echo "\n--- Second query ---\n"`,
    `curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"prompt": "Write a haiku about programming"}'`,
    `echo "\n--- Third query ---\n"`,
    `curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d '{"prompt": "What are transformers?"}'`
  ];

  runInTerminal(terminal, commands.join(' && '));

  vscode.window.showInformationMessage(
    'Running multiple API tests. Watch the terminal for fast responses!'
  );
}

// ============================================================================
// LESSON 6: Production Inference with tt-inference-server
// ============================================================================

/**
 * Command: tenstorrent.verifyInferenceServerPrereqs
 * Verifies tt-inference-server is installed, model is downloaded, and hardware is detected
 */
function verifyInferenceServerPrereqs(): void {
  const terminal = getOrCreateTerminal('vllm-server');
  const command = TERMINAL_COMMANDS.VERIFY_INFERENCE_SERVER_PREREQS.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🔍 Checking tt-inference-server prerequisites. Watch the terminal for results.'
  );
}

/**
 * Command: tenstorrent.startTtInferenceServer
 * Starts vLLM server via tt-inference-server with basic configuration
 */
function startTtInferenceServer(): void {
  const terminal = getOrCreateTerminal('vllm-server');
  const command = TERMINAL_COMMANDS.START_TT_INFERENCE_SERVER.template;

  vscode.window.showInformationMessage(
    '🚀 Starting vLLM server via tt-inference-server. This may take 5-15 minutes on first run (downloads Docker image + model).'
  );

  runInTerminal(terminal, command);
}

/**
 * Command: tenstorrent.testTtInferenceServerSimple
 * Tests the vLLM server started by tt-inference-server with OpenAI-compatible API
 */
function testTtInferenceServerSimple(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.TEST_TT_INFERENCE_SERVER_SIMPLE.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🧪 Testing vLLM server (OpenAI-compatible API). Check the terminal for the response.'
  );
}

/**
 * Command: tenstorrent.testTtInferenceServerStreaming
 * Tests streaming responses from vLLM server
 */
function testTtInferenceServerStreaming(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.TEST_TT_INFERENCE_SERVER_STREAMING.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🌊 Testing streaming mode (Server-Sent Events). Watch tokens arrive progressively in the terminal.'
  );
}

/**
 * Command: tenstorrent.testTtInferenceServerSampling
 * Tests different sampling parameters with OpenAI-compatible API
 */
function testTtInferenceServerSampling(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.TEST_TT_INFERENCE_SERVER_SAMPLING.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🎲 Testing different sampling parameters. Compare high vs low temperature in the terminal.'
  );
}

/**
 * Command: tenstorrent.createTtInferenceServerClient
 * Creates a Python client using OpenAI SDK to connect to the vLLM server
 */
function createTtInferenceServerClient(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.CREATE_TT_INFERENCE_SERVER_CLIENT.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '📝 Created ~/tt-scratchpad/tt-inference-client.py (uses OpenAI SDK). Run with: python3 ~/tt-scratchpad/tt-inference-client.py'
  );
}

/**
 * Command: tenstorrent.createTtInferenceServerConfig
 * Shows that tt-inference-server uses command-line arguments, not config files
 */
function createTtInferenceServerConfig(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.CREATE_TT_INFERENCE_SERVER_CONFIG.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'ℹ️ tt-inference-server uses command-line arguments, not config files. Check the terminal for example usage.'
  );
}

// ============================================================================
// LESSON 7: Production Inference with vLLM (previously Lesson 6)
// ============================================================================

// vLLM Commands

/**
 * Command: tenstorrent.updateTTMetal
 * Updates and rebuilds TT-Metal to latest main branch
 */
async function updateTTMetal(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const ttMetalPath = path.join(homeDir, 'tt-metal');

  const terminal = getOrCreateTerminal('tt-metal');

  const command = `cd ${ttMetalPath} && git checkout main && git pull origin main && git submodule update --init --recursive && ./install_dependencies.sh && ./build_metal.sh`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🔧 Updating and rebuilding TT-Metal (including dependencies). This takes 5-15 minutes. Watch terminal for progress.'
  );
}

/**
 * Command: tenstorrent.cloneVllm
 * Clones the TT vLLM repository
 */
function cloneVllm(): void {
  const terminal = getOrCreateTerminal('vllm-server');

  const command = `cd ~ && git clone --branch dev https://github.com/tenstorrent/vllm.git tt-vllm && cd tt-vllm`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Cloning TT vLLM repository. This may take 1-2 minutes...'
  );
}

/**
 * Command: tenstorrent.installVllm
 * Creates a dedicated venv and installs vLLM and dependencies
 */
function installVllm(): void {
  const terminal = getOrCreateTerminal('vllm-server');

  const command = `cd ~/tt-vllm && python3 -m venv ~/tt-vllm-venv && source ~/tt-vllm-venv/bin/activate && pip install --upgrade pip && export vllm_dir=$(pwd) && source $vllm_dir/tt_metal/setup-metal.sh && pip install --upgrade ttnn pytest && pip install fairscale termcolor loguru blobfile fire pytz llama-models==0.0.48 && pip install -e . --extra-index-url https://download.pytorch.org/whl/cpu`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Creating venv and installing vLLM with all dependencies (ttnn, pytest, etc). This will take 5-10 minutes. Check terminal for progress.'
  );
}

/**
 * Command: tenstorrent.startVllmServer
 * Starts the vLLM OpenAI-compatible server with N150 configuration.
 * Creates a custom starter script that registers TT models and uses local model path.
 */
async function startVllmServer(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  // Create starter script if it doesn't exist
  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');
  const starterPath = path.join(scratchpadDir, 'start-vllm-server.py');

  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  if (!fs.existsSync(starterPath)) {
    const extensionPath = extensionContext.extensionPath;
    const templatePath = path.join(extensionPath, 'content', 'templates', 'start-vllm-server.py');

    if (fs.existsSync(templatePath)) {
      fs.copyFileSync(templatePath, starterPath);
      fs.chmodSync(starterPath, 0o755);
    }
  }

  const modelPath = await getModelBasePath();
  const terminal = getOrCreateTerminal('vllm-server');

  const command = `cd ~/tt-vllm && source ~/tt-vllm-venv/bin/activate && export TT_METAL_HOME=~/tt-metal && export MESH_DEVICE=N150 && export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH && source ~/tt-vllm/tt_metal/setup-metal.sh && python ~/tt-scratchpad/start-vllm-server.py --model ${modelPath} --host 0.0.0.0 --port 8000 --max-model-len 8192 --max-num-seqs 4 --block-size 64`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Starting vLLM server on N150 with 8K context (memory-optimized). First load takes 2-5 minutes...'
  );
}

/**
 * Hardware configuration map for vLLM server
 * Maps hardware names to their optimal vLLM settings
 */
const VLLM_HARDWARE_CONFIGS = {
  N150: {
    maxModelLen: 8192,
    maxNumSeqs: 4,
    blockSize: 64,
    tensorParallelSize: undefined,
    archName: undefined,
  },
  N300: {
    maxModelLen: 131072,
    maxNumSeqs: 32,
    blockSize: 64,
    tensorParallelSize: 2,
    archName: undefined,
  },
  T3K: {
    maxModelLen: 131072,
    maxNumSeqs: 64,
    blockSize: 64,
    tensorParallelSize: 8,
    archName: undefined,
  },
  P100: {
    maxModelLen: 8192,
    maxNumSeqs: 4,
    blockSize: 64,
    tensorParallelSize: undefined,
    archName: 'blackhole',
  },
} as const;

/**
 * Command: tenstorrent.startVllmServerWithHardware
 * Parameterized command to start vLLM server for any hardware.
 * This command accepts hardware as an argument, enabling DRY markdown lessons.
 *
 * Usage from markdown:
 * [Start N150](command:tenstorrent.startVllmServerWithHardware?%7B%22hardware%22%3A%22N150%22%7D)
 * [Start N300](command:tenstorrent.startVllmServerWithHardware?%7B%22hardware%22%3A%22N300%22%7D)
 * [Start T3K](command:tenstorrent.startVllmServerWithHardware?%7B%22hardware%22%3A%22T3K%22%7D)
 * [Start P100](command:tenstorrent.startVllmServerWithHardware?%7B%22hardware%22%3A%22P100%22%7D)
 *
 * @param args - Optional arguments object with hardware type
 */
async function startVllmServerWithHardware(args?: { hardware?: string }): Promise<void> {
  const hardware = (args?.hardware || 'N150') as keyof typeof VLLM_HARDWARE_CONFIGS;

  const config = VLLM_HARDWARE_CONFIGS[hardware];
  if (!config) {
    vscode.window.showErrorMessage(
      `Unknown hardware type: ${hardware}. Valid options: ${Object.keys(VLLM_HARDWARE_CONFIGS).join(', ')}`
    );
    return;
  }

  await startVllmServerForHardware(hardware, config);
}

/**
 * Helper function to start vLLM server with hardware-specific configuration.
 */
async function startVllmServerForHardware(
  hardware: string,
  config: {
    maxModelLen: number;
    maxNumSeqs: number;
    blockSize: number;
    tensorParallelSize?: number;
    archName?: string;
    modelPath?: string;  // Optional: override default Llama path
    maxNumBatchedTokens?: number;  // Optional: prevents batch size errors (required for Qwen)
  }
): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  // Create starter script if it doesn't exist
  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');
  const starterPath = path.join(scratchpadDir, 'start-vllm-server.py');

  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  if (!fs.existsSync(starterPath)) {
    const extensionPath = extensionContext.extensionPath;
    const templatePath = path.join(extensionPath, 'content', 'templates', 'start-vllm-server.py');

    if (fs.existsSync(templatePath)) {
      fs.copyFileSync(templatePath, starterPath);
      fs.chmodSync(starterPath, 0o755);
    }
  }

  const modelPath = config.modelPath || await getModelBasePath();
  const terminal = getOrCreateTerminal('vllm-server');

  // Build environment variables
  let envVars = `export TT_METAL_HOME=~/tt-metal && export MESH_DEVICE=${hardware} && export PYTHONPATH=$TT_METAL_HOME:$PYTHONPATH`;
  if (config.archName) {
    envVars += ` && export TT_METAL_ARCH_NAME=${config.archName}`;
  }

  // Build vLLM flags
  let vllmFlags = `--model ${modelPath} --host 0.0.0.0 --port 8000 --max-model-len ${config.maxModelLen} --max-num-seqs ${config.maxNumSeqs} --block-size ${config.blockSize}`;
  if (config.maxNumBatchedTokens) {
    vllmFlags += ` --max-num-batched-tokens ${config.maxNumBatchedTokens}`;
  }
  if (config.tensorParallelSize) {
    vllmFlags += ` --tensor-parallel-size ${config.tensorParallelSize}`;
  }

  const command = `cd ~/tt-vllm && source ~/tt-vllm-venv/bin/activate && ${envVars} && source ~/tt-vllm/tt_metal/setup-metal.sh && python ~/tt-scratchpad/start-vllm-server.py ${vllmFlags}`;

  runInTerminal(terminal, command);

  const modelName = config.modelPath ? path.basename(modelPath) : 'Llama-3.1-8B';
  const contextInfo = config.tensorParallelSize
    ? `${hardware} with ${modelName}, ${config.maxModelLen / 1024}K context, TP=${config.tensorParallelSize}`
    : `${hardware} with ${modelName}, ${config.maxModelLen / 1024}K context`;

  vscode.window.showInformationMessage(
    `🚀 Starting vLLM server on ${contextInfo}. First load takes 2-5 minutes...`
  );
}

/**
 * Command: tenstorrent.testVllmOpenai
 * Tests vLLM with OpenAI SDK
 */
function testVllmOpenai(): void {
  const terminal = getOrCreateTerminal('explore');
  const modelConfig = getModelConfig(DEFAULT_MODEL_KEY);

  const command = `python3 -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='dummy'); response = client.chat.completions.create(model='${modelConfig.huggingfaceId}', messages=[{'role': 'user', 'content': 'What is machine learning?'}], max_tokens=128); print(response.choices[0].message.content)"`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Testing vLLM with OpenAI SDK. Check terminal for response!'
  );
}

/**
 * Command: tenstorrent.testVllmCurl
 * Tests vLLM with curl
 */
function testVllmCurl(): void {
  const terminal = getOrCreateTerminal('explore');
  const modelConfig = getModelConfig(DEFAULT_MODEL_KEY);

  const command = `curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model": "${modelConfig.huggingfaceId}", "messages": [{"role": "user", "content": "Explain neural networks"}], "max_tokens": 128}'`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    'Testing vLLM with curl. Check terminal for OpenAI-formatted response!'
  );
}

// ============================================================================
// Lesson 7 - VSCode Chat Integration
// ============================================================================

/**
 * Helper function to check if a file should be overwritten.
 * Shows a confirmation dialog if the file exists.
 *
 * @param filePath - The path to check
 * @returns true if the file should be written (doesn't exist or user confirmed overwrite)
 */
async function shouldOverwriteFile(filePath: string): Promise<boolean> {
  const fs = await import('fs');

  if (!fs.existsSync(filePath)) {
    return true; // File doesn't exist, safe to write
  }

  // File exists, ask user
  const choice = await vscode.window.showWarningMessage(
    `File already exists: ${filePath}\n\nDo you want to overwrite it?`,
    'Overwrite',
    'Cancel'
  );

  return choice === 'Overwrite';
}

/**
 * Command: tenstorrent.createVllmStarter
 * Creates the vLLM starter script in ~/tt-scratchpad/ without starting the server.
 * This script registers TT models with vLLM before starting the API server.
 */
async function createVllmStarter(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');
  const starterPath = path.join(scratchpadDir, 'start-vllm-server.py');

  // Create directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(starterPath))) {
    return; // User cancelled
  }

  // Copy template
  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'start-vllm-server.py');

  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage('❌ Template not found: start-vllm-server.py');
    return;
  }

  fs.copyFileSync(templatePath, starterPath);
  fs.chmodSync(starterPath, 0o755);

  const selection = await vscode.window.showInformationMessage(
    `✅ Created vLLM starter script at ${starterPath}`,
    'Open File',
    'Show in Folder'
  );

  if (selection === 'Open File') {
    const doc = await vscode.workspace.openTextDocument(starterPath);
    await vscode.window.showTextDocument(doc);
  } else if (selection === 'Show in Folder') {
    await vscode.commands.executeCommand('revealFileInOS', vscode.Uri.file(starterPath));
  }
}

/**
 * Command: tenstorrent.generateRetroImage
 * Generates a sample retro-style image using SD 3.5 Large on TT hardware
 */
async function generateRetroImage(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const terminal = getOrCreateTerminal('tt-metal');

  const command = replaceVariables(TERMINAL_COMMANDS.GENERATE_RETRO_IMAGE.template, {
    ttMetalPath,
  });

  runInTerminal(terminal, command);

  // Image will be saved to ~/tt-scratchpad/sd35_1024_1024.png
  const scratchpadPath = path.join(homeDir, 'tt-scratchpad');
  const imagePath = path.join(scratchpadPath, 'sd35_1024_1024.png');

  vscode.window.showInformationMessage(
    '🎨 Generating 1024x1024 image with Stable Diffusion 3.5 Large on TT hardware. First run downloads the model (~10 GB) and may take 5-10 minutes. Subsequent generations: ~12-15 seconds on N150. Image will auto-open when ready!'
  );

  // Automatically watch for the image and open it when created/updated
  const watcher = watchForImage(imagePath, 600000); // 10 minute timeout for first run
  extensionContext.subscriptions.push(watcher);
}

/**
 * Helper function to open a generated image in VSCode
 */
async function openGeneratedImage(imagePath: string): Promise<void> {
  const fs = await import('fs');

  if (!fs.existsSync(imagePath)) {
    vscode.window.showWarningMessage(
      `Image not found at ${imagePath}. Make sure generation completed successfully.`
    );
    return;
  }

  try {
    // Show in image preview panel
    const imagePreviewProvider = (global as any).imagePreviewProvider;
    if (imagePreviewProvider) {
      imagePreviewProvider.showImage(imagePath);
      vscode.window.showInformationMessage(`✅ Image displayed in Output Preview panel`);
    } else {
      vscode.window.showWarningMessage('Image preview panel not available');
    }
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to display image: ${error}`);
  }
}

/**
 * Watch for an image file to be created/updated and automatically open it
 * Returns a disposable to cancel the watch
 */
function watchForImage(imagePath: string, timeoutMs: number = 30000): vscode.Disposable {
  const fs = require('fs');
  const path = require('path');
  
  let watcher: vscode.Disposable | undefined;
  let timeoutHandle: NodeJS.Timeout | undefined;
  let initialMtime: number | undefined;

  // Get initial modification time if file exists
  if (fs.existsSync(imagePath)) {
    initialMtime = fs.statSync(imagePath).mtimeMs;
  }

  const checkAndOpen = async () => {
    if (fs.existsSync(imagePath)) {
      const currentMtime = fs.statSync(imagePath).mtimeMs;
      // Only open if file is new or has been modified since we started watching
      if (initialMtime === undefined || currentMtime > initialMtime) {
        cleanup();
        await openGeneratedImage(imagePath);
      }
    }
  };

  const cleanup = () => {
    if (watcher) {
      watcher.dispose();
      watcher = undefined;
    }
    if (timeoutHandle) {
      clearTimeout(timeoutHandle);
      timeoutHandle = undefined;
    }
  };

  // Watch the directory for changes
  const dir = path.dirname(imagePath);
  const filename = path.basename(imagePath);
  
  if (fs.existsSync(dir)) {
    const fileSystemWatcher = vscode.workspace.createFileSystemWatcher(
      new vscode.RelativePattern(dir, filename)
    );

    fileSystemWatcher.onDidCreate(checkAndOpen);
    fileSystemWatcher.onDidChange(checkAndOpen);
    
    watcher = fileSystemWatcher;
  }

  // Set timeout
  timeoutHandle = setTimeout(() => {
    cleanup();
  }, timeoutMs);

  return new vscode.Disposable(cleanup);
}

/**
 * Helper to open an image using CLI (code/code-insiders command)
 * Useful as fallback in environments where webview APIs don't work
 * Currently unused but available for future use
 */
/* 
async function openImageViaCLI(imagePath: string): Promise<void> {
  const fs = require('fs');
  
  if (!fs.existsSync(imagePath)) {
    vscode.window.showWarningMessage(`Image not found: ${imagePath}`);
    return;
  }

  // Try to determine which VS Code CLI is available
  const terminal = getOrCreateTerminal('tt-metal');
  
  // Try code-insiders first (common for remote), then code
  const command = `(command -v code-insiders >/dev/null 2>&1 && code-insiders "${imagePath}") || (command -v code >/dev/null 2>&1 && code "${imagePath}") || echo "⚠️  Could not find 'code' or 'code-insiders' command. Please open ${imagePath} manually."`;
  
  runInTerminal(terminal, command);
}
*/

/**
 * Command: tenstorrent.startInteractiveImageGen
 * Starts interactive SD 3.5 mode where users can enter custom prompts
 */
async function startInteractiveImageGen(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const terminal = getOrCreateTerminal('tt-metal');

  const command = replaceVariables(TERMINAL_COMMANDS.START_INTERACTIVE_IMAGE_GEN.template, {
    ttMetalPath,
  });

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🖼️ Starting interactive SD 3.5 Large. Model loads once (2-5 min), then enter custom prompts to generate 1024x1024 images (~12-15 sec each on N150)! Images will be saved to ~/tt-scratchpad/sd35_1024_1024.png'
  );
}

/**
 * Command: tenstorrent.copyImageGenDemo
 * Copies the SD 3.5 demo.py to ~/tt-scratchpad for experimentation
 */
async function copyImageGenDemo(): Promise<void> {
  // Get the tt-metal path from stored state (default to ~/tt-metal if not found)
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const terminal = getOrCreateTerminal('tt-metal');

  const command = replaceVariables(TERMINAL_COMMANDS.COPY_IMAGE_GEN_DEMO.template, {
    ttMetalPath,
  });

  runInTerminal(terminal, command);

  // Open the copied file for editing
  const scratchpadPath = path.join(homeDir, 'tt-scratchpad', 'sd35_demo.py');

  // Wait a moment for the file to be copied
  setTimeout(async () => {
    try {
      const doc = await vscode.workspace.openTextDocument(scratchpadPath);
      await vscode.window.showTextDocument(doc);

      vscode.window.showInformationMessage(
        '📝 SD 3.5 demo copied to ~/tt-scratchpad/sd35_demo.py! Experiment with prompts, parameters, batch generation, and more. The original in tt-metal remains untouched.'
      );
    } catch (error) {
      vscode.window.showWarningMessage(
        `Demo copied but couldn't auto-open. Find it at: ~/tt-scratchpad/sd35_demo.py`
      );
    }
  }, 500);
}

// ============================================================================
// Lesson 9 - Coding Assistant with Prompt Engineering
// ============================================================================

/**
 * Command: tenstorrent.openLatestImage
 * Opens the most recently generated image from ~/tt-scratchpad
 */
async function openLatestImage(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const fs = await import('fs');
  const homeDir = os.homedir();
  const scratchpadPath = path.join(homeDir, 'tt-scratchpad');

  if (!fs.existsSync(scratchpadPath)) {
    vscode.window.showWarningMessage('No images found. Generate an image first!');
    return;
  }

  // Find all image files
  const files = fs.readdirSync(scratchpadPath)
    .filter(f => /\.(png|jpg|jpeg)$/i.test(f))
    .map(f => ({
      name: f,
      path: path.join(scratchpadPath, f),
      mtime: fs.statSync(path.join(scratchpadPath, f)).mtimeMs
    }))
    .sort((a, b) => b.mtime - a.mtime); // Sort by modification time, newest first

  if (files.length === 0) {
    vscode.window.showWarningMessage('No images found in ~/tt-scratchpad. Generate an image first!');
    return;
  }

  // If multiple images, let user pick
  if (files.length > 1) {
    const selected = await vscode.window.showQuickPick(
      files.map(f => ({
        label: f.name,
        description: new Date(f.mtime).toLocaleString(),
        path: f.path
      })),
      { placeHolder: 'Select an image to open' }
    );

    if (selected) {
      await openGeneratedImage(selected.path);
    }
  } else {
    // Just open the only image
    await openGeneratedImage(files[0].path);
  }
}

/**
 * Command: tenstorrent.verifyCodingModel
 * Verifies Llama 3.1 8B model is available for coding assistant
 */
function verifyCodingModel(): void {
  const terminal = getOrCreateTerminal('explore');
  const command = TERMINAL_COMMANDS.VERIFY_CODING_MODEL.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🔍 Checking for Llama 3.1 8B model. Should be installed from Lesson 3.'
  );
}

/**
 * Command: tenstorrent.createCodingAssistantScript
 * Creates the coding assistant CLI script and opens it in editor
 */
async function createCodingAssistantScript(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-coding-assistant.py');

  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  // Create scratchpad directory if it doesn't exist
  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-coding-assistant.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    fs.copyFileSync(templatePath, destPath);
    fs.chmodSync(destPath, 0o755);

    // Open the file in the editor (to the side, preserving focus on terminal)
    const doc = await vscode.workspace.openTextDocument(destPath);
    await vscode.window.showTextDocument(doc, {
      viewColumn: vscode.ViewColumn.Beside,
      preserveFocus: true,
      preview: false
    });

    vscode.window.showInformationMessage(
      `✅ Created coding assistant script at ${destPath}. The file is now open - review the code!`
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to create coding assistant script: ${error}`);
  }
}

/**
 * Command: tenstorrent.startCodingAssistant
 * Starts the coding assistant CLI using Llama 3.1 8B with Direct API and prompt engineering
 */
async function startCodingAssistant(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  // Model path for Llama 3.1 8B (original subdirectory for Direct API)
  const modelPath = await getModelOriginalPath();

  const terminal = getOrCreateTerminal('api-server');

  const command = `cd "${ttMetalPath}" && export LLAMA_DIR="${modelPath}" && export PYTHONPATH=$(pwd) && python3 ~/tt-scratchpad/tt-coding-assistant.py`;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '💬 Starting coding assistant with prompt engineering. Model loads once (2-5 min), then fast responses (1-3 sec)!'
  );
}

// ============================================================================
// TT-Forge Commands (Lesson 11)
// ============================================================================

/**
 * Command: tenstorrent.buildForgeFromSource
 * Builds TT-Forge from source against your tt-metal installation
 */
function buildForgeFromSource(): void {
  const terminal = getOrCreateTerminal('tt-forge');
  const command = TERMINAL_COMMANDS.BUILD_FORGE_FROM_SOURCE.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🔨 Building TT-Forge from source (10-20 min). This ensures compatibility with your tt-metal!'
  );
}

/**
 * Command: tenstorrent.installForge
 * Installs TT-Forge-FE wheels (quick but may have version issues)
 */
function installForge(): void {
  const terminal = getOrCreateTerminal('tt-forge');
  const command = TERMINAL_COMMANDS.INSTALL_FORGE.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '📦 Installing TT-Forge wheels. If you get symbol errors, try building from source instead.'
  );
}

/**
 * Command: tenstorrent.testForgeInstall
 * Tests forge installation and device detection
 */
function testForgeInstall(): void {
  const terminal = getOrCreateTerminal('tt-forge');
  const command = TERMINAL_COMMANDS.TEST_FORGE_INSTALL.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🔍 Testing forge installation. Check terminal for version and device status.'
  );
}

/**
 * Command: tenstorrent.createForgeClassifier
 * Copies tt-forge-classifier.py template to ~/tt-scratchpad and opens it
 */
async function createForgeClassifier(): Promise<void> {
  const path = await import('path');
  const fs = await import('fs');
  const os = await import('os');

  const extensionPath = extensionContext.extensionPath;
  const templatePath = path.join(extensionPath, 'content', 'templates', 'tt-forge-classifier.py');

  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Template not found at ${templatePath}. Please reinstall the extension.`
    );
    return;
  }

  const homeDir = os.homedir();
  const scratchpadDir = path.join(homeDir, 'tt-scratchpad');

  if (!fs.existsSync(scratchpadDir)) {
    fs.mkdirSync(scratchpadDir, { recursive: true });
  }

  const destPath = path.join(scratchpadDir, 'tt-forge-classifier.py');

  // Check if file exists and ask for overwrite confirmation
  if (!(await shouldOverwriteFile(destPath))) {
    return; // User cancelled
  }

  try {
    fs.copyFileSync(templatePath, destPath);
    fs.chmodSync(destPath, 0o755);

    // Open in editor (to the side, preserving focus on terminal)
    const doc = await vscode.workspace.openTextDocument(destPath);
    await vscode.window.showTextDocument(doc, {
      viewColumn: vscode.ViewColumn.Beside,
      preserveFocus: true,
      preview: false
    });

    vscode.window.showInformationMessage(
      '✅ Created tt-forge-classifier.py. Review the MobileNetV2 implementation!'
    );
  } catch (error) {
    vscode.window.showErrorMessage(`Failed to create classifier script: ${error}`);
  }
}

/**
 * Command: tenstorrent.runForgeClassifier
 * Runs MobileNetV2 image classification on sample image
 */
function runForgeClassifier(): void {
  const terminal = getOrCreateTerminal('tt-forge');
  const command = TERMINAL_COMMANDS.RUN_FORGE_CLASSIFIER.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🎨 Running image classifier. First compilation takes 2-5 min, then inference is fast!'
  );
}

// ============================================================================
// Lesson 12 - TT-XLA JAX Integration Commands
// ============================================================================

/**
 * Command: tenstorrent.installTtXla
 * Installs TT-XLA PJRT plugin with JAX support
 */
function installTtXla(): void {
  const terminal = getOrCreateTerminal('tt-xla');
  const command = TERMINAL_COMMANDS.INSTALL_TT_XLA.template;

  runInTerminal(terminal, command);

  vscode.window.showInformationMessage(
    '🚀 Installing TT-XLA PJRT plugin with JAX. This may take a few minutes...'
  );
}

/**
 * Command: tenstorrent.testTtXlaInstall
 * Creates test script and runs it to verify TT-XLA installation
 */
function testTtXlaInstall(): void {
  const terminal = getOrCreateTerminal('tt-xla');

  // First create the test script
  const createCommand = TERMINAL_COMMANDS.CREATE_TT_XLA_TEST.template;
  runInTerminal(terminal, createCommand);

  // Then run it
  setTimeout(() => {
    const testCommand = TERMINAL_COMMANDS.TEST_TT_XLA_INSTALL.template;
    runInTerminal(terminal, testCommand);

    vscode.window.showInformationMessage(
      '🧪 Testing TT-XLA installation. You should see TtDevice in the output!'
    );
  }, 1000);
}

/**
 * Command: tenstorrent.runTtXlaDemo
 * Downloads and runs official GPT-2 demo with TT-XLA
 */
function runTtXlaDemo(): void {
  const terminal = getOrCreateTerminal('tt-xla');

  // First download the demo
  const downloadCommand = TERMINAL_COMMANDS.DOWNLOAD_TT_XLA_DEMO.template;
  runInTerminal(terminal, downloadCommand);

  // Then run it after a short delay
  setTimeout(() => {
    const runCommand = TERMINAL_COMMANDS.RUN_TT_XLA_DEMO.template;
    runInTerminal(terminal, runCommand);

    vscode.window.showInformationMessage(
      '🎯 Running GPT-2 demo on TT hardware via JAX. First run may take a few minutes!'
    );
  }, 2000);
}

// ============================================================================
// RISC-V Programming Commands (Lesson 13)
// ============================================================================

/**
 * Command: tenstorrent.buildProgrammingExamples
 * Builds tt-metal with programming examples including RISC-V demonstrations
 */
function buildProgrammingExamples(): void {
  const terminal = getOrCreateTerminal('tt-metal');

  const buildCommand = TERMINAL_COMMANDS.BUILD_PROGRAMMING_EXAMPLES.template;
  runInTerminal(terminal, buildCommand);

  vscode.window.showInformationMessage(
    '🔨 Building tt-metal with programming examples. This will take 5-10 minutes...'
  );
}

/**
 * Command: tenstorrent.runRiscvExample
 * Runs the RISC-V addition example on BRISC processor
 */
function runRiscvExample(): void {
  const terminal = getOrCreateTerminal('tt-metal');

  const runCommand = TERMINAL_COMMANDS.RUN_RISCV_EXAMPLE.template;
  runInTerminal(terminal, runCommand);

  vscode.window.showInformationMessage(
    '🚀 Running RISC-V addition example on BRISC processor. Watch for "Success: Result is 21"!'
  );
}

/**
 * Command: tenstorrent.openRiscvKernel
 * Opens the RISC-V kernel source code in VS Code
 */
async function openRiscvKernel(): Promise<void> {
  const kernelPath = `${process.env.HOME}/tt-metal/tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp`;

  try {
    const doc = await vscode.workspace.openTextDocument(kernelPath);
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage(
      '📖 Opened RISC-V kernel source. Look for the addition at line 41!'
    );
  } catch (error) {
    vscode.window.showErrorMessage(
      `❌ Could not open kernel file: ${kernelPath}. Make sure tt-metal is cloned to ~/tt-metal`
    );
  }
}

/**
 * Command: tenstorrent.openRiscvGuide
 * Opens the comprehensive RISC-V exploration guide
 */
async function openRiscvGuide(): Promise<void> {
  const guidePath = `${process.env.HOME}/code/tt-vscode-ext-clean/RISC-V_EXPLORATION.md`;

  try {
    const doc = await vscode.workspace.openTextDocument(guidePath);
    await vscode.window.showTextDocument(doc);
    vscode.window.showInformationMessage(
      '📘 Opened RISC-V Exploration Guide. Dive deep into 880 RISC-V cores!'
    );
  } catch (error) {
    vscode.window.showErrorMessage(
      `❌ Could not open guide: ${guidePath}`
    );
  }
}

// ============================================================================
// Bounty Program Commands
// ============================================================================

/**
 * Command: tenstorrent.browseOpenBounties
 * Opens GitHub issues page filtered for bounty label
 */
function browseOpenBounties(): void {
  const bountyUrl = 'https://github.com/tenstorrent/tt-metal/labels/bounty';
  vscode.env.openExternal(vscode.Uri.parse(bountyUrl));

  vscode.window.showInformationMessage(
    '🎯 Opening GitHub bounties page in your browser...'
  );
}

/**
 * Command: tenstorrent.copyBountyChecklist
 * Creates a bounty workflow checklist file in ~/tt-scratchpad/
 */
async function copyBountyChecklist(): Promise<void> {
  const scratchpadDir = `${process.env.HOME}/tt-scratchpad`;
  const checklistPath = `${scratchpadDir}/bounty-checklist.md`;

  const checklistContent = `# Bounty Program Workflow Checklist

## Phase 1: Setup & Preparation
- [ ] Find and claim a bounty on GitHub
- [ ] Clone tt-metal repository
- [ ] Build TT-Metal with ./build_metal.sh
- [ ] Set environment variables (TT_METAL_HOME, PYTHONPATH)
- [ ] Install Python dependencies
- [ ] Verify hardware with tt-smi
- [ ] Run a reference demo (e.g., Llama 3.1 8B)

## Phase 2: Baseline Validation
- [ ] Run reference model on CPU/GPU
- [ ] Validate model outputs
- [ ] Save reference logits for comparison
- [ ] Analyze model architecture (config.json)
- [ ] Check architecture compatibility with tt_transformers
- [ ] Verify model fits on target hardware
- [ ] Identify special features (RoPE, attention mechanisms)

## Phase 3: Component-Wise Bring-Up
- [ ] Identify similar models in tt_transformers
- [ ] Create unit test for RMSNorm/LayerNorm
- [ ] Create unit test for RotaryEmbedding (RoPE)
- [ ] Create unit test for Attention module
- [ ] Create unit test for MLP (feed-forward)
- [ ] Create unit test for full decoder layer
- [ ] Implement model-specific modifications (minimal!)
- [ ] Verify PCC (Pearson Correlation) >0.99 for each module

## Phase 4: Full Model Integration
- [ ] Implement decode stage (batch=32, single token)
- [ ] Test decode end-to-end
- [ ] Implement prefill stage (batch=1, long context)
- [ ] Test prefill end-to-end
- [ ] Run full generation (prefill + decode)
- [ ] Validate token accuracy vs reference
- [ ] Run teacher forcing test (top-1 >80%, top-5 >95%)
- [ ] Check generated text is coherent

## Phase 5: Performance Optimization
- [ ] Measure baseline performance (TTFT, throughput, latency)
- [ ] Calculate percentage of theoretical max
- [ ] Try different precision configs (bfp4, bfp8, bf16)
- [ ] Create custom decoder config if needed
- [ ] Apply Metal Trace for command buffer optimization
- [ ] Enable async mode if beneficial
- [ ] Profile with Tracy profiler
- [ ] Identify and fix bottlenecks
- [ ] Document final performance metrics

## Phase 6: Testing & CI Integration
- [ ] Create accuracy test (test_accuracy.py)
- [ ] Create performance test (test_perf.py)
- [ ] Create demo test (simple_text_demo.py)
- [ ] Generate reference outputs for CI
- [ ] Add model to CI test dispatch
- [ ] Run all tests locally
- [ ] Verify all tests pass

## Phase 7: Documentation & Submission
- [ ] Write or update README with setup instructions
- [ ] Document performance metrics (table format)
- [ ] Document accuracy results (top-1, top-5)
- [ ] Create pull request(s) following MODEL_ADD.md
- [ ] Write clear PR description with summary
- [ ] Link to bounty issue in PR
- [ ] Run CI pipelines (Pipeline Select)
- [ ] Respond to reviewer feedback promptly
- [ ] Make requested changes
- [ ] Get PR approved and merged
- [ ] Contribution complete! ✅

---

## Performance Tiers (Remember!)
- **Easy**: ≥25% of theoretical max throughput
- **Medium**: ≥50% of theoretical max throughput
- **Hard**: ≥70% of theoretical max throughput

## Accuracy Requirements
- **Top-1**: ≥80% token accuracy
- **Top-5**: ≥95% token accuracy

## Pro Tips
- ✅ Start with easy/warmup bounties to learn workflow
- ✅ Communicate progress in issue thread every few days
- ✅ Reuse existing tt_transformers code (reviewers love this!)
- ✅ Test incrementally - don't wait until the end
- ✅ Profile early to identify bottlenecks
- ✅ Break large PRs into smaller ones (easier to review)
- ❌ Don't copy-paste entire codebases
- ❌ Don't skip baseline validation
- ❌ Don't optimize before correctness
- ❌ Don't ignore CI failures

## Resources
- Model Bring-Up Guide: https://github.com/tenstorrent/tt-metal/blob/main/models/docs/model_bring_up.md
- TT-NN Docs: https://docs.tenstorrent.com/tt-metal/latest/ttnn/
- Bounty Terms: https://docs.tenstorrent.com/bounty_terms.html
- Discord: https://discord.gg/tenstorrent

---

Good luck with your contribution! 🚀
`;

  // Ensure scratchpad directory exists
  await vscode.workspace.fs.createDirectory(vscode.Uri.file(scratchpadDir));

  // Write checklist file
  await vscode.workspace.fs.writeFile(
    vscode.Uri.file(checklistPath),
    Buffer.from(checklistContent)
  );

  // Open the file in editor (to the side, preserving focus on terminal)
  const doc = await vscode.workspace.openTextDocument(checklistPath);
  await vscode.window.showTextDocument(doc, {
    viewColumn: vscode.ViewColumn.Beside,
    preserveFocus: true,
    preview: false
  });

  vscode.window.showInformationMessage(
    '✅ Bounty workflow checklist created! Check off items as you progress.'
  );
}

// ============================================================================
// Device Management Commands
// ============================================================================

/**
 * Command: tenstorrent.resetDevice
 * Performs a software reset of the Tenstorrent device using tt-smi -r
 */
async function resetDevice(): Promise<void> {
  const choice = await vscode.window.showWarningMessage(
    'This will reset the Tenstorrent device. Any running processes using the device will be interrupted. Continue?',
    'Reset Device',
    'Cancel'
  );

  if (choice !== 'Reset Device') {
    return;
  }

  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, 'tt-smi -r');

  vscode.window.showInformationMessage(
    '🔄 Resetting device... Check terminal for status.'
  );

  // Refresh status after a short delay
  setTimeout(async () => {
    await updateDeviceStatus();
  }, 3000);
}

/**
 * Command: tenstorrent.clearDeviceState
 * Clears device state by removing cached data and performing cleanup
 */
async function clearDeviceState(): Promise<void> {
  const choice = await vscode.window.showWarningMessage(
    'This will:\n• Kill any processes using the device\n• Clear /dev/shm Tenstorrent data\n• Reset device state\n\nThis operation requires sudo. Continue?',
    'Clear State',
    'Cancel'
  );

  if (choice !== 'Clear State') {
    return;
  }

  const terminal = getOrCreateTerminal('tt-metal');

  // Multi-step cleanup command
  const commands = [
    'echo "=== Killing processes using Tenstorrent devices ==="',
    'pgrep -f tt-metal && pkill -9 -f tt-metal || echo "No tt-metal processes found"',
    'pgrep -f vllm && pkill -9 -f vllm || echo "No vllm processes found"',
    'echo "=== Clearing /dev/shm ==="',
    'sudo rm -rf /dev/shm/tenstorrent* /dev/shm/tt_* || echo "No shared memory files found"',
    'echo "=== Resetting device ==="',
    'tt-smi -r',
    'echo "=== Cleanup complete ==="',
  ];

  runInTerminal(terminal, commands.join(' && '));

  vscode.window.showInformationMessage(
    '🧹 Clearing device state... Check terminal for progress. You may need to enter your sudo password.'
  );

  // Refresh status after cleanup
  setTimeout(async () => {
    await updateDeviceStatus();
  }, 5000);
}

// ============================================================================
// Welcome Page
// ============================================================================

/**
 * Command: tenstorrent.showWelcome
 *
 * Opens a welcome page in a webview panel with an overview of the extension,
 * links to all walkthroughs, and quick actions.
 */
async function showWelcome(context: vscode.ExtensionContext): Promise<void> {
  const panel = vscode.window.createWebviewPanel(
    'tenstorrentWelcome',
    'Welcome to Tenstorrent',
    { viewColumn: vscode.ViewColumn.One, preserveFocus: false },
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'content', 'pages')]
    }
  );

  // Load welcome HTML
  const fs = await import('fs');
  const path = await import('path');
  const welcomePath = path.join(context.extensionPath, 'content', 'pages', 'welcome.html');

  if (fs.existsSync(welcomePath)) {
    panel.webview.html = fs.readFileSync(welcomePath, 'utf8');
  } else {
    panel.webview.html = '<html><body><h1>Welcome to Tenstorrent</h1><p>Welcome content not found.</p></body></html>';
  }

  // Handle messages from the webview
  panel.webview.onDidReceiveMessage(
    async (message) => {
      switch (message.command) {
        case 'openWalkthrough':
          // Open lesson in new system (fallback to old walkthrough if stepId not found)
          const lessonId = message.stepId;
          await vscode.commands.executeCommand('tenstorrent.showLesson', lessonId);
          break;
        case 'executeCommand':
          // Execute a command by ID
          vscode.commands.executeCommand(message.commandId);
          break;
      }
    },
    undefined,
    context.subscriptions
  );
}

/**
 * Command: tenstorrent.showFaq
 *
 * Opens the FAQ in a rendered webview with search functionality.
 */
async function showFaq(context: vscode.ExtensionContext): Promise<void> {
  const panel = vscode.window.createWebviewPanel(
    'tenstorrentFaq',
    'Tenstorrent FAQ',
    { viewColumn: vscode.ViewColumn.One, preserveFocus: false },
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'content', 'pages')]
    }
  );

  const fs = await import('fs');
  const path = await import('path');

  try {
    // Read FAQ markdown
    const faqPath = path.join(context.extensionPath, 'content', 'pages', 'FAQ.md');
    const faqMarkdown = fs.readFileSync(faqPath, 'utf8');

    // Read template
    const templatePath = path.join(context.extensionPath, 'content', 'pages', 'faq-template.html');
    let template = fs.readFileSync(templatePath, 'utf8');

    // Convert markdown to HTML (simple conversion)
    const faqHtml = convertMarkdownToHtml(faqMarkdown);

    // Inject FAQ content into template
    panel.webview.html = template.replace('{{FAQ_CONTENT}}', faqHtml);
  } catch (error) {
    panel.webview.html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
          h1 { color: #4FD1C5; }
        </style>
      </head>
      <body>
        <h1>Error Loading FAQ</h1>
        <p>FAQ content could not be loaded. Please check the extension installation.</p>
        <p>Error: ${error}</p>
      </body>
      </html>
    `;
  }
}

/**
 * Simple markdown to HTML converter for FAQ content
 */
function convertMarkdownToHtml(markdown: string): string {
  let html = markdown;

  // Helper function to create URL-friendly slug from text
  // Matches the convention used in the FAQ TOC (e.g., "Remote Development & SSH" -> "remote-development--ssh")
  const slugify = (text: string): string => {
    return text
      .toLowerCase()
      .trim()
      .replace(/[^\w\s-]/g, '-') // Replace special characters with hyphens
      .replace(/\s+/g, '-')      // Replace spaces with hyphens
      .replace(/--+/g, '--')     // Normalize to max double hyphens (keep double from & replacement)
      .replace(/^-+|-+$/g, '');  // Remove leading/trailing hyphens
  };

  // Escape HTML characters first
  html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

  // Code blocks (```bash ... ```)
  html = html.replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    return `<pre><code class="language-${lang}">${code.trim()}</code></pre>`;
  });

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headers with proper slugified IDs
  html = html.replace(/^### (.+)$/gm, (_match, text) => {
    return `<h3 id="${slugify(text)}">${text}</h3>`;
  });
  html = html.replace(/^## (.+)$/gm, (_match, text) => {
    return `<h2 id="${slugify(text)}">${text}</h2>`;
  });
  html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

  // Bold and italic
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

  // Links [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>');

  // Unordered lists
  html = html.replace(/^- (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>.*<\/li>\n?)+/g, '<ul>$&</ul>');

  // Tables (basic support)
  const tableRegex = /\|(.+)\|\n\|[-:\s|]+\|\n((?:\|.+\|\n?)+)/g;
  html = html.replace(tableRegex, (_match, header, rows) => {
    const headerCells = header.split('|').filter((s: string) => s.trim()).map((h: string) => `<th>${h.trim()}</th>`).join('');
    const rowsHtml = rows.trim().split('\n').map((row: string) => {
      const cells = row.split('|').filter((s: string) => s.trim()).map((c: string) => `<td>${c.trim()}</td>`).join('');
      return `<tr>${cells}</tr>`;
    }).join('');
    return `<table><thead><tr>${headerCells}</tr></thead><tbody>${rowsHtml}</tbody></table>`;
  });

  // Paragraphs (lines separated by blank lines)
  html = html.split('\n\n').map(para => {
    para = para.trim();
    // Don't wrap if already wrapped in a tag
    if (para.match(/^<(h[1-6]|ul|ol|pre|table|div)/)) {
      return para;
    }
    // Split into lines but keep list items together
    if (para.includes('<li>')) {
      return para;
    }
    return para ? `<p>${para.replace(/\n/g, '<br>')}</p>` : '';
  }).join('\n');

  // Q: and A: formatting for FAQ blocks
  html = html.replace(/<h3>Q: (.+?)<\/h3>\s*<p><strong>A:<\/strong>(.+?)<\/p>/gs,
    '<div class="qa-block"><h3>Q: $1</h3><p><strong>A:</strong>$2</p></div>');

  // Sections (wrap each h2 and its content)
  const sections = html.split(/(?=<h2)/g);
  html = sections.map(section => {
    if (section.trim().startsWith('<h2')) {
      return `<div class="faq-section">${section}</div>`;
    }
    return section;
  }).join('');

  // Checkmarks and cross marks
  html = html.replace(/✅/g, '<span class="checkmark">✅</span>');
  html = html.replace(/❌/g, '<span class="crossmark">❌</span>');
  html = html.replace(/⚠️/g, '<span class="warning">⚠️</span>');

  return html;
}

/**
 * Command: tenstorrent.showRiscvGuide
 *
 * Opens the RISC-V Exploration Guide in a rendered webview with search functionality.
 */
async function showRiscvGuide(context: vscode.ExtensionContext): Promise<void> {
  const panel = vscode.window.createWebviewPanel(
    'tenstorrentRiscvGuide',
    'RISC-V Exploration Guide',
    { viewColumn: vscode.ViewColumn.One, preserveFocus: false },
    {
      enableScripts: true,
      localResourceRoots: [vscode.Uri.joinPath(context.extensionUri, 'content', 'pages')]
    }
  );

  const fs = await import('fs');
  const path = await import('path');

  try {
    // Read RISC-V guide markdown
    const guidePath = path.join(context.extensionPath, 'content', 'pages', 'riscv-guide.md');
    const guideMarkdown = fs.readFileSync(guidePath, 'utf8');

    // Read template (reuse FAQ template)
    const templatePath = path.join(context.extensionPath, 'content', 'pages', 'faq-template.html');
    let template = fs.readFileSync(templatePath, 'utf8');

    // Convert markdown to HTML
    const guideHtml = convertMarkdownToHtml(guideMarkdown);

    // Inject content into template (update title in template)
    template = template.replace('Tenstorrent FAQ', 'RISC-V Exploration Guide');
    template = template.replace('Frequently Asked Questions', 'Exploring Tenstorrent as a RISC-V Platform');
    panel.webview.html = template.replace('{{FAQ_CONTENT}}', guideHtml);
  } catch (error) {
    panel.webview.html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body { font-family: var(--vscode-font-family); padding: 20px; color: var(--vscode-foreground); }
          h1 { color: #4FD1C5; }
        </style>
      </head>
      <body>
        <h1>Error Loading RISC-V Guide</h1>
        <p>RISC-V Exploration Guide could not be loaded. Please check the extension installation.</p>
        <p>Error: ${error}</p>
      </body>
      </html>
    `;
  }
}

// ============================================================================
// Walkthrough Management
// ============================================================================

/**
 * Command: tenstorrent.openWalkthrough
 *
 * Opens (or reopens) the Tenstorrent setup walkthrough.
 * This allows users to access the walkthrough at any time from the Command Palette.
 */
function openWalkthrough(): void {
  // Open the walkthrough using VS Code's built-in command
  // Format: publisher.extensionName#walkthroughId
  vscode.commands.executeCommand(
    'workbench.action.openWalkthrough',
    'tenstorrent.tenstorrent-developer-extension#tenstorrent.setup',
    false
  );

  vscode.window.showInformationMessage('Opening Tenstorrent Setup Walkthrough...');
}

/**
 * Command: tenstorrent.resetProgress
 *
 * Resets walkthrough progress by clearing stored paths and state.
 * This allows users to start the walkthrough from scratch.
 */
async function resetProgress(): Promise<void> {
  const choice = await vscode.window.showWarningMessage(
    'This will reset your walkthrough progress and clear stored paths. Continue?',
    'Reset Progress',
    'Cancel'
  );

  if (choice === 'Reset Progress') {
    // Clear all stored state
    await extensionContext.globalState.update(STATE_KEYS.TT_METAL_PATH, undefined);
    await extensionContext.globalState.update(STATE_KEYS.MODEL_PATH, undefined);

    vscode.window.showInformationMessage(
      '✓ Walkthrough progress reset. You can now start from the beginning.'
    );

    // Optionally, reopen the walkthrough
    openWalkthrough();
  }
}

// ============================================================================
// Lesson 11 Commands - Exploring TT-Metalium
// ============================================================================

/**
 * Command: tenstorrent.launchTtnnTutorials
 *
 * Opens the TTNN tutorials directory in the tt-metal repository.
 * Launches Jupyter notebooks for interactive learning.
 */
async function launchTtnnTutorials(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const tutorialsPath = path.join(ttMetalPath, 'ttnn', 'tutorials');

  // Check if tt-metal exists
  const fs = await import('fs');
  if (!fs.existsSync(ttMetalPath)) {
    const choice = await vscode.window.showWarningMessage(
      'tt-metal repository not found. Would you like to clone it first?',
      'Clone tt-metal',
      'Cancel'
    );

    if (choice === 'Clone tt-metal') {
      await cloneTTMetal();
    }
    return;
  }

  // Check if tutorials directory exists
  if (!fs.existsSync(tutorialsPath)) {
    vscode.window.showErrorMessage(
      `Tutorials directory not found at ${tutorialsPath}. Please ensure tt-metal is up to date.`
    );
    return;
  }

  // Configure Jupyter to use tt-metal python_env
  const vscodePath = path.join(ttMetalPath, '.vscode');
  const settingsPath = path.join(vscodePath, 'settings.json');
  const pythonEnvPath = path.join(ttMetalPath, 'python_env', 'bin', 'python');

  try {
    // Create .vscode directory if it doesn't exist
    if (!fs.existsSync(vscodePath)) {
      fs.mkdirSync(vscodePath, { recursive: true });
    }

    // Read existing settings or create new
    let settings: any = {};
    if (fs.existsSync(settingsPath)) {
      const content = fs.readFileSync(settingsPath, 'utf-8');
      settings = JSON.parse(content);
    }

    // Set Python interpreter for Jupyter notebooks
    settings['python.defaultInterpreterPath'] = pythonEnvPath;
    settings['jupyter.kernels.filter'] = [
      {
        "path": pythonEnvPath,
        "type": "pythonEnvironment"
      }
    ];

    // Write settings
    fs.writeFileSync(settingsPath, JSON.stringify(settings, null, 2));

    vscode.window.showInformationMessage(
      `✅ Configured Jupyter to use ${pythonEnvPath}`
    );
  } catch (error) {
    console.error('Failed to configure Jupyter settings:', error);
  }

  // Open the tutorials folder in VS Code
  const uri = vscode.Uri.file(tutorialsPath);
  await vscode.commands.executeCommand('vscode.openFolder', uri, { forceNewWindow: false });

  vscode.window.showInformationMessage(
    `📓 Opening TTNN Tutorials folder. Start with 001.ipynb for tensor basics!`
  );
}

/**
 * Command: tenstorrent.browseModelZoo
 *
 * Opens the model zoo directory and displays information about available demos.
 */
async function browseModelZoo(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const modelZooPath = path.join(ttMetalPath, 'models', 'demos');

  // Check if tt-metal exists
  const fs = await import('fs');
  if (!fs.existsSync(ttMetalPath)) {
    const choice = await vscode.window.showWarningMessage(
      'tt-metal repository not found. Would you like to clone it first?',
      'Clone tt-metal',
      'Cancel'
    );

    if (choice === 'Clone tt-metal') {
      await cloneTTMetal();
    }
    return;
  }

  // Check if model zoo exists
  if (!fs.existsSync(modelZooPath)) {
    vscode.window.showErrorMessage(
      `Model zoo not found at ${modelZooPath}. Please ensure tt-metal is up to date.`
    );
    return;
  }

  // Open the model zoo folder
  const uri = vscode.Uri.file(modelZooPath);
  await vscode.commands.executeCommand('revealInExplorer', uri);

  // Show information panel
  const message = `
🔍 Model Zoo Browser

**Production Models:**
- Llama 3.1 8B - Text generation
- Whisper - Audio transcription
- ResNet50 - Image classification
- BERT - NLP tasks
- Stable Diffusion 3.5 - Image generation

**Experimental Models:**
- BlazePose - Pose estimation
- YOLOv4-v12 - Object detection
- nanoGPT - Train your own GPT

📂 Location: ${modelZooPath}

Each model has:
- demo/ - Runnable examples
- tt/ - TT hardware implementation
- tests/ - Unit tests
- README.md - Setup guide
  `;

  const panel = vscode.window.createWebviewPanel(
    'modelZoo',
    'TT-Metal Model Zoo',
    vscode.ViewColumn.Two,
    {}
  );

  panel.webview.html = `
    <!DOCTYPE html>
    <html>
    <head>
      <style>
        body { font-family: var(--vscode-font-family); padding: 20px; }
        h1 { color: var(--vscode-foreground); }
        pre { background: var(--vscode-editor-background); padding: 15px; border-radius: 5px; }
        code { color: var(--vscode-textPreformat-foreground); }
      </style>
    </head>
    <body>
      <pre>${message}</pre>
    </body>
    </html>
  `;
}

/**
 * Command: tenstorrent.exploreProgrammingExamples
 *
 * Opens the programming examples directory showing low-level TT-Metalium examples.
 */
async function exploreProgrammingExamples(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const homeDir = os.homedir();
  const defaultPath = path.join(homeDir, 'tt-metal');
  const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);

  const examplesPath = path.join(ttMetalPath, 'tt_metal', 'programming_examples');

  // Check if tt-metal exists
  const fs = await import('fs');
  if (!fs.existsSync(ttMetalPath)) {
    const choice = await vscode.window.showWarningMessage(
      'tt-metal repository not found. Would you like to clone it first?',
      'Clone tt-metal',
      'Cancel'
    );

    if (choice === 'Clone tt-metal') {
      await cloneTTMetal();
    }
    return;
  }

  // Check if examples exist
  if (!fs.existsSync(examplesPath)) {
    vscode.window.showErrorMessage(
      `Programming examples not found at ${examplesPath}. Please ensure tt-metal is up to date.`
    );
    return;
  }

  // Open the examples folder in file explorer
  const uri = vscode.Uri.file(examplesPath);

  // Show quick pick with options
  const action = await vscode.window.showInformationMessage(
    `⚡ Programming examples found at ${examplesPath}`,
    'Open in Terminal',
    'Show in Explorer',
    'Open Folder'
  );

  if (action === 'Open in Terminal') {
    const terminal = getOrCreateTerminal('explore');
    terminal.show();
    terminal.sendText(`cd "${examplesPath}"`);
  } else if (action === 'Show in Explorer') {
    await vscode.commands.executeCommand('revealFileInOS', uri);
  } else if (action === 'Open Folder') {
    await vscode.commands.executeCommand('vscode.openFolder', uri, { forceNewWindow: false });
  }
}

// ============================================================================
// Lesson 12 Commands - TT-Metalium Cookbook
// ============================================================================

/**
 * Command: tenstorrent.createCookbookProjects
 *
 * Deploys all cookbook project templates to ~/tt-scratchpad/cookbook/
 * Creates the complete project structure with all 4 projects.
 */
async function createCookbookProjects(): Promise<void> {
  const os = await import('os');
  const path = await import('path');
  const fs = await import('fs');
  const homeDir = os.homedir();
  const scratchpadPath = path.join(homeDir, 'tt-scratchpad', 'cookbook');

  // Get extension's template directory
  const extensionPath = extensionContext.extensionPath;

  // Try dist/ first (production), then content/ (development)
  let templatePath = path.join(extensionPath, 'dist', 'content', 'templates', 'cookbook');
  if (!fs.existsSync(templatePath)) {
    templatePath = path.join(extensionPath, 'content', 'templates', 'cookbook');
  }

  // Check if templates exist
  if (!fs.existsSync(templatePath)) {
    vscode.window.showErrorMessage(
      `Cookbook templates not found. Checked:\n- ${path.join(extensionPath, 'dist', 'content', 'templates', 'cookbook')}\n- ${path.join(extensionPath, 'content', 'templates', 'cookbook')}`
    );
    return;
  }

  // Check if destination already exists
  if (fs.existsSync(scratchpadPath)) {
    const choice = await vscode.window.showWarningMessage(
      `Cookbook directory already exists at ${scratchpadPath}. Overwrite?`,
      'Overwrite',
      'Cancel'
    );

    if (choice !== 'Overwrite') {
      return;
    }

    // Remove existing directory
    fs.rmSync(scratchpadPath, { recursive: true, force: true });
  }

  // Create cookbook directory
  fs.mkdirSync(scratchpadPath, { recursive: true });

  // Copy all templates recursively
  function copyDir(src: string, dest: string) {
    fs.mkdirSync(dest, { recursive: true });
    const entries = fs.readdirSync(src, { withFileTypes: true });

    for (const entry of entries) {
      const srcPath = path.join(src, entry.name);
      const destPath = path.join(dest, entry.name);

      if (entry.isDirectory()) {
        copyDir(srcPath, destPath);
      } else {
        fs.copyFileSync(srcPath, destPath);
      }
    }
  }

  try {
    copyDir(templatePath, scratchpadPath);

    // Create .env file for Jupyter notebooks to use tt-metal environment
    const defaultPath = path.join(homeDir, 'tt-metal');
    const ttMetalPath = extensionContext.globalState.get<string>(STATE_KEYS.TT_METAL_PATH, defaultPath);
    const envContent = `# Environment variables for TT-Metal cookbook notebooks\n# These ensure Jupyter uses the correct Python environment\nPYTHONPATH=${ttMetalPath}\nTT_METAL_HOME=${ttMetalPath}\n`;
    const envPath = path.join(scratchpadPath, '.env');
    fs.writeFileSync(envPath, envContent);

    // Show success message with file count
    const projects = ['game_of_life', 'audio_processor', 'mandelbrot', 'image_filters'];
    const fileCount = projects.reduce((count, project) => {
      const projectPath = path.join(scratchpadPath, project);
      if (fs.existsSync(projectPath)) {
        return count + fs.readdirSync(projectPath).length;
      }
      return count;
    }, 0);

    vscode.window.showInformationMessage(
      `✓ Created ${projects.length} cookbook projects with ${fileCount} files in ${scratchpadPath}. .env file created for Jupyter integration!`
    );

    // Open the cookbook folder in explorer
    const uri = vscode.Uri.file(scratchpadPath);
    await vscode.commands.executeCommand('revealInExplorer', uri);

    // Show informational panel
    const panel = vscode.window.createWebviewPanel(
      'cookbookProjects',
      'TT-Metalium Cookbook Projects',
      vscode.ViewColumn.Two,
      {}
    );

    panel.webview.html = `
      <!DOCTYPE html>
      <html>
      <head>
        <style>
          body {
            font-family: var(--vscode-font-family);
            padding: 20px;
            line-height: 1.6;
          }
          h1 { color: var(--vscode-foreground); }
          h2 { color: var(--vscode-foreground); margin-top: 20px; }
          code {
            background: var(--vscode-editor-background);
            padding: 2px 6px;
            border-radius: 3px;
            color: var(--vscode-textPreformat-foreground);
          }
          pre {
            background: var(--vscode-editor-background);
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
          }
          .project { margin: 15px 0; }
        </style>
      </head>
      <body>
        <h1>🎉 Cookbook Projects Created!</h1>
        <p>All 4 projects have been deployed to: <code>${scratchpadPath}</code></p>

        <h2>Projects</h2>

        <div class="project">
          <h3>🎮 Game of Life</h3>
          <p>Cellular automaton with parallel tile computing</p>
          <pre>cd ${scratchpadPath}/game_of_life
pip install -r requirements.txt
python game_of_life.py</pre>
        </div>

        <div class="project">
          <h3>🎵 Audio Processor</h3>
          <p>Real-time audio signal processing</p>
          <pre>cd ${scratchpadPath}/audio_processor
pip install -r requirements.txt
python processor.py examples/sample.wav</pre>
        </div>

        <div class="project">
          <h3>🌀 Mandelbrot Explorer</h3>
          <p>Interactive fractal renderer</p>
          <pre>cd ${scratchpadPath}/mandelbrot
pip install -r requirements.txt
python explorer.py</pre>
        </div>

        <div class="project">
          <h3>🖼️ Image Filters</h3>
          <p>Creative image processing</p>
          <pre>cd ${scratchpadPath}/image_filters
pip install -r requirements.txt
python filters.py examples/sample.jpg</pre>
        </div>

        <h2>Next Steps</h2>
        <ol>
          <li>Install dependencies: <code>cd [project] && pip install -r requirements.txt</code></li>
          <li>Follow along with Lesson 12 for complete implementations</li>
          <li>Experiment and extend the projects!</li>
        </ol>

        <p><strong>📖 See Lesson 12 for detailed explanations and extensions</strong></p>
      </body>
      </html>
    `;

  } catch (error) {
    vscode.window.showErrorMessage(
      `Failed to create cookbook projects: ${error instanceof Error ? error.message : String(error)}`
    );
  }
}

/**
 * Command: tenstorrent.runGameOfLife
 * Runs Conway's Game of Life with random initial state
 */
async function runGameOfLife(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_GAME_OF_LIFE.template);
  vscode.window.showInformationMessage(
    '🎮 Running Game of Life with random initial state. Watch the cellular automaton evolve!'
  );
}

/**
 * Command: tenstorrent.runGameOfLifeGlider
 * Runs Game of Life with classic glider pattern
 */
async function runGameOfLifeGlider(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_GAME_OF_LIFE_GLIDER.template);
  vscode.window.showInformationMessage(
    '🎮 Running Game of Life with glider pattern. Watch it move diagonally across the grid!'
  );
}

/**
 * Command: tenstorrent.runGameOfLifeGliderGun
 * Runs Game of Life with Gosper Glider Gun
 */
async function runGameOfLifeGliderGun(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_GAME_OF_LIFE_GLIDER_GUN.template);
  vscode.window.showInformationMessage(
    '🎮 Running Game of Life with Gosper Glider Gun. Watch it generate gliders infinitely!'
  );
}

/**
 * Command: tenstorrent.runMandelbrotExplorer
 * Launches interactive Mandelbrot fractal explorer
 */
async function runMandelbrotExplorer(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_MANDELBROT_EXPLORER.template);
  vscode.window.showInformationMessage(
    '🌀 Launching Mandelbrot Explorer! Click to zoom, press R to reset, C for colors, Q to quit.'
  );
}

/**
 * Command: tenstorrent.runMandelbrotJulia
 * Displays comparison of 6 Julia set fractals
 */
async function runMandelbrotJulia(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_MANDELBROT_JULIA.template);
  vscode.window.showInformationMessage(
    '🌀 Displaying 6 classic Julia set fractals for comparison!'
  );
}

/**
 * Command: tenstorrent.runAudioProcessor
 * Runs audio processor demo with mel-spectrogram
 */
async function runAudioProcessor(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_AUDIO_PROCESSOR.template);
  vscode.window.showInformationMessage(
    '🎵 Running audio processor demo. Provide your own audio file or use the example!'
  );
}

/**
 * Command: tenstorrent.runImageFilters
 * Runs image filters demo showing edge detect, blur, sharpen, etc.
 */
async function runImageFilters(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_IMAGE_FILTERS.template);
  vscode.window.showInformationMessage(
    '🖼️ Running image filters demo. See edge detect, blur, sharpen, emboss, and oil painting effects!'
  );
}

// ============================================================================
// Lesson 17: Native Video Animation with AnimateDiff
// ============================================================================

/**
 * Command: tenstorrent.installAnimateDiff
 * Installs the animatediff-ttnn standalone package
 */
async function installAnimateDiff(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.INSTALL_ANIMATEDIFF.template);
  vscode.window.showInformationMessage(
    '📦 Installing AnimateDiff package... Check terminal for progress.'
  );
}

/**
 * Command: tenstorrent.runAnimateDiff2Frame
 * Tests temporal attention with minimal 2-frame sequence
 */
async function runAnimateDiff2Frame(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_ANIMATEDIFF_2FRAME.template);
  vscode.window.showInformationMessage(
    '🎬 Running 2-frame temporal attention test...'
  );
}

/**
 * Command: tenstorrent.runAnimateDiff16Frame
 * Generates full 16-frame animated sequence
 */
async function runAnimateDiff16Frame(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.RUN_ANIMATEDIFF_16FRAME.template);
  vscode.window.showInformationMessage(
    '🎥 Generating 16-frame animated sequence... This will take a moment.'
  );
}

/**
 * Command: tenstorrent.viewAnimateDiffOutput
 * Views the generated animation file
 */
async function viewAnimateDiffOutput(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.VIEW_ANIMATEDIFF_OUTPUT.template);
}

/**
 * Command: tenstorrent.setupAnimateDiffProject
 * Sets up the AnimateDiff project from the bundled extension files
 */
async function setupAnimateDiffProject(): Promise<void> {
  const extensionPath = vscode.extensions.getExtension('tenstorrent.tt-vscode-toolkit')?.extensionPath;
  if (!extensionPath) {
    vscode.window.showErrorMessage('Could not find extension path');
    return;
  }

  const projectPath = `${extensionPath}/dist/content/projects/animatediff`;

  const terminal = getOrCreateTerminal('tt-metal');
  const command = replaceVariables(TERMINAL_COMMANDS.SETUP_ANIMATEDIFF_PROJECT.template, {
    projectPath,
  });
  runInTerminal(terminal, command);
  vscode.window.showInformationMessage(
    '📦 Setting up AnimateDiff project at ~/tt-scratchpad/tt-animatediff/...'
  );
}

/**
 * Command: tenstorrent.generateAnimateDiffVideoSD35
 * Generates animated video using SD 3.5 + AnimateDiff (gnu cinemagraph)
 */
async function generateAnimateDiffVideoSD35(): Promise<void> {
  const terminal = getOrCreateTerminal('tt-metal');
  runInTerminal(terminal, TERMINAL_COMMANDS.GENERATE_ANIMATEDIFF_VIDEO_SD35.template);
  vscode.window.showInformationMessage(
    '🎬 Generating animated video with SD 3.5 + AnimateDiff! This will take 5-7 minutes on N150...'
  );
}

/**
 * Command: tenstorrent.viewAnimateDiffTutorial
 * Opens the comprehensive model bring-up tutorial
 */
async function viewAnimateDiffTutorial(): Promise<void> {
  const tutorialPath = vscode.Uri.file(
    `${process.env.HOME}/tt-animatediff/MODEL_BRINGUP_TUTORIAL.md`
  );
  const doc = await vscode.workspace.openTextDocument(tutorialPath);
  await vscode.window.showTextDocument(doc, { preview: false });
  vscode.window.showInformationMessage(
    '📖 Opened model bring-up tutorial - complete walkthrough from research to implementation!'
  );
}

/**
 * Command: tenstorrent.exploreAnimateDiffPackage
 * Opens the AnimateDiff package directory in explorer
 */
async function exploreAnimateDiffPackage(): Promise<void> {
  const packagePath = vscode.Uri.file(`${process.env.HOME}/tt-animatediff`);
  await vscode.commands.executeCommand('revealFileInOS', packagePath);
  vscode.window.showInformationMessage(
    '📂 Opening AnimateDiff package directory...'
  );
}

// ============================================================================
// Command Menu
// ============================================================================

/**
 * Shows a quick-pick menu with all available Tenstorrent commands organized by category.
 * Triggered by clicking the Tenstorrent icon in the status bar.
 */
async function showCommandMenu(): Promise<void> {
  interface CommandItem extends vscode.QuickPickItem {
    command?: string;
    isCategory?: boolean;
  }

  const items: CommandItem[] = [
    // Welcome & Getting Started
    { label: '🏠 Welcome & Getting Started', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(home) Show Welcome Page', description: 'Overview and lesson cards', command: 'tenstorrent.showWelcome' },
    { label: '$(question) Show FAQ', description: 'Frequently asked questions and troubleshooting', command: 'tenstorrent.showFaq' },
    { label: '$(book) Open Walkthrough', description: 'Step-by-step setup guide', command: 'tenstorrent.openWalkthrough' },
    { label: '$(refresh) Reset Walkthrough Progress', description: 'Start walkthrough from beginning', command: 'tenstorrent.resetProgress' },

    // Hardware & Setup
    { label: '⚙️ Hardware & Setup', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(device-desktop) Run Hardware Detection', description: 'Detect Tenstorrent devices (tt-smi)', command: 'tenstorrent.runHardwareDetection' },
    { label: '$(check) Verify Installation', description: 'Test tt-metal installation', command: 'tenstorrent.verifyInstallation' },
    { label: '$(pulse) Show Device Actions', description: 'Device status and management', command: 'tenstorrent.showDeviceActions' },
    { label: '$(sync) Reset Device', description: 'Soft reset with tt-smi -r', command: 'tenstorrent.resetDevice' },
    { label: '$(trash) Clear Device State', description: 'Full cleanup (processes + /dev/shm)', command: 'tenstorrent.clearDeviceState' },

    // Models & Downloads
    { label: '📦 Models & Downloads', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(cloud-download) Download Model', description: 'Download Llama 3.1 8B from HuggingFace', command: 'tenstorrent.downloadModel' },
    { label: '$(key) Set HuggingFace Token', description: 'Configure HF authentication', command: 'tenstorrent.setHuggingFaceToken' },
    { label: '$(sign-in) Login to HuggingFace', description: 'Authenticate with HF CLI', command: 'tenstorrent.loginHuggingFace' },

    // Inference & Chat
    { label: '💬 Inference & Chat', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(play) Run Llama Inference', description: 'Test model with pytest demo', command: 'tenstorrent.runInference' },
    { label: '$(comment) Create Chat Script', description: 'Generate interactive chat script', command: 'tenstorrent.createChatScriptDirect' },
    { label: '$(comment-discussion) Start Chat Session', description: 'Run interactive chat with Direct API', command: 'tenstorrent.startChatSessionDirect' },
    { label: '$(code) Create Coding Assistant', description: 'AI coding assistant with prompt engineering', command: 'tenstorrent.createCodingAssistantScript' },
    { label: '$(robot) Start Coding Assistant', description: 'Launch coding assistant chat', command: 'tenstorrent.startCodingAssistant' },

    // API Server
    { label: '🌐 API Server', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(file-code) Create API Server', description: 'Generate Flask API server script', command: 'tenstorrent.createApiServerDirect' },
    { label: '$(server) Start API Server', description: 'Launch Flask API server', command: 'tenstorrent.startApiServerDirect' },
    { label: '$(debug-start) Test API (Basic)', description: 'Send test query to API', command: 'tenstorrent.testApiBasicDirect' },
    { label: '$(debug-console) Test API (Multiple)', description: 'Send multiple queries', command: 'tenstorrent.testApiMultipleDirect' },

    // vLLM Production
    { label: '🚀 vLLM Production', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(git-branch) Clone vLLM', description: 'Clone Tenstorrent vLLM fork', command: 'tenstorrent.cloneVllm' },
    { label: '$(package) Install vLLM', description: 'Install vLLM with dependencies', command: 'tenstorrent.installVllm' },
    { label: '$(server-process) Start vLLM Server', description: 'Launch OpenAI-compatible API', command: 'tenstorrent.startVllmServer' },
    { label: '$(beaker) Test vLLM (OpenAI SDK)', description: 'Test with Python SDK', command: 'tenstorrent.testVllmOpenai' },
    { label: '$(symbol-method) Test vLLM (curl)', description: 'Test with HTTP request', command: 'tenstorrent.testVllmCurl' },

    // Image Generation
    { label: '🎨 Image Generation', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(file-media) Generate Sample Image', description: 'SD 3.5 Large (1024x1024)', command: 'tenstorrent.generateRetroImage' },
    { label: '$(paintcan) Interactive Image Gen', description: 'Custom prompts with SD 3.5', command: 'tenstorrent.startInteractiveImageGen' },

    // TT-Forge (Image Classification)
    { label: '🔨 TT-Forge (MLIR Compiler)', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(building) Build Forge from Source', description: 'Recommended installation method', command: 'tenstorrent.buildForgeFromSource' },
    { label: '$(package) Install Forge (Wheels)', description: 'Quick install via pip', command: 'tenstorrent.installForge' },
    { label: '$(test-view-icon) Test Forge Install', description: 'Verify forge module loads', command: 'tenstorrent.testForgeInstall' },
    { label: '$(file-code) Create Forge Classifier', description: 'MobileNetV2 image classifier', command: 'tenstorrent.createForgeClassifier' },
    { label: '$(play) Run Forge Classifier', description: 'Classify images with PyTorch', command: 'tenstorrent.runForgeClassifier' },

    // Bounty Program
    { label: '💰 Bounty Program & Contribution', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(globe) Browse Open Bounties', description: 'View available model bring-up bounties', command: 'tenstorrent.browseOpenBounties' },
    { label: '$(checklist) Copy Bounty Checklist', description: 'Workflow template for contributions', command: 'tenstorrent.copyBountyChecklist' },

    // Exploration & Learning
    { label: '🔍 Exploration & Learning', kind: vscode.QuickPickItemKind.Separator },
    { label: '$(mortar-board) Launch TTNN Tutorials', description: 'Interactive Jupyter notebooks', command: 'tenstorrent.launchTtnnTutorials' },
    { label: '$(library) Browse Model Zoo', description: 'Explore validated models', command: 'tenstorrent.browseModelZoo' },
    { label: '$(code) Programming Examples', description: 'Sample code and patterns', command: 'tenstorrent.exploreProgrammingExamples' },
    { label: '$(book) Create Cookbook Projects', description: 'Deploy 4 complete recipes', command: 'tenstorrent.createCookbookProjects' },
  ];

  const selected = await vscode.window.showQuickPick(items, {
    placeHolder: 'Search Tenstorrent commands...',
    matchOnDescription: true,
    matchOnDetail: true,
  });

  if (selected?.command) {
    vscode.commands.executeCommand(selected.command);
  }
}

// ============================================================================
// Image Preview WebviewView Provider
// ============================================================================

/**
 * WebviewViewProvider for displaying generated images and the Tenstorrent logo
 */
class TenstorrentImagePreviewProvider implements vscode.WebviewViewProvider {
  private _view?: vscode.WebviewView;
  private _currentImage?: string; // Path to currently displayed image

  constructor(private readonly context: vscode.ExtensionContext) {}

  resolveWebviewView(
    webviewView: vscode.WebviewView,
    _context: vscode.WebviewViewResolveContext,
    _token: vscode.CancellationToken
  ): void {
    this._view = webviewView;

    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(this.context.extensionUri, 'assets'),
        vscode.Uri.file(require('os').homedir())
      ]
    };

    webviewView.webview.html = this.getHtmlContent(webviewView.webview);

    // Handle messages from webview
    webviewView.webview.onDidReceiveMessage(async (message) => {
      if (message.command === 'openImage') {
        if (this._currentImage) {
          // Open in editor when double-clicked
          const imageUri = vscode.Uri.file(this._currentImage);
          await vscode.commands.executeCommand('vscode.open', imageUri, { preview: false });
        }
      }
    });
  }

  /**
   * Update telemetry data in the logo animation
   * @param telemetry Telemetry data from hardware
   */
  public updateTelemetry(telemetry: any): void {
    if (this._view && !this._currentImage) {
      // Only animate logo when not showing an image
      this._view.webview.postMessage({
        command: 'updateTelemetry',
        telemetry: telemetry
      });
    }
  }

  /**
   * Update the preview to show a specific image or video
   * @param imagePath Absolute path to the image or video file
   */
  public showImage(imagePath: string): void {
    this._currentImage = imagePath;
    if (this._view) {
      this._view.webview.html = this.getHtmlContent(this._view.webview, imagePath);
      this._view.show?.(true); // Reveal the view
    }
  }

  /**
   * Reset preview to show the default logo
   */
  public showLogo(): void {
    this._currentImage = undefined;
    if (this._view) {
      this._view.webview.html = this.getHtmlContent(this._view.webview);
    }
  }

  private getHtmlContent(webview: vscode.Webview, imagePath?: string): string {
    let imageUri: vscode.Uri;
    let altText: string;
    let isGeneratedImage = false;
    let isVideo = false;
    let caption = '';

    if (imagePath) {
      // Check if it's a video file
      const path = require('path');
      const ext = path.extname(imagePath).toLowerCase();
      isVideo = ['.mp4', '.webm', '.ogg', '.mov'].includes(ext);

      // Show generated image or video
      imageUri = webview.asWebviewUri(vscode.Uri.file(imagePath));
      altText = `Generated: ${path.basename(imagePath)}`;
      isGeneratedImage = true;
      caption = `<div class="caption">${path.basename(imagePath)}</div>`;
    } else {
      // Show default logo with telemetry animation
      imageUri = webview.asWebviewUri(
        vscode.Uri.joinPath(this.context.extensionUri, 'assets', 'img', 'tt_logo_color_dark_backgrounds.svg')
      );
      altText = 'Tenstorrent';
    }

    const clickHandler = isGeneratedImage
      ? 'ondblclick="vscode.postMessage({ command: \'openImage\' })" style="cursor: pointer;" title="Double-click to open in editor, right-click to save"'
      : '';

    // Add telemetry animation script only for logo (not generated images)
    const telemetryScript = !isGeneratedImage ? `
<script>
  const vscode = acquireVsCodeApi();

  // Current and target telemetry values
  let currentTelemetry = {
    temp: 40,
    power: 15,
    clock: 1000
  };
  let targetTelemetry = { ...currentTelemetry };

  // Animation state
  let animationFrame = null;
  let lastUpdate = Date.now();

  // Listen for telemetry updates
  window.addEventListener('message', event => {
    const message = event.data;

    if (message.command === 'updateTelemetry' && message.telemetry) {
      targetTelemetry = {
        temp: message.telemetry.asic_temp || 40,
        power: message.telemetry.power || 15,
        clock: message.telemetry.aiclk || 1000
      };
      lastUpdate = Date.now();
    }
  });

  // Map telemetry to visual properties
  function telemetryToVisuals(telemetry) {
    // Temperature → Hue (blue when cool, orange/red when hot)
    // 40°C = 200° (blue), 70°C = 30° (orange)
    const tempHue = Math.max(0, Math.min(360, 200 - (telemetry.temp - 40) * 5.67));

    // Power → Brightness (dim when idle, bright when active)
    // 10W = 60% brightness, 50W = 100% brightness
    const brightness = Math.max(60, Math.min(100, 60 + (telemetry.power - 10) * 1));

    return { hue: tempHue, brightness };
  }

  // Smooth interpolation (lerp)
  function lerp(start, end, t) {
    return start + (end - start) * t;
  }

  // Animation loop
  function animate() {
    const now = Date.now();
    const timeSinceUpdate = (now - lastUpdate) / 1000; // seconds

    // Interpolate over 5 seconds (to match telemetry update interval)
    const t = Math.min(timeSinceUpdate / 5, 1);

    // Smooth interpolation
    currentTelemetry.temp = lerp(currentTelemetry.temp, targetTelemetry.temp, t * 0.1);
    currentTelemetry.power = lerp(currentTelemetry.power, targetTelemetry.power, t * 0.1);
    currentTelemetry.clock = lerp(currentTelemetry.clock, targetTelemetry.clock, t * 0.1);

    // Calculate visual properties
    const visuals = telemetryToVisuals(currentTelemetry);

    // Apply to logo (color and brightness only)
    const logo = document.querySelector('.image-container img');
    if (logo) {
      logo.style.filter = \`hue-rotate(\${visuals.hue - 200}deg) brightness(\${visuals.brightness}%)\`;
    }

    animationFrame = requestAnimationFrame(animate);
  }

  // Start animation when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      animate();
    });
  } else {
    animate();
  }
</script>
    ` : '<script>const vscode = acquireVsCodeApi();</script>';

    return `<!DOCTYPE html>
<html style="height: 100%;">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src ${webview.cspSource} data:; media-src ${webview.cspSource}; style-src 'unsafe-inline'; script-src 'unsafe-inline';">
  <style>
    html, body {
      height: 100%;
      margin: 0;
      padding: 0;
      overflow: hidden;
    }
    body {
      padding: 8px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: center;
      background: transparent;
      box-sizing: border-box;
    }
    .image-container {
      text-align: center;
      width: 100%;
      max-width: ${isGeneratedImage ? '100%' : '140px'};
      max-height: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
    }
    img, video {
      width: 100%;
      height: auto;
      max-height: ${isGeneratedImage ? 'calc(100vh - 40px)' : '80px'};
      object-fit: contain;
      display: block;
      ${isGeneratedImage ? 'border-radius: 4px; box-shadow: 0 2px 8px rgba(0,0,0,0.3);' : ''}
      transition: filter 0.3s ease;
    }
    ${isGeneratedImage ? `
    img:hover, video:hover {
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      transition: box-shadow 0.2s;
    }` : ``}
    .caption {
      margin-top: 4px;
      padding: 2px 4px;
      font-size: 10px;
      color: var(--vscode-descriptionForeground);
      word-wrap: break-word;
      font-family: var(--vscode-font-family);
      max-width: 100%;
      user-select: text;
    }
  </style>
</head>
<body>
  <div class="image-container">
    ${isVideo
      ? `<video src="${imageUri}" ${clickHandler} controls autoplay loop muted style="max-width: 100%;">
           Your browser does not support the video tag.
         </video>`
      : `<img src="${imageUri}" alt="${altText}" ${clickHandler} onerror="this.style.display='none'">`
    }
    ${caption}
  </div>
  ${telemetryScript}
</body>
</html>`;
  }
}

// ============================================================================
// Extension Lifecycle
// ============================================================================

/**
 * Called when the extension is activated.
 *
 * Registers all commands that are referenced by the walkthrough steps.
 * The walkthrough itself is automatically shown by VS Code based on the
 * configuration in package.json.
 *
 * @param context - Extension context provided by VS Code
 */
export async function activate(context: vscode.ExtensionContext): Promise<void> {
  console.log('Tenstorrent Developer Extension is now active');

  // Store context globally for use in command handlers
  extensionContext = context;

  // ============================================================================
  // New Lesson System Initialization
  // ============================================================================

  // Initialize core managers
  const stateManager = new StateManager(context);
  const progressTracker = new ProgressTracker(context, stateManager);
  const lessonRegistry = new LessonRegistry(context);

  // Initialize environment manager for Python venv tracking
  environmentManager = new EnvironmentManager(context);
  context.subscriptions.push(environmentManager);

  // Load lesson registry - MUST complete before creating TreeView
  try {
    await lessonRegistry.load();
    console.log(`Loaded ${lessonRegistry.getTotalCount()} lessons`);
  } catch (error) {
    console.error('Failed to load lesson registry:', error);
    vscode.window.showErrorMessage(
      'Failed to load Tenstorrent lessons. Please reload the window and check the extension logs.'
    );
    throw error; // Fail activation if registry doesn't load
  }

  // Create WebviewView for image preview (globally accessible for updating from commands)
  const imagePreviewProvider = new TenstorrentImagePreviewProvider(context);
  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('tenstorrentImagePreview', imagePreviewProvider)
  );
  
  // Make provider globally accessible so commands can update the preview
  (global as any).imagePreviewProvider = imagePreviewProvider;

  // Create TreeView for lessons
  const treeDataProvider = new LessonTreeDataProvider(lessonRegistry, progressTracker);
  const treeView = vscode.window.createTreeView('tenstorrentLessons', {
    treeDataProvider,
    showCollapseAll: true,
  });

  // Create Webview Manager
  const webviewManager = new LessonWebviewManager(context, lessonRegistry, progressTracker);

  // Initialize telemetry monitor for hardware status (with logo animation callback)
  const telemetryMonitor = new TelemetryMonitor(context, (telemetry) => {
    imagePreviewProvider.updateTelemetry(telemetry);
  });

  // Make telemetry monitor globally accessible so commands can access current details
  (global as any).telemetryMonitor = telemetryMonitor;

  // Note: Tree item clicks are handled via the command property set in LessonTreeDataProvider
  // No need for onDidChangeSelection handler - it would cause lessons to open twice

  // Hook progress tracking into existing commands
  // NOTE: Progress is tracked in the commands themselves via recordCommandExecution

  // Register new lesson system commands
  context.subscriptions.push(
    vscode.commands.registerCommand('tenstorrent.showLesson', async (lessonId: string) => {
      const lesson = lessonRegistry.get(lessonId);
      if (lesson) {
        await stateManager.setCurrentLesson(lessonId);
        await webviewManager.showLesson(lesson);
      }
    }),
    vscode.commands.registerCommand('tenstorrent.refreshLessons', () => {
      treeDataProvider.refresh();
    }),
    vscode.commands.registerCommand('tenstorrent.filterLessons', async () => {
      // Show quick pick for filter options
      const filterType = await vscode.window.showQuickPick(
        [
          { label: '$(device-desktop) Hardware', value: 'hardware' },
          { label: '$(check) Validated Only', value: 'validated' },
          { label: '$(tag) Tags', value: 'tags' },
          { label: '$(circle-slash) Clear Filters', value: 'clear' },
        ],
        { placeHolder: 'Select filter type' }
      );

      if (!filterType) {
        return;
      }

      if (filterType.value === 'clear') {
        treeDataProvider.clearFilters();
        vscode.window.showInformationMessage('Filters cleared');
        return;
      }

      if (filterType.value === 'validated') {
        treeDataProvider.applyFilter({ validatedOnly: true });
        vscode.window.showInformationMessage('Showing validated lessons only');
        return;
      }

      // More filter logic can be added here
    }),
    vscode.commands.registerCommand('tenstorrent.toggleShowAllLessons', async () => {
      const config = vscode.workspace.getConfiguration('tenstorrent');
      const currentValue = config.get<boolean>('showUnvalidatedLessons', false);
      const newValue = !currentValue;

      await config.update('showUnvalidatedLessons', newValue, vscode.ConfigurationTarget.Global);

      if (newValue) {
        vscode.window.showInformationMessage('✅ Now showing all lessons (including draft and blocked)');
      } else {
        vscode.window.showInformationMessage('✅ Now showing validated lessons only');
      }
    }),
    treeView,
    webviewManager
  );

  // ============================================================================
  // End New Lesson System
  // ============================================================================

  // Register all commands (walkthrough management + step commands)
  const commands = [
    // Command menu
    vscode.commands.registerCommand('tenstorrent.showCommandMenu', showCommandMenu),

    // Welcome page
    vscode.commands.registerCommand('tenstorrent.showWelcome', () => showWelcome(context)),

    // FAQ
    vscode.commands.registerCommand('tenstorrent.showFaq', () => showFaq(context)),

    // RISC-V Exploration Guide
    vscode.commands.registerCommand('tenstorrent.showRiscvGuide', () => showRiscvGuide(context)),

    // Telemetry details - show same content as hover tooltip
    vscode.commands.registerCommand('tenstorrent.showTelemetryDetails', () => {
      const monitor = (global as any).telemetryMonitor;
      if (monitor) {
        const details = monitor.getCurrentDetails();
        if (details) {
          vscode.window.showInformationMessage(details, 'OK');
        } else {
          vscode.window.showInformationMessage(
            'Telemetry data not yet available. Please wait a moment and try again.',
            'OK'
          );
        }
      } else {
        vscode.window.showInformationMessage(
          'Telemetry monitor not initialized.',
          'OK'
        );
      }
    }),

    // Walkthrough management commands
    vscode.commands.registerCommand('tenstorrent.openWalkthrough', openWalkthrough),
    vscode.commands.registerCommand('tenstorrent.resetProgress', resetProgress),

    // Walkthrough step commands

    // Lesson 0 - Modern Setup with tt-installer 2.0
    vscode.commands.registerCommand('tenstorrent.runQuickInstall', runQuickInstall),
    vscode.commands.registerCommand('tenstorrent.downloadInstaller', downloadInstaller),
    vscode.commands.registerCommand('tenstorrent.runInteractiveInstall', runInteractiveInstall),
    vscode.commands.registerCommand('tenstorrent.runNonInteractiveInstall', runNonInteractiveInstall),
    vscode.commands.registerCommand('tenstorrent.testMetaliumContainer', testMetaliumContainer),

    // Lesson 1 - Hardware Detection
    vscode.commands.registerCommand('tenstorrent.runHardwareDetection', runHardwareDetection),

    // Lesson 2 - Verify Installation
    vscode.commands.registerCommand('tenstorrent.verifyInstallation', verifyInstallation),

    // Lesson 3 - Download Model
    vscode.commands.registerCommand('tenstorrent.setHuggingFaceToken', setHuggingFaceToken),
    vscode.commands.registerCommand('tenstorrent.loginHuggingFace', loginHuggingFace),
    vscode.commands.registerCommand('tenstorrent.downloadModel', downloadModel),
    vscode.commands.registerCommand('tenstorrent.cloneTTMetal', cloneTTMetal),
    vscode.commands.registerCommand('tenstorrent.setupEnvironment', setupEnvironment),
    vscode.commands.registerCommand('tenstorrent.runInference', runInference),
    vscode.commands.registerCommand('tenstorrent.installInferenceDeps', installInferenceDeps),
    vscode.commands.registerCommand('tenstorrent.createChatScript', createChatScript),
    vscode.commands.registerCommand('tenstorrent.startChatSession', startChatSession),
    vscode.commands.registerCommand('tenstorrent.createApiServer', createApiServer),
    vscode.commands.registerCommand('tenstorrent.installFlask', installFlask),
    vscode.commands.registerCommand('tenstorrent.startApiServer', startApiServer),
    vscode.commands.registerCommand('tenstorrent.testApiBasic', testApiBasic),
    vscode.commands.registerCommand('tenstorrent.testApiMultiple', testApiMultiple),

    // Lesson 4 - Direct API Chat
    vscode.commands.registerCommand('tenstorrent.createChatScriptDirect', createChatScriptDirect),
    vscode.commands.registerCommand('tenstorrent.startChatSessionDirect', startChatSessionDirect),

    // Lesson 5 - Direct API Server
    vscode.commands.registerCommand('tenstorrent.createApiServerDirect', createApiServerDirect),
    vscode.commands.registerCommand('tenstorrent.startApiServerDirect', startApiServerDirect),
    vscode.commands.registerCommand('tenstorrent.testApiBasicDirect', testApiBasicDirect),
    vscode.commands.registerCommand('tenstorrent.testApiMultipleDirect', testApiMultipleDirect),

    // Lesson 6 - tt-inference-server
    vscode.commands.registerCommand('tenstorrent.verifyInferenceServerPrereqs', verifyInferenceServerPrereqs),
    vscode.commands.registerCommand('tenstorrent.startTtInferenceServer', startTtInferenceServer),
    vscode.commands.registerCommand('tenstorrent.testTtInferenceServerSimple', testTtInferenceServerSimple),
    vscode.commands.registerCommand('tenstorrent.testTtInferenceServerStreaming', testTtInferenceServerStreaming),
    vscode.commands.registerCommand('tenstorrent.testTtInferenceServerSampling', testTtInferenceServerSampling),
    vscode.commands.registerCommand('tenstorrent.createTtInferenceServerClient', createTtInferenceServerClient),
    vscode.commands.registerCommand('tenstorrent.createTtInferenceServerConfig', createTtInferenceServerConfig),

    // Lesson 7 - vLLM (previously Lesson 6)
    vscode.commands.registerCommand('tenstorrent.updateTTMetal', updateTTMetal),
    vscode.commands.registerCommand('tenstorrent.cloneVllm', cloneVllm),
    vscode.commands.registerCommand('tenstorrent.installVllm', installVllm),
    vscode.commands.registerCommand('tenstorrent.startVllmServer', startVllmServer),
    vscode.commands.registerCommand('tenstorrent.startVllmServerWithHardware', startVllmServerWithHardware), // Parameterized command (replaces N150/N300/T3K/P100 variants)
    vscode.commands.registerCommand('tenstorrent.createVllmStarter', createVllmStarter),
    vscode.commands.registerCommand('tenstorrent.testVllmOpenai', testVllmOpenai),
    vscode.commands.registerCommand('tenstorrent.testVllmCurl', testVllmCurl),

    // Lesson 8 - VSCode Chat Integration (previously Lesson 7)

    // Lesson 9 - Image Generation with SD 3.5 Large (previously Lesson 8)
    vscode.commands.registerCommand('tenstorrent.generateRetroImage', generateRetroImage),
    vscode.commands.registerCommand('tenstorrent.startInteractiveImageGen', startInteractiveImageGen),
    vscode.commands.registerCommand('tenstorrent.copyImageGenDemo', copyImageGenDemo),
    vscode.commands.registerCommand('tenstorrent.openLatestImage', openLatestImage),

    // Lesson 10 - Coding Assistant with Prompt Engineering (previously Lesson 9)
    vscode.commands.registerCommand('tenstorrent.verifyCodingModel', verifyCodingModel),
    vscode.commands.registerCommand('tenstorrent.createCodingAssistantScript', createCodingAssistantScript),
    vscode.commands.registerCommand('tenstorrent.startCodingAssistant', startCodingAssistant),

    // Lesson 11 - Image Classification with TT-Forge
    vscode.commands.registerCommand('tenstorrent.buildForgeFromSource', buildForgeFromSource),
    vscode.commands.registerCommand('tenstorrent.installForge', installForge),
    vscode.commands.registerCommand('tenstorrent.testForgeInstall', testForgeInstall),
    vscode.commands.registerCommand('tenstorrent.createForgeClassifier', createForgeClassifier),
    vscode.commands.registerCommand('tenstorrent.runForgeClassifier', runForgeClassifier),

    // Lesson 12 - TT-XLA JAX Integration
    vscode.commands.registerCommand('tenstorrent.installTtXla', installTtXla),
    vscode.commands.registerCommand('tenstorrent.testTtXlaInstall', testTtXlaInstall),
    vscode.commands.registerCommand('tenstorrent.runTtXlaDemo', runTtXlaDemo),

    // Lesson 13 - RISC-V Programming on Tensix Cores
    vscode.commands.registerCommand('tenstorrent.buildProgrammingExamples', buildProgrammingExamples),
    vscode.commands.registerCommand('tenstorrent.runRiscvExample', runRiscvExample),
    vscode.commands.registerCommand('tenstorrent.openRiscvKernel', openRiscvKernel),
    vscode.commands.registerCommand('tenstorrent.openRiscvGuide', openRiscvGuide),

    // Lesson 14 - Exploring TT-Metalium
    vscode.commands.registerCommand('tenstorrent.launchTtnnTutorials', launchTtnnTutorials),
    vscode.commands.registerCommand('tenstorrent.browseModelZoo', browseModelZoo),
    vscode.commands.registerCommand('tenstorrent.exploreProgrammingExamples', exploreProgrammingExamples),

    // Lesson 12 - TT-Metalium Cookbook
    vscode.commands.registerCommand('tenstorrent.createCookbookProjects', createCookbookProjects),
    vscode.commands.registerCommand('tenstorrent.runGameOfLife', runGameOfLife),
    vscode.commands.registerCommand('tenstorrent.runGameOfLifeGlider', runGameOfLifeGlider),
    vscode.commands.registerCommand('tenstorrent.runGameOfLifeGliderGun', runGameOfLifeGliderGun),
    vscode.commands.registerCommand('tenstorrent.runMandelbrotExplorer', runMandelbrotExplorer),
    vscode.commands.registerCommand('tenstorrent.runMandelbrotJulia', runMandelbrotJulia),
    vscode.commands.registerCommand('tenstorrent.runAudioProcessor', runAudioProcessor),
    vscode.commands.registerCommand('tenstorrent.runImageFilters', runImageFilters),

    // Lesson 17 - Native Video Animation with AnimateDiff
    vscode.commands.registerCommand('tenstorrent.setupAnimateDiffProject', setupAnimateDiffProject),
    vscode.commands.registerCommand('tenstorrent.installAnimateDiff', installAnimateDiff),
    vscode.commands.registerCommand('tenstorrent.runAnimateDiff2Frame', runAnimateDiff2Frame),
    vscode.commands.registerCommand('tenstorrent.runAnimateDiff16Frame', runAnimateDiff16Frame),
    vscode.commands.registerCommand('tenstorrent.viewAnimateDiffOutput', viewAnimateDiffOutput),
    vscode.commands.registerCommand('tenstorrent.generateAnimateDiffVideoSD35', generateAnimateDiffVideoSD35),
    vscode.commands.registerCommand('tenstorrent.viewAnimateDiffTutorial', viewAnimateDiffTutorial),
    vscode.commands.registerCommand('tenstorrent.exploreAnimateDiffPackage', exploreAnimateDiffPackage),

    // Bounty Program
    vscode.commands.registerCommand('tenstorrent.browseOpenBounties', browseOpenBounties),
    vscode.commands.registerCommand('tenstorrent.copyBountyChecklist', copyBountyChecklist),

    // Device Management
    vscode.commands.registerCommand('tenstorrent.resetDevice', resetDevice),
    vscode.commands.registerCommand('tenstorrent.clearDeviceState', clearDeviceState),

    // Python Environment Management
    vscode.commands.registerCommand('tenstorrent.selectPythonEnvironment', async () => {
      const activeTerminal = vscode.window.activeTerminal;
      if (!activeTerminal) {
        vscode.window.showWarningMessage('No active terminal. Open or select a terminal first.');
        return;
      }
      await environmentManager.switchEnvironment(activeTerminal);
    }),
    vscode.commands.registerCommand('tenstorrent.refreshEnvironmentStatus', async () => {
      const activeTerminal = vscode.window.activeTerminal;
      if (!activeTerminal) {
        vscode.window.showWarningMessage('No active terminal. Open or select a terminal first.');
        return;
      }
      await environmentManager.detectActiveEnvironment(activeTerminal);
      vscode.window.showInformationMessage('Environment status refreshed');
    }),
  ];

  // Add all command registrations to subscriptions for proper cleanup
  context.subscriptions.push(...commands);

  // Initialize command menu statusbar item (left side, high priority)
  commandMenuStatusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Left,
    1000 // High priority = far left
  );

  commandMenuStatusBarItem.text = '$(circuit-board) Tenstorrent';
  commandMenuStatusBarItem.tooltip = 'Tenstorrent Commands';
  commandMenuStatusBarItem.command = 'tenstorrent.showCommandMenu';
  commandMenuStatusBarItem.show();

  context.subscriptions.push(commandMenuStatusBarItem);

  // Initialize device status statusbar item (right side)
  statusBarItem = vscode.window.createStatusBarItem(
    vscode.StatusBarAlignment.Right,
    100 // Priority (higher = further left)
  );

  statusBarItem.command = {
    title: 'Show Device Actions',
    command: 'tenstorrent.showDeviceActions',
  };

  context.subscriptions.push(statusBarItem);

  // Register statusbar click command
  context.subscriptions.push(
    vscode.commands.registerCommand('tenstorrent.showDeviceActions', showDeviceActionsMenu)
  );

  // Run initial device status check (but don't start auto-polling)
  updateDeviceStatus();

  // Start periodic updates only if enabled (default: disabled)
  startStatusUpdateTimer();

  // Create a persistent tt-metal terminal on activation (most common use case)
  // This terminal stays open and can be reused for tt-metal commands
  const defaultTerminal = vscode.window.createTerminal({
    name: 'TT: Metal',
    cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath,
  });
  // Show the terminal by default for better UX
  defaultTerminal.show(true); // preserveFocus=true keeps focus on editor
  terminals['tt-metal'] = defaultTerminal;  // Register in terminal management system
  context.subscriptions.push(defaultTerminal);

  // Auto-configure user experience on first activation
  const hasSeenWelcome = context.globalState.get<boolean>('hasSeenWelcome', false);
  if (!hasSeenWelcome) {
    // Mark as seen first to avoid reopening if command fails
    context.globalState.update('hasSeenWelcome', true);

    // Set Tenstorrent theme for brand consistency and optimal visibility
    const config = vscode.workspace.getConfiguration();
    const currentTheme = config.get<string>('workbench.colorTheme');

    // Only set theme if it's still the default (to respect user's existing preference)
    if (currentTheme === 'Default Dark Modern' || currentTheme === 'Default Light Modern' || !currentTheme) {
      config.update('workbench.colorTheme', 'Tenstorrent', vscode.ConfigurationTarget.Global);
    }

    // Prompt to install recommended extensions (non-blocking)
    setTimeout(() => {
      promptRecommendedExtensions();
    }, 1500);

    // Open the welcome page automatically on first run
    setTimeout(() => {
      showWelcome(context);
    }, 2000); // Small delay to ensure extension is fully activated
  }
}

/**
 * Called when the extension is deactivated.
 *
 * Cleans up terminal references and stops status monitoring.
 * Note that VS Code automatically disposes of terminals and statusbar items
 * when the extension is deactivated, but we explicitly clear our references
 * for good measure.
 */
export function deactivate(): void {
  // Stop status update timer
  stopStatusUpdateTimer();

  // Clear all terminal references
  // VS Code will handle actual disposal
  terminals['tt-metal'] = undefined;
  terminals['tt-forge'] = undefined;
  terminals['tt-xla'] = undefined;
  terminals['vllm-server'] = undefined;
  terminals['api-server'] = undefined;
  terminals['explore'] = undefined;

  // Clear statusbar references
  statusBarItem = undefined;
  commandMenuStatusBarItem = undefined;

  console.log('Tenstorrent Developer Extension has been deactivated');
}
