/**
 * vizCommands.ts — Tensix Grid Visualizer VSCode commands
 *
 * Registers `tenstorrent.showTensixViz` which opens a webview panel
 * containing the tensix-viz.js Canvas visualizer. The panel loads scene
 * JSON from the extension's bundled scenes directory or accepts a custom
 * script passed as a command argument.
 */

import * as vscode from 'vscode';
import * as fs from 'fs';
import * as path from 'path';

const SCENES: Record<string, string> = {
  'NOC Routing':          'noc-routing.json',
  'Parallelism Scale-out':'parallelism-scale.json',
  'Kernel Dispatch':      'kernel-dispatch.json',
  'Single Core':          'single-core.json',
};

export function registerVizCommands(context: vscode.ExtensionContext): void {
  context.subscriptions.push(
    vscode.commands.registerCommand(
      'tenstorrent.showTensixViz',
      (opts?: { scene?: string; arch?: string; script?: unknown[] }) =>
        showTensixVizPanel(context, opts)
    )
  );
}

async function showTensixVizPanel(
  context: vscode.ExtensionContext,
  opts?: { scene?: string; arch?: string; script?: unknown[] }
): Promise<void> {
  // Scene selection — either from opts or via quick pick
  let sceneName = opts?.scene;
  let customScript = opts?.script;

  if (!sceneName && !customScript) {
    const pick = await vscode.window.showQuickPick(Object.keys(SCENES), {
      placeHolder: 'Select a Tensix visualizer scene',
      title: 'Tenstorrent: Tensix Grid Visualizer',
    });
    if (!pick) return;
    sceneName = pick;
  }

  // Load scene JSON from bundled scenes directory
  let scriptJson = '[]';
  if (customScript) {
    scriptJson = JSON.stringify(customScript);
  } else if (sceneName && SCENES[sceneName]) {
    const scenePath = path.join(
      context.extensionPath,
      'dist', 'src', 'webview', 'tensix-viz', 'scenes',
      SCENES[sceneName]
    );
    // Fall back to src/ when running in Extension Dev Host (not packaged)
    const devPath = path.join(
      context.extensionPath,
      'src', 'webview', 'tensix-viz', 'scenes',
      SCENES[sceneName]
    );
    const resolvedPath = fs.existsSync(scenePath) ? scenePath : devPath;
    try {
      scriptJson = fs.readFileSync(resolvedPath, 'utf8');
    } catch {
      scriptJson = '[]';
    }
  }

  const arch = opts?.arch || 'wormhole';
  const title = sceneName ? `Tensix Viz — ${sceneName}` : 'Tensix Grid Visualizer';

  const panel = vscode.window.createWebviewPanel(
    'tenstorrentTensixViz',
    title,
    vscode.ViewColumn.Beside,
    {
      enableScripts: true,
      localResourceRoots: [
        vscode.Uri.joinPath(context.extensionUri, 'src', 'webview', 'tensix-viz'),
        vscode.Uri.joinPath(context.extensionUri, 'dist', 'src', 'webview', 'tensix-viz'),
      ],
    }
  );

  // Resolve URIs for webview resources
  const distBase  = vscode.Uri.joinPath(context.extensionUri, 'dist', 'src', 'webview', 'tensix-viz');
  const srcBase   = vscode.Uri.joinPath(context.extensionUri, 'src',  'webview', 'tensix-viz');
  const jsUri     = panel.webview.asWebviewUri(
    fs.existsSync(path.join(distBase.fsPath, 'tensix-viz.js')) ? distBase : srcBase
  );

  panel.webview.html = buildVizHtml(
    panel.webview,
    jsUri,
    arch,
    sceneName || 'Custom',
    scriptJson
  );
}

function buildVizHtml(
  webview: vscode.Webview,
  jsBaseUri: vscode.Uri,
  arch: string,
  sceneName: string,
  scriptJson: string
): string {
  const jsUri  = `${jsBaseUri}/tensix-viz.js`;
  const cssUri = `${jsBaseUri}/tensix-viz.css`;
  const nonce  = getNonce();

  return /* html */ `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none';
             style-src ${webview.cspSource} 'unsafe-inline';
             script-src 'nonce-${nonce}' ${webview.cspSource};">
  <title>Tensix Grid Visualizer</title>
  <link rel="stylesheet" href="${cssUri}">
  <style>
    body {
      background: #0F2A35;
      color: #E8F0F2;
      font-family: var(--vscode-font-family, sans-serif);
      margin: 0;
      padding: 16px;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      box-sizing: border-box;
    }
    h2 {
      font-size: 14px;
      color: #4FD1C5;
      margin: 0 0 4px;
      font-family: monospace;
      letter-spacing: 0.05em;
    }
    .arch-label {
      font-size: 11px;
      color: #607D8B;
      margin-bottom: 12px;
    }
    .viz-area {
      background: #0D2030;
      border: 1px solid #2D6675;
      border-radius: 8px;
      padding: 12px;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 10px;
    }
    canvas { border-radius: 4px; }
    .controls { display: flex; gap: 8px; }
    .tv-play, .tv-step {
      background: #1A3C47; border: 1px solid #2D6675; border-radius: 4px;
      color: #4FD1C5; cursor: pointer; font-size: 13px; padding: 4px 14px;
    }
    .tv-play:hover, .tv-step:hover { background: #2D6675; border-color: #4FD1C5; }
    .tv-legend { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 6px; }
    .tv-legend-item { display: flex; align-items: center; gap: 5px; font-size: 11px; color: #607D8B; }
    .tv-legend-dot { width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }
    .scene-select {
      background: #1A3C47; border: 1px solid #2D6675; border-radius: 4px;
      color: #E8F0F2; font-size: 12px; padding: 3px 8px; margin-bottom: 8px;
    }
  </style>
</head>
<body>
  <h2>⬡ Tensix Grid Visualizer</h2>
  <div class="arch-label">${arch === 'blackhole' ? 'Blackhole (P100/P150/P300c)' : 'Wormhole (N150/N300/T3K)'} · ${sceneName}</div>

  <div class="viz-area">
    <div class="tensix-viz-container" data-arch="${arch}">
      <canvas class="tensix-viz-canvas" width="520" height="340"></canvas>
      <script class="tensix-viz-script" type="application/json">${scriptJson}</script>
    </div>
    <div class="controls">
      <button class="tv-play">▶ Play</button>
      <button class="tv-step">⏭ Step</button>
    </div>
    <div class="tv-legend"></div>
  </div>

  <script nonce="${nonce}" src="${jsUri}"></script>
</body>
</html>`;
}

function getNonce(): string {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}
