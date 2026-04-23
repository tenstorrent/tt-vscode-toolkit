// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// cloud-playground.js — cloud-backed variant of the browser playground.
// Connects to TTSIM_API_URL via WebSocket to execute kernels server-side.
// Falls back to the Pyodide (local) playground if the API is unreachable.

(function () {
    'use strict';

    // Injected by build-web.js from $TTSIM_API_URL env var (may be empty string).
    const CLOUD_API_URL = window.TTSIM_API_URL || '';

    // ─── Kernel snippets (same set as playground.js) ─────────────────────────

    const KERNELS = {
        'eltwise_add': {
            label: 'Element-wise Add',
            code: `\
import numpy as np

# This code runs on the TT cloud simulator.
# ttl / ttnn are pre-imported automatically.

TILE = 32
dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
ref = a_np + b_np

a = ttnn.from_numpy(a_np, device=device)
b = ttnn.from_numpy(b_np, device=device)
out = ttnn.zeros_like(a)

@ttl.operation(grid="auto")
def eltwise_add(a, b, out):
    rows = a.shape[0] // TILE
    cols = a.shape[1] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1,1), block_count=2)
    o_dfb = ttl.make_dataflow_buffer_like(out, shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.wait() as ab, b_dfb.wait() as bb, o_dfb.reserve() as ob:
                    ob.store(ab + bb)

    @ttl.datamovement()
    def read():
        for r in range(rows):
            for c in range(cols):
                with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                    ttl.copy(a[r:r+1, c:c+1], ab).wait()
                    ttl.copy(b[r:r+1, c:c+1], bb).wait()

    @ttl.datamovement()
    def write():
        for r in range(rows):
            for c in range(cols):
                with o_dfb.wait() as ob:
                    ttl.copy(ob, out[r:r+1, c:c+1]).wait()

eltwise_add(a, b, out)
result = ttnn.to_numpy(out)
max_err = float(np.abs(result - ref).max())
print(f"eltwise_add  dim={dim}x{dim}  max_err={max_err:.6f}")
print("PASSED" if max_err < 1e-4 else "FAILED")
`
        },
        'matmul_1d': {
            label: 'Matmul (row-partitioned)',
            code: `\
import numpy as np

# Row-partitioned C = A @ B on the cloud simulator.
dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
ref = a_np @ b_np

a = ttnn.from_numpy(a_np, device=device)
b = ttnn.from_numpy(b_np, device=device)
c = ttnn.zeros([dim, dim], dtype=ttnn.float32, device=device)

result = ttnn.to_numpy(c)
max_err = float(np.abs(result - ref).max())
print(f"matmul  dim={dim}x{dim}  max_err={max_err:.6f}")
print("PASSED" if max_err < 1e-3 else "FAILED")
`
        },
        'hello_tensor': {
            label: 'Hello Tensor',
            code: `\
import numpy as np

a = ttnn.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), device=device)
b = ttnn.from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32), device=device)
c = a + b
print("a + b =", ttnn.to_numpy(c))
print("PASSED")
`
        },
    };

    // ─── CloudPlaygroundController ────────────────────────────────────────────

    class CloudPlaygroundController {
        constructor(mount) {
            this._mount = mount;
            this._ws = null;
            this._running = false;

            this._buildUI();
            this._selectKernel('eltwise_add');
        }

        _buildUI() {
            this._mount.innerHTML = `
<div class="tt-pg-cloud-notice" id="tt-pg-cloud-notice"></div>
<div class="tt-pg-layout">
  <div class="tt-pg-editor-col">
    <div class="tt-pg-toolbar">
      <label class="tt-pg-label">Kernel</label>
      <select class="tt-pg-kernel-select" id="tt-pg-kernel-sel"></select>
      <label class="tt-pg-label">Backend</label>
      <select class="tt-pg-backend-select" id="tt-pg-backend-sel">
        <option value="ttlang-sim">ttlang-sim (Python)</option>
        <option value="ttsim-wh">ttsim-wh (Wormhole emulation)</option>
        <option value="ttsim-bh">ttsim-bh (Blackhole emulation)</option>
      </select>
      <button class="tt-pg-btn tt-pg-run-btn" id="tt-pg-run">&#9654; Run on Simulator</button>
      <button class="tt-pg-btn tt-pg-clear-btn" id="tt-pg-clear">&#10006; Clear</button>
    </div>
    <textarea class="tt-pg-code" id="tt-pg-code" spellcheck="false"></textarea>
  </div>
  <div class="tt-pg-output-col">
    <div class="tt-pg-output-header">Output</div>
    <pre class="tt-pg-output" id="tt-pg-output"></pre>
  </div>
</div>`;

            const sel = this._mount.querySelector('#tt-pg-kernel-sel');
            for (const [key, { label }] of Object.entries(KERNELS)) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = label;
                sel.appendChild(opt);
            }
            sel.addEventListener('change', () => this._selectKernel(sel.value));

            this._mount.querySelector('#tt-pg-run').addEventListener('click', () => this._run());
            this._mount.querySelector('#tt-pg-clear').addEventListener('click', () => this._clearOutput());

            this._noticeEl = this._mount.querySelector('#tt-pg-cloud-notice');
            this._codeEl = this._mount.querySelector('#tt-pg-code');
            this._outputEl = this._mount.querySelector('#tt-pg-output');
            this._runBtn = this._mount.querySelector('#tt-pg-run');
            this._backendSel = this._mount.querySelector('#tt-pg-backend-sel');

            this._showCloudStatus();
        }

        _selectKernel(key) {
            if (KERNELS[key]) {
                this._codeEl.value = KERNELS[key].code.trim();
            }
        }

        _showCloudStatus() {
            if (!CLOUD_API_URL) {
                this._noticeEl.innerHTML = `
<span class="tt-pg-notice-warn">
  ⚠ No cloud simulator URL configured. Set <code>TTSIM_API_URL</code> at build time.
  <a href="#pyodide-playground">Use the local Pyodide playground instead.</a>
</span>`;
                this._runBtn.disabled = true;
                return;
            }
            // Quick connectivity check via HTTP health endpoint
            const healthUrl = CLOUD_API_URL.replace(/^ws/, 'http').replace(/\/execute$/, '') + '/health';
            fetch(healthUrl, { signal: AbortSignal.timeout(5000) })
                .then(r => r.json())
                .then(data => {
                    const backends = Object.entries(data.backends || {})
                        .filter(([, ok]) => ok)
                        .map(([b]) => `<code>${b}</code>`)
                        .join(', ');
                    this._noticeEl.innerHTML =
                        `<span class="tt-pg-notice-ok">✓ Cloud simulator connected. Available: ${backends || 'none'}</span>`;
                })
                .catch(() => {
                    this._noticeEl.innerHTML = `
<span class="tt-pg-notice-warn">
  ⚠ Cloud simulator unreachable at <code>${CLOUD_API_URL}</code>.
  <a href="#pyodide-playground">Use the local Pyodide playground instead.</a>
</span>`;
                    this._runBtn.disabled = true;
                });
        }

        _appendOutput(text, cls) {
            const span = document.createElement('span');
            if (cls) span.className = cls;
            span.textContent = text;
            this._outputEl.appendChild(span);
            this._outputEl.scrollTop = this._outputEl.scrollHeight;
        }

        _clearOutput() {
            this._outputEl.textContent = '';
        }

        _run() {
            if (this._running) return;
            if (!CLOUD_API_URL) return;

            this._clearOutput();
            this._running = true;
            this._runBtn.disabled = true;
            this._runBtn.textContent = '⏳ Running…';

            const code = this._codeEl.value;
            const backend = this._backendSel.value;

            // Build preamble that imports ttl/ttnn inside the server environment
            const preamble = `
import sys, importlib
# Ensure ttlang-sim is importable if installed in the server environment
try:
    import ttl
    import ttnn
    device = ttnn.open_device(device_id=0)
except ImportError:
    pass
`;
            const fullCode = preamble + '\n' + code;

            const wsUrl = CLOUD_API_URL.endsWith('/execute')
                ? CLOUD_API_URL
                : CLOUD_API_URL.replace(/\/?$/, '/execute');

            try {
                this._ws = new WebSocket(wsUrl);
            } catch (e) {
                this._appendOutput(`WebSocket error: ${e.message}\n`, 'tt-pg-stderr');
                this._done();
                return;
            }

            this._ws.onopen = () => {
                this._ws.send(JSON.stringify({ code: fullCode, backend, timeout: 30 }));
            };

            this._ws.onmessage = (evt) => {
                let msg;
                try { msg = JSON.parse(evt.data); } catch { return; }
                if (msg.type === 'stdout') {
                    this._appendOutput(msg.data, 'tt-pg-stdout');
                } else if (msg.type === 'stderr') {
                    this._appendOutput(msg.data, 'tt-pg-stderr');
                } else if (msg.type === 'error') {
                    this._appendOutput(`Error: ${msg.data}\n`, 'tt-pg-stderr');
                } else if (msg.type === 'exit') {
                    this._appendOutput(`\n[exit code ${msg.code}]\n`, msg.code === 0 ? 'tt-pg-ok' : 'tt-pg-stderr');
                    this._done();
                }
            };

            this._ws.onerror = () => {
                this._appendOutput('\n[WebSocket error — is the simulator API running?]\n', 'tt-pg-stderr');
                this._done();
            };

            this._ws.onclose = () => {
                if (this._running) this._done();
            };
        }

        _done() {
            this._running = false;
            this._runBtn.disabled = false;
            this._runBtn.textContent = '▶ Run on Simulator';
            if (this._ws) {
                try { this._ws.close(); } catch { }
                this._ws = null;
            }
        }
    }

    // ─── Auto-mount on DOMContentLoaded ──────────────────────────────────────

    function mount() {
        document.querySelectorAll('.tt-cloud-playground-mount').forEach(el => {
            new CloudPlaygroundController(el);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', mount);
    } else {
        mount();
    }

    window.CloudPlaygroundController = CloudPlaygroundController;
})();
