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

    // Each kernel declares which backend(s) it actually runs against.
    // ttlang-sim's shim ttnn supports from_numpy/ttl.operation; the real
    // tt-metal ttnn used by ttsim-wh/ttsim-bh supports neither (only
    // from_torch, no ttl) -- these are genuinely different APIs, so a
    // kernel written for one is not portable to the other by accident.
    const KERNELS = {
        'eltwise_add': {
            label: 'Element-wise Add',
            backends: ['ttlang-sim'],
            code: `\
import numpy as np

# This code runs on the ttlang-sim backend specifically.
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
            backends: ['ttlang-sim'],
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
            backends: ['ttlang-sim'],
            code: `\
import numpy as np

a = ttnn.from_numpy(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), device=device)
b = ttnn.from_numpy(np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32), device=device)
c = a + b
print("a + b =", ttnn.to_numpy(c))
print("PASSED")
`
        },
        // The three kernels above use ttlang-sim's shim ttnn (from_numpy,
        // ttl.operation) and are not portable to real tt-metal/ttnn. These
        // three are the ttsim-wh/ttsim-bh equivalents, using the real API
        // (from_torch; no ttl).
        'hello_tensor_ttsim': {
            label: 'Hello Tensor (ttsim)',
            backends: ['ttsim-wh', 'ttsim-bh'],
            code: `\
import torch

a = ttnn.from_torch(torch.tensor([[1.0, 2.0], [3.0, 4.0]]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.tensor([[10.0, 20.0], [30.0, 40.0]]), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = a + b
print("a + b =", ttnn.to_torch(ttnn.from_device(c)))
print("PASSED")
`
        },
        'eltwise_add_ttsim': {
            label: 'Element-wise Add (ttsim)',
            backends: ['ttsim-wh', 'ttsim-bh'],
            code: `\
import numpy as np
import torch

# This code runs on the ttsim-wh/ttsim-bh backend (real tt-metal/ttnn).
# dtype=ttnn.bfloat16 is required, not optional: the tensix unpacker on this
# backend rejects float32 tiles outright. numpy has no bfloat16 of its own,
# so .float() first is needed before .numpy() can convert the result back.
dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
ref = a_np + b_np

a = ttnn.from_torch(torch.from_numpy(a_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.from_numpy(b_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.add(a, b)
result = ttnn.to_torch(ttnn.from_device(c)).float().numpy()

max_err = float(np.abs(result - ref).max())
print(f"eltwise_add  dim={dim}x{dim}  max_err={max_err:.6f}")
print("PASSED" if max_err < 1e-2 else "FAILED")
`
        },
        'matmul_ttsim': {
            label: 'Matmul (ttsim)',
            backends: ['ttsim-wh', 'ttsim-bh'],
            code: `\
import numpy as np
import torch

dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
ref = a_np @ b_np

a = ttnn.from_torch(torch.from_numpy(a_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b = ttnn.from_torch(torch.from_numpy(b_np), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
c = ttnn.matmul(a, b)
result = ttnn.to_torch(ttnn.from_device(c)).float().numpy()

max_err = float(np.abs(result - ref).max())
print(f"matmul  dim={dim}x{dim}  max_err={max_err:.6f}")
# bfloat16 has ~7 bits of mantissa; a dim=64 dot product accumulates that
# per-element error across 64 terms, so max abs error routinely lands
# between 0.1 and 0.3 against a float32 reference -- live-tested against
# ttsim-wh, not just estimated. 1e-1 (tuned for float32-precision inputs)
# false-failed on real bfloat16 output; 5e-1 has margin without being loose
# enough to hide an actually-broken kernel.
print("PASSED" if max_err < 5e-1 else "FAILED")
`
        },
    };

    // Strip the minimum common leading whitespace from every non-empty line,
    // preserving relative indentation (e.g. a try/except body). Used to keep
    // multi-line Python template literals immune to how this file itself
    // happens to be indented -- see the preamble construction below.
    function _dedent(str) {
        const lines = str.replace(/^\n/, '').replace(/\s+$/, '').split('\n');
        // [ \t]*, not just space: a tab-indented source (nothing in this repo
        // enforces spaces-only) would otherwise measure 0 for every line,
        // leave the tabs in place, and produce an IndentationError on the
        // first line of the emitted Python -- exactly the failure mode this
        // helper exists to prevent.
        const indents = lines.filter(l => l.trim().length > 0).map(l => l.match(/^[ \t]*/)[0].length);
        const minIndent = indents.length ? Math.min(...indents) : 0;
        return lines.map(l => l.slice(minIndent)).join('\n');
    }

    function _kernelsForBackend(backend) {
        return Object.entries(KERNELS).filter(([, k]) => k.backends.includes(backend));
    }

    // ─── CloudPlaygroundController ────────────────────────────────────────────

    class CloudPlaygroundController {
        constructor(mount) {
            this._mount = mount;
            this._ws = null;
            this._running = false;
            this._currentKernel = null;

            this._buildUI();
            this._onBackendChange();
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

            const kernelSel = this._mount.querySelector('#tt-pg-kernel-sel');
            const backendSel = this._mount.querySelector('#tt-pg-backend-sel');
            // Kernel options are populated per-backend in _onBackendChange(),
            // not once here -- eltwise_add/matmul_1d/hello_tensor use
            // ttlang-sim's shim ttnn (from_numpy, ttl.operation) and are not
            // portable to real tt-metal/ttnn, so the list of runnable
            // kernels genuinely differs by backend.
            kernelSel.addEventListener('change', () => this._selectKernel(kernelSel.value));
            backendSel.addEventListener('change', () => this._onBackendChange());

            this._mount.querySelector('#tt-pg-run').addEventListener('click', () => this._run());
            this._mount.querySelector('#tt-pg-clear').addEventListener('click', () => this._clearOutput());

            this._noticeEl = this._mount.querySelector('#tt-pg-cloud-notice');
            this._codeEl = this._mount.querySelector('#tt-pg-code');
            this._outputEl = this._mount.querySelector('#tt-pg-output');
            this._runBtn = this._mount.querySelector('#tt-pg-run');
            this._backendSel = backendSel;
            this._kernelSel = kernelSel;

            this._showCloudStatus();
        }

        _onBackendChange() {
            const compatible = _kernelsForBackend(this._backendSel.value);
            const previousKey = this._currentKernel;
            this._kernelSel.innerHTML = '';
            for (const [key, { label }] of compatible) {
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = label;
                this._kernelSel.appendChild(opt);
            }
            if (!compatible.length) {
                this._currentKernel = null;
                this._codeEl.value = '';
                return;
            }
            // Only reload the template -- clobbering whatever the user is
            // currently editing -- when the previously-selected kernel
            // genuinely isn't runnable against the new backend. Several
            // kernels (e.g. "Matmul (ttsim)") are compatible with BOTH
            // ttsim-wh and ttsim-bh, and switching between those backends
            // previously reloaded the pristine template over any
            // in-progress edits on every single change, even though the
            // same kernel key stays selectable either way.
            const stillCompatible = compatible.some(([key]) => key === previousKey);
            if (stillCompatible) {
                this._kernelSel.value = previousKey;
            } else {
                this._selectKernel(compatible[0][0]);
            }
        }

        _selectKernel(key) {
            if (KERNELS[key]) {
                this._codeEl.value = KERNELS[key].code.trim();
                this._currentKernel = key;
                // Keep the dropdown in sync with the loaded code -- without
                // this, calling _selectKernel() with anything other than
                // whatever the browser defaults the <select> to (its first
                // option) leaves the visible selection and the actual
                // loaded code silently out of sync.
                if (this._kernelSel) this._kernelSel.value = key;
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
                    const okBackends = Object.entries(data.backends || {})
                        .filter(([, ok]) => ok)
                        .map(([b]) => b);
                    const span = document.createElement('span');
                    span.className = 'tt-pg-notice-ok';
                    span.appendChild(document.createTextNode('✓ Cloud simulator connected. Available: '));
                    if (okBackends.length === 0) {
                        span.appendChild(document.createTextNode('none'));
                    } else {
                        okBackends.forEach((b, i) => {
                            const code = document.createElement('code');
                            code.textContent = b;
                            span.appendChild(code);
                            if (i < okBackends.length - 1) {
                                span.appendChild(document.createTextNode(', '));
                            }
                        });
                    }
                    this._noticeEl.textContent = '';
                    this._noticeEl.appendChild(span);
                })
                .catch(() => {
                    const span = document.createElement('span');
                    span.className = 'tt-pg-notice-warn';
                    span.appendChild(document.createTextNode('⚠ Cloud simulator unreachable at '));
                    const code = document.createElement('code');
                    code.textContent = CLOUD_API_URL;
                    span.appendChild(code);
                    span.appendChild(document.createTextNode('. '));
                    const link = document.createElement('a');
                    link.href = '#pyodide-playground';
                    link.textContent = 'Use the local Pyodide playground instead.';
                    span.appendChild(link);
                    this._noticeEl.textContent = '';
                    this._noticeEl.appendChild(span);
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

            // One preamble for every backend: ttnn and device are ALWAYS
            // set up (unconditionally, outside any try/except), since every
            // kernel needs them regardless of backend. Only `ttl` is
            // guarded -- it exists in ttlang-sim's environment but not in
            // ttsim-wh/ttsim-bh's real tt-metal/ttnn -- and set to None
            // rather than left undefined on import failure, so a kernel
            // that needs it but was run against an incompatible backend (it
            // shouldn't be reachable via the UI, which filters the kernel
            // list by backend in _onBackendChange(), but the API can be hit
            // directly) fails with a clear AttributeError on `ttl.whatever`
            // instead of leaving `device` undefined too and failing on an
            // unrelated NameError first.
            //
            // _dedent() below guards against a future re-indent of this file
            // silently breaking the emitted Python: template literals keep
            // whatever leading whitespace precedes each line in the source,
            // and Python is indentation-sensitive, so an editor auto-format
            // that nests these lines deeper would otherwise produce a
            // hard-to-diagnose IndentationError at execution time.
            const preamble = _dedent(`
                try:
                    import ttl
                except ImportError:
                    ttl = None
                import ttnn
                device = ttnn.open_device(device_id=0)
            `);
            // Wrap the kernel body in try/finally so ttnn.close_device()
            // always runs, even when the kernel raises -- without this, a
            // failing run (or even a passing one, since nothing ever called
            // it) leaves the simulated device open for the lifetime of the
            // server process, and the next run's ttnn.open_device() either
            // queues behind it or fails outright depending on the backend.
            // Every line of the user's code is indented once to sit inside
            // the try: block; a uniform per-line indent is always valid
            // Python regardless of the kernel's own internal structure.
            const indentedCode = code.split('\n').map(l => (l.length ? '    ' + l : l)).join('\n');
            const fullCode =
                preamble + '\n' +
                'try:\n' +
                indentedCode + '\n' +
                'finally:\n' +
                '    ttnn.close_device(device)\n';

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
                // No `timeout` field -- let the server apply its own
                // EXEC_TIMEOUT-derived default per backend instead of a
                // second hardcoded value here drifting out of sync with it
                // (this previously sent 30s while the server's own default
                // was already 60-180s, silently cutting every run short).
                this._ws.send(JSON.stringify({ code: fullCode, backend }));
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
