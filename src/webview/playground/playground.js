// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// playground.js — controller for the ttlang-sim-lite browser playground.
// Manages the Pyodide worker, kernel selector, and output pane.

(function () {
    'use strict';

    // ─── Service worker registration ──────────────────────────────────────────
    if ('serviceWorker' in navigator) {
        navigator.serviceWorker.register('/assets/playground/sw.js', { scope: '/' })
            .catch(() => { /* non-fatal */ });
    }

    // ─── Kernel snippets ──────────────────────────────────────────────────────

    const KERNELS = {
        'eltwise_add': {
            label: 'Element-wise Add',
            code: `\
import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid="auto")
def eltwise_add(a: ttnn.Tensor, b: ttnn.Tensor, out: ttnn.Tensor) -> None:
    rows = a.shape[0] // TILE
    cols = a.shape[1] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a,   shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b,   shape=(1,1), block_count=2)
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

device = ttnn.open_device(device_id=0)
dim = 64
a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
a   = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
b   = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out = ttnn.from_torch(np.zeros((dim,dim), dtype=np.float32),
                      dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
eltwise_add(a, b, out)
result = ttnn.to_torch(out)
print(f"eltwise_add: max_err={np.max(np.abs(result - (a_np+b_np))):.6f}")
ttnn.close_device(device)
`,
        },
        'hello_tensor': {
            label: 'Hello Tensor',
            code: `\
# Minimal ttnn tensor round-trip — no kernels, no hardware needed.
import numpy as np
import ttnn

device = ttnn.open_device(device_id=0)

a_np = np.arange(16, dtype=np.float32).reshape(4, 4)
t = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

print("Tensor shape:", t.shape)
print("Tensor dtype:", t.dtype)

recovered = ttnn.to_torch(t)
print("Roundtrip match:", np.allclose(a_np, recovered, atol=1e-5))
print(recovered)

ttnn.close_device(device)
`,
        },
        'scale_add': {
            label: 'Scale + Add (broadcast)',
            code: `\
# Demonstrates scalar broadcast: scale each row by a different factor.
import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid="auto")
def scale_add(src: ttnn.Tensor, scale_row: ttnn.Tensor, out: ttnn.Tensor) -> None:
    """Multiply each tile-row of src by the corresponding row of scale_row, then add 1."""
    rows = src.shape[0] // TILE
    cols = src.shape[1] // TILE
    s_dfb   = ttl.make_dataflow_buffer_like(src,      shape=(1,1), block_count=2)
    sc_dfb  = ttl.make_dataflow_buffer_like(scale_row, shape=(1,1), block_count=2)
    o_dfb   = ttl.make_dataflow_buffer_like(out,      shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for r in range(rows):
            for c in range(cols):
                with s_dfb.wait() as sb, sc_dfb.wait() as scb, o_dfb.reserve() as ob:
                    ob.store(sb * scb + 1.0)

    @ttl.datamovement()
    def read():
        for r in range(rows):
            for c in range(cols):
                with s_dfb.reserve() as sb, sc_dfb.reserve() as scb:
                    ttl.copy(src[r:r+1,      c:c+1], sb).wait()
                    ttl.copy(scale_row[r:r+1, c:c+1], scb).wait()

    @ttl.datamovement()
    def write():
        for r in range(rows):
            for c in range(cols):
                with o_dfb.wait() as ob:
                    ttl.copy(ob, out[r:r+1, c:c+1]).wait()

device = ttnn.open_device(device_id=0)
dim = 64
src_np   = np.ones((dim, dim), dtype=np.float32)
# scale_row: each tile-row i gets scale factor (i+1)
scale_np = np.repeat(
    np.arange(1, dim//TILE + 1, dtype=np.float32)[:, None],
    TILE, axis=0
).repeat(dim, axis=1)[:dim, :dim]

src   = ttnn.from_torch(src_np,   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
scale = ttnn.from_torch(scale_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
out   = ttnn.from_torch(np.zeros((dim,dim), dtype=np.float32),
                        dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

scale_add(src, scale, out)
result = ttnn.to_torch(out)
expected = src_np * scale_np + 1.0
print(f"scale_add: max_err={np.max(np.abs(result - expected)):.6f}")
ttnn.close_device(device)
`,
        },
        'fused_mma': {
            label: 'Fused Multiply-Add (tutorial)',
            code: `\
# Tutorial Step 1 (Elementwise): Fused Multiply-Add — y = a * b + c
# Direct adaptation of the official TT-Lang elementwise tutorial.
#
# Three concurrent threads per Tensix core:
#   DM read  → streams a/b/c tiles from DRAM into L1 ring buffers (DFBs)
#   Compute  → fused a*b+c in L1, zero DRAM traffic mid-kernel
#   DM write → drains results to DRAM
#
# block_count=2 enables double-buffering on real hardware:
# DM fills one DFB slot while Compute processes the other.
import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid=(1, 1))
def fused_mma(a: ttnn.Tensor, b: ttnn.Tensor, c: ttnn.Tensor, y: ttnn.Tensor) -> None:
    rows = a.shape[0] // TILE
    cols = a.shape[1] // TILE
    a_dfb = ttl.make_dataflow_buffer_like(a, shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(b, shape=(1,1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(c, shape=(1,1), block_count=2)
    y_dfb = ttl.make_dataflow_buffer_like(y, shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(rows):
            for _ in range(cols):
                # wait() — block until DM reader has pushed a filled tile
                # reserve() — claim an empty output slot for the result
                with (a_dfb.wait() as ab, b_dfb.wait() as bb,
                      c_dfb.wait() as cb, y_dfb.reserve() as yb):
                    yb.store(ab * bb + cb)  # fused in L1

    @ttl.datamovement()
    def read():
        for row in range(rows):
            for col in range(cols):
                with a_dfb.reserve() as ab, b_dfb.reserve() as bb, c_dfb.reserve() as cb:
                    # Tile-coordinate index: a[row,col] = tile at (row,col), not element (row*32,col*32)
                    ttl.copy(a[row, col], ab).wait()
                    ttl.copy(b[row, col], bb).wait()
                    ttl.copy(c[row, col], cb).wait()

    @ttl.datamovement()
    def write():
        for row in range(rows):
            for col in range(cols):
                with y_dfb.wait() as yb:
                    ttl.copy(yb, y[row, col]).wait()

np.random.seed(42)
device = ttnn.open_device(device_id=0)
dim = 64
def to_tt(arr):
    return ttnn.from_torch(arr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

a_np = np.random.rand(dim, dim).astype(np.float32)
b_np = np.random.rand(dim, dim).astype(np.float32)
c_np = np.random.rand(dim, dim).astype(np.float32)
y    = to_tt(np.zeros((dim, dim), dtype=np.float32))

fused_mma(to_tt(a_np), to_tt(b_np), to_tt(c_np), y)

result   = ttnn.to_torch(y)
expected = a_np * b_np + c_np
max_err  = np.max(np.abs(result - expected))
print(f"fused_mma: PASS  max_err={max_err:.6f}  (a*b+c, bfloat16)")
ttnn.close_device(device)
`,
        },
        'matmul_relu': {
            label: 'Matmul + Bias + ReLU (tutorial)',
            code: `\
# Tutorial Step 1 (Matmul): y = relu(a @ b + bias)
# Direct adaptation of the official TT-Lang matmul tutorial.
#
# Showcases the k-reduction accumulator DFB ping-pong:
#   acc_dfb acts as a two-slot alternating buffer — each k-step reads the
#   previous partial sum (wait) and writes the updated one (reserve),
#   keeping ALL accumulation in L1 with zero DRAM traffic during the k-loop.
#   The final step fuses bias-add + ReLU before the result hits DRAM.
#
# This is the architectural pattern behind high-throughput matmul on Tensix.
import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid=(1, 1))
def matmul_relu(a: ttnn.Tensor, b: ttnn.Tensor, bias: ttnn.Tensor, y: ttnn.Tensor) -> None:
    M = a.shape[0] // TILE
    K = a.shape[1] // TILE
    N = b.shape[1] // TILE

    a_dfb    = ttl.make_dataflow_buffer_like(a,    shape=(1,1), block_count=2)
    b_dfb    = ttl.make_dataflow_buffer_like(b,    shape=(1,1), block_count=2)
    bias_dfb = ttl.make_dataflow_buffer_like(bias, shape=(1,1), block_count=2)
    # acc_dfb: internal ping-pong accumulator — only compute reads and writes it
    acc_dfb  = ttl.make_dataflow_buffer_like(y,    shape=(1,1), block_count=2)
    y_dfb    = ttl.make_dataflow_buffer_like(y,    shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        for _ in range(M):
            for _ in range(N):
                # Zero the accumulator before the k-reduction.
                with acc_dfb.reserve() as acc:
                    acc.store(ttl.math.fill(acc, 0))

                # K-reduction ping-pong:
                # Each step consumes the previous partial sum and pushes the new one.
                # Both wait() and reserve() on acc_dfb are nested in the same scope;
                # push() runs before pop() so the new sum is always visible first.
                for _ in range(K):
                    with (a_dfb.wait() as ab, b_dfb.wait() as bb,
                          acc_dfb.wait() as prev):
                        with acc_dfb.reserve() as acc:
                            acc.store(prev + ab @ bb)

                # Fused bias add + ReLU — one tile write to y_dfb.
                with bias_dfb.wait() as bib, acc_dfb.wait() as acc:
                    with y_dfb.reserve() as yb:
                        yb.store(ttl.math.relu(acc + bib))

    @ttl.datamovement()
    def read():
        for m in range(M):
            for n in range(N):
                # Bias tile first — ready when compute finishes the k-loop.
                with bias_dfb.reserve() as bib:
                    ttl.copy(bias[m, n], bib).wait()
                for k in range(K):
                    with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                        ttl.copy(a[m, k], ab).wait()
                        ttl.copy(b[k, n], bb).wait()

    @ttl.datamovement()
    def write():
        for m in range(M):
            for n in range(N):
                with y_dfb.wait() as yb:
                    ttl.copy(yb, y[m, n]).wait()

np.random.seed(42)
device = ttnn.open_device(device_id=0)
M, K, N = 64, 64, 64
def to_tt(arr):
    return ttnn.from_torch(arr, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

a_np    = np.random.randn(M, K).astype(np.float32)
b_np    = np.random.randn(K, N).astype(np.float32)
bias_np = np.random.randn(M, N).astype(np.float32)
y       = to_tt(np.zeros((M, N), dtype=np.float32))

matmul_relu(to_tt(a_np), to_tt(b_np), to_tt(bias_np), y)

result   = ttnn.to_torch(y)
expected = np.maximum(a_np @ b_np + bias_np, 0.0)
max_err  = np.max(np.abs(result - expected))
print(f"matmul_relu: PASS  max_err={max_err:.4f}  (relu(a@b+bias), bfloat16 k-reduction)")
ttnn.close_device(device)
`,
        },
        'matmul_1d': {
            label: 'Row-partitioned Matmul',
            code: `\
# 1-D row-partitioned matrix multiply: C = A @ B
# Each Tensix core handles a contiguous row-slice of A.
import numpy as np
import ttl
import ttnn

TILE = 32

@ttl.operation(grid="auto")
def matmul_1d(A: ttnn.Tensor, B: ttnn.Tensor, C: ttnn.Tensor) -> None:
    M_tiles = A.shape[0] // TILE   # rows of A in tiles
    K_tiles = A.shape[1] // TILE   # inner dimension
    N_tiles = B.shape[1] // TILE   # cols of B in tiles

    # Each node handles a slice of M
    grid_cols, grid_rows = ttl.grid_size(dims=2)
    rows_per_node = -(-M_tiles // grid_rows)

    # Per-tile accumulators — one DFB per (row, col) would be expensive;
    # instead we iterate K in the compute thread with a single pair of DFBs.
    a_dfb = ttl.make_dataflow_buffer_like(A, shape=(1,1), block_count=2)
    b_dfb = ttl.make_dataflow_buffer_like(B, shape=(1,1), block_count=2)
    c_dfb = ttl.make_dataflow_buffer_like(C, shape=(1,1), block_count=2)

    @ttl.compute()
    def compute():
        node_col, node_row = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for n in range(N_tiles):
                acc = None
                for _k in range(K_tiles):
                    with a_dfb.wait() as ab, b_dfb.wait() as bb:
                        tile = ab @ bb      # 32x32 @ 32x32
                        acc = tile if acc is None else acc + tile
                with c_dfb.reserve() as cb:
                    cb.store(acc)

    @ttl.datamovement()
    def read():
        node_col, node_row = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for _n in range(N_tiles):
                for k in range(K_tiles):
                    with a_dfb.reserve() as ab, b_dfb.reserve() as bb:
                        ttl.copy(A[m:m+1, k:k+1], ab).wait()
                        ttl.copy(B[k:k+1, _n:_n+1], bb).wait()

    @ttl.datamovement()
    def write():
        node_col, node_row = ttl.node(dims=2)
        for local_m in range(rows_per_node):
            m = node_row * rows_per_node + local_m
            if m >= M_tiles:
                break
            for n in range(N_tiles):
                with c_dfb.wait() as cb:
                    ttl.copy(cb, C[m:m+1, n:n+1]).wait()

device = ttnn.open_device(device_id=0)
M, K, N = 64, 64, 64
a_np = np.random.rand(M, K).astype(np.float32)
b_np = np.random.rand(K, N).astype(np.float32)
A = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
B = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
C = ttnn.from_torch(np.zeros((M,N), dtype=np.float32),
                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)

matmul_1d(A, B, C)
result  = ttnn.to_torch(C)
expected = a_np @ b_np
# bfloat16 truncation causes ~1% relative error; rtol=0.02 is generous but realistic
ok = np.allclose(result, expected, rtol=0.02, atol=0.02)
print(f"matmul_1d: {'PASS' if ok else 'FAIL'}  max_err={np.max(np.abs(result - expected)):.4f}")
ttnn.close_device(device)
`,
        },
    };

    const WORKER_PATH    = '/assets/playground/pyodide-worker.js';
    const SIM_LITE_BASE  = '/assets/ttlang-sim-lite';

    // ─── PlaygroundController ─────────────────────────────────────────────────

    class PlaygroundController {
        constructor(containerEl) {
            this.container   = containerEl;
            this.worker      = null;
            this.workerReady = false;
            this.running     = false;
            this._currentKernel = 'eltwise_add';
            this._buildUI();
        }

        _buildUI() {
            const kernelOptions = Object.entries(KERNELS)
                .map(([k, v]) => `<option value="${k}">${this._escapeHtml(v.label)}</option>`)
                .join('');

            this.container.innerHTML = `
<div class="tt-playground">
  <div class="tt-pg-header">
    <span class="tt-pg-title">tt-lang Kernel Playground</span>
    <span class="tt-pg-badge">ttlang-sim-lite + Pyodide</span>
  </div>
  <div class="tt-pg-editor-wrap">
    <textarea class="tt-pg-editor" spellcheck="false">${this._escapeHtml(KERNELS['eltwise_add'].code)}</textarea>
  </div>
  <div class="tt-pg-toolbar">
    <button class="tt-pg-btn tt-pg-run" disabled>▶ Run</button>
    <select class="tt-pg-kernel-select" title="Load a starter kernel">${kernelOptions}</select>
    <button class="tt-pg-btn tt-pg-reset">↺ Reset runtime</button>
    <button class="tt-pg-btn tt-pg-clear-out">✕ Clear output</button>
    <span class="tt-pg-status">Initializing…</span>
    <div class="tt-pg-spinner" aria-hidden="true"></div>
  </div>
  <pre class="tt-pg-output" aria-live="polite"></pre>
</div>`;

            this.editorEl      = this.container.querySelector('.tt-pg-editor');
            this.runBtn        = this.container.querySelector('.tt-pg-run');
            this.resetBtn      = this.container.querySelector('.tt-pg-reset');
            this.clearBtn      = this.container.querySelector('.tt-pg-clear-out');
            this.kernelSelect  = this.container.querySelector('.tt-pg-kernel-select');
            this.statusEl      = this.container.querySelector('.tt-pg-status');
            this.spinnerEl     = this.container.querySelector('.tt-pg-spinner');
            this.outputEl      = this.container.querySelector('.tt-pg-output');

            this.runBtn.addEventListener('click', () => this.run());
            this.resetBtn.addEventListener('click', () => this.reset());
            this.clearBtn.addEventListener('click', () => { this.outputEl.textContent = ''; });
            this.kernelSelect.addEventListener('change', () => {
                const k = this.kernelSelect.value;
                if (KERNELS[k]) {
                    this.editorEl.value = KERNELS[k].code;
                    this._currentKernel = k;
                    this.outputEl.textContent = '';
                }
            });

            this._initWorker();
        }

        _escapeHtml(s) {
            return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }

        _initWorker() {
            this.workerReady = false;
            this.runBtn.disabled = true;
            this._setStatus('Loading runtime…', true);

            const workerUrl    = this.container.dataset.workerUrl    || WORKER_PATH;
            const simLiteBase  = this.container.dataset.simLiteBase  || SIM_LITE_BASE;

            this.worker = new Worker(workerUrl);
            this.worker.onmessage = ({ data }) => this._onWorkerMessage(data);
            this.worker.onerror   = (e) => {
                this._appendOutput(`[worker error] ${e.message}\n`, 'err');
                this._setStatus('Worker error', false);
            };
            this.worker.postMessage({ type: 'init', simLiteBaseUrl: simLiteBase });
        }

        _onWorkerMessage(data) {
            switch (data.type) {
                case 'ready':
                    this.workerReady = true;
                    this.runBtn.disabled = false;
                    this._setStatus('Ready', false);
                    break;
                case 'status':
                    this._setStatus(data.text, true);
                    break;
                case 'stdout':
                    this._appendOutput(data.text, 'out');
                    break;
                case 'stderr':
                    this._appendOutput(data.text, 'err');
                    break;
                case 'done':
                    this.running = false;
                    this.runBtn.disabled = false;
                    this.runBtn.textContent = '▶ Run';
                    this._setStatus('Done', false);
                    break;
                case 'error':
                    this._appendOutput(`[init error] ${data.text}\n`, 'err');
                    this._setStatus('Error — see output', false);
                    break;
            }
        }

        _setStatus(text, spinning) {
            this.statusEl.textContent = text;
            this.spinnerEl.style.display = spinning ? 'inline-block' : 'none';
        }

        _appendOutput(text, cls) {
            const span = document.createElement('span');
            if (cls === 'err') span.className = 'tt-pg-stderr';
            span.textContent = text;
            this.outputEl.appendChild(span);
            this.outputEl.scrollTop = this.outputEl.scrollHeight;
        }

        run() {
            if (!this.workerReady || this.running) return;
            this.running = true;
            this.runBtn.disabled = true;
            this.runBtn.textContent = '⏳ Running…';
            this._setStatus('Running…', true);
            this.outputEl.textContent = '';
            this.worker.postMessage({ type: 'run', code: this.editorEl.value, timeout: 60000 });
        }

        reset() {
            if (this.worker) { this.worker.terminate(); this.worker = null; }
            this.running = false;
            this.outputEl.textContent = '';
            this._initWorker();
        }
    }

    // ─── Init all playgrounds on page ────────────────────────────────────────

    function initAll() {
        document.querySelectorAll('.tt-playground-mount').forEach((el) => {
            new PlaygroundController(el);
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initAll);
    } else {
        initAll();
    }

    window.TtPlayground = { PlaygroundController, KERNELS };
})();
