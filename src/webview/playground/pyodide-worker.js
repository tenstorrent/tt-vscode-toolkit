// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
//
// Pyodide Web Worker — loads ttlang-sim-lite and executes user Python code.
// Runs in a dedicated worker so the main thread stays responsive during load.

const PYODIDE_VERSION = '0.26.4';

let pyodide = null;
let ready = false;
let simLiteBaseUrl = '../ttlang-sim-lite';

// Ordered list matching test_sim_lite.py bootstrap (dfb/pipe/sharding before copyhandlers).
const SIM_MODULES = [
    'torch_compat', 'typedefs', 'constants', 'errors', 'blockstate',
    'context_types', 'diagnostics', 'debug_print', 'dfbstate',
    'trace', 'ttnnsim', 'greenlet_scheduler', 'context', 'decorators',
    'corecontext', 'dfb', 'pipe', 'sharding',
    'operation', 'copyhandlers', 'copy',
    'program', 'torch_utils', 'math', 'ttlang_sim', '__init__',
];

const SIM_KERNELS = ['eltwise_add', 'matmul_1d', 'fused_mma', 'matmul_relu'];

// Python bootstrap: injects greenlet shim, then registers the ttlang_sim_lite package.
const BOOTSTRAP_PY = `
import sys, types, importlib.util

_script_dir = '/ttlang_sim_lite'
sys.path = [p for p in sys.path if p != _script_dir]

# ── Greenlet shim ────────────────────────────────────────────────────────────
# greenlet is a C extension not available in Pyodide.  We provide a pure-Python
# drop-in backed by threading.Thread + threading.Event.
if 'greenlet' not in sys.modules:
    _gl_spec = importlib.util.spec_from_file_location(
        'greenlet', '/ttlang_sim_lite/greenlet_shim.py'
    )
    _gl_mod = importlib.util.module_from_spec(_gl_spec)
    sys.modules['greenlet'] = _gl_mod
    _gl_spec.loader.exec_module(_gl_mod)
    print("greenlet: using sequential shim (Pyodide)", flush=True)

# ── ttlang-sim-lite package ───────────────────────────────────────────────────
PKG = "ttlang_sim_lite"
BASE = "/ttlang_sim_lite"

_pkg = types.ModuleType(PKG)
_pkg.__path__ = [BASE]
_pkg.__package__ = PKG
sys.modules[PKG] = _pkg

def _load(name):
    full = f"{PKG}.{name}"
    spec = importlib.util.spec_from_file_location(
        full, f"{BASE}/{name}.py",
        submodule_search_locations=[BASE],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = PKG
    sys.modules[full] = mod
    spec.loader.exec_module(mod)
    return mod

for _name in ${JSON.stringify(SIM_MODULES)}:
    _load(_name)

_sim_init = sys.modules[f"{PKG}.__init__"]
_ttnn = sys.modules[f"{PKG}.ttnnsim"]
_ttl = _sim_init.ttl
sys.modules["ttl"] = _ttl
sys.modules["ttnn"] = _ttnn

# ── Pyodide sequential-mode patch ────────────────────────────────────────────
# The sequential greenlet shim runs each thread function to completion without
# cooperative switching.  This requires that DFB reserve()/wait() never block.
# Patching make_dataflow_buffer_like to use block_count=4096 ensures that
# every DFB has enough slots for the entire kernel to run without blocking.
# Suppress the L1-overflow warning that the inflated block_count triggers.
import warnings
warnings.filterwarnings("ignore", message=".*DataflowBuffer capacity.*L1 memory limit.*")

_dfb_mod = sys.modules[f"{PKG}.dfb"]
_orig_make_dfb = _dfb_mod.make_dataflow_buffer_like

def _make_dfb_unlimited(tensor, shape, block_count=2, **kw):
    return _orig_make_dfb(tensor, shape, block_count=4096, **kw)

_dfb_mod.make_dataflow_buffer_like = _make_dfb_unlimited
_ttl.make_dataflow_buffer_like = _make_dfb_unlimited
print("DFB block_count patched to 4096 for sequential execution", flush=True)

print("ttlang-sim-lite ready", flush=True)
`;

function emit(type, payload) {
    self.postMessage({ type, ...payload });
}

async function fetchText(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`fetch ${url}: ${resp.status} ${resp.statusText}`);
    return resp.text();
}

async function loadSimLiteIntoFS() {
    try {
        pyodide.FS.mkdir('/ttlang_sim_lite');
    } catch (_) { /* already exists */ }

    // Fetch greenlet_shim.py first — bootstrap needs it before any sim module loads.
    const shimText = await fetchText(`${simLiteBaseUrl}/greenlet_shim.py`);
    pyodide.FS.writeFile('/ttlang_sim_lite/greenlet_shim.py', shimText);

    for (const name of SIM_MODULES) {
        const text = await fetchText(`${simLiteBaseUrl}/${name}.py`);
        pyodide.FS.writeFile(`/ttlang_sim_lite/${name}.py`, text);
    }

    try {
        pyodide.FS.mkdir('/ttlang_sim_lite/kernels');
    } catch (_) {}

    for (const name of SIM_KERNELS) {
        try {
            const text = await fetchText(`${simLiteBaseUrl}/kernels/${name}.py`);
            pyodide.FS.writeFile(`/ttlang_sim_lite/kernels/${name}.py`, text);
        } catch (_) { /* optional kernel — skip */ }
    }
}

async function initPyodide() {
    try {
        emit('status', { text: 'Loading Pyodide runtime…' });

        importScripts(`https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/pyodide.js`);

        pyodide = await loadPyodide({
            indexURL: `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`,
            stdout: (t) => emit('stdout', { text: t + '\n' }),
            stderr: (t) => emit('stderr', { text: t + '\n' }),
        });

        emit('status', { text: 'Loading numpy + pydantic…' });
        await pyodide.loadPackage(['numpy', 'pydantic']);

        emit('status', { text: 'Loading ttlang-sim-lite…' });
        await loadSimLiteIntoFS();

        await pyodide.runPythonAsync(BOOTSTRAP_PY);

        ready = true;
        emit('ready', {});
    } catch (e) {
        emit('error', { text: String(e) });
    }
}

async function runCode(code, timeout = 30000) {
    if (!ready) {
        emit('stderr', { text: 'Runtime not ready yet — please wait.\n' });
        emit('done', {});
        return;
    }
    try {
        // Wrap in a coroutine so asyncio.wait_for can enforce timeout if needed
        await Promise.race([
            pyodide.runPythonAsync(code),
            new Promise((_, reject) =>
                setTimeout(() => reject(new Error(`Execution timed out after ${timeout / 1000}s`)), timeout)
            ),
        ]);
    } catch (e) {
        emit('stderr', { text: String(e) + '\n' });
    }
    emit('done', {});
}

self.onmessage = async ({ data }) => {
    switch (data.type) {
        case 'init':
            if (data.simLiteBaseUrl) simLiteBaseUrl = data.simLiteBaseUrl;
            await initPyodide();
            break;
        case 'run':
            await runCode(data.code, data.timeout);
            break;
        case 'reset':
            ready = false;
            pyodide = null;
            await initPyodide();
            break;
    }
};
