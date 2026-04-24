"""
Test script for ttlang-sim-lite (numpy backend).

Run from the tt-vscode-toolkit root:
    python content/web/ttlang-sim-lite/test_sim_lite.py

This verifies that the torch→numpy swap leaves the simulator semantics intact.
"""

import sys
import os

# Remove the script directory from sys.path immediately.
# Python adds it when running a script directly, which would shadow stdlib
# modules (copy, math) with our local copies.
_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _script_dir]

import types
import importlib.util

# The Python-importable canonical copy lives in the sibling ttlang_sim_lite/ directory.
# Point BASE there so this test exercises the actual importable package, not the
# hyphen-named mirror that Python cannot import by name.
_HYPHEN_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(os.path.dirname(_HYPHEN_DIR), 'ttlang_sim_lite')
PKG = "ttlang_sim_lite"

# ──────────────────────────────────────────────────────────────────────────────
# Bootstrap the sim-lite package without polluting sys.path
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_sim_lite():
    pkg = types.ModuleType(PKG)
    pkg.__path__ = [BASE]
    pkg.__package__ = PKG
    sys.modules[PKG] = pkg

    def load(name):
        full = f"{PKG}.{name}"
        spec = importlib.util.spec_from_file_location(
            full, os.path.join(BASE, f"{name}.py"),
            submodule_search_locations=[BASE],
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        return mod

    for name in [
        "torch_compat", "typedefs", "constants", "errors", "blockstate",
        "context_types", "diagnostics", "debug_print", "dfbstate",
        "trace", "ttnnsim", "greenlet_scheduler", "context", "decorators",
        # dfb, pipe, sharding must come before copyhandlers/copy which import from them.
        # Loading them late would cause auto-import to create a different class object,
        # breaking HANDLER_REGISTRY key identity at runtime.
        "corecontext", "dfb", "pipe", "sharding",
        "operation", "copyhandlers", "copy",
        "program", "torch_utils", "math", "ttlang_sim", "__init__",
    ]:
        load(name)

    return sys.modules[PKG]


pkg = bootstrap_sim_lite()
sim_init = sys.modules[f"{PKG}.__init__"]
ttnn = sys.modules[f"{PKG}.ttnnsim"]
ttl = sim_init.ttl

# Expose as top-level modules so kernels can `import ttl; import ttnn`
sys.modules["ttl"] = sim_init.ttl  # type: ignore[assignment]
sys.modules["ttnn"] = ttnn

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────

PASSED = 0
FAILED = 0


def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  PASS  {name}")
        PASSED += 1
    except Exception as e:
        print(f"  FAIL  {name}: {e}")
        import traceback; traceback.print_exc()
        FAILED += 1


def test_tensor_creation():
    device = ttnn.open_device(device_id=0)
    a = ttnn.rand((32, 32), dtype=ttnn.bfloat16)
    assert a.shape == (32, 32), f"shape mismatch: {a.shape}"
    assert a._tensor.dtype == np.float32


def test_from_to_torch():
    device = ttnn.open_device()
    a_np = np.random.rand(64, 64).astype(np.float32)
    t = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    recovered = ttnn.to_torch(t)
    assert np.allclose(a_np, recovered, atol=1e-5), "from_torch/to_torch roundtrip failed"


def _run_kernel(name):
    """Load and run a kernel's main() from the kernels/ subdirectory."""
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        name,
        os.path.join(BASE, "kernels", f"{name}.py"),
    )
    mod = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()


def test_eltwise_add_kernel():
    """Run eltwise_add kernel through the full greenlet-scheduled simulator."""
    _run_kernel("eltwise_add")


def test_fused_mma_kernel():
    """Run fused multiply-add (a*b+c) kernel: verifies the three-thread DFB model."""
    _run_kernel("fused_mma")


def test_matmul_relu_kernel():
    """Run fused matmul+bias+ReLU kernel: verifies k-reduction accumulator ping-pong."""
    _run_kernel("matmul_relu")


def test_tensor_arithmetic():
    device = ttnn.open_device()
    a_np = np.ones((64, 64), dtype=np.float32) * 2.0
    b_np = np.ones((64, 64), dtype=np.float32) * 3.0
    a = ttnn.from_torch(a_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    b = ttnn.from_torch(b_np, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device)
    result_tensor = a + b
    result = ttnn.to_torch(result_tensor)
    assert np.allclose(result, 5.0, atol=1e-4), f"Expected 5.0, got {result.mean()}"


def test_dtype_aliases():
    assert ttnn.bfloat16 == np.float32
    assert ttnn.float32 == np.float32


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────

print("ttlang-sim-lite test suite")
print("=" * 40)
test("tensor_creation", test_tensor_creation)
test("from_to_torch roundtrip", test_from_to_torch)
test("dtype aliases", test_dtype_aliases)
test("tensor arithmetic", test_tensor_arithmetic)
test("eltwise_add kernel (full sim)", test_eltwise_add_kernel)
test("fused_mma kernel (a*b+c, 3-thread DFB)", test_fused_mma_kernel)
test("matmul_relu kernel (k-reduction + ReLU)", test_matmul_relu_kernel)

print("=" * 40)
print(f"Results: {PASSED} passed, {FAILED} failed")
if FAILED:
    sys.exit(1)
