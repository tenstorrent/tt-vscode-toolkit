# `tt_patches` — Ninja Monkeypatch Harness for TT-NN / TT-Metalium

Change TT-NN behavior with the **smallest possible trace**, and stay
**upgrade-safe** — no `tt-metal` fork, no editing installed package files.

## Why

On a TT-QuietBox 2, `ttnn` is an installed Python package with **no
`~/tt-metal` source tree**. Editing files under `site-packages` is invisible
and gets wiped by the next upgrade. This harness patches at runtime instead:
every change is saved, reversible, logged, and fails loud if upstream renames
the thing you patched.

## Install

Copy `tt_patches.py` into your project (e.g. next to your entry point). It has
no third-party dependencies.

## The one rule

`ttnn` reads config at **import time**. Apply env changes and import your patch
module **before** `import ttnn`.

## API

| Call | Use |
|---|---|
| `reg.wrap(obj, attr, make_wrapper, label=...)` | Log/profile or fix behavior; `make_wrapper(original)` returns the replacement |
| `reg.set_default(obj, attr, value, label=...)` | Change a constant/default (dtype, trace size) |
| `reg.unwrap_all()` | Restore everything |
| `with patched(reg): ...` | Scope patches to a block; auto-restore |
| `version_at_most(current, ceiling)` | Retire a bugfix patch once upstream is fixed |
| `verify((name, probe), ...)` | Mechanical check an AI agent can gate on |

## Example: log every `ttnn.add`

```python
import tt_patches
from tt_patches import PatchRegistry
import ttnn

reg = PatchRegistry()

def log_calls(original):
    def wrapper(*args, **kwargs):
        print(f"ttnn.add called with {len(args)} tensors")
        return original(*args, **kwargs)
    return wrapper

reg.wrap(ttnn, "add", log_calls, label="trace-add")
# ... run your model ...
reg.unwrap_all()
```

## Example: AI-agent verification after patching

```python
ok = tt_patches.verify(
    ("add-exists", lambda: getattr(ttnn, "add")),
    ("smoke", lambda: ttnn.add(ttnn.zeros([32, 32]), ttnn.zeros([32, 32]))),
)
assert ok, "patch verification failed — do not proceed"
```

## Testing

`python3 -m pytest test_tt_patches.py -v` — runs anywhere, no hardware needed.

**Rule of thumb:** *patch* to change behavior; *wrap a thin library* to add it.
