---
id: monkeypatch-ttnn
title: "Monkeypatching TT-NN"
description: >-
  Change TT-NN / TT-Metalium behavior with the smallest possible trace and
  without forking tt-metal — add logging, work around a bug, tweak a default,
  or register a model, all while staying upgrade-safe. Built for TT-QuietBox 2,
  where ttnn is an installed package with no source tree.
category: advanced
tags:
  - ttnn
  - metalium
  - patching
  - workflow
supportedHardware:
  - n150
  - n300
  - t3k
  - p100
  - p150
  - p300c
  - galaxy
status: validated
validatedOn:
  - p300c
estimatedMinutes: 25
---

# Monkeypatching TT-NN<sup>™</sup>

You want to change how **TT-NN** or **TT-Metalium<sup>™</sup>** behaves — add a log line, work
around a bug, bump a default, register a new model — but you *don't* want to fork
`tt-metal` and you *definitely* don't want to hand-edit files inside an installed
package. This lesson teaches monkeypatching as a discipline: change behavior with
the **smallest possible trace**, and in a way that stays **upgrade-safe**.

This matters most on a **TT-QuietBox<sup>®</sup> 2**, where `ttnn` arrives as a *pre-installed
Python package* and there is usually **no `~/tt-metal` source tree at all**. There's
nothing to fork. So you reach into the package at runtime instead — carefully.

## The two axes every technique is judged on

- **Smallest trace** — how little of the system you disturb, and how easily your
  change can be *seen*, *reasoned about*, and *reversed*.
- **Upgrade-safety** — will a `pip`/image/`tt-metal` upgrade silently break or
  silently revert your change, and does it **fail loud** when its assumptions no
  longer hold?

> 💡 **Rule of thumb:** *Patch* to **change** behavior. *Wrap a thin library* to
> **add** behavior. Editing the source tree is the last resort, not the first.

## What You'll Learn

- 📍 Where `ttnn` actually lives, and why editing it in place is a trap
- ⏱️ The one rule underneath everything: **env + import order**
- 🔍 A reusable, hardware-free-testable patch harness (`tt_patches.py`)
- 🎯 The right technique for each goal: observe, fix, change a default, add new
- 🧪 A verification recipe an **AI agent** can gate its own work on

---

## Orientation: find what you're patching

First, see where the package you're about to patch lives:

```bash
python3 -c "import ttnn, os; print(ttnn.__file__)"
```

On a TT-QuietBox 2 this prints somewhere under `site-packages`, **not** your home directory.

> ⚠️ **Editing files under `site-packages` is not a patch.** It's invisible to
> anyone reading your project, it's not in your git history, and the next
> `pip install --upgrade` or image refresh wipes it without warning. Everything
> below keeps your changes in *your* code instead.

### The one rule: env + import order

`ttnn` reads a lot of its configuration — visible devices, arch name, mesh shape,
dispatch — **at `import` time**. That means any environment variable or namespace
patch you want to take effect must land **before** `import ttnn` runs.

Set environment in the shell before Python starts:

```bash
# blackhole for P-series (p100/p150/p300c/TT-QuietBox 2); wormhole_b0 for N-series.
# The := form honours a value you already exported.
: "${TT_METAL_ARCH_NAME:=wormhole_b0}"
export TT_METAL_ARCH_NAME
```

Or set it in Python — but only *above* the import, and defer the import so it
can't accidentally run first:

```python
import os

os.environ.setdefault("TT_METAL_ARCH_NAME", "blackhole")
os.environ.setdefault("TT_VISIBLE_DEVICES", "0")

import ttnn  # noqa: E402 — intentionally after the env is set
```

Pushing `import ttnn` down into a function (a *deferred import*) is a common way
to guarantee your setup runs first. If you take nothing else from this lesson:
**get your patch in before the import.**

---

## Meet the harness: `tt_patches.py`

Rather than sprinkle raw `setattr` calls across your project, use one small,
dependency-free helper. Copy `content/templates/monkeypatch/tt_patches.py` into
your project (it ships with `test_tt_patches.py`, which runs anywhere — no
hardware needed).

It gives you:

| Call | Use |
|---|---|
| `reg.wrap(obj, attr, make_wrapper, label=...)` | Log/profile or fix behavior; `make_wrapper(original)` returns the replacement |
| `reg.set_default(obj, attr, value, label=...)` | Change a constant/default (dtype, trace size) |
| `reg.unwrap_all()` | Restore everything |
| `with patched(reg): ...` | Scope patches to a block; auto-restore even on error |
| `version_at_most(current, ceiling)` | Retire a bugfix patch once upstream is fixed |
| `verify((name, probe), ...)` | Mechanical check an AI agent can gate on |

Every `wrap`/`set_default` **saves the original**, **logs a line**, and **raises
`PatchError` if the target is missing** — so an upstream rename fails loud instead
of silently doing nothing. That's upgrade-safety built into the tool.

Now, by goal.

---

## Goal 1 — "I want to see what's happening"

The lowest-trace patch there is: wrap a function to observe it, call the original
inside, change nothing about the result. Perfect for logging, shape checks, or
timing. This is exactly the shape of a host-side profiler (the pattern behind
`tt-qwythos`'s decode profiler): save → wrap → restore, math untouched.

```python
import tt_patches
from tt_patches import PatchRegistry
import ttnn

reg = PatchRegistry()

def trace_shapes(original):
    def wrapper(*args, **kwargs):
        shapes = [tuple(a.shape) for a in args if hasattr(a, "shape")]
        print(f"ttnn.matmul inputs: {shapes}")
        return original(*args, **kwargs)   # original runs unchanged
    return wrapper

reg.wrap(ttnn, "matmul", trace_shapes, label="trace-matmul")

# ... run your model ...

reg.unwrap_all()   # leave no trace
```

Prefer a scoped patch when you only need it for one region:

```python
from tt_patches import patched

with patched(reg):          # reg already has wraps queued
    output = model(inputs)  # observed here
# automatically unwrapped here, even if model() raised
```

**Upgrade-safety:** purely additive and reversible — the safest technique. If
`ttnn.matmul` is ever renamed, `wrap` raises `PatchError` and you find out
immediately.

---

## Goal 2 — "I want to fix a bug before upstream does"

Sometimes upstream has a bug and you can't wait for the release. Wrap the function
to *replace* its behavior — but **guard the patch with a version check** so it
retires itself the moment the fix lands. Otherwise you'll silently keep overriding
a now-correct implementation and quietly miss the upgrade.

```python
import importlib.metadata
import tt_patches
from tt_patches import PatchRegistry, version_at_most
import ttnn

reg = PatchRegistry()

# TT-NN has no `ttnn.__version__`; read the installed package version instead.
ttnn_version = importlib.metadata.version("ttnn")   # e.g. "0.65.1rc17.dev6200"
FIXED_IN = "0.51.0"   # the release that fixes this upstream

if version_at_most(ttnn_version, FIXED_IN):
    def corrected(original):
        def wrapper(*args, **kwargs):
            # ... the corrected behavior; call original where still valid ...
            return original(*args, **kwargs)
        return wrapper

    reg.wrap(ttnn, "some_buggy_op", corrected, label=f"bugfix<= {FIXED_IN}")
    print(f"tt_patches: applied bugfix for ttnn {ttnn_version}")
else:
    print(f"tt_patches: bugfix no longer needed on ttnn {ttnn_version} — remove it")
```

> ⚠️ **There is no `ttnn.__version__`.** Use
> `importlib.metadata.version("ttnn")` (verified on TT-NN `0.65.x`). `version_at_most`
> parses the leading numeric segments, so a `…rc17.dev6200` suffix is handled fine.

**Upgrade-safety:** the `else` branch is the important half — it tells you (loudly)
when the patch is dead weight, so patches don't accumulate silently across upgrades.

---

## Goal 3 — "I want to change a default"

Defaults like a dtype, a mesh shape, a dispatch config, or a trace-region size.
Two tools: set the constant with `set_default`, or set an environment variable
before import (Goal-0 rule).

**A real one — trace region size.** Large models can exhaust the default trace
buffer on a P300X2 during warmup; the fix is to raise `trace_region_size` on the
config/runner rather than editing source (this is what `tt-local-generator` does
for its FLUX runner on TT-QuietBox 2):

```python
# Bump a runner/config default without touching the installed package.
reg.set_default(my_runner_config, "trace_region_size", 64_000_000, label="flux-trace")
```

**A real gotcha — dispatch core axis.** Never hard-code
`ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER, ttnn.DispatchCoreAxis.ROW)`:
`DispatchCoreAxis.ROW` **crashes on Blackhole**. Prefer letting TT-NN auto-detect
the axis (COL on Blackhole, ROW on Wormhole):

```python
# ✅ Let TT-NN choose the axis for the detected architecture.
config = ttnn.DispatchCoreConfig(ttnn.DispatchCoreType.WORKER)
```

If you must force a default across a whole app, prefer an env var set before
`import ttnn` over patching the constructor — it's less surprising to the next
reader.

**Upgrade-safety:** `set_default` saves the original value, so `unwrap_all()`
restores it, and a renamed attribute raises `PatchError`.

---

## Goal 4 — "I want to add something new"

Registering a model, adding a composed op, or extending a request model is often
more than one function. You have two good options — and one even better one.

**(a) Inject into the namespace.** For a pure-Python addition, attach your new
callable to the package so the rest of your code finds it in the usual place:

```python
def rms_then_add(x, weight, bias):
    return ttnn.add(ttnn.rms_norm(x, weight), bias)

# Register a composed op. Guard so you don't clobber a future real one.
if not hasattr(ttnn, "rms_then_add"):
    setattr(ttnn, "rms_then_add", rms_then_add)
```

**(b) Overlay files onto an installed package/image.** When you need to add or
replace whole modules inside a package you don't own — for example new server
runners inside a `tt-inference-server` container — **bind-mount** your files over
the package path at startup instead of rebuilding the image. `tt-local-generator`
does exactly this: patch files are bind-mounted over `~/tt-metal/server/` and
`~/tt-metal/models/` when the container launches, so nothing in the image is
edited and no rebuild is needed. The overlay lives in *your* repo, versioned and
reviewable.

**(c) When *not* to patch at all — wrap, don't patch.** If you're purely *adding*
capability and not altering existing behavior, a thin **external library on top of**
`ttnn` is lower-trace and more upgrade-safe than any patch: you own the new
surface, `ttnn` stays untouched, and an upgrade can't revert you. This is the
model behind Martin Chang's [`ttPseudoRowMajor`](https://github.com/marty1885/ttPseudoRowMajor)
(an additive, non-invasive TT-NN-facing library) and his upstream-first ggml /
llama.cpp Tenstorrent backend. Reach for a patch when you must *change* behavior;
reach for a wrapper when you're *adding* it.

---

## Goal 5 — Escape hatch: editing the source tree (last resort)

> 🛑 **This is the heaviest, least TT-QuietBox 2-friendly option.** It requires a cloned and
> built `tt-metal` — which a stock TT-QuietBox 2 does **not** have. See
> [Build TT-Metalium from Source](command:tenstorrent.showLesson?["build-tt-metal"])
> if you need one. Exhaust Goals 1–4 first.

Some changes a runtime patch simply can't reach: compiled C++ / kernel behavior,
or edits so invasive that a Python wrap would be more fragile than an honest source
change. When you genuinely hit that wall, don't fork-and-drift — keep your change
as an **additive, tracked diff** you can re-apply after each upstream update. This
is how `tt-qwythos` manages its `tt_metal_patches/` (for example, registering a new model
in `tt_transformers`): every edit is additive, documented with a "why this is
safe" note, and stored as a `git diff` snapshot.

```bash
# Capture your change as a re-appliable patch, scoped to just what you touched.
git -C ~/tt-metal diff models/tt_transformers > my_project/patches/tt_transformers.diff

# Re-apply after pulling a new tt-metal.
git -C ~/tt-metal apply my_project/patches/tt_transformers.diff
```

**Upgrade-safety:** a tracked diff *tells you* when it stops applying cleanly —
that conflict is your signal that upstream moved, which is exactly the loud
failure you want. Keep the diff small and additive so it keeps applying across
versions.

---

## Make your patches durable

A few habits keep patches from rotting into mystery bugs:

- **One `patches.py`, imported first.** Put every patch in a single module and
  import it before you touch `ttnn`. One place to read, one place to remove.
- **Document why & when.** Each patch gets a docstring: *why it's safe* and *when
  to remove it* (a version, a fixed issue, a condition).
- **Fail loud on upgrade.** The harness already raises `PatchError` when a target
  is missing — don't swallow it. A crash on startup after an upgrade beats a
  model that silently behaves differently.

### For AI agents: verify before you proceed

If an agent is applying patches autonomously, it needs a mechanical way to confirm
the patch took and nothing broke — no human in the loop. Use `verify`:

```python
import tt_patches
import ttnn

ok = tt_patches.verify(
    ("op-exists",  lambda: getattr(ttnn, "matmul")),                     # target present?
    ("smoke",      lambda: ttnn.add(ttnn.zeros([32, 32]),
                                    ttnn.zeros([32, 32]))),              # tiny op runs?
)
assert ok, "patch verification failed — do not proceed"
```

Pair that at the file level with a compile check the agent can run before importing
anything:

```bash
python3 -m py_compile patches.py && echo PATCH_MODULE_OK
```

`verify` returns a boolean and logs any failing probe, so an agent can branch on
the result instead of crashing.

---

## Recap

| Goal | Technique | Seen in | Upgrade-safety |
|------|-----------|---------|----------------|
| See what's happening | `wrap` (observe, call original) | tt-qwythos profiler | Additive, reversible — safest |
| Fix a bug early | `wrap` to replace + `version_at_most` guard | — | Retires itself when upstream fixes it |
| Change a default | `set_default` or env-before-import | tt-local-generator (trace size) | Original saved; rename raises |
| Add something new | namespace inject / file overlay / **thin wrapper lib** | tt-local-generator overlays; marty1885 `ttPseudoRowMajor` | Wrapper untouched by upgrades |
| Source change (last resort) | additive tracked `git diff` | tt-qwythos `tt_metal_patches/` | Conflict = loud "upstream moved" signal |

The through-line: **do the least, and make it loud when it stops being valid.**
That's how you keep moving fast on TT hardware without a fork — and without missing
the upgrades that matter.
