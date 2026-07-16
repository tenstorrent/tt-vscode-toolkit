# Monkeypatching TT-NN Lesson — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a toolkit lesson that teaches upgrade-safe, smallest-trace monkeypatching of TT-NN / TT-Metalium (organized by goal), backed by a reusable, hardware-free-testable `tt_patches.py` harness.

**Architecture:** A pure-Python patch harness (`content/templates/monkeypatch/tt_patches.py`) implements save→setattr→restore with fail-loud missing-target detection, a scoped context manager, a version guard, and an agent verification recipe — all unit-testable against fake objects with no `ttnn` import. A markdown lesson (`content/lessons/monkeypatch-ttnn.md`) walks through the harness organized by developer goal. The lesson is wired into the extension via `content/lesson-registry.json` only (this repo no longer uses `package.json` walkthroughs).

**Tech Stack:** TypeScript VSCode extension (build/validation), Python 3 (template + pytest), Node scripts for lesson-registry sync, Vale for docs lint.

## Global Constraints

- **Version bump required:** increment the `package.json` version by one PATCH level for this content change (from whatever the current released version is).
- **Lesson-registry sync:** markdown front matter is source of truth for `id, title, description, category, tags, supportedHardware, status, validatedOn, estimatedMinutes`; `order, previousLesson, nextLesson, completionEvents, markdownFile` are manually maintained in `content/lesson-registry.json`. `npm run validate:lessons` must pass (it is wired into `npm run build`).
- **No `package.json` walkthrough edits:** `contributes.walkthroughs` does not exist in this repo; do not add it.
- **Template test gate:** every `.py` under `content/templates/` must pass `python3 -m py_compile` (enforced by `npm run test:templates`).
- **Command URIs:** `npm run validate:command-uris` runs in build. `command:` links ARE allowed as long as they resolve to a command registered via `registerCommand(...)` in `extension.ts` (e.g. `command:tenstorrent.showLesson?[...]`); do not reference commands that aren't registered.
- **WH/BH compatibility copy rules (from CLAUDE.md):** use `hf` CLI (never `huggingface-cli`); never use `ttnn.DispatchCoreAxis.ROW`; do not assume `~/tt-metal` exists; `TT_METAL_ARCH_NAME` = `blackhole` (P-series) / `wormhole_b0` (N-series).
- **No proprietary code:** re-express patterns generically; cite source repos by name, do not paste their code.
- **Trademark superscripts:** follow existing lesson convention — TT-Metalium<sup>™</sup>, TT-NN<sup>™</sup>, TT-Forge<sup>™</sup> on first prominent use.

---

## File Structure

- Create: `content/templates/monkeypatch/tt_patches.py` — the reusable harness.
- Create: `content/templates/monkeypatch/test_tt_patches.py` — hardware-free pytest for the harness.
- Create: `content/templates/monkeypatch/README.md` — one-page usage of the harness.
- Create: `content/lessons/monkeypatch-ttnn.md` — the lesson (front matter + by-goal body).
- Modify: `content/lesson-registry.json` — add the `monkeypatch-ttnn` entry (via generator) + set manual fields and neighbor navigation.
- Modify: `package.json` — version bump.
- Modify: `CHANGELOG.md` — add entry.

---

## Task 1: Reusable patch harness `tt_patches.py` (TDD, hardware-free)

**Files:**
- Create: `content/templates/monkeypatch/tt_patches.py`
- Test: `content/templates/monkeypatch/test_tt_patches.py`

**Interfaces:**
- Produces:
  - `class PatchError(RuntimeError)`
  - `class PatchRegistry` with:
    - `wrap(self, obj, attr: str, make_wrapper: Callable[[Callable], Callable], *, label: str | None = None) -> None`
    - `set_default(self, obj, attr: str, value, *, label: str | None = None) -> None`
    - `unwrap_all(self) -> None`
    - `applied` property → `list[tuple[object, str, object]]`
  - `patched(registry: PatchRegistry)` — context manager, unwraps on exit (even on exception)
  - `version_at_most(current: str, ceiling: str) -> bool`
  - `verify(*probes: tuple[str, Callable[[], object]]) -> bool`

- [ ] **Step 1: Write the failing test**

Create `content/templates/monkeypatch/test_tt_patches.py`:

```python
"""Hardware-free tests for the tt_patches harness.

These test the save/restore/guard/verify logic against plain fake objects —
no ttnn import, no device, so they run anywhere (CI, laptop, QB2).
Run: pytest content/templates/monkeypatch/test_tt_patches.py -v
"""
import logging

import pytest

from tt_patches import (
    PatchError,
    PatchRegistry,
    patched,
    verify,
    version_at_most,
)


class FakeOps:
    """Stand-in for a module/object we want to patch (e.g. ttnn)."""

    DEFAULT_DTYPE = "bfloat16"

    def add(self, a, b):
        return a + b


def test_wrap_intercepts_then_restores():
    ops = FakeOps()
    reg = PatchRegistry()
    calls = []

    def make_logging_wrapper(original):
        def wrapper(*args, **kwargs):
            calls.append(args)
            return original(*args, **kwargs)
        return wrapper

    reg.wrap(ops, "add", make_logging_wrapper, label="log-add")
    assert ops.add(2, 3) == 5           # behavior preserved
    assert calls == [(2, 3)]            # wrapper observed the call

    reg.unwrap_all()
    assert ops.add(4, 5) == 9
    assert calls == [(2, 3)]            # original no longer records
    assert reg.applied == []


def test_wrap_missing_attribute_fails_loud():
    ops = FakeOps()
    reg = PatchRegistry()
    with pytest.raises(PatchError):
        reg.wrap(ops, "nonexistent_op", lambda orig: orig)


def test_set_default_changes_and_restores():
    ops = FakeOps()
    reg = PatchRegistry()
    reg.set_default(ops, "DEFAULT_DTYPE", "float32", label="dtype")
    assert ops.DEFAULT_DTYPE == "float32"
    reg.unwrap_all()
    assert ops.DEFAULT_DTYPE == "bfloat16"


def test_set_default_missing_attribute_fails_loud():
    ops = FakeOps()
    reg = PatchRegistry()
    with pytest.raises(PatchError):
        reg.set_default(ops, "missing", 1)


def test_patched_context_manager_restores_on_exception():
    ops = FakeOps()
    reg = PatchRegistry()
    reg.set_default(ops, "DEFAULT_DTYPE", "float32")
    with pytest.raises(ValueError):
        with patched(reg):
            assert ops.DEFAULT_DTYPE == "float32"
            raise ValueError("boom")
    assert ops.DEFAULT_DTYPE == "bfloat16"   # restored despite exception


def test_version_at_most():
    assert version_at_most("0.51.0", "0.51.0") is True
    assert version_at_most("0.50.9", "0.51.0") is True
    assert version_at_most("0.52.0", "0.51.0") is False
    # tolerates suffixes / non-numeric trailing segments
    assert version_at_most("0.51.0rc1", "0.51.0") is True


def test_verify_true_when_all_probes_pass():
    assert verify(("one", lambda: 1), ("two", lambda: [][0:0])) is True


def test_verify_false_when_a_probe_raises(caplog):
    def boom():
        raise RuntimeError("nope")
    with caplog.at_level(logging.ERROR):
        assert verify(("ok", lambda: 1), ("bad", boom)) is False
    assert any("bad" in r.message for r in caplog.records)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd content/templates/monkeypatch && python3 -m pytest test_tt_patches.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tt_patches'`

- [ ] **Step 3: Write minimal implementation**

Create `content/templates/monkeypatch/tt_patches.py`:

```python
"""tt_patches — a ninja monkeypatch harness for TT-NN / TT-Metalium.

Change behavior with the smallest possible trace and stay upgrade-safe:

  * Every patch SAVES the original and can be RESTORED (`unwrap_all`).
  * Patching a missing attribute RAISES (`PatchError`) instead of silently
    no-op'ing — so an upstream rename fails loud on the next upgrade.
  * Every applied patch logs a line (visible trace).
  * `version_at_most` lets a bugfix patch retire itself once upstream fixes it.
  * `verify` is a mechanical check an AI agent can gate its work on.

Keep all your project's patches in ONE module that you import BEFORE you use
ttnn (ttnn reads config at import time). Each patch should carry a docstring
saying WHY it is safe and WHEN to remove it.

Rule of thumb: PATCH to change behavior; WRAP a thin library to ADD behavior.
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Callable

logger = logging.getLogger("tt_patches")


class PatchError(RuntimeError):
    """A patch target was missing — usually an upstream rename/removal."""


class PatchRegistry:
    """Tracks applied patches so they can all be reversed."""

    def __init__(self) -> None:
        self._applied: list[tuple[object, str, object]] = []

    @property
    def applied(self) -> list[tuple[object, str, object]]:
        return list(self._applied)

    def _require(self, obj: object, attr: str) -> object:
        if not hasattr(obj, attr):
            raise PatchError(
                f"cannot patch {getattr(obj, '__name__', obj)!r}.{attr}: "
                "attribute missing (did upstream rename/remove it?)"
            )
        return getattr(obj, attr)

    def wrap(
        self,
        obj: object,
        attr: str,
        make_wrapper: Callable[[Callable], Callable],
        *,
        label: str | None = None,
    ) -> None:
        """Replace ``obj.attr`` with ``make_wrapper(original)``.

        ``make_wrapper`` receives the original callable and returns the
        replacement. Use for logging/profiling (call the original inside) or
        for behavior fixes (compute differently).
        """
        original = self._require(obj, attr)
        wrapper = make_wrapper(original)
        setattr(obj, attr, wrapper)
        self._applied.append((obj, attr, original))
        logger.info(
            "tt_patches: wrapped %s.%s%s",
            getattr(obj, "__name__", obj),
            attr,
            f" [{label}]" if label else "",
        )

    def set_default(
        self,
        obj: object,
        attr: str,
        value: object,
        *,
        label: str | None = None,
    ) -> None:
        """Replace a constant/default value (e.g. a dtype or trace size)."""
        original = self._require(obj, attr)
        setattr(obj, attr, value)
        self._applied.append((obj, attr, original))
        logger.info(
            "tt_patches: set %s.%s = %r%s",
            getattr(obj, "__name__", obj),
            attr,
            value,
            f" [{label}]" if label else "",
        )

    def unwrap_all(self) -> None:
        """Restore every patched attribute, in reverse order."""
        for obj, attr, original in reversed(self._applied):
            setattr(obj, attr, original)
        self._applied.clear()


@contextmanager
def patched(registry: PatchRegistry):
    """Scope patches to a block; restore on exit even if it raises."""
    try:
        yield registry
    finally:
        registry.unwrap_all()


def _version_tuple(version: str) -> tuple[int, ...]:
    """Parse leading numeric dotted segments; ignore non-numeric suffixes.

    '0.51.0rc1' -> (0, 51, 0); 'v1.2' -> (1, 2). Dependency-free (no packaging).
    """
    parts: list[int] = []
    for segment in version.lstrip("vV").split("."):
        digits = ""
        for ch in segment:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def version_at_most(current: str, ceiling: str) -> bool:
    """True if ``current`` <= ``ceiling`` — use to retire a bugfix patch once
    upstream is fixed: ``if version_at_most(ttnn.__version__, "0.51.0"): apply()``.
    """
    return _version_tuple(current) <= _version_tuple(ceiling)


def verify(*probes: tuple[str, Callable[[], object]]) -> bool:
    """Run each ``(name, callable)`` probe; return True only if all succeed.

    An AI agent can gate on this after applying patches: assert the patched
    symbol exists, run a tiny smoke op, etc. Failures are logged, not raised.
    """
    ok = True
    for name, probe in probes:
        try:
            probe()
        except Exception as exc:  # noqa: BLE001 - report, don't crash the agent
            logger.error("tt_patches: verify probe %r failed: %s", name, exc)
            ok = False
    return ok
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd content/templates/monkeypatch && python3 -m pytest test_tt_patches.py -v`
Expected: PASS — 8 passed

- [ ] **Step 5: Verify py_compile gate (matches `npm run test:templates`)**

Run: `python3 -m py_compile content/templates/monkeypatch/tt_patches.py content/templates/monkeypatch/test_tt_patches.py && echo COMPILE_OK`
Expected: `COMPILE_OK`

- [ ] **Step 6: Commit**

```bash
git add content/templates/monkeypatch/tt_patches.py content/templates/monkeypatch/test_tt_patches.py
git commit -m "feat(templates): add tt_patches ninja monkeypatch harness + hardware-free tests"
```

---

## Task 2: Harness README + confirm template suite picks it up

**Files:**
- Create: `content/templates/monkeypatch/README.md`
- Test: `npm run test:templates`

**Interfaces:**
- Consumes: `tt_patches.py` public API from Task 1 (`PatchRegistry.wrap/set_default/unwrap_all`, `patched`, `version_at_most`, `verify`, `PatchError`).

- [ ] **Step 1: Write the README**

Create `content/templates/monkeypatch/README.md`:

````markdown
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
````

- [ ] **Step 2: Run the template validation suite**

Run: `npm run test:templates`
Expected: PASS — all Python templates (including `content/templates/monkeypatch/*.py`) have valid syntax.

- [ ] **Step 3: Commit**

```bash
git add content/templates/monkeypatch/README.md
git commit -m "docs(templates): document tt_patches harness usage"
```

---

## Task 3: Lesson markdown `monkeypatch-ttnn.md`

**Files:**
- Create: `content/lessons/monkeypatch-ttnn.md`

**Interfaces:**
- Consumes: the harness at `content/templates/monkeypatch/tt_patches.py` (referenced by relative path in prose).
- Produces: front-matter fields that Task 4 syncs into the registry — `id: monkeypatch-ttnn`, `title`, `description`, `category: advanced`, `tags`, `supportedHardware`, `status: draft`, `validatedOn: []`, `estimatedMinutes: 25`.

- [ ] **Step 1: Write the lesson file**

Create `content/lessons/monkeypatch-ttnn.md` with this front matter, then the body (below):

```yaml
---
id: monkeypatch-ttnn
title: "Monkeypatching TT-NN — Ninja Edition"
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
status: draft
validatedOn: []
estimatedMinutes: 25
---
```

Body requirements (write full prose + fenced code; keep every code block copy-pasteable; obey Global Constraints on copy/trademarks):

1. **Title + intro** — the QB2 reality: `ttnn` is an installed package, usually no `~/tt-metal`. Two axes every technique is judged on: *smallest trace* and *upgrade-safety*. State the rule of thumb: **patch to change behavior, wrap to add behavior**.
2. **Orientation** —
   - Locate the package: ` ```bash\npython3 -c "import ttnn, os; print(ttnn.__file__)"\n``` ` and note editing files there is not a patch (invisible, wiped by upgrades).
   - **The import-order rule:** ttnn reads config at import time; env + patch module must load *before* `import ttnn`. Show a deferred-import snippet and a "set env then import" snippet (e.g. `os.environ["TT_METAL_ARCH_NAME"] = "blackhole"` before `import ttnn`, with the `: "${TT_METAL_ARCH_NAME:=wormhole_b0}"` shell idiom mentioned).
3. **Meet the harness** — introduce `content/templates/monkeypatch/tt_patches.py`; show the `PatchRegistry` API table (wrap / set_default / unwrap_all / patched / version_at_most / verify). One short line: "Full file + tests are in the template; copy it into your project."
4. **Goal 1 — "I want to see what's happening"** — reversible `wrap` for logging/profiling; show the `log_calls` example calling the original inside; note it's host-side, math unchanged, lowest trace. Cite: this is the pattern behind tt-qwythos's decode profiler.
5. **Goal 2 — "I want to fix a bug before upstream does"** — `wrap` to replace behavior, guarded by `version_at_most(ttnn.__version__, "X.Y.Z")` so it retires itself; log on apply AND when detected redundant.
6. **Goal 3 — "I want to change a default"** — `set_default` for a constant, OR env-before-import. Use real cases: never use `ttnn.DispatchCoreConfig(..., ttnn.DispatchCoreAxis.ROW)` (crashes on Blackhole — prefer auto-detect); bumping a `trace_region_size` to avoid P300X2 OOM (as tt-local-generator does for FLUX).
7. **Goal 4 — "I want to add something new"** — register a model / composed op / extend a request model. Show (a) inject a new module into the namespace (`setattr(ttnn, "my_op", ...)` / `sys.modules`), and (b) **file-overlay** onto an installed package/image without editing source (bind-mount pattern, as tt-local-generator does over the tt-inference-server image). Then **"when not to patch"**: if you're purely *adding* capability, a thin external wrapper library is lower-trace and more upgrade-safe than a patch — cite Martin Chang's `ttPseudoRowMajor` (non-invasive additive lib) and his upstream-first ggml backend.
8. **Goal 5 — Escape hatch (LAST RESORT)** — additive, tracked source diff against a real `~/tt-metal` checkout (as tt-qwythos keeps in `tt_metal_patches/`). Front-load: this needs a cloned/built tree — link the `build-tt-metal` lesson — and is the heaviest, least QB2-friendly option; use only when a runtime wrap can't reach (compiled C++/kernels, or edits so invasive a tracked diff is safer).
9. **Make patches ninja & durable (cross-cutting)** — one `patches.py` imported first; each patch documents *why safe / when to remove*; fail-loud via the harness's missing-attr `PatchError`; **AI-agent recipe** with `verify(...)`: assert target exists → `python3 -m py_compile` the patch module → run a smoke op; branch on the boolean.
10. **Recap table** — goal → technique → source repo → upgrade-safety note.

- [ ] **Step 2: Verify front matter parses and required fields exist**

Run:
```bash
node -e "const y=require('js-yaml');const fs=require('fs');const m=fs.readFileSync('content/lessons/monkeypatch-ttnn.md','utf8').match(/^---\n([\s\S]*?)\n---/);const d=y.load(m[1]);const req=['id','title','description','category','tags','supportedHardware','status','validatedOn','estimatedMinutes'];const miss=req.filter(k=>!(k in d));console.log(miss.length?'MISSING '+miss:'FRONTMATTER_OK id='+d.id)"
```
Expected: `FRONTMATTER_OK id=monkeypatch-ttnn`

- [ ] **Step 3: Lint the prose (non-blocking review)**

Run: `npm run lint:docs:errors -- content/lessons/monkeypatch-ttnn.md` (or `npx vale content/lessons/monkeypatch-ttnn.md`)
Expected: no error-level alerts (warnings acceptable; fix obvious issues).

- [ ] **Step 4: Commit**

```bash
git add content/lessons/monkeypatch-ttnn.md
git commit -m "docs(lessons): add Monkeypatching TT-NN (Ninja Edition) lesson"
```

---

## Task 4: Wire the lesson into the registry

**Files:**
- Modify: `content/lesson-registry.json`

**Interfaces:**
- Consumes: front matter from Task 3 (`content/lessons/monkeypatch-ttnn.md`).
- Produces: registry entry `monkeypatch-ttnn` with manual fields `order: 16`, `previousLesson: explore-metalium`, `nextLesson: animatediff-video-generation`, `completionEvents: []`, `markdownFile: content/lessons/monkeypatch-ttnn.md`; and updated neighbor navigation.

- [ ] **Step 1: Preview the generator diff**

Run: `npm run generate:lessons`
Expected: dry-run shows a new `ADD: monkeypatch-ttnn` entry (content fields from front matter). No unexpected modifications to other lessons.

- [ ] **Step 2: Apply the generator (adds content fields + backup)**

Run: `npm run generate:lessons -- --execute --force`
Expected: `✅ Successfully updated lesson-registry.json` and a backup written under `.backups/`.

- [ ] **Step 3: Set the manual fields on the new entry**

Edit `content/lesson-registry.json` — on the `monkeypatch-ttnn` object set:
```json
"order": 16,
"previousLesson": "explore-metalium",
"nextLesson": "animatediff-video-generation",
"completionEvents": [],
"markdownFile": "content/lessons/monkeypatch-ttnn.md"
```

- [ ] **Step 4: Update neighbor navigation to insert into the chain**

Edit `content/lesson-registry.json`:
- On the `explore-metalium` entry: change `"nextLesson": "animatediff-video-generation"` → `"nextLesson": "monkeypatch-ttnn"`.
- On the `animatediff-video-generation` entry: set `"previousLesson": "monkeypatch-ttnn"` (add the field if absent).

- [ ] **Step 5: Validate registry ↔ markdown sync**

Run: `npm run validate:lessons`
Expected: exit 0, no drift errors.

- [ ] **Step 6: Commit**

```bash
git add content/lesson-registry.json
git commit -m "chore(lessons): register monkeypatch-ttnn (order 16, after explore-metalium)"
```

---

## Task 5: Version bump, changelog, and full build

**Files:**
- Modify: `package.json` (version)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: all prior tasks (template, lesson, registry entry).

- [ ] **Step 1: Bump the version**

Edit `package.json`: increment the `version` field by one PATCH level from the
current released version (e.g. `X.Y.Z` → `X.Y.(Z+1)`).

- [ ] **Step 2: Add a CHANGELOG entry**

Edit `CHANGELOG.md` — add a new `## [X.Y.Z] - <date>` heading matching the
version you just set (Keep a Changelog format), e.g.:
```markdown
### Added
- New advanced lesson "Monkeypatching TT-NN": upgrade-safe, smallest-trace
  patching of TT-NN / TT-Metalium organized by goal, plus a reusable,
  hardware-free-testable `tt_patches` harness template.
```
(Describe by feature area — no line numbers, per the changelog policy.)

- [ ] **Step 3: Run the full build (runs lesson + command-uri validation)**

Run: `npm run build`
Expected: build succeeds; `dist/content/lessons/monkeypatch-ttnn.md` and `dist/content/templates/monkeypatch/tt_patches.py` are copied.

- [ ] **Step 4: Confirm the lesson shipped to dist**

Run: `ls dist/content/lessons/monkeypatch-ttnn.md dist/content/templates/monkeypatch/tt_patches.py && echo DIST_OK`
Expected: `DIST_OK`

- [ ] **Step 5: Commit**

```bash
git add package.json CHANGELOG.md
git commit -m "chore: bump to X.Y.Z for Monkeypatching TT-NN lesson"
```

---

## Self-Review

**Spec coverage:**
- QB2 "installed package, no tree" framing → Task 3 intro + orientation. ✅
- Two axes (trace, upgrade-safety) → Task 3 intro; enforced by harness (Task 1). ✅
- Import-order rule → Task 3 orientation. ✅
- Goals 1–5 → Task 3 body sections 4–8. ✅
- "wrap don't patch to add" / Martin Chang → Task 3 Goal 4. ✅
- Fail-loud + one-module + AI-agent verify → Task 1 (`PatchError`, `verify`) + Task 3 closing. ✅
- Reusable `tt_patches.py` template → Tasks 1–2. ✅
- Registry + version + changelog wiring → Tasks 4–5. ✅
- Escape hatch links `build-tt-metal` → Task 3 Goal 5. ✅

**Placeholder scan:** All code steps contain complete code; test code is concrete; no TBD/TODO. ✅

**Type consistency:** `wrap`, `set_default`, `unwrap_all`, `applied`, `patched`, `version_at_most`, `verify`, `PatchError` are used identically in Task 1 implementation, Task 1 tests, and Task 2 README. ✅
