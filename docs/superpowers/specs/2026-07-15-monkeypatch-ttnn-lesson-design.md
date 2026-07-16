# Design: "Monkeypatching TT-NN / TT-Metalium — Ninja Edition" lesson

**Date:** 2026-07-15
**Branch:** `how-to-monkeypatch`
**Author:** Taylor Singletary (with Claude Code)
**Status:** Approved for planning

## Problem

Developers — increasingly AI agents — want to bring up models and experiment on
Tenstorrent hardware *without* modifying an existing `tt-metal` tree, and often
without even cloning `tt-metal` at all. This is the normal situation on a
**TT-QuietBox 2 (QB2)**: `ttnn` (TT-NN) is present as an *installed Python
package* with **no `~/tt-metal` source tree**. People still need to change
behavior — add logging, work around a bug, tweak a default, register a new
model — and today they either fork `tt-metal` (heavyweight, loses upstream
upgrades) or edit installed package files in place (invisible, fragile, wiped by
the next `pip`/image upgrade).

We want a lesson that teaches **monkeypatching as a discipline**: change behavior
with the *smallest possible trace* and in a way that is *upgrade-safe* — you keep
receiving upstream improvements and you find out loudly if an upgrade invalidates
your patch, rather than silently reverting or silently no-op'ing.

## Audience

- Developers on QB2 (and any machine where `ttnn` is a pip-installed package).
- **AI agents** doing autonomous model bring-up — explicitly a first-class
  audience. The lesson must give agents a mechanical, verifiable recipe (assert
  the target exists, `py_compile`, run a smoke op) so a patch can be confirmed
  without a human in the loop.

## Guiding principles (the two axes every technique is judged on)

1. **Smallest trace** — how little of the system you disturb, and how easily the
   change is seen, reasoned about, and reverted.
2. **Upgrade-safety** — will a `pip`/image/`tt-metal` upgrade silently break or
   silently revert the patch, and does the patch *fail loud* when its assumptions
   no longer hold?

## Grounding: real techniques already in use across TT projects

These are not invented patterns — each is pulled from a real repo on this
machine. The lesson cites them (by pattern, not by pasting proprietary code).

| Source | Technique | Role in lesson |
|---|---|---|
| `tt-qwythos/benchmarks/decode_profile.py` (`_wrap` / `_unwrap_all`) | Runtime attribute wrap with save → `setattr` → restore; host-side only, math unchanged | Canonical base pattern (Goal 1) |
| `tt-local-generator/patches/` (bind-mount overlay into tt-inference-server image; `trace_region_size` bump) | File-overlay onto an installed package/image without editing source; changing a default | Goals 3 & 4 |
| `tt-animatediff` (deferred `import ttnn`), `tt-qwythos/inference/generate.py` (env set before `import ttnn`) | Env + import-order control — ttnn reads config at import time | Orientation rule (applies throughout) |
| `tt-qwythos/tt_metal_patches/` (additive `git diff` snapshots vs a real checkout, documented "why safe") | Additive source diff against a real tree | Escape hatch (Goal 5, last resort) |
| `tt-zork1/ttlang/*` (ttnn from a TT-Lang pyenv, no `~/tt-metal`, raw device work) | — | Supporting context: confirms the "installed package, no tree" QB2 reality |
| Martin Chang (`marty1885`) `ttPseudoRowMajor` (non-invasive external C++ lib that *extends* ttnn without forking or editing the tree); his upstream-first ggml/llama.cpp Tenstorrent backend | Thin additive wrapper *on top of* ttnn instead of patching it | "When not to patch" nuance (Goal 4 + closing) — the smallest-trace / most upgrade-safe move is sometimes a wrapper, not a patch |

Note: `tt-zork` / `tt-zork-and-more` were not present on this machine;
`tt-zork1` was checked and contributes context but no new patching technique.
Martin Chang's tt-awesome repos (`ttnn-helloworld-cpp`, `ttVecAdd`,
`ttMandelbrot`, `ttPseudoRowMajor`) are C++, not Python monkeypatching, but his
non-invasive/upstreamable philosophy is the north star for the whole lesson.

## Deliverables

1. **Lesson markdown:** `content/lessons/monkeypatch-ttnn.md`
2. **Reusable template:** a copy-paste patch harness under `content/templates/`
   (see "Template" below).
3. **Wiring:** entry in `content/lesson-registry.json` (via generator + manual
   `order`/navigation fields). NOTE: this repo has no `contributes.walkthroughs`
   in `package.json` — lessons are driven entirely by the registry, so no
   `package.json` walkthrough step is needed.
4. **Version bump** in `package.json` (PATCH; content addition) + `CHANGELOG.md`
   entry.

## Lesson front matter (markdown-owned fields)

```yaml
id: monkeypatch-ttnn
title: "Monkeypatching TT-NN — Ninja Edition"
description: >-
  Change TT-NN / TT-Metalium behavior with the smallest possible trace and
  without forking tt-metal — add logging, work around a bug, tweak a default,
  or register a model, all while staying upgrade-safe. Built for QB2, where
  ttnn is an installed package with no source tree.
category: advanced
tags:
  - ttnn
  - metalium
  - patching
  - workflow
supportedHardware: [n150, n300, t3k, p100, p150, p300c, galaxy]
status: draft            # promote to validated after testing on p300c
validatedOn: []          # add p300c once verified
estimatedMinutes: 25
```

(JSON-owned fields — `order`, `previousLesson`, `nextLesson`,
`completionEvents`, `markdownFile` — added in `lesson-registry.json` per the
registry sync workflow in CLAUDE.md.)

## Structure — organized **by goal**

### Orientation (short)

- **Locate what you're patching.** `python -c "import ttnn, os; print(ttnn.__file__)"`.
  Explain the QB2 reality: ttnn is an installed package; there is usually no
  `~/tt-metal`. Editing files under `site-packages` in place is *not* a patch —
  it is invisible and the next upgrade wipes it.
- **The one rule underneath everything: env + import order.** `ttnn` reads
  configuration (visible devices, arch name, mesh, dispatch) at `import` time.
  Any env change or namespace patch must land **before** `import ttnn`. Show the
  deferred-import and "set env then import" patterns from tt-animatediff /
  tt-qwythos. This rule recurs in several goals, so it is established up front.

### Goal 1 — "I want to see what's happening" (observe / profile / log)

Runtime attribute wrap with **save → `setattr` → restore**, host-side, math
unchanged. Anchored on qwythos `decode_profile.py`'s `_wrap`/`_unwrap_all`.
Teach the base pattern:

- Grab the original with `getattr`.
- Wrap with `functools.wraps`, bind the original via default arg to avoid late
  binding.
- Keep a registry of `(obj, attr, original)` and restore in reverse.
- Prefer a context manager / decorator for scoped patches.

Upgrade-safety note: purely additive + reversible; the lowest-trace technique.

### Goal 2 — "I want to fix a bug before upstream does"

Wrap-to-*replace* behavior, **guarded by a version check** so the patch no-ops
once the fix lands upstream (so you don't silently keep overriding a
now-correct implementation, and you don't miss the upgrade). Show:

- `if ttnn.__version__ <= "X": apply()` style guard (or a capability probe when
  no clean version exists).
- A loud log line when the patch applies AND when it detects it's now redundant.

### Goal 3 — "I want to change a default"

Defaults like mesh config, dispatch-core config, trace-region size, dtype. Two
tools: (a) wrap a factory/config function, (b) set env before import. Ties to
the real cases:

- `DispatchCoreAxis.ROW` crashing on Blackhole (from CLAUDE.md) → prefer letting
  ttnn auto-detect; if you must, patch the config constructor.
- `trace_region_size` OOM on P300X2 (tt-local-generator FLUX runner) → bump a
  default via a wrapped runner/config rather than editing source.
- `TT_METAL_ARCH_NAME` / `TT_VISIBLE_DEVICES` via env-before-import.

### Goal 4 — "I want to add something new"

Register a model, add a composed op, extend a request model — where wrapping one
function isn't enough. **File-overlay onto the installed package / image**
without editing source, anchored on `tt-local-generator/patches/` bind-mount
mechanism. Cover:

- Bind-mount / overlay a replacement or *additional* module over the package
  path at container/process startup (no image rebuild, no source edit).
- The "add a module + inject it into the namespace" variant for a pure-Python
  environment (assign into `ttnn.<name>` or `sys.modules`).
- Sync/precedence caveats (which copy actually wins).
- **"When not to patch at all":** if you're purely *adding* capability (not
  altering existing behavior), a thin external wrapper library that sits on top
  of ttnn is often lower-trace and more upgrade-safe than any patch — you own the
  new surface, ttnn stays untouched, and upgrades can't revert you. Cite Martin
  Chang's `ttPseudoRowMajor` (additive external lib) and his upstream-first
  ggml backend as the model. Rule of thumb: **patch to change behavior, wrap to
  add behavior.**

### Goal 5 — Escape hatch: "patching isn't enough" (LAST RESORT)

Additive source diff against a real `~/tt-metal` checkout, kept as a re-appliable
`.diff` and documented "why safe / additive." Anchored on qwythos
`tt_metal_patches/`. **Front-loaded caveat:** this is the heaviest, least
QB2-friendly option — it *requires* cloning/building tt-metal. Link the
`build-tt-metal` lesson. Explain when you've genuinely hit the limit of
patching (e.g. changes to compiled C++ / kernels, or edits so invasive that a
runtime wrap would be more fragile than a tracked diff).

### Closing — making patches ninja & durable (cross-cutting)

- **One patch module, imported first.** All patches live in a single
  `patches.py` (or package) imported before `ttnn` is used; each patch carries a
  docstring saying *why it's safe* and *when to remove it*.
- **Fail loud on upgrade.** `assert hasattr(obj, attr)` (or an explicit raise)
  before wrapping, so an upstream rename raises instead of silently no-op'ing.
- **For AI agents:** a verification recipe — (1) assert the target symbol exists,
  (2) `python -m py_compile` the patch module, (3) run a smoke import + tiny op
  to confirm the patch took and nothing broke. Framed so an agent can gate its
  own work on the result.

## Template deliverable

Ship a reusable patch harness devs drop into their project. Proposed location:
`content/templates/monkeypatch/tt_patches.py` (plus a short `README.md` in that
dir). It provides:

- `PatchRegistry` (or module-level `_WRAPS` list) with `wrap(obj, attr, fn)` /
  `unwrap_all()` implementing save→setattr→restore.
- `wrap` asserts `hasattr(obj, attr)` first and logs on apply (fail-loud +
  visible-trace built in).
- A `@patches` context manager for scoped patching.
- A `require_version(pred)` / capability-probe helper for version-guarded
  patches (Goal 2).
- A `verify()` function embodying the agent recipe (assert-exists + smoke op),
  returning a boolean an agent can branch on.
- Header comment template for documenting *why safe / when to remove*.

The lesson references and walks through this file rather than duplicating it
inline; short illustrative snippets still appear in the markdown for reading
flow.

## Non-goals (YAGNI)

- Not a general Python metaprogramming tutorial — scoped to the TT use cases.
- Not teaching how to build tt-metal (that's the `build-tt-metal` lesson; we link
  it from Goal 5).
- No custom extension UI/commands — this is content + a template only.
- Not pasting proprietary code from qwythos/local-generator/animatediff; patterns
  are re-expressed generically and cited.

## Testing / validation plan

- `npm run validate:lessons` passes (markdown ↔ registry in sync).
- `npm run build` succeeds (validation is wired into build).
- Template `tt_patches.py` passes `python -m py_compile`.
- Manual: launch Extension Development Host (F5), open the walkthrough, confirm
  the new step renders and code blocks/links display correctly.
- Hardware validation on p300c (QB2) deferred; `status: draft` until done, then
  promote to `validated` / add `p300c` to `validatedOn`.

## Open questions

None blocking. Hardware validation on p300c is the one follow-up before the
lesson is promoted from `draft` to `validated`.
