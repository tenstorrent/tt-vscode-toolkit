#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Drift detection: ttlang-sim-lite vs. upstream ~/code/tt-lang/python/sim/

Compares the public API (functions, classes) of every module shared between
the vendored sim-lite fork and the upstream tt-lang simulator. Run this after
pulling upstream to see what changed before manually porting the diff.

Usage:
    python scripts/check-sim-lite-drift.py [--upstream PATH]

Exit code:
    0 — no drift detected
    1 — drift found (new upstream symbols, signature changes, or new module files)

Intentional sim-lite additions (not reported as drift):
    torch_compat.py     — torch→numpy shim (sim-lite only)
    greenlet_shim.py    — greenlet fallback for environments without C extension
    test_sim_lite.py    — test harness (sim-lite only)
    kernels/            — example kernels (sim-lite addition)
"""

import ast
import os
import sys
import argparse
from pathlib import Path
from typing import NamedTuple

# ──────────────────────────────────────────────────────────────────────────────
# Files that exist only in sim-lite by design — not reported as drift
# ──────────────────────────────────────────────────────────────────────────────
SIM_LITE_ONLY_FILES = {
    "torch_compat.py",
    "greenlet_shim.py",
    "test_sim_lite.py",
}

# Public symbols that exist only in sim-lite (intentional extensions, not drift)
# Format: {filename: {symbol_name}}
SIM_LITE_ONLY_SYMBOLS: dict[str, set[str]] = {
    "diagnostics.py": {"format", "format_error"},
}


class Symbol(NamedTuple):
    kind: str           # 'func' or 'class'
    name: str
    args: tuple[str, ...]   # positional arg names (excludes 'self')
    lineno: int


def extract_public_symbols(filepath: str) -> list[Symbol]:
    """Return all public-level functions and classes in a Python file."""
    try:
        with open(filepath, encoding="utf-8") as f:
            src = f.read()
        tree = ast.parse(src, filename=filepath)
    except SyntaxError as e:
        print(f"  [WARN] Syntax error parsing {filepath}: {e}", file=sys.stderr)
        return []
    except OSError as e:
        print(f"  [WARN] Cannot read {filepath}: {e}", file=sys.stderr)
        return []

    symbols: list[Symbol] = []
    # Only top-level definitions (direct children of Module)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                args = tuple(
                    a.arg for a in node.args.args if a.arg != "self"
                )
                symbols.append(Symbol("func", node.name, args, node.lineno))
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                symbols.append(Symbol("class", node.name, (), node.lineno))
    return symbols


def compare_file(fname: str, upstream_dir: Path, sim_lite_dir: Path) -> list[str]:
    """Return list of human-readable drift lines for a single file. Empty = clean."""
    up_path = upstream_dir / fname
    sl_path = sim_lite_dir / fname

    up_syms = {s.name: s for s in extract_public_symbols(str(up_path))}
    sl_syms = {s.name: s for s in extract_public_symbols(str(sl_path))}

    # Symbols intentionally only in sim-lite for this file
    sl_only_ok = SIM_LITE_ONLY_SYMBOLS.get(fname, set())

    issues: list[str] = []

    # New in upstream but missing from sim-lite → must be ported
    for name, sym in sorted(up_syms.items()):
        if name not in sl_syms:
            issues.append(
                f"  + MISSING in sim-lite: {sym.kind} {name}"
                f"({', '.join(sym.args)})  [upstream line {sym.lineno}]"
            )

    # Signature changed in upstream
    for name, up_sym in sorted(up_syms.items()):
        if name in sl_syms:
            sl_sym = sl_syms[name]
            if up_sym.kind == "func" and up_sym.args != sl_sym.args:
                issues.append(
                    f"  ~ SIGNATURE CHANGED: {name}\n"
                    f"      upstream : ({', '.join(up_sym.args)})\n"
                    f"      sim-lite : ({', '.join(sl_sym.args)})"
                )

    # In sim-lite but removed from upstream (may indicate upstream refactor)
    for name, sl_sym in sorted(sl_syms.items()):
        if name not in up_syms and name not in sl_only_ok:
            issues.append(
                f"  - REMOVED in upstream: {sl_sym.kind} {name}"
                f"  [sim-lite line {sl_sym.lineno}]"
            )

    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream",
        default=os.path.expanduser("~/code/tt-lang/python/sim"),
        help="Path to upstream ~/code/tt-lang/python/sim (default: %(default)s)",
    )
    args = parser.parse_args()

    upstream_dir = Path(args.upstream)
    sim_lite_dir = Path(__file__).parent.parent / "content" / "web" / "ttlang-sim-lite"

    if not upstream_dir.exists():
        print(f"ERROR: upstream directory not found: {upstream_dir}", file=sys.stderr)
        print("  Clone it first: git clone https://github.com/tenstorrent/tt-lang vendor/tt-lang", file=sys.stderr)
        return 1

    if not sim_lite_dir.exists():
        print(f"ERROR: sim-lite directory not found: {sim_lite_dir}", file=sys.stderr)
        return 1

    upstream_files = {
        f for f in os.listdir(upstream_dir)
        if f.endswith(".py") and f not in ("__pycache__",)
    }
    sim_lite_files = {
        f for f in os.listdir(sim_lite_dir)
        if f.endswith(".py") and f not in ("__pycache__",)
    }

    # New module files in upstream that sim-lite doesn't have
    new_upstream_modules = upstream_files - sim_lite_files
    # sim-lite files that don't exist upstream (expected additions)
    sim_lite_additions = sim_lite_files - upstream_files - SIM_LITE_ONLY_FILES

    total_issues = 0
    any_drift = False

    # ── New module files ──────────────────────────────────────────────────────
    if new_upstream_modules:
        any_drift = True
        total_issues += len(new_upstream_modules)
        print("NEW MODULE FILES in upstream (not in sim-lite):")
        for f in sorted(new_upstream_modules):
            print(f"  + {f}")
        print()

    if sim_lite_additions:
        print("SIM-LITE ONLY module files (unexpected — review if intentional):")
        for f in sorted(sim_lite_additions):
            print(f"  ? {f}")
        print()

    # ── Per-file API comparison ───────────────────────────────────────────────
    common_files = upstream_files & sim_lite_files
    file_issues: dict[str, list[str]] = {}

    for fname in sorted(common_files):
        issues = compare_file(fname, upstream_dir, sim_lite_dir)
        if issues:
            file_issues[fname] = issues

    if file_issues:
        any_drift = True
        print("API DRIFT (upstream changed, sim-lite not updated):")
        for fname, issues in sorted(file_issues.items()):
            print(f"\n  {fname}")
            for issue in issues:
                print(issue)
            total_issues += len(issues)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("─" * 60)
    if any_drift:
        print(f"DRIFT DETECTED: {total_issues} issue(s) found.")
        print("  Review the upstream changes and port relevant ones to sim-lite.")
        print("  Intentional sim-lite-only files (ignored):", ", ".join(sorted(SIM_LITE_ONLY_FILES)))
        return 1
    else:
        checked = len(common_files)
        print(f"No drift detected. Checked {checked} shared module(s).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
