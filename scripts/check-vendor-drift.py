#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Vendor repo drift detector.

Checks whether each vendored reference repository has fallen behind its
upstream remote, and — for Python-heavy repos — reports which public APIs
changed in the commits you haven't yet pulled.

Vendor repos live in vendor/ and are read-only reference checkouts used when
authoring lessons.  When upstream advances, lesson content (commands, model
paths, Python API calls) may need updating.

Supported repos (configured below):
  tt-metal          TT-Metal hardware runtime + TTNN Python API
  tt-vllm           Tenstorrent vLLM fork
  tt-inference-server  Inference server + model_spec.json catalog
  tt-forge-models   TT-Forge model library
  ttsim             Simulator reference
  tt-xla            TT-XLA / JAX integration  (clone if missing)
  tt-forge-onnx     TT-Forge ONNX frontend    (clone if missing)

Usage:
    python scripts/check-vendor-drift.py [--fetch] [--repo NAME]

Options:
    --fetch          Run `git fetch origin` before comparing (needs network)
    --repo NAME      Check only this repo (e.g. --repo tt-metal)
    --show-api       Show detailed Python API diff for affected modules
                     (default: show file-level diff only)

Exit codes:
    0  all repos up-to-date (or behind but only in non-watched paths)
    1  one or more repos are behind upstream in watched paths
    2  configuration or git error
"""

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Vendor repo configuration
# ──────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).parent.parent  # tt-vscode-toolkit/

VENDOR_REPOS: dict[str, dict] = {
    "tt-metal": {
        "path": "vendor/tt-metal",
        "clone_url": "https://github.com/tenstorrent/tt-metal.git",
        "remote_branch": "main",
        "description": "TT-Metal hardware runtime + TTNN Python API",
        # Subdirectories to watch for Python API drift
        "watch_py_dirs": ["ttnn/ttnn/", "models/"],
        # Key files whose content we summarize
        "watch_files": [],
    },
    "tt-vllm": {
        "path": "vendor/tt-vllm",
        "clone_url": "https://github.com/tenstorrent/vllm.git",
        "remote_branch": "main",
        "description": "Tenstorrent vLLM fork",
        "watch_py_dirs": ["vllm/"],
        "watch_files": [],
    },
    "tt-inference-server": {
        "path": "vendor/tt-inference-server",
        "clone_url": "https://github.com/tenstorrent/tt-inference-server.git",
        "remote_branch": "main",
        "description": "Inference server + model catalog",
        "watch_py_dirs": [],
        # model_spec.json is data-driven; we summarize model count changes
        "watch_files": ["model_spec.json", "release_model_spec.json"],
    },
    "tt-forge-models": {
        "path": "vendor/tt-forge-models",
        "clone_url": "https://github.com/tenstorrent/tt-forge-models.git",
        "remote_branch": "main",
        "description": "TT-Forge model library",
        "watch_py_dirs": [],
        "watch_files": [],
    },
    "ttsim": {
        "path": "vendor/ttsim",
        "clone_url": "https://github.com/tenstorrent/ttsim.git",
        "remote_branch": "main",
        "description": "TT Simulator reference",
        "watch_py_dirs": [],
        "watch_files": [],
    },
    # Repos not yet vendored — script reports clone instructions
    "tt-xla": {
        "path": "vendor/tt-xla",
        "clone_url": "https://github.com/tenstorrent/tt-xla.git",
        "remote_branch": "main",
        "description": "TT-XLA / JAX integration",
        "watch_py_dirs": ["tt_xla/"],
        "watch_files": [],
    },
    "tt-forge-onnx": {
        "path": "vendor/tt-forge-onnx",
        "clone_url": "https://github.com/tenstorrent/tt-forge.git",
        "remote_branch": "main",
        "description": "TT-Forge ONNX frontend",
        "watch_py_dirs": [],
        "watch_files": [],
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Git helpers
# ──────────────────────────────────────────────────────────────────────────────

def _git(repo_path: Path, *args: str, check: bool = True) -> Optional[str]:
    """Run a git command in repo_path; return stdout or None on error."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path)] + list(args),
            capture_output=True, text=True, check=check,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def fetch_origin(repo_path: Path) -> bool:
    """Fetch from origin. Returns True on success."""
    result = subprocess.run(
        ["git", "-C", str(repo_path), "fetch", "origin", "--quiet"],
        capture_output=True, text=True,
    )
    return result.returncode == 0


def commits_behind(repo_path: Path, remote_branch: str) -> Optional[int]:
    """Return how many commits local HEAD is behind origin/<remote_branch>."""
    ref = f"origin/{remote_branch}"
    count = _git(repo_path, "rev-list", "--count", f"HEAD..{ref}", check=False)
    if count is None:
        return None
    try:
        return int(count)
    except ValueError:
        return None


def changed_files(repo_path: Path, remote_branch: str, extension: str = ".py") -> list[str]:
    """Return list of files changed between HEAD and origin/<remote_branch>."""
    ref = f"origin/{remote_branch}"
    out = _git(repo_path, "diff", "--name-only", f"HEAD..{ref}", check=False)
    if not out:
        return []
    return [f for f in out.splitlines() if f.endswith(extension)]


def recent_commits_summary(repo_path: Path, remote_branch: str, n: int = 5) -> list[str]:
    """Return one-line summaries of the N most recent upstream-only commits."""
    ref = f"origin/{remote_branch}"
    out = _git(repo_path, "log", "--oneline", f"HEAD..{ref}", f"-{n}", check=False)
    if not out:
        return []
    return out.splitlines()


# ──────────────────────────────────────────────────────────────────────────────
# Python API diff
# ──────────────────────────────────────────────────────────────────────────────

def _public_api(src: str) -> set[str]:
    """Return set of 'kind:name' public symbols in a Python source string."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    syms: set[str] = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                syms.add(f"func:{node.name}")
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith("_"):
                syms.add(f"class:{node.name}")
    return syms


def api_diff_for_file(repo_path: Path, remote_branch: str, rel_path: str) -> dict:
    """
    Return {'added': [...], 'removed': [...]} public API changes for a file
    between local HEAD and origin/<remote_branch>.
    """
    ref = f"origin/{remote_branch}"
    local = _git(repo_path, "show", f"HEAD:{rel_path}", check=False) or ""
    remote = _git(repo_path, "show", f"{ref}:{rel_path}", check=False) or ""
    local_api = _public_api(local)
    remote_api = _public_api(remote)
    return {
        "added": sorted(remote_api - local_api),
        "removed": sorted(local_api - remote_api),
    }


# ──────────────────────────────────────────────────────────────────────────────
# model_spec.json diff
# ──────────────────────────────────────────────────────────────────────────────

def model_spec_diff(repo_path: Path, remote_branch: str, spec_file: str) -> dict:
    """Compare model_spec.json model counts and new/removed model IDs."""
    ref = f"origin/{remote_branch}"

    def parse_models(content: str) -> set[str]:
        try:
            d = json.loads(content)
            return set(d.get("model_specs", {}).keys())
        except (json.JSONDecodeError, AttributeError):
            return set()

    local_src = _git(repo_path, "show", f"HEAD:{spec_file}", check=False) or ""
    remote_src = _git(repo_path, "show", f"{ref}:{spec_file}", check=False) or ""
    local_models = parse_models(local_src)
    remote_models = parse_models(remote_src)
    return {
        "added": sorted(remote_models - local_models),
        "removed": sorted(local_models - remote_models),
        "local_count": len(local_models),
        "remote_count": len(remote_models),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Per-repo check
# ──────────────────────────────────────────────────────────────────────────────

def check_repo(name: str, config: dict, do_fetch: bool, show_api: bool) -> int:
    """
    Check one vendor repo for drift. Returns 0 (clean), 1 (behind), 2 (error).
    """
    repo_path = REPO_ROOT / config["path"]
    remote_branch = config["remote_branch"]
    clone_url = config["clone_url"]

    print(f"\n{'─' * 60}")
    print(f"  {name}  —  {config['description']}")
    print(f"  path: {config['path']}")

    # Repo not cloned yet
    if not repo_path.exists():
        print(f"  [MISSING] Not found. Clone with:")
        print(f"    git clone {clone_url} {config['path']}")
        return 0  # Not an error — just informational

    # Fetch if requested
    if do_fetch:
        ok = fetch_origin(repo_path)
        if not ok:
            print(f"  [WARN] git fetch failed (no network?); using cached refs")

    # Count commits behind
    behind = commits_behind(repo_path, remote_branch)
    if behind is None:
        print(f"  [WARN] Could not compare to origin/{remote_branch} — is the repo fetched?")
        return 2

    if behind == 0:
        print(f"  ✓  Up-to-date with origin/{remote_branch}")
        return 0

    print(f"  ⚠  {behind} commit(s) behind origin/{remote_branch}")

    # Show recent commits summary
    commits = recent_commits_summary(repo_path, remote_branch, n=min(behind, 8))
    if commits:
        print("  Recent upstream commits:")
        for c in commits:
            print(f"    {c}")
        if behind > 8:
            print(f"    ... and {behind - 8} more")

    result = 1

    # Python API diff for watched directories
    if config.get("watch_py_dirs") or show_api:
        py_files = changed_files(repo_path, remote_branch, ".py")
        if py_files:
            watched_dirs = config.get("watch_py_dirs", [])
            watched = [
                f for f in py_files
                if not watched_dirs or any(f.startswith(d) for d in watched_dirs)
            ]
            other = len(py_files) - len(watched)

            if watched:
                print(f"\n  Python files changed in watched paths ({len(watched)}):")
                for f in watched[:15]:
                    print(f"    {f}")
                if len(watched) > 15:
                    print(f"    ... and {len(watched) - 15} more")

                if show_api:
                    print("\n  API changes (watched modules):")
                    any_api_change = False
                    for f in watched[:20]:
                        diff = api_diff_for_file(repo_path, remote_branch, f)
                        if diff["added"] or diff["removed"]:
                            any_api_change = True
                            print(f"\n    {f}:")
                            for sym in diff["added"]:
                                print(f"      + {sym}")
                            for sym in diff["removed"]:
                                print(f"      - {sym}")
                    if not any_api_change:
                        print("    (no public API changes — internal only)")

            if other > 0:
                print(f"  + {other} changed Python file(s) outside watched paths (skipped)")

    # model_spec.json diff for inference-server-style repos
    for spec_file in config.get("watch_files", []):
        spec_changed = _git(
            repo_path, "diff", "--name-only", f"HEAD..origin/{remote_branch}", "--",
            spec_file, check=False
        )
        if not spec_changed:
            continue

        if spec_file.endswith(".json") and "model_spec" in spec_file:
            diff = model_spec_diff(repo_path, remote_branch, spec_file)
            if diff["added"] or diff["removed"] or diff["local_count"] != diff["remote_count"]:
                print(f"\n  {spec_file} changes:")
                print(f"    local: {diff['local_count']} models → upstream: {diff['remote_count']} models")
                for m in diff["added"]:
                    print(f"    + {m}")
                for m in diff["removed"]:
                    print(f"    - {m}")
        else:
            print(f"\n  {spec_file} changed (run `git diff HEAD..origin/{remote_branch} -- {spec_file}` for details)")

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true",
                        help="Run git fetch origin before comparing (needs network)")
    parser.add_argument("--repo", metavar="NAME",
                        help="Check only this repo (e.g. tt-metal)")
    parser.add_argument("--show-api", action="store_true",
                        help="Show detailed Python API changes (slower)")
    args = parser.parse_args()

    if args.repo:
        if args.repo not in VENDOR_REPOS:
            print(f"ERROR: unknown repo '{args.repo}'. Known: {', '.join(VENDOR_REPOS)}")
            return 2
        repos = {args.repo: VENDOR_REPOS[args.repo]}
    else:
        repos = VENDOR_REPOS

    print("Vendor repo drift check")
    print(f"Repos: {', '.join(repos)}")
    if args.fetch:
        print("Fetching from upstream remotes...")

    results = {}
    for name, config in repos.items():
        results[name] = check_repo(name, config, args.fetch, args.show_api)

    # Summary
    print(f"\n{'═' * 60}")
    ok = [n for n, r in results.items() if r == 0]
    behind = [n for n, r in results.items() if r == 1]
    errors = [n for n, r in results.items() if r == 2]

    if ok:
        print(f"✓ Up-to-date ({len(ok)}): {', '.join(ok)}")
    if behind:
        print(f"⚠ Behind upstream ({len(behind)}): {', '.join(behind)}")
        print("  → Update with: git -C vendor/<name> pull origin main")
    if errors:
        print(f"✗ Check errors ({len(errors)}): {', '.join(errors)}")

    if args.fetch:
        pass  # Already fetched inline
    else:
        print("\nTip: run with --fetch to check for new upstream commits (needs network)")
        print("     run with --show-api to see Python API changes per file")

    return 1 if behind else (2 if errors else 0)


if __name__ == "__main__":
    sys.exit(main())
