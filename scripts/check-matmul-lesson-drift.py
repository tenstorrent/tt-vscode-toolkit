#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Check upstream drift for Module 8 matmul lesson source files.

This script tracks a curated set of upstream tt-metal files that Module 8
references directly (Lab 1/2/3 RSTs + executable example sources). It compares
pinned SHAs in this repository to current SHAs on tt-metal/main and reports a
simple drift metric.

Exit codes:
  0 - no drift
  1 - one or more tracked files drifted
  2 - fetch/config error
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

TRACKED_FILES: list[tuple[str, str]] = [
    ("docs/source/tt-metalium/tt_metal/labs/matmul/lab1/lab1.rst", "eab34153b412660e6e20eb58e83b9df92aee7e46"),
    ("docs/source/tt-metalium/tt_metal/labs/matmul/lab2/lab2.rst", "6a051f778cb157da5aec1b8143ff7eefe469d407"),
    ("docs/source/tt-metalium/tt_metal/labs/matmul/lab3/lab3.rst", "ff2bf0f7e63cd1c8a98dcce948f9bf854bc75905"),
    (
        "tt_metal/programming_examples/matmul/matmul_single_core/matmul_single_core.cpp",
        "85ae4cf343a2be09732cb020f68eb5ceb5fbe89b",
    ),
    (
        "tt_metal/programming_examples/matmul/matmul_multi_core/matmul_multi_core.cpp",
        "12a103e5e7807b80614a84a1294962b24ba2da31",
    ),
    (
        "tt_metal/programming_examples/matmul/matmul_multicore_reuse/matmul_multicore_reuse.cpp",
        "8cb6965bb7d5ca63c7f4730f7770d8dc5efe7adc",
    ),
    ("ttnn/examples/lab_multicast/lab_multicast.cpp", "1d15a4e20babbf09dec155e8c6ca44a4dc2231c8"),
]


def github_sha(path: str) -> str | None:
    url = f"https://raw.githubusercontent.com/tenstorrent/tt-metal/main/{path}"
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "tt-vscode-toolkit-drift-check",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            content = resp.read()
            # GitHub uses git blob SHA-1 object IDs for file identity; we intentionally
            # reproduce that exact legacy format for git-compatibility (identity comparison),
            # not for cryptographic integrity/security.
            return hashlib.sha1(b"blob " + str(len(content)).encode("utf-8") + b"\0" + content).hexdigest()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
        print(f"[WARN] Unable to fetch {path} from github source: {exc}", file=sys.stderr)
        return None


def vendor_sha(path: str, vendor_path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(vendor_path), "rev-parse", f"HEAD:{path}"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.strip() if exc.stderr else str(exc)
        print(f"[WARN] Unable to resolve vendor SHA for {path}: {message}", file=sys.stderr)
        return None
    except FileNotFoundError as exc:
        print(f"[WARN] git executable not found while checking {path}: {exc}", file=sys.stderr)
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=["github", "vendor"],
        default="github",
        help="Where to read current SHAs from (default: github)",
    )
    parser.add_argument(
        "--vendor-path",
        default="vendor/tt-metal",
        help="Path to vendor tt-metal checkout when --source=vendor",
    )
    args = parser.parse_args()

    vendor_path = (REPO_ROOT / args.vendor_path).resolve()
    if args.source == "vendor" and not vendor_path.exists():
        print(f"ERROR: vendor checkout missing at {vendor_path}")
        print("Clone first: git clone https://github.com/tenstorrent/tt-metal.git vendor/tt-metal")
        return 2

    changed = 0
    errors = 0

    print("Matmul lesson drift check")
    print(f"Source: {args.source}")

    for path, pinned_sha in TRACKED_FILES:
        current_sha = github_sha(path) if args.source == "github" else vendor_sha(path, vendor_path)

        if not current_sha:
            errors += 1
            print(f"[ERR] {path}")
            continue

        if current_sha == pinned_sha:
            print(f"[OK ] {path}")
        else:
            changed += 1
            print(f"[DRIFT] {path}")
            print(f"        pinned : {pinned_sha}")
            print(f"        current: {current_sha}")

    total = len(TRACKED_FILES)
    drift_pct = (changed / total) * 100 if total else 0.0
    print("-" * 60)
    print(f"Tracked files: {total}")
    print(f"Drifted files: {changed}")
    print(f"Drift metric : {drift_pct:.1f}%")

    if errors:
        print(f"Errors       : {errors}")
        return 2

    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
