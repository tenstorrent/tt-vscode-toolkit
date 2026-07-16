#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Guard against drift between the Monkeypatching TT-NN lesson and the harness.

The lesson `content/lessons/monkeypatch-ttnn.md` embeds the full source of
`content/templates/monkeypatch/tt_patches.py` inside a collapsible <details>
block, for reader transparency. That embedded copy is a snapshot — if the
template changes and the lesson does not (or vice versa), the "read what you're
about to copy" promise silently breaks.

This check extracts the embedded ```python block and compares it, line for line,
to the actual template file.

Exit codes:
  0 - in sync
  1 - drift detected (embedded copy != template)
  2 - could not locate the embedded block / files
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LESSON = REPO_ROOT / "content" / "lessons" / "monkeypatch-ttnn.md"
TEMPLATE = REPO_ROOT / "content" / "templates" / "monkeypatch" / "tt_patches.py"

# Match the ```python fence whose body begins with the tt_patches module
# docstring — anchoring on that content (rather than "first fence after
# <details>") keeps this correct if other <details>/python blocks are added.
# `read_text` already normalizes CRLF to \n, but tolerate a stray \r anyway.
EMBED_RE = re.compile(r'```python\r?\n("""tt_patches.*?)\r?\n```', re.S)


def main() -> int:
    if not LESSON.exists() or not TEMPLATE.exists():
        print(f"❌ missing file(s): lesson={LESSON.exists()} template={TEMPLATE.exists()}")
        return 2

    match = EMBED_RE.search(LESSON.read_text(encoding="utf-8"))
    if not match:
        print('❌ could not find the embedded ```python block starting with the tt_patches docstring in the lesson')
        return 2

    embedded = match.group(1).strip()
    actual = TEMPLATE.read_text(encoding="utf-8").strip()

    if embedded == actual:
        print("✅ monkeypatch lesson embed is in sync with tt_patches.py")
        return 0

    print("❌ DRIFT: the tt_patches.py source embedded in the lesson does not match the template.")
    print("   Re-copy the template's contents into the <details> block in")
    print(f"   {LESSON.relative_to(REPO_ROOT)} (source of truth: {TEMPLATE.relative_to(REPO_ROOT)}).\n")
    diff = difflib.unified_diff(
        actual.splitlines(),
        embedded.splitlines(),
        fromfile="tt_patches.py (template)",
        tofile="lesson <details> embed",
        lineterm="",
    )
    for line in list(diff)[:40]:
        print("   " + line)
    return 1


if __name__ == "__main__":
    sys.exit(main())
