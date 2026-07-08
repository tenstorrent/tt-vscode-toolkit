# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Shared helper: put TT-Lang's functional simulator on sys.path.

The kernel files in this directory are *lesson reference code* for the
"Build an LLM from Scratch, TT-Native" arc. They run against TT-Lang's
in-process functional simulator (`vendor/tt-lang/python/sim`), which
executes the reader/compute/writer threads on the host with greenlets —
no Tenstorrent device is required.

Resolution order for the simulator package:
  1. the ``TTLANG_PYTHON`` environment variable, if it points at a
     ``python`` dir that contains ``sim/``;
  2. a ``vendor/tt-lang/python`` directory found by walking up from this
     file toward the repository root.

`vendor/` is git-ignored (see the repo CLAUDE.md); clone tt-lang there with
``git clone https://github.com/tenstorrent/tt-lang.git vendor/tt-lang`` if it
is missing. `import ttnn` must also succeed (the sim uses it for tensor
conversion / golden references).
"""

import os
import sys


def add_ttlang_sim_to_path() -> str:
    """Locate ``vendor/tt-lang/python`` and prepend it to ``sys.path``.

    Returns the resolved path. Raises RuntimeError with actionable guidance
    if the simulator cannot be found.
    """
    candidates = []
    env = os.environ.get("TTLANG_PYTHON")
    if env:
        candidates.append(env)

    here = os.path.dirname(os.path.abspath(__file__))
    directory = here
    for _ in range(8):
        candidates.append(os.path.join(directory, "vendor", "tt-lang", "python"))
        directory = os.path.dirname(directory)

    for candidate in candidates:
        if os.path.isdir(os.path.join(candidate, "sim")):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate

    raise RuntimeError(
        "Could not find the TT-Lang functional simulator "
        "(vendor/tt-lang/python/sim).\n"
        "Clone it with:\n"
        "  git clone https://github.com/tenstorrent/tt-lang.git vendor/tt-lang\n"
        "or set TTLANG_PYTHON to the tt-lang 'python' directory."
    )
