# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""
Sequential drop-in for the greenlet C extension — Pyodide / WASM build.

Pyodide's WASM build does not support threading (_start_new_thread is not
implemented), so the thread-based cooperative-switch approach cannot be used.

Instead this shim runs each greenlet's function directly to completion when
switch() is called.  This works because:

  1. Program._run_cooperative adds threads in DM0 → compute → DM1 order
     (see program.py), so the data is always available before it is consumed.
  2. DataflowBuffer block_count is patched to 4096 in pyodide-worker.js
     BOOTSTRAP_PY, so reserve() / wait() never actually need to block.

Because the DFB operations never block, block_current_thread() (which would
call _MainGreenlet.switch()) is never reached.  _MainGreenlet.switch() is
kept as a no-op for safety.
"""

import threading
from typing import Any, Callable, Optional

_tls = threading.local()


def _current() -> "_MainGreenlet | greenlet":
    g = getattr(_tls, "greenlet", None)
    return g if g is not None else _MAIN


def _set_current(g) -> None:
    _tls.greenlet = g


class _MainGreenlet:
    """Pseudo-greenlet representing the scheduler (main / calling context)."""

    dead = False
    parent = None

    def __init__(self):
        pass

    @staticmethod
    def getcurrent():
        return _current()

    def switch(self, *args) -> Any:
        """No-op: with sequential execution and large DFBs this is never called."""
        return None


class greenlet:
    """Sequential coroutine: runs func() to completion on the first switch().

    API surface:
        g = greenlet(func)    — create
        g.switch()            — run func to completion (caller does NOT sleep)
        g.dead                — True once func has returned
        greenlet.getcurrent() — returns the running greenlet
    """

    def __init__(self, run: Optional[Callable] = None, parent=None):
        self._run = run
        self.parent = parent or _current()
        self.dead = False
        self._exc: Optional[BaseException] = None

    @staticmethod
    def getcurrent():
        return _current()

    def switch(self, *args) -> Any:
        """Run self._run() to completion, then return.

        The caller is NOT suspended — this is a direct call, not a coroutine
        switch.  Works correctly as long as the called function never actually
        needs to block (i.e., DFB can_reserve/can_wait always return True).
        """
        if self._run is None or self.dead:
            return None

        prev = _current()
        _set_current(self)
        try:
            self._run()
        except BaseException as e:
            self._exc = e
        finally:
            self.dead = True
            _set_current(prev)

        if self._exc is not None:
            exc, self._exc = self._exc, None
            raise exc

        return None


_MAIN = _MainGreenlet()
_tls.greenlet = _MAIN


def getcurrent():
    """Module-level alias — matches `from greenlet import getcurrent`."""
    return _current()
