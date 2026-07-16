"""tt_patches — a low-trace monkeypatch harness for TT-NN / TT-Metalium.

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
    upstream is fixed. TT-NN has no ``ttnn.__version__``; read the package version::

        import importlib.metadata
        if version_at_most(importlib.metadata.version("ttnn"), "0.51.0"):
            apply()

    Leading numeric segments are compared, so a ``…rc17.dev6200`` suffix is fine.
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
