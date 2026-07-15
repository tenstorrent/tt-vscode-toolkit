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
