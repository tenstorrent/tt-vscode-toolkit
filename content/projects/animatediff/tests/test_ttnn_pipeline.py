# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Unit tests for ttnn_pipeline.py — run without Blackhole hardware.

ttnn_pipeline.py uses lazy imports (ttnn imported inside functions), so the
module can be imported on any machine. Tests mock sys.modules at call time.
"""

import os
import sys
from unittest.mock import MagicMock, patch
import torch
import pytest


def test_setup_blackhole_sets_env_var(tmp_path, monkeypatch):
    """setup_blackhole() sets TT_METAL_ARCH_NAME=blackhole if not already set."""
    monkeypatch.delenv("TT_METAL_ARCH_NAME", raising=False)
    (tmp_path).mkdir(exist_ok=True)  # stand in for ~/tt-metal

    from animatediff_ttnn import ttnn_pipeline

    with patch.dict(sys.modules, {
        "ttnn": MagicMock(),
        "models.demos.wormhole.stable_diffusion.common": MagicMock(SD_L1_SMALL_SIZE=21056),
    }):
        with patch.object(ttnn_pipeline, "TT_METAL_PATH", tmp_path):
            ttnn_pipeline.setup_blackhole()

    assert os.environ.get("TT_METAL_ARCH_NAME") == "blackhole"


def test_setup_blackhole_preserves_existing_arch_name(monkeypatch):
    """setup_blackhole() does not overwrite TT_METAL_ARCH_NAME if already set."""
    monkeypatch.setenv("TT_METAL_ARCH_NAME", "wormhole_b0")

    from animatediff_ttnn import ttnn_pipeline

    with patch.object(ttnn_pipeline, "_ensure_tt_metal_path"):
        with patch.dict(sys.modules, {
            "ttnn": MagicMock(),
            "models.demos.wormhole.stable_diffusion.common": MagicMock(SD_L1_SMALL_SIZE=21056),
        }):
            ttnn_pipeline.setup_blackhole()

    assert os.environ.get("TT_METAL_ARCH_NAME") == "wormhole_b0"


def test_build_tlist_returns_one_entry_per_timestep():
    """build_tlist() returns a list with one element per scheduler timestep."""
    from animatediff_ttnn import ttnn_pipeline

    mock_scheduler = MagicMock()
    mock_scheduler.timesteps = torch.tensor([900, 700, 500, 300, 100])
    mock_device = MagicMock()
    mock_time_proj = MagicMock()

    # Patch _constant_prop_time_embeddings at module level so build_tlist() sees it
    with patch.object(ttnn_pipeline, "_constant_prop_time_embeddings",
                      return_value=torch.randn(2, 320)):
        with patch.dict(sys.modules, {"ttnn": MagicMock()}):
            result = ttnn_pipeline.build_tlist(mock_scheduler, mock_time_proj, mock_device)

    assert len(result) == 5


def test_ensure_tt_metal_path_raises_when_absent(tmp_path):
    """_ensure_tt_metal_path() raises RuntimeError when ~/tt-metal is missing."""
    from animatediff_ttnn import ttnn_pipeline

    original = ttnn_pipeline.TT_METAL_PATH
    ttnn_pipeline.TT_METAL_PATH = tmp_path / "nonexistent"
    try:
        with pytest.raises(RuntimeError, match="tt-metal not found"):
            ttnn_pipeline._ensure_tt_metal_path()
    finally:
        ttnn_pipeline.TT_METAL_PATH = original
