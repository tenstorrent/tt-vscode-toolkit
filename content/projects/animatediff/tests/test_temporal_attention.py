# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC

"""Tests for cross_frame_attention() — runs without hardware (pure PyTorch)."""

import torch
import pytest

from animatediff_ttnn.temporal_attention import cross_frame_attention


def test_passthrough_when_single_frame():
    """N=1 returns the input unchanged."""
    x = torch.randn(1, 4, 8, 8)
    out = cross_frame_attention(x)
    assert torch.equal(out, x)


def test_output_shape_preserved():
    """Output shape matches input for typical N=8 case."""
    x = torch.randn(8, 4, 8, 8)
    out = cross_frame_attention(x)
    assert out.shape == x.shape


def test_dtype_preserved():
    """Output dtype matches input dtype."""
    x = torch.randn(4, 4, 8, 8).to(torch.float16)
    out = cross_frame_attention(x)
    assert out.dtype == torch.float16


def test_alpha_zero_is_identity():
    """alpha=0 blends nothing — output equals input."""
    x = torch.randn(4, 4, 8, 8)
    out = cross_frame_attention(x, alpha=0.0)
    assert torch.allclose(out, x, atol=1e-6)


def test_alpha_one_is_full_attention():
    """alpha=1 returns pure attention output — different from input (for N>1)."""
    torch.manual_seed(0)
    x = torch.randn(4, 4, 8, 8)
    out = cross_frame_attention(x, alpha=1.0)
    assert not torch.allclose(out, x, atol=1e-3)


def test_reproducible_with_same_input():
    """Same input always produces same output (no randomness in attention)."""
    x = torch.randn(4, 4, 8, 8)
    out1 = cross_frame_attention(x)
    out2 = cross_frame_attention(x)
    assert torch.equal(out1, out2)


def test_attention_softmax_range():
    """Blended output values stay in a plausible range (not exploding)."""
    x = torch.randn(8, 4, 8, 8)
    out = cross_frame_attention(x, alpha=0.35)
    assert out.isfinite().all()
    assert out.abs().max() < 100.0
