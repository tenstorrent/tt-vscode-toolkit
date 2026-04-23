# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
numpy-backed tensor utilities for ttlang-sim-lite.

Drop-in replacement for the original torch_utils.py — same API, numpy backend.
"""

from typing import List, Union

import numpy as np

from .typedefs import Count, Shape


def randn(*shape: int) -> np.ndarray:
    """Create an array with random normal values."""
    return np.random.randn(*shape).astype(np.float32)


def zeros(*shape: int) -> np.ndarray:
    """Create an array filled with zeros."""
    return np.zeros(shape, dtype=np.float32)


def ones(*shape: int) -> np.ndarray:
    """Create an array filled with ones."""
    return np.ones(shape, dtype=np.float32)


def full(shape: Shape, fill_value: Union[int, float]) -> np.ndarray:
    """Create an array filled with a specific value."""
    return np.full(shape, fill_value, dtype=np.float32)


def all_true(condition: np.ndarray) -> bool:
    """Check if all elements are True."""
    return bool(np.all(condition))


def allclose(
    a: np.ndarray, b: np.ndarray, rtol: float = 1e-05, atol: float = 1e-08
) -> bool:
    """Check if arrays are element-wise close."""
    return bool(np.allclose(a, b, rtol=rtol, atol=atol))


def equal(a: np.ndarray, b: np.ndarray) -> bool:
    """Check if arrays are exactly equal."""
    return bool(np.array_equal(a, b))


def cat(tensors: List[np.ndarray], dim: Count = 0) -> np.ndarray:
    """Concatenate arrays along an axis."""
    return np.concatenate(tensors, axis=dim)


def stack(tensors: List[np.ndarray], dim: Count = 0) -> np.ndarray:
    """Stack arrays along a new axis."""
    return np.stack(tensors, axis=dim)


def is_tiled(tensor: np.ndarray, tile_shape: Shape) -> bool:
    """Check if all tensor dimensions are evenly divisible by tile dimensions."""
    if len(tensor.shape) != len(tile_shape):
        return False
    return all(dim % tile_dim == 0 for dim, tile_dim in zip(tensor.shape, tile_shape))
