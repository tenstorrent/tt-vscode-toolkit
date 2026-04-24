# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
numpy-backed torch compatibility shim for ttlang-sim-lite.

Replaces torch with numpy equivalents so the sim runs in Pyodide
(which has numpy but not torch). bfloat16 maps to float32 since
numpy has no bfloat16 dtype.
"""

from __future__ import annotations

import builtins
from typing import Any, List, Optional, Union

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# dtype aliases  (mirror torch.float32, torch.bfloat16, etc.)
# ──────────────────────────────────────────────────────────────────────────────

float32 = np.float32
float16 = np.float16
bfloat16 = np.float32   # numpy has no bfloat16; treat as float32 for sim purposes
int32 = np.int32
int64 = np.int64
uint8 = np.uint8
bool_ = np.bool_
dtype = type(np.float32)  # valid type annotation target


# ──────────────────────────────────────────────────────────────────────────────
# Tensor type alias so `torch.Tensor` works as a type name / pattern match
# ──────────────────────────────────────────────────────────────────────────────

Tensor = np.ndarray


# ──────────────────────────────────────────────────────────────────────────────
# nn.functional namespace
# ──────────────────────────────────────────────────────────────────────────────

class _NNFunctional:
    @staticmethod
    def gelu(x: np.ndarray) -> np.ndarray:
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))

    @staticmethod
    def silu(x: np.ndarray) -> np.ndarray:
        return x * (1.0 / (1.0 + np.exp(-x)))

    @staticmethod
    def softsign(x: np.ndarray) -> np.ndarray:
        return x / (1.0 + np.abs(x))

    @staticmethod
    def hardsigmoid(x: np.ndarray) -> np.ndarray:
        return np.clip(x / 6.0 + 0.5, 0.0, 1.0)

    @staticmethod
    def selu(x: np.ndarray) -> np.ndarray:
        alpha, scale = 1.6732632423543772, 1.0507009873554804
        return scale * np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))

    @staticmethod
    def leaky_relu(x: np.ndarray, negative_slope: float = 0.01) -> np.ndarray:
        return np.where(x >= 0, x, negative_slope * x)

    @staticmethod
    def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x >= 0, x, alpha * (np.exp(x) - 1.0))

    @staticmethod
    def celu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        return np.where(x >= 0, x, alpha * (np.exp(x / alpha) - 1.0))

    @staticmethod
    def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
        return (1.0 / beta) * np.log1p(np.exp(np.clip(beta * x, -500, threshold)))

    @staticmethod
    def hardtanh(x: np.ndarray, min_val: float = -1.0, max_val: float = 1.0) -> np.ndarray:
        return np.clip(x, min_val, max_val)


class _NN:
    functional = _NNFunctional()


nn = _NN()


# ──────────────────────────────────────────────────────────────────────────────
# Tensor creation
# ──────────────────────────────────────────────────────────────────────────────

def rand(*args: Any, dtype: Any = float32, **kwargs: Any) -> np.ndarray:
    shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return np.random.rand(*shape).astype(dtype)


def randn(*args: Any, dtype: Any = float32, **kwargs: Any) -> np.ndarray:
    shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return np.random.randn(*shape).astype(dtype)


def empty(shape: Any, dtype: Any = float32, **kwargs: Any) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def zeros(shape: Any, dtype: Any = float32, **kwargs: Any) -> np.ndarray:
    return np.zeros(shape, dtype=dtype)


def zeros_like(t: np.ndarray, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    return np.zeros_like(t, dtype=dtype)


def ones(*args: Any, dtype: Any = float32, **kwargs: Any) -> np.ndarray:
    shape = args[0] if len(args) == 1 and isinstance(args[0], (list, tuple)) else args
    return np.ones(shape, dtype=dtype)


def full(shape: Any, fill_value: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    return np.full(shape, fill_value, dtype=dtype)


def full_like(t: np.ndarray, fill_value: Any, dtype: Any = None) -> np.ndarray:
    return np.full_like(t, fill_value, dtype=dtype)


def tensor(data: Any, dtype: Any = None, **kwargs: Any) -> np.ndarray:
    return np.array(data, dtype=dtype)


def arange(start: Any, end: Any = None, step: int = 1, dtype: Any = None) -> np.ndarray:
    if end is None:
        start, end = 0, start
    return np.arange(start, end, step, dtype=dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Aggregation
# ──────────────────────────────────────────────────────────────────────────────

def stack(tensors: List[np.ndarray], dim: int = 0, **kwargs: Any) -> np.ndarray:
    return np.stack(tensors, axis=dim)


def cat(tensors: List[np.ndarray], dim: int = 0, **kwargs: Any) -> np.ndarray:
    return np.concatenate(tensors, axis=dim)


def all(x: Any) -> builtins.bool:  # noqa: A001
    return builtins.bool(np.all(x))


def allclose(a: np.ndarray, b: np.ndarray, rtol: float = 1e-5, atol: float = 1e-8,
             **kwargs: Any) -> builtins.bool:
    return builtins.bool(np.allclose(a, b, rtol=rtol, atol=atol))


def equal(a: np.ndarray, b: np.ndarray) -> builtins.bool:
    return builtins.bool(np.array_equal(a, b))


def broadcast_shapes(*shapes: tuple) -> tuple:
    return np.broadcast_shapes(*shapes)


def max(x: np.ndarray, dim: Optional[int] = None, **kwargs: Any) -> Any:  # noqa: A001
    if dim is None:
        return np.max(x)
    return np.max(x, axis=dim)


def sum(x: np.ndarray, dim: Optional[int] = None, **kwargs: Any) -> Any:  # noqa: A001
    if dim is None:
        return np.sum(x)
    return np.sum(x, axis=dim)


# ──────────────────────────────────────────────────────────────────────────────
# Element-wise ops
# ──────────────────────────────────────────────────────────────────────────────

def abs(x: np.ndarray) -> np.ndarray:         return np.abs(x)           # noqa: A001
def neg(x: np.ndarray) -> np.ndarray:         return np.negative(x)
def exp(x: np.ndarray) -> np.ndarray:         return np.exp(x)
def exp2(x: np.ndarray) -> np.ndarray:        return np.exp2(x)
def expm1(x: np.ndarray) -> np.ndarray:       return np.expm1(x)
def log(x: np.ndarray) -> np.ndarray:         return np.log(x)           # noqa: A001
def log1p(x: np.ndarray) -> np.ndarray:       return np.log1p(x)
def sqrt(x: np.ndarray) -> np.ndarray:        return np.sqrt(x)
def square(x: np.ndarray) -> np.ndarray:      return np.square(x)
def rsqrt(x: np.ndarray) -> np.ndarray:       return 1.0 / np.sqrt(x)
def reciprocal(x: np.ndarray) -> np.ndarray:  return 1.0 / x
def tan(x: np.ndarray) -> np.ndarray:         return np.tan(x)
def tanh(x: np.ndarray) -> np.ndarray:        return np.tanh(x)
def atan(x: np.ndarray) -> np.ndarray:        return np.arctan(x)
def atanh(x: np.ndarray) -> np.ndarray:       return np.arctanh(x)
def sin(x: np.ndarray) -> np.ndarray:         return np.sin(x)
def asin(x: np.ndarray) -> np.ndarray:        return np.arcsin(x)
def asinh(x: np.ndarray) -> np.ndarray:       return np.arcsinh(x)
def cos(x: np.ndarray) -> np.ndarray:         return np.cos(x)
def acos(x: np.ndarray) -> np.ndarray:        return np.arccos(x)
def acosh(x: np.ndarray) -> np.ndarray:       return np.arccosh(x)
def relu(x: np.ndarray) -> np.ndarray:        return np.maximum(0.0, x)
def sigmoid(x: np.ndarray) -> np.ndarray:     return 1.0 / (1.0 + np.exp(-x))
def floor(x: np.ndarray) -> np.ndarray:       return np.floor(x)         # noqa: A001
def ceil(x: np.ndarray) -> np.ndarray:        return np.ceil(x)          # noqa: A001
def frac(x: np.ndarray) -> np.ndarray:        return x - np.trunc(x)
def trunc(x: np.ndarray) -> np.ndarray:       return np.trunc(x)         # noqa: A001
def sign(x: np.ndarray) -> np.ndarray:        return np.sign(x)
def signbit(x: np.ndarray) -> np.ndarray:     return np.signbit(x)
def maximum(a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.maximum(a, b)
def minimum(a: np.ndarray, b: np.ndarray) -> np.ndarray: return np.minimum(a, b)

def clamp(x: np.ndarray, min: Any = None, max: Any = None, **kwargs: Any) -> np.ndarray:  # noqa: A001,A002
    return np.clip(x, a_min=min, a_max=max)


def where(condition: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.where(condition, x, y)


def round(x: np.ndarray, decimals: int = 0) -> np.ndarray:  # noqa: A001
    return np.round(x, decimals=decimals)


# ──────────────────────────────────────────────────────────────────────────────
# Matrix ops
# ──────────────────────────────────────────────────────────────────────────────

def matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.matmul(a, b)
