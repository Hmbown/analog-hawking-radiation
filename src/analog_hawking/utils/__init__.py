"""Utility helpers for array backends and numerical acceleration."""

from .array_module import (
    xp,
    numpy,
    cupy,
    using_cupy,
    get_array_module,
    to_numpy,
    ensure_numpy,
)

__all__ = [
    "xp",
    "numpy",
    "cupy",
    "using_cupy",
    "get_array_module",
    "to_numpy",
    "ensure_numpy",
]
