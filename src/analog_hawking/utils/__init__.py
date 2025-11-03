"""Utility helpers for array backends and numerical acceleration."""

from .array_module import (
    cupy,
    ensure_numpy,
    get_array_module,
    numpy,
    to_numpy,
    using_cupy,
    xp,
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
