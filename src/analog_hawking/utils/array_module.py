"""Helpers for selecting NumPy or CuPy at runtime.

The module exposes ``xp`` as the preferred array namespace. By default the
project stays on NumPy, but if CuPy is installed and the environment variable
``ANALOG_HAWKING_USE_CUPY`` is not set to ``0``/``false`` the helpers promote
arrays to CuPy automatically. Consumers can also force CPU execution by setting
``ANALOG_HAWKING_FORCE_CPU=1``.

Utilities:
    - ``xp``: preferred array module (NumPy or CuPy)
    - ``numpy``: the canonical NumPy module (always available)
    - ``cupy``: the CuPy module or ``None`` when unavailable
    - ``using_cupy()``: quick boolean check for GPU execution
    - ``get_array_module(*arrays)``: infer backend from provided arrays
    - ``to_numpy`` / ``ensure_numpy``: convert arrays back to NumPy for downstream APIs
    - ``xp_trapz`` / ``xp_gradient``: thin wrappers that dispatch to the active backend
"""

from __future__ import annotations

import importlib
import os
from typing import Any, Iterable, Optional

import numpy as _np

_FORCE_CPU = os.getenv("ANALOG_HAWKING_FORCE_CPU", "").lower() in {"1", "true", "yes", "on"}
_USE_CUPY = os.getenv("ANALOG_HAWKING_USE_CUPY", "1").lower() not in {"0", "false", "no", "off"}

cupy = None
if not _FORCE_CPU and _USE_CUPY:
    if importlib.util.find_spec("cupy") is not None:  # type: ignore[attr-defined]
        _cupy = importlib.import_module("cupy")
        try:
            # Trigger a tiny kernel launch to make sure NVRTC/driver pieces exist.
            _ = _cupy.arange(1, dtype=_cupy.float32)
            cupy = _cupy  # type: ignore[assignment]
        except Exception:  # pragma: no cover - fall back to numpy when GPU unusable
            cupy = None

numpy = _np

if cupy is not None:
    xp = cupy
else:
    xp = numpy


def using_cupy() -> bool:
    """Return True when CuPy is the active backend."""
    return cupy is not None and xp is cupy


def get_array_module(*arrays: Any):
    """Infer array module from arguments, defaulting to the preferred backend."""
    if cupy is None:
        return numpy
    for arr in arrays:
        if isinstance(arr, cupy.ndarray):
            return cupy
    return cupy if xp is cupy else numpy


def to_numpy(array: Any) -> _np.ndarray:
    """Convert an array-like to a NumPy ndarray without copying when possible."""
    if cupy is not None and isinstance(array, cupy.ndarray):
        return cupy.asnumpy(array)
    return numpy.asarray(array)


def ensure_numpy(array: Any) -> _np.ndarray:
    """Guarantee a NumPy ndarray (alias of :func:`to_numpy`)."""
    return to_numpy(array)


def xp_trapezoid(y: Any, x: Optional[Any] = None) -> Any:
    """Backend-aware trapezoidal integration (numpy.trapezoid / cupy.trapezoid)."""
    module = get_array_module(y, x)
    if module is numpy:
        trap_fn = getattr(numpy, "trapezoid", numpy.trapz)
        return trap_fn(ensure_numpy(y), x=ensure_numpy(x) if x is not None else None)
    y_arr = module.asarray(y)
    trap_fn = getattr(module, "trapezoid", None)
    if trap_fn is None:
        trap_fn = getattr(module, "trapz")
    if x is None:
        return trap_fn(y_arr)
    return trap_fn(y_arr, x=module.asarray(x))


def xp_gradient(y: Any, x: Optional[Any] = None) -> Any:
    """Backend-aware gradient that mirrors ``numpy.gradient`` semantics."""
    module = get_array_module(y, x)
    if module is numpy:
        if x is None:
            return numpy.gradient(ensure_numpy(y))
        return numpy.gradient(ensure_numpy(y), ensure_numpy(x))
    y_arr = module.asarray(y)
    if x is None:
        return module.gradient(y_arr)
    return module.gradient(y_arr, module.asarray(x))


def xp_clip(a: Any, a_min: Any = None, a_max: Any = None) -> Any:
    """Backend-aware clipping helper with support for open bounds."""
    module = get_array_module(a, a_min, a_max)
    if module is numpy:
        return numpy.clip(a, a_min, a_max)
    if a_min is None and a_max is None:
        return module.asarray(a)
    a_arr = module.asarray(a)
    if a_min is None:
        return module.minimum(a_arr, module.asarray(a_max))
    if a_max is None:
        return module.maximum(a_arr, module.asarray(a_min))
    return module.clip(a_arr, module.asarray(a_min), module.asarray(a_max))


def xp_abs(a: Any) -> Any:
    """Absolute value respecting the active backend."""
    module = get_array_module(a)
    if module is numpy:
        return numpy.abs(a)
    return module.abs(a)


def as_scalar(value: Any) -> float:
    """Return a Python float from a backend scalar."""
    if cupy is not None and isinstance(value, cupy.ndarray):
        return float(value.item())
    if hasattr(value, "item"):
        try:
            return float(value.item())
        except Exception:  # pragma: no cover - defensive
            pass
    return float(value)


__all__ = [
    "xp",
    "numpy",
    "cupy",
    "using_cupy",
    "get_array_module",
    "to_numpy",
    "ensure_numpy",
    "xp_trapezoid",
    "xp_gradient",
    "xp_clip",
    "xp_abs",
    "as_scalar",
]
