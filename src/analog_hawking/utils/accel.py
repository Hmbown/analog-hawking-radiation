"""Optional acceleration utilities (Numba).

Provides a minimal interface that only activates if Numba is installed.
"""

from __future__ import annotations

try:
    import numba  # type: ignore

    njit = numba.njit  # type: ignore[attr-defined]
    AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency

    def njit(*args, **kwargs):  # type: ignore
        def wrapper(func):
            return func

        return wrapper

    AVAILABLE = False

__all__ = ["njit", "AVAILABLE"]
