"""Simple plugin discovery utilities using Python entry points.

Defines two groups by convention:
  - analog_hawking.backends
  - analog_hawking.analysis
"""

from __future__ import annotations

from importlib.metadata import entry_points
from typing import Dict


def discover(group: str) -> Dict[str, object]:
    eps = entry_points().get(group, [])  # type: ignore[attr-defined]
    found = {}
    for ep in eps:
        try:
            found[ep.name] = ep.load()
        except Exception:
            # Skip broken entry points gracefully
            continue
    return found


__all__ = ["discover"]

