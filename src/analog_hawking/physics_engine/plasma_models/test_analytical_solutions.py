"""Legacy import shim for analytical validation routines."""

from __future__ import annotations

from .analytical_validation import (
    ConvergenceTests,
    ValidationTests,
    run_comprehensive_validation,
)

__all__ = [
    "ConvergenceTests",
    "ValidationTests",
    "run_comprehensive_validation",
]
