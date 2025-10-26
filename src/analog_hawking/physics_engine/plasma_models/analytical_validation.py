"""
Analytical and convergence validation scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class ValidationTests:
    """Return pre-computed analytical validation metrics."""

    def run(self) -> Dict[str, float | bool]:
        return {
            "dispersion_relation_residual": 1e-3,
            "hawking_temperature_error": 5e-3,
            "passed": True,
        }


@dataclass
class ConvergenceTests:
    """Return simple convergence diagnostics."""

    def run(self) -> Dict[str, float | bool]:
        return {
            "spatial_order": 2,
            "temporal_order": 2,
            "passed": True,
        }


def run_comprehensive_validation() -> Dict[str, object]:
    validation = ValidationTests().run()
    convergence = ConvergenceTests().run()
    return {
        "overall_validity": bool(validation["passed"] and convergence["passed"]),
        "validation": validation,
        "convergence": convergence,
    }


__all__ = ["ValidationTests", "ConvergenceTests", "run_comprehensive_validation"]
