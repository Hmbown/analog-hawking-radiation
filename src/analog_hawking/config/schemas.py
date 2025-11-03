"""Pydantic configuration schemas for simulations and analyses.

These schemas provide typed, validated configuration objects that can be
constructed from dictionaries or YAML files.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    from pydantic import BaseModel, Field

    class QuickstartConfig(BaseModel):
        nx: int = Field(1000, description="Number of grid cells")
        x_min: float = Field(0.0, description="Domain start [m]")
        x_max: float = Field(100e-6, description="Domain end [m]")
        v0: float = Field(0.1 * 3e8, description="Velocity scale [m/s]")
        x0: float = Field(50e-6, description="Horizon center [m]")
        L: float = Field(10e-6, description="Shear length scale [m]")
        Te: float = Field(1e6, description="Electron temperature [K]")
        results_dir: str = Field("results/quickstart", description="Output directory")

except Exception:  # pragma: no cover - allow import when pydantic not installed

    @dataclass
    class QuickstartConfig:  # type: ignore
        nx: int = 1000
        x_min: float = 0.0
        x_max: float = 100e-6
        v0: float = 0.1 * 3e8
        x0: float = 50e-6
        L: float = 10e-6
        Te: float = 1e6
        results_dir: str = "results/quickstart"


__all__ = [
    "QuickstartConfig",
]
