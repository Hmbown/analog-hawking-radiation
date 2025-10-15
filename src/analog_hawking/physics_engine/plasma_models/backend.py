"""Backend interface definitions for plasma simulations.

Provides abstract base classes and data containers that decouple
front-end drivers (laser coupling, horizon analysis, optimizers)
from specific numerical implementations (fluid solver, PIC engines).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Protocol

import numpy as np


@dataclass
class GridSpec:
    """Describes discretization for a simulation domain."""

    ndim: int
    shape: Iterable[int]
    spacing: Iterable[float]


@dataclass
class SpeciesConfig:
    """Defines particle species metadata for PIC-style backends."""

    name: str
    charge: float
    mass: float
    macro_weight: float = 1.0


@dataclass
class PlasmaState:
    """Minimal state summary shared with downstream modules."""

    density: np.ndarray
    velocity: np.ndarray
    sound_speed: np.ndarray
    electric_field: Optional[np.ndarray] = None
    magnetic_field: Optional[np.ndarray] = None
    grid: Optional[np.ndarray] = None
    observables: Dict[str, np.ndarray] = field(default_factory=dict)


class DiagnosticsSink(Protocol):
    """Protocol for custom diagnostic hooks."""

    def emit(self, name: str, data: Mapping[str, np.ndarray]) -> None:
        ...


class PlasmaBackend(ABC):
    """Abstract base class for plasma simulation backends."""

    @abstractmethod
    def configure(self, run_config: Mapping[str, object]) -> None:
        """Configure simulation from provided run configuration."""

    @abstractmethod
    def set_diagnostics_sink(self, sink: DiagnosticsSink) -> None:
        """Register sink for emitting diagnostics (optional)."""

    @abstractmethod
    def step(self, dt: float) -> PlasmaState:
        """Advance simulation by ``dt`` and return current state summary."""

    @abstractmethod
    def export_observables(self, requests: Iterable[str]) -> Dict[str, np.ndarray]:
        """Return additional observables such as spectra or particle moments."""

    @abstractmethod
    def shutdown(self) -> None:
        """Tear down internal state and release resources."""

    def attach_fluctuation_injector(self, injector: Any) -> None:  # pragma: no cover - optional extension point
        """Attach a quantum fluctuation injector (optional)."""
        raise NotImplementedError("Quantum fluctuation injection not supported by this backend")


class NullDiagnosticsSink:
    """No-op sink used when clients do not supply diagnostics."""

    def emit(self, name: str, data: Mapping[str, np.ndarray]) -> None:  # pragma: no cover - trivial
        return None


