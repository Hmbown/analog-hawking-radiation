"""WarpX backend skeleton providing hooks for PIC integration."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState
from .backend import SpeciesConfig

from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState

try:
    import warpx  # type: ignore
    from pywarpx import libwarpx  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    warpx = None
    libwarpx = None


class WarpXBackend(PlasmaBackend):
    """Provides structure for WarpX-driven PIC simulations."""

    def __init__(self) -> None:
        self._sink: DiagnosticsSink = NullDiagnosticsSink()
        self._configured = False
        self._species: Iterable[SpeciesConfig] = []
        self._grid: Optional[np.ndarray] = None
        self._fluctuation_injector = None

    def configure(self, run_config: Mapping[str, object]) -> None:
        if warpx is None:
            raise RuntimeError("WarpX backend requested but warpx module not available")
        self._species = run_config.get("species", [])  # type: ignore[assignment]
        self._grid = np.asarray(run_config.get("grid")) if run_config.get("grid") is not None else None
        self._build_geometry(run_config)
        self._build_species(run_config)
        self._build_lasers(run_config)
        self._build_diagnostics(run_config)
        self._configured = True

    def set_diagnostics_sink(self, sink: Optional[DiagnosticsSink]) -> None:
        self._sink = sink or NullDiagnosticsSink()

    def step(self, dt: float) -> PlasmaState:
        if not self._configured:
            raise RuntimeError("WarpX backend used before configure()")
        libwarpx.warpx.step(1)

        if self._fluctuation_injector is not None:
            self._fluctuation_injector.inject()

        density = self._extract_field("rho")
        velocity = self._compute_bulk_velocity()
        sound_speed = self._estimate_sound_speed(density)

        state = PlasmaState(
            density=density,
            velocity=velocity,
            sound_speed=sound_speed,
            electric_field=self._extract_field("Ex"),
            magnetic_field=self._extract_field("Bx"),
            grid=self._grid,
        )
        self._sink.emit("warpx_state", {
            "density": state.density,
            "velocity": state.velocity,
        })
        return state

    def export_observables(self, requests: Iterable[str]) -> Dict[str, np.ndarray]:
        observables: Dict[str, np.ndarray] = {}
        for name in requests:
            observables[name] = self._extract_field(name)
        return observables

    def shutdown(self) -> None:
        if warpx is not None:
            libwarpx.warpx.finalize()
        self._configured = False

    def _extract_field(self, name: str) -> np.ndarray:
        if warpx is None:
            raise RuntimeError("WarpX backend unavailable")
        # Placeholder: real implementation would pull data via warpx.fields or openPMD interface.
        return np.array([])

    def _compute_bulk_velocity(self) -> np.ndarray:
        # Placeholder: compute fluid moment from particle data.
        return np.array([])

    def _estimate_sound_speed(self, density: np.ndarray) -> np.ndarray:
        if density.size == 0:
            return np.array([])
        # Placeholder: use local plasma parameters to infer sound speed.
        return np.full_like(density, 0.1)

    def _build_geometry(self, run_config: Mapping[str, object]) -> None:
        # Placeholder: configure WarpX geometry from run_config
        return None

    def _build_species(self, run_config: Mapping[str, object]) -> None:
        # Placeholder: register particle species with WarpX
        _ = run_config

    def _build_lasers(self, run_config: Mapping[str, object]) -> None:
        # Placeholder: configure laser drivers
        _ = run_config

    def _build_diagnostics(self, run_config: Mapping[str, object]) -> None:
        # Placeholder: configure diagnostics (openPMD, field probes)
        _ = run_config

    def attach_fluctuation_injector(self, injector) -> None:
        self._fluctuation_injector = injector


