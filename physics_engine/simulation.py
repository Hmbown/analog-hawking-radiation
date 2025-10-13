"""Simulation orchestration utilities bridging modules to backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

import numpy as np

from .horizon import HorizonResult, find_horizons_with_uncertainty
from .plasma_models.backend import PlasmaBackend, PlasmaState


@dataclass
class SimulationOutputs:
    state: PlasmaState
    horizons: Optional[HorizonResult]


class SimulationRunner:
    """Coordinates plasma backend stepping and horizon diagnostics."""

    def __init__(self, backend: PlasmaBackend) -> None:
        self._backend = backend

    def run_step(self, dt: float) -> SimulationOutputs:
        state = self._backend.step(dt)
        horizons = self._compute_horizons(state)
        return SimulationOutputs(state=state, horizons=horizons)

    def configure(self, run_config: Mapping[str, object]) -> None:
        self._backend.configure(run_config)

    def export(self, requests: Iterable[str]):
        return self._backend.export_observables(requests)

    def shutdown(self) -> None:
        self._backend.shutdown()

    def _compute_horizons(self, state: PlasmaState) -> Optional[HorizonResult]:
        if state.velocity.ndim != 1:
            return None
        if state.grid is not None:
            x = state.grid
        else:
            x = np.arange(len(state.velocity))
        horizons = find_horizons_with_uncertainty(
            state.velocity,
            state.sound_speed,
        )
        return horizons


