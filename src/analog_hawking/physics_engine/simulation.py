"""Simulation orchestration utilities bridging modules to backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

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
        self._result_dir: Optional[Path] = None
        self._write_sidecar: bool = False

    def run_step(self, dt: float) -> SimulationOutputs:
        state = self._backend.step(dt)
        horizons = self._compute_horizons(state)
        return SimulationOutputs(state=state, horizons=horizons)

    def configure(self, run_config: Mapping[str, object]) -> None:
        self._backend.configure(run_config)
        results_path = run_config.get("results_dir")
        if results_path is not None:
            self._result_dir = Path(results_path)
            self._result_dir.mkdir(parents=True, exist_ok=True)
        self._write_sidecar = bool(run_config.get("write_sidecar", False))

    def export(self, requests: Iterable[str]):
        return self._backend.export_observables(requests)

    def shutdown(self) -> None:
        self._backend.shutdown()

    def _compute_horizons(self, state: PlasmaState) -> Optional[HorizonResult]:
        """Find horizons in the current plasma state."""
        if state.grid is None or state.velocity is None or state.sound_speed is None:
            return None
        x = state.grid
        horizons = find_horizons_with_uncertainty(
            x,
            state.velocity,
            state.sound_speed,
        )
        if self._write_sidecar and horizons is not None and horizons.positions.size:
            self._write_horizon_sidecar(x, state, horizons)
        return horizons

    def _write_horizon_sidecar(self, x: np.ndarray, state: PlasmaState, horizons: HorizonResult) -> None:
        if self._result_dir is None:
            return
        sidecar = {
            "grid": x.tolist(),
            "velocity": state.velocity.tolist(),
            "sound_speed": state.sound_speed.tolist(),
            "density": state.density.tolist(),
            "horizon_positions": horizons.positions.tolist(),
            "kappa": horizons.kappa.tolist(),
            "kappa_err": horizons.kappa_err.tolist(),
            "dvdx": horizons.dvdx.tolist(),
            "dcsdx": horizons.dcsdx.tolist(),
        }
        meta: Dict[str, object] = {}
        if state.observables is not None:
            meta.update({k: v.tolist() for k, v in state.observables.items()})
        sidecar["metadata"] = meta
        step_index = meta.get("step_index", 0) if isinstance(meta.get("step_index", 0), int) else 0
        filename = self._result_dir / f"horizon_step_{step_index:05d}.json"
        import json

        filename.write_text(json.dumps(sidecar, indent=2))

