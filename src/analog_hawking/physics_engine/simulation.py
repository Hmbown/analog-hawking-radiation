"""Simulation orchestration utilities bridging modules to backends."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import yaml

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
        self._grid_3d: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None

    def run_step(self, dt: float) -> SimulationOutputs:
        state = self._backend.step(dt)
        horizons = self._compute_horizons(state)
        return SimulationOutputs(state=state, horizons=horizons)

    def configure(self, run_config: Mapping[str, object]) -> None:
        # Handle 3D grid configuration
        if "3d_grid_config" in run_config:
            config_path = run_config["3d_grid_config"]
            with open(config_path, "r") as f:
                grid_config = yaml.safe_load(f)
            dimensions = grid_config.get("dimensions", [100, 50, 50])
            dx = grid_config.get("dx", 0.1e-6)
            dy = grid_config.get("dy", dx)
            dz = grid_config.get("dz", dx)
            x_min = grid_config.get("x_min", -5.0e-6)
            y_min = grid_config.get("y_min", -2.5e-6)
            z_min = grid_config.get("z_min", -2.5e-6)

            nx, ny, nz = dimensions
            x = np.linspace(x_min, x_min + nx * dx, nx)
            y = np.linspace(y_min, y_min + ny * dy, ny)
            z = np.linspace(z_min, z_min + nz * dz, nz)
            self._grid_3d = (x, y, z)
            # For backward compatibility, pass 1D x-grid to backend
            run_config["grid"] = x
        else:
            # Inline defaults for 3D if not specified
            nx, ny, nz = 100, 50, 50
            dx = 0.1e-6
            x = np.linspace(-5.0e-6, 5.0e-6, nx)
            y = np.linspace(-2.5e-6, 2.5e-6, ny)
            z = np.linspace(-2.5e-6, 2.5e-6, nz)
            self._grid_3d = (x, y, z)
            run_config["grid"] = x

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
        # For 3D support, slice along central y,z if multi-D
        if self._grid_3d is not None:
            x, y, z = self._grid_3d
            ny, nz = len(y), len(z)
            # Slice at center
            if state.velocity.ndim > 1:
                v_slice = state.velocity[:, ny // 2, nz // 2]
                cs_slice = state.sound_speed[:, ny // 2, nz // 2]
            else:
                v_slice = state.velocity
                cs_slice = state.sound_speed
            x_slice = x
        else:
            x_slice = state.grid
            v_slice = state.velocity
            cs_slice = state.sound_speed
        horizons = find_horizons_with_uncertainty(
            x_slice,
            v_slice,
            cs_slice,
        )
        if self._write_sidecar and horizons is not None and horizons.positions.size:
            self._write_horizon_sidecar(x_slice, state, horizons)
        return horizons

    def _write_horizon_sidecar(
        self, x: np.ndarray, state: PlasmaState, horizons: HorizonResult
    ) -> None:
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
            "c_H": horizons.c_h.tolist() if getattr(horizons, "c_h", None) is not None else [],
            "d_c2_minus_v2_dx": (
                horizons.d_c2_minus_v2_dx.tolist()
                if getattr(horizons, "d_c2_minus_v2_dx", None) is not None
                else []
            ),
        }
        meta: Dict[str, object] = {}
        if state.observables is not None:
            meta.update({k: v.tolist() for k, v in state.observables.items()})
        sidecar["metadata"] = meta
        step_index = meta.get("step_index", 0) if isinstance(meta.get("step_index", 0), int) else 0
        filename = self._result_dir / f"horizon_step_{step_index:05d}.json"
        import json

        filename.write_text(json.dumps(sidecar, indent=2))
