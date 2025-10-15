"""Fluid solver backend implementing the PlasmaBackend interface."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np
from scipy.constants import c

from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState
from .plasma_physics import PlasmaPhysicsModel


class FluidBackend(PlasmaBackend):
    """Adapter exposing existing fluid model via generic backend API."""

    def __init__(self) -> None:
        self._model: Optional[PlasmaPhysicsModel] = None
        self._sink = NullDiagnosticsSink()
        self._x: Optional[np.ndarray] = None
        self._last_state: Optional[PlasmaState] = None

    def configure(self, run_config: Mapping[str, object]) -> None:
        density = float(run_config.get("plasma_density", 1e18))
        wavelength = float(run_config.get("laser_wavelength", 800e-9))
        intensity = float(run_config.get("laser_intensity", 1e17))
        grid = np.asarray(run_config.get("grid", np.linspace(0.0, 50e-6, 1000)))

        self._model = PlasmaPhysicsModel(density, wavelength, intensity)
        self._x = grid
        self._last_state = None

    def set_diagnostics_sink(self, sink: Optional[DiagnosticsSink]) -> None:
        self._sink = sink or NullDiagnosticsSink()

    def step(self, dt: float) -> PlasmaState:  # pylint: disable=unused-argument
        assert self._model is not None and self._x is not None

        t = np.array([0.0])

        def E_laser_func(x_pos, _t):
            return np.sin(2 * np.pi * x_pos / self._model.lambda_l)

        response = self._model.simulate_plasma_response(t, self._x, E_laser_func)
        sound_speed = np.full_like(response["density"], 0.1 * c)

        state = PlasmaState(
            density=response["density"],
            velocity=response["velocity"],
            sound_speed=sound_speed,
            electric_field=response["electric_field"],
            grid=self._x,
        )
        self._last_state = state
        self._sink.emit("fluid_state", {
            "density": state.density,
            "velocity": state.velocity,
        })
        return state

    def export_observables(self, requests: Iterable[str]) -> Dict[str, np.ndarray]:
        assert self._last_state is not None
        obs: Dict[str, np.ndarray] = {}
        for key in requests:
            if key == "density":
                obs[key] = self._last_state.density
            elif key == "velocity":
                obs[key] = self._last_state.velocity
            elif key == "sound_speed":
                obs[key] = self._last_state.sound_speed
        return obs

    def shutdown(self) -> None:
        self._model = None
        self._x = None
        self._last_state = None


