"""Fluid solver backend implementing the PlasmaBackend interface."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState
from .plasma_physics import PlasmaPhysicsModel


class FluidBackend(PlasmaBackend):
    """Adapter exposing existing fluid model via generic backend API."""

    def __init__(self) -> None:
        self._model: Optional[PlasmaPhysicsModel] = None
        self._sink = NullDiagnosticsSink()
        self._x: Optional[np.ndarray] = None
        self._last_state: Optional[PlasmaState] = None
        self._temperature_config: Optional[Mapping[str, object]] = None
        self._magnetic_field: Optional[np.ndarray] = None
        self._velocity_override: Optional[np.ndarray] = None
        self._use_fast_magnetosonic: bool = False

    def configure(self, run_config: Mapping[str, object]) -> None:
        density = float(run_config.get("plasma_density", 1e18))
        wavelength = float(run_config.get("laser_wavelength", 800e-9))
        intensity = float(run_config.get("laser_intensity", 1e17))
        grid = np.asarray(run_config.get("grid", np.linspace(0.0, 50e-6, 1000)))

        self._model = PlasmaPhysicsModel(density, wavelength, intensity)
        self._x = grid
        self._last_state = None
        self._temperature_config = run_config.get("temperature_settings")  # type: ignore[assignment]
        self._magnetic_field = None
        self._velocity_override = None
        self._use_fast_magnetosonic = bool(run_config.get("use_fast_magnetosonic", False))

        if "magnetic_field" in run_config:
            B = run_config["magnetic_field"]
            if callable(B):
                self._magnetic_field = np.asarray([B(x_val) for x_val in grid])
            else:
                self._magnetic_field = np.asarray(B, dtype=float)
                if self._magnetic_field.ndim == 0:
                    self._magnetic_field = np.full_like(grid, float(self._magnetic_field))

        if "velocity_profile" in run_config:
            v_profile = run_config["velocity_profile"]
            if callable(v_profile):
                self._velocity_override = np.asarray([v_profile(x_val) for x_val in grid])
            else:
                self._velocity_override = np.asarray(v_profile, dtype=float)

    def set_diagnostics_sink(self, sink: Optional[DiagnosticsSink]) -> None:
        self._sink = sink or NullDiagnosticsSink()

    def step(self, dt: float) -> PlasmaState:  # pylint: disable=unused-argument
        assert self._model is not None and self._x is not None

        t = np.array([0.0])

        def E_laser_func(x_pos, _t):
            return np.sin(2 * np.pi * x_pos / self._model.lambda_l)

        response = self._model.simulate_plasma_response(t, self._x, E_laser_func)

        if self._velocity_override is not None:
            velocity = self._velocity_override
        else:
            velocity = response["velocity"]

        electron_temperature = self._compute_temperature_profile(response["density"])
        if self._use_fast_magnetosonic or self._magnetic_field is not None:
            magnetosonic_speed = self._model.fast_magnetosonic_speed(
                electron_temperature,
                density=response["density"],
                B=self._magnetic_field,
            )
            sound_speed = magnetosonic_speed
        else:
            sound_speed = self._model.sound_speed(electron_temperature)

        magnetosonic_speed = None
        if self._magnetic_field is not None:
            magnetosonic_speed = self._model.fast_magnetosonic_speed(
                electron_temperature,
                density=response["density"],
                B=self._magnetic_field,
            )

        state = PlasmaState(
            density=response["density"],
            velocity=velocity,
            sound_speed=sound_speed,
            electric_field=response["electric_field"],
            grid=self._x,
            temperature=electron_temperature,
            magnetosonic_speed=magnetosonic_speed,
        )
        self._last_state = state
        self._sink.emit("fluid_state", {
            "density": state.density,
            "velocity": state.velocity,
            "temperature": state.temperature if state.temperature is not None else np.array([]),
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
            elif key == "temperature" and self._last_state.temperature is not None:
                obs[key] = self._last_state.temperature
            elif key == "magnetosonic_speed" and self._last_state.magnetosonic_speed is not None:
                obs[key] = self._last_state.magnetosonic_speed
        return obs

    def _compute_temperature_profile(self, density: np.ndarray) -> np.ndarray:
        if self._temperature_config is None:
            return np.full_like(density, 1e4, dtype=float)

        if "constant" in self._temperature_config:
            value = float(self._temperature_config["constant"])  # type: ignore[index]
            return np.full_like(density, value, dtype=float)

        if "profile" in self._temperature_config:
            profile = self._temperature_config["profile"]
            if callable(profile):
                return np.asarray([profile(x_val) for x_val in self._x])
            arr = np.asarray(profile, dtype=float)
            if arr.ndim == 0:
                return np.full_like(density, float(arr))
            return arr

        if "file" in self._temperature_config:
            path = self._temperature_config["file"]
            data = np.load(path)  # type: ignore[arg-type]
            if isinstance(data, np.ndarray):
                return np.asarray(data, dtype=float)
            if "temperature" in data:
                return np.asarray(data["temperature"], dtype=float)

        return np.full_like(density, 1e4, dtype=float)

    def shutdown(self) -> None:
        self._model = None
        self._x = None
        self._last_state = None


