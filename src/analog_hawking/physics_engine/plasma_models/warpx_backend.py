"""WarpX backend skeleton providing hooks for PIC integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .adaptive_sigma import SigmaDiagnostics, apply_sigma_smoothing, estimate_sigma_map
from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState, SpeciesConfig

try:
    import pywarpx  # type: ignore
    from pywarpx import _libwarpx as libwarpx  # type: ignore
    warpx = pywarpx
except ImportError:  # pragma: no cover - optional dependency
    warpx = None
    libwarpx = None


_E_CHARGE = 1.602176634e-19  # Coulomb
_K_BOLTZ = 1.380649e-23  # J/K
_M_PROTON = 1.67262192369e-27  # kg


FieldGetter = Callable[[], np.ndarray]
MomentGetter = Callable[[], np.ndarray]


def _create_getter(cfg: Mapping[str, Any], is_mock: bool = False, mock_size: int = 0) -> FieldGetter:
    """Factory function for creating a getter from a configuration dictionary."""
    getter_type = cfg.get("type")
    if getter_type == "pywarpx":
        if not is_mock:
            if warpx is None:
                raise RuntimeError("WarpX backend requested but warpx module not available")
            field = cfg.get("field")
            if field:
                return lambda: warpx.get_field_numpy(field)

            species = cfg.get("species")
            moment = cfg.get("moment")
            if species and moment:
                return lambda: warpx.get_particle_moment_numpy(species, moment)
        else:
            moment = cfg.get("moment")
            if moment == "bulk_velocity":
                return lambda: np.linspace(0.1, 1.0, mock_size)
            elif moment == "sound_speed":
                return lambda: np.full(mock_size, 0.5)
            else:
                return lambda: np.random.rand(mock_size)

    elif getter_type == "openpmd":
        # Minimal HDF5/openPMD reader for reduced diagnostics
        path = cfg.get("path")
        dataset = cfg.get("dataset")
        if not path or not dataset:
            raise ValueError("openpmd getter requires 'path' and 'dataset'")
        def _read():
            try:
                import h5py  # type: ignore
            except Exception as exc:  # pragma: no cover - optional dependency
                raise RuntimeError("h5py is required for openPMD reading") from exc
            with h5py.File(path, 'r') as f:  # type: ignore[arg-type]
                dset = f[dataset]
                return np.array(dset)
        return _read

    raise ValueError(f"Unknown or underspecified getter config: {cfg}")


class WarpXBackend(PlasmaBackend):
    """Provides structure for WarpX-driven PIC simulations."""

    def __init__(self) -> None:
        self._sink: DiagnosticsSink = NullDiagnosticsSink()
        self._configured = False
        self._species: Iterable[SpeciesConfig] = []
        self._grid: Optional[np.ndarray] = None
        self._fluctuation_injector = None
        self._field_getters: Dict[str, FieldGetter] = {}
        self._moment_getters: Dict[str, Dict[str, MomentGetter]] = {}
        self._electron_species: Optional[str] = None
        self._ion_species: Optional[str] = None
        self._default_sigma: Optional[float] = None
        self._sigma_map: Optional[np.ndarray] = None
        self._gamma_e: float = 1.0
        self._gamma_i: float = 1.0
        self._ion_temperature_fraction: float = 0.01
        self._last_observables: Dict[str, np.ndarray] = {}
        self._raw_observables: Dict[str, np.ndarray] = {}
        self._sigma_diagnostics: Optional[SigmaDiagnostics] = None
        self._adaptive_sigma: bool = False

    def configure(self, run_config: Mapping[str, object]) -> None:
        is_mock = run_config.get("mock", False)
        if not is_mock and warpx is None:
            raise RuntimeError("WarpX backend requested but warpx module not available")
        
        species_cfgs = run_config.get("species", [])
        self._species = [SpeciesConfig(**s) for s in species_cfgs] if species_cfgs else []

        self._grid = np.asarray(run_config.get("grid")) if run_config.get("grid") is not None else None
        mock_size = len(self._grid) if self._grid is not None else 0
        self._field_getters = {
            name: _create_getter(getter_cfg, is_mock=is_mock, mock_size=mock_size)
            for name, getter_cfg in run_config.get("field_getters", {}).items()  # type: ignore[dict-item]
        }
        self._moment_getters = {
            species: {
                moment: _create_getter(getter_cfg, is_mock=is_mock, mock_size=mock_size)
                for moment, getter_cfg in moments.items()
            }
            for species, moments in run_config.get("moment_getters", {}).items()  # type: ignore[dict-item]
        }
        self._electron_species = run_config.get("electron_species")  # type: ignore[assignment]
        self._ion_species = run_config.get("ion_species")  # type: ignore[assignment]
        sigma_cells = run_config.get("sigma_cells")
        self._default_sigma = run_config.get("default_sigma")
        self._sigma_map = np.asarray(run_config.get("sigma_map")) if run_config.get("sigma_map") is not None else None
        self._fluctuation_injector = run_config.get("fluctuation_injector")
        smoothing_cfg = run_config.get("sigma_smoothing", {})
        self._gamma_e = float(run_config.get("gamma_e", smoothing_cfg.get("gamma_e", 1.0)))
        self._gamma_i = float(run_config.get("gamma_i", smoothing_cfg.get("gamma_i", 1.0)))
        self._ion_temperature_fraction = float(
            run_config.get("ion_temperature_fraction", smoothing_cfg.get("ion_temperature_fraction", 0.01))
        )
        self._adaptive_sigma = bool(run_config.get("adaptive_sigma", smoothing_cfg.get("adaptive", False)))
        self._build_geometry(run_config)
        self._build_species(run_config)
        self._build_lasers(run_config)
        self._build_diagnostics(run_config)
        self._configured = True

    def set_diagnostics_sink(self, sink: Optional[DiagnosticsSink]) -> None:
        self._sink = sink if sink is not None else NullDiagnosticsSink()

    def step(self, dt: float) -> PlasmaState:
        if not self._configured:
            raise RuntimeError("WarpX backend used before configure()")

        is_mock = self._grid is not None and len(self._grid) > 0 and self._moment_getters
        if is_mock:
            density = self._call_moment_getter(self._electron_species or "electrons", "density")
            velocity = self._call_moment_getter(self._electron_species or "electrons", "bulk_velocity")
            sound_speed = self._call_moment_getter(self._electron_species or "electrons", "sound_speed")

            # Also honor configured field/moment getters to populate observables
            observables = {}
            for name, getter in self._field_getters.items():
                observables[name] = getter()
            for species, moments in self._moment_getters.items():
                for moment, getter in moments.items():
                    observables[f"{species}_{moment}"] = getter()
            self._raw_observables.update(observables)
            self._sink.emit("step_observables", observables)

            return PlasmaState(
                density=density,
                velocity=velocity,
                sound_speed=sound_speed,
                electric_field=self._extract_field("Ex"),
                magnetic_field=self._extract_field("Bx"),
                grid=self._grid,
                observables=observables,
            )

        libwarpx.warpx.step(1)

        if self._fluctuation_injector is not None:
            self._fluctuation_injector.inject()

        if self._adaptive_sigma and velocity_raw.size and sound_speed_raw.size and temperature_e.size:
            self.update_sigma_from_diagnostics(
                velocity_raw=velocity_raw,
                sound_speed_raw=sound_speed_raw,
                temperature_e=temperature_e,
            )

        observables = {}
        # Extract fields via field_getters
        for name, getter in self._field_getters.items():
            observables[name] = getter()
        # Extract moments via moment_getters
        for species, moments in self._moment_getters.items():
            for moment, getter in moments.items():
                observables[f"{species}_{moment}"] = getter()

        # Provide mock data if no getters are configured
        if not observables:
            import random
            grid_size = len(self._grid) if self._grid is not None else 100
            observables["density"] = np.array([1.0 + 0.1 * random.random() for _ in range(grid_size)])
            observables["velocity"] = np.array([0.1 + 0.05 * random.random() for _ in range(grid_size)])
            observables["sound_speed"] = np.array([0.5 + 0.1 * random.random() for _ in range(grid_size)])
        # Apply adaptive sigma smoothing if requested
        if self._adaptive_sigma and self._sigma_diagnostics is not None:
            sigma_map, diagnostics = apply_sigma_smoothing(
                density=observables.get("density", np.zeros_like(self._sigma_map)),
                sigma_map=self._sigma_map,
                gamma_e=self._gamma_e,
                gamma_i=self._gamma_i,
                ion_temperature_fraction=self._ion_temperature_fraction,
            )
            self._sigma_map = sigma_map
            self._sigma_diagnostics = diagnostics
            observables["sigma_map"] = sigma_map
            observables["sigma_diagnostics"] = _pack_sigma_diagnostics(diagnostics)
        # Store raw observables for potential export
        self._raw_observables.update(observables)
        self._sink.emit("step_observables", observables)
        return PlasmaState(
            density=density,
            velocity=velocity_raw,
            sound_speed=sound_speed_raw,
            electric_field=self._extract_field("Ex"),
            magnetic_field=observables.get("B", None),
            grid=self._grid,
            observables=observables,
        )

    def export_observables(self, requests: Iterable[str]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        for name in requests:
            if name in self._raw_observables:
                result[name] = self._raw_observables[name]
            else:
                result[name] = np.array([])
        return result

    def shutdown(self) -> None:
        if hasattr(self._sink, 'shutdown'):
            self._sink.shutdown()

    def _extract_field(self, name: str) -> np.ndarray:
        # Placeholder: extract field data from WarpX
        return np.array([])

    def _compute_density(self) -> np.ndarray:
        # Placeholder: compute number density from particle data
        return np.array([])

    def _compute_bulk_velocity(self, density: np.ndarray, apply_smoothing: bool = True) -> np.ndarray:
        # Placeholder: compute bulk velocity from momentum density
        velocity = np.array([])
        if apply_smoothing:
            velocity = self._apply_smoothing(velocity)
        return velocity

    def _compute_sound_speed(
        self,
        density: np.ndarray,
        temperature: Optional[np.ndarray] = None,
        apply_smoothing: bool = True,
    ) -> np.ndarray:
        if temperature is None:
            # Estimate temperature from pressure or use a default
            temperature = np.full_like(density, 1.0)
        # Simplified isothermal sound speed
        sound_speed_raw = np.sqrt(temperature / density)
        if apply_smoothing:
            sound_speed_raw = self._apply_smoothing(sound_speed_raw)
        return sound_speed_raw

    def _apply_smoothing(self, data: np.ndarray) -> np.ndarray:
        # Placeholder: implement Gaussian kernel smoothing
        return data

    def _call_moment_getter(self, species: str, moment: str) -> np.ndarray:
        if species in self._moment_getters and moment in self._moment_getters[species]:
            return self._moment_getters[species][moment]()
        return np.array([])

    def _species_charge(self, species_name: str) -> Optional[float]:
        for species in self._species:
            if species.name == species_name:
                return species.charge
        return None

    def update_sigma_from_diagnostics(
        self,
        density: np.ndarray,
        sigma_map: Optional[np.ndarray] = None,
        gamma_e: float = 1.0,
        gamma_i: float = 1.0,
        ion_temperature_fraction: float = 0.01,
    ) -> Tuple[np.ndarray, SigmaDiagnostics]:
        if sigma_map is None:
            sigma_map = self._sigma_map
        if sigma_map is None:
            raise ValueError("No sigma map available")
        sigma_map, diagnostics = apply_sigma_smoothing(
            density=density,
            sigma_map=sigma_map,
            gamma_e=gamma_e,
            gamma_i=gamma_i,
            ion_temperature_fraction=ion_temperature_fraction,
            sound_speed=self._compute_sound_speed,
        )
        self._sigma_map = sigma_map
        self._sigma_diagnostics = diagnostics
        return sigma_map, diagnostics

    def _species_mass(self, species_name: str) -> Optional[float]:
        for species in self._species:
            if species.name == species_name:
                return species.mass
        return None

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
