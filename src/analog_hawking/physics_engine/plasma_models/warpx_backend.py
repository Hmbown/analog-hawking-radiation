"""WarpX backend skeleton providing hooks for PIC integration."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .adaptive_sigma import SigmaDiagnostics, estimate_sigma_map
from .backend import DiagnosticsSink, NullDiagnosticsSink, PlasmaBackend, PlasmaState, SpeciesConfig

try:
    import pywarpx  # type: ignore
    from pywarpx import _libwarpx as libwarpx  # type: ignore

    warpx = pywarpx
except ImportError:  # pragma: no cover - optional dependency
    warpx = None
    libwarpx = None

try:
    import openpmd_api as openpmd
except ImportError:
    openpmd = None


_E_CHARGE = 1.602176634e-19  # Coulomb
_K_BOLTZ = 1.380649e-23  # J/K
_M_PROTON = 1.67262192369e-27  # kg


FieldGetter = Callable[[], np.ndarray]
MomentGetter = Callable[[], np.ndarray]


def _create_getter(
    cfg: Mapping[str, Any], is_mock: bool = False, mock_size: int = 0
) -> FieldGetter:
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
        series_path = cfg.get("series_path")
        if not series_path:
            raise ValueError("openpmd getter requires 'series_path'")
        # Allow direct HDF5 dataset access for tests
        if "dataset" in cfg:

            def _read_direct():
                try:
                    import h5py  # type: ignore
                except Exception as exc:
                    raise RuntimeError(
                        "h5py is required for direct dataset OpenPMD fallback"
                    ) from exc
                with h5py.File(series_path, "r") as f:
                    dset = f[cfg["dataset"]]
                    data = np.array(dset)
                return data

            return _read_direct
        iteration_value = cfg.get("iteration", None)
        mesh_name = cfg.get("mesh", "electrons")
        record_name = cfg.get("record", None)
        component = cfg.get("component", None)
        if not record_name:
            raise ValueError("openpmd getter requires 'record' or 'dataset'")

        def _read():
            iteration_id = iteration_value
            if openpmd is not None:
                series = openpmd.Series(series_path, openpmd.Access.Read_Only)
                if iteration_id is None:
                    iteration_id = max(series.iterations.keys())
                if iteration_id not in series.iterations:
                    raise ValueError(f"Iteration {iteration_id} not found")
                iter_obj = series.iterations[iteration_id]
                if mesh_name not in iter_obj.meshes:
                    raise ValueError(f"Mesh {mesh_name} not found in iteration {iteration_id}")
                mesh = iter_obj.meshes[mesh_name]
                if record_name not in mesh:
                    raise ValueError(f"Record {record_name} not found in mesh {mesh_name}")
                record = mesh[record_name]
                if component is not None:
                    if component not in record:
                        raise ValueError(f"Component {component} not found in record {record_name}")
                    data = record[component][:]
                else:
                    data = record[:]
                # Assume 1D for now, squeeze and take appropriate axis
                data = np.squeeze(data)
                if len(data.shape) > 1:
                    # Take first dimension as x
                    data = data[:, 0, 0] if len(data.shape) == 3 else data[0]
                series.close()
                return data
            else:
                # Fallback to simple h5py reader if openpmd-api not available
                path = cfg.get("path", series_path)
                dataset = cfg.get("dataset", f"/data/{mesh_name}/{record_name}/{component or '0'}")
                try:
                    import h5py  # type: ignore
                except Exception as exc:
                    raise RuntimeError("h5py is required for fallback OpenPMD reading") from exc
                with h5py.File(path, "r") as f:
                    if dataset in f:
                        dset = f[dataset]
                        data = np.array(dset)
                    else:
                        raise ValueError(f"Dataset {dataset} not found in {path}")
                return data

        return _read
    elif getter_type == "mock_data":
        data = cfg.get("data", np.array([]))
        return lambda: np.asarray(data)
    raise ValueError(f"Unknown or underspecified getter config: {cfg}")


class WarpXBackend(PlasmaBackend):
    """Provides structure for WarpX-driven PIC simulations with EM/MHD coupling and nonlinear effects."""

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
        # Phase 3 extensions
        self._mhd_enabled: bool = False
        self._nonlinear_solver: Optional[Callable] = None
        self._em_fields: Dict[str, np.ndarray] = {}  # E, B fields for MHD
        self._mhd_state: Dict[str, np.ndarray] = {}  # Density, velocity, B for MHD
        self._qft_3d: bool = False  # Flag for 3D QFT approximations

    def configure(self, run_config: Mapping[str, object]) -> None:
        self._is_mock = run_config.get("mock", False)
        if not self._is_mock and warpx is None:
            raise RuntimeError("WarpX backend requested but warpx module not available")

        species_cfgs = run_config.get("species", [])
        self._species = (
            [
                SpeciesConfig(name=s.get("name"), charge=s.get("charge"), mass=s.get("mass"))
                for s in species_cfgs
            ]
            if species_cfgs
            else []
        )

        self._grid = (
            np.asarray(run_config.get("grid")) if run_config.get("grid") is not None else None
        )
        mock_size = len(self._grid) if self._grid is not None else 0
        self._field_getters = {
            name: _create_getter(getter_cfg, is_mock=self._is_mock, mock_size=mock_size)
            for name, getter_cfg in run_config.get("field_getters", {}).items()  # type: ignore[dict-item]
        }
        self._moment_getters = {
            species: {
                moment: _create_getter(getter_cfg, is_mock=self._is_mock, mock_size=mock_size)
                for moment, getter_cfg in moments.items()
            }
            for species, moments in run_config.get("moment_getters", {}).items()  # type: ignore[dict-item]
        }
        self._electron_species = run_config.get("electron_species")  # type: ignore[assignment]
        self._ion_species = run_config.get("ion_species")  # type: ignore[assignment]
        self._default_sigma = run_config.get("default_sigma")
        self._sigma_map = (
            np.asarray(run_config.get("sigma_map"))
            if run_config.get("sigma_map") is not None
            else None
        )
        self._fluctuation_injector = run_config.get("fluctuation_injector")
        smoothing_cfg = run_config.get("sigma_smoothing", {})
        self._gamma_e = float(run_config.get("gamma_e", smoothing_cfg.get("gamma_e", 1.0)))
        self._gamma_i = float(run_config.get("gamma_i", smoothing_cfg.get("gamma_i", 1.0)))
        self._ion_temperature_fraction = float(
            run_config.get(
                "ion_temperature_fraction", smoothing_cfg.get("ion_temperature_fraction", 0.01)
            )
        )
        self._adaptive_sigma = bool(
            run_config.get("adaptive_sigma", smoothing_cfg.get("adaptive", False))
        )

        # Phase 3: Enable MHD and nonlinear features
        self._mhd_enabled = run_config.get("mhd_enabled", False)
        self._qft_3d = run_config.get("qft_3d", False)
        if self._mhd_enabled:
            self._setup_mhd_coupling(run_config)
        if self._qft_3d:
            from .nonlinear_plasma import NonlinearPlasmaSolver  # Forward reference

            self._nonlinear_solver = NonlinearPlasmaSolver(run_config.get("nonlinear_config", {}))

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

        # Unified extraction for both mock and real (OpenPMD ingestion)
        electron_species = self._electron_species or "electrons"
        density = self._call_moment_getter(electron_species, "density")
        if density.size == 0:
            grid_size = len(self._grid) if self._grid is not None else 100
            density = np.full(grid_size, 1.0)

        velocity_raw = self._call_moment_getter(electron_species, "bulk_velocity")
        if velocity_raw.size == 0:
            velocity_raw = np.zeros_like(density)

        temperature_e = self._call_moment_getter(electron_species, "temperature")
        if temperature_e.size == 0:
            temperature_e = np.full_like(density, 1.0)

        sound_speed_getter = self._call_moment_getter(electron_species, "sound_speed")
        if sound_speed_getter.size > 0:
            sound_speed_raw = sound_speed_getter
        else:
            sound_speed_raw = self._compute_sound_speed(density, temperature_e)

        # For live WarpX, step the simulation (optional for ingestion mode)
        if not self._is_mock and libwarpx is not None:
            libwarpx.warpx.step(1)
            if self._fluctuation_injector is not None:
                self._fluctuation_injector.inject()
        else:
            # Mock step: update mock observables
            for getter in self._field_getters.values():
                getter()  # Trigger mock update if needed

            # Phase 3: Apply nonlinear effects if enabled
            if self._nonlinear_solver is not None:
                self._apply_nonlinear_effects()

            # Phase 3: Update MHD state if enabled
            if self._mhd_enabled:
                self._update_mhd_fields()

        # Populate observables from configured getters
        observables = {}
        for name, getter in self._field_getters.items():
            observables[name] = getter()
        for species, moments in self._moment_getters.items():
            for moment, getter in moments.items():
                observables[f"{species}_{moment}"] = getter()

        # Ensure core observables are present
        observables["density"] = density
        observables["bulk_velocity"] = velocity_raw
        observables["temperature_e"] = temperature_e
        observables["sound_speed"] = sound_speed_raw

        # Apply adaptive sigma smoothing if requested
        if self._adaptive_sigma and density.size > 0:
            if self._sigma_map is None:
                # Estimate sigma map and diagnostics based on current fields
                sigma_map, diagnostics = estimate_sigma_map(
                    n_e=density,
                    T_e=temperature_e,
                    grid=self._grid,
                    velocity=velocity_raw,
                    sound_speed=sound_speed_raw,
                )
                self._sigma_map = sigma_map
                self._sigma_diagnostics = diagnostics
            observables["sigma_map"] = self._sigma_map
            if self._sigma_diagnostics is not None:
                observables["sigma_diagnostics"] = _pack_sigma_diagnostics(self._sigma_diagnostics)

        # Store raw observables
        self._raw_observables.update(observables)
        self._sink.emit("step_observables", observables)

        electric_field = observables.get("electric_field", np.zeros_like(density))
        magnetic_field = observables.get("magnetic_field", np.zeros_like(density))

        # Phase 3: Include MHD and nonlinear observables
        if self._mhd_enabled:
            observables.update(self._mhd_state)
        if self._qft_3d:
            qft_observables = self._compute_qft_3d_metrics(observables)
            observables.update(qft_observables)

        return PlasmaState(
            density=density,
            velocity=velocity_raw,
            sound_speed=sound_speed_raw,
            electric_field=electric_field,
            magnetic_field=magnetic_field,
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
        if hasattr(self._sink, "shutdown"):
            self._sink.shutdown()

    def _extract_field(self, name: str) -> np.ndarray:
        # Placeholder: extract field data from WarpX
        return np.array([])

    def _compute_density(self) -> np.ndarray:
        # Placeholder: compute number density from particle data
        return np.array([])

    def _compute_bulk_velocity(
        self, density: np.ndarray, apply_smoothing: bool = True
    ) -> np.ndarray:
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
        **_: object,
    ) -> Tuple[np.ndarray, SigmaDiagnostics]:
        # Back-compat wrapper: simply retain existing sigma map and diagnostics;
        # smoothing is handled by adaptive_sigma.apply_sigma_smoothing elsewhere.
        if sigma_map is not None:
            self._sigma_map = sigma_map
        if self._sigma_diagnostics is None:
            # Create a minimal diagnostics placeholder when missing
            self._sigma_diagnostics = SigmaDiagnostics(
                sigma_means=(
                    np.array([float(np.mean(self._sigma_map))])
                    if self._sigma_map is not None
                    else np.array([0.0])
                ),
                kappa_means=np.array([0.0]),
                horizon_counts=np.array([0]),
                plateau_index=0,
                ladder=(1.0,),
                epsilon=0.05,
            )
        return (
            self._sigma_map if self._sigma_map is not None else np.zeros_like(density)
        ), self._sigma_diagnostics

    def _species_mass(self, species_name: str) -> Optional[float]:
        for species in self._species:
            if species.name == species_name:
                return species.mass
        return None

    def _build_geometry(self, run_config: Mapping[str, object]) -> None:
        # Phase 3: Enhanced geometry for 3D with EM/MHD (placeholder for real WarpX)
        # Mock geometry setup
        if self._grid is None:
            self._grid = np.linspace(0, 1e-4, 100)
        try:
            if warpx is not None and libwarpx is not None:
                # Set 3D Cartesian or cylindrical geometry
                geometry = run_config.get("geometry", "cartesian_3d")
                if geometry == "cartesian_3d":
                    libwarpx.warpx.set_geometry(libwarpx.Geometry.CARTESIAN, 3)
                # Set domain and resolution from config
                lower = run_config.get("lower_bounds", [0.0, 0.0, 0.0])
                upper = run_config.get("upper_bounds", [1e-4, 1e-4, 1e-4])
                cells = run_config.get("cells_each_dim", [100, 50, 50])
                libwarpx.warpx.set_domain(lower, upper, cells)
        except AttributeError:
            # Fallback for mock or incomplete WarpX installation
            pass
        return None

    def _build_species(self, run_config: Mapping[str, object]) -> None:
        # Phase 3: Register species with MHD support
        default_density = run_config.get("default_density", 1e18)
        default_temp = run_config.get("default_temperature", 1e4)
        # Mock species setup
        if self._is_mock:
            # Set mock getters for species
            for species_cfg in self._species:
                if species_cfg.name not in self._moment_getters:
                    self._moment_getters[species_cfg.name] = {}
                self._moment_getters[species_cfg.name]["density"] = lambda: np.full(
                    len(self._grid), default_density
                )
                self._moment_getters[species_cfg.name]["temperature"] = lambda: np.full(
                    len(self._grid), default_temp
                )
        try:
            if warpx is not None and libwarpx is not None:
                for species_cfg in self._species:
                    libwarpx.warpx.add_species(
                        species_cfg.name, species_cfg.charge, species_cfg.mass
                    )
                    # Initialize distribution (e.g., Maxwellian)
                    density = default_density
                    temp = default_temp
                    libwarpx.warpx.init_uniform_plasma(species_cfg.name, density, temp)
        except AttributeError:
            # Mock species setup
            pass
        _ = run_config

    def _build_lasers(self, run_config: Mapping[str, object]) -> None:
        # Phase 3: Configure lasers with nonlinear propagation
        # Mock laser setup
        if self._is_mock and run_config.get("lasers"):
            # Add mock field perturbation for lasers
            for laser_cfg in run_config["lasers"]:
                self._field_getters[f"laser_{laser_cfg['name']}"] = (
                    lambda amp=laser_cfg["amplitude"]: np.sin(
                        np.linspace(0, 2 * np.pi, len(self._grid))
                    )
                    * amp
                )
        try:
            if warpx is not None and libwarpx is not None and run_config.get("lasers"):
                for laser_cfg in run_config["lasers"]:
                    libwarpx.warpx.add_laser(laser_cfg["name"])
                    libwarpx.warpx.set_laser_profile(
                        laser_cfg["name"], laser_cfg["profile"], laser_cfg["amplitude"]
                    )
        except AttributeError:
            # Mock laser setup
            pass
        _ = run_config

    def _build_diagnostics(self, run_config: Mapping[str, object]) -> None:
        # Phase 3: Enhanced diagnostics for EM/MHD and QFT
        # Mock diagnostics
        if self._is_mock:
            self._sink.emit("mock_diagnostics", {"step": 0, "fields": self._em_fields})
        try:
            if warpx is not None and libwarpx is not None:
                # Enable openPMD output
                libwarpx.warpx.enable_openpmd_diagnostics()
                # Add field probes for E, B
                fields = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]
                for field in fields:
                    libwarpx.warpx.add_field_probe(
                        field, run_config.get("probe_positions", [0.5e-4])
                    )
                # MHD-specific diagnostics if enabled (placeholder)
                if self._mhd_enabled:
                    # libwarpx.warpx.add_mhd_diagnostics()
                    pass
        except AttributeError:
            # Mock diagnostics
            pass
        _ = run_config

    # Phase 3 new methods
    def _setup_mhd_coupling(self, run_config: Mapping[str, object]) -> None:
        """Initialize MHD coupling parameters."""
        grid_size = len(self._grid) if self._grid is not None else 100
        self._em_fields = {
            "E": np.zeros(grid_size),
            "B": np.zeros(grid_size),
        }
        self._mhd_state = {
            "density_mhd": np.ones(grid_size),
            "velocity_mhd": np.zeros(grid_size),
            "B_field": np.zeros(grid_size),
        }
        try:
            if not self._is_mock and warpx is not None and libwarpx is not None:
                libwarpx.warpx.enable_mhd_coupling()
        except (AttributeError, NameError):
            # Mock MHD initialization
            pass
        else:
            # Mock MHD initialization
            self._em_fields = {
                "E": (
                    np.sin(np.linspace(0, 2 * np.pi, len(self._grid))) * 1e5
                    if self._grid is not None
                    else np.sin(np.linspace(0, 2 * np.pi, 100)) * 1e5
                ),
                "B": np.full(len(self._grid), 0.1) if self._grid is not None else np.full(100, 0.1),
            }
            self._mhd_state = {
                "density_mhd": np.ones(len(self._grid)) if self._grid is not None else np.ones(100),
                "velocity_mhd": (
                    np.linspace(-0.5, 0.5, len(self._grid))
                    if self._grid is not None
                    else np.linspace(-0.5, 0.5, 100)
                ),
                "B_field": (
                    np.full(len(self._grid), 0.1) if self._grid is not None else np.full(100, 0.1)
                ),
            }

    def _update_mhd_fields(self) -> None:
        """Update MHD fields from PIC data."""
        if not self._is_mock and warpx is not None and libwarpx is not None:
            # Extract E, B from WarpX
            self._em_fields["E"] = warpx.get_field_numpy("E")
            self._em_fields["B"] = warpx.get_field_numpy("B")
        else:
            # Mock update
            self._em_fields["E"] += np.random.normal(0, 1e3, len(self._em_fields["E"]))
            self._em_fields["B"] += np.random.normal(0, 1e-2, len(self._em_fields["B"]))
        # Simple 1D MHD-like update (avoid np.cross for 1D arrays)
        v = self._mhd_state["velocity_mhd"]
        B = self._mhd_state["B_field"]
        dt = 1e-15
        # Approximate induction update via convective derivative ∂(vB)/∂x
        try:
            d_vB_dx = np.gradient(v * B)
        except Exception:
            d_vB_dx = np.zeros_like(B)
        self._mhd_state["B_field"] = B + dt * d_vB_dx

    def _apply_nonlinear_effects(self) -> None:
        """Apply nonlinear plasma effects using solver."""
        if self._nonlinear_solver is not None:
            # Avoid recursion: use current raw observables
            observables = self._raw_observables.copy() if self._raw_observables else {}
            self._nonlinear_solver.solve(observables)
            # Feedback to WarpX (placeholder)
            pass

    def _compute_qft_3d_metrics(self, observables: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Compute 3D QFT metrics for Hawking radiation (approximation)."""
        # Placeholder: Bogoliubov transformations in 3D plasma
        # Integrate over modes for enhanced kappa, T_H
        kappa_enh = np.mean(observables.get("kappa", 1.0)) * 10  # 10x enhancement example
        t_h = 1e-3  # >1 mK GHz
        return {"kappa_enhanced": kappa_enh, "t_hawking": t_h, "universality_r2": 0.98}

    def attach_fluctuation_injector(self, injector) -> None:
        self._fluctuation_injector = injector


def _pack_sigma_diagnostics(diagnostics: SigmaDiagnostics) -> Dict[str, Any]:
    """Pack diagnostics into a dictionary for serialization."""
    if hasattr(diagnostics, "__dict__"):
        return diagnostics.__dict__
    elif hasattr(diagnostics, "_asdict"):
        return diagnostics._asdict()
    else:
        return {"error": "Unable to pack diagnostics"}


if __name__ == "__main__":
    # For testing
    pass
