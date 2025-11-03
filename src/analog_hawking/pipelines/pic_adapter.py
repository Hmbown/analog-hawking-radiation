"""Adapters for turning PIC/openPMD diagnostics into horizon-ready profiles."""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

try:
    import openpmd_api as openpmd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    openpmd = None

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

from ..physics_engine.horizon import HorizonResult, find_horizons_with_uncertainty, sound_speed


@dataclass
class OpenPMDAdapterResult:
    """Datasets parsed from an openPMD series and derived horizon diagnostics."""

    grid: np.ndarray
    velocity: np.ndarray
    sound_speed: np.ndarray
    density: Optional[np.ndarray] = None
    temperature: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    horizon: HorizonResult | None = None

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "grid": self.grid,
            "velocity": self.velocity,
            "sound_speed": self.sound_speed,
            "density": self.density,
            "temperature": self.temperature,
            "metadata": self.metadata,
        }
        if self.horizon is not None:
            result["horizon"] = dataclasses.asdict(self.horizon)
        return result


def _resolve_iteration(series, iteration: Optional[int], latest: bool) -> int:
    keys = sorted(series.iterations.keys())
    if not keys:
        raise ValueError("openPMD series contains no iterations")
    if iteration is not None:
        if iteration not in series.iterations:
            raise ValueError(f"iteration {iteration} not found in series (available: {keys})")
        return iteration
    return keys[-1] if latest else keys[0]


def _squeeze_1d(data) -> np.ndarray:
    arr = np.asarray(data)
    while arr.ndim > 1:
        arr = arr.take(0, axis=-1)
    return arr


def _read_openpmd_record(
    iteration, mesh: str, record: str, component: Optional[str]
) -> Tuple[np.ndarray, Mapping[str, Any]]:
    if mesh not in iteration.meshes:
        raise KeyError(
            f"mesh '{mesh}' not found; available meshes: {list(iteration.meshes.keys())}"
        )
    mesh_obj = iteration.meshes[mesh]
    if record not in mesh_obj:
        raise KeyError(f"record '{record}' not found in mesh '{mesh}'")
    rec = mesh_obj[record]
    if component is not None:
        if component not in rec:
            raise KeyError(f"component '{component}' not found in record '{record}'")
        data = rec[component][:]
    else:
        data = rec[:]
    attrs: Dict[str, Any] = {}
    try:
        attrs["axis_labels"] = list(rec.axis_labels)
    except Exception:
        pass
    try:
        attrs["grid_spacing"] = tuple(float(v) for v in rec.grid_spacing)
    except Exception:
        pass
    try:
        attrs["grid_global_offset"] = tuple(float(v) for v in rec.grid_global_offset)
    except Exception:
        pass
    try:
        attrs["geometry"] = getattr(rec, "geometry", None)
    except Exception:
        pass
    return _squeeze_1d(data), attrs


def _construct_grid(length: int, attrs: Mapping[str, Any]) -> np.ndarray:
    spacing = attrs.get("grid_spacing")
    offset = attrs.get("grid_global_offset")
    if spacing and offset:
        dx = float(spacing[0])
        x0 = float(offset[0])
        return x0 + dx * np.arange(length)
    return np.linspace(0.0, float(length - 1), length)


def _read_h5_dataset(path: str, dataset: str) -> np.ndarray:
    if h5py is None:
        raise RuntimeError(
            "h5py is required to read openPMD datasets without the openPMD-api package"
        )
    with h5py.File(path, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"dataset '{dataset}' not found in file '{path}'")
        return _squeeze_1d(handle[dataset][()])


def _compute_uncertainty(arr: np.ndarray, window: int = 3) -> np.ndarray:
    if arr.size < 2 or window < 2:
        return np.zeros_like(arr)
    pad = window // 2
    extended = np.pad(arr, pad_width=pad, mode="edge")
    local_std = []
    for i in range(arr.size):
        segment = extended[i : i + window]
        local_std.append(float(np.std(segment)))
    return np.asarray(local_std)


def from_openpmd(
    series_uri: str,
    t: Optional[int | str] = None,
    *,
    velocity_source: Mapping[str, Any] | None = None,
    sound_speed_source: Mapping[str, Any] | None = None,
    density_source: Mapping[str, Any] | None = None,
    temperature_source: Mapping[str, Any] | None = None,
    auto_sound_speed: bool = True,
    horizon: bool = True,
    kappa_method: str = "acoustic_exact",
    uncertainty_window: int = 5,
) -> OpenPMDAdapterResult:
    """Extract 1D profiles from an openPMD series and compute horizon metadata.

    Args:
        series_uri: Path to the openPMD series (directory or file pattern). Accepts
            ``\"%T\"`` placeholders as defined by openPMD.
        t: Iteration index or \"latest\"/\"first\". When ``None`` uses the latest iteration.
        velocity_source, sound_speed_source, density_source, temperature_source:
            Optional mappings with keys ``mesh``, ``record``, and ``component`` describing
            where to load the respective quantity. By default the adapter attempts to
            use WarpX-derived diagnostics:

            - Velocity: mesh=\"derived\", record=\"bulk_velocity\", component=\"x\"
            - Sound speed: mesh=\"derived\", record=\"sound_speed\", component=\"0\"
            - Density: mesh=\"derived\", record=\"density\", component=\"0\"
            - Temperature: mesh=\"derived\", record=\"temperature\", component=\"0\"

            When ``sound_speed_source`` is not provided and ``auto_sound_speed`` is True,
            the adapter derives :math:`c_s` from the electron temperature using
            :func:`analog_hawking.physics_engine.horizon.sound_speed`.
        auto_sound_speed: When True and a dedicated sound-speed record is not supplied,
            the returned profile computes :math:`c_s` from the temperature dataset.
        horizon: Whether to run horizon finding on the extracted profile.
        kappa_method: κ definition forwarded to :func:`find_horizons_with_uncertainty`.
        uncertainty_window: Window size (samples) for simple local standard-deviation
            based uncertainties on the velocity and sound-speed arrays.

    Returns:
        :class:`OpenPMDAdapterResult` with arrays and optional horizon metadata.
    """

    iteration_label = None
    if isinstance(t, str):
        lowered = t.lower()
        if lowered in {"latest", "last", "final"}:
            iteration_label = "latest"
        elif lowered in {"first", "initial"}:
            iteration_label = "first"
        else:
            try:
                t = int(t)
            except ValueError as exc:
                raise ValueError(f"unrecognized iteration selector '{t}'") from exc
    if isinstance(t, (int, np.integer)):
        iteration_index = int(t)
    else:
        iteration_index = None

    default_velocity = {"mesh": "derived", "record": "bulk_velocity", "component": "x"}
    default_cs = {"mesh": "derived", "record": "sound_speed", "component": "0"}
    default_density = {"mesh": "derived", "record": "density", "component": "0"}
    default_temperature = {"mesh": "derived", "record": "temperature", "component": "0"}

    v_src = dict(default_velocity, **(velocity_source or {}))
    cs_src = dict(default_cs, **(sound_speed_source or {}))
    ne_src = dict(default_density, **(density_source or {}))
    te_src = dict(default_temperature, **(temperature_source or {}))

    grid = None
    velocity = None
    sound_speed_arr = None
    density = None
    temperature = None
    metadata: Dict[str, Any] = {"series_uri": series_uri}

    if openpmd is not None and (
        "%T" in series_uri
        or os.path.isdir(os.path.dirname(series_uri))
        or os.path.isdir(series_uri)
    ):
        access_uri = series_uri
        if "%T" not in access_uri and os.path.isdir(access_uri):
            access_uri = os.path.join(access_uri, "openpmd_%T.h5")
        series = openpmd.Series(access_uri, openpmd.Access.read_only)
        try:
            idx = _resolve_iteration(series, iteration_index, latest=(iteration_label != "first"))
            iteration = series.iterations[idx]
            velocity, v_attrs = _read_openpmd_record(
                iteration, v_src["mesh"], v_src["record"], v_src.get("component")
            )
            grid = _construct_grid(len(velocity), v_attrs)
            metadata["iteration"] = idx
            metadata["velocity_attrs"] = dict(v_attrs)
            try:
                sound_speed_arr, cs_attrs = _read_openpmd_record(
                    iteration, cs_src["mesh"], cs_src["record"], cs_src.get("component")
                )
                metadata["sound_speed_attrs"] = dict(cs_attrs)
            except KeyError:
                sound_speed_arr = None
            try:
                density, ne_attrs = _read_openpmd_record(
                    iteration, ne_src["mesh"], ne_src["record"], ne_src.get("component")
                )
                metadata["density_attrs"] = dict(ne_attrs)
            except KeyError:
                density = None
            try:
                temperature, te_attrs = _read_openpmd_record(
                    iteration, te_src["mesh"], te_src["record"], te_src.get("component")
                )
                metadata["temperature_attrs"] = dict(te_attrs)
            except KeyError:
                temperature = None
        finally:
            series.close()
    else:
        # Assume direct HDF5 dataset paths when openPMD python bindings are unavailable.
        if not isinstance(v_src.get("path"), str):
            raise RuntimeError(
                "openPMD-api not available; please provide explicit 'path' and 'dataset' "
                "entries in velocity_source/sound_speed_source mappings."
            )
        velocity = _read_h5_dataset(v_src["path"], v_src["dataset"])
        grid = np.linspace(0.0, float(velocity.size - 1), velocity.size)
        metadata["iteration"] = iteration_index
        if cs_src.get("dataset"):
            sound_speed_arr = _read_h5_dataset(cs_src["path"], cs_src["dataset"])
        if ne_src.get("dataset"):
            density = _read_h5_dataset(ne_src["path"], ne_src["dataset"])
        if te_src.get("dataset"):
            temperature = _read_h5_dataset(te_src["path"], te_src["dataset"])

    if velocity is None or grid is None:
        raise RuntimeError("failed to extract velocity profile from openPMD series")

    grid = np.asarray(grid, dtype=float)
    velocity = np.asarray(velocity, dtype=float)

    if sound_speed_arr is None:
        if auto_sound_speed:
            if temperature is None:
                raise RuntimeError(
                    "sound_speed_source not found and auto_sound_speed=True, but temperature dataset is unavailable"
                )
            sound_speed_arr = sound_speed(np.asarray(temperature, dtype=float))
            metadata["sound_speed_derived_from_temperature"] = True
        else:
            raise RuntimeError("sound_speed_source not provided and auto_sound_speed disabled")
    else:
        sound_speed_arr = np.asarray(sound_speed_arr, dtype=float)

    density = np.asarray(density, dtype=float) if density is not None else None
    temperature = np.asarray(temperature, dtype=float) if temperature is not None else None

    # Simple uncertainty estimates from local variance
    velocity_unc = _compute_uncertainty(velocity, window=uncertainty_window)
    cs_unc = _compute_uncertainty(sound_speed_arr, window=uncertainty_window)

    metadata["velocity_uncertainty"] = velocity_unc
    metadata["sound_speed_uncertainty"] = cs_unc

    horizon_result: HorizonResult | None = None
    if horizon:
        horizon_result = find_horizons_with_uncertainty(
            grid,
            velocity,
            sound_speed_arr,
            kappa_method=kappa_method,
        )

    return OpenPMDAdapterResult(
        grid=grid,
        velocity=velocity,
        sound_speed=sound_speed_arr,
        density=density,
        temperature=temperature,
        metadata=metadata,
        horizon=horizon_result,
    )


__all__ = ["OpenPMDAdapterResult", "from_openpmd"]
