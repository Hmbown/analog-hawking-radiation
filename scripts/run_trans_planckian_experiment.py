#!/usr/bin/env python3
"""
High-fidelity Trans-Planckian experiment driver.

This script orchestrates a WarpX (or mock) backend run, captures horizon
diagnostics each step, and emits summary artefacts for downstream spectrum
analysis.  It is intentionally light on physics inline; detailed configuration
is supplied via an external YAML/JSON file describing geometry, species,
diagnostics, and storage paths.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np

# Repository imports (add scripts/ and src/ to sys.path)
REPO_ROOT = Path(__file__).resolve().parent.parent
os.sys.path.insert(0, str(REPO_ROOT))
os.sys.path.insert(0, str(REPO_ROOT / "src"))

from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend  # type: ignore  # noqa: E402
from analog_hawking.physics_engine.simulation import SimulationRunner  # type: ignore  # noqa: E402
from analog_hawking.physics_engine.horizon import HorizonResult  # type: ignore  # noqa: E402
from scripts.hawking_detection_experiment import calculate_hawking_spectrum  # type: ignore  # noqa: E402

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


DEFAULT_GRID_POINTS = 512
DEFAULT_DOMAIN_LENGTH = 50e-6  # meters
DEFAULT_DT = 1e-15  # seconds


@dataclass
class ExperimentSummary:
    steps_executed: int
    horizons_detected: int
    peak_kappa: Optional[float]
    spectra_path: Optional[str]
    notes: Dict[str, Any]


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() in {".json"}:
        return json.loads(path.read_text())
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs")
        return yaml.safe_load(path.read_text())
    raise ValueError(f"Unsupported config extension: {path.suffix}")


def _default_mock_config(grid_points: int = DEFAULT_GRID_POINTS) -> Dict[str, Any]:
    grid = np.linspace(0.0, DEFAULT_DOMAIN_LENGTH, grid_points)
    return {
        "mock": True,
        "grid": grid,
        "field_getters": {
            "Ex": {"type": "pywarpx", "field": "Ex"}
        },
        "moment_getters": {
            "electrons": {
                "density": {"type": "pywarpx", "moment": "density"},
                "bulk_velocity": {"type": "pywarpx", "moment": "bulk_velocity"},
                "sound_speed": {"type": "pywarpx", "moment": "sound_speed"},
            }
        },
        "electron_species": "electrons",
        "ion_species": "ions",
        "results_dir": "results/trans_planckian_experiment",
        "write_sidecar": True,
        "sigma_smoothing": {
            "adaptive": True,
            "gamma_e": 1.0,
            "gamma_i": 1.0,
            "ion_temperature_fraction": 0.01,
        },
    }


def _save_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _summarise_horizons(horizons: Optional[HorizonResult]) -> Dict[str, Any]:
    if horizons is None or horizons.positions.size == 0:
        return {"positions": [], "kappa": [], "kappa_err": []}
    return {
        "positions": horizons.positions.tolist(),
        "kappa": horizons.kappa.tolist(),
        "kappa_err": horizons.kappa_err.tolist(),
        "dvdx": horizons.dvdx.tolist(),
        "dcsdx": horizons.dcsdx.tolist(),
    }


def _generate_spectrum(kappa: float, out_dir: Path) -> Optional[str]:
    spec = calculate_hawking_spectrum(
        kappa,
        emitting_area_m2=1e-6,
        solid_angle_sr=5e-2,
        coupling_efficiency=0.1,
    )
    if not spec.get("success", False):
        return None
    payload = {
        "frequencies": spec["frequencies"].tolist(),
        "power_spectrum": spec["power_spectrum"].tolist(),
        "peak_frequency": float(spec.get("peak_frequency", 0.0)),
        "temperature": float(spec.get("temperature", 0.0)),
    }
    out_path = out_dir / "hawking_spectrum.json"
    _save_json(out_path, payload)
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Trans-Planckian WarpX experiment with horizon diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="Path to YAML/JSON run configuration.")
    parser.add_argument("--mock", action="store_true", help="Force mock mode (no WarpX dependency).")
    parser.add_argument("--steps", type=int, default=10, help="Number of WarpX steps to execute.")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Timestep passed to backend (seconds).")
    parser.add_argument("--results-dir", type=Path, default=Path("results/trans_planckian_experiment"),
                        help="Directory for summaries and diagnostics.")
    parser.add_argument("--spectrum", action="store_true", help="Emit Hawking spectrum JSON for best κ.")
    parser.add_argument("--grid-points", type=int, default=DEFAULT_GRID_POINTS,
                        help="Grid points for mock configuration.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.config:
        config = _load_config(args.config)
    else:
        config = _default_mock_config(grid_points=args.grid_points)

    if args.mock:
        config.update(_default_mock_config(grid_points=args.grid_points))
    elif config.get("mock", False):
        # configuration already requests mock; ensure grid exists
        config.setdefault("grid", np.linspace(0.0, DEFAULT_DOMAIN_LENGTH, args.grid_points))

    config["results_dir"] = str(args.results_dir)
    config.setdefault("write_sidecar", True)

    backend = WarpXBackend()
    runner = SimulationRunner(backend)

    runner.configure(config)

    step_summaries: Dict[str, Any] = {}
    peak_kappa: Optional[float] = None

    for step in range(args.steps):
        outputs = runner.run_step(args.dt)
        horizons = outputs.horizons
        step_info = _summarise_horizons(horizons)
        step_info["step_index"] = step
        step_summaries[f"step_{step:05d}"] = step_info
        if step_info["kappa"]:
            kappa_step = float(max(step_info["kappa"]))
            if peak_kappa is None or kappa_step > peak_kappa:
                peak_kappa = kappa_step

    runner.shutdown()

    spectrum_path: Optional[str] = None
    if args.spectrum and peak_kappa is not None:
        spectrum_path = _generate_spectrum(peak_kappa, args.results_dir)

    summary = ExperimentSummary(
        steps_executed=args.steps,
        horizons_detected=sum(1 for info in step_summaries.values() if info["kappa"]),
        peak_kappa=peak_kappa,
        spectra_path=spectrum_path,
        notes={
            "config_path": str(args.config) if args.config else None,
            "mock": bool(config.get("mock", False)),
        },
    )

    metadata = {
        "summary": asdict(summary),
        "steps": step_summaries,
    }
    _save_json(args.results_dir / "trans_planckian_summary.json", metadata)
    print(f"Wrote summary to {args.results_dir / 'trans_planckian_summary.json'}")
    if spectrum_path:
        print(f"Hawking spectrum saved to {spectrum_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
