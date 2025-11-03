#!/usr/bin/env python3
"""Driver for running WarpX horizon diagnostics with adaptive smoothing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from physics_engine.plasma_models.warpx_backend import WarpXBackend
from physics_engine.simulation import SimulationRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WarpX horizon diagnostics")
    parser.add_argument(
        "--config", type=Path, help="Path to JSON configuration file for WarpX setup"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="Number of simulation steps to execute"
    )
    parser.add_argument(
        "--dt", type=float, default=0.0, help="Time step per iteration (WarpX uses internal dt)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results"), help="Directory for diagnostic outputs"
    )
    parser.add_argument(
        "--sigma-policy",
        choices=["fixed", "adaptive"],
        default="adaptive",
        help="Smoothing policy for diagnostics",
    )
    parser.add_argument(
        "--sigma", type=float, default=2.0, help="Default Gaussian smoothing width in cells"
    )
    parser.add_argument(
        "--sigma-ladder",
        type=str,
        default="0.5,1.0,2.0,4.0",
        help="Comma-separated ladder for adaptive sigma exploration",
    )
    parser.add_argument(
        "--sigma-epsilon",
        type=float,
        default=0.05,
        help="Plateau tolerance for adaptive sigma selection",
    )
    parser.add_argument(
        "--save-profiles",
        type=Path,
        help="Path to save velocity and sound speed profiles (NPZ format)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Allow appending to existing results directory"
    )
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data


def _prepare_backend_config(
    namespace: argparse.Namespace, base_config: Dict[str, object]
) -> Dict[str, object]:
    smoothing_cfg = base_config.get("smoothing", {})
    if namespace.sigma_policy == "fixed":
        smoothing_cfg = dict(smoothing_cfg, sigma=namespace.sigma, adaptive=False)
    else:
        ladder = tuple(float(x.strip()) for x in namespace.sigma_ladder.split(","))
        smoothing_cfg = dict(
            smoothing_cfg,
            sigma=namespace.sigma,
            adaptive=True,
            ladder=ladder,
            epsilon=namespace.sigma_epsilon,
        )
    config = dict(base_config)
    config["smoothing"] = smoothing_cfg
    if namespace.sigma_policy == "adaptive":
        config["adaptive_sigma"] = True
    if namespace.sigma_policy == "fixed":
        config["sigma_cells"] = namespace.sigma
    return config


def _ensure_results_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"results directory {path} already exists (use --overwrite to override)"
        )
    path.mkdir(parents=True, exist_ok=True)


def _write_profiles(path: Path, state: "PlasmaState") -> None:
    """Save plasma profiles to a compressed NumPy archive."""
    if state.grid is not None and state.velocity is not None and state.sound_speed is not None:
        np.savez_compressed(
            path,
            grid=state.grid,
            velocity=state.velocity,
            sound_speed=state.sound_speed,
        )


def _write_summary(results_dir: Path, horizon_records: List[Dict[str, object]]) -> None:
    summary_path = results_dir / "horizon_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(horizon_records, handle, indent=2)


def main() -> None:
    args = _parse_args()
    base_config = _load_config(args.config)
    backend_config = _prepare_backend_config(args, base_config)
    _ensure_results_dir(args.results_dir, args.overwrite)

    backend = WarpXBackend()
    runner = SimulationRunner(backend)

    runner.configure(
        {
            **backend_config,
            "results_dir": str(args.results_dir),
            "write_sidecar": True,
        }
    )

    horizon_records: List[Dict[str, object]] = []

    for step in range(args.steps):
        outputs = runner.run_step(args.dt)
        state = outputs.state
        horizons = outputs.horizons
        record: Dict[str, object] = {
            "step": step,
            "density_mean": float(np.mean(state.density)) if state.density.size else 0.0,
            "sigma_policy": args.sigma_policy,
        }
        if horizons is not None and horizons.positions.size:
            record.update(
                {
                    "horizon_count": int(horizons.positions.size),
                    "kappa_mean": float(np.mean(horizons.kappa)),
                    "kappa_std": float(np.std(horizons.kappa)),
                }
            )
        else:
            record["horizon_count"] = 0
        horizon_records.append(record)

    if args.save_profiles and state is not None:
        _write_profiles(args.save_profiles, state)

    _write_summary(args.results_dir, horizon_records)
    runner.shutdown()


if __name__ == "__main__":
    main()
