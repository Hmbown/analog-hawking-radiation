#!/usr/bin/env python3
"""Scan magnetic field strengths to evaluate horizon diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from physics_engine.plasma_models.warpx_backend import WarpXBackend
from physics_engine.simulation import SimulationRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B-field horizon scan")
    parser.add_argument("--config", type=Path, required=True, help="Base WarpX configuration JSON")
    parser.add_argument(
        "--b-fields",
        type=str,
        default="0.0,0.1,0.5,1.0",
        help="Comma-separated magnetic field strengths (Tesla)",
    )
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--results", type=Path, default=Path("results/B_scan_horizons.json"))
    return parser.parse_args()


def _load_config(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = _parse_args()
    base_config = _load_config(args.config)
    b_values = [float(val) for val in args.b_fields.split(",")]

    records: List[Dict[str, object]] = []
    for b_field in b_values:
        config = dict(base_config)
        config.setdefault("magnetic_field", {})["magnitude"] = b_field
        backend = WarpXBackend()
        runner = SimulationRunner(backend)
        runner.configure(config)
        for step in range(args.steps):
            outputs = runner.run_step(0.0)
            horizons = outputs.horizons
            record = {
                "B": b_field,
                "step": step,
                "horizon_count": 0,
            }
            if horizons is not None and horizons.positions.size:
                record.update(
                    {
                        "horizon_count": int(horizons.positions.size),
                        "kappa_mean": float(np.mean(horizons.kappa)),
                        "kappa_std": float(np.std(horizons.kappa)),
                    }
                )
            records.append(record)
        runner.shutdown()

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
