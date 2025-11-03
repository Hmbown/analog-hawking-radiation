#!/usr/bin/env python3
from __future__ import annotations

"""Sweep threshold parameters and record κ_max sensitivity.

This script varies breakdown thresholds (e.g., v_max_fraction_c, dv_dx_max_s)
and runs a reduced gradient sweep to estimate how κ_max changes.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.analysis.gradient_sweep import run_sweep  # type: ignore


@dataclass
class ThresholdConfig:
    v_max_fraction_c: float
    dv_dx_max_s: float
    intensity_max_W_m2: float


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=60)
    p.add_argument("--out", type=str, default="results/threshold_sensitivity.json")
    p.add_argument("--v-fracs", type=str, default="0.4,0.5,0.6")
    p.add_argument("--dv-max", type=str, default="2e12,4e12,8e12")
    p.add_argument("--intensity-max", type=float, default=1.0e24)
    args = p.parse_args()

    v_fracs = [float(s) for s in args.v_fracs.split(",") if s]
    dv_vals = [float(s) for s in args.dv_max.split(",") if s]

    results = []
    for vf in v_fracs:
        for dv in dv_vals:
            thr = ThresholdConfig(
                v_max_fraction_c=vf, dv_dx_max_s=dv, intensity_max_W_m2=float(args.intensity_max)
            )
            sweep = run_sweep(
                n_samples=int(args.n_samples),
                output_dir="results/threshold_sweeps",
                vmax_frac=thr.v_max_fraction_c,
                dvdx_max=thr.dv_dx_max_s,
                intensity_max=thr.intensity_max_W_m2,
                seed=12345,
            )
            kmax = float(sweep.get("analysis", {}).get("max_kappa", np.nan))
            results.append(
                {
                    "v_max_fraction_c": vf,
                    "dv_dx_max_s": dv,
                    "intensity_max_W_m2": float(args.intensity_max),
                    "kappa_max": kmax,
                }
            )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"sensitivity": results}, fh, indent=2)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
