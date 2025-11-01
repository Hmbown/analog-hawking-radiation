#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.analysis.gradient_sweep import generate_catastrophe_plots, run_sweep


def main() -> int:
    p = argparse.ArgumentParser(description="Stratified, range-focused gradient sweep")
    p.add_argument("--out", dest="output", type=str, required=True)
    p.add_argument("--a0-min", type=float, required=True)
    p.add_argument("--a0-max", type=float, required=True)
    p.add_argument("--ne-min", type=float, required=True)
    p.add_argument("--ne-max", type=float, required=True)
    p.add_argument("--grad-min", type=float, required=True)
    p.add_argument("--grad-max", type=float, required=True)
    p.add_argument("--n-per-axis", type=int, default=10)
    p.add_argument("--thresholds", type=str, default=None)
    p.add_argument("--vmax-frac", type=float, default=None)
    p.add_argument("--dvdx-max", type=float, default=None)
    p.add_argument("--intensity-max", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)

    args = p.parse_args()

    sweep_data = run_sweep(
        n_samples=args.n_per_axis**3,
        output_dir=args.output,
        thresholds_path=args.thresholds,
        vmax_frac=args.vmax_frac,
        dvdx_max=args.dvdx_max,
        intensity_max=args.intensity_max,
        a0_min=args.a0_min,
        a0_max=args.a0_max,
        ne_min=args.ne_min,
        ne_max=args.ne_max,
        grad_min=args.grad_min,
        grad_max=args.grad_max,
        n_per_axis=args.n_per_axis,
        seed=args.seed,
        stratified=True,
    )

    generate_catastrophe_plots(sweep_data, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

