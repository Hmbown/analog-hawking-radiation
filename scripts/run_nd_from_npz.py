#!/usr/bin/env python3
from __future__ import annotations

"""Run nD horizon detection from a grid NPZ (produced by openpmd_to_grid_nd.py).

Example:
  python scripts/run_nd_from_npz.py results/grid_nd_profile.npz --scan-axis 0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("npz_path", type=str)
    p.add_argument("--scan-axis", type=int, default=0)
    args = p.parse_args()

    data = np.load(args.npz_path)
    grids = []
    for key in ("x", "y", "z"):
        if key in data:
            grids.append(np.array(data[key]))
    if not grids:
        raise SystemExit("NPZ must include at least 'x' coordinate")
    comps = []
    for comp in ("vx", "vy", "vz"):
        if comp in data:
            comps.append(np.array(data[comp]))
    if not comps:
        raise SystemExit("NPZ must include at least 'vx'")
    v_field = np.stack(comps, axis=-1)
    if "c_s" not in data:
        raise SystemExit("NPZ must include 'c_s'")
    cs = np.array(data["c_s"])

    surf = find_horizon_surface_nd(grids, v_field, cs, scan_axis=args.scan_axis)
    out_dir = Path("results") / "horizon_nd"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "n_points": int(surf.positions.shape[0]),
        "kappa_median": float(np.median(surf.kappa)) if surf.kappa.size else 0.0,
        "kappa_mean": float(np.mean(surf.kappa)) if surf.kappa.size else 0.0,
        "kappa_std": float(np.std(surf.kappa)) if surf.kappa.size else 0.0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"Saved nD horizon summary to {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

