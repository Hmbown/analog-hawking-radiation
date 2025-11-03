#!/usr/bin/env python3
from __future__ import annotations

"""Demo: 2D/3D synthetic horizon detection and κ estimation.

Examples:
  2D sheet:
    python scripts/run_horizon_nd_demo.py --dim 2 --nx 160 --ny 40 --sigma 4e-7 \
      --v0 2.0e6 --cs0 1.0e6 --x0 5e-6

  3D sheet:
    python scripts/run_horizon_nd_demo.py --dim 3 --nx 64 --ny 24 --nz 16 --sigma 6e-7 \
      --v0 1.8e6 --cs0 1.0e6 --x0 5e-6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
import matplotlib

from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_profile(dim: int, nx: int, ny: int, nz: int, x0: float, Lx: float, Ly: float, Lz: float, v0: float, cs0: float, sigma: float):
    if dim == 2:
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        X, Y = np.meshgrid(x, y, indexing='ij')
        vx = v0 * np.tanh((X - x0) / sigma)
        vy = np.zeros_like(vx)
        v = np.stack([vx, vy], axis=-1)
        cs = np.full((nx, ny), cs0, dtype=float)
        return [x, y], v, cs
    elif dim == 3:
        x = np.linspace(0.0, Lx, nx)
        y = np.linspace(0.0, Ly, ny)
        z = np.linspace(0.0, Lz, nz)
        X = x[:, None, None]
        vx = v0 * np.tanh((X - x0) / sigma)
        vx = np.broadcast_to(vx, (nx, ny, nz))
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)
        v = np.stack([vx, vy, vz], axis=-1)
        cs = np.full((nx, ny, nz), cs0, dtype=float)
        return [x, y, z], v, cs
    else:
        raise ValueError("dim must be 2 or 3")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--nx", type=int, default=160)
    p.add_argument("--ny", type=int, default=40)
    p.add_argument("--nz", type=int, default=16)
    p.add_argument("--Lx", type=float, default=10e-6)
    p.add_argument("--Ly", type=float, default=5e-6)
    p.add_argument("--Lz", type=float, default=5e-6)
    p.add_argument("--x0", type=float, default=5e-6)
    p.add_argument("--sigma", type=float, default=4e-7)
    p.add_argument("--v0", type=float, default=2.0e6)
    p.add_argument("--cs0", type=float, default=1.0e6)
    p.add_argument("--scan-axis", type=int, default=0)
    args = p.parse_args()

    grids, v, cs = build_profile(
        args.dim, args.nx, args.ny, args.nz, args.x0, args.Lx, args.Ly, args.Lz, args.v0, args.cs0, args.sigma
    )
    surf = find_horizon_surface_nd(grids, v, cs, scan_axis=args.scan_axis)

    out_dir = Path("results") / "horizon_nd_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "dim": args.dim,
        "nx": args.nx,
        "ny": args.ny,
        "nz": args.nz,
        "n_points": int(surf.positions.shape[0]),
        "kappa_median": float(np.median(surf.kappa)) if surf.kappa.size else 0.0,
        "kappa_mean": float(np.mean(surf.kappa)) if surf.kappa.size else 0.0,
        "kappa_std": float(np.std(surf.kappa)) if surf.kappa.size else 0.0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # Optional 2D plot
    try:
        if args.dim == 2 and surf.positions.shape[0] > 0:
            x, y = grids
            vmag = np.sqrt(np.sum(v**2, axis=-1))
            plt.figure(figsize=(6, 4))
            plt.contourf(x, y, vmag.T, levels=30, cmap="magma")
            plt.colorbar(label="|v| [m/s]")
            plt.contour(x, y, cs.T, levels=10, colors="white", linewidths=0.5, alpha=0.5)
            pts = surf.positions
            plt.plot(pts[:, 0], pts[:, 1], "c.", ms=2, alpha=0.7, label="horizon")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.savefig(out_dir / "horizon_2d.png", dpi=180)
            plt.close()
    except Exception:
        pass

    print(f"Saved horizon nD demo summary to {out_dir}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

