#!/usr/bin/env python3
from __future__ import annotations

"""Generate a small synthetic 2D/3D HDF5 file with (x[,y[,z]], vx,vy[,vz], c_s).

Usage:
  python scripts/generate_synthetic_nd_h5.py --out results/synthetic_nd.h5 --dim 2
"""

import argparse
import os
from pathlib import Path
import numpy as np

try:
    import h5py  # type: ignore
except Exception as exc:
    raise SystemExit("h5py is required to create synthetic HDF5 files")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--dim", type=int, choices=[2, 3], default=2)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--ny", type=int, default=48)
    p.add_argument("--nz", type=int, default=16)
    p.add_argument("--Lx", type=float, default=10e-6)
    p.add_argument("--Ly", type=float, default=5e-6)
    p.add_argument("--Lz", type=float, default=5e-6)
    p.add_argument("--x0", type=float, default=5e-6)
    p.add_argument("--v0", type=float, default=2.0e6)
    p.add_argument("--cs0", type=float, default=1.0e6)
    p.add_argument("--sigma", type=float, default=4e-7)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    x = np.linspace(0.0, args.Lx, args.nx)
    y = np.linspace(0.0, args.Ly, args.ny)
    z = np.linspace(0.0, args.Lz, args.nz)

    if args.dim == 2:
        X, Y = np.meshgrid(x, y, indexing='ij')
        vx = args.v0 * np.tanh((X - args.x0) / args.sigma)
        vy = np.zeros_like(vx)
        cs = np.full((args.nx, args.ny), args.cs0, dtype=float)
        with h5py.File(args.out, 'w') as f:
            f.create_dataset('/mesh/x', data=x)
            f.create_dataset('/mesh/y', data=y)
            f.create_dataset('/fields/vx', data=vx)
            f.create_dataset('/fields/vy', data=vy)
            f.create_dataset('/fields/c_s', data=cs)
    else:
        X = x[:, None, None]
        vx = args.v0 * np.tanh((X - args.x0) / args.sigma)
        vx = np.broadcast_to(vx, (args.nx, args.ny, args.nz))
        vy = np.zeros_like(vx)
        vz = np.zeros_like(vx)
        cs = np.full((args.nx, args.ny, args.nz), args.cs0, dtype=float)
        with h5py.File(args.out, 'w') as f:
            f.create_dataset('/mesh/x', data=x)
            f.create_dataset('/mesh/y', data=y)
            f.create_dataset('/mesh/z', data=z)
            f.create_dataset('/fields/vx', data=vx)
            f.create_dataset('/fields/vy', data=vy)
            f.create_dataset('/fields/vz', data=vz)
            f.create_dataset('/fields/c_s', data=cs)

    print(f"Wrote synthetic nD file to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

