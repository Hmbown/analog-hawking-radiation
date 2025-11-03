#!/usr/bin/env python3
from __future__ import annotations

"""Convert OpenPMD/HDF5 datasets to an nD grid NPZ for horizon_nd.

Example:
  python scripts/openpmd_to_grid_nd.py --in sample.h5 \
    --x /mesh/x --y /mesh/y --z /mesh/z \
    --vx /fields/vx --vy /fields/vy --vz /fields/vz \
    --cs /fields/c_s --out results/grid_nd_profile.npz
"""

import argparse
import os
from typing import Optional

import numpy as np

try:
    import h5py  # type: ignore
except Exception:
    raise SystemExit("h5py is required to read OpenPMD/HDF5 files")


def _read(f: h5py.File, path: Optional[str]) -> Optional[np.ndarray]:
    if path is None:
        return None
    if path not in f:
        return None
    return np.array(f[path])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", default="results/grid_nd_profile.npz")
    p.add_argument("--x", default=None)
    p.add_argument("--y", default=None)
    p.add_argument("--z", default=None)
    p.add_argument("--vx", default=None)
    p.add_argument("--vy", default=None)
    p.add_argument("--vz", default=None)
    p.add_argument("--cs", default=None)
    p.add_argument("--Te", default=None)
    p.add_argument("--Te-unit", choices=["K", "eV"], default="K")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with h5py.File(args.infile, 'r') as f:
        x = _read(f, args.x)
        y = _read(f, args.y)
        z = _read(f, args.z)
        vx = _read(f, args.vx)
        vy = _read(f, args.vy)
        vz = _read(f, args.vz)
        cs = _read(f, args.cs)
        Te = _read(f, args.Te)

    if cs is None and Te is not None:
        # Avoid import cycles; compute rough c_s from Te (Kelvin)
        from analog_hawking.physics_engine.horizon import sound_speed
        Te_arr = np.array(Te)
        if args.Te_unit == "eV":
            Te_arr = Te_arr * 11604.51812
        cs = sound_speed(Te_arr)

    out = {}
    if x is not None:
        out["x"] = x
    if y is not None:
        out["y"] = y
    if z is not None:
        out["z"] = z
    if vx is not None:
        out["vx"] = vx
    if vy is not None:
        out["vy"] = vy
    if vz is not None:
        out["vz"] = vz
    if cs is not None:
        out["c_s"] = cs

    np.savez(args.outfile, **out)
    print(f"Wrote {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

