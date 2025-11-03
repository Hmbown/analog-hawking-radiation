#!/usr/bin/env python3
from __future__ import annotations

"""
Convert a 1D openPMD/HDF5 slice to a (x, v, c_s[, B, T_e, n_e]) profile.

Examples:
  python scripts/openpmd_slice_to_profile.py --in sample.h5 \
      --vel-dataset /data/vel --Te-dataset /data/Te --ne-dataset /data/ne \
      --out results/warpx_profile.npz
"""

import argparse
import os
from typing import Optional

import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    raise SystemExit("h5py is required to read openPMD/HDF5 files")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.physics_engine.horizon import sound_speed


def _read_dataset(f: h5py.File, path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    if path not in f:
        return None
    return np.array(f[path])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True, help="Input HDF5 file path")
    p.add_argument("--out", dest="outfile", default="results/warpx_profile.npz")
    p.add_argument("--x-dataset", default=None)
    p.add_argument("--vel-dataset", default=None)
    p.add_argument("--cs-dataset", default=None)
    p.add_argument("--Te-dataset", default=None)
    p.add_argument("--Te-unit", choices=["K", "eV"], default="K", help="Unit for Te dataset")
    p.add_argument("--B-dataset", default=None)
    p.add_argument("--ne-dataset", default=None)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)
    with h5py.File(args.infile, "r") as f:
        x = _read_dataset(f, args.x_dataset) if args.x_dataset else None
        v = _read_dataset(f, args.vel_dataset) if args.vel_dataset else None
        cs = _read_dataset(f, args.cs_dataset) if args.cs_dataset else None
        Te = _read_dataset(f, args.Te_dataset) if args.Te_dataset else None
        B = _read_dataset(f, args.B_dataset) if args.B_dataset else None
        ne = _read_dataset(f, args.ne_dataset) if args.ne_dataset else None

    # Construct c_s from Te if needed (unit-aware)
    if cs is None and Te is not None:
        Te_arr = np.array(Te)
        if args.Te_unit == "eV":
            TeK = Te_arr * 11604.51812  # eV -> K
        else:
            TeK = Te_arr
        cs = sound_speed(TeK)
    if cs is None:
        cs = np.zeros_like(v) if v is not None else np.array([])

    # X-grid fallback
    if x is None:
        n = len(v) if v is not None else len(cs)
        x = np.linspace(0.0, 1.0, n)

    out = {"x": x, "v": v, "c_s": cs}
    if B is not None:
        out["B"] = B
    if Te is not None:
        out["T_e"] = Te
        out["T_e_unit"] = args.Te_unit
    if ne is not None:
        out["n_e"] = ne

    np.savez(args.outfile, **out)
    print(f"Wrote profile to {args.outfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
