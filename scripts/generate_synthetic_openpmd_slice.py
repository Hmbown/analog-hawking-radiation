#!/usr/bin/env python3
from __future__ import annotations

"""
Generate a small synthetic openPMD-like HDF5 slice with datasets suitable
for testing the PIC ingestion pipeline:

Creates: results/synthetic_slice.h5 with datasets /x, /vel, /Te
"""

import os
from pathlib import Path

import numpy as np

try:
    import h5py  # type: ignore
except Exception:
    raise SystemExit("h5py is required to write HDF5 files; pip install h5py")


def main() -> int:
    os.makedirs("results", exist_ok=True)
    path = Path("results/synthetic_slice.h5")
    N = 256
    x = np.linspace(-1.0, 1.0, N)
    v = 1.0 * x  # linear flow
    Te = 1.0e4 * np.ones_like(x)  # 1e4 K constant
    with h5py.File(path, "w") as f:
        f.create_dataset("/x", data=x)
        f.create_dataset("/vel", data=v)
        f.create_dataset("/Te", data=Te)
    print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
