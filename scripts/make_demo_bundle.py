#!/usr/bin/env python3
from __future__ import annotations

"""
Create a minimal demo dataset bundle for reproducible runs.

Outputs directory: data/demo_bundle/
 - slice.h5           (openPMD-like HDF5 slice with /x, /vel, /Te)
 - profile.npz        (NPZ profile with x, v, c_s)
 - README.md          (how to run)
 - MANIFEST.txt       (file list with sizes and SHA256)
"""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTDIR = ROOT / 'data' / 'demo_bundle'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def ensure_dirs() -> None:
    os.makedirs(OUTDIR, exist_ok=True)


def generate_slice(path: Path) -> None:
    # Use the existing helper to write an HDF5 slice
    cmd = [sys.executable, str(ROOT / 'scripts' / 'generate_synthetic_openpmd_slice.py')]
    subprocess.check_call(cmd)
    # Move generated file into bundle
    src = ROOT / 'results' / 'synthetic_slice.h5'
    path.write_bytes(src.read_bytes())


def generate_profile(h5_path: Path, npz_path: Path) -> None:
    cmd = [
        sys.executable, str(ROOT / 'scripts' / 'openpmd_slice_to_profile.py'),
        '--in', str(h5_path), '--x-dataset', '/x', '--vel-dataset', '/vel', '--Te-dataset', '/Te', '--out', str(npz_path)
    ]
    subprocess.check_call(cmd)


def write_readme(path: Path) -> None:
    text = (
        "# Demo Bundle\n\n"
        "This folder contains a minimal, reproducible dataset for the PIC-to-pipeline path.\n\n"
        "Files:\n\n"
        "- `slice.h5`: synthetic openPMD-like HDF5 slice with `/x`, `/vel`, `/Te`\n"
        "- `profile.npz`: profile built from the slice with arrays `x`, `v`, `c_s`\n\n"
        "Reproduce end-to-end metrics (dimensionless graybody):\n\n"
        "```bash\n"
        "python scripts/run_pic_pipeline.py --profile data/demo_bundle/profile.npz --graybody dimensionless\n"
        "```\n\n"
        "Or use the acoustic-WKB graybody and exact acoustic κ:\n\n"
        "```bash\n"
        "python scripts/run_pic_pipeline.py --profile data/demo_bundle/profile.npz --graybody acoustic_wkb --kappa-method acoustic_exact\n"
        "```\n"
    )
    path.write_text(text)


def write_manifest(paths: list[Path]) -> None:
    lines = []
    for p in paths:
        lines.append(f"{p.name}\t{p.stat().st_size} B\tSHA256={sha256(p)}")
    (OUTDIR / 'MANIFEST.txt').write_text("\n".join(lines) + "\n")


def main() -> int:
    ensure_dirs()
    slice_path = OUTDIR / 'slice.h5'
    prof_path = OUTDIR / 'profile.npz'
    generate_slice(slice_path)
    generate_profile(slice_path, prof_path)
    write_readme(OUTDIR / 'README.md')
    write_manifest([slice_path, prof_path])
    print(f"Demo bundle created at {OUTDIR}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

