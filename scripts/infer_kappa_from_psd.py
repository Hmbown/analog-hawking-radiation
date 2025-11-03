#!/usr/bin/env python3
"""Infer κ from precomputed PSD files using scikit-optimize."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analog_hawking.inference.kappa_mle import infer_kappa as infer_kappa_from_psd
from analog_hawking.inference.kappa_mle import make_graybody_model


def _load_profile(path: Optional[str]) -> Optional[Dict[str, np.ndarray]]:
    if not path:
        return None
    data = np.load(path)
    keys = {"x", "v", "c_s"}
    if not keys.issubset(data.keys()):
        raise ValueError(f"profile file {path} must contain {keys}")
    return {"x": data["x"], "v": data["v"], "c_s": data["c_s"]}


def _load_psd(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    freq_keys = [k for k in ("frequencies", "freq", "f") if k in data]
    psd_keys = [k for k in ("power_spectrum", "psd", "spectrum") if k in data]
    if not freq_keys or not psd_keys:
        raise ValueError(f"{path} must contain 'frequencies' and 'power_spectrum' arrays")
    freqs = np.asarray(data[freq_keys[0]], dtype=float)
    psd = np.asarray(data[psd_keys[0]], dtype=float)
    return freqs, psd


def main() -> int:
    parser = argparse.ArgumentParser(description="Infer κ from PSD files.")
    parser.add_argument(
        "psd_glob",
        nargs="+",
        help="NPZ files or glob patterns with frequencies/power_spectrum arrays",
    )
    parser.add_argument(
        "--graybody-profile", help="Optional NPZ with x,v,c_s for graybody modelling"
    )
    parser.add_argument(
        "--graybody-method",
        default="dimensionless",
        choices=["dimensionless", "wkb", "acoustic_wkb"],
    )
    parser.add_argument("--alpha-gray", type=float, default=1.0)
    parser.add_argument(
        "--bounds",
        type=str,
        default="1e4,1e12",
        help="κ bounds as min,max in SI (default: 1e4,1e12)",
    )
    parser.add_argument("--calls", type=int, default=40, help="Number of optimizer evaluations")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/kappa_inference"),
        help="Directory for inference outputs",
    )
    parser.add_argument("--emit-json", action="store_true", help="Print JSON summary to stdout")
    args = parser.parse_args()

    psd_paths: Iterable[str] = []
    for pattern in args.psd_glob:
        expanded = glob.glob(pattern)
        if expanded:
            psd_paths = list(psd_paths) + expanded
        elif os.path.exists(pattern):
            psd_paths = list(psd_paths) + [pattern]
        else:
            raise FileNotFoundError(f"No PSD files matched pattern '{pattern}'")

    bounds_tokens = args.bounds.split(",")
    if len(bounds_tokens) != 2:
        raise ValueError("--bounds must be provided as min,max")
    bounds = (float(bounds_tokens[0]), float(bounds_tokens[1]))
    if bounds[0] <= 0 or bounds[1] <= bounds[0]:
        raise ValueError("invalid κ bounds")

    profile = _load_profile(args.graybody_profile)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for psd_path in psd_paths:
        path = Path(psd_path)
        freqs, psd = _load_psd(path)
        model = make_graybody_model(
            freqs,
            graybody_profile=profile,
            graybody_method=args.graybody_method,
            alpha_gray=args.alpha_gray,
            emitting_area_m2=1e-6,
            solid_angle_sr=5e-2,
            coupling_efficiency=0.1,
        )
        inference = infer_kappa_from_psd(
            freqs,
            psd,
            model,
            bounds=bounds,
            n_calls=args.calls,
        )
        stem = path.stem
        posterior_npz = args.out_dir / f"{stem}_posterior.npz"
        np.savez(
            posterior_npz,
            kappa_grid=inference.posterior_grid,
            posterior_density=inference.posterior_density,
            trace=np.asarray(inference.trace, dtype=float),
        )
        summary = {
            "source": str(path),
            "kappa_hat": inference.kappa_hat,
            "kappa_err": inference.kappa_err,
            "credible_interval": inference.credible_interval,
            "bounds": bounds,
        }
        meta_path = args.out_dir / f"{stem}_summary.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(
            f"{path}: κ̂={inference.kappa_hat:.4e} ± {inference.kappa_err:.2e} (95% CI ~ {inference.credible_interval[0]:.4e}-{inference.credible_interval[1]:.4e})"
        )
        summaries.append(summary)

    if args.emit_json:
        print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
