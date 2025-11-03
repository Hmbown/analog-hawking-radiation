#!/usr/bin/env python3
"""Compute horizon-crossing g^(2) correlation maps from openPMD diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analog_hawking.pipelines import OpenPMDAdapterResult, from_openpmd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


def _select_observable(profile: OpenPMDAdapterResult, name: str) -> np.ndarray:
    name = name.lower()
    if name == "density" and profile.density is not None:
        return profile.density
    if name == "velocity":
        return profile.velocity
    if name == "sound_speed":
        return profile.sound_speed
    raise ValueError(f"Observable '{name}' not available in profile (choose density/velocity/sound_speed)")


def _sample_iterations(center: int, window: int) -> List[int]:
    if window <= 0:
        return [center]
    indices = [center]
    for offset in range(1, window + 1):
        if center - offset >= 0:
            indices.append(center - offset)
        indices.append(center + offset)
    return sorted(set(indices))


def _build_reference_grid(extent: float, cells: int) -> np.ndarray:
    return np.linspace(-extent, extent, cells)


def _align_to_reference(
    x_rel: np.ndarray,
    values: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return np.interp(reference, x_rel, values, left=values[0], right=values[-1])


def _compute_correlation(samples: np.ndarray) -> np.ndarray:
    delta = samples - np.mean(samples, axis=0, keepdims=True)
    return delta.T @ delta / max(delta.shape[0], 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate horizon-crossing g^2 correlation maps.")
    parser.add_argument("--series", required=True, help="Path to openPMD series (directory or pattern).")
    parser.add_argument("--t-index", type=str, default="latest", help="Iteration index or 'latest'.")
    parser.add_argument("--window", type=int, default=2, help="Number of iterations on each side of t-index.")
    parser.add_argument("--observable", choices=["density", "velocity", "sound_speed"], default="density")
    parser.add_argument("--extent", type=float, default=5e-5, help="Spatial extent around horizon (meters).")
    parser.add_argument("--cells", type=int, default=128, help="Number of spatial cells in the aligned window.")
    parser.add_argument("--out", type=Path, default=Path("results/correlation/g2_horizon.npz"), help="Output NPZ path.")
    parser.add_argument("--figure", type=Path, default=Path("figures/g2_horizon_map.png"), help="Optional PNG output.")
    parser.add_argument("--save-metadata", action="store_true", help="Write JSON sidecar with metadata.")
    args = parser.parse_args()

    try:
        series = from_openpmd(args.series, t=args.t_index)
    except Exception as exc:
        raise SystemExit(f"Failed to load iteration '{args.t_index}': {exc}") from exc

    if series.horizon is None or series.horizon.positions.size == 0:
        raise SystemExit("No horizon detected at the requested iteration.")

    grid_ref = _build_reference_grid(args.extent, args.cells)
    iterations = []
    observables = []
    horizons = []

    # Determine iteration indices available by inspecting metadata generated earlier
    # For series files, we attempt sequential offsets around requested index.
    center = series.metadata.get("iteration", 0)
    sample_indices = _sample_iterations(center, args.window)

    for idx in sample_indices:
        try:
            profile = series if idx == center else from_openpmd(args.series, t=idx)
        except Exception:
            continue
        if profile.horizon is None or profile.horizon.positions.size == 0:
            continue
        observable = _select_observable(profile, args.observable)
        h_pos = float(profile.horizon.positions[0])
        x_rel = profile.grid - h_pos
        aligned = _align_to_reference(x_rel, observable, grid_ref)
        iterations.append(idx)
        observables.append(aligned)
        horizons.append(
            {
                "iteration": idx,
                "position": h_pos,
                "kappa": float(profile.horizon.kappa[0]) if profile.horizon.kappa.size else None,
            }
        )

    if not observables:
        raise SystemExit("No usable iterations found for correlation calculation.")

    samples = np.vstack(observables)
    g2 = _compute_correlation(samples)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        grid=grid_ref,
        g2=g2,
        iterations=np.asarray(iterations, dtype=int),
        observable=args.observable,
    )

    if args.save_metadata:
        meta_path = args.out.with_suffix(".json")
        meta = {
            "series": args.series,
            "iterations": iterations,
            "horizons": horizons,
            "extent": args.extent,
            "cells": args.cells,
            "observable": args.observable,
        }
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    if plt is not None and args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        extent_plot = [grid_ref[0] * 1e6, grid_ref[-1] * 1e6, grid_ref[0] * 1e6, grid_ref[-1] * 1e6]
        im = plt.imshow(
            g2,
            origin="lower",
            extent=extent_plot,
            cmap="magma",
        )
        plt.colorbar(im, label=r"$g^{(2)}(x_1, x_2)$")
        plt.xlabel(r"$x_1 - x_H$ [$\mu$m]")
        plt.ylabel(r"$x_2 - x_H$ [$\mu$m]")
        plt.title("Horizon-Crossing Correlation Map")
        plt.tight_layout()
        plt.savefig(args.figure, dpi=220)
        plt.close()

    print(f"Saved correlation map to {args.out}")
    if args.figure and plt is not None:
        print(f"Figure written to {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
