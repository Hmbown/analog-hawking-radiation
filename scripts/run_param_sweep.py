#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_full_pipeline import run_full_pipeline


@dataclass
class SweepResult:
    intensity_values_Wcm2: list[float]
    density_values_cm3: list[float]
    temperature_values: list[float]
    magnetic_field_values: list[float]
    entries: list[dict]
    probability_map: list[list[float]]
    kappa_max_map: list[list[float]]


def run_sweep(
    intensities_Wcm2: Sequence[float],
    densities_cm3: Sequence[float],
    temperatures_K: Sequence[float],
    magnetic_fields_T: Sequence[float],
    grid_points: int = 512,
    progress: bool = False,
    progress_every: int = 50,
) -> SweepResult:
    entries: list[dict] = []
    prob_map: list[list[float]] = []
    kappa_map: list[list[float]] = []
    total_cases = (
        len(intensities_Wcm2) * len(densities_cm3) * len(temperatures_K) * len(magnetic_fields_T)
    )
    done = 0
    for I_cm2 in intensities_Wcm2:
        row_prob: list[float] = []
        row_kappa: list[float] = []
        for n_cm3 in densities_cm3:
            success = 0
            total = 0
            local_kappa_max = 0.0
            for T in temperatures_K:
                for B in magnetic_fields_T:
                    I_SI = float(I_cm2) * 1e4
                    n_SI = float(n_cm3) * 1e6
                    summary = run_full_pipeline(
                        plasma_density=n_SI,
                        laser_intensity=I_SI,
                        temperature_constant=float(T),
                        magnetic_field=float(B),
                        scale_with_intensity=True,
                        grid_points=grid_points,
                        save_graybody_figure=False,
                    )
                    entry = asdict(summary)
                    entry.update(
                        {
                            "input_intensity_Wcm2": float(I_cm2),
                            "input_density_cm3": float(n_cm3),
                            "input_temperature_K": float(T),
                            "input_B_T": float(B),
                        }
                    )
                    entries.append(entry)
                    total += 1
                    done += 1
                    if progress and (done % max(1, int(progress_every)) == 0):
                        print(
                            f"[sweep] {done}/{total_cases} cases \u2014 I={I_cm2:.2e} W/cm^2, n={n_cm3:.2e} cm^-3, T={T:.2e} K, B={B:.2e} T"
                        )
                    if summary.kappa:
                        success += 1
                        local_kappa_max = max(local_kappa_max, float(max(summary.kappa)))
            p = success / total if total > 0 else 0.0
            row_prob.append(p)
            row_kappa.append(local_kappa_max)
        prob_map.append(row_prob)
        kappa_map.append(row_kappa)
    return SweepResult(
        intensity_values_Wcm2=[float(x) for x in intensities_Wcm2],
        density_values_cm3=[float(x) for x in densities_cm3],
        temperature_values=[float(x) for x in temperatures_K],
        magnetic_field_values=[float(x) for x in magnetic_fields_T],
        entries=entries,
        probability_map=prob_map,
        kappa_max_map=kappa_map,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nI", type=int, default=6)
    parser.add_argument("--nN", type=int, default=6)
    parser.add_argument("--nT", type=int, default=4)
    parser.add_argument("--nB", type=int, default=4)
    parser.add_argument("--grid_points", type=int, default=512)
    parser.add_argument("--mode", type=str, choices=["default", "radio"], default="default")
    parser.add_argument(
        "--progress", action="store_true", help="Print periodic progress updates during the sweep"
    )
    parser.add_argument(
        "--progress-every", type=int, default=50, help="How many cases between progress prints"
    )
    args = parser.parse_args()

    if args.mode == "radio":
        I_vals = np.logspace(15, 17, args.nI)
        n_vals = np.logspace(16, 18, args.nN)
        T_vals = np.geomspace(1e4, 1e5, args.nT)
        B_vals = np.linspace(0.0, 0.1, args.nB)
    else:
        I_vals = np.logspace(17, 19, args.nI)
        n_vals = np.logspace(17, 19, args.nN)
        T_vals = np.geomspace(1e4, 1e6, args.nT)
        B_vals = np.linspace(0.0, 0.1, args.nB)

    result = run_sweep(
        I_vals,
        n_vals,
        T_vals,
        B_vals,
        grid_points=args.grid_points,
        progress=bool(args.progress),
        progress_every=int(args.progress_every),
    )
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    extended_path = os.path.join("results", "extended_param_sweep.json")
    payload = {
        "intensity_values_Wcm2": result.intensity_values_Wcm2,
        "density_values_cm3": result.density_values_cm3,
        "temperature_values": result.temperature_values,
        "magnetic_field_values": result.magnetic_field_values,
        "probability_map": result.probability_map,
        "kappa_max_map": result.kappa_max_map,
        "entries": result.entries,
    }
    with open(extended_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved extended sweep to {extended_path}")

    success_cases = []
    for e in result.entries:
        kappa = e.get("kappa", [])
        if kappa:
            e_copy = dict(e)
            e_copy["kappa_max"] = float(max(kappa))
            e_copy["meets_kappa_1e10"] = bool(e_copy["kappa_max"] > 1e10)
            success_cases.append(e_copy)
    success_path = os.path.join("results", "horizon_success_cases.json")
    with open(success_path, "w") as f:
        json.dump(success_cases, f, indent=2)
    print(f"Saved horizon success cases to {success_path}")

    I_plot = np.asarray(result.intensity_values_Wcm2)
    N_plot = np.asarray(result.density_values_cm3)
    P = np.asarray(result.probability_map)
    plt.figure(figsize=(8, 5))
    im = plt.contourf(N_plot, I_plot, P, levels=21, cmap="plasma")
    plt.colorbar(im, label="Horizon formation probability")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Plasma density [cm^-3]")
    plt.ylabel("Laser intensity [W/cm^2]")
    plt.title("Horizon formation probability vs intensity, density")
    prob_fig = os.path.join("figures", "horizon_analysis_probability_map.png")
    plt.tight_layout()
    plt.savefig(prob_fig, dpi=200)
    print(f"Saved {prob_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
