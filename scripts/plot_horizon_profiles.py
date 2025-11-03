#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend


def plot_profiles_for_case(case: dict, idx: int) -> str:
    grid_points = int(case.get("grid_points", 512))
    grid = np.linspace(0.0, 50e-6, grid_points)

    backend = FluidBackend()
    cfg = {
        "plasma_density": float(case["plasma_density"]),
        "laser_wavelength": float(case.get("laser_wavelength", 800e-9)),
        "laser_intensity": float(case["laser_intensity"]),
        "grid": grid,
        "temperature_settings": {"constant": float(case.get("temperature_constant", 5e5))},
        "use_fast_magnetosonic": bool(case.get("use_fast_magnetosonic", True)),
        "scale_with_intensity": True,
    }
    if case.get("magnetic_field") is not None:
        cfg["magnetic_field"] = float(case["magnetic_field"])
    backend.configure(cfg)

    state = backend.step(0.0)

    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)

    plt.figure(figsize=(8, 4))
    plt.plot(state.grid * 1e6, state.velocity, label="|v(x)|", lw=2)
    plt.plot(state.grid * 1e6, state.sound_speed, label="c_s(x)", lw=2)

    for xh in horizons.positions:
        plt.axvline(x=xh * 1e6, color="k", ls="--", alpha=0.5)

    plt.xlabel("x [µm]")
    plt.ylabel("Speed [m/s]")
    plt.title("Velocity and sound speed with horizons")
    plt.legend()
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    out = Path("figures") / f"horizon_analysis_profile_{idx:02d}.png"
    plt.savefig(out, dpi=200)
    return str(out)


def main() -> int:
    cases_path = Path("results") / "horizon_success_cases.json"
    if not cases_path.exists():
        print(f"No success cases found at {cases_path}. Run run_param_sweep.py first.")
        return 1
    with open(cases_path, "r") as f:
        cases = json.load(f)

    # Sort by kappa_max descending and take top 3
    cases = sorted(cases, key=lambda c: float(c.get("kappa_max", 0.0)), reverse=True)
    top_cases = cases[:3]

    outputs = []
    for i, case in enumerate(top_cases):
        out = plot_profiles_for_case(case, i)
        outputs.append(out)
        print(f"Saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
