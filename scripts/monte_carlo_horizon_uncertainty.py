#!/usr/bin/env python3
from __future__ import annotations

"""
Monte Carlo horizon uncertainty: sample around nominal (n_e, T, B) and
estimate horizon formation probability and kappa statistics.

Outputs
- results/horizon_probability_bands.json
- figures/horizon_probability_bands.png
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend


@dataclass
class MCConfig:
    n_samples: int = 200
    density_mean: float = 5e17
    density_spread_frac: float = 0.2  # 20% std (log-normal)
    temperature_mean: float = 5e5
    temperature_spread_frac: float = 0.3
    magnetic_field: Optional[float] = 0.01
    wavelength: float = 800e-9
    intensity: float = 5e16
    use_fast_magnetosonic: bool = True
    scale_with_intensity: bool = True
    grid_min: float = 0.0
    grid_max: float = 50e-6
    grid_points: int = 512


def _lognormal_samples(mean: float, frac_std: float, n: int) -> np.ndarray:
    if frac_std <= 0:
        return np.full(n, mean, dtype=float)
    sigma = np.sqrt(np.log(1.0 + frac_std**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma, size=n)


def run_mc(cfg: MCConfig):
    rng = np.random.default_rng()
    N = cfg.n_samples

    densities = _lognormal_samples(cfg.density_mean, cfg.density_spread_frac, N)
    temperatures = _lognormal_samples(cfg.temperature_mean, cfg.temperature_spread_frac, N)

    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points)

    horizon_flags = np.zeros(N, dtype=bool)
    kappas = np.full(N, np.nan, dtype=float)

    backend = FluidBackend()

    for i in range(N):
        backend.configure(
            {
                "plasma_density": float(densities[i]),
                "laser_wavelength": cfg.wavelength,
                "laser_intensity": cfg.intensity,
                "grid": grid,
                "temperature_settings": {"constant": float(temperatures[i])},
                "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
                "scale_with_intensity": bool(cfg.scale_with_intensity),
                "magnetic_field": cfg.magnetic_field,
            }
        )
        state = backend.step(0.0)
        hz = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)
        if hz.positions.size:
            horizon_flags[i] = True
            kappas[i] = float(hz.kappa[0])

    prob = float(np.mean(horizon_flags))
    kappa_valid = kappas[~np.isnan(kappas)]
    kappa_mean = float(np.mean(kappa_valid)) if kappa_valid.size else 0.0
    kappa_std = float(np.std(kappa_valid)) if kappa_valid.size else 0.0

    os.makedirs("results", exist_ok=True)
    out = {
        "config": asdict(cfg),
        "horizon_probability": prob,
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
    }
    with open("results/horizon_probability_bands.json", "w") as f:
        json.dump(out, f, indent=2)

    # Figure: scatter of samples colored by horizon outcome; annotate probability
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(6, 4))
    sc0 = plt.scatter(
        densities[~horizon_flags],
        temperatures[~horizon_flags],
        s=10,
        c="#bbbbbb",
        label="no horizon",
    )
    sc1 = plt.scatter(
        densities[horizon_flags], temperatures[horizon_flags], s=10, c="#d62728", label="horizon"
    )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("density n_e")
    plt.ylabel("temperature T")
    plt.title(
        f"Horizon probability ≈ {prob:.2f}; κ_mean={kappa_mean:.2e} ± {kappa_std:.2e} s$^{{-1}}$"
    )
    plt.legend(markerscale=2)
    plt.tight_layout()
    plt.savefig("figures/horizon_probability_bands.png", dpi=200)
    plt.close()

    print("Saved results/horizon_probability_bands.json and figures/horizon_probability_bands.png")


if __name__ == "__main__":
    raise SystemExit(run_mc(MCConfig()))
