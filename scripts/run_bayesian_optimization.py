#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

# Ensure package imports work if editable install is not active
import sys
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.physics_engine.optimization.merit_function import GlowMeritFunction


def merit_from_params(
    plasma_density_m3: float,
    laser_intensity_Wm2: float,
    T_peak_K: float,
    T_background_K: float = 2e6,
    n_samples: int = 200,
) -> float:
    base_params = {
        "plasma_density": float(plasma_density_m3),
        "laser_intensity": float(laser_intensity_Wm2),
        "T_peak": float(T_peak_K),
        "T_background": float(T_background_K),
    }
    # 10% uncertainties
    param_uncertainties = {
        k: 0.1 * v for k, v in base_params.items() if isinstance(v, (int, float))
    }

    snr_config = {
        "system_temperature": 30.0,  # K
        "bandwidth": 1e8,  # 100 MHz
        "integration_time": 3600.0,  # 1 hr
    }

    # Spatial-temporal context for merit calculation
    x_grid = np.linspace(-50e-6, 50e-6, 500)
    t0 = 50e-15

    merit_func = GlowMeritFunction(
        base_params=base_params,
        param_uncertainties=param_uncertainties,
        snr_config=snr_config,
        n_samples=n_samples,
    )
    merit = float(merit_func.calculate_merit(x_grid, t0))
    return merit


def make_log_prob(bounds_SI: dict[str, tuple[float, float]]):
    # emcee log-prob function over params vector [density_m3, intensity_Wm2, T_peak_K]
    def log_prob(theta: np.ndarray) -> float:
        n_m3, I_Wm2, T_K = [float(x) for x in theta]
        (n_lo, n_hi) = bounds_SI["plasma_density"]
        (I_lo, I_hi) = bounds_SI["laser_intensity"]
        (T_lo, T_hi) = bounds_SI["T_peak"]
        # Bounds check
        if not (n_lo < n_m3 < n_hi and I_lo < I_Wm2 < I_hi and T_lo < T_K < T_hi):
            return -np.inf
        merit = merit_from_params(n_m3, I_Wm2, T_K)
        # Use merit as a pseudo log-probability as in existing scripts
        return merit

    return log_prob


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--steps", type=int, default=100, help="Number of MCMC steps (50-100 recommended)"
    )
    parser.add_argument("--walkers", type=int, default=24, help="Number of MCMC walkers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--samples", type=int, default=200, help="Monte Carlo samples in merit evaluation"
    )
    # Ranges in lab units for convenience; converted internally to SI
    parser.add_argument("--I_min_Wcm2", type=float, default=1e17)
    parser.add_argument("--I_max_Wcm2", type=float, default=1e19)
    parser.add_argument("--n_min_cm3", type=float, default=1e17)
    parser.add_argument("--n_max_cm3", type=float, default=1e19)
    parser.add_argument("--T_min_K", type=float, default=1e4)
    parser.add_argument("--T_max_K", type=float, default=1e6)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # Convert bounds to SI used by the internal models
    bounds_SI = {
        "plasma_density": (args.n_min_cm3 * 1e6, args.n_max_cm3 * 1e6),  # cm^-3 -> m^-3
        "laser_intensity": (args.I_min_Wcm2 * 1e4, args.I_max_Wcm2 * 1e4),  # W/cm^2 -> W/m^2
        "T_peak": (args.T_min_K, args.T_max_K),
    }

    # Initialize walkers around midpoints
    n0 = np.sqrt(bounds_SI["plasma_density"][0] * bounds_SI["plasma_density"][1])
    I0 = np.sqrt(bounds_SI["laser_intensity"][0] * bounds_SI["laser_intensity"][1])
    T0 = np.sqrt(bounds_SI["T_peak"][0] * bounds_SI["T_peak"][1])
    p0_center = np.array([n0, I0, T0], dtype=float)
    p0 = p0_center[None, :] * (1.0 + 1e-3 * rng.standard_normal(size=(args.walkers, 3)))

    sampler = emcee.EnsembleSampler(args.walkers, 3, make_log_prob(bounds_SI))
    print(f"Running Bayesian-style optimization with {args.walkers} walkers, {args.steps} steps...")
    sampler.run_mcmc(p0, args.steps, progress=True)

    # Extract trace and best-so-far per iteration
    chain = sampler.get_chain()  # shape (steps, walkers, 3)
    logp = sampler.get_log_prob()  # shape (steps, walkers)

    best_merit_by_step = np.max(logp, axis=1)
    best_idx_by_step = np.argmax(logp, axis=1)
    best_params_by_step = chain[np.arange(chain.shape[0]), best_idx_by_step, :]

    # Save trace
    os.makedirs("results", exist_ok=True)
    trace_out = Path("results") / "bayesian_optimization_trace.json"
    trace_records = []
    for s in range(chain.shape[0]):
        for w in range(chain.shape[1]):
            n_m3, I_Wm2, T_K = [float(x) for x in chain[s, w, :]]
            trace_records.append(
                {
                    "step": int(s),
                    "walker": int(w),
                    "plasma_density_m3": n_m3,
                    "laser_intensity_Wm2": I_Wm2,
                    "T_peak_K": T_K,
                    "log_merit": float(logp[s, w]),
                    "plasma_density_cm3": n_m3 / 1e6,
                    "laser_intensity_Wcm2": I_Wm2 / 1e4,
                }
            )
    with open(trace_out, "w") as f:
        json.dump(trace_records, f, indent=2)
    print(f"Saved optimization trace to {trace_out}")

    # Convergence plot
    os.makedirs("figures", exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(best_merit_by_step)), best_merit_by_step, lw=2)
    plt.xlabel("Iteration (step)")
    plt.ylabel("Best merit so far (pseudo log-prob)")
    plt.title("Bayesian Optimization Convergence")
    plt.tight_layout()
    fig_out = Path("figures") / "horizon_analysis_bo_convergence.png"
    plt.savefig(fig_out, dpi=200)
    print(f"Saved {fig_out}")

    # Report final best parameters in lab units
    n_best, I_best, T_best = [float(x) for x in best_params_by_step[-1]]
    print("Best-found parameters (lab units):")
    print(f"  plasma_density: {n_best/1e6:.3e} cm^-3")
    print(f"  laser_intensity: {I_best/1e4:.3e} W/cm^2")
    print(f"  T_peak: {T_best:.3e} K")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
