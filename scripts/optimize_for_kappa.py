"""Bayesian optimization targeting maximal horizon gradient (kappa)."""

import os
import sys

import emcee
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.optimization.probabilistic_horizon import ProbabilisticHorizonModel

param_space = {
    "plasma_density": (1e23, 5e24),
    "laser_intensity": (1e22, 1e23),
    "T_peak": (5e6, 20e6),
}


def log_prob(params):
    plasma_density, laser_intensity, T_peak = params

    if not (
        param_space["plasma_density"][0] < plasma_density < param_space["plasma_density"][1]
        and param_space["laser_intensity"][0] < laser_intensity < param_space["laser_intensity"][1]
        and param_space["T_peak"][0] < T_peak < param_space["T_peak"][1]
    ):
        return -np.inf

    base_params = {
        "plasma_density": plasma_density,
        "laser_intensity": laser_intensity,
        "T_peak": T_peak,
        "T_background": 2e6,
    }
    param_uncertainties = {key: 0.1 * val for key, val in base_params.items()}

    horizon_model = ProbabilisticHorizonModel(
        base_params=base_params,
        param_uncertainties=param_uncertainties,
        n_samples=200,
    )

    x_grid = np.linspace(-50e-6, 50e-6, 500)
    t0 = 50e-15
    _, kappa_stats = horizon_model.calculate_horizon_probability(x_grid, t0)
    return kappa_stats["mean"]


def run_optimization():
    print("Starting Bayesian Optimization for maximal kappa...")

    n_walkers = 32
    n_dim = len(param_space)

    initial_guess = np.array([2e24, 5e22, 12e6])
    p0 = initial_guess + 1e-4 * initial_guess * np.random.randn(n_walkers, n_dim)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)
    sampler.run_mcmc(p0, 1000, progress=True)

    log_prob_chain = sampler.get_log_prob(discard=100)
    chain = sampler.get_chain(discard=100)

    max_prob_index = np.unravel_index(np.argmax(log_prob_chain), log_prob_chain.shape)
    best_params = chain[max_prob_index]

    print("\n--- Optimization Complete ---")
    for i, key in enumerate(param_space.keys()):
        print(f"  - {key}: {best_params[i]:.3e}")

    optimal_log_prob = log_prob(best_params)
    print(f"\nMean kappa at optimal point: {optimal_log_prob:.4e}")

    return best_params


if __name__ == "__main__":
    run_optimization()

