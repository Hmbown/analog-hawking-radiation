"""
Test Script for the Probabilistic Horizon Model

This script serves as a unit test and example for the ProbabilisticHorizonModel.
It initializes the model with a set of parameters and uncertainties, calculates
the horizon formation probability, and prints the results.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.optimization.probabilistic_horizon import ProbabilisticHorizonModel


def test_probabilistic_model():
    """
    Tests the functionality of the ProbabilisticHorizonModel.
    """
    print("Testing Probabilistic Horizon Model...")

    # --- 1. Define Base Parameters and Uncertainties ---
    base_params = {
        "plasma_density": 1e24,  # m^-3
        "laser_intensity": 5e22,  # W/m^2
        "T_peak": 10e6,  # K
        "T_background": 2e6,  # K
    }

    param_uncertainties = {
        "plasma_density": 0.2e24,  # 20% uncertainty
        "laser_intensity": 1e22,  # 20% uncertainty
        "T_peak": 2e6,  # 20% uncertainty
    }

    # --- 2. Initialize the Model ---
    prob_model = ProbabilisticHorizonModel(
        base_params=base_params,
        param_uncertainties=param_uncertainties,
        n_samples=500,  # A smaller number of samples for a quick test
    )

    # --- 3. Define Simulation Grid ---
    x_grid = np.linspace(-50e-6, 50e-6, 500)
    t0 = 50e-15  # s

    # --- 4. Calculate Probability ---
    probability, kappa_stats = prob_model.calculate_horizon_probability(x_grid, t0)

    # --- 5. Print Results ---
    print(f"  Base Parameters: {base_params}")
    print(f"  Uncertainties: {param_uncertainties}")
    print("-" * 30)
    print(f"  Calculated Horizon Formation Probability: {probability:.2%}")
    print(f"  Mean Surface Gravity (kappa): {kappa_stats['mean']:.2e} s^-1")
    print(f"  Std Dev of Surface Gravity (kappa): {kappa_stats['std']:.2e} s^-1")
    print("-" * 30)

    # --- 6. Assertions for Basic Validation ---
    assert 0.0 <= probability <= 1.0, "Probability should be between 0 and 1."
    assert kappa_stats["mean"] >= 0, "Mean kappa should not be negative."
    assert kappa_stats["std"] >= 0, "Kappa std dev should not be negative."

    print("✅ Probabilistic model test completed successfully.")


if __name__ == "__main__":
    test_probabilistic_model()
