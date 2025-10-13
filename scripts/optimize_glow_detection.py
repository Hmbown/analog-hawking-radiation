"""
Bayesian Optimization for "Glow" Detection

This script uses Bayesian optimization to find the experimental parameters
that maximize the probability of forming a detectable analog event horizon.
"""

import os
import sys
import numpy as np
import emcee

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.optimization.merit_function import GlowMeritFunction

# --- 1. Define the Parameter Space for Optimization ---
# The parameters we want to optimize, with their ranges [min, max]
param_space = {
    'plasma_density': (1e23, 5e24),       # m^-3
    'laser_intensity': (1e22, 1e23),      # W/m^2
    'T_peak': (5e6, 20e6)                 # K
}

# --- 2. Define the Objective Function to be Maximized ---
# This function will take a set of parameters and return the merit score.
# emcee works by sampling a probability distribution, so we treat our merit
# score as a log-probability. A higher merit score means a higher probability.

def log_prob(params):
    # Unpack the parameters
    plasma_density, laser_intensity, T_peak = params

    # Check if the parameters are within our defined bounds
    if not (param_space['plasma_density'][0] < plasma_density < param_space['plasma_density'][1] and
            param_space['laser_intensity'][0] < laser_intensity < param_space['laser_intensity'][1] and
            param_space['T_peak'][0] < T_peak < param_space['T_peak'][1]):
        return -np.inf  # Log-probability is -infinity for out-of-bounds parameters

    # --- Configuration for the Merit Function ---
    base_params = {
        'plasma_density': plasma_density,
        'laser_intensity': laser_intensity,
        'T_peak': T_peak,
        'T_background': 2e6
    }
    # Assume a fixed 10% uncertainty for all parameters for this optimization
    param_uncertainties = {key: 0.1 * val for key, val in base_params.items()}
    snr_config = {'system_temperature': 50, 'bandwidth': 10e6, 'integration_time': 3600}

    # Initialize the merit function
    merit_func = GlowMeritFunction(
        base_params=base_params,
        param_uncertainties=param_uncertainties,
        snr_config=snr_config,
        n_samples=200  # Fewer samples for faster optimization steps
    )

    # --- Simulation Grid ---
    x_grid = np.linspace(-50e-6, 50e-6, 500)
    t0 = 50e-15

    # --- Calculate and Return Merit Score ---
    # We return the score directly, as emcee will interpret it as the log-probability
    return merit_func.calculate_merit(x_grid, t0)

# --- 3. Set up and Run the MCMC Sampler ---
def run_optimization():
    print("Starting Bayesian Optimization for 'Glow' Detection...")
    
    n_walkers = 32
    n_dim = len(param_space)
    
    # Initialize walkers in a small ball around a starting guess
    initial_guess = np.array([2e24, 5e22, 12e6])
    p0 = initial_guess + 1e-4 * initial_guess * np.random.randn(n_walkers, n_dim)

    # Set up the emcee sampler
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_prob)

    # Run the MCMC for a number of steps
    n_steps = 1000
    sampler.run_mcmc(p0, n_steps, progress=True)

    # --- 4. Extract and Display Results ---
    # Get the chain of samples and discard the "burn-in" phase
    log_prob_chain = sampler.get_log_prob(discard=100)
    chain = sampler.get_chain(discard=100)

    # Find the parameters that correspond to the highest probability (merit score)
    max_prob_index = np.unravel_index(np.argmax(log_prob_chain), log_prob_chain.shape)
    best_params = chain[max_prob_index]


    print("\n--- Optimization Complete ---")
    print(f"Optimal Parameters Found:")
    for i, key in enumerate(param_space.keys()):
        print(f"  - {key}: {best_params[i]:.3e}")

    # For verification, calculate the merit score at the optimal point
    optimal_log_prob = log_prob(best_params)
    print(f"\nMerit Score at Optimal Point: {optimal_log_prob:.4f}")
    
    return best_params

if __name__ == '__main__':
    optimal_parameters = run_optimization()
