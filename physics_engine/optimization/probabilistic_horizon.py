"""
Probabilistic Horizon Formation Model

This module provides a framework for estimating the probability of analog event
horizon formation by incorporating uncertainties in the input physical parameters.
It uses a Monte Carlo approach to sample from parameter distributions and 
calculates a formation probability based on the results.
"""

import numpy as np
from scipy.constants import c
from ..plasma_models.laser_plasma_interaction import MaxwellFluidModel, AnalogHorizonFormation

class ProbabilisticHorizonModel:
    """
    Calculates the probability of horizon formation using Monte Carlo sampling
    to account for parameter uncertainties.
    """

    def __init__(self, base_params, param_uncertainties, n_samples=1000):
        """
        Initializes the probabilistic model.

        Args:
            base_params (dict): A dictionary of the mean values for the simulation parameters.
                e.g., {'plasma_density': 1e24, 'laser_intensity': 5e22, 'T_peak': 10e6}
            param_uncertainties (dict): A dictionary of the standard deviations for each parameter.
                e.g., {'plasma_density': 0.1e24, 'laser_intensity': 0.5e22, 'T_peak': 1e6}
            n_samples (int): The number of Monte Carlo samples to run.
        """
        self.base_params = base_params
        self.param_uncertainties = param_uncertainties
        self.n_samples = n_samples
        self.samples = self._generate_samples()

    def _generate_samples(self):
        """Generates random samples for each parameter based on a normal distribution."""
        samples = {}
        for param, mean in self.base_params.items():
            std_dev = self.param_uncertainties.get(param, 0)
            samples[param] = np.random.normal(loc=mean, scale=std_dev, size=self.n_samples)
        return samples

    def calculate_horizon_probability(self, x_grid, t0):
        """
        Runs the Monte Carlo simulation to calculate the probability of horizon formation.

        Args:
            x_grid (np.ndarray): The spatial grid for the simulation.
            t0 (float): A representative time slice for the analysis.

        Returns:
            float: The probability of horizon formation (0.0 to 1.0).
            dict: A dictionary containing the mean and std dev of the resulting surface gravity `kappa`.
        """
        horizon_formation_count = 0
        kappa_values = []

        # Simplified fluid velocity profile for this probabilistic assessment
        # In a full implementation, this would also be subject to uncertainty
        v_fluid = 0.05 * c * np.exp(-((x_grid - 5e-6) / (15e-6))**2)

        for i in range(self.n_samples):
            # Instantiate models with the sampled parameters for this run
            maxwell_model = MaxwellFluidModel(
                plasma_density=self.samples['plasma_density'][i],
                laser_intensity=self.samples['laser_intensity'][i]
            )
            
            # Create the temperature profile from sampled parameters
            T_peak = self.samples['T_peak'][i]
            T_background = self.base_params.get('T_background', 2e6) # Assume background is certain
            T_profile = T_background + (T_peak - T_background) * np.exp(-(x_grid / (20e-6))**2)

            analog_model = AnalogHorizonFormation(
                maxwell_model,
                plasma_temperature_profile=T_profile
            )

            # Check for horizon formation
            horizon_mask = analog_model.horizon_position(x_grid, t0, v_actual=v_fluid)
            
            if np.any(horizon_mask):
                horizon_formation_count += 1
                
                # If a horizon forms, calculate and store its surface gravity
                kappa = analog_model.surface_gravity_at_horizon(x_grid, t0, v_actual=v_fluid)
                # We are interested in the maximum kappa value along the grid
                kappa_values.append(np.max(kappa))

        probability = horizon_formation_count / self.n_samples
        
        kappa_stats = {
            'mean': np.mean(kappa_values) if kappa_values else 0,
            'std': np.std(kappa_values) if kappa_values else 0
        }

        return probability, kappa_stats
