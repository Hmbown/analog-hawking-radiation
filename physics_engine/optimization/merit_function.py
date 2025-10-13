"""
Unified Merit Function for "Glow" Detection

This module defines a merit function that combines the probability of horizon
formation with the signal-to-noise ratio of the resulting Hawking radiation
to produce a single score for a given set of experimental parameters.
"""

import numpy as np
from .probabilistic_horizon import ProbabilisticHorizonModel
from .snr_model import RadioSNRModel

class GlowMeritFunction:
    """
    Calculates a merit score for detecting the "glow" of Hawking radiation.
    """

    def __init__(self, base_params, param_uncertainties, snr_config, n_samples=1000):
        """
        Initializes the merit function.

        Args:
            base_params (dict): Mean values for the simulation parameters.
            param_uncertainties (dict): Standard deviations for the simulation parameters.
            snr_config (dict): Configuration for the RadioSNRModel.
                               e.g., {'system_temperature': 50, 'bandwidth': 10e6, 'integration_time': 3600}
            n_samples (int): Number of Monte Carlo samples for the probabilistic model.
        """
        self.prob_model = ProbabilisticHorizonModel(base_params, param_uncertainties, n_samples)
        self.snr_model = RadioSNRModel(**snr_config)

    def calculate_merit(self, x_grid, t0):
        """
        Calculates the final merit score.

        The merit score is defined as: Merit = P_horizon * E[SNR(T_H(kappa))]
        This represents the horizon formation probability multiplied by the expected
        signal-to-noise ratio of the Hawking radiation.

        Args:
            x_grid (np.ndarray): The spatial grid for the simulation.
            t0 (float): A representative time slice for the analysis.

        Returns:
            float: The final merit score.
        """
        # Calculate the probability of horizon formation and the stats for kappa
        p_horizon, kappa_stats = self.prob_model.calculate_horizon_probability(x_grid, t0)

        # If no horizons are ever formed, the merit is zero
        if p_horizon == 0:
            return 0.0

        # Convert the mean kappa to an expected Hawking temperature
        mean_kappa = kappa_stats['mean']
        expected_T_H = self.snr_model.kappa_to_hawking_temperature(mean_kappa)

        # Calculate the SNR for this expected temperature
        expected_snr = self.snr_model.calculate_snr(expected_T_H)

        # The merit is the probability of forming a horizon multiplied by the expected SNR
        merit_score = p_horizon * expected_snr

        return merit_score
