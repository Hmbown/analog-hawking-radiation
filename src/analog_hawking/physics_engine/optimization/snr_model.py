"""
Signal-to-Noise Ratio (SNR) Model for Radio Astronomy

This module provides a simple model for calculating the signal-to-noise ratio (SNR)
for a radio telescope observing a thermal source, such as analog Hawking radiation.
"""

import numpy as np
from scipy.constants import k, hbar

class RadioSNRModel:
    """
    Calculates the SNR for a radiometer.
    """

    def __init__(self, system_temperature=50, bandwidth=10e6, integration_time=3600):
        """
        Initializes the radio SNR model.

        Args:
            system_temperature (float): The total system noise temperature in Kelvin (T_sys).
            bandwidth (float): The observing bandwidth in Hz (B).
            integration_time (float): The total integration time in seconds (t).
        """
        self.T_sys = system_temperature
        self.B = bandwidth
        self.t = integration_time

    def calculate_snr(self, hawking_temperature):
        """
        Calculates the signal-to-noise ratio.

        The SNR is given by the radiometer equation: SNR = (T_sig / T_sys) * sqrt(B * t)

        Args:
            hawking_temperature (float): The effective temperature of the Hawking radiation
                                         signal in Kelvin (T_sig).

        Returns:
            float: The calculated signal-to-noise ratio.
        """
        if self.T_sys <= 0 or self.B <= 0 or self.t <= 0:
            return 0.0

        T_sig = hawking_temperature
        snr = (T_sig / self.T_sys) * np.sqrt(self.B * self.t)
        return snr

    @staticmethod
    def kappa_to_hawking_temperature(kappa):
        """
        Converts surface gravity (kappa) to Hawking temperature.

        The relationship is T_H = (hbar * kappa) / (2 * pi * k_B).

        Args:
            kappa (float): The surface gravity in s^-1.

        Returns:
            float: The Hawking temperature in Kelvin.
        """
        if kappa <= 0:
            return 0.0
            
        T_H = (hbar * kappa) / (2 * np.pi * k)
        return T_H
