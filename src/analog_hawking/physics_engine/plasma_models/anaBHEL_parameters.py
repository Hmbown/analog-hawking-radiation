"""
AnaBHEL-style parameter utilities.

These helpers supply deterministic, physics-inspired values that are sufficient
for documentation snippets and the archived demo workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.constants import c, hbar, k, e, epsilon_0, m_e


@dataclass
class AnaBHELExperiment:
    """Minimal description of the AnaBHEL down-ramp configuration."""

    plasma_density: float = 5.0e19
    laser_intensity: float = 8.0e18
    detection_solid_angle: float = 5.0e-2
    detector_area: float = 5.0e-6
    system_temperature: float = 30.0

    def realistic_simulation_parameters(self) -> Dict[str, float | Tuple[float, float]]:
        """Return representative parameters for quick simulations."""
        hawking_temp = self._hawking_temperature_estimate()
        return {
            "plasma_density": self.plasma_density,
            "laser_intensity": self.laser_intensity,
            "expected_hawking_temp_range": (
                0.25 * hawking_temp,
                4.0 * hawking_temp,
            ),
            "relativistic_parameter": self._a0_parameter(self.laser_intensity),
        }

    def calculate_signal_strength(self, hawking_temperature: float) -> Dict[str, float]:
        """Crude radiometer estimate of the detectable power."""
        hawking_temperature = max(hawking_temperature, 0.0)
        peak_frequency = 2.82 * k * hawking_temperature / hbar
        bandwidth = 0.1 * peak_frequency
        total_power = (
            k * hawking_temperature * bandwidth * self.detection_solid_angle
        )
        photon_energy = hbar * peak_frequency
        photon_flux = total_power / photon_energy if photon_energy > 0 else 0.0
        snr = (
            total_power
            / (k * self.system_temperature * np.sqrt(bandwidth))
            if bandwidth > 0
            else 0.0
        )
        return {
            "total_power": float(total_power),
            "photon_flux": float(photon_flux),
            "signal_to_noise_ratio": float(max(snr, 0.0)),
        }

    def time_to_detect_signal(self, sigma: float, hawking_temperature: float) -> float:
        """Return integration time (seconds) required to reach the given sigma."""
        sigma = max(sigma, 1.0)
        stats = self.calculate_signal_strength(hawking_temperature)
        snr = stats["signal_to_noise_ratio"]
        if snr <= 0:
            return float("inf")
        return float((sigma / snr) ** 2)

    def _hawking_temperature_estimate(self) -> float:
        a0 = self._a0_parameter(self.laser_intensity)
        omega_pe = np.sqrt(e ** 2 * self.plasma_density / (epsilon_0 * m_e))
        kappa = omega_pe * max(a0, 1e-6)
        return float(hbar * kappa / (2.0 * np.pi * k))

    @staticmethod
    def _a0_parameter(intensity: float) -> float:
        return float(np.sqrt(2.0 * intensity / (epsilon_0 * c)) * (e / (m_e * c)))


__all__ = ["AnaBHELExperiment"]
