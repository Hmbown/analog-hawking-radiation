from __future__ import annotations

import numpy as np
from scipy.constants import c, k

from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def test_thermal_spectral_density_rj_limit():
    # Choose a modest temperature where h*nu << k*T for the chosen nu range
    T = 10.0  # Kelvin
    qft = QuantumFieldTheory(surface_gravity=1.0, temperature=T)

    # Frequencies well within Rayleigh-Jeans regime for T=10 K
    freqs = np.geomspace(1e3, 1e8, 12)  # 1 kHz .. 100 MHz
    B_nu = qft.thermal_spectral_density(freqs)

    # Rayleigh-Jeans approximation: B_nu ~ 2 k T nu^2 / c^2
    B_rj = (2.0 * k * T * freqs**2) / (c**2)

    # Allow 10% relative error across the range
    rel_err = np.max(np.abs(B_nu - B_rj) / np.maximum(B_rj, 1e-300))
    assert rel_err < 0.1, f"RJ limit deviation too large: {rel_err:.3e}"
