import numpy as np

from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def test_graybody_low_omega_vanishes():
    qft = QuantumFieldTheory(surface_gravity=1e12)
    # Frequencies far below kappa so fallback should be near zero
    omega = np.logspace(0, 4, 50)  # rad/s
    g = qft.graybody_factor(omega)
    assert np.all((g >= 0.0) & (g <= 1.0))
    assert g[0] < 1e-6


def test_graybody_applied_once_matches_manual():
    # Build manual base PSD using B_nu scaled by A*Omega*eta
    kappa = 1e12
    A, Omega, eta = 1e-6, 0.05, 0.1
    qft = QuantumFieldTheory(surface_gravity=kappa,
                             emitting_area_m2=A,
                             solid_angle_sr=Omega,
                             coupling_efficiency=eta)
    freqs = np.logspace(6, 8, 100)  # 1 MHz .. 100 MHz
    omega = 2 * np.pi * freqs

    Bnu = qft.thermal_spectral_density(freqs)
    base_psd = Bnu * A * Omega * eta

    # Arbitrary transmission shape in [0,1]
    T = np.linspace(0.0, 1.0, freqs.size) ** 2

    actual = qft.hawking_spectrum(omega, transmission=T)
    expected = base_psd * T

    # If graybody is applied exactly once, these should match closely
    assert np.allclose(actual, expected, rtol=1e-12, atol=0.0)
