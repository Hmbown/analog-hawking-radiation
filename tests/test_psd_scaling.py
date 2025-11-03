import numpy as np

from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def test_psd_scales_linearly_with_area_and_solid_angle():
    kappa = 1e12
    freqs = np.logspace(8, 10, 50)
    omega = 2 * np.pi * freqs

    q1 = QuantumFieldTheory(
        surface_gravity=kappa, emitting_area_m2=1e-6, solid_angle_sr=0.05, coupling_efficiency=0.1
    )
    q2 = QuantumFieldTheory(
        surface_gravity=kappa, emitting_area_m2=2e-6, solid_angle_sr=0.05, coupling_efficiency=0.1
    )

    P1 = q1.hawking_spectrum(omega, transmission=np.ones_like(omega))
    P2 = q2.hawking_spectrum(omega, transmission=np.ones_like(omega))

    # Exact linearity across all frequencies
    np.testing.assert_allclose(P2, 2.0 * P1, rtol=0.0, atol=0.0)
