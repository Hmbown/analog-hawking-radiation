import numpy as np
from scipy.constants import hbar, k, pi

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def test_hawking_temperature_units_scale():
    kappa = 1.0  # s^-1
    qft = QuantumFieldTheory(surface_gravity=kappa)
    T_H = qft.hawking_temperature_from_kappa(kappa)
    expected = float(hbar * kappa / (2.0 * pi * k))
    assert np.isclose(T_H, expected, rtol=1e-10, atol=0.0)


def test_kappa_methods_relative_scaling_linear_profile():
    # v = a x, c = c0 => κ_acoustic ≈ |a|, κ_exact ≈ |a|, κ_legacy ≈ 0.5|a|
    a = 1.0
    c0 = 0.2
    x = np.linspace(-1.0, 1.0, 2001)
    v = a * x
    c = np.full_like(x, c0)
    res_ac = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic")
    res_ex = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic_exact")
    res_legacy = find_horizons_with_uncertainty(x, v, c, kappa_method="legacy")
    assert res_ac.kappa.size and res_ex.kappa.size and res_legacy.kappa.size
    assert np.isclose(res_ac.kappa[0], abs(a), rtol=5e-2)
    assert np.isclose(res_ex.kappa[0], abs(a), rtol=5e-2)
    assert np.isclose(res_legacy.kappa[0], 0.5 * abs(a), rtol=5e-2)

