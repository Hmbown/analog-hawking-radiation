import numpy as np

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty


def test_horizon_kappa_linear_flow_constant_cs_acoustic_and_legacy():
    # v(x) = a x, c_s = c0 -> |v| - c_s crosses zero near x = c0/a
    a = 1.0  # 1/s (arbitrary units compatible with x in meters)
    c0 = 0.2  # m/s
    x = np.linspace(-1.0, 1.0, 2001)
    v = a * x
    c_s = np.full_like(x, c0)

    # Acoustic default: κ ≈ |∂x(c_s − |v|)| = |0 − sign(v)·a| = a
    res_ac = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic")
    assert res_ac.positions.size >= 1
    expected_ac = abs(a)
    assert np.isclose(res_ac.kappa[0], expected_ac, rtol=5e-2)
    assert res_ac.kappa_err[0] >= 0.0

    # Legacy: κ = 0.5·|∂x(|v| − c_s)| = 0.5·|a - 0| = 0.5 a
    res_legacy = find_horizons_with_uncertainty(x, v, c_s, kappa_method="legacy")
    expected_legacy = 0.5 * abs(a)
    assert np.isclose(res_legacy.kappa[0], expected_legacy, rtol=5e-2)

    # Acoustic exact should reduce to |a|
    res_exact = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic_exact")
    assert np.isclose(res_exact.kappa[0], expected_ac, rtol=5e-2)


def test_horizon_kappa_constant_flow_linear_cs_all_methods_agree():
    # v(x) = v0, c_s = |v0| + b x -> crossing at x=0
    v0 = 0.3
    b = 2.5  # 1/s
    x = np.linspace(-1.0, 1.0, 2001)
    v = np.full_like(x, v0)
    c_s = np.abs(v0) + b * x

    res_ac = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic")
    res_legacy = find_horizons_with_uncertainty(x, v, c_s, kappa_method="legacy")
    res_exact = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic_exact")

    expected = abs(b)
    assert res_ac.positions.size >= 1
    assert np.isclose(res_ac.kappa[0], expected, rtol=5e-2)
    assert np.isclose(res_exact.kappa[0], expected, rtol=5e-2)
    assert np.isclose(res_legacy.kappa[0], 0.5 * expected, rtol=5e-2)


def test_horizon_kappa_mixed_linear_v_cs_all_methods():
    # v(x) = a x, c_s(x) = c0 + b x -> crossing at x = c0/(a-b) if a>b
    a = 1.5
    b = 0.4
    c0 = 0.2
    x = np.linspace(-1.0, 2.0, 4001)
    v = a * x
    c_s = c0 + b * x

    res_ac = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic")
    res_legacy = find_horizons_with_uncertainty(x, v, c_s, kappa_method="legacy")
    res_exact = find_horizons_with_uncertainty(x, v, c_s, kappa_method="acoustic_exact")

    # Two horizons exist solving c0 + b x = a |x|: at x = c0/(a-b) (v>0) and x = -c0/(a+b) (v<0)
    expected_pos = abs(a - b)
    expected_neg = abs(a + b)
    assert res_ac.positions.size >= 2
    # Sort kappas consistently with positions
    order = np.argsort(res_ac.positions)
    k_ac_sorted = res_ac.kappa[order]
    k_exact_sorted = res_exact.kappa[order]
    k_leg_sorted = res_legacy.kappa[order]
    # Negative-x horizon first
    assert np.isclose(k_ac_sorted[0], expected_neg, rtol=5e-2)
    assert np.isclose(k_exact_sorted[0], expected_neg, rtol=5e-2)
    assert np.isclose(k_leg_sorted[0], 0.5 * expected_neg, rtol=5e-2)
    # Positive-x horizon second
    assert np.isclose(k_ac_sorted[1], expected_pos, rtol=5e-2)
    assert np.isclose(k_exact_sorted[1], expected_pos, rtol=5e-2)
    assert np.isclose(k_leg_sorted[1], 0.5 * expected_pos, rtol=5e-2)
