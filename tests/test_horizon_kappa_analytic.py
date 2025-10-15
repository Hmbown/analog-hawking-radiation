import numpy as np

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty


def test_horizon_kappa_linear_flow_constant_cs():
    # v(x) = a x, c_s = c0 -> |v| - c_s crosses zero near x = c0/a
    a = 1.0  # 1/s (arbitrary units compatible with x in meters)
    c0 = 0.2  # m/s
    x = np.linspace(-1.0, 1.0, 2001)
    v = a * x
    c_s = np.full_like(x, c0)

    res = find_horizons_with_uncertainty(x, v, c_s)
    assert res.positions.size >= 1

    # For v >= 0 near root, d|v|/dx = +a, so kappa = 0.5*|a - 0| = 0.5*a
    # Accept small numerical error from discretization
    expected = 0.5 * abs(a)
    assert np.isclose(res.kappa[0], expected, rtol=5e-2)
    assert res.kappa_err[0] >= 0.0
