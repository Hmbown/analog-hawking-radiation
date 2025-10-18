import numpy as np

from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty


def test_graybody_acoustic_wkb_monotonic_and_limits():
    # Simple near-horizon profile: linear flow crossing constant sound speed
    x = np.linspace(-1.0, 1.0, 2001)
    a = 1.0
    c0 = 0.2
    v = a * x
    c = np.full_like(x, c0)
    # Estimate kappa from analytic methods (|a|)
    kappa = abs(a)

    # Choose frequency band around κ (κ ~ 1 s^-1)
    freqs = np.logspace(-3, 1, 200)
    gb_wkb = compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=0.3)
    T = gb_wkb.transmission
    assert T.min() >= 0.0 and T.max() <= 1.0
    # Low-omega suppression and high-omega transparency
    assert T[0] < 0.2
    assert T[-1] > 0.8

    # Around omega ~ kappa, compare to dimensionless model
    gb_dim = compute_graybody(x, v, c, freqs, method="dimensionless", kappa=kappa)
    # Use a mid-frequency index
    mid = len(freqs) // 2
    ratio = T[mid] / (gb_dim.transmission[mid] + 1e-12)
    assert 0.25 <= ratio <= 4.5  # agree within a loose factor near turnover


def test_graybody_acoustic_wkb_step_barrier_matches_closed_form_within_factor():
    # Construct a piecewise-constant gap profile to mimic a square barrier in x*
    # gap = |c - |v|| sets dx* = dx / gap and potential shape S ~ (gap^2) / max
    x = np.linspace(-1.0, 1.0, 4001)
    gap = np.zeros_like(x)
    # Barrier region with constant gap
    L = 0.2
    gap_val = 0.5
    mask = (x >= -L) & (x <= L)
    gap[mask] = gap_val
    # Outside barrier, very small gap (approach horizon)
    gap[~mask] = 1e-6
    # Recover v and c from gap: choose c = 1, |v| = c - gap
    c = np.ones_like(x)
    v = (c - gap) * np.sign(x)  # sign irrelevant for |v|

    kappa = 1.0  # set scale so V0 = (alpha kappa)^2 ~ alpha^2
    freqs = np.logspace(-2, 1, 60)
    gb = compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=1.0)
    T_wkb = gb.transmission

    # Closed-form for constant barrier in x*: T = exp(-2 sqrt(V0 - ω^2) * Delta x*) for ω < sqrt(V0)
    # Compute Delta x* across barrier: dx* = dx / gap_val in barrier region (constant)
    dx = x[1] - x[0]
    N = int(np.round(2 * L / dx)) + 1
    L_star = (N * dx) / gap_val
    V0 = (kappa ** 2) * 1.0  # since alpha=1 and S=1 in barrier region
    omega = 2 * np.pi * freqs
    T_cf = np.ones_like(omega)
    under = V0 - omega**2
    mask_u = under > 0
    T_cf[mask_u] = np.exp(-2.0 * np.sqrt(under[mask_u]) * L_star)
    # Compare over the grid within a factor ~ 5 to be conservative
    ratio = T_wkb / (T_cf + 1e-30)
    # Limit to a sensible omega range to avoid extreme tails
    sel = (omega > 1e-2) & (omega < 2.0)
    if np.any(sel):
        r = ratio[sel]
        finite = np.isfinite(r) & (T_wkb[sel] > 0)
        if np.any(finite):
            r = r[finite]
            assert np.nanmedian(r) > 1/5 and np.nanmedian(r) < 5.0
