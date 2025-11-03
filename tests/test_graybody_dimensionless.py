import numpy as np

from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody


def test_graybody_dimensionless_monotonic_in_frequency_ratio():
    x = np.linspace(-1, 1, 200)
    velocity = np.tanh(x) * 0.2
    sound_speed = np.full_like(x, 0.15)
    # Choose a kappa such that lowest freq is well below turnover
    kappa = 1e8  # s^-1
    freqs = np.logspace(5, 9, 40)
    gb = compute_graybody(x, velocity, sound_speed, freqs, method="dimensionless", kappa=kappa)
    T = gb.transmission
    # Transmission should be non-decreasing with frequency
    assert np.all(np.diff(T) >= -1e-12)
    # Low-frequency suppression and high-frequency transparency
    assert T[0] < 0.1 and T[-1] > 0.9
