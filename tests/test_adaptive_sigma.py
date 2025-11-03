import numpy as np

from analog_hawking.physics_engine.plasma_models.adaptive_sigma import (
    apply_sigma_smoothing,
    estimate_sigma_map,
)


def test_estimate_sigma_map_plateau_detection():
    x = np.linspace(0, 1, 100)
    n_e = np.full_like(x, 1e18)
    T_e = 100 * np.ones_like(x)
    velocity = np.tanh((x - 0.5) * 20)
    sound_speed = np.full_like(x, 0.5)
    sigma_map, diagnostics = estimate_sigma_map(n_e, T_e, x, velocity, sound_speed)
    assert sigma_map.shape == velocity.shape
    assert diagnostics.plateau_index >= 0
    assert diagnostics.sigma_means[diagnostics.plateau_index] > 0

def test_apply_sigma_smoothing_reduces_variance():
    data = np.random.randn(100)
    sigma_map = np.full_like(data, 2.0)
    smoothed = apply_sigma_smoothing(data, sigma_map)
    assert smoothed.shape == data.shape
    assert np.var(smoothed) < np.var(data)
