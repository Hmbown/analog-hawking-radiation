import numpy as np

from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody

def test_compute_graybody_monotonic_transmission():
    x = np.linspace(-1, 1, 200)
    velocity = np.tanh(x) * 0.5
    sound_speed = np.full_like(x, 0.3)
    freqs = np.logspace(6, 9, 20)
    result = compute_graybody(x, velocity, sound_speed, freqs)
    assert result.transmission.shape == freqs.shape
    assert np.all((result.transmission >= 0.0) & (result.transmission <= 1.0))

def test_graybody_uncertainty_positive():
    x = np.linspace(-1, 1, 200)
    velocity = np.tanh(x) * 0.5
    sound_speed = np.full_like(x, 0.3)
    freqs = np.logspace(6, 9, 10)
    result = compute_graybody(x, velocity, sound_speed, freqs)
    assert np.all(result.uncertainties >= 0.0)
