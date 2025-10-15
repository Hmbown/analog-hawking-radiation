import os
import sys

# Ensure we can import the helper from scripts/
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from hawking_detection_experiment import _choose_frequency_band  # type: ignore


def test_frequency_gating_boundary_at_10K():
    fmin, fmax = _choose_frequency_band(10.0)
    assert fmin == 1e6 and fmax == 1e11


def test_frequency_gating_above_10K_goes_thz_ehz():
    fmin, fmax = _choose_frequency_band(10.0001)
    assert fmin == 1e12 and fmax == 1e18
