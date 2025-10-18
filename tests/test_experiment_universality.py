from __future__ import annotations

import numpy as np

from scripts.experiment_universality_collapse import (
    make_linear_profile,
    make_tanh_profile,
    make_piecewise_ramp_profile,
    make_no_horizon_profile,
    _spectrum_for_profile,
)
from analog_hawking.detection.psd_collapse import (
    omega_over_kappa_axis,
    resample_on_x,
    collapse_stats,
)


def _quick_psd(profile, alpha=0.8, B=1e8, T_sys=30.0):
    # reuse helper; returns f, psd, record
    return _spectrum_for_profile(profile, alpha=alpha, B=B, T_sys=T_sys)


def test_universality_pipeline_shapes_and_metrics():
    # Small analytic subset; keep fast (<10s)
    profiles = [make_linear_profile(1), make_tanh_profile(2), make_piecewise_ramp_profile(3)]
    curves = []
    x_common = np.linspace(0.2, 5.0, 120)
    for p in profiles:
        f, psd, rec = _quick_psd(p)
        assert psd.size > 0 and rec.kappa > 0
        x = omega_over_kappa_axis(f, rec.kappa)
        y = resample_on_x(x, psd, x_common)
        curves.append(y)

    stats = collapse_stats(curves)
    assert np.isfinite(stats.rms_relative)
    assert stats.mean.shape == x_common.shape


def test_control_no_horizon_yields_no_spectrum():
    # No-horizon profile should return no spectrum / success False
    p = make_no_horizon_profile(5)
    f, psd, rec = _quick_psd(p)
    assert psd.size == 0
    assert rec.success is False

