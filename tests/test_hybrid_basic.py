from __future__ import annotations

import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.horizon_hybrid import (
    HybridHorizonParams,
    find_hybrid_horizons,
)
from analog_hawking.detection.hybrid_spectrum import (
    calculate_enhanced_hawking_spectrum,
)
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty


def _synthetic_profile():
    # 1D grid
    x = np.linspace(-20e-6, 20e-6, 801)
    # Fluid velocity with a smooth sign change
    v = 0.02 * np.tanh(x / 4e-6)  # arbitrary units
    # Constant sound speed smaller than max |v|
    c_s = np.full_like(x, 0.01)
    return x, v, c_s


def test_plasma_mirror_dynamics_and_kappa_mapping():
    x, _, _ = _synthetic_profile()
    t = np.linspace(0.0, 100e-15, 401)
    p = PlasmaMirrorParams(n_p0=1e24, omega_p0=1e14, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel")
    mirror = calculate_plasma_mirror_dynamics(x, laser_intensity=5e17, params=p, t=t)
    assert mirror.t.shape == t.shape
    assert mirror.xM.shape == t.shape
    assert mirror.vM.shape == t.shape
    assert mirror.aM.shape == t.shape
    assert mirror.kappa_mirror > 0.0


def test_hybrid_reduces_to_fluid_when_coupling_zero():
    x, v, c_s = _synthetic_profile()
    # fluid-only
    fluid = find_horizons_with_uncertainty(x, v, c_s)
    # minimal mirror (non-zero but will be suppressed by zero coupling)
    t = np.linspace(0.0, 100e-15, 401)
    p = PlasmaMirrorParams(n_p0=1e24, omega_p0=1e14, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="unruh")
    mirror = calculate_plasma_mirror_dynamics(x, laser_intensity=5e17, params=p, t=t)

    hh = find_hybrid_horizons(x, v, c_s, mirror, HybridHorizonParams(coupling_strength=0.0))
    assert hh.fluid.positions.shape == fluid.positions.shape
    assert np.allclose(hh.hybrid_kappa, fluid.kappa)


def test_hybrid_spectrum_success():
    res = calculate_enhanced_hawking_spectrum(kappa_fluid=1e12, kappa_mirror=1e12, coupling_weight=0.3)
    assert res.get("success", False)
    assert "frequencies" in res and "power_spectrum" in res
    f = np.asarray(res["frequencies"])  # type: ignore[index]
    P = np.asarray(res["power_spectrum"])  # type: ignore[index]
    assert f.size > 0 and P.size == f.size
