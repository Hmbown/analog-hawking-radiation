import importlib
import os

import numpy as np
import pytest


def test_graybody_acoustic_wkb_cpu_gpu_parity():
    cupy = pytest.importorskip("cupy")  # noqa: F841

    # Prepare simple profile
    N = 1024
    x = np.linspace(-1.0, 1.0, N)
    a = 1.0
    c0 = 0.2
    v = a * x
    c = np.full_like(x, c0)
    kappa = abs(a)
    freqs = np.logspace(-2, 2, 128)

    # CPU run
    os.environ["ANALOG_HAWKING_FORCE_CPU"] = "1"
    from analog_hawking.utils import array_module as am
    importlib.reload(am)
    from analog_hawking.physics_engine.optimization import graybody_1d as gb_cpu
    importlib.reload(gb_cpu)
    res_cpu = gb_cpu.compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=0.3)

    # GPU run
    os.environ.pop("ANALOG_HAWKING_FORCE_CPU", None)
    os.environ["ANALOG_HAWKING_USE_CUPY"] = "1"
    importlib.reload(am)
    from analog_hawking.physics_engine.optimization import graybody_1d as gb_gpu
    importlib.reload(gb_gpu)
    res_gpu = gb_gpu.compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=0.3)

    # Compare transmissions within reasonable tolerance
    assert res_cpu.transmission.shape == res_gpu.transmission.shape
    np.testing.assert_allclose(res_cpu.transmission, res_gpu.transmission, rtol=1e-3, atol=1e-6)
