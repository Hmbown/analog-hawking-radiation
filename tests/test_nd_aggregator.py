from __future__ import annotations

import numpy as np

from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd


def _make_grid_2d(nx=64, ny=24, x0=5e-6, Lx=10e-6, Ly=5e-6, v0=2.0e6, cs0=1.0e6, sigma=0.5e-6):
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    vx = v0 * np.tanh((X - x0) / sigma)
    vy = np.zeros_like(vx)
    v = np.stack([vx, vy], axis=-1)
    cs = np.full((nx, ny), cs0, dtype=float)
    return [x, y], v, cs


def test_nd_horizon_exists_and_kappa_positive():
    grids, v, cs = _make_grid_2d()
    surf = find_horizon_surface_nd(grids, v, cs, scan_axis=0)
    assert surf.positions.shape[0] > 0
    assert np.median(surf.kappa) > 0
