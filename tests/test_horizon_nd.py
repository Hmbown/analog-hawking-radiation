from __future__ import annotations

import numpy as np

from analog_hawking.physics_engine.horizon_nd import find_horizon_surface_nd


def _tanh_profile_2d(nx=200, ny=50, x0=5e-6, Lx=10e-6, Ly=5e-6, v0=1.5e6, cs0=1.0e6, sigma=0.5e-6):
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    vx = v0 * np.tanh((X - x0) / sigma)
    vy = np.zeros_like(vx)
    v = np.stack([vx, vy], axis=-1)  # (nx, ny, 2)
    cs = np.full((nx, ny), cs0, dtype=float)
    return (x, y), v, cs


def _expected_kappa_tanh(v0: float, cs0: float, sigma: float) -> float:
    # dv/dx at horizon: v = v0 tanh((x-x0)/sigma) => dv/dx|_h = (v0^2 - cs0^2)/(sigma v0)
    return abs((v0 * v0 - cs0 * cs0) / (sigma * v0))


def test_horizon_nd_2d_tanh_matches_dvdx():
    x0 = 5e-6
    v0 = 2.0e6
    cs0 = 1.0e6
    sigma = 0.4e-6
    grids, v, cs = _tanh_profile_2d(nx=160, ny=40, x0=x0, v0=v0, cs0=cs0, sigma=sigma)
    surf = find_horizon_surface_nd(grids, v, cs, scan_axis=0)
    assert surf.positions.shape[0] > 0
    kappa_med = float(np.median(surf.kappa))
    kappa_exp = _expected_kappa_tanh(v0, cs0, sigma)
    # 5% relative tolerance
    assert np.isclose(kappa_med, kappa_exp, rtol=0.1), (kappa_med, kappa_exp)


def _tanh_profile_3d(nx=80, ny=24, nz=16, x0=5e-6, Lx=10e-6, Ly=5e-6, Lz=5e-6, v0=1.8e6, cs0=1.0e6, sigma=0.6e-6):
    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, Lz, nz)
    X = x[:, None, None]
    vx = v0 * np.tanh((X - x0) / sigma)
    vx = np.broadcast_to(vx, (nx, ny, nz))
    vy = np.zeros_like(vx)
    vz = np.zeros_like(vx)
    v = np.stack([vx, vy, vz], axis=-1)
    cs = np.full((nx, ny, nz), cs0, dtype=float)
    return (x, y, z), v, cs


def test_horizon_nd_3d_tanh_sheet_basic():
    x0 = 4e-6
    v0 = 1.6e6
    cs0 = 1.0e6
    sigma = 0.5e-6
    grids, v, cs = _tanh_profile_3d(nx=64, ny=16, nz=12, x0=x0, v0=v0, cs0=cs0, sigma=sigma)
    surf = find_horizon_surface_nd(grids, v, cs, scan_axis=0)
    assert surf.positions.shape[0] > 0
    # κ should be approximately constant across patches
    kappa_med = float(np.median(surf.kappa))
    kappa_std = float(np.std(surf.kappa))
    assert kappa_med > 0
    assert kappa_std / kappa_med < 0.25

