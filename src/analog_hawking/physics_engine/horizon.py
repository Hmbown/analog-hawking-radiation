"""
Horizon finding utilities for analog Hawking radiation in 1D profiles.

Provides:
- sound_speed(T_e, ion_mass=m_p, gamma=5/3)
- find_horizons_with_uncertainty(x, v, c_s): robust root finding on f(x)=|v|-c_s
  with simple numerical uncertainty estimates from multi-scale finite differences.

Notes on uncertainty: the returned uncertainty on the surface gravity (kappa) is a
numerical estimate from varying the finite-difference stencil (grid sensitivity),
not a propagation of physical uncertainties in the underlying model parameters.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
from scipy.constants import k, m_p, mu_0


def sound_speed(T_e, ion_mass: float = m_p, gamma: float = 5.0/3.0):
    """Compute adiabatic sound speed c_s = sqrt(gamma k T_e / m_i).
    T_e can be scalar or array (Kelvin)."""
    T_e = np.asarray(T_e)
    return np.sqrt(np.maximum(gamma * k * T_e / ion_mass, 0.0))


def fast_magnetosonic_speed(T_e,
                            n_e,
                            B,
                            ion_mass: float = m_p,
                            gamma: float = 5.0/3.0):
    """Approximate fast magnetosonic speed c_f ≈ sqrt(c_s^2 + v_A^2).
    Args:
        T_e: electron temperature (K)
        n_e: number density (m^-3)
        B: magnetic field (Tesla)
    Returns:
        c_fast (m/s)
    Note: This is a simplified approximation for guidance; real MHD is directional.
    """
    c_s = sound_speed(T_e, ion_mass=ion_mass, gamma=gamma)
    rho = n_e * ion_mass
    v_A = np.where(rho > 0, B / np.sqrt(mu_0 * rho), 0.0)
    return np.sqrt(c_s**2 + v_A**2)


@dataclass
class HorizonResult:
    positions: np.ndarray        # horizon x-positions
    kappa: np.ndarray            # surface gravity estimates at horizons (s^-1)
    kappa_err: np.ndarray        # numerical (grid) uncertainty estimates
    dvdx: np.ndarray             # dv/dx at horizon
    dcsdx: np.ndarray            # dc_s/dx at horizon


def _refine_root(xl, xr, fl, fr, f, max_iter=20):
    """Bisect-refine a root of f between xl and xr where fl and fr have opposite signs."""
    a, b = xl, xr
    fa, fb = fl, fr
    for _ in range(max_iter):
        c = 0.5 * (a + b)
        fc = f(c)
        if np.sign(fa) == np.sign(fc):
            a, fa = c, fc
        else:
            b, fb = c, fc
    return 0.5 * (a + b)


def _finite_grad(x, y, idx, stencil=1):
    """Central diff gradient dy/dx at index idx with stencil."""
    i = idx
    i0 = max(0, i - stencil)
    i1 = min(len(x) - 1, i + stencil)
    if i1 == i0:
        return 0.0
    return (y[i1] - y[i0]) / (x[i1] - x[i0])


def find_horizons_with_uncertainty(x: np.ndarray,
                                    v: np.ndarray,
                                    c_s: np.ndarray,
                                    sigma_cells: Optional[np.ndarray] = None) -> HorizonResult:
    """
    Find positions where |v| = c_s using sign changes in f(x)=|v|-c_s.
    Return kappa = 0.5*|d/dx(|v|-c_s)| at horizon with simple uncertainty from
    multiple finite-difference stencils.
    """
    x = np.asarray(x)
    v = np.asarray(v)
    c_s = np.asarray(c_s)
    assert x.ndim == v.ndim == c_s.ndim == 1 and len(x) == len(v) == len(c_s)

    f = np.abs(v) - c_s
    roots = []
    # detect sign changes excluding exact equalities
    for i in range(len(x) - 1):
        f0, f1 = f[i], f[i + 1]
        if np.sign(f0) == 0 and np.sign(f1) == 0:
            # rare exact equality at multiple points: pick midpoint
            roots.append(0.5 * (x[i] + x[i+1]))
        elif f0 == 0:
            roots.append(x[i])
        elif f1 == 0:
            roots.append(x[i+1])
        elif f0 * f1 < 0:
            # bracketed root; refine with bisection on f(x)
            def f_interp(xi):
                # linear interpolation for v and c_s
                # improve with local linear segments
                j = i
                t = (xi - x[j]) / (x[j+1] - x[j])
                vxi = v[j] * (1 - t) + v[j+1] * t
                csxi = c_s[j] * (1 - t) + c_s[j+1] * t
                return abs(vxi) - csxi
            root = _refine_root(x[i], x[i+1], f0, f1, f_interp)
            roots.append(root)

    roots = np.array(sorted(set([float(r) for r in roots])))
    if roots.size == 0:
        return HorizonResult(positions=np.array([]), kappa=np.array([]), kappa_err=np.array([]),
                             dvdx=np.array([]), dcsdx=np.array([]))

    # compute gradients and kappa at nearest grid index to each root
    positions = []
    kappas = []
    dk = []
    dvdx_list = []
    dcsdx_list = []
    for r in roots:
        idx = int(np.clip(np.searchsorted(x, r), 1, len(x)-2))
        local_sigma = None
        if sigma_cells is not None and sigma_cells.size == len(x):
            local_sigma = float(sigma_cells[idx])
        # multi-stencil estimates
        grads = []
        for st in (1, 2, 3):
            dv = _finite_grad(x, v, idx, stencil=st)
            dcs = _finite_grad(x, c_s, idx, stencil=st)
            df = np.sign(v[idx]) * dv if v[idx] != 0 else abs(dv)  # d|v|/dx at root
            grads.append(0.5 * abs(df - dcs))
        kappa_est = float(np.median(grads))
        kappa_err = float(np.std(grads))
        positions.append(r)
        kappas.append(kappa_est)
        dk.append(kappa_err)
        # also return single-stencil grads for info
        dv = _finite_grad(x, v, idx, stencil=1)
        dcs = _finite_grad(x, c_s, idx, stencil=1)
        dvdx_list.append(dv)
        dcsdx_list.append(dcs)

    return HorizonResult(
        positions=np.array(positions),
        kappa=np.array(kappas),
        kappa_err=np.array(dk),
        dvdx=np.array(dvdx_list),
        dcsdx=np.array(dcsdx_list)
    )

# Backward-compatible alias for clarity in downstream code/documentation
setattr(HorizonResult, "kappa_numerical_err", property(lambda self: self.kappa_err))
