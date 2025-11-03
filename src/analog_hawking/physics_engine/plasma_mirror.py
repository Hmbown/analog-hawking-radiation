"""
Status: ⚠️ Experimental hybrid plasma mirror mapping.

Plasma Mirror Dynamics for Analog Hawking Radiation

This module implements plasma mirror acceleration calculations based on the
AnaBHEL (Analog Black Hole Evaporation via Lasers) framework developed by:

- Prof. Pisin Chen (National Taiwan University, LeCosPA)
- Prof. Gerard Mourou (École Polytechnique, IZEST) - Nobel Laureate
- AnaBHEL Collaboration

Key References:
- Chen & Mourou (2017). Phys. Rev. Lett. 118, 045001
- Chen et al. (2022). Photonics 9(12), 1003

IMPORTANT: This implementation explores speculative hybrid coupling with fluid
systems that extends beyond established AnaBHEL theory. Treat it as
experimental scaffolding and subject any quantitative usage to an explicit
units-checked derivation against the cited references.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c


@dataclass
class PlasmaMirrorParams:
    n_p0: float  # reference plasma density [m^-3]
    omega_p0: float  # reference plasma freq [rad/s]
    a: float  # AnaBHEL profile parameter a
    b: float  # AnaBHEL profile parameter b
    D: float  # density scale length [m]
    eta_a: float = 1.0  # dimensionless factor from AnaBHEL
    model: str = "unruh"  # 'unruh' | 'anabhel'


@dataclass
class MirrorDynamics:
    t: np.ndarray
    xM: np.ndarray
    vM: np.ndarray
    aM: np.ndarray
    kappa_mirror: float


def _density_profile(x: np.ndarray, p: PlasmaMirrorParams) -> np.ndarray:
    return p.n_p0 * (p.a + p.b * np.exp(x / p.D)) ** 2


def _plasma_freq(x: np.ndarray, p: PlasmaMirrorParams) -> np.ndarray:
    return p.omega_p0 * (p.a + p.b * np.exp(x / p.D))


def calculate_plasma_mirror_dynamics(
    x: np.ndarray, laser_intensity: float, params: PlasmaMirrorParams, t: np.ndarray
) -> MirrorDynamics:
    """
    Compute an accelerating plasma mirror trajectory consistent with AnaBHEL-style
    profiles. This is a compact surrogate capturing the qualitative features:
    - Down-ramp density enhances mirror acceleration.
    - Acceleration is localized in time (bell-shaped) and bounded.

    Mapping to κ_mirror options (SI units throughout):
    - model='unruh':    κ_mirror = max(aM) / c
    - model='anabhel':  k_B T_H = (ħ/D) * η_a  => κ_mirror = (2π c η_a) / D
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    if x.ndim != 1 or t.ndim != 1 or x.size < 3 or t.size < 3:
        raise ValueError("x and t must be 1D arrays with >=3 points")

    # Choose a working point near the down-ramp side (upper quartile of x)
    x0 = float(np.quantile(x, 0.75))

    # Characteristic acceleration scale from density gradient and laser intensity
    # a0 ~ (c^2/|D|) * f(laser_intensity) * g(grad ω_p)
    domega_dx = (params.omega_p0 * params.b / params.D) * np.exp(x0 / params.D)
    grad_factor = np.tanh(abs(domega_dx) / max(params.omega_p0, 1e-30))
    intensity_scale = np.tanh(laser_intensity / (1e18))  # saturating around ~1e18 W/m^2
    a0 = (c**2 / max(abs(params.D), 1e-30)) * grad_factor * intensity_scale * 0.1

    # Bell-shaped acceleration vs time (sech^2), centered at peak
    tmin, tmax = float(t[0]), float(t[-1])
    tc = 0.5 * (tmin + tmax)
    tau = max(0.05 * (tmax - tmin), 1e-16)
    s = (t - tc) / tau
    aM = a0 / (np.cosh(s) ** 2)

    # Integrate to get velocity and position (trapezoidal)
    vM = np.zeros_like(t)
    xM = np.zeros_like(t)
    for i in range(1, t.size):
        dt = t[i] - t[i - 1]
        vM[i] = vM[i - 1] + 0.5 * dt * (aM[i] + aM[i - 1])
        xM[i] = xM[i - 1] + 0.5 * dt * (vM[i] + vM[i - 1])

    a_peak = float(np.max(np.abs(aM)))
    if not np.isfinite(a_peak) or a_peak < 0:
        raise ValueError("Computed mirror acceleration must be finite and non-negative.")

    if params.model.lower() == "unruh":
        kappa_mirror = a_peak / c
    elif params.model.lower() == "anabhel":
        if params.D <= 0:
            raise ValueError("Scale length D must be positive when using the AnaBHEL mapping.")
        # Chen & Mourou mapping with explicit c factor to ensure κ has units of s^-1
        kappa_mirror = (2.0 * np.pi * c * params.eta_a) / params.D
    else:
        kappa_mirror = a_peak / c

    return MirrorDynamics(t=t, xM=xM + x0, vM=vM, aM=aM, kappa_mirror=float(kappa_mirror))
