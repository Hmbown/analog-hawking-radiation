"""3D graybody transmission estimator for analog horizons using multi-D WKB approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Tuple, Optional

import numpy as np
from scipy.integrate import quad

from analog_hawking.physics_engine.optimization.graybody_1d import GraybodyResult, _wkb_transmission  # Reuse 1D utilities
from ...horizon import find_horizons_with_uncertainty


@dataclass
class Graybody3DResult(GraybodyResult):
    """Extended result for 3D graybody with transverse factors."""
    transverse_scattering: np.ndarray  # Approximate scattering correction per frequency




def _effective_potential_3d(r: np.ndarray, v_r: np.ndarray, c_s: np.ndarray, l: int = 0) -> np.ndarray:
    """Effective potential in 3D radial coordinate, including centrifugal term."""
    acoustic_pot = (np.abs(v_r) - c_s) ** 2
    centrifugal = l * (l + 1) / r**2 if l > 0 else 0.0
    return acoustic_pot + centrifugal


def compute_graybody_3d(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    velocity: np.ndarray,  # Shape (nx, ny, nz)
    sound_speed: np.ndarray,  # Shape (nx, ny, nz)
    frequencies: Iterable[float],
    l_max: int = 2,  # Max angular momentum for approximation
    perturbation: float = 0.05,
    *,
    method: str = "wkb_3d",
    kappa: Optional[float] = None,
    alpha: float = 1.0,
    r_horizon: float = 1.0,  # Approximate horizon radius
) -> Graybody3DResult:
    """Compute 3D graybody transmission using WKB approximation in multi-D.

    Approximates scattering by summing over low-l partial waves; for each ω, T(ω) ≈ sum_l (2l+1) T_l(ω) / (2l_max + 1).
    T_l uses radial WKB with effective potential including centrifugal barrier.
    Assumes cylindrical symmetry; slices along central plane for radial profile.
    """
    frequencies = np.asarray(list(frequencies), dtype=float)
    omega_values = 2.0 * np.pi * frequencies

    # Extract central slice for radial approximation (assume x is propagation direction)
    ny, nz = velocity.shape[1], velocity.shape[2]
    y_c, z_c = y[ny//2], z[nz//2]
    r = np.sqrt(y_c**2 + z_c**2)  # Radial distance in transverse plane
    v_r = velocity[:, ny//2, nz//2]  # Radial velocity approximation
    cs_r = sound_speed[:, ny//2, nz//2]

    # Estimate kappa if not provided
    if kappa is None or kappa <= 0.0:
        from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
        hr = find_horizons_with_uncertainty(x, v_r, cs_r, kappa_method="acoustic_exact")
        kappa_eff = float(np.max(hr.kappa)) if hr.kappa.size else 1.0
    else:
        kappa_eff = float(kappa)

    transmission = np.zeros_like(omega_values)
    transverse_scattering = np.zeros_like(omega_values)
    for i, omega in enumerate(omega_values):
        T_l_sum = 0.0
        for l in range(l_max + 1):
            pot_func = lambda xi: _effective_potential_3d(r_horizon + xi, v_r, cs_r, l)
            T_l = _wkb_transmission(omega, pot_func, (-r_horizon, len(x) * np.mean(np.gradient(x))))
            T_l_sum += (2 * l + 1) * T_l
        transmission[i] = T_l_sum / (2 * l_max + 1)

        # Approximate scattering correction: simple geometric for low l
        scattering_factor = 1.0 - (l_max * (l_max + 1)) / (2 * (omega * r_horizon / kappa_eff)**2 + 1e-6)
        transverse_scattering[i] = np.clip(scattering_factor, 0.0, 1.0)

    # Uncertainty via perturbation (reuse 1D logic on slice)
    v_pert = v_r * (1.0 + perturbation)
    cs_pert = cs_r * (1.0 - perturbation)
    # Recompute for upper/lower bounds (simplified)
    uncertainties = perturbation * transmission  # Heuristic

    return Graybody3DResult(
        frequencies=frequencies,
        transmission=np.clip(transmission * np.mean(transverse_scattering), 0.0, 1.0),
        uncertainties=uncertainties,
        transverse_scattering=transverse_scattering,
    )