"""3D graybody transmission estimator for analog horizons using multi-D WKB approximation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from analog_hawking.physics_engine.optimization.graybody_1d import (  # Reuse 1D utilities
    GraybodyResult,
    _wkb_transmission,
)


@dataclass
class Graybody3DResult(GraybodyResult):
    """Extended result for 3D graybody with transverse factors."""
    transverse_scattering: np.ndarray  # Approximate scattering correction per frequency




def _effective_potential_3d(
    r: np.ndarray,
    v_r: np.ndarray,
    c_s: np.ndarray,
    angular_mode: int = 0,
) -> np.ndarray:
    """Effective potential in 3D radial coordinate, including centrifugal term."""
    acoustic_pot = (np.abs(v_r) - c_s) ** 2
    centrifugal = angular_mode * (angular_mode + 1) / r**2 if angular_mode > 0 else 0.0
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
        transmission_sum = 0.0
        for angular_mode in range(l_max + 1):

            def pot_func(xi: float, *, mode: int = angular_mode) -> float:
                return float(_effective_potential_3d(r_horizon + xi, v_r, cs_r, mode))

            transmission_mode = _wkb_transmission(
                omega, pot_func, (-r_horizon, len(x) * np.mean(np.gradient(x)))
            )
            transmission_sum += (2 * angular_mode + 1) * transmission_mode
        transmission[i] = transmission_sum / (2 * l_max + 1)

        # Approximate scattering correction: simple geometric for low angular indices
        scattering_factor = 1.0 - (l_max * (l_max + 1)) / (2 * (omega * r_horizon / kappa_eff)**2 + 1e-6)
        transverse_scattering[i] = np.clip(scattering_factor, 0.0, 1.0)

    # Uncertainty via perturbation (reuse 1D logic on slice; simplified heuristic)
    uncertainties = perturbation * transmission  # Heuristic

    return Graybody3DResult(
        frequencies=frequencies,
        transmission=np.clip(transmission * np.mean(transverse_scattering), 0.0, 1.0),
        uncertainties=uncertainties,
        transverse_scattering=transverse_scattering,
    )
