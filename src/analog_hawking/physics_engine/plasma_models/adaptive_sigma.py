"""Adaptive coarse graining utilities for WarpX diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.constants import c, epsilon_0 as EPSILON_0, m_e

from ..horizon import find_horizons_with_uncertainty


@dataclass
class SigmaDiagnostics:
    """Summary of sigma-ladder evaluation for plateau detection."""

    sigma_means: np.ndarray
    kappa_means: np.ndarray
    horizon_counts: np.ndarray
    plateau_index: int
    ladder: Tuple[float, ...]
    epsilon: float


_E_CHARGE = 1.602176634e-19  # Coulomb


def estimate_sigma_map(
    n_e: np.ndarray,
    T_e: np.ndarray,
    grid: Optional[np.ndarray],
    velocity: np.ndarray,
    sound_speed: np.ndarray,
    sigma_factor: float = 1.0,
    ladder: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
    epsilon: float = 0.05,
    minimum_sigma_cells: float = 0.5,
) -> Tuple[np.ndarray, SigmaDiagnostics]:
    """Estimate adaptive sigma(x) by searching for κ plateaus.

    Args:
        n_e: electron number density (m^-3).
        T_e: electron temperature (K).
        grid: spatial grid positions (meters or arbitrary units).
        velocity: raw bulk velocity profile (m/s).
        sound_speed: raw sound speed profile (m/s).
        sigma_factor: multiplicative factor applied to σ₀.
        ladder: multiplicative ladder for σ exploration.
        epsilon: plateau tolerance parameter.
        minimum_sigma_cells: lower bound for σ expressed in grid cells.

    Returns:
        sigma_map: per-cell sigma in grid cells.
        diagnostics: SigmaDiagnostics describing ladder exploration.
    """

    if n_e.size == 0 or T_e.size == 0 or velocity.size == 0 or sound_speed.size == 0:
        raise ValueError("estimate_sigma_map requires non-empty diagnostics arrays")

    x = grid if grid is not None and grid.size else np.arange(len(velocity), dtype=float)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Grid for sigma estimation must be 1D")

    if len(x) != len(velocity):
        raise ValueError("Grid and velocity arrays must share the same length")

    dx = float(np.mean(np.diff(x))) if len(x) > 1 else 1.0
    dx = max(dx, 1e-12)

    n_e = np.asarray(n_e, dtype=float)
    T_e = np.asarray(T_e, dtype=float)

    ne_safe = np.where(n_e > 0, n_e, 1e6)  # prevent divide-by-zero; fallback to small density

    lambda_D = np.sqrt(np.maximum(EPSILON_0 * T_e * _E_CHARGE / (ne_safe * _E_CHARGE**2), 0.0))
    omega_pe = np.sqrt(ne_safe * _E_CHARGE**2 / (EPSILON_0 * m_e))
    skin_depth = np.where(omega_pe > 0.0, c / omega_pe, np.inf)

    sigma_length = np.maximum(lambda_D, skin_depth) * sigma_factor
    sigma_cells = np.clip(sigma_length / dx, minimum_sigma_cells, None)

    ladder = tuple(float(f) for f in ladder)
    sigma_candidates = [np.asarray(sigma_cells * factor, dtype=float) for factor in ladder]

    sigma_means = np.array([float(np.nanmean(candidate)) for candidate in sigma_candidates], dtype=float)
    kappa_means = np.zeros(len(sigma_candidates), dtype=float)
    horizon_counts = np.zeros(len(sigma_candidates), dtype=int)

    for idx, candidate in enumerate(sigma_candidates):
        v_smooth = apply_sigma_smoothing(velocity, candidate)
        cs_smooth = apply_sigma_smoothing(sound_speed, candidate)
        try:
            horizons = find_horizons_with_uncertainty(x, v_smooth, cs_smooth)
        except AssertionError:
            horizons = None
        if horizons is not None and horizons.kappa.size:
            kappa_means[idx] = float(np.mean(horizons.kappa))
            horizon_counts[idx] = int(horizons.kappa.size)
        else:
            kappa_means[idx] = 0.0
            horizon_counts[idx] = 0

    plateau_index = _select_plateau_index(sigma_means, kappa_means, epsilon)

    diagnostics = SigmaDiagnostics(
        sigma_means=sigma_means,
        kappa_means=kappa_means,
        horizon_counts=horizon_counts,
        plateau_index=plateau_index,
        ladder=tuple(ladder),
        epsilon=epsilon,
    )

    return sigma_candidates[plateau_index], diagnostics


def apply_sigma_smoothing(data: np.ndarray, sigma_map: np.ndarray) -> np.ndarray:
    """Smooth data using per-cell Gaussian kernels specified in sigma_map."""

    data = np.asarray(data, dtype=float)
    sigma_map = np.asarray(sigma_map, dtype=float)
    if data.size == 0:
        return data
    if sigma_map.shape != data.shape:
        raise ValueError("Sigma map must match data shape for smoothing")

    smoothed = np.empty_like(data, dtype=float)
    n = data.size
    for i in range(n):
        sigma = float(max(sigma_map[i], 0.0))
        if sigma <= 1e-6:
            smoothed[i] = data[i]
            continue
        radius = max(int(3.0 * sigma + 0.5), 1)
        left = max(0, i - radius)
        right = min(n - 1, i + radius)
        window = data[left:right + 1]
        offsets = np.arange(left, right + 1, dtype=float) - float(i)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights_sum = weights.sum()
        if weights_sum <= 0.0:
            smoothed[i] = data[i]
        else:
            smoothed[i] = float(np.dot(weights, window) / weights_sum)
    return smoothed


def _select_plateau_index(sigma_means: np.ndarray, kappa_means: np.ndarray, epsilon: float) -> int:
    if sigma_means.size == 0:
        return 0
    if sigma_means.size == 1:
        return 0

    first_derivative = np.gradient(kappa_means, sigma_means)
    second_derivative = np.gradient(first_derivative, sigma_means)

    for idx in range(1, len(sigma_means)):
        if abs(second_derivative[idx]) < epsilon * abs(first_derivative[idx]):
            return idx
    return len(sigma_means) - 1


