"""
Utilities to normalize Hawking spectra by surface gravity and quantify
universality-collapse across disparate flow profiles, extended for PIC data and MLE κ recovery.

Key functions:
- omega_over_kappa_axis(frequencies, kappa)
- resample_on_x(X, Y, X_common)
- collapse_stats(curves)
- mle_kappa_recovery(psd_data, frequencies, initial_guess=1e12)
- band_temperature_and_t5sig(f, psd, B=1e8, T_sys=30.0)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from ..physics_engine.optimization.graybody_1d import compute_graybody
from ..physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory
from .radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)


def omega_over_kappa_axis(frequencies: Iterable[float], kappa: float) -> np.ndarray:
    """Return dimensionless axis x = ω/κ from frequencies (Hz) and κ (s⁻¹)."""
    f = np.asarray(list(frequencies), dtype=float)
    if kappa <= 0:
        # Avoid division by zero: return zeros (caller should mask)
        return np.zeros_like(f)
    omega = 2.0 * np.pi * f
    return omega / float(kappa)


def resample_on_x(x: np.ndarray, y: np.ndarray, x_common: np.ndarray) -> np.ndarray:
    """Linearly resample y(x) onto x_common; extrapolation is clipped to endpoints."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xc = np.asarray(x_common, dtype=float)
    if x.size == 0 or y.size == 0 or xc.size == 0:
        return np.zeros_like(xc)
    # Ensure strictly increasing for interpolation
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    # Clip to valid interpolation range
    x_min, x_max = float(x_sorted[0]), float(x_sorted[-1])
    xc_clipped = np.clip(xc, x_min, x_max)
    return np.interp(xc_clipped, x_sorted, y_sorted)


@dataclass
class CollapseStats:
    grid: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    rms_relative: float
    per_curve_rms_relative: List[float]


def collapse_stats(curves: List[np.ndarray]) -> CollapseStats:
    """Given a list of curves sampled on the same grid, compute collapse metrics.

    The primary metric is the RMS of relative deviation from the family mean
    across the grid, averaged over curves. Values near or below 0.1 indicate
    a reasonably tight collapse under the user's acceptance criterion.
    """
    if not curves:
        return CollapseStats(
            grid=np.array([]),
            mean=np.array([]),
            std=np.array([]),
            rms_relative=np.nan,
            per_curve_rms_relative=[],
        )

    Y = np.vstack([np.asarray(c, dtype=float) for c in curves])
    mu = np.mean(Y, axis=0)
    sigma = np.std(Y, axis=0)
    # Avoid division by zero in relative deviation by flooring with small epsilon
    denom = np.clip(np.abs(mu), 1e-30, None)
    rel_dev = np.abs((Y - mu) / denom)
    per_curve = np.sqrt(np.mean(rel_dev**2, axis=1))
    overall = float(np.mean(per_curve))

    # grid is not known here; caller provides it separately alongside CollapseStats
    return CollapseStats(
        grid=np.arange(mu.size),
        mean=mu,
        std=sigma,
        rms_relative=overall,
        per_curve_rms_relative=[float(v) for v in per_curve],
    )


def mle_kappa_recovery(
    frequencies: np.ndarray,
    observed_psd: np.ndarray,
    x_profile: np.ndarray,
    v_profile: np.ndarray,
    cs_profile: np.ndarray,
    initial_guess: float = 1e12,
    bounds: Tuple[float, float] = (1e10, 1e15),
) -> Tuple[float, float]:
    """Maximum likelihood estimation of κ from observed PSD using model comparison.

    Minimizes negative log-likelihood between observed PSD and model Hawking spectrum
    for varying κ, using fixed profiles for graybody and QFT.

    Args:
        frequencies: Observed frequencies (Hz)
        observed_psd: Observed power spectrum (W/Hz)
        x_profile, v_profile, cs_profile: Fixed flow profiles for graybody computation
        initial_guess: Starting κ (s^-1)
        bounds: Search bounds for κ

    Returns:
        (estimated_kappa, negative_log_likelihood)
    """

    def neg_log_lik(kappa: float) -> float:
        if kappa <= 0:
            return np.inf
        # Compute model PSD for this κ
        gb = compute_graybody(
            x_profile, v_profile, cs_profile, frequencies, method="acoustic_wkb", kappa=kappa
        )
        qft = QuantumFieldTheory(
            surface_gravity=kappa,
            emitting_area_m2=1e-6,
            solid_angle_sr=5e-2,
            coupling_efficiency=0.1,
        )
        model_psd = qft.hawking_spectrum(2 * np.pi * frequencies, transmission=gb.transmission)
        # Simple Gaussian likelihood (assuming Poisson-like for counts, but simplified)
        # Negative log lik = sum (observed - model)^2 / (2 * model) + const
        diff = observed_psd - model_psd
        nll = np.sum(diff**2 / (2 * np.maximum(model_psd, 1e-30)))
        return nll

    res = minimize_scalar(neg_log_lik, bounds=bounds, method="bounded", options={"xatol": 1e8})
    kappa_mle = res.x if res.success else initial_guess
    nll = res.fun
    return kappa_mle, nll


def band_temperature_and_t5sig(
    frequencies: np.ndarray,
    power_spectrum: np.ndarray,
    B: float = 1e8,
    T_sys: float = 30.0,
    f_center: float | None = None,
) -> Tuple[float, float]:
    """Compute equivalent signal temperature and 5σ integration time for a band.

    If f_center is not given, use the peak of the spectrum as the band center.
    Returns (T_sig, t_5sigma_seconds).
    """
    f = np.asarray(frequencies, dtype=float)
    psd = np.asarray(power_spectrum, dtype=float)
    if f_center is None:
        idx = int(np.argmax(psd)) if psd.size else 0
        f_center = float(f[idx]) if psd.size else float(0.0)
    P_sig = band_power_from_spectrum(f, psd, f_center=f_center, bandwidth=B)
    T_sig = equivalent_signal_temperature(P_sig, B)
    t_grid = sweep_time_for_5sigma(
        np.array([T_sys], dtype=float), np.array([B], dtype=float), T_sig
    )
    t_5 = float(t_grid[0, 0])
    return T_sig, t_5
