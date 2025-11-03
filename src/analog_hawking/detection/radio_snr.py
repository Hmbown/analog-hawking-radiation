"""
Radio-band detection utilities for thermal spectra, extended for 3D volume integration and Unruh metrics.

Provides:
- band_power_from_spectrum(f, psd, f_center, B) [1D/2D]
- volume_integrated_psd for 3D PSD over volumes
- equivalent_signal_temperature(P_sig, B)
- sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)
- unruh_correlation for simple Unruh mode entanglement proxy via correlation functions
"""

from __future__ import annotations

import numpy as np
from scipy.constants import k
from scipy.stats import pearsonr


def band_power_from_spectrum(
    frequencies: np.ndarray, power_spectrum: np.ndarray, f_center: float, bandwidth: float
) -> float:
    """Integrate power spectrum over a band centered at ``f_center`` with width ``bandwidth``.

    Args:
        frequencies: 1D array of frequency samples (Hz)
        power_spectrum: 1D array of spectral power density (W/Hz) aligned with ``frequencies``
        f_center: Center frequency (Hz)
        bandwidth: Bandwidth (Hz)

    Returns:
        In-band power (W).
    """
    f_lo = f_center - 0.5 * bandwidth
    f_hi = f_center + 0.5 * bandwidth
    if f_lo >= f_hi:
        return 0.0
    mask = (frequencies >= f_lo) & (frequencies <= f_hi)
    if not np.any(mask):
        return 0.0
    # Use trapezoidal integration; prefer numpy.trapezoid if available, else fall back to trapz
    try:
        trap = np.trapezoid  # type: ignore[attr-defined]
    except AttributeError:
        trap = np.trapz  # Older NumPy
    return float(trap(power_spectrum[mask], x=frequencies[mask]))


def volume_integrated_psd(
    frequencies: np.ndarray,
    psd_3d: np.ndarray,  # Shape (nf, nx, ny, nz) or (nx, ny, nz, nf)
    volume_elements: np.ndarray,  # dx*dy*dz per voxel
    f_center: float,
    bandwidth: float,
    axis: int = -1,  # Frequency axis
) -> float:
    """Integrate 3D PSD over volume and band for total power.

    Args:
        frequencies: 1D array of frequency samples (Hz)
        psd_3d: 3D/4D array of PSD (W/Hz per voxel)
        volume_elements: Volume per voxel (m^3)
        f_center, bandwidth: As in band_power_from_spectrum
        axis: Frequency axis in psd_3d

    Returns:
        Total in-band power (W).
    """
    # Integrate over frequency band first
    f_lo, f_hi = f_center - 0.5 * bandwidth, f_center + 0.5 * bandwidth
    mask = (frequencies >= f_lo) & (frequencies <= f_hi)
    if not np.any(mask):
        return 0.0
    try:
        trap = np.trapezoid  # type: ignore[attr-defined]
    except AttributeError:
        trap = np.trapz
    psd_band = trap(psd_3d.take(mask, axis=axis), x=frequencies[mask], axis=axis)
    # Integrate over volume
    total_power = np.sum(psd_band * volume_elements)
    return float(total_power)


def equivalent_signal_temperature(P_sig: float, bandwidth: float) -> float:
    """Convert in-band power to equivalent antenna temperature via radiometer relation.

    T_sig = P_sig / (k B).
    """
    if bandwidth <= 0:
        return 0.0
    return float(P_sig / (k * bandwidth))


def unruh_correlation(
    psd_hawking: np.ndarray,
    psd_unruh: np.ndarray,
    frequencies: np.ndarray,
    corr_window: float = 0.1,  # Relative bandwidth for correlation
) -> float:
    """Simple correlation metric between Hawking and Unruh modes as entanglement proxy.

    Computes Pearson correlation over frequency window around peak.
    Assumes psd_hawking and psd_unruh are PSD arrays for partner modes.

    Args:
        psd_hawking, psd_unruh: PSD arrays (W/Hz)
        frequencies: Frequency array (Hz)
        corr_window: Relative bandwidth for local correlation

    Returns:
        Pearson r in [-1,1]; higher |r| indicates stronger mode entanglement.
    """
    # Find peak frequency (assume shared)
    peak_idx = np.argmax(psd_hawking)
    f_peak = frequencies[peak_idx]
    window = corr_window * f_peak
    mask = (frequencies >= f_peak - 0.5 * window) & (frequencies <= f_peak + 0.5 * window)
    if not np.any(mask):
        return 0.0
    haw_mask = psd_hawking[mask]
    unr_mask = psd_unruh[mask]
    if len(haw_mask) < 2:
        return 0.0
    r, _ = pearsonr(haw_mask, unr_mask)
    return float(r)


def sweep_time_for_5sigma(T_sys_vals: np.ndarray, B_vals: np.ndarray, T_sig: float) -> np.ndarray:
    """Compute integration time grid for 5σ detection using the radiometer equation.

    For SNR = (T_sig/T_sys) * sqrt(B t), set SNR = 5 and solve for t.
    t = (5 * T_sys / (T_sig * sqrt(B)))^2
    """
    T_sys_vals = np.asarray(T_sys_vals, dtype=float)
    B_vals = np.asarray(B_vals, dtype=float)
    T = np.empty((T_sys_vals.size, B_vals.size), dtype=float)
    if T_sig <= 0:
        T.fill(np.inf)
        return T
    for i, T_sys in enumerate(T_sys_vals):
        for j, B in enumerate(B_vals):
            if T_sys <= 0 or B <= 0:
                T[i, j] = np.inf
            else:
                T[i, j] = (5.0 * T_sys / (T_sig * np.sqrt(B))) ** 2
    return T
