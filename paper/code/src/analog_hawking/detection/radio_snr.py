"""
Radio-band detection utilities for thermal spectra.

Provides:
- band_power_from_spectrum(f, psd, f_center, B)
- equivalent_signal_temperature(P_sig, B)
- sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)
"""

from __future__ import annotations

import numpy as np
from scipy.constants import k


def band_power_from_spectrum(frequencies: np.ndarray,
                             power_spectrum: np.ndarray,
                             f_center: float,
                             bandwidth: float) -> float:
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
    return float(np.trapezoid(power_spectrum[mask], x=frequencies[mask]))


def equivalent_signal_temperature(P_sig: float, bandwidth: float) -> float:
    """Convert in-band power to equivalent antenna temperature via radiometer relation.

    T_sig = P_sig / (k B).
    """
    if bandwidth <= 0:
        return 0.0
    return float(P_sig / (k * bandwidth))


def sweep_time_for_5sigma(T_sys_vals: np.ndarray,
                          B_vals: np.ndarray,
                          T_sig: float) -> np.ndarray:
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

