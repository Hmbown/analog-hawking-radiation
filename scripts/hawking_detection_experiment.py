#!/usr/bin/env python3
"""
Hawking spectrum helper used by radio SNR scripts.

Exports:
- calculate_hawking_spectrum(kappa): returns dict with keys
  {success, frequencies, power_spectrum, peak_frequency, temperature}.

Frequency band gating:
- If T_H <= 10 K, select radio/microwave band (<= 1e11 Hz) to satisfy
  low-temperature scenarios and downstream gating checks.
- Otherwise, use a broader optical→X-ray band.
"""
from __future__ import annotations

import numpy as np
from scipy.constants import hbar, k, pi

from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody
from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def _temperature_from_kappa(kappa: float) -> float:
    return float(hbar * kappa / (2 * pi * k))


def _choose_frequency_band(T_H: float) -> tuple[float, float]:
    # Radio/microwave band (up to 100 GHz) for low temperatures
    if T_H <= 10.0:
        return 1e6, 1e11  # 1 MHz .. 100 GHz
    # Otherwise, use a wide band spanning THz to PHz
    return 1e12, 1e18


def calculate_hawking_spectrum(kappa: float,
                               n_points: int = 1200,
                               graybody_profile: dict | None = None,
                               emitting_area_m2: float | None = None,
                               solid_angle_sr: float | None = None,
                               coupling_efficiency: float = 1.0,
                               graybody_method: str = "dimensionless",
                               alpha_gray: float = 1.0) -> dict:
    try:
        T_H = _temperature_from_kappa(float(kappa))
        f_min, f_max = _choose_frequency_band(T_H)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))

        if emitting_area_m2 is None:
            emitting_area_m2 = 1.0
        if solid_angle_sr is None:
            solid_angle_sr = 1.0

        qft = QuantumFieldTheory(surface_gravity=float(kappa),
                                 emitting_area_m2=emitting_area_m2,
                                 solid_angle_sr=solid_angle_sr,
                                 coupling_efficiency=coupling_efficiency)
        omega = 2 * np.pi * freqs

        transmission = None
        uncertainties = None
        if graybody_profile is not None:
            x = np.asarray(graybody_profile["x"])
            velocity = np.asarray(graybody_profile["v"])
            sound_speed = np.asarray(graybody_profile["c_s"])
            graybody = compute_graybody(x, velocity, sound_speed, freqs, method=graybody_method, kappa=float(kappa), alpha=float(alpha_gray))
            transmission = graybody.transmission
            uncertainties = graybody.uncertainties

        psd = qft.hawking_spectrum(omega, transmission=transmission)

        peak_idx = int(np.argmax(psd)) if psd.size else 0
        peak_f = float(freqs[peak_idx]) if psd.size else float(0.0)

        result = {
            'success': True,
            'frequencies': freqs,
            'power_spectrum': psd,
            'peak_frequency': peak_f,
            'temperature': T_H,
        }
        if transmission is not None:
            result['transmission'] = transmission
        if uncertainties is not None:
            result['transmission_uncertainty'] = uncertainties
        return result
    except Exception as exc:  # pragma: no cover - robustness for scripts
        return {
            'success': False,
            'error': str(exc),
        }


if __name__ == "__main__":
    # Simple smoke test
    from scipy.constants import hbar, k, pi
    target_T = 0.01  # K (radio regime)
    kappa = 2 * pi * k * target_T / hbar
    out = calculate_hawking_spectrum(kappa)
    print(f"success={out.get('success')} peak={out.get('peak_frequency', 0):.3e} Hz band=[{out['frequencies'][0]:.3e},{out['frequencies'][-1]:.3e}] Hz")
