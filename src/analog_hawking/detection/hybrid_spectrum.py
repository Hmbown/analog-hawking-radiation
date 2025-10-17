from __future__ import annotations

import numpy as np
from scipy.constants import hbar, k, pi

from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory
from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody


def _temperature_from_kappa(kappa: float) -> float:
    return float(hbar * kappa / (2 * pi * k))


def _choose_frequency_band(T_H: float) -> tuple[float, float]:
    # Align band gating with scripts/hawking_detection_experiment.py
    if T_H <= 10.0:
        return 1e6, 1e11  # 1 MHz .. 100 GHz (radio/microwave)
    return 1e12, 1e18      # THz .. PHz


def calculate_enhanced_hawking_spectrum(kappa_fluid: float,
                                         kappa_mirror: float,
                                         coupling_weight: float,
                                         cross_coupling: float = 0.2,
                                         n_points: int = 1200,
                                         emitting_area_m2: float | None = 1.0,
                                         solid_angle_sr: float | None = 1.0,
                                         coupling_efficiency: float = 1.0,
                                         graybody_profile: dict | None = None) -> dict:
    """
    Construct an effective Hawking spectrum combining fluid and mirror contributions.

    T_f = ħ κ_f / (2π k_B), T_m = ħ κ_m / (2π k_B)
    T_eff = max(0, T_f + coupling_weight*T_m + cross_coupling*sqrt(T_f*T_m))

    Returns: dict like calculate_hawking_spectrum():
      {success, frequencies, power_spectrum, peak_frequency, temperature}
    """
    try:
        k_f = max(float(kappa_fluid), 0.0)
        k_m = max(float(kappa_mirror), 0.0)
        w = max(float(coupling_weight), 0.0)

        T_f = _temperature_from_kappa(k_f)
        T_m = _temperature_from_kappa(k_m)
        T_eff = float(max(0.0, T_f + w * T_m + float(cross_coupling) * np.sqrt(max(T_f, 0.0) * max(T_m, 0.0))))

        f_min, f_max = _choose_frequency_band(T_eff)
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), int(n_points))
        omega = 2 * np.pi * freqs

        qft = QuantumFieldTheory(surface_gravity=0.0,
                                 temperature=T_eff,
                                 emitting_area_m2=emitting_area_m2,
                                 solid_angle_sr=solid_angle_sr,
                                 coupling_efficiency=coupling_efficiency)

        transmission = None
        if graybody_profile is not None:
            x = np.asarray(graybody_profile["x"])  # type: ignore[index]
            v = np.asarray(graybody_profile["v"])  # type: ignore[index]
            c_s = np.asarray(graybody_profile["c_s"])  # type: ignore[index]
            gb = compute_graybody(x, v, c_s, freqs)
            transmission = gb.transmission

        psd = qft.hawking_spectrum(omega, transmission=transmission)
        peak_idx = int(np.argmax(psd)) if psd.size else 0
        peak_f = float(freqs[peak_idx]) if psd.size else float(0.0)

        return {
            'success': True,
            'frequencies': freqs,
            'power_spectrum': psd,
            'peak_frequency': peak_f,
            'temperature': T_eff,
        }
    except Exception as exc:  # pragma: no cover
        return {'success': False, 'error': str(exc)}
