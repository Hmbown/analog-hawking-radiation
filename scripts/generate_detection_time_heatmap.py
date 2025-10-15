#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.hawking_detection_experiment import calculate_hawking_spectrum
from analog_hawking.detection.radio_snr import band_power_from_spectrum, equivalent_signal_temperature, sweep_time_for_5sigma


def main() -> int:
    cases_path = Path('results') / 'horizon_success_cases.json'
    if not cases_path.exists():
        print(f"Missing {cases_path}. Run scripts/run_param_sweep.py first.")
        return 1

    with open(cases_path, 'r') as f:
        cases = json.load(f)

    if not cases:
        print("No horizon-forming configurations found. Cannot generate detection time heatmap.")
        return 0

    # Use best case by kappa_max
    best = max(cases, key=lambda c: float(c.get('kappa_max', 0.0)))
    kappa = float(best.get('kappa_max', 0.0))
    if not np.isfinite(kappa) or kappa <= 0:
        print("Best case has invalid kappa_max; aborting heatmap generation.")
        return 1

    spec = calculate_hawking_spectrum(kappa)
    if not spec.get('success', False):
        print("Spectrum calculation failed; aborting heatmap generation.")
        return 1

    freqs = np.asarray(spec['frequencies'])
    P = np.asarray(spec['power_spectrum'])
    f_center = float(spec.get('peak_frequency', freqs[np.argmax(P)]))

    # Compute T_sig at a reference bandwidth using dense linear sampling around peak to avoid zero-mask on coarse log grid
    B_vals = np.logspace(5, 11, 61)  # 100 kHz .. 100 GHz
    T_sys_vals = np.linspace(5.0, 80.0, 41)  # 5 K .. 80 K

    B_ref = 1e8  # 100 MHz reference for T_sig
    f_band = np.linspace(f_center - 0.5 * B_ref, f_center + 0.5 * B_ref, 2001)
    if f_band[0] < freqs[0] or f_band[-1] > freqs[-1]:
        # Clamp the band to available spectrum if needed
        f_band = np.clip(f_band, freqs[0], freqs[-1])
    psd_band = np.interp(f_band, freqs, P)
    P_sig = float(np.trapezoid(psd_band, x=f_band))
    T_sig = equivalent_signal_temperature(P_sig, B_ref)

    Tgrid = sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)

    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    im = plt.contourf(B_vals * 1e-6, T_sys_vals, np.log10(Tgrid / 3600.0), levels=24, cmap='viridis')
    plt.colorbar(im, label='log10(t_5σ) [hours]')
    plt.xlabel('Bandwidth [MHz]')
    plt.ylabel('System Temperature T_sys [K]')
    plt.title('Detection time heatmap (best horizon configuration)')
    plt.tight_layout()
    out = Path('figures') / 'horizon_analysis_detection_time.png'
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    t_min = float(np.nanmin(Tgrid))
    print(f"Min t_5sigma across grid: {t_min:.3e} s ({t_min/3600:.3e} hours)")

    # Also produce a surrogate heatmap using T_H as brightness temperature
    T_H = float(spec.get('temperature', 0.0))
    if T_H > 0:
        Tgrid_TH = sweep_time_for_5sigma(T_sys_vals, B_vals, T_H)
        plt.figure(figsize=(8, 5))
        im2 = plt.contourf(B_vals * 1e-6, T_sys_vals, np.log10(Tgrid_TH / 3600.0), levels=24, cmap='plasma')
        plt.colorbar(im2, label='log10(t_5σ) [hours] (T_H surrogate)')
        plt.xlabel('Bandwidth [MHz]')
        plt.ylabel('System Temperature T_sys [K]')
        plt.title('Detection time heatmap (T_H as T_sig surrogate)')
        plt.tight_layout()
        out2 = Path('figures') / 'horizon_analysis_detection_time_TH.png'
        plt.savefig(out2, dpi=200)
        print(f"Saved {out2}")
        t_min_th = float(np.nanmin(Tgrid_TH))
        print(f"Min t_5sigma (T_H surrogate): {t_min_th:.3e} s ({t_min_th/3600:.3e} hours)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
