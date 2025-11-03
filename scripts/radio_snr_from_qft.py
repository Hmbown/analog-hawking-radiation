#!/usr/bin/env python3
"""
Compute radio SNR sweep from the actual QFT spectrum produced by
calculate_hawking_spectrum(kappa) in hawking_detection_experiment.

Usage:
  python scripts/radio_snr_from_qft.py  # uses a default small kappa→radio band

Saves radio_snr_from_qft.png
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scipy.constants import hbar, k, pi

from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from hawking_detection_experiment import calculate_hawking_spectrum


def default_kappa_for_radio(T_H=0.01):
    return 2*pi*k*T_H / hbar


def main():
    kappa = default_kappa_for_radio(0.01)
    graybody_profile = None
    sidecar_path = os.path.join('results', 'warpx_profiles.npz')
    if os.path.exists(sidecar_path):
        npz = np.load(sidecar_path)
        graybody_profile = {
            'x': npz['grid'],
            'v': npz['velocity'],
            'c_s': npz['sound_speed'],
        }
    spec = calculate_hawking_spectrum(kappa, graybody_profile=graybody_profile)
    if not spec.get('success', False):
        print('Spectrum calculation failed')
        return
    frequencies = spec['frequencies']
    power_spectrum = spec['power_spectrum']
    f_center = spec['peak_frequency']

    B_vals = np.logspace(5, 9, 25)  # 100 kHz .. 1 GHz
    T_sys_vals = np.linspace(5, 80, 25)

    # Reference to compute T_sig at B_ref, then sweep
    B_ref = 1e8
    P_sig = band_power_from_spectrum(frequencies, power_spectrum, f_center, B_ref)
    T_sig = equivalent_signal_temperature(P_sig, B_ref)

    Tgrid = sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)

    plt.figure(figsize=(8, 5))
    im = plt.contourf(B_vals*1e-6, T_sys_vals, np.log10(Tgrid/3600), levels=20, cmap='viridis')
    plt.colorbar(im, label='log10(Time for 5σ) [hours]')
    plt.xlabel('Bandwidth [MHz]')
    plt.ylabel('System Temperature T_sys [K]')
    plt.title('Radiometer 5σ Time (from QFT spectrum)')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = os.path.join('figures', 'radio_snr_from_qft.png')
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
