#!/usr/bin/env python3
"""
Generate a simple radio SNR sweep plot for a given power spectrum.

This script expects to receive frequencies (Hz) and power spectrum (W/Hz) from
an upstream step; for demonstration, it synthesizes a notional spectrum with a
peak in the hundreds of MHz.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for CI
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analog_hawking.detection.radio_snr import band_power_from_spectrum, equivalent_signal_temperature, sweep_time_for_5sigma


def synth_spectrum():
    f = np.logspace(6, 11, 2000)  # 1 MHz .. 100 GHz
    # Gaussian-like bump centered at 200 MHz, arbitrary amplitude
    f0 = 2e8
    bw = 5e7
    psd = 1e-24 * np.exp(-0.5 * ((f - f0)/bw)**2)  # W/Hz
    return f, psd


def main():
    frequencies, power_spectrum = synth_spectrum()
    # Choose band centered at the spectral peak
    f_center = frequencies[np.argmax(power_spectrum)]
    # Sweep bandwidths and system temperatures
    B_vals = np.logspace(5, 9, 25)  # 100 kHz .. 1 GHz
    T_sys_vals = np.linspace(5, 80, 25)  # 5 K .. 80 K

    # For each bandwidth, compute in-band power and T_sig (fixed center)
    # Use representative B (e.g., 100 MHz) for T_sig; we’ll then sweep around it
    B_ref = 1e8
    P_sig = band_power_from_spectrum(frequencies, power_spectrum, f_center, B_ref)
    T_sig = equivalent_signal_temperature(P_sig, B_ref)

    Tgrid = sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)

    plt.figure(figsize=(8, 5))
    im = plt.contourf(B_vals*1e-6, T_sys_vals, np.log10(Tgrid/3600), levels=20, cmap='viridis')
    plt.colorbar(im, label='log10(Time for 5σ) [hours]')
    plt.xlabel('Bandwidth [MHz]')
    plt.ylabel('System Temperature T_sys [K]')
    plt.title('Radiometer 5σ Time (synthetic spectrum)')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = os.path.join('figures', 'radio_snr_sweep.png')
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
