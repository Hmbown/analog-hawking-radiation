#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import hbar, k


def main() -> int:
    path = Path('results') / 'extended_param_sweep.json'
    if not path.exists():
        print(f"Missing {path}. Run scripts/run_param_sweep.py first.")
        return 1

    with open(path, 'r') as f:
        data = json.load(f)

    I_vals = np.asarray(data['intensity_values_Wcm2'], dtype=float)
    N_vals = np.asarray(data['density_values_cm3'], dtype=float)
    kappa_map = np.asarray(data['kappa_max_map'], dtype=float)  # shape (len(I), len(N))

    # Plot kappa map
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(8, 5))
    im = plt.contourf(N_vals, I_vals, np.log10(np.maximum(kappa_map, 1e-20)), levels=21, cmap='magma')
    plt.colorbar(im, label='log10(κ [s^-1])')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Plasma density [cm^-3]')
    plt.ylabel('Laser intensity [W/cm^2]')
    plt.title('Surface gravity κ across parameter space')
    plt.tight_layout()
    out1 = Path('figures') / 'horizon_analysis_kappa_map.png'
    plt.savefig(out1, dpi=200)
    print(f"Saved {out1}")

    # Hawking temperature map: T_H = hbar * kappa / (2*pi*k_B)
    T_H_map = (hbar * kappa_map) / (2.0 * np.pi * k)

    plt.figure(figsize=(8, 5))
    im = plt.contourf(N_vals, I_vals, np.log10(np.maximum(T_H_map, 1e-30)), levels=21, cmap='viridis')
    plt.colorbar(im, label='log10(T_H [K])')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Plasma density [cm^-3]')
    plt.ylabel('Laser intensity [W/cm^2]')
    plt.title('Hawking temperature T_H across parameter space')
    plt.tight_layout()
    out2 = Path('figures') / 'horizon_analysis_TH_map.png'
    plt.savefig(out2, dpi=200)
    print(f"Saved {out2}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
