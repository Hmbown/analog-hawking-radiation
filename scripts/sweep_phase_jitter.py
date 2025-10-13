#!/usr/bin/env python3
"""
Phase jitter sweep for multi-beam enhancement stability.

Runs simulate_gradient_enhancement with phase_align=False multiple times to collect
statistics of enhancement under random phases, for several configurations.
Saves a bar chart with mean±std for each configuration.
"""
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.multi_beam_superposition import simulate_gradient_enhancement


def main():
    configs = ['two_opposed','triangular','square','pentagram','hexagon','standing_wave']
    n_trials = 30
    wavelength = 800e-9
    w0 = 5e-6
    coarse_len = 2.5e-6  # set to ~ skin depth or w0/2

    means = []
    stds = []
    for name in configs:
        vals = []
        rng = np.random.default_rng(42)
        for _ in range(n_trials):
            # simulate with random phases by setting phase_align=False
            res = simulate_gradient_enhancement(name, wavelength=wavelength, w0=w0, I_total=1.0,
                                                grid_half_width=12e-6, n_grid=141, n_time=12,
                                                radius_for_max=2.5e-6, phase_align=False,
                                                coarse_grain_length=coarse_len)
            vals.append(res['enhancement'])
        vals = np.array(vals)
        means.append(np.nanmean(vals))
        stds.append(np.nanstd(vals))

    x = np.arange(len(configs))
    plt.figure(figsize=(9,5))
    plt.bar(x, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=5)
    plt.xticks(x, configs, rotation=30)
    plt.ylabel('Gradient enhancement (mean ± std)')
    plt.title('Phase jitter robustness (equal total power, coarse-grained)')
    plt.tight_layout()
    os.makedirs('figures', exist_ok=True)
    out = os.path.join('figures', 'phase_jitter_stability.png')
    plt.savefig(out, dpi=200)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
