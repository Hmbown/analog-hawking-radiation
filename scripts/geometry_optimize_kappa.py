#!/usr/bin/env python3
from __future__ import annotations

"""
Search beam geometries under a fixed total power budget and report
relative gradient/kappa surrogates using multi_beam_superposition.

Outputs
- results/geometry_vs_kappa.json
- figures/geometry_vs_kappa.png
"""

import json
import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analog_hawking.physics_engine.multi_beam_superposition import compare_configurations


def main() -> int:
    configs: List[str] = [
        'single', 'two_opposed', 'triangular', 'square', 'pentagram', 'hexagon',
        'standing_wave',
    ]

    # Envelope-scale, power-conserving settings; modest grid for speed
    sim = compare_configurations(
        configs=configs,
        wavelength=800e-9,
        w0=5e-6,
        I_total=1.0,
        grid_half_width=12e-6,
        n_grid=161,
        n_time=12,
        radius_for_max=2.5e-6,
        coarse_grain_length=None,
        phase_align=True,
        # κ surrogate knobs
        tau_response=10e-15,
        c_s_value=3e5,
    )

    # Collate and normalize κ surrogate against each config's own single-beam baseline
    order = configs
    enh = [float(sim[name]['enhancement']) for name in order]
    kappa_sur = []
    for name in order:
        ks = float(sim[name].get('kappa_surrogate_single', np.nan))
        km = float(sim[name].get('kappa_surrogate_multi', np.nan))
        ratio = km / max(ks, 1e-30) if np.isfinite(ks) and np.isfinite(km) else np.nan
        kappa_sur.append(ratio)

    os.makedirs('results', exist_ok=True)
    with open('results/geometry_vs_kappa.json', 'w') as f:
        json.dump({'order': order, 'enhancement': enh, 'kappa_surrogate_enhancement': kappa_sur}, f, indent=2)

    # Figure
    os.makedirs('figures', exist_ok=True)
    x = np.arange(len(order))
    width = 0.38
    plt.figure(figsize=(8, 4))
    plt.bar(x - width/2, enh, width=width, label='|∇I| enhancement')
    plt.bar(x + width/2, kappa_sur, width=width, label='κ surrogate enhancement')
    plt.xticks(x, [s.replace('_', '\n') for s in order], rotation=0)
    plt.ylabel('Relative enhancement (× single-beam)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('figures/geometry_vs_kappa.png', dpi=200)
    plt.close()

    print('Saved results/geometry_vs_kappa.json and figures/geometry_vs_kappa.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
