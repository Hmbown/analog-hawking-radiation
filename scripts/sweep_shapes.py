#!/usr/bin/env python3
"""
Batch sweep of multi-beam shapes/geometries with conserved total power and
coarse-grained envelope-scale response, saving figures and JSON results.

Outputs:
  - results/enhancement_stats.json
  - figures/enhancement_bar.png
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from analog_hawking.physics_engine.multi_beam_superposition import simulate_gradient_enhancement


def ensure_dirs():
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)


def sweep():
    ensure_dirs()
    results = {}
    common = dict(
        wavelength=800e-9,
        w0=5e-6,
        I_total=1.0,
        grid_half_width=12e-6,
        n_grid=161,
        n_time=12,
        radius_for_max=2.5e-6,
        coarse_grain_length=2.5e-6,
    )

    configs = [
        ("single", {}),
        ("two_opposed", {}),
        ("triangular", {}),
        ("square", {}),
        ("pentagram", {}),
        ("hexagon", {}),
        ("standing_wave", {}),
        ("ring", {"ring_N": 8}),
        ("ring", {"ring_N": 12}),
        ("angled_crossing", {"angle_deg": 10.0}),
        ("angled_crossing", {"angle_deg": 20.0}),
        ("angled_crossing", {"angle_deg": 40.0}),
        # non-equal weights example: 2-strong + 2-weak square
        ("square", {"weights": [0.35, 0.35, 0.15, 0.15]}),
    ]

    for name, kw in configs:
        res = simulate_gradient_enhancement(name, phase_align=True, **common, **kw)
        key = f"{name}_{kw}" if kw else name
        results[key] = {
            "enhancement": float(res["enhancement"]),
            "kappa_surrogate_enhancement": float(res.get("kappa_surrogate_enhancement") or np.nan),
        }

    with open("results/enhancement_stats.json", "w") as f:
        json.dump(results, f, indent=2)

    # Bar chart
    labels = list(results.keys())
    enh = [results[k]["enhancement"] for k in labels]
    x = np.arange(len(labels))
    plt.figure(figsize=(12, 5))
    plt.bar(x, enh)
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("Gradient enhancement (vs single)")
    plt.title("Multi-beam gradient enhancements (conserved power, coarse-grained)")
    plt.tight_layout()
    plt.savefig("figures/enhancement_bar.png", dpi=200)
    print("Saved results/enhancement_stats.json and figures/enhancement_bar.png")


if __name__ == "__main__":
    sweep()
