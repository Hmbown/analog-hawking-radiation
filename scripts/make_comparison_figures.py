#!/usr/bin/env python3
from __future__ import annotations

"""
Generate comparison figures for README:
 - figures/graybody_methods_comparison.png
 - figures/kappa_methods_comparison.png

Also copies to docs/img/ if available.
"""

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody


def ensure_dirs():
    os.makedirs(ROOT / "figures", exist_ok=True)
    os.makedirs(ROOT / "docs" / "img", exist_ok=True)


def graybody_methods_comparison():
    # Simple near-horizon profile: v = a x, c = c0
    x = np.linspace(-1.0, 1.0, 2001)
    a = 1.0
    c0 = 0.2
    v = a * x
    c = np.full_like(x, c0)
    kappa = abs(a)

    freqs = np.logspace(-3, 1, 300)
    gb_dim = compute_graybody(x, v, c, freqs, method="dimensionless", kappa=kappa, alpha=1.0)
    gb_wkb = compute_graybody(x, v, c, freqs, method="acoustic_wkb", kappa=kappa, alpha=0.8)

    plt.figure(figsize=(7, 4))
    plt.semilogx(freqs, gb_dim.transmission, label="dimensionless", lw=2)
    plt.semilogx(freqs, gb_wkb.transmission, label="acoustic_wkb (α=0.8)", lw=2)
    plt.xlabel("Frequency [arb. units]")
    plt.ylabel("Transmission")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    out = ROOT / "figures" / "graybody_methods_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    (ROOT / "docs" / "img" / out.name).write_bytes(out.read_bytes())


def kappa_methods_comparison():
    # Linear v, constant c: expected κ_acoustic = |a|, κ_exact = |a|, κ_legacy = 0.5|a|
    a = 1.0
    c0 = 0.2
    x = np.linspace(-1.0, 1.0, 2001)
    v = a * x
    c = np.full_like(x, c0)

    ka = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic").kappa
    ke = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic_exact").kappa
    kl = find_horizons_with_uncertainty(x, v, c, kappa_method="legacy").kappa
    vals = [
        float(ka[0]) if ka.size else 0.0,
        float(ke[0]) if ke.size else 0.0,
        float(kl[0]) if kl.size else 0.0,
    ]
    labels = ["acoustic", "acoustic_exact", "legacy"]

    plt.figure(figsize=(6, 4))
    xs = np.arange(len(labels))
    plt.bar(xs, vals, color=["#4C78A8", "#72B7B2", "#F58518"])
    plt.xticks(xs, labels)
    plt.ylabel("κ [arb. s⁻¹]")
    plt.title("Surface gravity definitions (analytic profile)")
    plt.tight_layout()
    out = ROOT / "figures" / "kappa_methods_comparison.png"
    plt.savefig(out, dpi=200)
    plt.close()
    (ROOT / "docs" / "img" / out.name).write_bytes(out.read_bytes())


def main() -> int:
    ensure_dirs()
    graybody_methods_comparison()
    kappa_methods_comparison()
    print("Saved comparison figures to figures/ and docs/img/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
