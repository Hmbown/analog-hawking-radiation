#!/usr/bin/env python3
"""
Compare fluid-only vs hybrid (fluid + plasma mirror) spectra and simple detection metrics.
Saves figures/hybrid_vs_fluid_spectrum.png and prints t_5sigma estimates.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from scipy.constants import k
from scripts.hawking_detection_experiment import calculate_hawking_spectrum
from analog_hawking.detection.hybrid_spectrum import calculate_enhanced_hawking_spectrum
from analog_hawking.detection.radio_snr import sweep_time_for_5sigma


def main():
    # Representative parameters (radio regime by design)
    kappa_fluid = 1.0e12  # s^-1
    kappa_mirror = 1.0e12  # s^-1 (order-of-magnitude match)
    w_couple = 0.3
    B_ref = 1e8  # 100 MHz
    T_sys = 30.0  # K

    # Fluid-only spectrum
    fluid = calculate_hawking_spectrum(kappa_fluid)
    if not fluid.get("success"):
        print("Failed to compute fluid-only spectrum")
        return 1

    # Hybrid spectrum
    hybrid = calculate_enhanced_hawking_spectrum(kappa_fluid, kappa_mirror, w_couple)
    if not hybrid.get("success"):
        print("Failed to compute hybrid spectrum")
        return 1

    # Plot
    f_fl = np.asarray(fluid["frequencies"])  # type: ignore[index]
    P_fl = np.asarray(fluid["power_spectrum"])  # type: ignore[index]
    f_hy = np.asarray(hybrid["frequencies"])  # type: ignore[index]
    P_hy = np.asarray(hybrid["power_spectrum"])  # type: ignore[index]

    plt.figure(figsize=(7, 4))
    plt.loglog(f_fl, P_fl, label="fluid-only")
    plt.loglog(f_hy, P_hy, label="hybrid (w=0.3)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [W/Hz]")
    plt.legend()
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "hybrid_vs_fluid_spectrum.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")

    # Simple t_5sigma comparison at each peak
    def _t5(spec: dict) -> float:
        f = np.asarray(spec["frequencies"])  # type: ignore[index]
        P = np.asarray(spec["power_spectrum"])  # type: ignore[index]
        if not P.size:
            return float("inf")
        f0 = float(spec.get("peak_frequency", f[np.argmax(P)]))
        f_lo, f_hi = f0 - 0.5 * B_ref, f0 + 0.5 * B_ref
        f_band = np.linspace(max(f_lo, float(f[0])), min(f_hi, float(f[-1])), 2001)
        if f_band[-1] <= f_band[0]:
            return float("inf")
        psd_band = np.interp(f_band, f, P)
        P_sig = float(np.trapz(psd_band, x=f_band))
        T_sig = P_sig / (k * B_ref) if B_ref > 0 else 0.0
        if T_sig <= 0:
            return float("inf")
        Tgrid = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), T_sig)
        return float(Tgrid[0, 0])

    t5_fluid = _t5(fluid)
    t5_hybrid = _t5(hybrid)
    print(f"t_5sigma (fluid):  {t5_fluid:.3e} s | {t5_fluid/3600:.3e} h")
    print(f"t_5sigma (hybrid): {t5_hybrid:.3e} s | {t5_hybrid/3600:.3e} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
