#!/usr/bin/env python3
"""
Apples-to-apples comparison of fluid-only vs hybrid (fluid + plasma mirror) spectra and t_5sigma.
Uses identical graybody transmission, area, solid angle, and efficiency for both branches.
Saves figures/hybrid_apples_to_apples.png and prints metrics.
"""
from __future__ import annotations

import os
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from scipy.constants import e, epsilon_0, m_e
from scripts.hawking_detection_experiment import calculate_hawking_spectrum

from analog_hawking.detection.hybrid_spectrum import calculate_enhanced_hawking_spectrum
from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.horizon_hybrid import HybridHorizonParams, find_hybrid_horizons
from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend


def main() -> int:
    # Match pipeline demo defaults
    plasma_density = 5e17
    laser_wavelength = 800e-9
    laser_intensity = 5e17
    temperature_constant = 1e4
    grid = np.linspace(0.0, 50e-6, 512)

    # Configure fluid backend
    backend = FluidBackend()
    backend.configure(
        {
            "plasma_density": plasma_density,
            "laser_wavelength": laser_wavelength,
            "laser_intensity": laser_intensity,
            "grid": grid,
            "temperature_settings": {"constant": temperature_constant},
            "use_fast_magnetosonic": False,
            "scale_with_intensity": True,
        }
    )
    state = backend.step(0.0)

    # Horizons
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)
    if not horizons.positions.size:
        print("No horizons found; aborting apples-to-apples comparison")
        return 0

    # Near-horizon profile window for graybody
    x = state.grid
    v = state.velocity
    cs = state.sound_speed
    idx = int(np.clip(np.searchsorted(x, float(horizons.positions[0])), 1, len(x) - 2))
    f = np.abs(v) - cs
    df_dx = np.gradient(f, x)
    slope = float(abs(df_dx[idx])) if np.isfinite(df_dx[idx]) else 0.0
    dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
    f_thresh = 0.1 * float(np.nanmax(np.abs(f))) if np.isfinite(np.nanmax(np.abs(f))) else 0.0
    if slope > 0 and dx > 0:
        L_half = f_thresh / slope
        cells_half = int(np.clip(np.ceil(L_half / dx), 10, len(x) // 5))
    else:
        cells_half = 20
    i0 = max(0, idx - cells_half)
    i1 = min(len(x), idx + cells_half + 1)
    gray_profile = {"x": x[i0:i1], "v": v[i0:i1], "c_s": cs[i0:i1]}

    # Fluid spectrum (aperture/solid angle consistent with pipeline)
    kappa0 = float(horizons.kappa[0])
    spec_fluid = calculate_hawking_spectrum(
        kappa0,
        graybody_profile=gray_profile,
        emitting_area_m2=1e-6,
        solid_angle_sr=5e-2,
        coupling_efficiency=0.1,
    )
    if not spec_fluid.get("success"):
        print("Failed to compute fluid spectrum")
        return 1

    # Mirror dynamics
    n_p0 = 1.0e24
    omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
    p = PlasmaMirrorParams(
        n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel"
    )
    t_m = np.linspace(0.0, 100e-15, 401)
    mirror = calculate_plasma_mirror_dynamics(state.grid, float(laser_intensity), p, t_m)

    # Hybrid horizons and spectrum
    hh = find_hybrid_horizons(
        state.grid, state.velocity, state.sound_speed, mirror, HybridHorizonParams()
    )
    if not hh.hybrid_kappa.size:
        print("No hybrid horizons computed")
        return 0
    k_fluid_ref = float(np.max(horizons.kappa))
    w = float(np.max(hh.coupling_weight)) if hh.coupling_weight.size else 0.0
    spec_hybrid = calculate_enhanced_hawking_spectrum(
        k_fluid_ref,
        float(mirror.kappa_mirror),
        w,
        emitting_area_m2=1e-6,
        solid_angle_sr=5e-2,
        coupling_efficiency=0.1,
        graybody_profile=gray_profile,
    )
    if not spec_hybrid.get("success"):
        print("Failed to compute hybrid spectrum")
        return 1

    # Apples-to-apples in-band power at fluid peak
    freqs_f = np.asarray(spec_fluid["frequencies"])  # type: ignore[index]
    P_f = np.asarray(spec_fluid["power_spectrum"])  # type: ignore[index]
    peak_frequency = float(spec_fluid.get("peak_frequency", freqs_f[np.argmax(P_f)]))
    inband_power_f = band_power_from_spectrum(freqs_f, P_f, peak_frequency, 1e8)
    if inband_power_f == 0.0:
        f_lo = peak_frequency - 0.5 * 1e8
        f_hi = peak_frequency + 0.5 * 1e8
        fb = np.linspace(f_lo, f_hi, 2001)
        fb = np.clip(fb, float(freqs_f[0]), float(freqs_f[-1]))
        if fb[-1] > fb[0]:
            psd_band = np.interp(fb, freqs_f, P_f)
            inband_power_f = float(np.trapezoid(psd_band, x=fb))

    freqs_h = np.asarray(spec_hybrid["frequencies"])  # type: ignore[index]
    P_h = np.asarray(spec_hybrid["power_spectrum"])  # type: ignore[index]
    inband_power_h = band_power_from_spectrum(freqs_h, P_h, peak_frequency, 1e8)
    if inband_power_h == 0.0:
        f_lo = peak_frequency - 0.5 * 1e8
        f_hi = peak_frequency + 0.5 * 1e8
        fb = np.linspace(f_lo, f_hi, 2001)
        fb = np.clip(fb, float(freqs_h[0]), float(freqs_h[-1]))
        if fb[-1] > fb[0]:
            psd_band = np.interp(fb, freqs_h, P_h)
            inband_power_h = float(np.trapezoid(psd_band, x=fb))

    T_sig_f = equivalent_signal_temperature(inband_power_f, 1e8)
    T_sig_h = equivalent_signal_temperature(inband_power_h, 1e8)
    t_f = (
        float(sweep_time_for_5sigma(np.array([30.0]), np.array([1e8]), T_sig_f)[0, 0])
        if T_sig_f > 0
        else float("inf")
    )
    t_h = (
        float(sweep_time_for_5sigma(np.array([30.0]), np.array([1e8]), T_sig_h)[0, 0])
        if T_sig_h > 0
        else float("inf")
    )

    # Plot overlay
    plt.figure(figsize=(7, 4))
    plt.loglog(freqs_f, P_f, label="fluid (profile graybody)")
    plt.loglog(freqs_h, P_h, label="hybrid (same graybody)")
    plt.axvline(peak_frequency, color="k", ls=":", lw=1, label="fluid peak")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [W/Hz]")
    plt.legend()
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    out = os.path.join("figures", "hybrid_apples_to_apples.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out}")
    print(f"T_sig fluid={T_sig_f:.3e} K, t5sigma fluid={t_f:.3e} s")
    print(f"T_sig hybrid={T_sig_h:.3e} K, t5sigma hybrid={t_h:.3e} s")
    print(f"Improvement factor (t_fluid/t_hybrid) = {t_f/max(t_h,1e-30):.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
