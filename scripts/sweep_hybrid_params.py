#!/usr/bin/env python3
"""
Parameter sweep for hybrid coupling (apples-to-apples):
- Loops over coupling_strength and mirror scale D (eta_a fixed or varied lightly)
- Uses the same graybody transmission and normalization as fluid for fair comparison
- Saves results/hybrid_sweep.csv and figures/hybrid_t5_ratio_map.png (eta_a=1.0 slice)
"""
from __future__ import annotations

import csv
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


def compute_baseline(plasma_density=5e17, laser_wavelength=800e-9, laser_intensity=5e17, temperature_constant=1e4):
    grid = np.linspace(0.0, 50e-6, 512)
    backend = FluidBackend()
    backend.configure({
        "plasma_density": plasma_density,
        "laser_wavelength": laser_wavelength,
        "laser_intensity": laser_intensity,
        "grid": grid,
        "temperature_settings": {"constant": temperature_constant},
        "use_fast_magnetosonic": False,
        "scale_with_intensity": True,
    })
    state = backend.step(0.0)
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)
    if not horizons.positions.size:
        raise RuntimeError("No horizons found in baseline config")
    # Graybody near first horizon
    x = state.grid
    v = state.velocity
    cs = state.sound_speed
    idx = int(np.clip(np.searchsorted(x, float(horizons.positions[0])), 1, len(x)-2))
    f = np.abs(v) - cs
    df_dx = np.gradient(f, x)
    slope = float(abs(df_dx[idx])) if np.isfinite(df_dx[idx]) else 0.0
    dx = float(x[1]-x[0]) if len(x) > 1 else 1.0
    f_thresh = 0.1 * float(np.nanmax(np.abs(f))) if np.isfinite(np.nanmax(np.abs(f))) else 0.0
    if slope > 0 and dx > 0:
        L_half = f_thresh / slope
        cells_half = int(np.clip(np.ceil(L_half / dx), 10, len(x)//5))
    else:
        cells_half = 20
    i0 = max(0, idx - cells_half)
    i1 = min(len(x), idx + cells_half + 1)
    gray_profile = {"x": x[i0:i1], "v": v[i0:i1], "c_s": cs[i0:i1]}

    # Fluid spectrum
    kappa0 = float(horizons.kappa[0])
    spec_fluid = calculate_hawking_spectrum(
        kappa0,
        graybody_profile=gray_profile,
        emitting_area_m2=1e-6,
        solid_angle_sr=5e-2,
        coupling_efficiency=0.1,
    )
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
    T_sig_f = equivalent_signal_temperature(inband_power_f, 1e8)
    t_f = float(sweep_time_for_5sigma(np.array([30.0]), np.array([1e8]), T_sig_f)[0,0]) if T_sig_f > 0 else float("inf")

    return state, horizons, gray_profile, peak_frequency, T_sig_f, t_f


def main() -> int:
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    state, horizons, gray_profile, peak_frequency, T_sig_f, t_f = compute_baseline()
    k_fluid_ref = float(np.max(horizons.kappa))

    # Sweep grids
    coupling_strengths = np.array([0.05, 0.1, 0.2, 0.3, 0.5])
    Ds = np.array([5e-6, 10e-6, 20e-6, 40e-6])
    eta_a = 1.0

    # Storage
    ratio_map = np.full((Ds.size, coupling_strengths.size), np.nan)

    with open(os.path.join("results", "hybrid_sweep.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coupling_strength","D","eta_a","w_effective","kappa_mirror","T_sig_fluid","T_sig_hybrid","t5_fluid","t5_hybrid","ratio_fluid_over_hybrid"])

        for iD, D in enumerate(Ds):
            # Mirror for this D
            n_p0 = 1.0e24
            omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
            p = PlasmaMirrorParams(n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=float(D), eta_a=float(eta_a), model="anabhel")
            t_m = np.linspace(0.0, 100e-15, 401)
            mirror = calculate_plasma_mirror_dynamics(state.grid, float(5e17), p, t_m)

            for iw, cs in enumerate(coupling_strengths):
                hh = find_hybrid_horizons(state.grid, state.velocity, state.sound_speed, mirror, HybridHorizonParams(coupling_strength=float(cs)))
                w_eff = float(np.max(hh.coupling_weight)) if hh.coupling_weight.size else 0.0
                spec_h = calculate_enhanced_hawking_spectrum(
                    k_fluid_ref,
                    float(mirror.kappa_mirror),
                    w_eff,
                    emitting_area_m2=1e-6,
                    solid_angle_sr=5e-2,
                    coupling_efficiency=0.1,
                    graybody_profile=gray_profile,
                )
                if not spec_h.get("success"):
                    continue
                freqs_h = np.asarray(spec_h["frequencies"])  # type: ignore[index]
                P_h = np.asarray(spec_h["power_spectrum"])  # type: ignore[index]
                inband_power_h = band_power_from_spectrum(freqs_h, P_h, float(peak_frequency), 1e8)
                if inband_power_h == 0.0:
                    f_lo = float(peak_frequency) - 0.5 * 1e8
                    f_hi = float(peak_frequency) + 0.5 * 1e8
                    fb = np.linspace(f_lo, f_hi, 2001)
                    fb = np.clip(fb, float(freqs_h[0]), float(freqs_h[-1]))
                    if fb[-1] > fb[0]:
                        psd_band = np.interp(fb, freqs_h, P_h)
                        inband_power_h = float(np.trapezoid(psd_band, x=fb))
                T_sig_h = equivalent_signal_temperature(inband_power_h, 1e8)
                t_h = float(sweep_time_for_5sigma(np.array([30.0]), np.array([1e8]), T_sig_h)[0,0]) if T_sig_h > 0 else float("inf")
                ratio = t_f / max(t_h, 1e-30)
                ratio_map[iD, iw] = ratio

                w.writerow([cs, D, eta_a, w_eff, float(mirror.kappa_mirror), T_sig_f, T_sig_h, t_f, t_h, ratio])

    # Heatmap (eta_a=1 slice): rows=D, cols=coupling_strength
    plt.figure(figsize=(6, 3.5))
    im = plt.imshow(ratio_map, aspect="auto", origin="lower", interpolation="nearest",
                    extent=(coupling_strengths[0], coupling_strengths[-1], Ds[0]*1e6, Ds[-1]*1e6))
    plt.colorbar(im, label="t_5σ fluid / t_5σ hybrid (ratio)")
    plt.xlabel("coupling_strength")
    plt.ylabel("D [µm]")
    plt.title("Hybrid improvement (eta_a=1.0)")
    plt.tight_layout()
    out = os.path.join("figures", "hybrid_t5_ratio_map.png")
    plt.savefig(out, dpi=200)
    print(f"Saved {out} and results/hybrid_sweep.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
