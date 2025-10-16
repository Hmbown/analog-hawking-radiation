#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

import sys
from pathlib import Path
# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from hawking_detection_experiment import calculate_hawking_spectrum
from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from scipy.constants import hbar, k, pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import argparse


@dataclass
class FullPipelineSummary:
    plasma_density: float
    laser_wavelength: float
    laser_intensity: float
    temperature_constant: float
    magnetic_field: Optional[float]
    use_fast_magnetosonic: bool
    grid_points: int
    horizon_positions: list[float]
    kappa: list[float]
    spectrum_peak_frequency: Optional[float]
    inband_power_W: Optional[float]
    T_sig_K: Optional[float]
    t5sigma_s: Optional[float]
    T_H_K: Optional[float]
    t5sigma_TH_s: Optional[float]
    graybody_window_cells: int | None = None


def run_full_pipeline(
    plasma_density: float = 5e17,
    laser_wavelength: float = 800e-9,
    laser_intensity: float = 5e17,
    temperature_constant: float = 1e4,
    magnetic_field: Optional[float] = None,
    use_fast_magnetosonic: bool = False,
    scale_with_intensity: bool = True,
    grid_min: float = 0.0,
    grid_max: float = 50e-6,
    grid_points: int = 512,
    B_ref: float = 1e8,  # 100 MHz
    T_sys: float = 30.0,
    graybody_window_cells: Optional[int] = None,
    save_graybody_figure: bool = True,
) -> FullPipelineSummary:
    # 1) Configure backend
    grid = np.linspace(grid_min, grid_max, grid_points)
    backend = FluidBackend()
    cfg = {
        "plasma_density": plasma_density,
        "laser_wavelength": laser_wavelength,
        "laser_intensity": laser_intensity,
        "grid": grid,
        "temperature_settings": {"constant": temperature_constant},
        "use_fast_magnetosonic": bool(use_fast_magnetosonic),
        "scale_with_intensity": bool(scale_with_intensity),
    }
    if magnetic_field is not None:
        cfg["magnetic_field"] = float(magnetic_field)
    backend.configure(cfg)

    # 2) Step and collect state
    state = backend.step(0.0)

    # 3) Horizon detection
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)
    positions = horizons.positions.tolist() if horizons.positions.size else []
    kappa = horizons.kappa.tolist() if horizons.kappa.size else []

    # 4) QFT spectrum and detection metrics
    peak_frequency = None
    inband_power = None
    T_sig = None
    t5sigma = None
    T_H = None
    t5sigma_TH = None

    chosen_window_cells: Optional[int] = None
    if kappa:
        # Build a profile window around the first horizon for graybody
        x = state.grid
        v = state.velocity
        cs = state.sound_speed
        idx = int(np.clip(np.searchsorted(x, positions[0]), 1, len(x) - 2)) if positions else None

        # Adaptive window based on local slope of f(x)=|v|-c_s at the horizon
        # length scale ~ f_thresh / |f'|; choose f_thresh as 0.1*max(|f|) for a conservative near-horizon region
        if idx is not None:
            f = np.abs(v) - cs
            df_dx = np.gradient(f, x)
            slope = float(abs(df_dx[idx])) if np.isfinite(df_dx[idx]) else 0.0
            dx = float(x[1] - x[0]) if len(x) > 1 else 1.0
            f_thresh = 0.1 * float(np.nanmax(np.abs(f))) if np.isfinite(np.nanmax(np.abs(f))) else 0.0
            if graybody_window_cells is not None:
                chosen_window_cells = int(graybody_window_cells)
            else:
                if slope > 0 and dx > 0:
                    L_half = f_thresh / slope  # meters
                    cells_half = int(np.clip(np.ceil(L_half / dx), 10, len(x) // 5))
                else:
                    cells_half = 20
                chosen_window_cells = cells_half
            w = int(chosen_window_cells)
            i0 = max(0, idx - w)
            i1 = min(len(x), idx + w + 1)
            gray_profile = {"x": x[i0:i1], "v": v[i0:i1], "c_s": cs[i0:i1]}
        else:
            gray_profile = None

        # Spectrum with profile-derived graybody
        spec_prof = {}
        if gray_profile is not None:
            spec_prof = calculate_hawking_spectrum(
                float(kappa[0]),
                graybody_profile=gray_profile,
                emitting_area_m2=1e-6,
                solid_angle_sr=5e-2,
                coupling_efficiency=0.1,
            )
        # Spectrum with fallback graybody
        spec_fallback = calculate_hawking_spectrum(
            float(kappa[0]),
            emitting_area_m2=1e-6,
            solid_angle_sr=5e-2,
            coupling_efficiency=0.1,
        )

        # Prefer profile-derived transmission for metrics
        spec = spec_prof if spec_prof.get("success") else spec_fallback
        if spec.get("success"):
            freqs = spec["frequencies"]
            P = spec["power_spectrum"]
            peak_frequency = float(spec.get("peak_frequency", float(freqs[np.argmax(P)])))
            inband_power = band_power_from_spectrum(freqs, P, peak_frequency, B_ref)
            if inband_power == 0.0:
                f_lo = peak_frequency - 0.5 * B_ref
                f_hi = peak_frequency + 0.5 * B_ref
                fb = np.linspace(f_lo, f_hi, 2001)
                fb = np.clip(fb, float(freqs[0]), float(freqs[-1]))
                psd_band = np.interp(fb, freqs, P)
                inband_power = float(np.trapezoid(psd_band, x=fb))
            T_sig = equivalent_signal_temperature(inband_power, B_ref)
            if not np.isfinite(T_sig) or T_sig <= 0.0:
                T_sig = float(spec.get("temperature", 0.0))
            t_grid = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), T_sig)
            t5sigma = float(t_grid[0, 0])

        # Save comparison figure (profile vs fallback) unless suppressed (e.g. during sweeps)
        if save_graybody_figure:
            try:
                if spec_prof.get("success") and spec_fallback.get("success"):
                    freqs_prof = spec_prof["frequencies"]
                    P_prof = spec_prof["power_spectrum"]
                    freqs_fb = spec_fallback["frequencies"]
                    P_fb = spec_fallback["power_spectrum"]
                    plt.figure(figsize=(7, 4))
                    plt.loglog(freqs_fb, P_fb, label="fallback graybody", alpha=0.8)
                    plt.loglog(freqs_prof, P_prof, label="profile graybody", alpha=0.8)
                    plt.xlabel("Frequency [Hz]")
                    plt.ylabel("PSD [W/Hz]")
                    plt.legend()
                    plt.tight_layout()
                    os.makedirs("figures", exist_ok=True)
                    plt.savefig(os.path.join("figures", "graybody_impact.png"), dpi=200)
                    plt.close()
            except Exception:
                pass
        # Compute Hawking temperature and TH-based detection time surrogate
        T_H = float(hbar * float(kappa[0]) / (2.0 * pi * k))
        if T_H > 0:
            t_grid_TH = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), float(T_H))
            t5sigma_TH = float(t_grid_TH[0, 0])

    return FullPipelineSummary(
        plasma_density=plasma_density,
        laser_wavelength=laser_wavelength,
        laser_intensity=laser_intensity,
        temperature_constant=temperature_constant,
        magnetic_field=magnetic_field,
        use_fast_magnetosonic=use_fast_magnetosonic,
        grid_points=grid_points,
        horizon_positions=positions,
        kappa=kappa,
        spectrum_peak_frequency=peak_frequency,
        inband_power_W=inband_power,
        T_sig_K=T_sig,
        t5sigma_s=t5sigma,
        T_H_K=T_H,
        t5sigma_TH_s=t5sigma_TH,
        graybody_window_cells=chosen_window_cells,
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Use a horizon-forming demo configuration")
    p.add_argument("--intensity", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--window-cells", type=int, default=None)
    args = p.parse_args()

    kwargs = {}
    if args.demo:
        kwargs.update(dict(
            laser_intensity=5e17,
            temperature_constant=1e4,
            magnetic_field=None,
            use_fast_magnetosonic=False,
            scale_with_intensity=True,
        ))
    if args.intensity is not None:
        kwargs["laser_intensity"] = args.intensity
    if args.temperature is not None:
        kwargs["temperature_constant"] = args.temperature
    if args.window_cells is not None:
        kwargs["graybody_window_cells"] = args.window_cells

    summary = run_full_pipeline(**kwargs)
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "full_pipeline_summary.json")
    with open(out_path, "w") as f:
        json.dump(asdict(summary), f, indent=2)
    print(f"Saved pipeline summary to {out_path}")
    if summary.kappa:
        print(f"First kappa: {summary.kappa[0]:.3e} s^-1")
    if summary.t5sigma_s is not None:
        print(f"t_5sigma: {summary.t5sigma_s:.2e} s (T_sys=30K, B=100 MHz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
