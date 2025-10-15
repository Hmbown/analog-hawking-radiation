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

from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from hawking_detection_experiment import calculate_hawking_spectrum
from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from scipy.constants import hbar, k, pi


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


def run_full_pipeline(
    plasma_density: float = 5e17,
    laser_wavelength: float = 800e-9,
    laser_intensity: float = 5e16,
    temperature_constant: float = 5e5,
    magnetic_field: Optional[float] = 0.01,  # Tesla; set None to disable
    use_fast_magnetosonic: bool = True,
    scale_with_intensity: bool = False,
    grid_min: float = 0.0,
    grid_max: float = 50e-6,
    grid_points: int = 512,
    B_ref: float = 1e8,  # 100 MHz
    T_sys: float = 30.0,
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

    if kappa:
        spec = calculate_hawking_spectrum(float(kappa[0]))
        if spec.get("success"):
            freqs = spec["frequencies"]
            P = spec["power_spectrum"]
            peak_frequency = float(spec["peak_frequency"]) if "peak_frequency" in spec else float(freqs[np.argmax(P)])
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
                # Fallback: use Hawking temperature as an effective signal temperature
                T_sig = float(spec.get("temperature", 0.0))
            t_grid = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), T_sig)
            t5sigma = float(t_grid[0, 0])
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
    )


def main() -> int:
    summary = run_full_pipeline()
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
