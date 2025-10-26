#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

import sys
from pathlib import Path
from uuid import uuid4
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
from scipy.constants import hbar, k, pi, e, m_e, epsilon_0
from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.horizon_hybrid import (
    HybridHorizonParams,
    find_hybrid_horizons,
)
from analog_hawking.detection.hybrid_spectrum import (
    calculate_enhanced_hawking_spectrum,
)
try:
    from analog_hawking.inference.kappa_mle import infer_kappa as infer_kappa_from_psd
    from analog_hawking.inference.kappa_mle import make_graybody_model
except Exception:
    infer_kappa_from_psd = None  # type: ignore[assignment]
    make_graybody_model = None  # type: ignore[assignment]
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
    kappa_err: list[float]
    spectrum_peak_frequency: Optional[float]
    inband_power_W: Optional[float]
    T_sig_K: Optional[float]
    t5sigma_s: Optional[float]
    t5sigma_s_low: Optional[float] = None
    t5sigma_s_high: Optional[float] = None
    T_H_K: Optional[float] = None
    T_H_K_low: Optional[float] = None
    T_H_K_high: Optional[float] = None
    t5sigma_TH_s: Optional[float] = None
    graybody_window_cells: int | None = None
    hybrid_used: bool = False
    hybrid_kappa_eff: float | None = None
    hybrid_coupling_weight: float | None = None
    hybrid_T_sig_K: float | None = None
    hybrid_t5sigma_s: float | None = None
    kappa_inferred: float | None = None
    kappa_inferred_err: float | None = None
    kappa_credible_interval: tuple[float, float] | None = None


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
    kappa_method: str = "acoustic",
    graybody: str = "dimensionless",
    alpha_gray: float = 1.0,
    bands: Optional[str] = None,
    graybody_window_cells: Optional[int] = None,
    save_graybody_figure: bool = True,
    enable_hybrid: bool = False,
    hybrid_model: str = "anabhel",
    mirror_D: float = 10e-6,
    mirror_eta: float = 1.0,
    perform_kappa_inference: bool = False,
    inference_bounds: tuple[float, float] | None = None,
    inference_calls: int = 40,
) -> FullPipelineSummary:
    # Ensure defaults for variables used in return even if certain branches are not taken
    t5sigma_TH = float('inf')
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
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed, kappa_method=str(kappa_method))
    positions = horizons.positions.tolist() if horizons.positions.size else []
    kappa = horizons.kappa.tolist() if horizons.kappa.size else []
    kappa_err_list = horizons.kappa_err.tolist() if horizons.kappa_err.size else []

    # 4) QFT spectrum and detection metrics
    peak_frequency = None
    inband_power = None
    T_sig = None
    t5sigma = None
    t5sigma_low = None
    t5sigma_high = None
    T_H = None
    T_H_low = None
    T_H_high = None
    hybrid_used = False
    hybrid_kappa_eff = None
    hybrid_coupling_weight = None
    hybrid_T_sig = None
    hybrid_t5 = None
    kappa_inferred = None
    kappa_inferred_err = None
    kappa_ci = None

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
                graybody_method=str(graybody) if graybody in {"dimensionless", "wkb", "acoustic_wkb"} else "dimensionless",
                alpha_gray=float(alpha_gray),
            )
        # Spectrum with fallback graybody
        spec_fallback = calculate_hawking_spectrum(
            float(kappa[0]),
            emitting_area_m2=1e-6,
            solid_angle_sr=5e-2,
            coupling_efficiency=0.1,
            graybody_method=str(graybody) if graybody in {"dimensionless", "wkb", "acoustic_wkb"} else "dimensionless",
            alpha_gray=float(alpha_gray),
        )

        # Prefer profile-derived transmission for metrics
        spec = spec_prof if spec_prof.get("success") else spec_fallback
        if spec.get("success"):
            freqs = spec["frequencies"]
            P = np.asarray(spec["power_spectrum"])  # type: ignore[index]
            peak_frequency = float(spec.get("peak_frequency", float(freqs[np.argmax(P)])))
            inband_power = band_power_from_spectrum(freqs, P, peak_frequency, B_ref)
            if inband_power == 0.0:
                f_lo = peak_frequency - 0.5 * B_ref
                f_hi = peak_frequency + 0.5 * B_ref
                fb = np.linspace(f_lo, f_hi, 2001)
                fb = np.clip(fb, float(freqs[0]), float(freqs[-1]))
                psd_band = np.interp(fb, freqs, P)
                inband_power = float(np.trapz(psd_band, x=fb))
            T_sig = equivalent_signal_temperature(inband_power, B_ref)
            if not np.isfinite(T_sig) or T_sig <= 0.0:
                T_sig = float(spec.get("temperature", 0.0))
            t_grid = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), T_sig)
            t5sigma = float(t_grid[0, 0])

            # Transmission uncertainty envelope → band power low/high
            T_unc = spec.get("transmission_uncertainty")
            T_base = spec.get("transmission")
            if T_unc is not None and T_base is not None:
                T_base = np.asarray(T_base)
                T_unc = np.asarray(T_unc)
                # Avoid division by zero; where T_base=0, use absolute scaling via clipping
                scale_low = np.ones_like(P)
                scale_high = np.ones_like(P)
                with np.errstate(divide='ignore', invalid='ignore'):
                    scale_low = np.clip((T_base - T_unc) / np.clip(T_base, 1e-30, None), 0.0, None)
                    scale_high = np.clip((T_base + T_unc) / np.clip(T_base, 1e-30, None), 0.0, None)
                P_low = P * scale_low
                P_high = P * scale_high
                inband_low = band_power_from_spectrum(freqs, P_low, float(peak_frequency), B_ref)
                inband_high = band_power_from_spectrum(freqs, P_high, float(peak_frequency), B_ref)
                T_sig_low_env = equivalent_signal_temperature(inband_low, B_ref)
                T_sig_high_env = equivalent_signal_temperature(inband_high, B_ref)
                # Initialize bounds
                T_sig_b_lo = T_sig_low_env
                T_sig_b_hi = T_sig_high_env
                # Combine with kappa-derived bounds if available below

            # Propagate numerical κ uncertainty to bounds via recomputation
            if kappa_err_list:
                k0 = float(kappa[0])
                dk = float(kappa_err_list[0])
                if dk > 0 and np.isfinite(dk):
                    for sign, holder in ((-1.0, "low"), (1.0, "high")):
                        k_bound = max(k0 + sign * dk, 0.0)
                        spec_b = calculate_hawking_spectrum(
                            k_bound,
                            graybody_profile=gray_profile if 'gray_profile' in locals() else None,
                            emitting_area_m2=1e-6,
                            solid_angle_sr=5e-2,
                            coupling_efficiency=0.1,
                        )
                        if spec_b.get("success"):
                            fb = np.asarray(spec_b["frequencies"])  # type: ignore[index]
                            Pb = np.asarray(spec_b["power_spectrum"])  # type: ignore[index]
                            inband_b = band_power_from_spectrum(fb, Pb, float(peak_frequency), B_ref)
                            if inband_b == 0.0:
                                f_lo = float(peak_frequency) - 0.5 * B_ref
                                f_hi = float(peak_frequency) + 0.5 * B_ref
                                f_band = np.linspace(max(f_lo, float(fb[0])), min(f_hi, float(fb[-1])), 2001)
                                if f_band[-1] > f_band[0]:
                                    psd_band = np.interp(f_band, fb, Pb)
                                    inband_b = float(np.trapz(psd_band, x=f_band))
                            T_sig_b = equivalent_signal_temperature(inband_b, B_ref)
                            # Combine with transmission envelope if available, conservatively
                            if 'T_sig_b_lo' in locals() and 'T_sig_b_hi' in locals():
                                if holder == "low":
                                    T_sig_combined = min(T_sig_b, T_sig_b_lo)
                                else:
                                    T_sig_combined = max(T_sig_b, T_sig_b_hi)
                            else:
                                T_sig_combined = T_sig_b
                            t_b = float(sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), max(T_sig_combined, 0.0))[0, 0]) if T_sig_combined > 0 else float("inf")
                            if holder == "low":
                                t5sigma_low = t_b
                            else:
                                t5sigma_high = t_b

            # If only transmission envelope exists (no kappa bounds), still record bounds
            if t5sigma_low is None and 'T_sig_b_lo' in locals():
                t5sigma_low = float(sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), max(T_sig_b_lo, 0.0))[0, 0]) if T_sig_b_lo > 0 else float("inf")
            if t5sigma_high is None and 'T_sig_b_hi' in locals():
                t5sigma_high = float(sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), max(T_sig_b_hi, 0.0))[0, 0]) if T_sig_b_hi > 0 else float("inf")

            if perform_kappa_inference and infer_kappa_from_psd is not None and make_graybody_model is not None:
                try:
                    model = make_graybody_model(
                        freqs,
                        graybody_profile=gray_profile if 'gray_profile' in locals() else None,
                        graybody_method=str(graybody),
                        alpha_gray=float(alpha_gray),
                        emitting_area_m2=1e-6,
                        solid_angle_sr=5e-2,
                        coupling_efficiency=0.1,
                    )
                    bounds = inference_bounds or (1e4, 1e12)
                    inference = infer_kappa_from_psd(
                        freqs,
                        P,
                        model,
                        bounds=bounds,
                        n_calls=int(max(inference_calls, 20)),
                    )
                    kappa_inferred = float(inference.kappa_hat)
                    kappa_inferred_err = float(inference.kappa_err)
                    kappa_ci = tuple(float(v) for v in inference.credible_interval)
                    inference_dir = Path("results") / "kappa_inference"
                    inference_dir.mkdir(parents=True, exist_ok=True)
                    run_id = uuid4().hex[:8]
                    posterior_path = inference_dir / f"posterior_{run_id}.npz"
                    np.savez(
                        posterior_path,
                        kappa_grid=inference.posterior_grid,
                        posterior_density=inference.posterior_density,
                        trace=np.asarray(inference.trace, dtype=float),
                    )
                    meta_path = inference_dir / f"posterior_{run_id}.json"
                    with meta_path.open("w", encoding="utf-8") as fh:
                        json.dump(
                            {
                                "kappa_hat": inference.kappa_hat,
                                "kappa_err": inference.kappa_err,
                                "credible_interval": inference.credible_interval,
                                "diagnostics": inference.diagnostics,
                                "bounds": bounds,
                            },
                            fh,
                            indent=2,
                        )
                except Exception as exc:
                    print(f"[WARN] κ inference failed: {exc}")
        else:
            if perform_kappa_inference and infer_kappa_from_psd is None:
                print("[INFO] κ inference requested but scikit-optimize is not installed; skipping.")

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
        if kappa_err_list:
            dk = float(kappa_err_list[0])
            if dk > 0 and np.isfinite(dk):
                T_H_low = float(hbar * max(float(kappa[0]) - dk, 0.0) / (2.0 * pi * k))
                T_H_high = float(hbar * (float(kappa[0]) + dk) / (2.0 * pi * k))
        # Always define t5sigma_TH; if T_H <= 0 set to inf
        if T_H > 0:
            t_grid_TH = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), float(T_H))
            t5sigma_TH = float(t_grid_TH[0, 0])
        else:
            t5sigma_TH = float('inf')

        # Optional multi-band evaluation if bands provided (format: "f1:B1,f2:B2,...")
        if bands:
            try:
                t_best = None
                for pair in str(bands).split(','):
                    if not pair:
                        continue
                    if ':' not in pair:
                        continue
                    fc_str, bw_str = pair.split(':', 1)
                    fc = float(eval(fc_str)) if any(ch.isalpha() for ch in fc_str) else float(fc_str)
                    bw = float(eval(bw_str)) if any(ch.isalpha() for ch in bw_str) else float(bw_str)
                    Psel = np.asarray(spec.get("power_spectrum", P))
                    fsel = np.asarray(spec.get("frequencies", freqs))
                    inband = band_power_from_spectrum(fsel, Psel, fc, bw)
                    Tsig_band = equivalent_signal_temperature(inband, bw)
                    t_band = float(sweep_time_for_5sigma(np.array([T_sys]), np.array([bw]), Tsig_band)[0, 0]) if Tsig_band > 0 else float("inf")
                    if t_best is None or t_band < t_best:
                        t_best = t_band
                if t_best is not None:
                    t5sigma = t_best
            except Exception:
                pass

        # Optional hybrid branch
        if enable_hybrid:
            try:
                n_p0 = 1.0e24
                omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
                from analog_hawking.physics_engine.plasma_mirror import PlasmaMirrorParams, calculate_plasma_mirror_dynamics
                from analog_hawking.physics_engine.horizon_hybrid import HybridHorizonParams, find_hybrid_horizons
                p = PlasmaMirrorParams(n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=float(mirror_D), eta_a=float(mirror_eta), model=str(hybrid_model))
                t_m = np.linspace(0.0, 100e-15, 401)
                mirror = calculate_plasma_mirror_dynamics(state.grid, float(laser_intensity), p, t_m)
                hh = find_hybrid_horizons(state.grid, state.velocity, state.sound_speed, mirror, HybridHorizonParams())
                if hh.hybrid_kappa.size:
                    j = int(np.argmax(hh.hybrid_kappa))
                    hybrid_kappa_eff = float(hh.hybrid_kappa[j])
                    hybrid_coupling_weight = float(hh.coupling_weight[j])
                    k_fluid_ref = float(np.max(horizons.kappa)) if horizons.kappa.size else 0.0
                    # Apples-to-apples: same graybody and normalization as fluid spectrum
                    spec_h = calculate_enhanced_hawking_spectrum(
                        k_fluid_ref,
                        float(mirror.kappa_mirror),
                        float(hybrid_coupling_weight),
                        emitting_area_m2=1e-6,
                        solid_angle_sr=5e-2,
                        coupling_efficiency=0.1,
                        graybody_profile=gray_profile if 'gray_profile' in locals() else None,
                    )
                    if spec_h.get("success"):
                        freqs_h = np.asarray(spec_h["frequencies"])
                        P_h = np.asarray(spec_h["power_spectrum"])
                        # Apples-to-apples: integrate around the fluid's peak_frequency
                        if peak_frequency is not None and P_h.size:
                            inband_power_h = band_power_from_spectrum(freqs_h, P_h, float(peak_frequency), B_ref)
                            if inband_power_h == 0.0:
                                f_lo = float(peak_frequency) - 0.5 * B_ref
                                f_hi = float(peak_frequency) + 0.5 * B_ref
                                f_band = np.linspace(max(f_lo, float(freqs_h[0])), min(f_hi, float(freqs_h[-1])), 2001)
                                if f_band[-1] > f_band[0]:
                                    psd_band = np.interp(f_band, freqs_h, P_h)
                                    inband_power_h = float(np.trapz(psd_band, x=f_band))
                            T_sig_h = equivalent_signal_temperature(inband_power_h, B_ref)
                            if T_sig_h > 0:
                                t_grid_h = sweep_time_for_5sigma(np.array([T_sys]), np.array([B_ref]), T_sig_h)
                                hybrid_t5 = float(t_grid_h[0, 0])
                                hybrid_T_sig = float(T_sig_h)
                                hybrid_used = True
            except Exception:
                pass

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
            kappa_err=kappa_err_list,
            spectrum_peak_frequency=peak_frequency,
            inband_power_W=inband_power,
            T_sig_K=T_sig,
            t5sigma_s=t5sigma,
            t5sigma_s_low=t5sigma_low,
            t5sigma_s_high=t5sigma_high,
            T_H_K=T_H,
            T_H_K_low=T_H_low,
            T_H_K_high=T_H_high,
            t5sigma_TH_s=t5sigma_TH,
            graybody_window_cells=chosen_window_cells,
            hybrid_used=bool(hybrid_used),
            hybrid_kappa_eff=hybrid_kappa_eff,
            hybrid_coupling_weight=hybrid_coupling_weight,
            hybrid_T_sig_K=hybrid_T_sig,
            hybrid_t5sigma_s=hybrid_t5,
            kappa_inferred=kappa_inferred,
            kappa_inferred_err=kappa_inferred_err,
            kappa_credible_interval=kappa_ci,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="Use a horizon-forming demo configuration")
    p.add_argument("--intensity", type=float, default=None)
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--window-cells", type=int, default=None)
    p.add_argument("--kappa-method", type=str, choices=["acoustic", "legacy", "acoustic_exact"], default="acoustic")
    p.add_argument("--graybody", type=str, choices=["dimensionless", "wkb", "acoustic_wkb"], default="dimensionless")
    p.add_argument("--alpha-gray", type=float, default=1.0)
    p.add_argument("--bands", type=str, default=None, help="Comma-separated list of f_center:bandwidth pairs (e.g., '1e8:1e8,2e8:5e7')")
    p.add_argument("--Tsys", type=float, default=None, help="System temperature [K]")
    p.add_argument("--hybrid", action="store_true")
    p.add_argument("--hybrid-model", type=str, choices=["unruh", "anabhel"], default="anabhel")
    p.add_argument("--mirror-D", type=float, default=10e-6)
    p.add_argument("--mirror-eta", type=float, default=1.0)
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
    if args.kappa_method is not None:
        kwargs["kappa_method"] = args.kappa_method
    if args.graybody is not None:
        kwargs["graybody"] = args.graybody
    if args.alpha_gray is not None:
        kwargs["alpha_gray"] = args.alpha_gray
    if args.bands is not None:
        kwargs["bands"] = args.bands
    if args.Tsys is not None:
        kwargs["T_sys"] = args.Tsys

    if args.hybrid:
        kwargs["enable_hybrid"] = True
        kwargs["hybrid_model"] = args.hybrid_model
        kwargs["mirror_D"] = args.mirror_D
        kwargs["mirror_eta"] = args.mirror_eta
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
