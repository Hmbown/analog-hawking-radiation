#!/usr/bin/env python3
from __future__ import annotations

"""
Experiment: Universality of Hawking spectra after κ-normalization.

Generates analytic flow families (and optional PIC slices), computes horizons
and κ via acoustic_exact, constructs graybody via acoustic_wkb, forms PSD via
QFT, and evaluates collapse when plotted vs ω/κ. Also computes radio band
T_sig and t_5σ and writes JSON summaries and figures.

Usage examples:
  python scripts/experiment_universality_collapse.py \
    --out results/experiments/universality --n 24 --alpha 0.8 --seed 7

  python scripts/experiment_universality_collapse.py --include-controls
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import numpy as np
import matplotlib.pyplot as plt

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody
from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory
from analog_hawking.detection.psd_collapse import (
    omega_over_kappa_axis,
    resample_on_x,
    collapse_stats,
    band_temperature_and_t5sig,
)


# ------------------------------
# Analytic flow families
# ------------------------------

def _grid(n: int = 2048, L: float = 1.0) -> np.ndarray:
    return np.linspace(-L, L, int(n))


def make_linear_profile(seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _grid()
    c0 = float(rng.uniform(1.0e5, 3.0e5))  # m/s
    a = float(rng.uniform(5.0e5, 1.5e6))   # s^-1 scale (via dv/dx)
    x0 = float(rng.uniform(-0.2, 0.2))
    v = a * (x - x0)
    c_s = np.full_like(x, c0)
    return {"x": x, "v": v, "c_s": c_s}


def make_tanh_profile(seed: int = 1) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _grid()
    c0 = float(rng.uniform(1.2e5, 3.5e5))
    dv = float(rng.uniform(2.0e5, 8.0e5))
    L = float(rng.uniform(0.05, 0.3))
    x0 = float(rng.uniform(-0.1, 0.1))
    v = dv * np.tanh((x - x0) / L)
    c_s = np.full_like(x, c0)
    return {"x": x, "v": v, "c_s": c_s}


def make_exponential_profile(seed: int = 2) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _grid()
    c0 = float(rng.uniform(1.0e5, 3.0e5))
    A = float(rng.uniform(1.0e5, 6.0e5))
    L = float(rng.uniform(0.05, 0.2))
    x0 = float(rng.uniform(-0.3, 0.3))
    sgn = rng.choice([-1.0, 1.0])
    v = sgn * A * (np.exp((x - x0) / L) - 1.0)
    # clip extremes to avoid numerical overflow
    v = np.clip(v, -2.0e6, 2.0e6)
    c_s = np.full_like(x, c0)
    return {"x": x, "v": v, "c_s": c_s}


def make_piecewise_ramp_profile(seed: int = 3) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _grid()
    c0 = float(rng.uniform(1.0e5, 3.0e5))
    x_break = float(rng.uniform(-0.2, 0.2))
    v_left = float(rng.uniform(-6.0e5, -1.0e5))
    v_right = float(rng.uniform(1.0e5, 6.0e5))
    # Linear ramps to create a crossing
    v = np.where(x < x_break,
                 np.interp(x, [x[0], x_break], [v_left, 0.0]),
                 np.interp(x, [x_break, x[-1]], [0.0, v_right]))
    c_s = np.full_like(x, c0)
    return {"x": x, "v": v, "c_s": c_s}


def make_no_horizon_profile(seed: int = 4) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = _grid()
    c0 = float(rng.uniform(2.0e5, 3.0e5))
    v_amp = float(rng.uniform(2.0e4, 5.0e4))  # keep |v| well below c_s
    k = float(rng.uniform(4.0, 8.0))
    v = v_amp * np.sin(k * x)
    c_s = np.full_like(x, c0)
    return {"x": x, "v": v, "c_s": c_s}


def make_white_hole_profile(seed: int = 5) -> Dict[str, np.ndarray]:
    # Flip gradient sign relative to typical BH-like configuration
    base = make_tanh_profile(seed)
    base["v"] = -base["v"]
    return base


FAMILIES = {
    "linear": make_linear_profile,
    "tanh": make_tanh_profile,
    "exponential": make_exponential_profile,
    "ramp": make_piecewise_ramp_profile,
}


@dataclass
class SpectrumRecord:
    family: str
    seed: int
    kappa: float
    peak_frequency: float
    T_sig: float
    t5sigma: float
    success: bool


def _spectrum_for_profile(profile: Dict[str, np.ndarray], *, alpha: float, B: float, T_sys: float) -> Tuple[np.ndarray, np.ndarray, SpectrumRecord]:
    x = np.asarray(profile["x"]) ; v = np.asarray(profile["v"]) ; c = np.asarray(profile["c_s"])  # noqa: E702
    hr = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic_exact")
    if hr.kappa.size == 0 or float(np.max(hr.kappa)) <= 0.0:
        # No horizon
        return np.array([]), np.array([]), SpectrumRecord("none", 0, 0.0, 0.0, 0.0, np.inf, False)
    kappa = float(np.max(hr.kappa))

    # Frequency grid chosen based on temperature scale
    qft = QuantumFieldTheory(surface_gravity=kappa, emitting_area_m2=1.0, solid_angle_sr=1.0, coupling_efficiency=1.0)
    # Use a band that spans radio→microwave to capture shape near the peak for low T
    f = np.logspace(6.0, 11.0, 1200)
    gb = compute_graybody(x, v, c, f, method="acoustic_wkb", kappa=kappa, alpha=float(alpha))
    psd = qft.hawking_spectrum(2.0 * np.pi * f, transmission=gb.transmission)

    T_sig, t5 = band_temperature_and_t5sig(f, psd, B=B, T_sys=T_sys)
    rec = SpectrumRecord("unknown", 0, kappa, float(f[int(np.argmax(psd))]), float(T_sig), float(t5), True)
    return f, psd, rec


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/experiments/universality")
    p.add_argument("--families", nargs="*", default=list(FAMILIES.keys()), choices=list(FAMILIES.keys()))
    p.add_argument("--n", type=int, default=24, help="Total spectra across families (approx)")
    p.add_argument("--alpha", type=float, default=0.8, help="Graybody acoustic_WKB alpha scale")
    p.add_argument("--B", type=float, default=1e8, help="Bandwidth in Hz (default 100 MHz)")
    p.add_argument("--Tsys", type=float, default=30.0, help="System temperature in K")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-controls", action="store_true")
    p.add_argument("--pic-profiles", nargs="*", default=[], help="Paths or globs to PIC/OpenPMD .npz profiles (keys: x, v, c_s)")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    outdir = Path(args.out)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    # Target dimensionless grid for collapse check
    x_common = np.linspace(0.2, 5.0, 240)
    all_curves: List[np.ndarray] = []
    all_records: List[SpectrumRecord] = []
    all_axes: List[np.ndarray] = []

    # Distribute seeds across families (analytic)
    per_family = max(1, args.n // max(len(args.families), 1))
    for fam in args.families:
        maker = FAMILIES[fam]
        for j in range(per_family):
            seed = int(rng.integers(0, 10_000_000))
            profile = maker(seed)
            f, psd, rec = _spectrum_for_profile(profile, alpha=args.alpha, B=args.B, T_sys=args.Tsys)
            if psd.size == 0:
                continue
            rec.family = fam
            rec.seed = seed
            # Map to ω/κ axis and resample on common grid
            x_dimless = omega_over_kappa_axis(f, rec.kappa)
            y_interp = resample_on_x(x_dimless, psd, x_common)
            all_curves.append(y_interp)
            all_records.append(rec)
            all_axes.append(x_dimless)

    # Include PIC/OpenPMD .npz profiles if provided
    def _expand_globs(paths: Iterable[str]) -> List[Path]:
        out: List[Path] = []
        for pth in paths:
            pth = str(pth)
            if any(ch in pth for ch in ["*", "?", "["]):
                for m in Path().glob(pth):
                    if m.suffix.lower() == ".npz":
                        out.append(m)
            else:
                pp = Path(pth)
                if pp.exists() and pp.suffix.lower() == ".npz":
                    out.append(pp)
        return out

    pic_files = _expand_globs(args.pic_profiles)
    for npz_path in pic_files:
        try:
            data = dict(np.load(npz_path))
            if not all(k in data for k in ("x", "v", "c_s")):
                continue
            profile = {"x": np.asarray(data["x"]), "v": np.asarray(data["v"]), "c_s": np.asarray(data["c_s"]) }
            f, psd, rec = _spectrum_for_profile(profile, alpha=args.alpha, B=args.B, T_sys=args.Tsys)
            if psd.size == 0:
                continue
            rec.family = f"pic:{npz_path.name}"
            rec.seed = 0
            x_dimless = omega_over_kappa_axis(f, rec.kappa)
            y_interp = resample_on_x(x_dimless, psd, x_common)
            all_curves.append(y_interp)
            all_records.append(rec)
            all_axes.append(x_dimless)
        except Exception:
            continue

    # Collapse stats on main set
    stats = collapse_stats(all_curves)
    stats_grid = x_common
    stats_mean = stats.mean
    stats_std = stats.std
    rms_rel = float(stats.rms_relative)

    # Save summary JSON
    summary = {
        "n_curves": len(all_curves),
        "families": args.families,
        "alpha": float(args.alpha),
        "B": float(args.B),
        "T_sys": float(args.Tsys),
        "seed": int(args.seed),
        "collapse_rms_relative": rms_rel,
        "grid_min": float(stats_grid[0]) if stats_grid.size else None,
        "grid_max": float(stats_grid[-1]) if stats_grid.size else None,
        "records": [asdict(r) for r in all_records],
    }
    with open(outdir / "universality_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # Plot collapse overlay
    plt.figure(figsize=(8, 5))
    for y in all_curves:
        plt.plot(stats_grid, y, color="tab:blue", alpha=0.35, linewidth=1.0)
    plt.plot(stats_grid, stats_mean, color="k", linewidth=2.0, label="mean")
    plt.fill_between(stats_grid, stats_mean - stats_std, stats_mean + stats_std,
                     color="orange", alpha=0.25, label="±1σ")
    plt.xscale("linear")
    plt.yscale("log")
    plt.xlabel(r"$\omega/\kappa$")
    plt.ylabel("PSD [arb W/Hz]")
    plt.title(f"κ-normalized collapse across families (RMS rel={rms_rel:.2%})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "collapse_overlay.png", dpi=180)
    plt.close()

    # Controls (optional): no-horizon and white-hole
    if args.include_controls:
        controls: List[Tuple[str, Dict[str, np.ndarray]]] = [
            ("no_horizon", make_no_horizon_profile(args.seed + 101)),
            ("white_hole", make_white_hole_profile(args.seed + 202)),
        ]
        ctrl_res: List[Dict[str, float]] = []
        for label, prof in controls:
            f, psd, rec = _spectrum_for_profile(prof, alpha=args.alpha, B=args.B, T_sys=args.Tsys)
            ctrl_res.append({
                "label": label,
                "has_horizon": bool(rec.kappa > 0),
                "T_sig": float(rec.T_sig),
                "t5sigma": float(rec.t5sigma),
            })
            # Quick side-by-side plot
            plt.figure(figsize=(6, 4))
            if f.size:
                plt.loglog(f, psd + 1e-60, label=label)
            plt.xlabel("f [Hz]")
            plt.ylabel("PSD [W/Hz]")
            plt.title(f"Control: {label} (T_sig={rec.T_sig:.2e} K)")
            plt.tight_layout()
            plt.savefig(figdir / f"control_{label}.png", dpi=160)
            plt.close()

        with open(outdir / "controls_summary.json", "w") as fh:
            json.dump(ctrl_res, fh, indent=2)

    print(f"Wrote: {outdir}/universality_summary.json ; RMS_rel={rms_rel:.3f} over [{stats_grid[0]:.2f},{stats_grid[-1]:.2f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
