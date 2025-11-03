#!/usr/bin/env python3
from __future__ import annotations

"""
κ-inference benchmark:
Simulate PSDs from ground-truth κ and graybody, add multiplicative noise,
invert for κ (and α) via grid-search MLE in log-PSD space, and summarize
coverage and calibration.

Example:
  python scripts/experiment_kappa_inference_benchmark.py \
    --n 64 --families linear tanh ramp --noise 0.08 --out results/experiments/kappa_inference
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.optimization.graybody_1d import compute_graybody
from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory
from experiment_universality_collapse import (
    make_exponential_profile,
    make_linear_profile,
    make_piecewise_ramp_profile,
    make_tanh_profile,
)

FAMILIES = {
    "linear": make_linear_profile,
    "tanh": make_tanh_profile,
    "exponential": make_exponential_profile,
    "ramp": make_piecewise_ramp_profile,
}


def _psd_from_profile(
    profile: Dict[str, np.ndarray], kappa: float, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    x = profile["x"]
    v = profile["v"]
    c = profile["c_s"]
    f = np.logspace(6.0, 11.0, 1000)
    gb = compute_graybody(x, v, c, f, method="acoustic_wkb", kappa=float(kappa), alpha=float(alpha))
    qft = QuantumFieldTheory(surface_gravity=float(kappa), emitting_area_m2=1.0, solid_angle_sr=1.0)
    psd = qft.hawking_spectrum(2.0 * np.pi * f, transmission=gb.transmission)
    return f, psd


def _nll_log_psd(y_obs: np.ndarray, y_model: np.ndarray, sigma_rel: float) -> float:
    # Gaussian residuals in log-space with relative noise sigma_rel
    eps = 1e-60
    r = np.log(y_obs + eps) - np.log(y_model + eps)
    var = float(sigma_rel) ** 2
    return 0.5 * float(np.sum(r * r / var))


def _fit_kappa_alpha(
    profile: Dict[str, np.ndarray],
    f: np.ndarray,
    y_obs: np.ndarray,
    kappa_ref: float,
    sigma_rel: float,
) -> Tuple[float, float, float]:
    # Coarse-to-fine grid search around kappa_ref and α ∈ [0.5, 1.5]
    k_lo, k_hi = 0.3 * kappa_ref, 3.0 * kappa_ref
    a_grid = np.linspace(0.5, 1.5, 11)
    best = (np.inf, float("nan"), float("nan"))
    for pass_no, n_k in enumerate([41, 61]):
        k_grid = np.geomspace(max(k_lo, 1e-6), max(k_hi, 1e-6), n_k)
        for a in a_grid:
            # generate model once per κ, α
            # reuse profile speeds for efficiency
            x = profile["x"]
            v = profile["v"]
            c = profile["c_s"]
            gb = compute_graybody(x, v, c, f, method="acoustic_wkb", kappa=None, alpha=float(a))
            # Override κ in QFT only (graybody uses κ from profile if None)
            for kappa in k_grid:
                qft = QuantumFieldTheory(
                    surface_gravity=float(kappa), emitting_area_m2=1.0, solid_angle_sr=1.0
                )
                y_model = qft.hawking_spectrum(2.0 * np.pi * f, transmission=gb.transmission)
                nll = _nll_log_psd(y_obs, y_model, sigma_rel)
                if nll < best[0]:
                    best = (nll, float(kappa), float(a))
        # Refine around best κ
        k_center = best[1]
        k_lo, k_hi = max(k_center / 3.0, 1e-6), k_center * 3.0
        a_grid = np.linspace(max(best[2] - 0.2, 0.3), min(best[2] + 0.2, 2.0), 11)

    # Approximate 1σ via profile-likelihood: ΔNLL = 0.5 for 1 parameter
    k_center = best[1]
    x = profile["x"]
    v = profile["v"]
    c = profile["c_s"]
    gb = compute_graybody(x, v, c, f, method="acoustic_wkb", kappa=None, alpha=float(best[2]))

    def nll_at(kval: float) -> float:
        qft = QuantumFieldTheory(
            surface_gravity=float(kval), emitting_area_m2=1.0, solid_angle_sr=1.0
        )
        y_model = qft.hawking_spectrum(2.0 * np.pi * f, transmission=gb.transmission)
        return _nll_log_psd(y_obs, y_model, sigma_rel)

    nll0 = nll_at(k_center)
    target = nll0 + 0.5
    # scan outward geometrically until we bracket the target on both sides
    k_lo = k_center
    while k_lo > k_center / 10.0 and nll_at(k_lo) < target:
        k_lo *= 0.8
    k_hi = k_center
    while k_hi < k_center * 10.0 and nll_at(k_hi) < target:
        k_hi *= 1.25

    # bisection to locate where NLL crosses target
    def bisect(a: float, b: float) -> float:
        fa = nll_at(a) - target
        fb = nll_at(b) - target
        for _ in range(25):
            c = 0.5 * (a + b)
            fc = nll_at(c) - target
            if np.sign(fa) == np.sign(fc):
                a, fa = c, fc
            else:
                b, fb = c, fc
        return 0.5 * (a + b)

    try:
        k_left = bisect(k_lo, k_center) if k_lo < k_center else k_center
        k_right = bisect(k_center, k_hi) if k_hi > k_center else k_center
        sigma_k = float(0.5 * (k_right - k_left))
    except Exception:
        sigma_k = 0.1 * k_center
    return best[1], best[2], sigma_k


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/experiments/kappa_inference")
    p.add_argument(
        "--families", nargs="*", default=list(FAMILIES.keys()), choices=list(FAMILIES.keys())
    )
    p.add_argument("--n", type=int, default=64)
    p.add_argument(
        "--noise", type=float, default=0.08, help="Relative multiplicative noise σ on PSD"
    )
    p.add_argument("--seed", type=int, default=123)
    args = p.parse_args()

    outdir = Path(args.out)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    per_family = max(1, args.n // max(len(args.families), 1))
    k_true_list: List[float] = []
    k_hat_list: List[float] = []
    k_sig_list: List[float] = []
    alpha_hat_list: List[float] = []
    fam_list: List[str] = []

    for fam in args.families:
        maker = FAMILIES[fam]
        for _ in range(per_family):
            seed = int(rng.integers(0, 10_000_000))
            prof = maker(seed)
            x = prof["x"]
            v = prof["v"]
            c = prof["c_s"]
            hr = find_horizons_with_uncertainty(x, v, c, kappa_method="acoustic_exact")
            if hr.kappa.size == 0:
                continue
            k_true = float(np.max(hr.kappa))
            f, y_true = _psd_from_profile(prof, k_true, alpha=1.0)
            # Inject multiplicative Gaussian noise
            noise = rng.normal(loc=0.0, scale=max(args.noise, 1e-6), size=y_true.shape)
            y_obs = y_true * (1.0 + noise)
            # prevent negatives (rare at large noise)
            y_obs = np.clip(y_obs, 1e-80, None)

            k_hat, a_hat, k_sig = _fit_kappa_alpha(prof, f, y_obs, k_true, sigma_rel=args.noise)
            k_true_list.append(k_true)
            k_hat_list.append(k_hat)
            k_sig_list.append(k_sig)
            alpha_hat_list.append(a_hat)
            fam_list.append(fam)

    k_true_arr = np.asarray(k_true_list)
    k_hat_arr = np.asarray(k_hat_list)
    k_sig_arr = np.asarray(k_sig_list)
    rel_err = np.abs(k_hat_arr - k_true_arr) / np.clip(k_true_arr, 1e-30, None)
    # coverage: |k_hat - k_true| < 1σ
    coverage = (
        float(np.mean(np.abs(k_hat_arr - k_true_arr) < k_sig_arr))
        if k_sig_arr.size
        else float("nan")
    )
    med_rel_err = float(np.median(rel_err)) if rel_err.size else float("nan")

    # Parity plot
    plt.figure(figsize=(5, 5))
    lim = (
        [float(np.min(k_true_arr)) * 0.8, float(np.max(k_true_arr)) * 1.2]
        if k_true_arr.size
        else [1e6, 1e7]
    )
    plt.loglog(k_true_arr, k_hat_arr, "o", alpha=0.6)
    plt.plot(lim, lim, "k--", lw=1.0)
    plt.xlabel("κ_true [s⁻¹]")
    plt.ylabel("κ_hat [s⁻¹]")
    plt.title(f"κ inference parity (coverage≈{coverage:.0%}, med rel err≈{med_rel_err:.1%})")
    plt.tight_layout()
    plt.savefig(figdir / "kappa_parity.png", dpi=180)
    plt.close()

    summary = {
        "n": int(k_true_arr.size),
        "families": args.families,
        "noise_sigma_rel": float(args.noise),
        "median_relative_error": med_rel_err,
        "coverage_within_1sigma": coverage,
    }
    with open(outdir / "kappa_inference_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print(
        f"Wrote: {outdir}/kappa_inference_summary.json ; coverage={coverage:.2%} ; med rel err={med_rel_err:.2%}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
