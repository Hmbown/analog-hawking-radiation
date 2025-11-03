from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, k, m_e
from tqdm import tqdm

from analog_hawking.config.thresholds import Thresholds, load_thresholds
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.laser_plasma_interaction import (
    MaxwellFluidModel,
)
from analog_hawking.physics_engine.plasma_models.validation_protocols import (
    PhysicsValidationFramework,
)


class GradientCatastropheDetector:
    def __init__(self, validation_tolerance: float = 0.1, thresholds: Thresholds | None = None):
        self.validation_tolerance = validation_tolerance
        self.validator = PhysicsValidationFramework(validation_tolerance)
        self.thresholds = thresholds or Thresholds()

    def compute_gradient_metrics(
        self, x_grid: np.ndarray, velocity: np.ndarray, sound_speed: np.ndarray
    ) -> Dict[str, float]:
        dv_dx = np.gradient(velocity, x_grid)
        dc_dx = np.gradient(sound_speed, x_grid)

        max_velocity_gradient = float(np.max(np.abs(dv_dx)))
        max_sound_gradient = float(np.max(np.abs(dc_dx)))
        horizon_length = (
            float(np.mean(sound_speed) / np.mean(np.abs(dv_dx))) if np.any(dv_dx != 0) else np.inf
        )
        max_mach = float(np.max(np.abs(velocity / sound_speed))) if np.any(sound_speed > 0) else 0.0

        return {
            "max_velocity_gradient": max_velocity_gradient,
            "max_sound_gradient": max_sound_gradient,
            "horizon_length_scale": horizon_length,
            "max_mach_number": max_mach,
            "gradient_steepness": max_velocity_gradient * horizon_length,
        }

    def detect_physics_breakdown(self, simulation_data: Dict) -> Dict[str, object]:
        breakdown_modes = {
            "relativistic_breakdown": False,
            "ionization_breakdown": False,
            "wave_breaking": False,
            "gradient_catastrophe": False,
            "intensity_breakdown": False,
            "numerical_instability": False,
            "validity_score": 1.0,
        }

        velocity = simulation_data.get("velocity", None)
        if velocity is not None:
            max_v = (
                float(np.max(np.abs(velocity))) if hasattr(velocity, "__iter__") else abs(velocity)
            )
            if max_v > self.thresholds.v_max_fraction_c * c:
                breakdown_modes["relativistic_breakdown"] = True
                breakdown_modes["validity_score"] *= 0.3

        density = simulation_data.get("density", None)
        if density is not None and hasattr(density, "__iter__"):
            if np.any(density <= 0) or np.any(density > 1e25):
                breakdown_modes["ionization_breakdown"] = True
                breakdown_modes["validity_score"] *= 0.1

        cs = simulation_data.get("sound_speed", None)
        if cs is not None and hasattr(cs, "__iter__") and np.any(cs <= 0):
            breakdown_modes["wave_breaking"] = True
            breakdown_modes["validity_score"] *= 0.2

        x = simulation_data.get("space_grid", None)
        v = simulation_data.get("velocity", None)
        if v is not None and x is not None and hasattr(v, "__iter__") and len(x) > 1:
            dv_dx = np.gradient(v, x)
            if np.any(np.abs(dv_dx) > self.thresholds.dv_dx_max_s):
                breakdown_modes["gradient_catastrophe"] = True
                breakdown_modes["validity_score"] *= 0.3
            if np.any(np.abs(dv_dx) > 1e20):
                breakdown_modes["validity_score"] *= 0.3

        intensity = simulation_data.get("intensity", None)
        try:
            if intensity is not None and float(intensity) > self.thresholds.intensity_max_W_m2:
                breakdown_modes["intensity_breakdown"] = True
                breakdown_modes["validity_score"] *= 0.3
        except Exception:
            pass

        validation_results = self.validator.validate_numerical_stability(simulation_data)
        if not validation_results["numerically_stable"]:
            breakdown_modes["numerical_instability"] = True
            breakdown_modes["validity_score"] *= 0.2

        return breakdown_modes


def run_single_configuration(
    a0: float, n_e: float, gradient_factor: float, thresholds: Thresholds | None = None
) -> Dict[str, object]:
    lambda_l = 800e-9
    omega_l = 2 * np.pi * c / lambda_l
    I_0 = 0.5 * epsilon_0 * c * (a0**2) * (m_e**2 * omega_l**2 * c**2) / (e**2)

    try:
        _ = MaxwellFluidModel(plasma_density=n_e, laser_wavelength=lambda_l, laser_intensity=I_0)
    except Exception as exc:  # pragma: no cover (defensive)
        return {
            "a0": a0,
            "n_e": n_e,
            "gradient_factor": gradient_factor,
            "kappa": 0.0,
            "validity_score": 0.0,
            "error": str(exc),
            "breakdown_modes": {"numerical_instability": True},
        }

    L = 100e-6
    x = np.linspace(-L / 2, L / 2, 500)
    x_transition = 0.0
    sigma = L / (20 * gradient_factor)

    cs_thermal = np.sqrt(k * 10000 / m_e)
    cs_base = cs_thermal * (1 + 0.2 * np.exp(-((x / sigma) ** 2)))
    sound_speed = cs_base

    v_scale = cs_thermal * 1.5 * a0 * gradient_factor
    velocity = v_scale * np.tanh((x - x_transition) / sigma)
    if float(np.max(np.abs(velocity))) < float(np.max(sound_speed)) * 0.5:
        velocity *= 2.0

    simulation_data = {
        "space_grid": x,
        "velocity": velocity,
        "sound_speed": sound_speed,
        "density": np.full_like(x, n_e),
        "electric_field": np.zeros_like(x),
        "a0": a0,
        "plasma_density": n_e,
        "gradient_steepness": gradient_factor,
        "intensity": I_0,
    }

    detector = GradientCatastropheDetector(thresholds=thresholds)
    breakdown_analysis = detector.detect_physics_breakdown(simulation_data)

    if breakdown_analysis["validity_score"] > 0.1:
        try:
            horizons = find_horizons_with_uncertainty(
                x, velocity, sound_speed, kappa_method="acoustic_exact"
            )
            kappa = float(np.mean(horizons.kappa)) if len(horizons.kappa) > 0 else 0.0
            gradient_metrics = detector.compute_gradient_metrics(x, velocity, sound_speed)
        except Exception as exc:
            kappa = 0.0
            gradient_metrics = {"error": str(exc)}
            breakdown_analysis["validity_score"] = 0.0
    else:
        kappa = 0.0
        gradient_metrics = {}

    return {
        "a0": float(a0),
        "n_e": float(n_e),
        "gradient_factor": float(gradient_factor),
        "kappa": float(kappa),
        "validity_score": float(breakdown_analysis["validity_score"]),
        "breakdown_modes": breakdown_analysis,
        "gradient_metrics": gradient_metrics,
        "intensity": float(I_0),
        "max_velocity": float(np.max(np.abs(velocity))),
        "gradient_steepness": float(np.max(np.abs(np.gradient(velocity, x)))),
    }


def analyze_catastrophe_boundaries(results: List[Dict[str, object]]) -> Dict[str, object]:
    valid_results = [r for r in results if float(r.get("validity_score", 0)) > 0.5]
    invalid_results = [r for r in results if float(r.get("validity_score", 0)) <= 0.5]

    kappas = [float(r.get("kappa", 0.0)) for r in valid_results]
    max_kappa = max(kappas) if kappas else 0.0
    max_kappa_config = (
        max(valid_results, key=lambda r: float(r.get("kappa", 0.0))) if valid_results else None
    )

    breakdown_stats: Dict[str, Dict[str, float]] = {}
    modes = [
        "relativistic_breakdown",
        "ionization_breakdown",
        "wave_breaking",
        "gradient_catastrophe",
        "intensity_breakdown",
        "numerical_instability",
    ]
    for mode in modes:
        count = sum(1 for r in results if r.get("breakdown_modes", {}).get(mode, False))
        breakdown_stats[mode] = {"count": count, "rate": count / len(results) if results else 0.0}
    breakdown_stats["total_breakdown_rate"] = (
        len(invalid_results) / len(results) if results else 0.0
    )

    valid_a0 = [float(r["a0"]) for r in valid_results]
    valid_ne = [float(r["n_e"]) for r in valid_results]
    valid_kappa = [float(r["kappa"]) for r in valid_results]

    def _logfit(xv: List[float], yv: List[float]) -> Tuple[float, Tuple[float, float]]:
        x = np.asarray([np.log10(x) for x, y in zip(xv, yv) if y > 0])
        y = np.asarray([np.log10(y) for y in yv if y > 0])
        if len(x) < 6 or len(y) != len(x):
            return float("nan"), (float("nan"), float("nan"))
        coeffs = np.polyfit(x, y, 1)
        slope = float(coeffs[0])
        yhat = coeffs[0] * x + coeffs[1]
        resid = y - yhat
        dof = max(len(x) - 2, 1)
        s_yx = float(np.sqrt(np.sum(resid**2) / dof))
        s_xx = float(np.sum((x - x.mean()) ** 2)) if len(x) > 1 else 1.0
        se_slope = s_yx / np.sqrt(max(s_xx, 1e-30))
        ci = (slope - 1.96 * se_slope, slope + 1.96 * se_slope)
        return slope, ci

    slope_a0, ci_a0 = _logfit(valid_a0, valid_kappa)
    slope_ne, ci_ne = _logfit(valid_ne, valid_kappa)

    return {
        "max_kappa": max_kappa,
        "max_kappa_config": max_kappa_config,
        "valid_configurations": len(valid_results),
        "invalid_configurations": len(invalid_results),
        "breakdown_statistics": breakdown_stats,
        "scaling_relationships": {
            "kappa_vs_a0_exponent": slope_a0,
            "kappa_vs_ne_exponent": slope_ne,
            "kappa_vs_a0_exponent_ci95": list(ci_a0),
            "kappa_vs_ne_exponent_ci95": list(ci_ne),
        },
        "parameter_boundaries": {
            "max_valid_a0": max(valid_a0) if valid_a0 else float("nan"),
            "max_valid_ne": max(valid_ne) if valid_ne else float("nan"),
            "max_valid_gradient": float(
                max([r.get("gradient_factor", 0) for r in valid_results])
                if valid_results
                else float("nan")
            ),
        },
    }


def run_sweep(
    n_samples: int = 500,
    output_dir: str = "results/gradient_limits",
    thresholds_path: str | None = None,
    vmax_frac: float | None = None,
    dvdx_max: float | None = None,
    intensity_max: float | None = None,
    a0_min: float | None = None,
    a0_max: float | None = None,
    ne_min: float | None = None,
    ne_max: float | None = None,
    grad_min: float | None = None,
    grad_max: float | None = None,
    n_per_axis: int | None = None,
    seed: int = 42,
    stratified: bool = False,
) -> Dict[str, object]:
    thr = load_thresholds(thresholds_path)
    if vmax_frac is not None:
        thr = Thresholds(
            v_max_fraction_c=vmax_frac,
            dv_dx_max_s=thr.dv_dx_max_s,
            intensity_max_W_m2=thr.intensity_max_W_m2,
        )
    if dvdx_max is not None:
        thr = Thresholds(
            v_max_fraction_c=thr.v_max_fraction_c,
            dv_dx_max_s=dvdx_max,
            intensity_max_W_m2=thr.intensity_max_W_m2,
        )
    if intensity_max is not None:
        thr = Thresholds(
            v_max_fraction_c=thr.v_max_fraction_c,
            dv_dx_max_s=thr.dv_dx_max_s,
            intensity_max_W_m2=intensity_max,
        )

    # Default ranges (log-spaced)
    default_a0 = np.logspace(0, 2, 20)
    default_ne = np.logspace(18, 22, 15)
    default_grad = np.logspace(0, 3, 10)

    if all(v is None for v in (a0_min, a0_max, ne_min, ne_max, grad_min, grad_max)):
        a0_range = default_a0
        ne_range = default_ne
        grad_range = default_grad
    else:
        # Build custom log ranges
        def _logspace(lo: float, hi: float, n: int) -> np.ndarray:
            return np.logspace(np.log10(lo), np.log10(hi), n)

        npa = n_per_axis or 10
        a0_range = _logspace(a0_min or 1.0, a0_max or 100.0, npa)
        ne_range = _logspace(ne_min or 1e18, ne_max or 1e22, npa)
        grad_range = _logspace(grad_min or 1.0, grad_max or 1000.0, npa)

    # Parameter selection
    param_combinations: List[Tuple[float, float, float]] = []
    if stratified:
        for a0 in a0_range:
            for ne in ne_range:
                for gf in grad_range:
                    param_combinations.append((float(a0), float(ne), float(gf)))
    else:
        total_combinations = len(a0_range) * len(ne_range) * len(grad_range)
        if total_combinations <= n_samples:
            for a0 in a0_range:
                for ne in ne_range:
                    for gf in grad_range:
                        param_combinations.append((float(a0), float(ne), float(gf)))
        else:
            rng = np.random.default_rng(seed)
            for _ in range(n_samples):
                param_combinations.append(
                    (
                        float(rng.choice(a0_range)),
                        float(rng.choice(ne_range)),
                        float(rng.choice(grad_range)),
                    )
                )

    results: List[Dict[str, object]] = []
    with tqdm(total=len(param_combinations), desc="Exploring parameter space") as pbar:
        for a0, ne, gf in param_combinations:
            results.append(run_single_configuration(a0, ne, gf, thresholds=thr))
            valid_kappas = [r["kappa"] for r in results if float(r.get("validity_score", 0)) > 0.5]
            max_kappa = max(valid_kappas) if valid_kappas else 0.0
            pbar.set_postfix(
                {"max_kappa": f"{max_kappa:.2e}", "valid": f"{len(valid_kappas)}/{len(results)}"}
            )
            pbar.update(1)

    analysis = analyze_catastrophe_boundaries(results)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sweep_data = {
        "results": results,
        "analysis": analysis,
        "n_samples": len(results),
        "parameter_ranges": {
            "a0_range": list(map(float, a0_range.tolist())),
            "n_e_range": list(map(float, ne_range.tolist())),
            "gradient_range": list(map(float, grad_range.tolist())),
        },
    }
    with (output_path / "gradient_catastrophe_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(sweep_data, f, indent=2, default=str)

    return sweep_data


def generate_catastrophe_plots(sweep_data: Dict[str, object], output_dir: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    results: List[Dict[str, object]] = list(sweep_data.get("results", []))

    a0 = np.array([float(r["a0"]) for r in results])
    ne = np.array([float(r["n_e"]) for r in results])
    kappa = np.array([float(r.get("kappa", 0.0)) for r in results])
    valid = np.array([float(r.get("validity_score", 0)) > 0.5 for r in results])

    # Scatter of valid points colored by kappa
    plt.figure(figsize=(7, 5))
    sc = plt.scatter(a0[valid], ne[valid], c=kappa[valid], cmap="viridis", s=12)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("a0")
    plt.ylabel("n_e [m^-3]")
    plt.colorbar(sc, label="kappa [s^-1]")
    plt.tight_layout()
    plt.savefig(output_path / "kappa_scatter.png", dpi=150)
    plt.close()

    # Kappa histogram
    plt.figure(figsize=(6, 4))
    plt.hist(kappa[valid], bins=40)
    plt.xlabel("kappa [s^-1]")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output_path / "kappa_hist.png", dpi=150)
    plt.close()
