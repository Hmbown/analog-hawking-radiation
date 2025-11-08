#!/usr/bin/env python3
"""
Gradient Catastrophe Hunt with Spatially Resolved Coupling

This script extends the original gradient catastrophe sweep to incorporate
spatially resolved hybrid plasma-mirror coupling. The key enhancement is
preserving per-patch kappa values instead of using mean values.

The hypothesis: Spatial coupling variation will increase the effective
kappa_max bound compared to the production value of 5.94×10¹² Hz.

Author: bern2025-k2
Date: 1905-11-06 (in spirit)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.constants import c, epsilon_0, k, m_e
from tqdm import tqdm

# Ensure package imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.config.thresholds import Thresholds, load_thresholds
from analog_hawking.physics_engine.enhanced_coupling import (
    compute_patchwise_effective_kappa,
    create_spatial_coupling_profile,
)
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.horizon_hybrid import (
    HybridHorizonParams,
    find_hybrid_horizons,
)
from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend



class SpatialGradientCatastropheDetector:
    """Extended detector that includes spatial coupling effects"""

    def __init__(self, validation_tolerance: float = 0.1, thresholds: Thresholds | None = None):
        self.validation_tolerance = validation_tolerance
        self.thresholds = thresholds or Thresholds()

    def compute_gradient_metrics(
        self, x_grid: np.ndarray, velocity: np.ndarray, sound_speed: np.ndarray
    ) -> Dict[str, float]:
        """Compute gradient steepness metrics"""
        dx = np.diff(x_grid)
        dv_dx = np.gradient(velocity, x_grid)
        dc_dx = np.gradient(sound_speed, x_grid)

        # Maximum gradient steepness
        max_velocity_gradient = np.max(np.abs(dv_dx))
        max_sound_gradient = np.max(np.abs(dc_dx))

        # Characteristic length scales
        horizon_length = (
            np.mean(sound_speed) / np.mean(np.abs(dv_dx)) if np.any(dv_dx != 0) else np.inf
        )

        # Relativistic parameter
        max_mach = np.max(np.abs(velocity / sound_speed)) if np.any(sound_speed > 0) else 0

        return {
            "max_velocity_gradient": max_velocity_gradient,
            "max_sound_gradient": max_sound_gradient,
            "horizon_length_scale": horizon_length,
            "max_mach_number": max_mach,
            "gradient_steepness": max_velocity_gradient * horizon_length,
        }

    def detect_physics_breakdown(self, simulation_data: Dict) -> Dict[str, any]:
        """Simple physics breakdown detection based on thresholds"""
        velocity = simulation_data["velocity"]
        sound_speed = simulation_data["sound_speed"]
        intensity = simulation_data["laser_intensity"]
        
        max_velocity = np.max(np.abs(velocity))
        max_gradient = np.max(np.abs(np.gradient(velocity, simulation_data["grid"])))
        
        # Check against thresholds
        velocity_ok = max_velocity / c < self.thresholds.v_max_fraction_c
        gradient_ok = max_gradient < self.thresholds.dv_dx_max_s
        intensity_ok = intensity < self.thresholds.intensity_max_W_m2
        
        validity_score = 1.0 if (velocity_ok and gradient_ok and intensity_ok) else 0.0
        
        return {
            "validity_score": validity_score,
            "velocity_ok": velocity_ok,
            "gradient_ok": gradient_ok,
            "intensity_ok": intensity_ok,
            "max_velocity": max_velocity,
            "max_gradient": max_gradient,
            "max_velocity_fraction_c": max_velocity / c,
        }


def evaluate_configuration_with_spatial_coupling(
    a0: float,
    n_e: float,
    gradient_factor: float,
    detector: SpatialGradientCatastropheDetector,
    grid_size: int = 512,
) -> Dict:
    """
    Evaluate a single configuration with spatially resolved coupling

    Args:
        a0: Laser amplitude (dimensionless)
        n_e: Electron density (m^-3)
        gradient_factor: Gradient steepness multiplier
        detector: Catastrophe detector instance
        grid_size: Number of grid points

    Returns:
        Dictionary with configuration results including spatial coupling effects
    """
    # Setup grid
    x_max = 50e-6  # 50 microns
    x = np.linspace(0, x_max, grid_size)

    # Calculate laser intensity from a0
    # a0 = e E0 / (m_e c omega) => I = (1/2) epsilon0 c E0^2
    lambda_laser = 800e-9  # 800 nm
    omega = 2 * np.pi * c / lambda_laser
    E0 = a0 * m_e * c * omega / const.e
    I_0 = 0.5 * epsilon_0 * c * E0**2

    # Setup fluid backend (more reliable than MaxwellFluidModel)
    # Use conservative parameters to stay within valid physics regime
    backend = FluidBackend()
    backend.configure({
        "plasma_density": max(n_e, 1e18),  # Lower bound for validity
        "laser_wavelength": lambda_laser,
        "laser_intensity": min(I_0, 1e18),  # Upper bound for validity
        "grid": x,
        "temperature_settings": {"constant": 1e4},
        "use_fast_magnetosonic": False,
        "scale_with_intensity": True,
    })
    
    try:
        state = backend.step(0.0)
        velocity = state.velocity
        sound_speed = state.sound_speed

        # Apply gradient factor correctly: compress spatial scale, don't amplify velocity
        # This preserves velocity magnitude while increasing gradient steepness
        if gradient_factor > 1.0:
            # Create compressed grid (makes gradients steeper)
            x_compressed = np.linspace(x[0], x[-1] / gradient_factor, len(x))
            # Interpolate velocity onto compressed grid
            velocity = np.interp(x_compressed, x, velocity)
            sound_speed = np.interp(x_compressed, x, sound_speed)
            x = x_compressed

    except Exception as e:
        return {
            "a0": a0,
            "n_e": n_e,
            "gradient_factor": gradient_factor,
            "kappa_mean": 0.0,
            "kappa_max": 0.0,
            "kappa_spatial_std": 0.0,
            "coupling_weight_mean": 0.0,
            "coupling_weight_std": 0.0,
            "validity_score": 0.0,
            "error": str(e),
        }

    # Validate physics
    simulation_data = {
        "grid": x,
        "velocity": velocity,
        "sound_speed": sound_speed,
        "density": n_e,
        "laser_intensity": I_0,
    }
    breakdown_analysis = detector.detect_physics_breakdown(simulation_data)

    # Calculate horizon properties if physics is valid
    kappa_mean = 0.0
    kappa_max = 0.0
    kappa_spatial_std = 0.0
    coupling_weight_mean = 0.0
    coupling_weight_std = 0.0
    n_horizons = 0

    if breakdown_analysis["validity_score"] > 0.1:
        try:
            # Fluid horizons
            fluid_horizons = find_horizons_with_uncertainty(
                x, velocity, sound_speed, kappa_method="acoustic_exact"
            )

            if fluid_horizons.positions.size > 0:
                # Plasma mirror dynamics
                n_p0 = 1.0e24
                omega_p0 = float(np.sqrt(const.e**2 * n_p0 / (epsilon_0 * m_e)))
                p = PlasmaMirrorParams(
                    n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel"
                )
                t_m = np.linspace(0.0, 100e-15, 401)
                mirror = calculate_plasma_mirror_dynamics(x, float(I_0), p, t_m)

                # Hybrid horizons with spatial coupling
                hybrid_params = HybridHorizonParams(coupling_strength=0.3, coupling_length=5e-6)
                hybrid_horizons = find_hybrid_horizons(x, velocity, sound_speed, mirror, hybrid_params)

                if hybrid_horizons.fluid.positions.size > 0:
                    # Create spatial coupling profile
                    profile = create_spatial_coupling_profile(hybrid_horizons)

                    # Compute per-patch effective kappa
                    kappa_per_patch = compute_patchwise_effective_kappa(profile)

                    kappa_mean = float(np.mean(kappa_per_patch))
                    kappa_max = float(np.max(kappa_per_patch))
                    kappa_spatial_std = float(np.std(kappa_per_patch))
                    coupling_weight_mean = float(np.mean(profile.coupling_weights))
                    coupling_weight_std = float(np.std(profile.coupling_weights))
                    n_horizons = len(kappa_per_patch)

            # Compute gradient metrics
            gradient_metrics = detector.compute_gradient_metrics(x, velocity, sound_speed)

        except Exception as e:
            breakdown_analysis["validity_score"] = 0.0
            gradient_metrics = {"error": str(e)}
    else:
        gradient_metrics = {}

    return {
        "a0": a0,
        "n_e": n_e,
        "gradient_factor": gradient_factor,
        "kappa_mean": kappa_mean,
        "kappa_max": kappa_max,
        "kappa_spatial_std": kappa_spatial_std,
        "coupling_weight_mean": coupling_weight_mean,
        "coupling_weight_std": coupling_weight_std,
        "n_horizons": n_horizons,
        "validity_score": breakdown_analysis["validity_score"],
        "breakdown_modes": breakdown_analysis,
        "gradient_metrics": gradient_metrics,
        "intensity": I_0,
        "max_velocity": np.max(np.abs(velocity)),
        "gradient_steepness": np.max(np.abs(np.gradient(velocity, x))),
    }


def run_spatial_gradient_catastrophe_sweep(
    n_samples: int = 500,
    output_dir: str = "results/gradient_limits_spatial",
    thresholds_path: str | None = None,
    vmax_frac: float | None = None,
    dvdx_max: float | None = None,
    intensity_max: float | None = None,
) -> Dict:
    """
    Run gradient catastrophe sweep with spatially resolved coupling

    Args:
        n_samples: Number of parameter combinations to test
        output_dir: Directory to save results

    Returns:
        Dictionary with complete sweep results
    """

    print("GRADIENT CATASTROPHE HUNT: Spatially Resolved Coupling Edition")
    print(f"Testing {n_samples} configurations to find kappa_max...")
    print("=" * 70)

    # Define parameter ranges (logarithmic spacing for wide exploration)
    a0_range = np.logspace(0, 2, 20)  # 1 to 100 (dimensionless laser amplitude)
    n_e_range = np.logspace(18, 22, 15)  # 10^18 to 10^22 m^-3 (underdense to overcritical)
    gradient_range = np.logspace(0, 3, 10)  # 1 to 1000 (gradient steepness factor)

    # Generate parameter combinations (stratified sampling)
    results = []
    param_combinations = []

    for i in range(n_samples):
        a0 = float(np.random.choice(a0_range))
        n_e = float(10 ** np.random.uniform(np.log10(n_e_range[0]), np.log10(n_e_range[-1])))
        gradient_factor = float(np.random.choice(gradient_range))

        param_combinations.append((a0, n_e, gradient_factor))

    # Initialize detector
    thresholds = load_thresholds(thresholds_path) if thresholds_path else None
    detector = SpatialGradientCatastropheDetector(thresholds=thresholds)

    # Override thresholds if provided (create new Thresholds object)
    if thresholds is None:
        thresholds = Thresholds()
    if vmax_frac is not None:
        thresholds = Thresholds(
            v_max_fraction_c=vmax_frac,
            dv_dx_max_s=thresholds.dv_dx_max_s,
            intensity_max_W_m2=thresholds.intensity_max_W_m2,
        )
    if dvdx_max is not None:
        thresholds = Thresholds(
            v_max_fraction_c=thresholds.v_max_fraction_c,
            dv_dx_max_s=dvdx_max,
            intensity_max_W_m2=thresholds.intensity_max_W_m2,
        )
    if intensity_max is not None:
        thresholds = Thresholds(
            v_max_fraction_c=thresholds.v_max_fraction_c,
            dv_dx_max_s=thresholds.dv_dx_max_s,
            intensity_max_W_m2=intensity_max,
        )
    detector.thresholds = thresholds

    # Run sweep with progress bar
    for a0, n_e, gradient_factor in tqdm(param_combinations, desc="Sweeping parameter space"):
        result = evaluate_configuration_with_spatial_coupling(a0, n_e, gradient_factor, detector)
        results.append(result)

    # Save raw results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    with open(output_path / "sweep_results_spatial.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Analysis
    valid_results = [r for r in results if r["validity_score"] > 0.1]

    analysis = {
        "total_samples": n_samples,
        "valid_samples": len(valid_results),
        "valid_rate": len(valid_results) / n_samples if n_samples > 0 else 0,
    }

    if valid_results:
        # Find max kappa using different metrics
        kappa_mean_values = [r["kappa_mean"] for r in valid_results if r["kappa_mean"] > 0]
        kappa_max_values = [r["kappa_max"] for r in valid_results if r["kappa_max"] > 0]

        if kappa_mean_values:
            analysis["max_kappa_mean"] = max(kappa_mean_values)
            analysis["mean_kappa_mean"] = np.mean(kappa_mean_values)
            analysis["median_kappa_mean"] = np.median(kappa_mean_values)

        if kappa_max_values:
            analysis["max_kappa_max"] = max(kappa_max_values)  # This is the key new metric!
            analysis["mean_kappa_max"] = np.mean(kappa_max_values)
            analysis["median_kappa_max"] = np.median(kappa_max_values)

        # Find configuration with maximum kappa_max
        if kappa_max_values:
            max_idx = kappa_max_values.index(max(kappa_max_values))
            max_config = valid_results[max_idx]
            analysis["max_kappa_config"] = {
                "a0": max_config["a0"],
                "n_e": max_config["n_e"],
                "gradient_factor": max_config["gradient_factor"],
                "intensity": max_config["intensity"],
                "kappa_mean": max_config["kappa_mean"],
                "kappa_max": max_config["kappa_max"],
                "kappa_spatial_std": max_config["kappa_spatial_std"],
                "coupling_weight_mean": max_config["coupling_weight_mean"],
                "coupling_weight_std": max_config["coupling_weight_std"],
                "n_horizons": max_config["n_horizons"],
            }

        # Statistics on spatial variation
        spatial_stds = [r["kappa_spatial_std"] for r in valid_results if r["kappa_spatial_std"] > 0]
        if spatial_stds:
            analysis["mean_spatial_std"] = np.mean(spatial_stds)
            analysis["max_spatial_std"] = max(spatial_stds)

        coupling_stds = [r["coupling_weight_std"] for r in valid_results if r["coupling_weight_std"] > 0]
        if coupling_stds:
            analysis["mean_coupling_std"] = np.mean(coupling_stds)

        # Scaling relationships
        if len(valid_results) > 10:
            a0_values = [r["a0"] for r in valid_results]
            kappa_max_values = [r["kappa_max"] for r in valid_results]

            # Filter out zeros for log regression
            valid_pairs = [(a, k) for a, k in zip(a0_values, kappa_max_values) if k > 0 and a > 0]
            if len(valid_pairs) > 5:
                a0_clean, kappa_clean = zip(*valid_pairs)
                a0_log = np.log10(a0_clean)
                kappa_log = np.log10(kappa_clean)

                # Linear regression: log(kappa) = m * log(a0) + b
                try:
                    m, b = np.polyfit(a0_log, kappa_log, 1)
                    analysis["scaling_relationships"] = {
                        "kappa_vs_a0_exponent": m,
                        "kappa_vs_a0_intercept": b,
                    }
                except:
                    analysis["scaling_relationships"] = {
                        "kappa_vs_a0_exponent": np.nan,
                        "kappa_vs_a0_intercept": np.nan,
                    }

    # Save analysis
    with open(output_path / "analysis_spatial.json", "w") as f:
        json.dump(analysis, f, indent=2, default=str)

    return {"results": results, "analysis": analysis}


def main():
    """Main entry point for spatial gradient catastrophe hunt"""

    import argparse

    parser = argparse.ArgumentParser(description="Hunt for gradient catastrophe boundaries with spatial coupling")
    parser.add_argument(
        "--n-samples", type=int, default=200, help="Number of parameter combinations to test"
    )
    parser.add_argument(
        "--output", type=str, default="results/gradient_limits_spatial", help="Output directory for results"
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=None,
        help="Path to thresholds YAML (defaults used if omitted)",
    )
    parser.add_argument(
        "--vmax-frac", type=float, default=None, help="Override: max |v| as fraction of c"
    )
    parser.add_argument(
        "--dvdx-max", type=float, default=None, help="Override: max |dv/dx| in s^-1"
    )
    parser.add_argument(
        "--intensity-max", type=float, default=None, help="Override: max intensity in W/m^2"
    )

    args = parser.parse_args()

    # Run the sweep
    sweep_data = run_spatial_gradient_catastrophe_sweep(
        args.n_samples,
        args.output,
        thresholds_path=args.thresholds,
        vmax_frac=args.vmax_frac,
        dvdx_max=args.dvdx_max,
        intensity_max=args.intensity_max,
    )

    # Print key findings
    analysis = sweep_data["analysis"]
    print("\n" + "=" * 70)
    print("KEY FINDINGS (Spatially Resolved Coupling)")
    print("=" * 70)

    production_kappa_max = 5.94e12  # From production sweep
    print(f"Production κ_max (collapsed physics): {production_kappa_max:.2e} Hz")

    if "max_kappa_max" in analysis:
        spatial_kappa_max = analysis["max_kappa_max"]
        print(f"Spatial κ_max (resolved coupling): {spatial_kappa_max:.2e} Hz")
        improvement = spatial_kappa_max / production_kappa_max
        print(f"Improvement factor: {improvement:.2f}x")

        if improvement > 1.5:
            print("\n🎉 BREAKTHROUGH: Spatial coupling significantly increases κ_max!")
        elif improvement > 1.1:
            print("\n✅ IMPROVEMENT: Spatial coupling moderately increases κ_max")
        else:
            print("\n⚠️  Limited impact from spatial coupling")

    if "max_kappa_config" in analysis:
        config = analysis["max_kappa_config"]
        print(f"\nOptimal configuration:")
        print(f"  - a0 = {config['a0']:.2f}")
        print(f"  - n_e = {config['n_e']:.2e} m^-3")
        print(f"  - Gradient factor = {config['gradient_factor']:.1f}")
        print(f"  - Required intensity = {config['intensity']:.2e} W/m^2")
        print(f"  - Mean effective κ = {config['kappa_mean']:.2e} Hz")
        print(f"  - Max effective κ = {config['kappa_max']:.2e} Hz")
        print(f"  - Spatial std dev = {config['kappa_spatial_std']:.2e} Hz")
        print(f"  - Coupling weight std = {config['coupling_weight_std']:.3e}")

    if "scaling_relationships" in analysis:
        scaling = analysis["scaling_relationships"]
        if not np.isnan(scaling["kappa_vs_a0_exponent"]):
            print(f"\nScaling relationships:")
            print(f"  - κ_max ∝ a0^{scaling['kappa_vs_a0_exponent']:.2f}")

    print(f"\nValid configurations: {analysis.get('valid_samples', 0)}/{analysis.get('total_samples', 0)}")
    print(f"Valid rate: {analysis.get('valid_rate', 0):.1%}")

    if "mean_spatial_std" in analysis:
        print(f"Mean spatial variation: {analysis['mean_spatial_std']:.2e} Hz")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
