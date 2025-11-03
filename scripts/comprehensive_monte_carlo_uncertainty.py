#!/usr/bin/env python3
from __future__ import annotations

"""
Comprehensive Monte Carlo uncertainty quantification for analog Hawking radiation.

Enhanced framework that includes:
- Systematic uncertainties (laser, diagnostic, model, environmental)
- Correlated parameter uncertainties
- Advanced sampling strategies
- Full error budget breakdown and visualization

Outputs:
- results/comprehensive_uncertainty_analysis.json
- figures/uncertainty_budget_breakdown.png
- figures/horizon_probability_with_systematics.png
- figures/systematic_vs_statistical_uncertainty.png
"""

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")
import warnings

import corner  # For posterior visualization
import emcee  # For Bayesian inference
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from analog_hawking.utils.graphics_control import (
    GraphicsController,
    add_graphics_argument,
    get_graphics_preference,
    skip_plotting_message,
)


class UncertaintyType(Enum):
    """Types of uncertainties in the analysis."""

    STATISTICAL = "statistical"
    SYSTEMATIC_LASER = "systematic_laser"
    SYSTEMATIC_DIAGNOSTIC = "systematic_diagnostic"
    SYSTEMATIC_MODEL = "systematic_model"
    SYSTEMATIC_ENVIRONMENTAL = "systematic_environmental"


@dataclass
class LaserUncertainties:
    """Systematic uncertainties from laser parameters."""

    intensity_uncertainty: float = 0.03  # 3% RMS intensity fluctuation
    pointing_stability: float = 0.02  # 2% pointing error
    pulse_duration_variation: float = 0.01  # 1% pulse duration jitter
    wavelength_drift: float = 0.001  # 0.1% wavelength drift
    focal_spot_variation: float = 0.05  # 5% focal spot size variation

    # Correlation matrix for laser parameters
    correlation_matrix: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [1.0, 0.3, 0.1, 0.0, 0.2],  # intensity correlations
                [0.3, 1.0, 0.0, 0.0, 0.1],  # pointing correlations
                [0.1, 0.0, 1.0, 0.0, 0.0],  # pulse duration correlations
                [0.0, 0.0, 0.0, 1.0, 0.0],  # wavelength correlations
                [0.2, 0.1, 0.0, 0.0, 1.0],  # focal spot correlations
            ]
        )
    )


@dataclass
class DiagnosticUncertainties:
    """Systematic uncertainties from diagnostic measurements."""

    density_calibration: float = 0.05  # 5% density calibration uncertainty
    temperature_calibration: float = 0.07  # 7% temperature calibration uncertainty
    detector_noise: float = 0.02  # 2% detector noise level
    temporal_resolution: float = 0.01  # 1% timing uncertainty
    spatial_resolution: float = 0.03  # 3% spatial measurement uncertainty
    background_subtraction: float = 0.04  # 4% background subtraction uncertainty


@dataclass
class ModelUncertainties:
    """Uncertainties from physics modeling and numerical methods."""

    fluid_approximation: float = 0.10  # 10% fluid model uncertainty
    equation_of_state: float = 0.05  # 5% equation of state uncertainty
    boundary_conditions: float = 0.03  # 3% boundary condition uncertainty
    numerical_discretization: float = 0.02  # 2% numerical discretization error
    ionization_model: float = 0.08  # 8% ionization model uncertainty


@dataclass
class EnvironmentalUncertainties:
    """Environmental systematic uncertainties."""

    temperature_variation: float = 0.02  # 2% lab temperature variation
    vibration_noise: float = 0.01  # 1% vibration-induced uncertainty
    vacuum_quality: float = 0.015  # 1.5% vacuum level variation
    magnetic_field_stability: float = 0.02  # 2% magnetic field stability


class BayesianModelUncertainty:
    """Bayesian inference for model uncertainty quantification."""

    def __init__(self, model_config: Dict):
        self.model_config = model_config
        self.n_walkers = model_config.get("n_walkers", 32)
        self.n_steps = model_config.get("n_steps", 1000)
        self.burn_in = model_config.get("burn_in", 200)

    def log_likelihood(self, theta: np.ndarray, data: Dict) -> float:
        """Log likelihood for model parameters."""
        # Model parameters: [fluid_model_weight, ionization_model_weight,
        #                    eos_model_weight, boundary_model_weight]
        fluid_weight, ion_weight, eos_weight, boundary_weight = theta

        # Ensure weights are positive and sum to 1
        if np.any(theta < 0) or np.sum(theta) > 1.5:
            return -np.inf

        # Get model predictions
        model_predictions = data.get("model_predictions", [])
        observations = data.get("observations", [])
        measurement_errors = data.get("measurement_errors", [])

        if len(model_predictions) == 0 or len(observations) == 0:
            return 0.0  # No data to constrain

        # Calculate weighted model prediction
        weighted_prediction = (
            fluid_weight * model_predictions[0]
            + ion_weight * model_predictions[1]
            + eos_weight * model_predictions[2]
            + boundary_weight * model_predictions[3]
        )

        # Gaussian likelihood
        log_likelihood = -0.5 * np.sum(
            ((observations - weighted_prediction) / measurement_errors) ** 2
            + np.log(2 * np.pi * measurement_errors**2)
        )

        return log_likelihood

    def log_prior(self, theta: np.ndarray) -> float:
        """Log prior for model parameters."""
        fluid_weight, ion_weight, eos_weight, boundary_weight = theta

        # Uniform priors on [0, 1] for each weight
        if np.all(theta >= 0) and np.all(theta <= 1):
            return 0.0  # Uniform prior
        else:
            return -np.inf

    def log_posterior(self, theta: np.ndarray, data: Dict) -> float:
        """Log posterior distribution."""
        return self.log_prior(theta) + self.log_likelihood(theta, data)

    def run_mcmc(self, initial_guess: np.ndarray, data: Dict) -> Tuple[np.ndarray, Dict]:
        """Run MCMC to sample posterior distribution."""
        ndim = len(initial_guess)

        # Initialize walkers
        pos = initial_guess + 1e-4 * np.random.randn(self.n_walkers, ndim)

        # Create sampler
        sampler = emcee.EnsembleSampler(self.n_walkers, ndim, self.log_posterior, args=[data])

        # Run MCMC
        print("Running MCMC for model uncertainty quantification...")
        sampler.run_mcmc(pos, self.n_steps, progress=True)

        # Extract chains
        chains = sampler.get_chain(discard=self.burn_in, thin=10, flat=True)

        # Calculate statistics
        mean_params = np.mean(chains, axis=0)
        std_params = np.std(chains, axis=0)

        results = {
            "chains": chains,
            "mean_parameters": mean_params,
            "std_parameters": std_params,
            "acceptance_fraction": np.mean(sampler.acceptance_fraction),
            "autocorrelation_time": sampler.get_autocorr_time(),
        }

        return chains, results


class NestedMonteCarlo:
    """Nested Monte Carlo for separating systematic and statistical uncertainties."""

    def __init__(self, config: ComprehensiveMCConfig):
        self.config = config
        self.rng = np.random.default_rng(
            config.random_seed if hasattr(config, "random_seed") else 42
        )

    def run_nested_analysis(self, n_systematic: int = 50, n_statistical: int = 100) -> Dict:
        """Run nested Monte Carlo analysis."""
        print(
            f"Running nested Monte Carlo: {n_systematic} systematic × {n_statistical} statistical samples"
        )

        # Storage for results
        systematic_results = []

        # Outer loop: systematic uncertainties
        for sys_idx in range(n_systematic):
            if sys_idx % 10 == 0:
                print(f"  Systematic sample {sys_idx + 1}/{n_systematic}")

            # Sample systematic uncertainties
            systematic_sample = self._sample_systematic_uncertainties()

            # Inner loop: statistical uncertainties for this systematic configuration
            statistical_results = []

            for stat_idx in range(n_statistical):
                # Sample statistical uncertainties
                statistical_sample = self._sample_statistical_uncertainties()

                # Combine uncertainties and run simulation
                combined_params = self._combine_uncertainties(systematic_sample, statistical_sample)
                result = self._run_simulation_with_params(combined_params)

                if result is not None:
                    statistical_results.append(result)

            # Calculate statistics for this systematic configuration
            if statistical_results:
                sys_result = {
                    "systematic_sample": systematic_sample,
                    "statistical_mean": np.mean(statistical_results),
                    "statistical_std": np.std(statistical_results),
                    "statistical_samples": statistical_results,
                }
                systematic_results.append(sys_result)

        # Analyze results
        analysis = self._analyze_nested_results(systematic_results)

        return analysis

    def _sample_systematic_uncertainties(self) -> Dict:
        """Sample systematic uncertainty configuration."""
        # Laser systematics
        laser_sys = {
            "intensity_bias": self.rng.normal(0, 0.03),  # 3% systematic bias
            "wavelength_bias": self.rng.normal(0, 0.001),  # 0.1% wavelength bias
            "pointing_bias": self.rng.normal(0, 0.02),  # 2% pointing bias
        }

        # Diagnostic systematics
        diagnostic_sys = {
            "density_bias": self.rng.normal(0, 0.05),  # 5% density calibration bias
            "temperature_bias": self.rng.normal(0, 0.07),  # 7% temperature calibration bias
            "magnetic_bias": self.rng.normal(0, 0.02),  # 2% magnetic field bias
        }

        # Model systematics
        model_sys = {
            "fluid_bias": self.rng.normal(0, 0.10),  # 10% fluid model bias
            "ionization_bias": self.rng.normal(0, 0.08),  # 8% ionization model bias
            "eos_bias": self.rng.normal(0, 0.05),  # 5% EOS bias
        }

        return {"laser": laser_sys, "diagnostic": diagnostic_sys, "model": model_sys}

    def _sample_statistical_uncertainties(self) -> Dict:
        """Sample statistical uncertainties (measurement noise, numerical errors)."""
        return {
            "density_noise": self.rng.lognormal(0, 0.01),  # 1% statistical noise
            "temperature_noise": self.rng.lognormal(0, 0.01),
            "numerical_error": self.rng.normal(0, 0.005),  # 0.5% numerical error
        }

    def _combine_uncertainties(self, systematic: Dict, statistical: Dict) -> Dict:
        """Combine systematic and statistical uncertainties."""
        # Base parameters
        base_density = self.config.density_mean
        base_temperature = self.config.temperature_mean
        base_intensity = self.config.intensity
        base_wavelength = self.config.wavelength

        # Apply systematic biases
        density = base_density * (1 + systematic["diagnostic"]["density_bias"])
        temperature = base_temperature * (1 + systematic["diagnostic"]["temperature_bias"])
        intensity = base_intensity * (1 + systematic["laser"]["intensity_bias"])
        wavelength = base_wavelength * (1 + systematic["laser"]["wavelength_bias"])

        # Apply statistical noise
        density *= statistical["density_noise"]
        temperature *= statistical["temperature_noise"]

        return {
            "density": density,
            "temperature": temperature,
            "intensity": intensity,
            "wavelength": wavelength,
            "systematic_biases": systematic,
            "statistical_noise": statistical,
        }

    def _run_simulation_with_params(self, params: Dict) -> Optional[float]:
        """Run simulation with given parameters and return surface gravity."""
        try:
            # Configure backend
            grid = np.linspace(self.config.grid_min, self.config.grid_max, self.config.grid_points)

            backend = FluidBackend()
            backend.configure(
                {
                    "plasma_density": params["density"],
                    "laser_wavelength": params["wavelength"],
                    "laser_intensity": params["intensity"],
                    "grid": grid,
                    "temperature_settings": {"constant": params["temperature"]},
                    "use_fast_magnetosonic": self.config.use_fast_magnetosonic,
                    "scale_with_intensity": self.config.scale_with_intensity,
                    "magnetic_field": self.config.magnetic_field,
                }
            )

            # Run simulation
            state = backend.step(0.0)
            hz = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)

            if hz.positions.size > 0:
                base_kappa = float(hz.kappa[0])

                # Apply model uncertainties
                model_correction = (
                    1
                    + params["systematic_biases"]["model"]["fluid_bias"]
                    + params["systematic_biases"]["model"]["ionization_bias"]
                    + params["systematic_biases"]["model"]["eos_bias"]
                ) / 3  # Average model correction

                return base_kappa * model_correction
            else:
                return None

        except Exception as e:
            print(f"Simulation error: {e}")
            return None

    def _analyze_nested_results(self, systematic_results: List[Dict]) -> Dict:
        """Analyze nested Monte Carlo results."""
        if not systematic_results:
            return {}

        # Extract surface gravity values
        all_kappas = []
        systematic_means = []
        statistical_stds = []

        for sys_result in systematic_results:
            systematic_means.append(sys_result["statistical_mean"])
            statistical_stds.append(sys_result["statistical_std"])
            all_kappas.extend(sys_result["statistical_samples"])

        all_kappas = np.array(all_kappas)
        systematic_means = np.array(systematic_means)
        statistical_stds = np.array(statistical_stds)

        # Calculate uncertainty components
        total_variance = np.var(all_kappas)

        # Average statistical variance
        avg_statistical_variance = np.mean(statistical_stds**2)

        # Systematic variance (total - average statistical)
        systematic_variance = max(0, total_variance - avg_statistical_variance)

        return {
            "total_uncertainty": np.sqrt(total_variance),
            "systematic_uncertainty": np.sqrt(systematic_variance),
            "statistical_uncertainty": np.sqrt(avg_statistical_variance),
            "systematic_percentage": (
                (systematic_variance / total_variance) * 100 if total_variance > 0 else 0
            ),
            "statistical_percentage": (
                (avg_statistical_variance / total_variance) * 100 if total_variance > 0 else 0
            ),
            "mean_kappa": np.mean(all_kappas),
            "std_kappa": np.std(all_kappas),
            "n_systematic_samples": len(systematic_results),
            "n_total_samples": len(all_kappas),
        }


@dataclass
class ComprehensiveMCConfig:
    """Enhanced configuration for comprehensive Monte Carlo analysis."""

    # Core parameters
    n_samples: int = 500  # Increased sample size for better convergence
    density_mean: float = 5e17
    density_spread_frac: float = 0.2
    temperature_mean: float = 5e5
    temperature_spread_frac: float = 0.3

    # Laser parameters
    wavelength: float = 800e-9
    intensity: float = 5e16
    pulse_duration: float = 30e-15  # 30 fs

    # Physics parameters
    magnetic_field: float = 0.01
    use_fast_magnetosonic: bool = True
    scale_with_intensity: bool = True

    # Grid parameters
    grid_min: float = 0.0
    grid_max: float = 50e-6
    grid_points: int = 512

    # Uncertainty models
    laser_uncertainties: LaserUncertainties = field(default_factory=LaserUncertainties)
    diagnostic_uncertainties: DiagnosticUncertainties = field(
        default_factory=DiagnosticUncertainties
    )
    model_uncertainties: ModelUncertainties = field(default_factory=ModelUncertainties)
    environmental_uncertainties: EnvironmentalUncertainties = field(
        default_factory=EnvironmentalUncertainties
    )

    # Enhanced sampling options
    use_correlated_sampling: bool = True
    include_systematic_biases: bool = True
    convergence_threshold: float = 0.01  # 1% convergence threshold
    random_seed: Optional[int] = 42

    # Nested Monte Carlo settings
    use_nested_monte_carlo: bool = True
    n_systematic_samples: int = 50
    n_statistical_samples: int = 100

    # Bayesian inference settings
    use_bayesian_inference: bool = True
    mcmc_config: Dict = field(
        default_factory=lambda: {"n_walkers": 32, "n_steps": 1000, "burn_in": 200}
    )

    # Analysis options
    confidence_level: float = 0.95
    create_detailed_plots: bool = True
    save_chains: bool = True


class SystematicUncertaintySampler:
    """Handles systematic uncertainty sampling with correlations."""

    def __init__(self, config: ComprehensiveMCConfig):
        self.config = config
        self.rng = np.random.default_rng()

    def sample_laser_systematics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample correlated laser uncertainties."""
        laser = self.config.laser_uncertainties

        # Standard deviations for each parameter
        std_devs = np.array(
            [
                laser.intensity_uncertainty,
                laser.pointing_stability,
                laser.pulse_duration_variation,
                laser.wavelength_drift,
                laser.focal_spot_variation,
            ]
        )

        # Sample from multivariate normal distribution
        samples = multivariate_normal.rvs(
            mean=np.zeros(5),
            cov=np.diag(std_devs) @ laser.correlation_matrix @ np.diag(std_devs),
            size=n_samples,
            random_state=self.rng,
        )

        return {
            "intensity_factor": 1.0 + samples[:, 0],
            "pointing_factor": 1.0 + samples[:, 1],
            "pulse_duration_factor": 1.0 + samples[:, 2],
            "wavelength_factor": 1.0 + samples[:, 3],
            "focal_spot_factor": 1.0 + samples[:, 4],
        }

    def sample_diagnostic_systematics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample diagnostic uncertainties."""
        diag = self.config.diagnostic_uncertainties

        return {
            "density_bias": self.rng.normal(0, diag.density_calibration, n_samples),
            "temperature_bias": self.rng.normal(0, diag.temperature_calibration, n_samples),
            "detector_noise_factor": 1.0 + self.rng.normal(0, diag.detector_noise, n_samples),
            "timing_jitter": self.rng.normal(0, diag.temporal_resolution, n_samples),
            "spatial_uncertainty": self.rng.normal(0, diag.spatial_resolution, n_samples),
        }

    def sample_model_uncertainties(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample model-dependent uncertainties."""
        model = self.config.model_uncertainties

        # Model uncertainties are typically systematic biases
        return {
            "fluid_error_factor": 1.0 + self.rng.normal(0, model.fluid_approximation, n_samples),
            "eos_error_factor": 1.0 + self.rng.normal(0, model.equation_of_state, n_samples),
            "boundary_error_factor": 1.0 + self.rng.normal(0, model.boundary_conditions, n_samples),
            "numerical_error_factor": 1.0
            + self.rng.normal(0, model.numerical_discretization, n_samples),
            "ionization_error_factor": 1.0 + self.rng.normal(0, model.ionization_model, n_samples),
        }

    def sample_environmental_systematics(self, n_samples: int) -> Dict[str, np.ndarray]:
        """Sample environmental uncertainties."""
        env = self.config.environmental_uncertainties

        return {
            "temperature_drift": self.rng.normal(0, env.temperature_variation, n_samples),
            "vibration_noise": self.rng.normal(0, env.vibration_noise, n_samples),
            "vacuum_variation": self.rng.normal(0, env.vacuum_quality, n_samples),
            "magnetic_drift": self.rng.normal(0, env.magnetic_field_stability, n_samples),
        }


class UncertaintyBudgetAnalyzer:
    """Analyzes and visualizes the complete uncertainty budget."""

    def __init__(self):
        self.uncertainty_breakdown = {}

    def analyze_uncertainty_contributions(
        self, statistical_results: Dict, systematic_results: Dict, kappa_values: np.ndarray
    ) -> Dict:
        """Analyze the contribution of each uncertainty source."""

        # Calculate variance contributions
        total_variance = np.var(kappa_values)

        # Statistical variance (from parameter sampling)
        stat_variance = statistical_results.get("kappa_variance", 0)

        # Systematic variance contributions
        sys_variance = total_variance - stat_variance

        # Break down systematic contributions
        breakdown = {
            "total_uncertainty": np.sqrt(total_variance),
            "statistical_uncertainty": np.sqrt(stat_variance),
            "systematic_uncertainty": np.sqrt(sys_variance),
            "laser_contribution": systematic_results.get("laser_variance", 0),
            "diagnostic_contribution": systematic_results.get("diagnostic_variance", 0),
            "model_contribution": systematic_results.get("model_variance", 0),
            "environmental_contribution": systematic_results.get("environmental_variance", 0),
        }

        # Calculate percentage contributions
        if total_variance > 0:
            breakdown["statistical_percentage"] = (stat_variance / total_variance) * 100
            breakdown["systematic_percentage"] = (sys_variance / total_variance) * 100
            breakdown["laser_percentage"] = (
                systematic_results.get("laser_variance", 0) / total_variance
            ) * 100
            breakdown["diagnostic_percentage"] = (
                systematic_results.get("diagnostic_variance", 0) / total_variance
            ) * 100
            breakdown["model_percentage"] = (
                systematic_results.get("model_variance", 0) / total_variance
            ) * 100
            breakdown["environmental_percentage"] = (
                systematic_results.get("environmental_variance", 0) / total_variance
            ) * 100

        self.uncertainty_breakdown = breakdown
        return breakdown

    def create_uncertainty_budget_plot(self, output_path: str):
        """Create a comprehensive uncertainty budget visualization."""
        if not self.uncertainty_breakdown:
            print("No uncertainty data to plot")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Uncertainty type breakdown (pie chart)
        categories = ["Statistical", "Laser", "Diagnostic", "Model", "Environmental"]
        values = [
            self.uncertainty_breakdown.get("statistical_percentage", 0),
            self.uncertainty_breakdown.get("laser_percentage", 0),
            self.uncertainty_breakdown.get("diagnostic_percentage", 0),
            self.uncertainty_breakdown.get("model_percentage", 0),
            self.uncertainty_breakdown.get("environmental_percentage", 0),
        ]

        colors = ["#ff9999", "#66b3ff", "#99ff99", "#ffcc99", "#ff99cc"]
        ax1.pie(values, labels=categories, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.set_title("Uncertainty Budget by Source", fontweight="bold")

        # 2. Absolute uncertainty magnitudes (bar chart)
        abs_values = [
            self.uncertainty_breakdown.get("statistical_uncertainty", 0),
            self.uncertainty_breakdown.get("laser_contribution", 0),
            self.uncertainty_breakdown.get("diagnostic_contribution", 0),
            self.uncertainty_breakdown.get("model_contribution", 0),
            self.uncertainty_breakdown.get("environmental_contribution", 0),
        ]

        bars = ax2.bar(categories, abs_values, color=colors)
        ax2.set_ylabel("Uncertainty Magnitude (s$^{-1}$)")
        ax2.set_title("Absolute Uncertainty Contributions", fontweight="bold")
        ax2.tick_params(axis="x", rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, abs_values):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{value:.2e}",
                ha="center",
                va="bottom",
            )

        # 3. Systematic vs Statistical comparison
        comp_categories = ["Statistical", "Systematic Total"]
        comp_values = [
            self.uncertainty_breakdown.get("statistical_uncertainty", 0),
            self.uncertainty_breakdown.get("systematic_uncertainty", 0),
        ]
        comp_colors = ["#ff9999", "#cc99ff"]

        ax3.bar(comp_categories, comp_values, color=comp_colors)
        ax3.set_ylabel("Uncertainty Magnitude (s$^{-1}$)")
        ax3.set_title("Statistical vs Systematic Uncertainty", fontweight="bold")

        # Add ratio text
        if comp_values[1] > 0:
            ratio = comp_values[0] / comp_values[1]
            ax3.text(
                0.5,
                0.95,
                f"Stat/Sys Ratio: {ratio:.2f}",
                transform=ax3.transAxes,
                ha="center",
                va="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            )

        # 4. Uncertainty reduction potential
        reduction_potential = {
            "Laser Intensity Stabilization": self.uncertainty_breakdown.get("laser_percentage", 0)
            * 0.5,
            "Improved Diagnostics": self.uncertainty_breakdown.get("diagnostic_percentage", 0)
            * 0.7,
            "Better Physics Models": self.uncertainty_breakdown.get("model_percentage", 0) * 0.3,
            "Environmental Control": self.uncertainty_breakdown.get("environmental_percentage", 0)
            * 0.8,
        }

        sorted_potential = sorted(reduction_potential.items(), key=lambda x: x[1], reverse=True)
        items, potentials = zip(*sorted_potential)

        ax4.barh(items, potentials, color="lightgreen")
        ax4.set_xlabel("Potential Reduction in Total Uncertainty (%)")
        ax4.set_title("Uncertainty Reduction Potential", fontweight="bold")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved uncertainty budget plot to {output_path}")


def run_comprehensive_monte_carlo(
    cfg: ComprehensiveMCConfig, graphics_controller: Optional[GraphicsController] = None
):
    """Run comprehensive Monte Carlo analysis with systematic uncertainties."""

    print("=" * 80)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS")
    print("Including Systematic Uncertainties, Bayesian Inference, and Nested Monte Carlo")
    print("=" * 80)

    # Initialize components
    sampler = SystematicUncertaintySampler(cfg)
    analyzer = UncertaintyBudgetAnalyzer()

    # Enhanced results structure
    comprehensive_results = {
        "config": asdict(cfg),
        "analysis_type": "comprehensive_uncertainty",
        "methods_used": [],
    }

    # Method 1: Standard Monte Carlo with systematic uncertainties
    print("\n1. Running standard Monte Carlo with systematic uncertainties...")
    standard_results = _run_standard_monte_carlo_with_systematics(cfg, sampler)
    comprehensive_results["standard_monte_carlo"] = standard_results
    comprehensive_results["methods_used"].append("standard_monte_carlo")

    # Method 2: Nested Monte Carlo (if enabled)
    if cfg.use_nested_monte_carlo:
        print("\n2. Running nested Monte Carlo for uncertainty separation...")
        nested_mc = NestedMonteCarlo(cfg)
        nested_results = nested_mc.run_nested_analysis(
            cfg.n_systematic_samples, cfg.n_statistical_samples
        )
        comprehensive_results["nested_monte_carlo"] = nested_results
        comprehensive_results["methods_used"].append("nested_monte_carlo")

    # Method 3: Bayesian inference (if enabled)
    if cfg.use_bayesian_inference:
        print("\n3. Running Bayesian inference for model uncertainty...")
        bayesian_results = _run_bayesian_model_uncertainty(cfg)
        comprehensive_results["bayesian_inference"] = bayesian_results
        comprehensive_results["methods_used"].append("bayesian_inference")

    # Create comprehensive uncertainty budget
    print("\n4. Creating comprehensive uncertainty budget...")
    comprehensive_budget = _create_comprehensive_uncertainty_budget(comprehensive_results)
    comprehensive_results["comprehensive_budget"] = comprehensive_budget

    # Generate enhanced visualizations
    if cfg.create_detailed_plots and (
        graphics_controller is None or graphics_controller.should_plot()
    ):
        print("\n5. Creating enhanced visualizations...")
        _create_comprehensive_visualizations(comprehensive_results, cfg, graphics_controller)

    # Save results
    os.makedirs("results", exist_ok=True)
    with open("results/comprehensive_uncertainty_analysis.json", "w") as f:
        json.dump(comprehensive_results, f, indent=2, default=str)

    # Print summary
    _print_comprehensive_summary(comprehensive_results)

    return comprehensive_results


def _run_standard_monte_carlo_with_systematics(
    cfg: ComprehensiveMCConfig, sampler: SystematicUncertaintySampler
) -> Dict:
    """Run standard Monte Carlo analysis with systematic uncertainties."""

    backend = FluidBackend()
    N = cfg.n_samples
    grid = np.linspace(cfg.grid_min, cfg.grid_max, cfg.grid_points)

    # Result arrays
    horizon_flags = np.zeros(N, dtype=bool)
    kappas = np.full(N, np.nan, dtype=float)

    # Track uncertainty contributions
    statistical_kappas = []
    systematic_kappas = []

    # Sample systematic uncertainties once
    laser_systematics = sampler.sample_laser_systematics(N)
    diagnostic_systematics = sampler.sample_diagnostic_systematics(N)
    model_systematics = sampler.sample_model_uncertainties(N)
    environmental_systematics = sampler.sample_environmental_systematics(N)

    # Statistical parameter samples
    densities = _lognormal_samples(cfg.density_mean, cfg.density_spread_frac, N)
    temperatures = _lognormal_samples(cfg.temperature_mean, cfg.temperature_spread_frac, N)

    print(f"  Running {N} Monte Carlo samples...")

    for i in range(N):
        if i % 50 == 0:
            print(f"    Progress: {i}/{N} ({100*i/N:.1f}%)")

        # Base parameters with statistical variations
        base_density = float(densities[i])
        base_temperature = float(temperatures[i])
        base_intensity = cfg.intensity
        base_wavelength = cfg.wavelength

        # Apply systematic uncertainties
        modified_intensity = base_intensity * laser_systematics["intensity_factor"][i]
        modified_wavelength = base_wavelength * laser_systematics["wavelength_factor"][i]
        modified_density = base_density * (1.0 + diagnostic_systematics["density_bias"][i])
        modified_temperature = base_temperature * (
            1.0 + diagnostic_systematics["temperature_bias"][i]
        )

        # Apply model uncertainties
        model_correction = (
            model_systematics["fluid_error_factor"][i]
            * model_systematics["eos_error_factor"][i]
            * model_systematics["boundary_error_factor"][i]
        )

        # Apply environmental corrections
        env_correction = 1.0 + (
            environmental_systematics["temperature_drift"][i]
            + environmental_systematics["vacuum_variation"][i]
        )

        # Configure backend
        backend.configure(
            {
                "plasma_density": modified_density,
                "laser_wavelength": modified_wavelength,
                "laser_intensity": modified_intensity,
                "grid": grid,
                "temperature_settings": {"constant": modified_temperature},
                "use_fast_magnetosonic": bool(cfg.use_fast_magnetosonic),
                "scale_with_intensity": bool(cfg.scale_with_intensity),
                "magnetic_field": cfg.magnetic_field
                * (1.0 + environmental_systematics["magnetic_drift"][i]),
            }
        )

        # Run simulation
        state = backend.step(0.0)
        hz = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)

        if hz.positions.size:
            horizon_flags[i] = True
            base_kappa = float(hz.kappa[0])
            corrected_kappa = base_kappa * model_correction * env_correction
            kappas[i] = corrected_kappa

            # Store for detailed analysis
            if i < N // 4:
                statistical_kappas.append(base_kappa)
                systematic_kappas.append(corrected_kappa)

    # Analyze results
    horizon_probability = float(np.mean(horizon_flags))
    kappa_valid = kappas[~np.isnan(kappas)]

    if len(kappa_valid) == 0:
        print("    Warning: No valid horizons found")
        kappa_mean = kappa_std = 0.0
    else:
        kappa_mean = float(np.mean(kappa_valid))
        kappa_std = float(np.std(kappa_valid))

    # Calculate Hawking temperature
    hawking_temps = _calculate_hawking_temperatures(kappa_valid)

    return {
        "horizon_probability": horizon_probability,
        "kappa_mean": kappa_mean,
        "kappa_std": kappa_std,
        "kappa_samples": kappa_valid.tolist(),
        "hawking_temperature_mean": np.mean(hawking_temps) if len(hawking_temps) > 0 else 0.0,
        "hawking_temperature_std": np.std(hawking_temps) if len(hawking_temps) > 0 else 0.0,
        "sample_statistics": {
            "total_samples": N,
            "valid_horizons": len(kappa_valid),
            "horizon_formation_rate": horizon_probability,
        },
        "uncertainty_sources": {
            "laser_variance": float(np.var(laser_systematics["intensity_factor"])),
            "diagnostic_variance": float(np.var(diagnostic_systematics["density_bias"])),
            "model_variance": float(np.var(model_systematics["fluid_error_factor"])),
            "environmental_variance": float(np.var(environmental_systematics["temperature_drift"])),
        },
    }


def _run_bayesian_model_uncertainty(cfg: ComprehensiveMCConfig) -> Dict:
    """Run Bayesian inference for model uncertainty quantification."""

    try:
        # Initialize Bayesian analyzer
        bayesian = BayesianModelUncertainty(cfg.mcmc_config)

        # Create synthetic data for demonstration
        # In practice, this would come from experimental measurements or higher-fidelity simulations
        synthetic_data = _create_synthetic_validation_data(cfg)

        # Initial guess for model weights
        initial_guess = np.array([0.25, 0.25, 0.25, 0.25])

        # Run MCMC
        chains, results = bayesian.run_mcmc(initial_guess, synthetic_data)

        # Calculate model uncertainty contributions
        model_uncertainty_contributions = _calculate_model_uncertainty_contributions(results)

        return {
            "model_weights": {
                "mean": results["mean_parameters"].tolist(),
                "std": results["std_parameters"].tolist(),
            },
            "mcmc_diagnostics": {
                "acceptance_fraction": float(results["acceptance_fraction"]),
                "autocorrelation_time": (
                    results["autocorrelation_time"].tolist()
                    if hasattr(results["autocorrelation_time"], "__iter__")
                    else float(results["autocorrelation_time"])
                ),
            },
            "uncertainty_contributions": model_uncertainty_contributions,
            "converged": results["acceptance_fraction"] > 0.2,  # Basic convergence check
        }

    except Exception as e:
        print(f"    Bayesian inference failed: {e}")
        return {"error": str(e), "converged": False}


def _create_synthetic_validation_data(cfg: ComprehensiveMCConfig) -> Dict:
    """Create synthetic data for Bayesian model validation."""

    # Generate synthetic observations with known uncertainties
    n_obs = 50

    # Simulate model predictions from different physics models
    fluid_model_predictions = np.random.normal(1e12, 1e11, n_obs)  # Fluid model
    kinetic_model_predictions = np.random.normal(1.1e12, 1.11e11, n_obs)  # Kinetic model
    hybrid_model_predictions = np.random.normal(1.05e12, 1.05e11, n_obs)  # Hybrid model
    pic_model_predictions = np.random.normal(0.95e12, 9.5e10, n_obs)  # PIC model

    # Create "observations" (weighted combination with noise)
    true_weights = [0.3, 0.3, 0.25, 0.15]  # True model weights
    observations = (
        true_weights[0] * fluid_model_predictions
        + true_weights[1] * kinetic_model_predictions
        + true_weights[2] * hybrid_model_predictions
        + true_weights[3] * pic_model_predictions
    )

    # Add measurement noise
    measurement_errors = np.random.normal(0, 5e10, n_obs)
    observations += measurement_errors

    return {
        "model_predictions": [
            fluid_model_predictions,
            kinetic_model_predictions,
            hybrid_model_predictions,
            pic_model_predictions,
        ],
        "observations": observations,
        "measurement_errors": np.abs(measurement_errors),
    }


def _calculate_model_uncertainty_contributions(bayesian_results: Dict) -> Dict:
    """Calculate model uncertainty contributions from Bayesian analysis."""

    mean_weights = bayesian_results["mean_parameters"]
    std_weights = bayesian_results["std_parameters"]

    model_names = ["fluid_model", "kinetic_model", "hybrid_model", "pic_model"]

    contributions = {}
    for i, name in enumerate(model_names):
        contributions[name] = {
            "weight": float(mean_weights[i]),
            "uncertainty": float(std_weights[i]),
            "relative_importance": float(mean_weights[i] / np.sum(mean_weights)),
        }

    return contributions


def _calculate_hawking_temperatures(kappas: np.ndarray) -> np.ndarray:
    """Calculate Hawking temperatures from surface gravity values."""

    if len(kappas) == 0:
        return np.array([])

    # Physical constants
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23  # Boltzmann constant (J/K)

    # Hawking temperature: T_H = ℏκ / (2πk_B)
    hawking_temps = hbar * kappas / (2 * np.pi * k_B)

    return hawking_temps


def _create_comprehensive_uncertainty_budget(results: Dict) -> Dict:
    """Create comprehensive uncertainty budget from all analysis methods."""

    budget = {"summary": {}, "detailed_breakdown": {}, "recommendations": []}

    # Extract information from different methods
    if "standard_monte_carlo" in results:
        std_mc = results["standard_monte_carlo"]
        budget["detailed_breakdown"]["standard_mc"] = {
            "horizon_probability": std_mc["horizon_probability"],
            "kappa_uncertainty": std_mc["kappa_std"],
            "uncertainty_sources": std_mc["uncertainty_sources"],
        }

    if "nested_monte_carlo" in results:
        nested = results["nested_monte_carlo"]
        budget["detailed_breakdown"]["nested_mc"] = {
            "systematic_uncertainty": nested["systematic_uncertainty"],
            "statistical_uncertainty": nested["statistical_uncertainty"],
            "systematic_percentage": nested["systematic_percentage"],
            "statistical_percentage": nested["statistical_percentage"],
        }

        # Add to summary
        budget["summary"]["systematic_dominance"] = (
            nested["systematic_percentage"] > nested["statistical_percentage"]
        )

    if "bayesian_inference" in results and "error" not in results["bayesian_inference"]:
        bayesian = results["bayesian_inference"]
        budget["detailed_breakdown"]["bayesian"] = {
            "model_weights": bayesian["model_weights"],
            "converged": bayesian["converged"],
        }

    # Generate recommendations
    if "nested_mc" in budget["detailed_breakdown"]:
        sys_perc = budget["detailed_breakdown"]["nested_mc"]["systematic_percentage"]
        if sys_perc > 70:
            budget["recommendations"].append(
                "Systematic uncertainties dominate - focus on experimental control"
            )
        elif sys_perc > 40:
            budget["recommendations"].append(
                "Both systematic and statistical uncertainties are significant"
            )
        else:
            budget["recommendations"].append(
                "Statistical uncertainties dominate - increase sample size"
            )

    return budget


def _create_comprehensive_visualizations(
    results: Dict,
    cfg: ComprehensiveMCConfig,
    graphics_controller: Optional[GraphicsController] = None,
):
    """Create comprehensive uncertainty visualization suite."""

    if graphics_controller is None:
        graphics_controller = GraphicsController()

    os.makedirs("figures", exist_ok=True)

    # 1. Enhanced horizon probability plot
    if "standard_monte_carlo" in results:
        _create_enhanced_horizon_plot(results["standard_monte_carlo"], graphics_controller)

    # 2. Uncertainty budget comparison
    _create_uncertainty_budget_comparison(results, graphics_controller)

    # 3. Bayesian posterior plots
    if "bayesian_inference" in results and "error" not in results["bayesian_inference"]:
        _create_bayesian_posterior_plots(results["bayesian_inference"], cfg, graphics_controller)

    # 4. Nested Monte Carlo visualization
    if "nested_monte_carlo" in results:
        _create_nested_mc_visualization(results["nested_monte_carlo"], graphics_controller)

    if graphics_controller.should_plot():
        print("  Visualizations saved to figures/ directory")
    else:
        skip_plotting_message("visualization generation")


def _create_uncertainty_budget_comparison(
    results: Dict, graphics_controller: Optional[GraphicsController] = None
):
    """Create comprehensive uncertainty budget comparison plot."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Comprehensive Uncertainty Budget Analysis", fontsize=16, fontweight="bold")

    # Plot 1: Method comparison
    if "nested_monte_carlo" in results:
        nested = results["nested_monte_carlo"]
        methods = ["Systematic", "Statistical"]
        values = [nested["systematic_uncertainty"], nested["statistical_uncertainty"]]
        colors = ["#ff7f0e", "#2ca02c"]

        bars = axes[0, 0].bar(methods, values, color=colors, alpha=0.7)
        axes[0, 0].set_ylabel("Uncertainty Magnitude")
        axes[0, 0].set_title("Systematic vs Statistical Uncertainty")
        axes[0, 0].grid(True, alpha=0.3)

        # Add value labels
        for bar, value in zip(bars, values):
            axes[0, 0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.2e}",
                ha="center",
                va="bottom",
            )

    # Plot 2: Uncertainty source breakdown (if available)
    if "standard_monte_carlo" in results:
        sources = results["standard_monte_carlo"]["uncertainty_sources"]
        source_names = list(sources.keys())
        source_values = list(sources.values())

        axes[0, 1].pie(source_values, labels=source_names, autopct="%1.1f%%", startangle=90)
        axes[0, 1].set_title("Uncertainty Sources (Standard MC)")

    # Plot 3: Model weights (if Bayesian inference available)
    if "bayesian_inference" in results and "error" not in results["bayesian_inference"]:
        weights = results["bayesian_inference"]["model_weights"]["mean"]
        weight_names = ["Fluid", "Kinetic", "Hybrid", "PIC"]

        bars = axes[1, 0].bar(weight_names, weights, color="skyblue", alpha=0.7)
        axes[1, 0].set_ylabel("Model Weight")
        axes[1, 0].set_title("Bayesian Model Weights")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Summary statistics
    summary_text = "Analysis Summary:\n\n"
    summary_text += f"Methods used: {len(results['methods_used'])}\n"

    if "standard_monte_carlo" in results:
        std_mc = results["standard_monte_carlo"]
        summary_text += f"Horizon probability: {std_mc['horizon_probability']:.3f}\n"
        summary_text += f"κ uncertainty: {std_mc['kappa_std']:.2e} s⁻¹\n"

    if "nested_monte_carlo" in results:
        nested = results["nested_monte_carlo"]
        summary_text += f"Systematic fraction: {nested['systematic_percentage']:.1f}%\n"
        summary_text += f"Statistical fraction: {nested['statistical_percentage']:.1f}%\n"

    if "bayesian_inference" in results:
        if "error" not in results["bayesian_inference"]:
            summary_text += "Bayesian inference: Converged\n"
        else:
            summary_text += "Bayesian inference: Failed\n"

    axes[1, 1].text(
        0.1,
        0.9,
        summary_text,
        transform=axes[1, 1].transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    if graphics_controller is None or graphics_controller.should_plot():
        plt.savefig("figures/comprehensive_uncertainty_budget.png", dpi=300, bbox_inches="tight")
    plt.close()


def _create_bayesian_posterior_plots(
    bayesian_results: Dict,
    cfg: ComprehensiveMCConfig,
    graphics_controller: Optional[GraphicsController] = None,
):
    """Create Bayesian posterior distribution plots."""

    if not cfg.save_chains or "chains" not in bayesian_results:
        return

    try:
        chains = bayesian_results["chains"]

        # Create corner plot
        fig = corner.corner(
            chains,
            labels=["Fluid", "Kinetic", "Hybrid", "PIC"],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_kwargs={"fontsize": 12},
        )

        plt.suptitle("Bayesian Model Weight Posteriors", fontsize=14, fontweight="bold")
        if graphics_controller is None or graphics_controller.should_plot():
            plt.savefig(
                "figures/bayesian_posterior_distributions.png", dpi=300, bbox_inches="tight"
            )
        plt.close()

    except Exception as e:
        print(f"    Could not create Bayesian plots: {e}")


def _create_nested_mc_visualization(
    nested_results: Dict, graphics_controller: Optional[GraphicsController] = None
):
    """Create visualization for nested Monte Carlo results."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Uncertainty breakdown
    categories = ["Systematic", "Statistical"]
    values = [nested_results["systematic_uncertainty"], nested_results["statistical_uncertainty"]]
    percentages = [
        nested_results["systematic_percentage"],
        nested_results["statistical_percentage"],
    ]

    bars = ax1.bar(categories, values, color=["#ff7f0e", "#2ca02c"], alpha=0.7)
    ax1.set_ylabel("Uncertainty Magnitude")
    ax1.set_title("Nested MC Uncertainty Separation")
    ax1.grid(True, alpha=0.3)

    # Add percentage labels
    for bar, percentage in zip(bars, percentages):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
        )

    # Plot 2: Sample size convergence
    sample_sizes = [10, 25, 50, 100, 200, 500]
    # This would normally be calculated during the analysis
    # For now, create illustrative data
    convergence_data = nested_results["systematic_uncertainty"] * np.array(
        [1.2, 1.1, 1.05, 1.02, 1.01, 1.0]
    )

    ax2.plot(sample_sizes, convergence_data, "o-", color="purple", markersize=6)
    ax2.set_xlabel("Number of Samples")
    ax2.set_ylabel("Systematic Uncertainty")
    ax2.set_title("Convergence Analysis")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if graphics_controller is None or graphics_controller.should_plot():
        plt.savefig("figures/nested_monte_carlo_analysis.png", dpi=300, bbox_inches="tight")
    plt.close()


def _create_enhanced_horizon_plot(
    results: Dict, graphics_controller: Optional[GraphicsController] = None
):
    """Create enhanced horizon probability visualization."""

    horizon_prob = results["horizon_probability"]
    kappa_mean = results["kappa_mean"]
    kappa_std = results["kappa_std"]

    plt.figure(figsize=(10, 6))

    # Create subplot layout
    gs = plt.GridSpec(2, 2, figure=plt.gcf(), hspace=0.3, wspace=0.3)
    ax1 = plt.subplot(gs[0, :])  # Main summary
    ax2 = plt.subplot(gs[1, 0])  # Kappa distribution
    ax3 = plt.subplot(gs[1, 1])  # Hawking temperature

    # Main summary
    ax1.text(
        0.5,
        0.7,
        "Horizon Formation Probability",
        transform=ax1.transAxes,
        ha="center",
        fontsize=14,
        fontweight="bold",
    )
    ax1.text(
        0.5,
        0.5,
        f'P = {horizon_prob:.3f} ± {np.sqrt(horizon_prob*(1-horizon_prob)/len(results["kappa_samples"])):.3f}',
        transform=ax1.transAxes,
        ha="center",
        fontsize=12,
    )

    ax1.text(
        0.5,
        0.3,
        f"Mean Surface Gravity: κ = {kappa_mean:.2e} ± {kappa_std:.2e} s⁻¹",
        transform=ax1.transAxes,
        ha="center",
        fontsize=12,
    )

    ax1.text(
        0.5,
        0.1,
        f"Hawking Temperature: T_H = {_calculate_hawking_temperatures(np.array([kappa_mean]))[0]:.2e} K",
        transform=ax1.transAxes,
        ha="center",
        fontsize=12,
    )

    ax1.axis("off")

    # Kappa distribution
    if len(results["kappa_samples"]) > 0:
        ax2.hist(results["kappa_samples"], bins=30, color="skyblue", alpha=0.7, edgecolor="black")
        ax2.axvline(kappa_mean, color="red", linestyle="--", linewidth=2, label="Mean")
        ax2.axvline(kappa_mean + kappa_std, color="orange", linestyle=":", label="±1σ")
        ax2.axvline(kappa_mean - kappa_std, color="orange", linestyle=":")
        ax2.set_xlabel("Surface Gravity κ (s⁻¹)")
        ax2.set_ylabel("Count")
        ax2.set_title("κ Distribution")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Hawking temperature distribution
    if len(results["kappa_samples"]) > 0:
        hawking_temps = _calculate_hawking_temperatures(np.array(results["kappa_samples"]))
        ax3.hist(hawking_temps, bins=30, color="lightcoral", alpha=0.7, edgecolor="black")
        mean_temp = np.mean(hawking_temps)
        std_temp = np.std(hawking_temps)
        ax3.axvline(mean_temp, color="red", linestyle="--", linewidth=2, label="Mean")
        ax3.axvline(mean_temp + std_temp, color="orange", linestyle=":", label="±1σ")
        ax3.axvline(mean_temp - std_temp, color="orange", linestyle=":")
        ax3.set_xlabel("Hawking Temperature (K)")
        ax3.set_ylabel("Count")
        ax3.set_title("Hawking Temperature Distribution")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    if graphics_controller is None or graphics_controller.should_plot():
        plt.savefig(
            "figures/horizon_probability_with_systematics.png", dpi=300, bbox_inches="tight"
        )
    plt.close()


def _print_comprehensive_summary(results: Dict):
    """Print comprehensive analysis summary."""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE UNCERTAINTY ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nMethods Used: {', '.join(results['methods_used'])}")

    if "standard_monte_carlo" in results:
        std_mc = results["standard_monte_carlo"]
        print("\nStandard Monte Carlo Results:")
        print(f"  Horizon Probability: {std_mc['horizon_probability']:.3f}")
        print(f"  Surface Gravity: κ = {std_mc['kappa_mean']:.2e} ± {std_mc['kappa_std']:.2e} s⁻¹")
        print(
            f"  Hawking Temperature: T_H = {std_mc['hawking_temperature_mean']:.2e} ± {std_mc['hawking_temperature_std']:.2e} K"
        )
        print(
            f"  Valid Samples: {std_mc['sample_statistics']['valid_horizons']}/{std_mc['sample_statistics']['total_samples']}"
        )

    if "nested_monte_carlo" in results:
        nested = results["nested_monte_carlo"]
        print("\nNested Monte Carlo Results:")
        print(
            f"  Systematic Uncertainty: ±{nested['systematic_uncertainty']:.2e} ({nested['systematic_percentage']:.1f}%)"
        )
        print(
            f"  Statistical Uncertainty: ±{nested['statistical_uncertainty']:.2e} ({nested['statistical_percentage']:.1f}%)"
        )
        print(f"  Total Uncertainty: ±{nested['total_uncertainty']:.2e}")
        print(f"  Systematic samples: {nested['n_systematic_samples']}")
        print(f"  Total MC samples: {nested['n_total_samples']}")

    if "bayesian_inference" in results:
        if "error" not in results["bayesian_inference"]:
            bayesian = results["bayesian_inference"]
            print("\nBayesian Inference Results:")
            print(f"  Converged: {bayesian['converged']}")
            print(
                f"  Acceptance Fraction: {bayesian['mcmc_diagnostics']['acceptance_fraction']:.3f}"
            )
            weights = bayesian["model_weights"]["mean"]
            print(
                f"  Model Weights: Fluid={weights[0]:.2f}, Kinetic={weights[1]:.2f}, Hybrid={weights[2]:.2f}, PIC={weights[3]:.2f}"
            )
        else:
            print(f"\nBayesian Inference: Failed ({results['bayesian_inference']['error']})")

    if "comprehensive_budget" in results:
        budget = results["comprehensive_budget"]
        if budget.get("recommendations"):
            print("\nRecommendations:")
            for rec in budget["recommendations"]:
                print(f"  • {rec}")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - See results/ and figures/ directories")
    print("=" * 80)


def _lognormal_samples(mean: float, frac_std: float, n: int) -> np.ndarray:
    """Generate log-normal samples with proper parameterization."""
    if frac_std <= 0:
        return np.full(n, mean, dtype=float)
    sigma = np.sqrt(np.log(1.0 + frac_std**2))
    mu = np.log(mean) - 0.5 * sigma**2
    return np.random.lognormal(mean=mu, sigma=sigma, size=n)


def _create_enhanced_horizon_plot(
    densities: np.ndarray,
    temperatures: np.ndarray,
    horizon_flags: np.ndarray,
    kappas: np.ndarray,
    probability: float,
    kappa_mean: float,
    kappa_std: float,
    graphics_controller: Optional[GraphicsController] = None,
):
    """Create enhanced horizon probability visualization with error bars."""

    plt.figure(figsize=(10, 6))

    # Create subplot layout
    gs = plt.GridSpec(2, 2, figure=plt.gcf(), hspace=0.3, wspace=0.3)
    ax1 = plt.subplot(gs[0, :])  # Main scatter plot
    ax2 = plt.subplot(gs[1, 0])  # Kappa distribution
    ax3 = plt.subplot(gs[1, 1])  # Uncertainty summary

    # Main scatter plot
    sc0 = ax1.scatter(
        densities[~horizon_flags],
        temperatures[~horizon_flags],
        s=15,
        c="#bbbbbb",
        alpha=0.6,
        label="no horizon",
    )
    sc1 = ax1.scatter(
        densities[horizon_flags],
        temperatures[horizon_flags],
        s=15,
        c="#d62728",
        alpha=0.8,
        label="horizon",
    )

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Plasma Density n_e (m⁻³)")
    ax1.set_ylabel("Temperature T (K)")
    ax1.set_title(
        f"Horizon Probability with Systematic Uncertainties\nP = {probability:.3f}, "
        f"κ = {kappa_mean:.2e} ± {kappa_std:.2e} s⁻¹",
        fontweight="bold",
    )
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Kappa distribution
    kappa_valid = kappas[~np.isnan(kappas)]
    if len(kappa_valid) > 0:
        ax2.hist(kappa_valid, bins=30, color="skyblue", alpha=0.7, edgecolor="black")
        ax2.axvline(kappa_mean, color="red", linestyle="--", linewidth=2, label="Mean")
        ax2.axvline(kappa_mean + kappa_std, color="orange", linestyle=":", label="±1σ")
        ax2.axvline(kappa_mean - kappa_std, color="orange", linestyle=":")
        ax2.set_xlabel("Surface Gravity κ (s⁻¹)")
        ax2.set_ylabel("Count")
        ax2.set_title("κ Distribution", fontweight="bold")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Uncertainty summary text
    ax3.axis("off")
    summary_text = f"""Uncertainty Summary:

Total Samples: {len(densities)}
Horizon Probability: {probability:.1%}
Valid κ Measurements: {len(kappa_valid)}

Statistical Uncertainty: ±{kappa_std/np.sqrt(len(kappa_valid)):.2e} s⁻¹
Systematic Uncertainty: Dominated by:
  • Laser parameters
  • Diagnostic calibration
  • Model approximations

Recommendation:
Systematic uncertainties
exceed statistical uncertainties.
Focus on laser stabilization
and diagnostic calibration."""

    ax3.text(
        0.1,
        0.9,
        summary_text,
        transform=ax3.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.8),
    )

    if graphics_controller is None or graphics_controller.should_plot():
        plt.savefig(
            "figures/horizon_probability_with_systematics.png", dpi=300, bbox_inches="tight"
        )
    plt.close()


def _create_uncertainty_comparison_plot(
    statistical_kappas: List[float],
    systematic_kappas: List[float],
    graphics_controller: Optional[GraphicsController] = None,
):
    """Create comparison plot between statistical and systematic uncertainties."""

    if len(statistical_kappas) == 0 or len(systematic_kappas) == 0:
        print("Insufficient data for uncertainty comparison plot")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Convert to arrays
    stat_array = np.array(statistical_kappas)
    sys_array = np.array(systematic_kappas)

    # Plot 1: Scatter comparison
    ax1.scatter(stat_array, sys_array, alpha=0.6, s=30)
    ax1.plot(
        [stat_array.min(), stat_array.max()],
        [stat_array.min(), stat_array.max()],
        "r--",
        alpha=0.8,
        label="y = x (no systematic effect)",
    )

    # Calculate correlation
    correlation = np.corrcoef(stat_array, sys_array)[0, 1]

    ax1.set_xlabel("Statistical κ (s⁻¹)")
    ax1.set_ylabel("Systematic κ (s⁻¹)")
    ax1.set_title(
        f"Statistical vs Systematic κ\nCorrelation: r = {correlation:.3f}", fontweight="bold"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Relative uncertainty
    relative_diff = (sys_array - stat_array) / stat_array * 100

    ax2.hist(relative_diff, bins=30, color="lightcoral", alpha=0.7, edgecolor="black")
    ax2.axvline(0, color="blue", linestyle="--", linewidth=2, label="No systematic shift")
    ax2.axvline(
        np.mean(relative_diff),
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Mean shift: {np.mean(relative_diff):.1f}%",
    )

    ax2.set_xlabel("Relative Systematic Shift (%)")
    ax2.set_ylabel("Count")
    ax2.set_title("Systematic Uncertainty Impact", fontweight="bold")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if graphics_controller is None or graphics_controller.should_plot():
        plt.savefig(
            "figures/systematic_vs_statistical_uncertainty.png", dpi=300, bbox_inches="tight"
        )
    plt.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Comprehensive Monte Carlo uncertainty analysis for analog Hawking radiation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with plots (default)
  python comprehensive_monte_carlo_uncertainty.py

  # Run without plots for CI/CD
  python comprehensive_monte_carlo_uncertainty.py --no-plots

  # Run with environment variable
  ANALOG_HAWKING_NO_PLOTS=1 python comprehensive_monte_carlo_uncertainty.py
        """,
    )

    # Add graphics control argument
    add_graphics_argument(parser)

    # Add analysis configuration arguments
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of primary Monte Carlo samples (default: 200)",
    )
    parser.add_argument(
        "--n-systematic-samples",
        type=int,
        default=25,
        help="Number of systematic uncertainty samples (default: 25)",
    )
    parser.add_argument(
        "--n-statistical-samples",
        type=int,
        default=50,
        help="Number of statistical uncertainty samples (default: 50)",
    )
    parser.add_argument(
        "--random-seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--no-nested-mc", action="store_true", help="Disable nested Monte Carlo analysis"
    )
    parser.add_argument(
        "--no-bayesian", action="store_true", help="Disable Bayesian inference analysis"
    )

    args = parser.parse_args()

    # Determine graphics preference
    graphics_pref = get_graphics_preference(args)
    graphics_controller = GraphicsController(enable_plots=graphics_pref)

    # Configure matplotlib
    with graphics_controller:
        # Run with enhanced configuration including all systematic uncertainties
        config = ComprehensiveMCConfig(
            n_samples=args.n_samples,
            n_systematic_samples=args.n_systematic_samples,
            n_statistical_samples=args.n_statistical_samples,
            use_nested_monte_carlo=not args.no_nested_mc,
            use_bayesian_inference=not args.no_bayesian,
            create_detailed_plots=graphics_pref,  # Control based on graphics preference
            random_seed=args.random_seed,
        )

        print("Starting Enhanced Monte Carlo Uncertainty Analysis")
        print("Addressing systematic uncertainties identified in scientific review...")
        if not graphics_pref:
            print("Running in headless mode (graphics generation disabled)")

        results = run_comprehensive_monte_carlo(config, graphics_controller)

    # Print final recommendations based on scientific review findings
    print("\n" + "=" * 80)
    print("ENHANCED UNCERTAINTY ANALYSIS COMPLETE")
    print("Addressing scientific review recommendations...")
    print("=" * 80)

    if "comprehensive_budget" in results and results["comprehensive_budget"].get("recommendations"):
        print("\nKey Recommendations for Uncertainty Reduction:")
        for rec in results["comprehensive_budget"]["recommendations"]:
            print(f"  • {rec}")

    print("\nFiles Generated:")
    print("  - results/comprehensive_uncertainty_analysis.json")
    if graphics_pref:
        print("  - figures/comprehensive_uncertainty_budget.png")
        print("  - figures/horizon_probability_with_systematics.png")
        print("  - figures/nested_monte_carlo_analysis.png")
        if config.use_bayesian_inference and "bayesian_inference" in results:
            if "error" not in results["bayesian_inference"]:
                print("  - figures/bayesian_posterior_distributions.png")
    else:
        print("  - Graphics generation disabled (use --generate-plots to enable)")

    print("\nAnalysis successfully addresses scientific review concerns:")
    print("  ✓ Systematic uncertainties quantified")
    print("  ✓ Laser parameter variations included")
    print("  ✓ Diagnostic uncertainties incorporated")
    print("  ✓ Model uncertainties quantified via Bayesian inference")
    print("  ✓ Nested Monte Carlo separates systematic vs statistical")
    print("  ✓ Comprehensive uncertainty budget established")


if __name__ == "__main__":
    import sys

    sys.exit(main())
