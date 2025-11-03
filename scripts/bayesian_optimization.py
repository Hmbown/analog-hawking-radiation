#!/usr/bin/env python3
"""
Bayesian optimization for analog Hawking radiation detection parameters.
Uses Gaussian Process optimization to efficiently explore parameter space
and find optimal detection conditions.
"""

import argparse
import json
import logging

# Add src to path
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Bayesian optimization
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.run_full_pipeline import run_full_pipeline


@dataclass
class OptimizationConfig:
    """Configuration for Bayesian optimization"""

    n_initial_points: int = 20
    n_calls: int = 100
    random_state: int = 42
    noise: float = 1e-4
    acq_func: str = "EI"  # Expected Improvement
    kappa: float = 1.96  # Exploration parameter

    # Parameter bounds
    intensity_bounds: Tuple[float, float] = (1e16, 1e20)
    density_bounds: Tuple[float, float] = (1e16, 1e20)
    magnetic_bounds: Tuple[float, float] = (0, 100)
    temperature_bounds: Tuple[float, float] = (1e3, 1e5)
    wavelength_bounds: Tuple[float, float] = (400e-9, 1200e-9)
    grid_size_bounds: Tuple[float, float] = (10e-6, 100e-6)
    hybrid_D_bounds: Tuple[float, float] = (1e-6, 100e-6)
    hybrid_eta_bounds: Tuple[float, float] = (0.1, 10.0)


class HawkingOptimization:
    """Bayesian optimization for Hawking radiation detection"""

    def __init__(self, config: OptimizationConfig, use_hybrid: bool = False):
        self.config = config
        self.use_hybrid = use_hybrid

        # Setup logging
        self.setup_logging()

        # Define parameter space
        self.space = self._define_parameter_space()

        # Results storage
        self.results = []
        self.best_params = None
        self.best_value = float("inf")

    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("results/optimization")
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_dir / "optimization.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _define_parameter_space(self) -> List:
        """Define the parameter space for optimization"""

        if self.use_hybrid:
            return [
                Real(*self.config.intensity_bounds, name="laser_intensity"),
                Real(*self.config.density_bounds, name="plasma_density"),
                Real(*self.config.temperature_bounds, name="temperature_constant"),
                Real(*self.config.hybrid_D_bounds, name="mirror_D"),
                Real(*self.config.hybrid_eta_bounds, name="mirror_eta"),
            ]
        else:
            return [
                Real(*self.config.intensity_bounds, name="laser_intensity"),
                Real(*self.config.density_bounds, name="plasma_density"),
                Real(*self.config.magnetic_bounds, name="magnetic_field"),
                Real(*self.config.temperature_bounds, name="temperature_constant"),
                Real(*self.config.wavelength_bounds, name="laser_wavelength"),
                Real(*self.config.grid_size_bounds, name="grid_max"),
            ]

    def objective_function(self, **params) -> float:
        """Objective function to minimize (negative detection rate)"""

        try:
            # Convert parameters to correct types
            params = {k: float(v) for k, v in params.items()}

            # Add fixed parameters
            params.update(
                {
                    "grid_points": 512,
                    "use_fast_magnetosonic": False,
                    "scale_with_intensity": True,
                    "kappa_method": "acoustic_exact",
                    "graybody": "acoustic_wkb",
                    "alpha_gray": 0.8,
                }
            )

            if self.use_hybrid:
                params.update(
                    {
                        "enable_hybrid": True,
                        "hybrid_model": "anabhel",
                        "magnetic_field": None,
                        "laser_wavelength": 800e-9,
                        "grid_max": 50e-6,
                    }
                )
            else:
                params["enable_hybrid"] = False
                if params["magnetic_field"] < 1e-6:
                    params["magnetic_field"] = None

            # Run simulation
            result = run_full_pipeline(**params)

            # Handle edge cases
            if not result.kappa or result.t5sigma_s is None:
                return 1e6  # Large penalty for invalid results

            # Objective: minimize detection time (maximize detection rate)
            detection_time = result.t5sigma_s

            # Add penalty for extreme parameters
            param_penalty = 0
            for key, value in params.items():
                if key.endswith("_bounds"):
                    continue
                if isinstance(value, (int, float)):
                    # Logarithmic penalty for extreme values
                    param_penalty += abs(np.log10(abs(value) + 1e-30))

            # Combined objective
            objective_value = detection_time + 0.01 * param_penalty

            # Store result
            self.results.append(
                {
                    "parameters": params,
                    "result": {
                        "kappa": result.kappa[0] if result.kappa else None,
                        "t5sigma_s": result.t5sigma_s,
                        "T_sig_K": result.T_sig_K,
                        "T_H_K": result.T_H_K,
                        "hybrid_kappa_eff": result.hybrid_kappa_eff,
                        "hybrid_t5sigma_s": result.hybrid_t5sigma_s,
                    },
                    "objective_value": float(objective_value),
                }
            )

            self.logger.info(f"Evaluated: {params} -> t5sigma={result.t5sigma_s}")

            return objective_value

        except Exception as e:
            self.logger.error(f"Error in objective function: {e}")
            return 1e6  # Large penalty for errors

    # Note: The use_named_args decorator cannot reference `self` at class
    # definition time. We therefore construct a wrapped objective at runtime
    # in `run_optimization()` using `use_named_args(self.space)`.

    def run_optimization(self) -> Dict[str, Any]:
        """Run the Bayesian optimization"""

        self.logger.info("Starting Bayesian optimization...")
        self.logger.info(f"Parameter space: {len(self.space)} dimensions")
        self.logger.info(f"Hybrid mode: {self.use_hybrid}")

        # Run optimization
        # Build named-args wrapper at runtime
        objective = use_named_args(self.space)(self.objective_function)

        result = gp_minimize(
            func=objective,
            dimensions=self.space,
            n_calls=self.config.n_calls,
            n_initial_points=self.config.n_initial_points,
            random_state=self.config.random_state,
            noise=self.config.noise,
            acq_func=self.config.acq_func,
            kappa=self.config.kappa,
            verbose=True,
        )

        # Extract results
        best_params = {}
        for i, (dim, value) in enumerate(zip(self.space, result.x)):
            best_params[dim.name] = float(value)

        self.best_params = best_params
        self.best_value = float(result.fun)

        optimization_results = {
            "best_parameters": best_params,
            "best_objective": self.best_value,
            "best_detection_time": self.best_value,  # Since we minimize detection time
            "optimization_result": {
                "x": [float(x) for x in result.x],
                "fun": float(result.fun),
                "func_vals": [float(f) for f in result.func_vals],
                "x_iters": [[float(x) for x in xi] for xi in result.x_iters],
            },
            "all_results": self.results,
            "config": {
                "n_calls": self.config.n_calls,
                "n_initial_points": self.config.n_initial_points,
                "use_hybrid": self.use_hybrid,
            },
        }

        return optimization_results

    def save_results(self, results: Dict[str, Any], filename: str = None):
        """Save optimization results"""

        if filename is None:
            mode = "hybrid" if self.use_hybrid else "standard"
            filename = f"bayesian_optimization_{mode}.json"

        output_dir = Path("results/optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / filename

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Results saved to {output_file}")

    def plot_convergence(self, results: Dict[str, Any]):
        """Plot optimization convergence"""

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Convergence plot
            func_vals = results["optimization_result"]["func_vals"]
            ax1.plot(func_vals, "b-", linewidth=2)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Detection Time (s)")
            ax1.set_title("Optimization Convergence")
            ax1.set_yscale("log")
            ax1.grid(True, alpha=0.3)

            # Parameter importance (simplified)
            if len(self.results) > 10:
                # Calculate parameter sensitivity
                params = [r["parameters"] for r in self.results[-50:]]  # Last 50 iterations
                values = [r["objective_value"] for r in self.results[-50:]]

                # Simple correlation analysis
                param_names = list(params[0].keys())
                sensitivities = []

                for name in param_names:
                    values_list = [p[name] for p in params]
                    corr = np.corrcoef(values_list, values)[0, 1]
                    sensitivities.append(abs(corr) if not np.isnan(corr) else 0)

                ax2.bar(param_names, sensitivities)
                ax2.set_xlabel("Parameters")
                ax2.set_ylabel("Sensitivity")
                ax2.set_title("Parameter Sensitivity")
                ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()

            output_dir = Path("results/optimization")
            output_dir.mkdir(parents=True, exist_ok=True)

            plt.savefig(
                output_dir
                / f"optimization_convergence_{'hybrid' if self.use_hybrid else 'standard'}.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

        except Exception as e:
            self.logger.error(f"Could not plot convergence: {e}")

    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate optimization report"""

        report = f"""
Bayesian Optimization Report
{'='*50}

Configuration:
- Hybrid Mode: {self.use_hybrid}
- Total Evaluations: {len(results['all_results'])}
- Best Detection Time: {results['best_detection_time']:.2e} seconds

Best Parameters:
"""

        for param, value in results["best_parameters"].items():
            if "intensity" in param:
                report += f"- {param}: {value:.2e} W/m²\n"
            elif "density" in param:
                report += f"- {param}: {value:.2e} m⁻³\n"
            elif "temperature" in param:
                report += f"- {param}: {value:.0f} K\n"
            elif "wavelength" in param:
                report += f"- {param}: {value*1e9:.0f} nm\n"
            elif "grid" in param:
                report += f"- {param}: {value*1e6:.0f} μm\n"
            elif "mirror" in param:
                report += f"- {param}: {value*1e6:.0f} μm\n"
            elif "eta" in param:
                report += f"- {param}: {value:.2f}\n"
            else:
                report += f"- {param}: {value}\n"

        # Calculate derived quantities
        if results["best_parameters"]:
            # Run final calculation with best parameters
            final_params = results["best_parameters"].copy()
            final_params.update(
                {
                    "grid_points": 512,
                    "use_fast_magnetosonic": False,
                    "kappa_method": "acoustic_exact",
                    "graybody": "acoustic_wkb",
                }
            )

            try:
                final_result = run_full_pipeline(**final_params)
                if final_result.kappa and final_result.T_H_K is not None:
                    sig_temp_str = (
                        f"{final_result.T_sig_K:.2e} K"
                        if final_result.T_sig_K is not None
                        else "N/A"
                    )
                    snr_str = (
                        f"{final_result.T_sig_K/30:.2f} (T_sys=30K)"
                        if final_result.T_sig_K is not None
                        else "N/A"
                    )
                    report += (
                        "\nDerived Quantities:\n"
                        f"- Surface Gravity (κ): {final_result.kappa[0]:.2e} s⁻¹\n"
                        f"- Hawking Temperature: {final_result.T_H_K:.2e} K\n"
                        f"- Signal Temperature: {sig_temp_str}\n"
                        f"- Signal-to-Noise Ratio: {snr_str}\n"
                    )
            except Exception as e:
                report += f"\nNote: Could not calculate derived quantities: {e}\n"

        return report


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian optimization for Hawking radiation detection"
    )
    parser.add_argument("--hybrid", action="store_true", help="Use hybrid plasma mirror model")
    parser.add_argument(
        "--n-calls", type=int, default=100, help="Number of optimization iterations"
    )
    parser.add_argument("--n-initial", type=int, default=20, help="Number of initial random points")
    parser.add_argument("--output", help="Output filename")
    parser.add_argument("--plot", action="store_true", help="Generate convergence plots")

    args = parser.parse_args()

    config = OptimizationConfig(n_calls=args.n_calls, n_initial_points=args.n_initial)

    optimizer = HawkingOptimization(config, use_hybrid=args.hybrid)

    # Run optimization
    results = optimizer.run_optimization()

    # Save results
    optimizer.save_results(results, args.output)

    # Generate plots
    if args.plot:
        optimizer.plot_convergence(results)

    # Print report
    report = optimizer.generate_report(results)
    print(report)

    # Save report
    mode = "hybrid" if args.hybrid else "standard"
    report_file = Path("results/optimization") / f"optimization_report_{mode}.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print("\nOptimization complete. Results saved to results/optimization/")


if __name__ == "__main__":
    main()
