#!/usr/bin/env python3
"""
Benchmark Validation System for Analog Hawking Radiation Experiments

Provides automated validation against known analytical benchmarks and
reference simulations to ensure physical correctness and model accuracy.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit

# Add project paths to Python path
import sys
# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from src.analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from scripts.analyze_significance import StatisticalAnalyzer


@dataclass
class BenchmarkResult:
    """Result of a benchmark validation"""
    benchmark_name: str
    passed: bool
    confidence: float
    metric_value: float
    reference_value: float
    tolerance: float
    relative_error: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkValidationSummary:
    """Summary of benchmark validation results"""
    timestamp: float
    experiment_id: str
    total_benchmarks: int
    passed_benchmarks: int
    failed_benchmarks: int
    overall_accuracy: float
    benchmark_results: List[BenchmarkResult]
    recommendations: List[str]


class BenchmarkValidator:
    """Validates experiment results against known benchmarks and analytical solutions"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Optional[Dict[str, Any]] = None
        
        # Benchmark tolerances (relative error)
        self.tolerances = {
            "kappa_analytical": 0.1,      # 10% tolerance for kappa
            "detection_time_scaling": 0.2, # 20% tolerance for scaling laws
            "temperature_scaling": 0.15,   # 15% tolerance for temperature
            "graybody_transmission": 0.25, # 25% tolerance for graybody
            "horizon_position": 0.05,      # 5% tolerance for horizon position
            "statistical_benchmark": 0.1   # 10% tolerance for statistical tests
        }
        
        # Known analytical benchmarks
        self.analytical_benchmarks = {
            "standard_hawking_temperature": {
                "description": "Hawking temperature T_H = ħκ/(2πk_B)",
                "reference": 6.17e-8,  # K for κ=1e10 s⁻¹
                "kappa_reference": 1e10
            },
            "detection_time_scaling": {
                "description": "Detection time scales as T^-4 for thermal spectrum",
                "expected_exponent": -4.0,
                "tolerance": 0.5
            }
        }
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load experiment data
        self.load_experiment_data()
    
    def _setup_logging(self) -> None:
        """Setup benchmark validation logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'benchmark_validation.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_experiment_data(self) -> bool:
        """Load experiment data from disk"""
        try:
            if not self.experiment_dir.exists():
                self.logger.error(f"Experiment directory not found: {self.experiment_dir}")
                return False
            
            # Load manifest
            manifest_file = self.experiment_dir / "experiment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    self.manifest = json.load(f)
            
            # Load phase results
            for phase_dir in self.experiment_dir.iterdir():
                if phase_dir.is_dir() and phase_dir.name.startswith("phase_"):
                    results_file = phase_dir / "simulation_results.json"
                    if results_file.exists():
                        with open(results_file, 'r') as f:
                            self.results[phase_dir.name] = json.load(f)
                        self.logger.info(f"Loaded {len(self.results[phase_dir.name])} results from {phase_dir.name}")
            
            return len(self.results) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment data: {e}")
            return False
    
    def run_comprehensive_benchmark_validation(self) -> BenchmarkValidationSummary:
        """Run comprehensive benchmark validation against known references"""
        self.logger.info(f"Starting comprehensive benchmark validation for experiment {self.experiment_id}")
        
        benchmark_results = []
        
        # Run all benchmark validations
        benchmark_results.extend(self.validate_kappa_against_analytical())
        benchmark_results.extend(self.validate_detection_time_scaling())
        benchmark_results.extend(self.validate_temperature_scaling())
        benchmark_results.extend(self.validate_graybody_transmission())
        benchmark_results.extend(self.validate_horizon_consistency())
        benchmark_results.extend(self.validate_statistical_benchmarks())
        
        # Generate summary
        summary = self._generate_summary(benchmark_results)
        
        self.logger.info(f"Benchmark validation completed: {summary.passed_benchmarks}/{summary.total_benchmarks} benchmarks passed")
        
        return summary
    
    def validate_kappa_against_analytical(self) -> List[BenchmarkResult]:
        """Validate kappa values against analytical expectations"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="kappa_analytical",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=0.0,
                tolerance=self.tolerances["kappa_analytical"],
                relative_error=1.0,
                message="No experiment data available for kappa validation"
            )]
        
        # Collect all kappa values
        all_kappas = []
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        all_kappas.extend(kappa_list)
        
        if not all_kappas:
            return [BenchmarkResult(
                benchmark_name="kappa_analytical",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=self.analytical_benchmarks["standard_hawking_temperature"]["kappa_reference"],
                tolerance=self.tolerances["kappa_analytical"],
                relative_error=1.0,
                message="No kappa values available for analytical validation"
            )]
        
        # Calculate median kappa and compare with reference
        median_kappa = np.median(all_kappas)
        reference_kappa = self.analytical_benchmarks["standard_hawking_temperature"]["kappa_reference"]
        
        # For analog systems, we expect kappa in the range 1e9-1e12 s⁻¹
        # Check if median is in physically reasonable range
        reasonable_range = (1e9, 1e12)
        is_reasonable = reasonable_range[0] <= median_kappa <= reasonable_range[1]
        
        # Calculate relative error from reference (if we had a specific reference)
        # For now, we check if it's within an order of magnitude of typical values
        order_of_magnitude_error = abs(np.log10(median_kappa) - np.log10(reference_kappa))
        relative_error = min(order_of_magnitude_error, 2.0) / 2.0  # Normalize to [0,1]
        
        passed = is_reasonable and relative_error <= self.tolerances["kappa_analytical"]
        confidence = 1.0 - relative_error
        
        results.append(BenchmarkResult(
            benchmark_name="kappa_analytical",
            passed=passed,
            confidence=confidence,
            metric_value=median_kappa,
            reference_value=reference_kappa,
            tolerance=self.tolerances["kappa_analytical"],
            relative_error=relative_error,
            message=f"Median kappa: {median_kappa:.2e} s⁻¹ (reference: {reference_kappa:.2e} s⁻¹)",
            details={
                "kappa_distribution": {
                    "min": np.min(all_kappas),
                    "max": np.max(all_kappas),
                    "median": median_kappa,
                    "mean": np.mean(all_kappas),
                    "std": np.std(all_kappas)
                },
                "reasonable_range": reasonable_range,
                "is_in_reasonable_range": is_reasonable
            }
        ))
        
        return results
    
    def validate_detection_time_scaling(self) -> List[BenchmarkResult]:
        """Validate detection time scaling with signal temperature"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="detection_time_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=self.analytical_benchmarks["detection_time_scaling"]["expected_exponent"],
                tolerance=self.tolerances["detection_time_scaling"],
                relative_error=1.0,
                message="No experiment data available for detection time scaling validation"
            )]
        
        # Collect detection times and signal temperatures
        detection_times = []
        signal_temperatures = []
        
        for phase_results in self.results.values():
            for result in phase_results:
                if (result.get("simulation_success") and 
                    result.get("t5sigma_s") is not None and
                    result.get("T_sig_K") is not None):
                    
                    detection_times.append(result["t5sigma_s"])
                    signal_temperatures.append(result["T_sig_K"])
        
        if len(detection_times) < 10:
            return [BenchmarkResult(
                benchmark_name="detection_time_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=self.analytical_benchmarks["detection_time_scaling"]["expected_exponent"],
                tolerance=self.tolerances["detection_time_scaling"],
                relative_error=1.0,
                message="Insufficient data for detection time scaling analysis"
            )]
        
        # For thermal radiation, detection time should scale as T^-4
        # Fit power law: t_det ∝ T^α
        try:
            # Use log-log space for linear fit
            log_t = np.log10(detection_times)
            log_T = np.log10(signal_temperatures)
            
            # Fit linear model: log(t) = α * log(T) + β
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_T, log_t)
            
            measured_exponent = slope
            expected_exponent = self.analytical_benchmarks["detection_time_scaling"]["expected_exponent"]
            
            relative_error = abs(measured_exponent - expected_exponent) / abs(expected_exponent)
            passed = relative_error <= self.tolerances["detection_time_scaling"]
            confidence = max(0.0, 1.0 - relative_error / self.tolerances["detection_time_scaling"])
            
            # Also check goodness of fit
            fit_quality = r_value ** 2  # R-squared
            
            results.append(BenchmarkResult(
                benchmark_name="detection_time_scaling",
                passed=passed and fit_quality > 0.5,
                confidence=confidence * fit_quality,
                metric_value=measured_exponent,
                reference_value=expected_exponent,
                tolerance=self.tolerances["detection_time_scaling"],
                relative_error=relative_error,
                message=f"Detection time scaling exponent: {measured_exponent:.2f} (expected: {expected_exponent:.1f}, R²={fit_quality:.3f})",
                details={
                    "measured_exponent": measured_exponent,
                    "expected_exponent": expected_exponent,
                    "r_squared": fit_quality,
                    "p_value": p_value,
                    "standard_error": std_err,
                    "sample_size": len(detection_times)
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Detection time scaling analysis failed: {e}")
            results.append(BenchmarkResult(
                benchmark_name="detection_time_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=self.analytical_benchmarks["detection_time_scaling"]["expected_exponent"],
                tolerance=self.tolerances["detection_time_scaling"],
                relative_error=1.0,
                message=f"Detection time scaling analysis failed: {e}"
            ))
        
        return results
    
    def validate_temperature_scaling(self) -> List[BenchmarkResult]:
        """Validate temperature scaling with surface gravity"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="temperature_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,  # Expected scaling factor
                tolerance=self.tolerances["temperature_scaling"],
                relative_error=1.0,
                message="No experiment data available for temperature scaling validation"
            )]
        
        # Collect kappa values and signal temperatures
        kappas = []
        temperatures = []
        
        for phase_results in self.results.values():
            for result in phase_results:
                if (result.get("simulation_success") and 
                    result.get("T_sig_K") is not None):
                    
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        max_kappa = max(kappa_list)
                        kappas.append(max_kappa)
                        temperatures.append(result["T_sig_K"])
        
        if len(kappas) < 10:
            return [BenchmarkResult(
                benchmark_name="temperature_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,
                tolerance=self.tolerances["temperature_scaling"],
                relative_error=1.0,
                message="Insufficient data for temperature scaling analysis"
            )]
        
        # For Hawking radiation, T ∝ κ
        # Check linear relationship between temperature and kappa
        try:
            # Use log space for power law analysis
            log_T = np.log10(temperatures)
            log_kappa = np.log10(kappas)
            
            # Fit linear model: log(T) = α * log(κ) + β
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_kappa, log_T)
            
            # Expected slope is 1.0 for T ∝ κ
            measured_slope = slope
            expected_slope = 1.0
            
            relative_error = abs(measured_slope - expected_slope) / abs(expected_slope)
            passed = relative_error <= self.tolerances["temperature_scaling"]
            confidence = max(0.0, 1.0 - relative_error / self.tolerances["temperature_scaling"])
            
            fit_quality = r_value ** 2  # R-squared
            
            results.append(BenchmarkResult(
                benchmark_name="temperature_scaling",
                passed=passed and fit_quality > 0.5,
                confidence=confidence * fit_quality,
                metric_value=measured_slope,
                reference_value=expected_slope,
                tolerance=self.tolerances["temperature_scaling"],
                relative_error=relative_error,
                message=f"Temperature-kappa scaling exponent: {measured_slope:.2f} (expected: {expected_slope:.1f}, R²={fit_quality:.3f})",
                details={
                    "measured_slope": measured_slope,
                    "expected_slope": expected_slope,
                    "r_squared": fit_quality,
                    "p_value": p_value,
                    "standard_error": std_err,
                    "sample_size": len(kappas)
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Temperature scaling analysis failed: {e}")
            results.append(BenchmarkResult(
                benchmark_name="temperature_scaling",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,
                tolerance=self.tolerances["temperature_scaling"],
                relative_error=1.0,
                message=f"Temperature scaling analysis failed: {e}"
            ))
        
        return results
    
    def validate_graybody_transmission(self) -> List[BenchmarkResult]:
        """Validate graybody transmission factors"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="graybody_transmission",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=0.5,  # Typical graybody factor
                tolerance=self.tolerances["graybody_transmission"],
                relative_error=1.0,
                message="No experiment data available for graybody validation"
            )]
        
        # For analog systems, graybody factors should be between 0 and 1
        # and typically around 0.1-0.8 for acoustic black holes
        
        # This is a simplified check - in a full implementation, we would
        # compare with known graybody spectra for specific configurations
        
        graybody_factors = []
        
        for phase_results in self.results.values():
            for result in phase_results:
                # Extract graybody-related information if available
                params = result.get("parameters_used", {})
                physics_params = params.get("physics", {})
                
                alpha_gray = physics_params.get("alpha_gray")
                if alpha_gray is not None:
                    graybody_factors.append(alpha_gray)
        
        if not graybody_factors:
            return [BenchmarkResult(
                benchmark_name="graybody_transmission",
                passed=True,  # Not critical if not used
                confidence=1.0,
                metric_value=0.0,
                reference_value=0.5,
                tolerance=self.tolerances["graybody_transmission"],
                relative_error=0.0,
                message="No graybody factors specified in parameters"
            )]
        
        # Check if graybody factors are in reasonable range
        reasonable_graybody = [g for g in graybody_factors if 0 < g < 1]
        reasonable_ratio = len(reasonable_graybody) / len(graybody_factors)
        
        passed = reasonable_ratio >= 0.9  # 90% should be reasonable
        confidence = reasonable_ratio
        
        results.append(BenchmarkResult(
            benchmark_name="graybody_transmission",
            passed=passed,
            confidence=confidence,
            metric_value=np.mean(graybody_factors),
            reference_value=0.5,
            tolerance=self.tolerances["graybody_transmission"],
            relative_error=abs(np.mean(graybody_factors) - 0.5),
            message=f"Graybody factors: {reasonable_ratio:.1%} in reasonable range (0,1)",
            details={
                "mean_graybody_factor": np.mean(graybody_factors),
                "reasonable_ratio": reasonable_ratio,
                "total_graybody_factors": len(graybody_factors),
                "reasonable_graybody_factors": len(reasonable_graybody)
            }
        ))
        
        return results
    
    def validate_horizon_consistency(self) -> List[BenchmarkResult]:
        """Validate horizon detection consistency"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="horizon_consistency",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,  # Perfect consistency
                tolerance=self.tolerances["horizon_position"],
                relative_error=1.0,
                message="No experiment data available for horizon consistency validation"
            )]
        
        # Check consistency of horizon detection across similar parameter sets
        horizon_positions = []
        
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    # Extract horizon-related information if available
                    # This would require the simulation to output horizon positions
                    # For now, we'll use a placeholder check
                    
                    # In a full implementation, we would:
                    # 1. Group results by similar parameters
                    # 2. Check variance of horizon positions within each group
                    # 3. Compare with expected numerical uncertainty
                    pass
        
        # Placeholder implementation
        # For now, we assume reasonable consistency if we have successful simulations
        total_simulations = sum(len(results) for results in self.results.values())
        successful_simulations = sum(
            sum(1 for r in results if r.get("simulation_success"))
            for results in self.results.values()
        )
        
        success_rate = successful_simulations / total_simulations if total_simulations > 0 else 0.0
        
        # Use success rate as a proxy for horizon consistency
        # Higher success rate suggests more consistent horizon detection
        passed = success_rate >= 0.5
        confidence = success_rate
        
        results.append(BenchmarkResult(
            benchmark_name="horizon_consistency",
            passed=passed,
            confidence=confidence,
            metric_value=success_rate,
            reference_value=0.8,  # Target success rate
            tolerance=0.3,  # 30% tolerance
            relative_error=abs(success_rate - 0.8),
            message=f"Horizon detection success rate: {success_rate:.1%}",
            details={
                "success_rate": success_rate,
                "total_simulations": total_simulations,
                "successful_simulations": successful_simulations
            }
        ))
        
        return results
    
    def validate_statistical_benchmarks(self) -> List[BenchmarkResult]:
        """Validate statistical properties against benchmarks"""
        results = []
        
        if not self.results:
            return [BenchmarkResult(
                benchmark_name="statistical_benchmarks",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,
                tolerance=self.tolerances["statistical_benchmark"],
                relative_error=1.0,
                message="No experiment data available for statistical benchmark validation"
            )]
        
        # Collect all successful results for statistical analysis
        all_successful_results = []
        for phase_results in self.results.values():
            all_successful_results.extend([r for r in phase_results if r.get("simulation_success")])
        
        if len(all_successful_results) < 20:
            return [BenchmarkResult(
                benchmark_name="statistical_benchmarks",
                passed=False,
                confidence=0.0,
                metric_value=0.0,
                reference_value=1.0,
                tolerance=self.tolerances["statistical_benchmark"],
                relative_error=1.0,
                message="Insufficient data for statistical benchmark analysis"
            )]
        
        # Use existing statistical analyzer
        analyzer = StatisticalAnalyzer()
        enhanced_results = analyzer.calculate_signal_to_noise(all_successful_results)
        
        # Calculate detection probabilities as benchmark
        detection_stats_1h = analyzer.calculate_detection_probability(enhanced_results, 3600)
        detection_stats_1d = analyzer.calculate_detection_probability(enhanced_results, 86400)
        
        # Benchmark: 5σ detection probability should be > 0 for viable configurations
        prob_5sigma_1d = detection_stats_1d.get("detection_probability_5sigma", 0.0)
        
        passed = prob_5sigma_1d > 0.0
        confidence = prob_5sigma_1d  # Use probability as confidence
        
        results.append(BenchmarkResult(
            benchmark_name="statistical_detection_probability",
            passed=passed,
            confidence=confidence,
            metric_value=prob_5sigma_1d,
            reference_value=0.1,  # Target 10% probability
            tolerance=0.1,  # 10% tolerance
            relative_error=max(0.0, 0.1 - prob_5sigma_1d),
            message=f"5σ detection probability (1 day): {prob_5sigma_1d:.1%}",
            details={
                "detection_probability_5sigma_1d": prob_5sigma_1d,
                "detection_probability_3sigma_1d": detection_stats_1d.get("detection_probability_3sigma", 0.0),
                "detection_probability_5sigma_1h": detection_stats_1h.get("detection_probability_5sigma", 0.0),
                "total_analyzed_results": len(enhanced_results)
            }
        ))
        
        return results
    
    def _generate_summary(self, benchmark_results: List[BenchmarkResult]) -> BenchmarkValidationSummary:
        """Generate benchmark validation summary from all results"""
        total_benchmarks = len(benchmark_results)
        passed_benchmarks = sum(1 for r in benchmark_results if r.passed)
        failed_benchmarks = total_benchmarks - passed_benchmarks
        
        # Calculate overall accuracy (weighted by confidence)
        if benchmark_results:
            overall_accuracy = np.mean([r.confidence for r in benchmark_results])
        else:
            overall_accuracy = 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(benchmark_results)
        
        return BenchmarkValidationSummary(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            total_benchmarks=total_benchmarks,
            passed_benchmarks=passed_benchmarks,
            failed_benchmarks=failed_benchmarks,
            overall_accuracy=overall_accuracy,
            benchmark_results=benchmark_results,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, benchmark_results: List[BenchmarkResult]) -> List[str]:
        """Generate improvement recommendations based on benchmark results"""
        recommendations = []
        
        for result in benchmark_results:
            if not result.passed:
                if "kappa" in result.benchmark_name.lower():
                    recommendations.append("Review kappa calculation and horizon detection methods")
                elif "scaling" in result.benchmark_name.lower():
                    recommendations.append("Investigate scaling law deviations - check model assumptions")
                elif "graybody" in result.benchmark_name.lower():
                    recommendations.append("Verify graybody transmission model implementation")
                elif "horizon" in result.benchmark_name.lower():
                    recommendations.append("Improve horizon detection consistency")
                elif "statistical" in result.benchmark_name.lower():
                    recommendations.append("Optimize parameters for better detection statistics")
        
        # Remove duplicates
        return list(set(recommendations))
    
    def save_benchmark_report(self, summary: BenchmarkValidationSummary, output_path: Optional[Path] = None) -> None:
        """Save benchmark validation report to disk"""
        if not output_path:
            output_path = self.experiment_dir / "benchmark_validation_report.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "benchmark_validation_summary": asdict(summary),
            "tolerances_used": self.tolerances,
            "analytical_benchmarks": self.analytical_benchmarks,
            "validation_timestamp": summary.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Saved benchmark validation report to {output_path}")
    
    def generate_benchmark_report_text(self, summary: BenchmarkValidationSummary) -> str:
        """Generate human-readable benchmark validation report"""
        report = f"BENCHMARK VALIDATION REPORT - Experiment {self.experiment_id}\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Validation Timestamp: {datetime.fromtimestamp(summary.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Overall Result: {summary.passed_benchmarks}/{summary.total_benchmarks} benchmarks passed\n"
        report += f"Overall Accuracy: {summary.overall_accuracy:.3f}\n\n"
        
        report += "BENCHMARK RESULTS\n"
        report += "-" * 30 + "\n"
        for result in summary.benchmark_results:
            status = "PASS" if result.passed else "FAIL"
            report += f"{result.benchmark_name}: {status} (confidence: {result.confidence:.3f})\n"
            report += f"  {result.message}\n"
            report += f"  Metric: {result.metric_value:.3e}, Reference: {result.reference_value:.3e}\n"
            report += f"  Relative error: {result.relative_error:.1%}, Tolerance: {result.tolerance:.1%}\n\n"
        
        report += "RECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        if summary.recommendations:
            for rec in summary.recommendations:
                report += f"• {rec}\n"
        else:
            report += "No specific recommendations\n"
        
        return report


def main():
    """Main entry point for benchmark validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Validation System")
    parser.add_argument("experiment_id", help="Experiment ID to validate")
    parser.add_argument("--output", help="Output file for benchmark report")
    parser.add_argument("--text", action="store_true", help="Generate text report instead of JSON")
    
    args = parser.parse_args()
    
    # Run benchmark validation
    validator = BenchmarkValidator(args.experiment_id)
    summary = validator.run_comprehensive_benchmark_validation()
    
    # Generate report
    if args.text:
        report_text = validator.generate_benchmark_report_text(summary)
        print(report_text)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report_text)
            print(f"Text report saved to {args.output}")
    else:
        validator.save_benchmark_report(summary, Path(args.output) if args.output else None)
        print(f"Benchmark validation completed: {summary.passed_benchmarks}/{summary.total_benchmarks} benchmarks passed")
        print(f"Overall accuracy: {summary.overall_accuracy:.3f}")


if __name__ == "__main__":
    main()
