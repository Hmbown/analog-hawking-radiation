#!/usr/bin/env python3
"""
Core Validation Framework for Analog Hawking Radiation Experiments

Provides automated validation against known benchmarks, cross-phase consistency
checking, statistical significance validation, and physics model validation.
"""

from __future__ import annotations

import json
import logging

# Add project paths to Python path
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy import stats

# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.analyze_significance import StatisticalAnalyzer
from scripts.convergence_detector import AdvancedConvergenceDetector


@dataclass
class ValidationResult:
    """Result of a validation check"""

    check_name: str
    passed: bool
    confidence: float
    severity: str  # "info", "warning", "error", "critical"
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    recommendation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationSummary:
    """Summary of all validation checks"""

    timestamp: float
    experiment_id: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    overall_confidence: float
    critical_issues: List[str]
    recommendations: List[str]
    results: List[ValidationResult]


class ValidationFramework:
    """Main validation framework with automated checks"""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Optional[Dict[str, Any]] = None

        # Validation thresholds
        self.thresholds = {
            "success_rate_min": 0.3,
            "kappa_min": 1e9,
            "detection_time_max": 1e6,
            "convergence_score_min": 0.6,
            "statistical_significance_min": 0.95,
            "parameter_sensitivity_max": 0.5,
            "cross_phase_correlation_min": 0.7,
            "physical_plausibility_min": 0.8,
        }

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Load experiment data
        self.load_experiment_data()

    def _setup_logging(self) -> None:
        """Setup validation logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_dir / "validation.log"), logging.StreamHandler()],
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
                with open(manifest_file, "r") as f:
                    self.manifest = json.load(f)

            # Load phase results
            for phase_dir in self.experiment_dir.iterdir():
                if phase_dir.is_dir() and phase_dir.name.startswith("phase_"):
                    results_file = phase_dir / "simulation_results.json"
                    if results_file.exists():
                        with open(results_file, "r") as f:
                            self.results[phase_dir.name] = json.load(f)
                        self.logger.info(
                            f"Loaded {len(self.results[phase_dir.name])} results from {phase_dir.name}"
                        )

            return len(self.results) > 0

        except Exception as e:
            self.logger.error(f"Failed to load experiment data: {e}")
            return False

    def run_comprehensive_validation(self) -> ValidationSummary:
        """Run all validation checks"""
        self.logger.info(f"Starting comprehensive validation for experiment {self.experiment_id}")

        validation_results = []

        # Run all validation checks
        validation_results.extend(self.validate_success_rates())
        validation_results.extend(self.validate_convergence())
        validation_results.extend(self.validate_statistical_significance())
        validation_results.extend(self.validate_parameter_sensitivity())
        validation_results.extend(self.validate_cross_phase_consistency())
        validation_results.extend(self.validate_physical_plausibility())
        validation_results.extend(self.validate_data_quality())
        validation_results.extend(self.validate_performance_metrics())

        # Generate summary
        summary = self._generate_summary(validation_results)

        self.logger.info(
            f"Validation completed: {summary.passed_checks}/{summary.total_checks} checks passed"
        )

        return summary

    def validate_success_rates(self) -> List[ValidationResult]:
        """Validate success rates across phases"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="success_rate_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for success rate validation",
                    recommendation="Run experiments first",
                )
            ]

        for phase_name, phase_results in self.results.items():
            successful = sum(1 for r in phase_results if r.get("simulation_success"))
            total = len(phase_results)
            success_rate = successful / total if total > 0 else 0.0

            passed = success_rate >= self.thresholds["success_rate_min"]
            confidence = min(success_rate / self.thresholds["success_rate_min"], 1.0)
            severity = "critical" if success_rate < 0.1 else "warning" if not passed else "info"

            results.append(
                ValidationResult(
                    check_name=f"success_rate_{phase_name}",
                    passed=passed,
                    confidence=confidence,
                    severity=severity,
                    message=f"Success rate for {phase_name}: {success_rate:.1%}",
                    metric_value=success_rate,
                    threshold=self.thresholds["success_rate_min"],
                    recommendation=(
                        "Adjust parameter ranges or improve simulation stability"
                        if not passed
                        else ""
                    ),
                )
            )

        return results

    def validate_convergence(self) -> List[ValidationResult]:
        """Validate convergence across phases"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="convergence_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for convergence validation",
                    recommendation="Run experiments first",
                )
            ]

        # Use the advanced convergence detector
        convergence_detector = AdvancedConvergenceDetector()

        for phase_name, phase_results in self.results.items():
            # Add results to convergence detector
            convergence_detector.add_results(phase_results)

            # Check convergence
            convergence_result = convergence_detector.check_convergence()

            passed = convergence_result.is_converged
            confidence = convergence_result.confidence
            severity = (
                "warning"
                if not passed and convergence_result.convergence_level == "partial"
                else "info"
            )

            results.append(
                ValidationResult(
                    check_name=f"convergence_{phase_name}",
                    passed=passed,
                    confidence=confidence,
                    severity=severity,
                    message=f"Convergence for {phase_name}: {convergence_result.convergence_level}",
                    metric_value=convergence_result.metrics.overall_convergence_score,
                    threshold=self.thresholds["convergence_score_min"],
                    recommendation=(
                        convergence_result.recommendations[0]
                        if convergence_result.recommendations
                        else ""
                    ),
                )
            )

        return results

    def validate_statistical_significance(self) -> List[ValidationResult]:
        """Validate statistical significance of results"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="statistical_significance_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for statistical significance validation",
                    recommendation="Run experiments first",
                )
            ]

        # Collect all successful results
        all_successful_results = []
        for phase_results in self.results.values():
            all_successful_results.extend([r for r in phase_results if r.get("simulation_success")])

        if not all_successful_results:
            return [
                ValidationResult(
                    check_name="statistical_significance_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No successful simulations for statistical significance analysis",
                    recommendation="Improve simulation success rate",
                )
            ]

        # Use existing statistical analyzer
        analyzer = StatisticalAnalyzer()
        enhanced_results = analyzer.calculate_signal_to_noise(all_successful_results)

        # Calculate detection probabilities
        detection_stats_1h = analyzer.calculate_detection_probability(enhanced_results, 3600)
        detection_stats_1d = analyzer.calculate_detection_probability(enhanced_results, 86400)

        # Check 5-sigma detection probability for 1-day observation
        prob_5sigma_1d = detection_stats_1d.get("detection_probability_5sigma", 0.0)
        passed = prob_5sigma_1d >= self.thresholds["statistical_significance_min"]
        confidence = prob_5sigma_1d

        results.append(
            ValidationResult(
                check_name="statistical_significance_5sigma_1d",
                passed=passed,
                confidence=confidence,
                severity="critical" if not passed else "info",
                message=f"5σ detection probability (1 day): {prob_5sigma_1d:.1%}",
                metric_value=prob_5sigma_1d,
                threshold=self.thresholds["statistical_significance_min"],
                recommendation=(
                    "Optimize parameters for better detection statistics" if not passed else ""
                ),
            )
        )

        return results

    def validate_parameter_sensitivity(self) -> List[ValidationResult]:
        """Validate parameter sensitivity analysis"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="parameter_sensitivity_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for parameter sensitivity validation",
                    recommendation="Run experiments first",
                )
            ]

        # Simple parameter sensitivity analysis
        all_results = []
        for phase_results in self.results.values():
            all_results.extend(
                [r for r in phase_results if r.get("simulation_success") and r.get("t5sigma_s")]
            )

        if len(all_results) < 10:
            return [
                ValidationResult(
                    check_name="parameter_sensitivity_validation",
                    passed=False,
                    confidence=0.0,
                    severity="warning",
                    message="Insufficient data for parameter sensitivity analysis",
                    recommendation="Collect more simulation results",
                )
            ]

        # Calculate correlations between parameters and detection time
        parameters_data = []
        detection_times = []

        for result in all_results:
            params = result.get("parameters_used", {})
            detection_time = result.get("t5sigma_s")

            if detection_time is None:
                continue

            # Extract key parameters
            param_vector = [
                np.log10(params.get("laser_intensity", 1e17)),
                np.log10(params.get("plasma_density", 5e17)),
                params.get("temperature_constant", 1e4),
                params.get("magnetic_field", 0.0),
            ]

            parameters_data.append(param_vector)
            detection_times.append(np.log10(detection_time))

        if len(parameters_data) < 10:
            return results

        # Calculate sensitivity scores
        param_names = ["log_intensity", "log_density", "temperature", "magnetic_field"]
        sensitivities = {}

        for i, param_name in enumerate(param_names):
            param_values = [vec[i] for vec in parameters_data]
            if len(set(param_values)) > 1:
                try:
                    corr_coef, _ = stats.pearsonr(param_values, detection_times)
                    sensitivities[param_name] = abs(corr_coef)
                except:
                    sensitivities[param_name] = 0.0

        # Check if any parameter has excessive sensitivity
        max_sensitivity = max(sensitivities.values()) if sensitivities else 0.0
        passed = max_sensitivity <= self.thresholds["parameter_sensitivity_max"]
        confidence = 1.0 - min(max_sensitivity / self.thresholds["parameter_sensitivity_max"], 1.0)

        results.append(
            ValidationResult(
                check_name="parameter_sensitivity_max",
                passed=passed,
                confidence=confidence,
                severity="warning" if not passed else "info",
                message=f"Maximum parameter sensitivity: {max_sensitivity:.3f}",
                metric_value=max_sensitivity,
                threshold=self.thresholds["parameter_sensitivity_max"],
                recommendation="Investigate parameter dependencies" if not passed else "",
            )
        )

        return results

    def validate_cross_phase_consistency(self) -> List[ValidationResult]:
        """Validate consistency across experiment phases"""
        results = []

        if len(self.results) < 2:
            return [
                ValidationResult(
                    check_name="cross_phase_consistency_validation",
                    passed=True,
                    confidence=1.0,
                    severity="info",
                    message="Single phase experiment - cross-phase consistency not applicable",
                    recommendation="",
                )
            ]

        phase_names = list(self.results.keys())

        # Compare adjacent phases
        for i in range(len(phase_names) - 1):
            phase1 = phase_names[i]
            phase2 = phase_names[i + 1]

            # Extract metrics from both phases
            metrics1 = self._extract_phase_metrics(phase1)
            metrics2 = self._extract_phase_metrics(phase2)

            if not metrics1 or not metrics2:
                continue

            # Calculate consistency score
            consistency_scores = []
            for metric in ["detection_time", "kappa"]:
                if metric in metrics1 and metric in metrics2:
                    values1 = metrics1[metric]
                    values2 = metrics2[metric]

                    if len(values1) > 5 and len(values2) > 5:
                        try:
                            # Use Kolmogorov-Smirnov test for distribution similarity
                            ks_stat, p_value = stats.ks_2samp(values1, values2)
                            consistency = 1.0 - ks_stat  # Higher is better
                            consistency_scores.append(consistency)
                        except:
                            continue

            if consistency_scores:
                avg_consistency = np.mean(consistency_scores)
                passed = avg_consistency >= self.thresholds["cross_phase_correlation_min"]
                confidence = avg_consistency

                results.append(
                    ValidationResult(
                        check_name=f"cross_phase_consistency_{phase1}_{phase2}",
                        passed=passed,
                        confidence=confidence,
                        severity="warning" if not passed else "info",
                        message=f"Cross-phase consistency ({phase1} ↔ {phase2}): {avg_consistency:.3f}",
                        metric_value=avg_consistency,
                        threshold=self.thresholds["cross_phase_correlation_min"],
                        recommendation=(
                            "Investigate phase transition consistency" if not passed else ""
                        ),
                    )
                )

        return results

    def validate_physical_plausibility(self) -> List[ValidationResult]:
        """Validate physical plausibility of results"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="physical_plausibility_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for physical plausibility validation",
                    recommendation="Run experiments first",
                )
            ]

        # Collect all kappa values
        all_kappas = []
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        all_kappas.extend(kappa_list)

        if not all_kappas:
            return [
                ValidationResult(
                    check_name="physical_plausibility_validation",
                    passed=False,
                    confidence=0.0,
                    severity="warning",
                    message="No kappa values available for physical plausibility check",
                    recommendation="Check horizon detection in simulations",
                )
            ]

        # Check if kappa values are in physically plausible range
        # Typical analog Hawking radiation: 1e9 - 1e12 s⁻¹
        plausible_kappas = [k for k in all_kappas if 1e9 <= k <= 1e12]
        plausibility_ratio = len(plausible_kappas) / len(all_kappas) if all_kappas else 0.0

        passed = plausibility_ratio >= self.thresholds["physical_plausibility_min"]
        confidence = plausibility_ratio

        results.append(
            ValidationResult(
                check_name="kappa_physical_plausibility",
                passed=passed,
                confidence=confidence,
                severity="critical" if not passed else "info",
                message=f"Physically plausible kappa values: {plausibility_ratio:.1%}",
                metric_value=plausibility_ratio,
                threshold=self.thresholds["physical_plausibility_min"],
                recommendation=(
                    "Review simulation parameters for physical consistency" if not passed else ""
                ),
            )
        )

        return results

    def validate_data_quality(self) -> List[ValidationResult]:
        """Validate data quality and completeness"""
        results = []

        if not self.results:
            return [
                ValidationResult(
                    check_name="data_quality_validation",
                    passed=False,
                    confidence=0.0,
                    severity="critical",
                    message="No experiment data available for data quality validation",
                    recommendation="Run experiments first",
                )
            ]

        # Check for missing or invalid data
        total_simulations = sum(len(results) for results in self.results.values())
        valid_simulations = 0

        for phase_name, phase_results in self.results.items():
            valid_in_phase = 0
            for result in phase_results:
                if (
                    result.get("simulation_success")
                    and result.get("t5sigma_s") is not None
                    and result.get("kappa") is not None
                ):
                    valid_in_phase += 1

            data_quality = valid_in_phase / len(phase_results) if phase_results else 0.0
            valid_simulations += valid_in_phase

            results.append(
                ValidationResult(
                    check_name=f"data_quality_{phase_name}",
                    passed=data_quality >= 0.8,
                    confidence=data_quality,
                    severity="warning" if data_quality < 0.8 else "info",
                    message=f"Data quality for {phase_name}: {data_quality:.1%}",
                    metric_value=data_quality,
                    threshold=0.8,
                    recommendation=(
                        "Check simulation outputs for completeness" if data_quality < 0.8 else ""
                    ),
                )
            )

        # Overall data quality
        overall_quality = valid_simulations / total_simulations if total_simulations > 0 else 0.0

        results.append(
            ValidationResult(
                check_name="overall_data_quality",
                passed=overall_quality >= 0.8,
                confidence=overall_quality,
                severity="warning" if overall_quality < 0.8 else "info",
                message=f"Overall data quality: {overall_quality:.1%}",
                metric_value=overall_quality,
                threshold=0.8,
                recommendation="Improve simulation reliability" if overall_quality < 0.8 else "",
            )
        )

        return results

    def validate_performance_metrics(self) -> List[ValidationResult]:
        """Validate performance metrics and efficiency"""
        results = []

        if not self.manifest:
            return [
                ValidationResult(
                    check_name="performance_metrics_validation",
                    passed=False,
                    confidence=0.0,
                    severity="warning",
                    message="No experiment manifest available for performance validation",
                    recommendation="",
                )
            ]

        # Check experiment duration
        start_time = self.manifest.get("start_time")
        end_time = self.manifest.get("end_time", time.time())

        if start_time:
            duration_hours = (end_time - start_time) / 3600.0

            # Reasonable duration: less than 24 hours for typical experiments
            passed = duration_hours <= 24.0
            confidence = max(0.0, 1.0 - duration_hours / 24.0)

            results.append(
                ValidationResult(
                    check_name="experiment_duration",
                    passed=passed,
                    confidence=confidence,
                    severity="warning" if not passed else "info",
                    message=f"Experiment duration: {duration_hours:.1f} hours",
                    metric_value=duration_hours,
                    threshold=24.0,
                    recommendation=(
                        "Optimize parallel processing or reduce phase iterations"
                        if not passed
                        else ""
                    ),
                )
            )

        return results

    def _extract_phase_metrics(self, phase_name: str) -> Dict[str, List[float]]:
        """Extract key metrics from a phase's results"""
        results = self.results.get(phase_name, [])
        metrics = {"detection_time": [], "kappa": []}

        for result in results:
            if result.get("simulation_success"):
                # Detection time
                t5sigma = result.get("t5sigma_s")
                if t5sigma is not None:
                    metrics["detection_time"].append(t5sigma)

                # Kappa
                kappa_list = result.get("kappa", [])
                if kappa_list:
                    metrics["kappa"].append(max(kappa_list))

        return metrics

    def _generate_summary(self, validation_results: List[ValidationResult]) -> ValidationSummary:
        """Generate validation summary from all results"""
        total_checks = len(validation_results)
        passed_checks = sum(1 for r in validation_results if r.passed)
        failed_checks = total_checks - passed_checks

        # Calculate overall confidence (weighted average)
        confidences = [r.confidence for r in validation_results]
        overall_confidence = np.mean(confidences) if confidences else 0.0

        # Collect critical issues and recommendations
        critical_issues = [
            r.message
            for r in validation_results
            if not r.passed and r.severity in ["critical", "error"]
        ]

        recommendations = list(
            set([r.recommendation for r in validation_results if r.recommendation and not r.passed])
        )

        return ValidationSummary(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            total_checks=total_checks,
            passed_checks=passed_checks,
            failed_checks=failed_checks,
            overall_confidence=overall_confidence,
            critical_issues=critical_issues,
            recommendations=recommendations,
            results=validation_results,
        )

    def save_validation_report(
        self, summary: ValidationSummary, output_path: Optional[Path] = None
    ) -> None:
        """Save validation report to disk"""
        if not output_path:
            output_path = self.experiment_dir / "validation_report.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "validation_summary": asdict(summary),
            "thresholds_used": self.thresholds,
            "validation_timestamp": summary.timestamp,
        }

        with open(output_path, "w") as f:
            json.dump(report_data, f, indent=2)

        self.logger.info(f"Saved validation report to {output_path}")

    def generate_validation_report_text(self, summary: ValidationSummary) -> str:
        """Generate human-readable validation report"""
        report = f"VALIDATION REPORT - Experiment {self.experiment_id}\n"
        report += "=" * 60 + "\n\n"

        report += f"Validation Timestamp: {datetime.fromtimestamp(summary.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Overall Result: {summary.passed_checks}/{summary.total_checks} checks passed\n"
        report += f"Overall Confidence: {summary.overall_confidence:.3f}\n\n"

        report += "CRITICAL ISSUES\n"
        report += "-" * 30 + "\n"
        if summary.critical_issues:
            for issue in summary.critical_issues:
                report += f"• {issue}\n"
        else:
            report += "No critical issues found\n"

        report += "\nRECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        if summary.recommendations:
            for rec in summary.recommendations:
                report += f"• {rec}\n"
        else:
            report += "No recommendations\n"

        report += "\nDETAILED RESULTS\n"
        report += "-" * 30 + "\n"
        for result in summary.results:
            status = "PASS" if result.passed else "FAIL"
            report += f"{result.check_name}: {status} (confidence: {result.confidence:.3f})\n"
            report += f"  {result.message}\n"
            if not result.passed and result.recommendation:
                report += f"  Recommendation: {result.recommendation}\n"
            report += "\n"

        return report


def main():
    """Main entry point for validation framework"""
    import argparse

    parser = argparse.ArgumentParser(description="Validation Framework")
    parser.add_argument("experiment_id", help="Experiment ID to validate")
    parser.add_argument("--output", help="Output file for validation report")
    parser.add_argument("--text", action="store_true", help="Generate text report instead of JSON")

    args = parser.parse_args()

    # Run validation
    validator = ValidationFramework(args.experiment_id)
    summary = validator.run_comprehensive_validation()

    # Generate report
    if args.text:
        report_text = validator.generate_validation_report_text(summary)
        print(report_text)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report_text)
            print(f"Text report saved to {args.output}")
    else:
        validator.save_validation_report(summary, Path(args.output) if args.output else None)
        print(f"Validation completed: {summary.passed_checks}/{summary.total_checks} checks passed")
        print(f"Overall confidence: {summary.overall_confidence:.3f}")


if __name__ == "__main__":
    main()
