#!/usr/bin/env python3
"""
Quality Assurance System for Analog Hawking Radiation Experiments

Provides automated data quality checks, anomaly detection, parameter boundary
validation, convergence validation, and reproducibility verification.
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
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.convergence_detector import AdvancedConvergenceDetector


@dataclass
class QualityMetric:
    """Quality metric with score and interpretation"""
    name: str
    score: float
    threshold: float
    passed: bool
    weight: float = 1.0
    description: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection analysis"""
    total_samples: int
    anomalies_detected: int
    anomaly_indices: List[int]
    anomaly_scores: List[float]
    contamination_estimate: float
    features_used: List[str]


@dataclass
class QualityAssuranceReport:
    """Comprehensive quality assurance report"""
    timestamp: float
    experiment_id: str
    overall_quality_score: float
    quality_metrics: List[QualityMetric]
    anomalies: AnomalyDetectionResult
    parameter_boundary_violations: List[str]
    convergence_quality: Dict[str, float]
    reproducibility_score: float
    recommendations: List[str]


class QualityAssuranceSystem:
    """Comprehensive quality assurance system with automated checks"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Optional[Dict[str, Any]] = None
        
        # Quality thresholds
        self.thresholds = {
            "data_completeness_min": 0.8,
            "parameter_boundary_violations_max": 0.05,
            "anomaly_contamination_max": 0.1,
            "convergence_quality_min": 0.7,
            "reproducibility_min": 0.9,
            "statistical_consistency_min": 0.8,
            "physical_plausibility_min": 0.8
        }
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load experiment data
        self.load_experiment_data()
    
    def _setup_logging(self) -> None:
        """Setup quality assurance logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'quality_assurance.log'),
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
    
    def run_comprehensive_quality_assurance(self) -> QualityAssuranceReport:
        """Run comprehensive quality assurance analysis"""
        self.logger.info(f"Starting comprehensive quality assurance for experiment {self.experiment_id}")
        
        quality_metrics = []
        
        # Run all quality checks
        quality_metrics.append(self.check_data_completeness())
        quality_metrics.append(self.check_parameter_boundaries())
        quality_metrics.append(self.check_statistical_consistency())
        quality_metrics.append(self.check_physical_plausibility())
        
        # Run advanced analyses
        anomalies = self.detect_anomalies()
        convergence_quality = self.assess_convergence_quality()
        reproducibility_score = self.assess_reproducibility()
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(quality_metrics)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_metrics, anomalies, overall_quality_score)
        
        report = QualityAssuranceReport(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            overall_quality_score=overall_quality_score,
            quality_metrics=quality_metrics,
            anomalies=anomalies,
            parameter_boundary_violations=self._get_parameter_boundary_violations(),
            convergence_quality=convergence_quality,
            reproducibility_score=reproducibility_score,
            recommendations=recommendations
        )
        
        self.logger.info(f"Quality assurance completed: overall score = {overall_quality_score:.3f}")
        
        return report
    
    def check_data_completeness(self) -> QualityMetric:
        """Check data completeness across all phases"""
        if not self.results:
            return QualityMetric(
                name="data_completeness",
                score=0.0,
                threshold=self.thresholds["data_completeness_min"],
                passed=False,
                description="No experiment data available"
            )
        
        total_simulations = 0
        complete_simulations = 0
        
        for phase_name, phase_results in self.results.items():
            for result in phase_results:
                total_simulations += 1
                
                # Check if result has all required fields
                required_fields = ["simulation_success", "t5sigma_s", "kappa", "T_sig_K"]
                has_required_fields = all(field in result for field in required_fields)
                
                # Check if values are valid
                has_valid_values = (
                    result.get("simulation_success") is not None and
                    result.get("t5sigma_s") is not None and
                    result.get("kappa") is not None and
                    len(result.get("kappa", [])) > 0 and
                    result.get("T_sig_K") is not None
                )
                
                if has_required_fields and has_valid_values:
                    complete_simulations += 1
        
        completeness_score = complete_simulations / total_simulations if total_simulations > 0 else 0.0
        passed = completeness_score >= self.thresholds["data_completeness_min"]
        
        return QualityMetric(
            name="data_completeness",
            score=completeness_score,
            threshold=self.thresholds["data_completeness_min"],
            passed=passed,
            weight=1.5,  # Higher weight for data completeness
            description=f"Data completeness: {completeness_score:.1%} ({complete_simulations}/{total_simulations} simulations)",
            details={
                "total_simulations": total_simulations,
                "complete_simulations": complete_simulations,
                "missing_data_simulations": total_simulations - complete_simulations
            }
        )
    
    def check_parameter_boundaries(self) -> QualityMetric:
        """Check for parameter boundary violations"""
        if not self.results:
            return QualityMetric(
                name="parameter_boundaries",
                score=0.0,
                threshold=self.thresholds["parameter_boundary_violations_max"],
                passed=False,
                description="No experiment data available"
            )
        
        total_parameters = 0
        boundary_violations = 0
        
        # Define reasonable parameter boundaries
        parameter_bounds = {
            "laser_intensity": (1e15, 1e21),  # W/m²
            "plasma_density": (1e15, 1e21),   # m⁻³
            "temperature_constant": (1e2, 1e6),  # K
            "magnetic_field": (0, 1000),      # T
            "mirror_D": (1e-7, 1e-4),         # m
            "mirror_eta": (0.1, 10.0)         # dimensionless
        }
        
        for phase_results in self.results.values():
            for result in phase_results:
                params = result.get("parameters_used", {})
                
                for param_name, (min_val, max_val) in parameter_bounds.items():
                    if param_name in params:
                        total_parameters += 1
                        value = params[param_name]
                        
                        # Check if value is within bounds
                        if value < min_val or value > max_val:
                            boundary_violations += 1
        
        violation_rate = boundary_violations / total_parameters if total_parameters > 0 else 0.0
        passed = violation_rate <= self.thresholds["parameter_boundary_violations_max"]
        
        # Score is inverse of violation rate (higher is better)
        score = max(0.0, 1.0 - violation_rate / self.thresholds["parameter_boundary_violations_max"])
        
        return QualityMetric(
            name="parameter_boundaries",
            score=score,
            threshold=self.thresholds["parameter_boundary_violations_max"],
            passed=passed,
            weight=1.2,
            description=f"Parameter boundary violations: {violation_rate:.1%} ({boundary_violations}/{total_parameters})",
            details={
                "total_parameters_checked": total_parameters,
                "boundary_violations": boundary_violations,
                "violation_rate": violation_rate
            }
        )
    
    def check_statistical_consistency(self) -> QualityMetric:
        """Check statistical consistency of results"""
        if not self.results:
            return QualityMetric(
                name="statistical_consistency",
                score=0.0,
                threshold=self.thresholds["statistical_consistency_min"],
                passed=False,
                description="No experiment data available"
            )
        
        # Collect detection times from successful simulations
        all_detection_times = []
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success") and result.get("t5sigma_s") is not None:
                    all_detection_times.append(result["t5sigma_s"])
        
        if len(all_detection_times) < 10:
            return QualityMetric(
                name="statistical_consistency",
                score=0.0,
                threshold=self.thresholds["statistical_consistency_min"],
                passed=False,
                description="Insufficient data for statistical consistency analysis"
            )
        
        # Calculate statistical consistency metrics
        detection_times_log = np.log10(all_detection_times)
        
        # Check for normality (log-normal distribution expected)
        try:
            _, p_value_normality = stats.normaltest(detection_times_log)
            normality_metric = min(p_value_normality, 1.0)  # Higher p-value is better
        except:
            normality_metric = 0.0
        
        # Check for outliers using IQR method
        Q1 = np.percentile(detection_times_log, 25)
        Q3 = np.percentile(detection_times_log, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = [x for x in detection_times_log if x < lower_bound or x > upper_bound]
        outlier_ratio = len(outliers) / len(detection_times_log)
        outlier_metric = 1.0 - outlier_ratio
        
        # Calculate coefficient of variation (lower is better for consistency)
        cv = np.std(detection_times_log) / np.mean(detection_times_log) if np.mean(detection_times_log) != 0 else 1.0
        cv_metric = 1.0 - min(cv, 1.0)
        
        # Combined statistical consistency score
        consistency_score = (normality_metric + outlier_metric + cv_metric) / 3.0
        passed = consistency_score >= self.thresholds["statistical_consistency_min"]
        
        return QualityMetric(
            name="statistical_consistency",
            score=consistency_score,
            threshold=self.thresholds["statistical_consistency_min"],
            passed=passed,
            weight=1.0,
            description=f"Statistical consistency: {consistency_score:.3f}",
            details={
                "normality_p_value": normality_metric,
                "outlier_ratio": outlier_ratio,
                "coefficient_of_variation": cv,
                "total_samples": len(all_detection_times)
            }
        )
    
    def check_physical_plausibility(self) -> QualityMetric:
        """Check physical plausibility of results"""
        if not self.results:
            return QualityMetric(
                name="physical_plausibility",
                score=0.0,
                threshold=self.thresholds["physical_plausibility_min"],
                passed=False,
                description="No experiment data available"
            )
        
        # Collect key physical metrics
        all_kappas = []
        all_temperatures = []
        all_detection_times = []
        
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    # Kappa values
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        all_kappas.extend(kappa_list)
                    
                    # Signal temperature
                    T_sig = result.get("T_sig_K")
                    if T_sig is not None:
                        all_temperatures.append(T_sig)
                    
                    # Detection time
                    t5sigma = result.get("t5sigma_s")
                    if t5sigma is not None:
                        all_detection_times.append(t5sigma)
        
        if not all_kappas:
            return QualityMetric(
                name="physical_plausibility",
                score=0.0,
                threshold=self.thresholds["physical_plausibility_min"],
                passed=False,
                description="No kappa values available for physical plausibility check"
            )
        
        # Check kappa values (typical analog Hawking: 1e9 - 1e12 s⁻¹)
        plausible_kappas = [k for k in all_kappas if 1e9 <= k <= 1e12]
        kappa_plausibility = len(plausible_kappas) / len(all_kappas) if all_kappas else 0.0
        
        # Check signal temperatures (reasonable range for radio detection)
        plausible_temperatures = [t for t in all_temperatures if 0.1 <= t <= 1e6]  # 0.1K to 1e6K
        temperature_plausibility = len(plausible_temperatures) / len(all_temperatures) if all_temperatures else 1.0
        
        # Check detection times (reasonable range for observatories)
        plausible_detection_times = [t for t in all_detection_times if 1 <= t <= 1e7]  # 1s to ~115 days
        detection_time_plausibility = len(plausible_detection_times) / len(all_detection_times) if all_detection_times else 1.0
        
        # Combined physical plausibility score
        plausibility_score = (kappa_plausibility + temperature_plausibility + detection_time_plausibility) / 3.0
        passed = plausibility_score >= self.thresholds["physical_plausibility_min"]
        
        return QualityMetric(
            name="physical_plausibility",
            score=plausibility_score,
            threshold=self.thresholds["physical_plausibility_min"],
            passed=passed,
            weight=1.3,
            description=f"Physical plausibility: {plausibility_score:.3f}",
            details={
                "kappa_plausibility": kappa_plausibility,
                "temperature_plausibility": temperature_plausibility,
                "detection_time_plausibility": detection_time_plausibility,
                "total_kappa_samples": len(all_kappas),
                "total_temperature_samples": len(all_temperatures),
                "total_detection_time_samples": len(all_detection_times)
            }
        )
    
    def detect_anomalies(self) -> AnomalyDetectionResult:
        """Detect anomalies in experiment results using isolation forest"""
        if not self.results:
            return AnomalyDetectionResult(
                total_samples=0,
                anomalies_detected=0,
                anomaly_indices=[],
                anomaly_scores=[],
                contamination_estimate=0.0,
                features_used=[]
            )
        
        # Prepare features for anomaly detection
        features = []
        sample_indices = []
        feature_names = []
        
        for phase_name, phase_results in self.results.items():
            for i, result in enumerate(phase_results):
                if result.get("simulation_success"):
                    feature_vector = []
                    
                    # Detection time (log scale)
                    t5sigma = result.get("t5sigma_s")
                    if t5sigma is not None and t5sigma > 0:
                        feature_vector.append(np.log10(t5sigma))
                        if "log_detection_time" not in feature_names:
                            feature_names.append("log_detection_time")
                    
                    # Maximum kappa (log scale)
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        max_kappa = max(kappa_list)
                        feature_vector.append(np.log10(max_kappa))
                        if "log_max_kappa" not in feature_names:
                            feature_names.append("log_max_kappa")
                    
                    # Signal temperature (log scale)
                    T_sig = result.get("T_sig_K")
                    if T_sig is not None and T_sig > 0:
                        feature_vector.append(np.log10(T_sig))
                        if "log_signal_temperature" not in feature_names:
                            feature_names.append("log_signal_temperature")
                    
                    # Only include samples with all features
                    if len(feature_vector) == 3:  # All three features available
                        features.append(feature_vector)
                        sample_indices.append((phase_name, i))
        
        if len(features) < 10:
            return AnomalyDetectionResult(
                total_samples=len(features),
                anomalies_detected=0,
                anomaly_indices=[],
                anomaly_scores=[],
                contamination_estimate=0.0,
                features_used=feature_names
            )
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Detect anomalies using Isolation Forest
        contamination = min(0.1, 5.0 / len(features))  # Adaptive contamination parameter
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(features_scaled)
        anomaly_scores = iso_forest.decision_function(features_scaled)
        
        # Convert labels: -1 for anomalies, 1 for normal
        anomaly_indices = [i for i, label in enumerate(anomaly_labels) if label == -1]
        anomalies_detected = len(anomaly_indices)
        
        return AnomalyDetectionResult(
            total_samples=len(features),
            anomalies_detected=anomalies_detected,
            anomaly_indices=[sample_indices[i] for i in anomaly_indices],
            anomaly_scores=[anomaly_scores[i] for i in anomaly_indices],
            contamination_estimate=contamination,
            features_used=feature_names
        )
    
    def assess_convergence_quality(self) -> Dict[str, float]:
        """Assess convergence quality across phases"""
        convergence_quality = {}
        
        if not self.results:
            return convergence_quality
        
        convergence_detector = AdvancedConvergenceDetector()
        
        for phase_name, phase_results in self.results.items():
            # Add results to convergence detector
            convergence_detector.add_results(phase_results)
            
            # Check convergence
            convergence_result = convergence_detector.check_convergence()
            
            # Quality score based on convergence level and confidence
            if convergence_result.convergence_level == "complete":
                quality_score = 1.0
            elif convergence_result.convergence_level == "strong":
                quality_score = 0.8
            elif convergence_result.convergence_level == "partial":
                quality_score = 0.5
            else:
                quality_score = 0.2
            
            # Adjust by confidence
            quality_score *= convergence_result.confidence
            
            convergence_quality[phase_name] = quality_score
        
        return convergence_quality
    
    def assess_reproducibility(self) -> float:
        """Assess reproducibility by comparing similar parameter sets"""
        if not self.results or len(self.results) < 2:
            return 0.0
        
        # This is a simplified reproducibility assessment
        # In a full implementation, we would run identical parameter sets multiple times
        
        reproducibility_scores = []
        
        # Compare results from different phases with similar parameters
        phase_names = list(self.results.keys())
        
        for i in range(len(phase_names) - 1):
            phase1 = phase_names[i]
            phase2 = phase_names[i + 1]
            
            # Extract detection times from both phases
            times1 = [r.get("t5sigma_s") for r in self.results[phase1] 
                     if r.get("simulation_success") and r.get("t5sigma_s") is not None]
            times2 = [r.get("t5sigma_s") for r in self.results[phase2] 
                     if r.get("simulation_success") and r.get("t5sigma_s") is not None]
            
            if len(times1) > 5 and len(times2) > 5:
                # Compare distributions using Wasserstein distance (Earth Mover's Distance)
                try:
                    from scipy.stats import wasserstein_distance
                    distance = wasserstein_distance(times1, times2)
                    
                    # Convert distance to similarity score (lower distance = higher similarity)
                    # Normalize by median values
                    median1 = np.median(times1)
                    median2 = np.median(times2)
                    normalization = max(median1, median2)
                    
                    if normalization > 0:
                        normalized_distance = distance / normalization
                        similarity = 1.0 - min(normalized_distance, 1.0)
                        reproducibility_scores.append(similarity)
                except:
                    continue
        
        if reproducibility_scores:
            return np.mean(reproducibility_scores)
        else:
            return 0.0
    
    def _get_parameter_boundary_violations(self) -> List[str]:
        """Get detailed list of parameter boundary violations"""
        violations = []
        
        if not self.results:
            return violations
        
        parameter_bounds = {
            "laser_intensity": (1e15, 1e21, "W/m²"),
            "plasma_density": (1e15, 1e21, "m⁻³"),
            "temperature_constant": (1e2, 1e6, "K"),
            "magnetic_field": (0, 1000, "T")
        }
        
        for phase_name, phase_results in self.results.items():
            for i, result in enumerate(phase_results):
                params = result.get("parameters_used", {})
                
                for param_name, (min_val, max_val, unit) in parameter_bounds.items():
                    if param_name in params:
                        value = params[param_name]
                        
                        if value < min_val:
                            violations.append(
                                f"{phase_name}[{i}].{param_name} = {value:.2e} {unit} < {min_val:.2e} {unit}"
                            )
                        elif value > max_val:
                            violations.append(
                                f"{phase_name}[{i}].{param_name} = {value:.2e} {unit} > {max_val:.2e} {unit}"
                            )
        
        return violations
    
    def _calculate_overall_quality_score(self, quality_metrics: List[QualityMetric]) -> float:
        """Calculate overall quality score from individual metrics"""
        if not quality_metrics:
            return 0.0
        
        total_weight = sum(metric.weight for metric in quality_metrics)
        if total_weight == 0:
            return 0.0
        
        weighted_sum = sum(metric.score * metric.weight for metric in quality_metrics)
        return weighted_sum / total_weight
    
    def _generate_recommendations(self, quality_metrics: List[QualityMetric], 
                                anomalies: AnomalyDetectionResult,
                                overall_score: float) -> List[str]:
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Recommendations based on quality metrics
        for metric in quality_metrics:
            if not metric.passed:
                if metric.name == "data_completeness":
                    recommendations.append("Improve data collection to ensure all required fields are populated")
                elif metric.name == "parameter_boundaries":
                    recommendations.append("Review parameter sampling to avoid boundary violations")
                elif metric.name == "statistical_consistency":
                    recommendations.append("Investigate statistical inconsistencies in results")
                elif metric.name == "physical_plausibility":
                    recommendations.append("Verify physical parameters and model assumptions")
        
        # Recommendations based on anomalies
        if anomalies.anomalies_detected > 0:
            anomaly_rate = anomalies.anomalies_detected / anomalies.total_samples
            if anomaly_rate > 0.1:
                recommendations.append(f"Investigate {anomalies.anomalies_detected} anomalous results detected")
        
        # General recommendations based on overall score
        if overall_score < 0.5:
            recommendations.append("Significant quality issues detected - comprehensive review recommended")
        elif overall_score < 0.8:
            recommendations.append("Moderate quality issues - address key recommendations")
        
        return list(set(recommendations))  # Remove duplicates
    
    def save_quality_report(self, report: QualityAssuranceReport, output_path: Optional[Path] = None) -> None:
        """Save quality assurance report to disk"""
        if not output_path:
            output_path = self.experiment_dir / "quality_assurance_report.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "quality_assurance_report": asdict(report),
            "thresholds_used": self.thresholds,
            "report_timestamp": report.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Saved quality assurance report to {output_path}")
    
    def generate_quality_report_text(self, report: QualityAssuranceReport) -> str:
        """Generate human-readable quality assurance report"""
        quality_text = f"QUALITY ASSURANCE REPORT - Experiment {self.experiment_id}\n"
        quality_text += "=" * 60 + "\n\n"
        
        quality_text += f"Report Timestamp: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
        quality_text += f"Overall Quality Score: {report.overall_quality_score:.3f}\n\n"
        
        quality_text += "QUALITY METRICS\n"
        quality_text += "-" * 30 + "\n"
        for metric in report.quality_metrics:
            status = "PASS" if metric.passed else "FAIL"
            quality_text += f"{metric.name}: {status} (score: {metric.score:.3f})\n"
            quality_text += f"  {metric.description}\n\n"
        
        quality_text += "ANOMALY DETECTION\n"
        quality_text += "-" * 30 + "\n"
        quality_text += f"Total samples analyzed: {report.anomalies.total_samples}\n"
        quality_text += f"Anomalies detected: {report.anomalies.anomalies_detected}\n"
        quality_text += f"Anomaly rate: {report.anomalies.anomalies_detected/report.anomalies.total_samples:.1%}\n\n"
        
        quality_text += "PARAMETER BOUNDARY VIOLATIONS\n"
        quality_text += "-" * 30 + "\n"
        if report.parameter_boundary_violations:
            for violation in report.parameter_boundary_violations[:10]:  # Show first 10
                quality_text += f"• {violation}\n"
            if len(report.parameter_boundary_violations) > 10:
                quality_text += f"... and {len(report.parameter_boundary_violations) - 10} more violations\n"
        else:
            quality_text += "No parameter boundary violations detected\n"
        
        quality_text += "\nCONVERGENCE QUALITY\n"
        quality_text += "-" * 30 + "\n"
        for phase, score in report.convergence_quality.items():
            quality_text += f"{phase}: {score:.3f}\n"
        
        quality_text += f"\nReproducibility Score: {report.reproducibility_score:.3f}\n\n"
        
        quality_text += "RECOMMENDATIONS\n"
        quality_text += "-" * 30 + "\n"
        if report.recommendations:
            for rec in report.recommendations:
                quality_text += f"• {rec}\n"
        else:
            quality_text += "No specific recommendations\n"
        
        return quality_text


def main():
    """Main entry point for quality assurance system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Assurance System")
    parser.add_argument("experiment_id", help="Experiment ID to analyze")
    parser.add_argument("--output", help="Output file for quality report")
    parser.add_argument("--text", action="store_true", help="Generate text report instead of JSON")
    
    args = parser.parse_args()
    
    # Run quality assurance
    qa_system = QualityAssuranceSystem(args.experiment_id)
    report = qa_system.run_comprehensive_quality_assurance()
    
    # Generate report
    if args.text:
        report_text = qa_system.generate_quality_report_text(report)
        print(report_text)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report_text)
            print(f"Text report saved to {args.output}")
    else:
        qa_system.save_quality_report(report, Path(args.output) if args.output else None)
        print(f"Quality assurance completed: overall score = {report.overall_quality_score:.3f}")


if __name__ == "__main__":
    main()
