#!/usr/bin/env python3
"""
Cross-Phase Validation System for Analog Hawking Radiation Experiments

Provides validation of consistency and progression across experiment phases,
including phase transition validation, metric progression analysis, and
cross-phase correlation checking.
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
from scipy.spatial.distance import jensenshannon

# Add project paths to Python path
import sys
# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.convergence_detector import AdvancedConvergenceDetector
from scripts.analyze_significance import StatisticalAnalyzer


@dataclass
class PhaseTransitionResult:
    """Result of phase transition validation"""
    from_phase: str
    to_phase: str
    passed: bool
    confidence: float
    improvement_metrics: Dict[str, float]
    consistency_metrics: Dict[str, float]
    message: str
    recommendations: List[str]


@dataclass
class CrossPhaseValidationSummary:
    """Summary of cross-phase validation results"""
    timestamp: float
    experiment_id: str
    total_transitions: int
    valid_transitions: int
    invalid_transitions: int
    overall_consistency: float
    phase_transition_results: List[PhaseTransitionResult]
    progression_analysis: Dict[str, Any]
    recommendations: List[str]


class CrossPhaseValidator:
    """Validates consistency and progression across experiment phases"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Optional[Dict[str, Any]] = None
        
        # Validation thresholds
        self.thresholds = {
            "improvement_min": 0.1,           # Minimum 10% improvement expected
            "consistency_min": 0.7,           # Minimum consistency score
            "success_rate_improvement_min": 0.05,  # 5% improvement in success rate
            "distribution_similarity_min": 0.8,    # Minimum distribution similarity
            "parameter_refinement_consistency": 0.6  # Parameter refinement consistency
        }
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load experiment data
        self.load_experiment_data()
    
    def _setup_logging(self) -> None:
        """Setup cross-phase validation logging"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'cross_phase_validation.log'),
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
    
    def run_comprehensive_cross_phase_validation(self) -> CrossPhaseValidationSummary:
        """Run comprehensive cross-phase validation"""
        self.logger.info(f"Starting comprehensive cross-phase validation for experiment {self.experiment_id}")
        
        phase_transition_results = []
        
        # Get phase order from manifest or directory structure
        phase_order = self._get_phase_order()
        
        # Validate each phase transition
        for i in range(len(phase_order) - 1):
            from_phase = phase_order[i]
            to_phase = phase_order[i + 1]
            
            transition_result = self.validate_phase_transition(from_phase, to_phase)
            phase_transition_results.append(transition_result)
        
        # Analyze overall progression
        progression_analysis = self.analyze_progression_patterns(phase_order)
        
        # Generate summary
        summary = self._generate_summary(phase_transition_results, progression_analysis)
        
        self.logger.info(f"Cross-phase validation completed: {summary.valid_transitions}/{summary.total_transitions} valid transitions")
        
        return summary
    
    def _get_phase_order(self) -> List[str]:
        """Get the order of phases from manifest or directory structure"""
        if self.manifest and "phases" in self.manifest:
            return self.manifest["phases"]
        
        # Fallback: sort phase directories by name
        phase_dirs = [name for name in self.results.keys() if name.startswith("phase_")]
        return sorted(phase_dirs)
    
    def validate_phase_transition(self, from_phase: str, to_phase: str) -> PhaseTransitionResult:
        """Validate a specific phase transition"""
        self.logger.info(f"Validating phase transition: {from_phase} -> {to_phase}")
        
        if from_phase not in self.results or to_phase not in self.results:
            return PhaseTransitionResult(
                from_phase=from_phase,
                to_phase=to_phase,
                passed=False,
                confidence=0.0,
                improvement_metrics={},
                consistency_metrics={},
                message=f"Missing phase data: {from_phase} or {to_phase}",
                recommendations=["Check phase result files"]
            )
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(from_phase, to_phase)
        
        # Calculate consistency metrics
        consistency_metrics = self._calculate_consistency_metrics(from_phase, to_phase)
        
        # Calculate overall transition score
        transition_score = self._calculate_transition_score(improvement_metrics, consistency_metrics)
        
        # Determine if transition is valid
        passed = transition_score >= 0.6  # Minimum threshold for valid transition
        confidence = transition_score
        
        # Generate message and recommendations
        message = self._generate_transition_message(from_phase, to_phase, improvement_metrics, consistency_metrics)
        recommendations = self._generate_transition_recommendations(improvement_metrics, consistency_metrics, passed)
        
        return PhaseTransitionResult(
            from_phase=from_phase,
            to_phase=to_phase,
            passed=passed,
            confidence=confidence,
            improvement_metrics=improvement_metrics,
            consistency_metrics=consistency_metrics,
            message=message,
            recommendations=recommendations
        )
    
    def _calculate_improvement_metrics(self, from_phase: str, to_phase: str) -> Dict[str, float]:
        """Calculate improvement metrics between phases"""
        metrics = {}
        
        # Extract key performance indicators
        from_kpis = self._extract_phase_kpis(from_phase)
        to_kpis = self._extract_phase_kpis(to_phase)
        
        if not from_kpis or not to_kpis:
            return metrics
        
        # Detection time improvement (lower is better)
        if from_kpis["best_detection_time"] and to_kpis["best_detection_time"]:
            improvement = (from_kpis["best_detection_time"] - to_kpis["best_detection_time"]) / from_kpis["best_detection_time"]
            metrics["detection_time_improvement"] = max(0.0, improvement)  # Negative improvement = degradation
        
        # Kappa improvement (higher is better)
        if from_kpis["best_kappa"] and to_kpis["best_kappa"]:
            improvement = (to_kpis["best_kappa"] - from_kpis["best_kappa"]) / from_kpis["best_kappa"]
            metrics["kappa_improvement"] = max(0.0, improvement)
        
        # Success rate improvement
        improvement = to_kpis["success_rate"] - from_kpis["success_rate"]
        metrics["success_rate_improvement"] = max(0.0, improvement)
        
        # Statistical significance improvement
        if from_kpis["detection_probability_5sigma"] and to_kpis["detection_probability_5sigma"]:
            improvement = to_kpis["detection_probability_5sigma"] - from_kpis["detection_probability_5sigma"]
            metrics["significance_improvement"] = max(0.0, improvement)
        
        return metrics
    
    def _calculate_consistency_metrics(self, from_phase: str, to_phase: str) -> Dict[str, float]:
        """Calculate consistency metrics between phases"""
        metrics = {}
        
        from_results = self.results[from_phase]
        to_results = self.results[to_phase]
        
        # Distribution similarity for key metrics
        detection_time_similarity = self._calculate_distribution_similarity(
            from_results, to_results, "t5sigma_s"
        )
        if detection_time_similarity is not None:
            metrics["detection_time_similarity"] = detection_time_similarity
        
        kappa_similarity = self._calculate_distribution_similarity(
            from_results, to_results, "kappa", aggregation="max"
        )
        if kappa_similarity is not None:
            metrics["kappa_similarity"] = kappa_similarity
        
        # Parameter space consistency
        parameter_consistency = self._calculate_parameter_consistency(from_phase, to_phase)
        metrics["parameter_consistency"] = parameter_consistency
        
        # Statistical consistency
        statistical_consistency = self._calculate_statistical_consistency(from_results, to_results)
        metrics["statistical_consistency"] = statistical_consistency
        
        return metrics
    
    def _calculate_distribution_similarity(self, results1: List[Dict], results2: List[Dict], 
                                         metric: str, aggregation: str = "direct") -> Optional[float]:
        """Calculate distribution similarity for a specific metric"""
        values1 = self._extract_metric_values(results1, metric, aggregation)
        values2 = self._extract_metric_values(results2, metric, aggregation)
        
        if not values1 or not values2:
            return None
        
        # Use Jensen-Shannon divergence for distribution similarity
        try:
            # Create histograms for comparison
            all_values = values1 + values2
            hist_range = (np.min(all_values), np.max(all_values))
            
            hist1, _ = np.histogram(values1, bins=20, range=hist_range, density=True)
            hist2, _ = np.histogram(values2, bins=20, range=hist_range, density=True)
            
            # Add small epsilon to avoid zeros
            hist1 = hist1 + 1e-10
            hist2 = hist2 + 1e-10
            
            # Normalize
            hist1 = hist1 / np.sum(hist1)
            hist2 = hist2 / np.sum(hist2)
            
            # Calculate Jensen-Shannon distance (0 = identical, 1 = completely different)
            js_distance = jensenshannon(hist1, hist2)
            
            # Convert to similarity (1 - distance)
            similarity = 1.0 - js_distance
            
            return max(0.0, similarity)
            
        except Exception as e:
            self.logger.warning(f"Distribution similarity calculation failed: {e}")
            return None
    
    def _extract_metric_values(self, results: List[Dict], metric: str, aggregation: str = "direct") -> List[float]:
        """Extract metric values from results with optional aggregation"""
        values = []
        
        for result in results:
            if not result.get("simulation_success"):
                continue
            
            if metric == "t5sigma_s":
                value = result.get("t5sigma_s")
                if value is not None:
                    values.append(value)
            
            elif metric == "kappa":
                kappa_list = result.get("kappa", [])
                if kappa_list:
                    if aggregation == "max":
                        values.append(max(kappa_list))
                    elif aggregation == "mean":
                        values.append(np.mean(kappa_list))
                    else:
                        values.extend(kappa_list)  # Use all kappa values
            
            elif metric == "T_sig_K":
                value = result.get("T_sig_K")
                if value is not None:
                    values.append(value)
        
        return values
    
    def _calculate_parameter_consistency(self, from_phase: str, to_phase: str) -> float:
        """Calculate parameter space consistency between phases"""
        # This would require access to phase configurations
        # For now, use a simplified approach based on result distributions
        
        from_results = self.results[from_phase]
        to_results = self.results[to_phase]
        
        # Extract parameter ranges from successful results
        from_params = self._extract_parameter_ranges(from_results)
        to_params = self._extract_parameter_ranges(to_results)
        
        if not from_params or not to_params:
            return 0.0
        
        # Calculate overlap in parameter spaces
        overlap_scores = []
        
        for param_name in from_params.keys():
            if param_name in to_params:
                from_min, from_max = from_params[param_name]
                to_min, to_max = to_params[param_name]
                
                # Calculate overlap ratio
                overlap_min = max(from_min, to_min)
                overlap_max = min(from_max, to_max)
                
                if overlap_min < overlap_max:
                    overlap_range = overlap_max - overlap_min
                    from_range = from_max - from_min
                    
                    if from_range > 0:
                        overlap_ratio = overlap_range / from_range
                        overlap_scores.append(overlap_ratio)
        
        if overlap_scores:
            return np.mean(overlap_scores)
        else:
            return 0.0
    
    def _extract_parameter_ranges(self, results: List[Dict]) -> Dict[str, Tuple[float, float]]:
        """Extract parameter ranges from results"""
        param_ranges = {}
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        
        for result in results:
            params = result.get("parameters_used", {})
            
            for param_name in key_params:
                if param_name in params:
                    value = params[param_name]
                    
                    if param_name not in param_ranges:
                        param_ranges[param_name] = (value, value)
                    else:
                        current_min, current_max = param_ranges[param_name]
                        param_ranges[param_name] = (min(current_min, value), max(current_max, value))
        
        return param_ranges
    
    def _calculate_statistical_consistency(self, results1: List[Dict], results2: List[Dict]) -> float:
        """Calculate statistical consistency between phase results"""
        # Extract detection times from successful results
        times1 = [r["t5sigma_s"] for r in results1 
                 if r.get("simulation_success") and r.get("t5sigma_s") is not None]
        times2 = [r["t5sigma_s"] for r in results2 
                 if r.get("simulation_success") and r.get("t5sigma_s") is not None]
        
        if len(times1) < 5 or len(times2) < 5:
            return 0.0
        
        # Use Kolmogorov-Smirnov test for distribution similarity
        try:
            ks_stat, p_value = stats.ks_2samp(times1, times2)
            
            # Higher p-value indicates more similar distributions
            # Convert to consistency score (0 to 1)
            consistency = min(p_value * 10, 1.0)  # Scale p-value
            
            return consistency
            
        except Exception as e:
            self.logger.warning(f"Statistical consistency calculation failed: {e}")
            return 0.0
    
    def _extract_phase_kpis(self, phase_name: str) -> Dict[str, Any]:
        """Extract key performance indicators for a phase"""
        results = self.results.get(phase_name, [])
        
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("simulation_success")]
        total_results = len(results)
        success_rate = len(successful_results) / total_results if total_results > 0 else 0.0
        
        # Best detection time (lowest is best)
        detection_times = [r.get("t5sigma_s") for r in successful_results if r.get("t5sigma_s") is not None]
        best_detection_time = min(detection_times) if detection_times else None
        
        # Best kappa (highest is best)
        kappa_values = []
        for r in successful_results:
            kappa_list = r.get("kappa", [])
            if kappa_list:
                kappa_values.append(max(kappa_list))
        best_kappa = max(kappa_values) if kappa_values else None
        
        # Statistical significance
        analyzer = StatisticalAnalyzer()
        enhanced_results = analyzer.calculate_signal_to_noise(successful_results)
        detection_stats = analyzer.calculate_detection_probability(enhanced_results, 3600)
        detection_probability_5sigma = detection_stats.get("detection_probability_5sigma", 0.0)
        
        return {
            "success_rate": success_rate,
            "best_detection_time": best_detection_time,
            "best_kappa": best_kappa,
            "detection_probability_5sigma": detection_probability_5sigma,
            "total_simulations": total_results,
            "successful_simulations": len(successful_results)
        }
    
    def _calculate_transition_score(self, improvement_metrics: Dict[str, float], 
                                  consistency_metrics: Dict[str, float]) -> float:
        """Calculate overall transition score"""
        if not improvement_metrics and not consistency_metrics:
            return 0.0
        
        # Weight improvement and consistency
        improvement_weight = 0.6
        consistency_weight = 0.4
        
        # Calculate average improvement (ignore negative improvements)
        if improvement_metrics:
            avg_improvement = np.mean(list(improvement_metrics.values()))
        else:
            avg_improvement = 0.0
        
        # Calculate average consistency
        if consistency_metrics:
            avg_consistency = np.mean(list(consistency_metrics.values()))
        else:
            avg_consistency = 0.0
        
        # Combined score
        transition_score = (improvement_weight * avg_improvement + 
                          consistency_weight * avg_consistency)
        
        return min(1.0, transition_score)
    
    def _generate_transition_message(self, from_phase: str, to_phase: str,
                                   improvement_metrics: Dict[str, float],
                                   consistency_metrics: Dict[str, float]) -> str:
        """Generate descriptive message for phase transition"""
        messages = []
        
        if improvement_metrics:
            avg_improvement = np.mean(list(improvement_metrics.values()))
            if avg_improvement > 0.1:
                messages.append(f"Good improvement ({avg_improvement:.1%})")
            elif avg_improvement > 0:
                messages.append(f"Modest improvement ({avg_improvement:.1%})")
            else:
                messages.append("No significant improvement")
        else:
            messages.append("No improvement data")
        
        if consistency_metrics:
            avg_consistency = np.mean(list(consistency_metrics.values()))
            if avg_consistency > 0.8:
                messages.append("excellent consistency")
            elif avg_consistency > 0.6:
                messages.append("good consistency")
            else:
                messages.append("poor consistency")
        else:
            messages.append("no consistency data")
        
        return f"Transition {from_phase} → {to_phase}: {', '.join(messages)}"
    
    def _generate_transition_recommendations(self, improvement_metrics: Dict[str, float],
                                           consistency_metrics: Dict[str, float],
                                           passed: bool) -> List[str]:
        """Generate recommendations for phase transition"""
        recommendations = []
        
        if not passed:
            recommendations.append("Review phase transition criteria")
        
        # Improvement-based recommendations
        if improvement_metrics:
            if improvement_metrics.get("detection_time_improvement", 0) < 0.05:
                recommendations.append("Focus on reducing detection times in next phase")
            
            if improvement_metrics.get("kappa_improvement", 0) < 0.05:
                recommendations.append("Work on increasing surface gravity values")
            
            if improvement_metrics.get("success_rate_improvement", 0) < 0.02:
                recommendations.append("Improve simulation success rate")
        
        # Consistency-based recommendations
        if consistency_metrics:
            if consistency_metrics.get("detection_time_similarity", 1) < 0.7:
                recommendations.append("Investigate detection time distribution changes")
            
            if consistency_metrics.get("parameter_consistency", 1) < 0.5:
                recommendations.append("Ensure parameter space continuity between phases")
        
        return recommendations
    
    def analyze_progression_patterns(self, phase_order: List[str]) -> Dict[str, Any]:
        """Analyze progression patterns across all phases"""
        progression = {
            "detection_time_trend": self._analyze_metric_trend(phase_order, "t5sigma_s", "decreasing"),
            "kappa_trend": self._analyze_metric_trend(phase_order, "kappa", "increasing", aggregation="max"),
            "success_rate_trend": self._analyze_metric_trend(phase_order, "success_rate", "increasing"),
            "convergence_progression": self._analyze_convergence_progression(phase_order)
        }
        
        return progression
    
    def _analyze_metric_trend(self, phase_order: List[str], metric: str, 
                            expected_direction: str, aggregation: str = "best") -> Dict[str, Any]:
        """Analyze trend of a specific metric across phases"""
        phase_values = []
        
        for phase in phase_order:
            kpis = self._extract_phase_kpis(phase)
            
            if metric == "t5sigma_s":
                value = kpis.get("best_detection_time")
                if value is not None:
                    phase_values.append(value)
            
            elif metric == "kappa":
                value = kpis.get("best_kappa")
                if value is not None:
                    phase_values.append(value)
            
            elif metric == "success_rate":
                value = kpis.get("success_rate")
                if value is not None:
                    phase_values.append(value)
        
        if len(phase_values) < 2:
            return {"trend": "insufficient_data", "strength": 0.0, "direction": "unknown"}
        
        # Calculate trend using linear regression
        x = np.arange(len(phase_values))
        y = np.array(phase_values)
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            # Determine trend direction and strength
            if abs(slope) < 1e-6:
                trend_direction = "stable"
            elif slope > 0:
                trend_direction = "increasing"
            else:
                trend_direction = "decreasing"
            
            trend_strength = r_value ** 2  # R-squared
            
            # Check if trend matches expected direction
            matches_expected = (trend_direction == expected_direction)
            
            return {
                "trend": trend_direction,
                "strength": trend_strength,
                "matches_expected": matches_expected,
                "slope": slope,
                "r_squared": trend_strength
            }
            
        except Exception as e:
            self.logger.warning(f"Trend analysis failed for {metric}: {e}")
            return {"trend": "analysis_failed", "strength": 0.0, "direction": "unknown"}
    
    def _analyze_convergence_progression(self, phase_order: List[str]) -> Dict[str, Any]:
        """Analyze convergence progression across phases"""
        # This would require convergence data from each phase
        # For now, use a simplified approach based on result variability
        
        convergence_scores = []
        
        for phase in phase_order:
            results = self.results.get(phase, [])
            
            if not results:
                continue
            
            # Calculate convergence proxy based on result variability
            detection_times = [r.get("t5sigma_s") for r in results 
                             if r.get("simulation_success") and r.get("t5sigma_s") is not None]
            
            if len(detection_times) > 5:
                # Lower coefficient of variation indicates better convergence
                cv = np.std(detection_times) / np.mean(detection_times) if np.mean(detection_times) > 0 else 1.0
                convergence_score = 1.0 - min(cv, 1.0)  # Higher is better
                convergence_scores.append(convergence_score)
        
        if len(convergence_scores) < 2:
            return {"progression": "insufficient_data", "improving": False}
        
        # Check if convergence is improving (scores increasing)
        improving = all(convergence_scores[i] <= convergence_scores[i+1] 
                       for i in range(len(convergence_scores)-1))
        
        return {
            "progression": "improving" if improving else "stable" if len(set(convergence_scores)) == 1 else "fluctuating",
            "improving": improving,
            "scores": convergence_scores
        }
    
    def _generate_summary(self, transition_results: List[PhaseTransitionResult],
                         progression_analysis: Dict[str, Any]) -> CrossPhaseValidationSummary:
        """Generate cross-phase validation summary"""
        total_transitions = len(transition_results)
        valid_transitions = sum(1 for r in transition_results if r.passed)
        invalid_transitions = total_transitions - valid_transitions
        
        # Calculate overall consistency
        if transition_results:
            overall_consistency = np.mean([r.confidence for r in transition_results])
        else:
            overall_consistency = 0.0
        
        # Generate recommendations
        recommendations = self._generate_overall_recommendations(transition_results, progression_analysis)
        
        return CrossPhaseValidationSummary(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            total_transitions=total_transitions,
            valid_transitions=valid_transitions,
            invalid_transitions=invalid_transitions,
            overall_consistency=overall_consistency,
            phase_transition_results=transition_results,
            progression_analysis=progression_analysis,
            recommendations=recommendations
        )
    
    def _generate_overall_recommendations(self, transition_results: List[PhaseTransitionResult],
                                        progression_analysis: Dict[str, Any]) -> List[str]:
        """Generate overall recommendations based on cross-phase analysis"""
        recommendations = []
        
        # Check transition success rate
        success_rate = len([r for r in transition_results if r.passed]) / len(transition_results) if transition_results else 0.0
        
        if success_rate < 0.5:
            recommendations.append("Review phase transition criteria and thresholds")
        
        # Check progression patterns
        detection_trend = progression_analysis.get("detection_time_trend", {})
        if detection_trend.get("trend") == "increasing":
            recommendations.append("Detection times are increasing - optimize parameters")
        
        kappa_trend = progression_analysis.get("kappa_trend", {})
        if kappa_trend.get("trend") == "decreasing":
            recommendations.append("Surface gravity values are decreasing - focus on horizon optimization")
        
        convergence_progression = progression_analysis.get("convergence_progression", {})
        if not convergence_progression.get("improving", False):
            recommendations.append("Convergence is not improving across phases - review optimization strategy")
        
        return recommendations
    
    def save_cross_phase_report(self, summary: CrossPhaseValidationSummary, output_path: Optional[Path] = None) -> None:
        """Save cross-phase validation report to disk"""
        if not output_path:
            output_path = self.experiment_dir / "cross_phase_validation_report.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report_data = {
            "cross_phase_validation_summary": asdict(summary),
            "thresholds_used": self.thresholds,
            "validation_timestamp": summary.timestamp
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        self.logger.info(f"Saved cross-phase validation report to {output_path}")
    
    def generate_cross_phase_report_text(self, summary: CrossPhaseValidationSummary) -> str:
        """Generate human-readable cross-phase validation report"""
        report = f"CROSS-PHASE VALIDATION REPORT - Experiment {self.experiment_id}\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Validation Timestamp: {datetime.fromtimestamp(summary.timestamp).strftime('%Y-%m-%d %H:%M:%S')}\n"
        report += f"Overall Result: {summary.valid_transitions}/{summary.total_transitions} valid transitions\n"
        report += f"Overall Consistency: {summary.overall_consistency:.3f}\n\n"
        
        report += "PHASE TRANSITION RESULTS\n"
        report += "-" * 30 + "\n"
        for result in summary.phase_transition_results:
            status = "VALID" if result.passed else "INVALID"
            report += f"{result.from_phase} → {result.to_phase}: {status} (confidence: {result.confidence:.3f})\n"
            report += f"  {result.message}\n"
            if result.recommendations:
                report += f"  Recommendations: {', '.join(result.recommendations)}\n"
            report += "\n"
        
        report += "PROGRESSION ANALYSIS\n"
        report += "-" * 30 + "\n"
        for metric, analysis in summary.progression_analysis.items():
            report += f"{metric}: {analysis.get('trend', 'unknown')} "
            report += f"(strength: {analysis.get('strength', 0):.3f})\n"
        
        report += "\nRECOMMENDATIONS\n"
        report += "-" * 30 + "\n"
        if summary.recommendations:
            for rec in summary.recommendations:
                report += f"• {rec}\n"
        else:
            report += "No specific recommendations\n"
        
        return report


def main():
    """Main entry point for cross-phase validator"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Cross-Phase Validation System")
    parser.add_argument("experiment_id", help="Experiment ID to validate")
    parser.add_argument("--output", help="Output file for cross-phase report")
    parser.add_argument("--text", action="store_true", help="Generate text report instead of JSON")
    
    args = parser.parse_args()
    
    # Run cross-phase validation
    validator = CrossPhaseValidator(args.experiment_id)
    summary = validator.run_comprehensive_cross_phase_validation()
    
    # Generate report
    if args.text:
        report_text = validator.generate_cross_phase_report_text(summary)
        print(report_text)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report_text)
            print(f"Text report saved to {args.output}")
    else:
        validator.save_cross_phase_report(summary, Path(args.output) if args.output else None)
        print(f"Cross-phase validation completed: {summary.valid_transitions}/{summary.total_transitions} valid transitions")
        print(f"Overall consistency: {summary.overall_consistency:.3f}")


if __name__ == "__main__":
    main()
