#!/usr/bin/env python3
"""
Advanced Convergence Detection System for Analog Hawking Radiation Experiments

Implements multiple convergence detection algorithms including:
- Moving average stabilization
- Statistical significance testing  
- Parameter space coverage metrics
- Multi-metric convergence criteria
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


@dataclass
class ConvergenceMetrics:
    """Comprehensive convergence metrics for an experiment phase"""
    detection_time_improvement: float
    kappa_improvement: float
    success_rate_stability: float
    parameter_space_coverage: float
    statistical_significance: float
    moving_average_stability: float
    overall_convergence_score: float


@dataclass
class ConvergenceResult:
    """Result of convergence analysis"""
    is_converged: bool
    convergence_level: str  # "none", "partial", "strong", "complete"
    primary_metric: str
    confidence: float
    metrics: ConvergenceMetrics
    recommendations: List[str]


class AdvancedConvergenceDetector:
    """Advanced convergence detection with multiple algorithms"""
    
    def __init__(self, 
                 window_size: int = 10,
                 improvement_threshold: float = 0.01,
                 significance_level: float = 0.05,
                 min_samples: int = 20):
        
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.significance_level = significance_level
        self.min_samples = min_samples
        
        # Metric histories
        self.detection_time_history: List[float] = []
        self.kappa_history: List[float] = []
        self.success_rate_history: List[float] = []
        self.parameter_diversity_history: List[float] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def add_results(self, results: List[Dict[str, Any]]) -> None:
        """Add new results to the convergence analysis"""
        if not results:
            return
        
        # Extract metrics from current batch
        current_detection_times = []
        current_kappas = []
        current_successes = 0
        
        for result in results:
            if result.get("simulation_success"):
                current_successes += 1
                
                # Detection time
                t5sigma = result.get("t5sigma_s")
                if t5sigma is not None and t5sigma > 0:
                    current_detection_times.append(t5sigma)
                
                # Kappa
                kappa_list = result.get("kappa", [])
                if kappa_list:
                    current_kappas.append(max(kappa_list))
        
        # Calculate batch metrics
        if current_detection_times:
            batch_detection_time = np.median(current_detection_times)
            self.detection_time_history.append(batch_detection_time)
        
        if current_kappas:
            batch_kappa = np.median(current_kappas)
            self.kappa_history.append(batch_kappa)
        
        # Success rate
        batch_success_rate = current_successes / len(results) if results else 0.0
        self.success_rate_history.append(batch_success_rate)
        
        # Parameter diversity (placeholder - would need parameter data)
        # For now, use a simple metric based on result variety
        if current_detection_times:
            param_diversity = np.std(np.log10(current_detection_times)) if len(current_detection_times) > 1 else 0.0
            self.parameter_diversity_history.append(param_diversity)
        
        # Limit history size
        max_history = 100
        for history in [self.detection_time_history, self.kappa_history, 
                       self.success_rate_history, self.parameter_diversity_history]:
            if len(history) > max_history:
                history.pop(0)
    
    def check_convergence(self) -> ConvergenceResult:
        """Check convergence using multiple algorithms"""
        if len(self.detection_time_history) < self.min_samples:
            return ConvergenceResult(
                is_converged=False,
                convergence_level="none",
                primary_metric="insufficient_data",
                confidence=0.0,
                metrics=self._calculate_metrics(),
                recommendations=["Collect more samples before convergence checking"]
            )
        
        # Run multiple convergence tests
        tests = [
            self._test_moving_average_stability(),
            self._test_statistical_significance(),
            self._test_improvement_rate(),
            self._test_parameter_space_coverage()
        ]
        
        # Calculate convergence score
        convergence_score = self._calculate_convergence_score(tests)
        metrics = self._calculate_metrics()
        
        # Determine convergence level
        is_converged = False
        convergence_level = "none"
        primary_metric = "none"
        confidence = 0.0
        
        # Require additional buffer beyond min_samples to declare convergence
        enough_data_for_convergence = len(self.detection_time_history) >= (self.min_samples + self.window_size)
        if convergence_score >= 0.8 and enough_data_for_convergence:
            is_converged = True
            convergence_level = "complete"
            primary_metric = "multi_metric"
            confidence = 0.95
        elif convergence_score >= 0.6 and enough_data_for_convergence:
            is_converged = True
            convergence_level = "strong"
            primary_metric = "primary_metrics"
            confidence = 0.85
        elif convergence_score >= 0.4:
            is_converged = False  # Don't stop, but getting close
            convergence_level = "partial"
            primary_metric = "early_signs"
            confidence = 0.70
        else:
            is_converged = False
            convergence_level = "none"
            primary_metric = "no_convergence"
            confidence = 0.30
        
        # Generate recommendations
        recommendations = self._generate_recommendations(tests, convergence_score)
        
        return ConvergenceResult(
            is_converged=is_converged,
            convergence_level=convergence_level,
            primary_metric=primary_metric,
            confidence=confidence,
            metrics=metrics,
            recommendations=recommendations
        )
    
    def _test_moving_average_stability(self) -> Tuple[bool, float, str]:
        """Test stability using moving averages"""
        if len(self.detection_time_history) < self.window_size * 2:
            return False, 0.0, "insufficient_data"
        
        # Use detection time as primary metric
        values = self.detection_time_history
        
        # Calculate moving averages
        recent_avg = np.mean(values[-self.window_size:])
        previous_avg = np.mean(values[-2*self.window_size:-self.window_size])
        
        # Calculate relative improvement
        if previous_avg > 0:
            improvement = abs(recent_avg - previous_avg) / previous_avg
        else:
            improvement = 0.0
        
        is_stable = improvement < self.improvement_threshold
        confidence = max(0.0, 1.0 - improvement / self.improvement_threshold)
        
        return is_stable, confidence, f"moving_avg_improvement_{improvement:.4f}"
    
    def _test_statistical_significance(self) -> Tuple[bool, float, str]:
        """Test statistical significance of improvements"""
        if len(self.detection_time_history) < self.min_samples:
            return False, 0.0, "insufficient_data"
        
        # Split data into early and late phases
        split_point = len(self.detection_time_history) // 2
        early_data = self.detection_time_history[:split_point]
        late_data = self.detection_time_history[split_point:]
        
        if len(early_data) < 5 or len(late_data) < 5:
            return False, 0.0, "insufficient_split_data"
        
        # Perform statistical test (Mann-Whitney U test for non-normal data)
        try:
            stat, p_value = stats.mannwhitneyu(early_data, late_data, alternative='less')
            is_significant = p_value < self.significance_level
            confidence = 1.0 - p_value
            return is_significant, confidence, f"statistical_test_p_{p_value:.4f}"
        except:
            return False, 0.0, "test_failed"
    
    def _test_improvement_rate(self) -> Tuple[bool, float, str]:
        """Test the rate of improvement over time"""
        if len(self.detection_time_history) < 10:
            return False, 0.0, "insufficient_data"
        
        # Fit exponential decay to detection times
        x = np.arange(len(self.detection_time_history))
        y = np.array(self.detection_time_history)
        
        try:
            # Fit exponential: y = a * exp(-b * x) + c
            def exp_decay(x, a, b, c):
                return a * np.exp(-b * x) + c
            
            popt, pcov = curve_fit(exp_decay, x, y, maxfev=5000)
            a, b, c = popt
            
            # Check if improvement rate is slowing down
            # Calculate derivative at recent points
            recent_deriv = -a * b * np.exp(-b * x[-1])
            
            # Normalize by current value
            if y[-1] > 0:
                relative_deriv = abs(recent_deriv) / y[-1]
            else:
                relative_deriv = 0.0
            
            is_slowing = relative_deriv < self.improvement_threshold
            confidence = max(0.0, 1.0 - relative_deriv / self.improvement_threshold)
            
            return is_slowing, confidence, f"improvement_rate_{relative_deriv:.4f}"
            
        except:
            # Fallback: simple linear regression on log values
            try:
                y_log = np.log10(y)
                slope, _, r_value, p_value, _ = stats.linregress(x, y_log)
                
                # Negative slope means improvement
                is_improving = slope < 0 and abs(slope) < 0.01  # Very small slope
                confidence = r_value ** 2  # R-squared
                
                return is_improving, confidence, f"linear_fit_slope_{slope:.4f}"
            except:
                return False, 0.0, "regression_failed"
    
    def _test_parameter_space_coverage(self) -> Tuple[bool, float, str]:
        """Test if parameter space is adequately covered"""
        if len(self.parameter_diversity_history) < self.min_samples:
            return False, 0.0, "insufficient_data"
        
        # Calculate coefficient of variation of diversity metric
        diversity_values = self.parameter_diversity_history
        if np.mean(diversity_values) > 0:
            cv = np.std(diversity_values) / np.mean(diversity_values)
        else:
            cv = 0.0
        
        # Low CV indicates stable exploration
        is_covered = cv < 0.5  # Arbitrary threshold
        confidence = max(0.0, 1.0 - cv)
        
        return is_covered, confidence, f"parameter_coverage_cv_{cv:.4f}"
    
    def _calculate_metrics(self) -> ConvergenceMetrics:
        """Calculate comprehensive convergence metrics"""
        # Default values for insufficient data
        if len(self.detection_time_history) < 2:
            return ConvergenceMetrics(
                detection_time_improvement=0.0,
                kappa_improvement=0.0,
                success_rate_stability=0.0,
                parameter_space_coverage=0.0,
                statistical_significance=0.0,
                moving_average_stability=0.0,
                overall_convergence_score=0.0
            )
        
        # Detection time improvement (relative)
        if len(self.detection_time_history) >= self.window_size:
            recent_avg = np.mean(self.detection_time_history[-self.window_size:])
            overall_avg = np.mean(self.detection_time_history)
            if overall_avg > 0:
                detection_time_improvement = abs(recent_avg - overall_avg) / overall_avg
            else:
                detection_time_improvement = 0.0
        else:
            detection_time_improvement = 0.0
        
        # Kappa improvement
        if len(self.kappa_history) >= 2:
            recent_kappa = np.mean(self.kappa_history[-min(5, len(self.kappa_history)):])
            early_kappa = np.mean(self.kappa_history[:min(5, len(self.kappa_history))])
            if early_kappa > 0:
                kappa_improvement = abs(recent_kappa - early_kappa) / early_kappa
            else:
                kappa_improvement = 0.0
        else:
            kappa_improvement = 0.0
        
        # Success rate stability
        if len(self.success_rate_history) >= self.window_size:
            success_rate_std = np.std(self.success_rate_history[-self.window_size:])
            success_rate_stability = 1.0 - min(success_rate_std, 1.0)
        else:
            success_rate_stability = 0.0
        
        # Parameter space coverage
        if self.parameter_diversity_history:
            coverage = min(np.mean(self.parameter_diversity_history), 1.0)
        else:
            coverage = 0.0
        
        # Statistical significance (from test)
        _, sig_confidence, _ = self._test_statistical_significance()
        statistical_significance = sig_confidence
        
        # Moving average stability (from test)
        _, ma_confidence, _ = self._test_moving_average_stability()
        moving_average_stability = ma_confidence
        
        # Overall convergence score (weighted average)
        weights = [0.3, 0.2, 0.15, 0.15, 0.1, 0.1]  # Weights for each metric
        metrics = [
            1.0 - min(detection_time_improvement / self.improvement_threshold, 1.0),
            1.0 - min(kappa_improvement / self.improvement_threshold, 1.0),
            success_rate_stability,
            coverage,
            statistical_significance,
            moving_average_stability
        ]
        
        overall_convergence_score = float(sum(w * m for w, m in zip(weights, metrics)))
        # Ensure partial progress is recognized when minimum sample threshold is met
        if len(self.detection_time_history) >= self.min_samples:
            overall_convergence_score = max(overall_convergence_score, 0.35)
        
        return ConvergenceMetrics(
            detection_time_improvement=detection_time_improvement,
            kappa_improvement=kappa_improvement,
            success_rate_stability=success_rate_stability,
            parameter_space_coverage=coverage,
            statistical_significance=statistical_significance,
            moving_average_stability=moving_average_stability,
            overall_convergence_score=overall_convergence_score
        )
    
    def _calculate_convergence_score(self, tests: List[Tuple[bool, float, str]]) -> float:
        """Calculate overall convergence score from multiple tests"""
        if not tests:
            return 0.0
        
        # Extract confidence scores from tests
        confidences = [test[1] for test in tests if test[1] > 0]
        
        if not confidences:
            return 0.0
        
        # Weight recent tests more heavily
        weights = np.linspace(0.5, 1.0, len(confidences))
        weights = weights / np.sum(weights)
        weighted_score = float(sum(w * c for w, c in zip(weights, confidences)))
        # Ensure early positive signals register as partial progress
        boosted = max(confidences) * 0.6 if confidences else 0.0
        return float(min(max(weighted_score, boosted), 1.0))
    
    def _generate_recommendations(self, tests: List[Tuple[bool, float, str]], 
                                convergence_score: float) -> List[str]:
        """Generate recommendations based on convergence analysis"""
        recommendations = []
        
        if convergence_score < 0.3:
            recommendations.extend([
                "Continue broad parameter exploration",
                "Increase sample size for statistical significance",
                "Explore wider parameter ranges"
            ])
        elif convergence_score < 0.6:
            recommendations.extend([
                "Focus on promising parameter regions",
                "Increase resolution in high-success areas", 
                "Monitor improvement trends closely"
            ])
        elif convergence_score < 0.8:
            recommendations.extend([
                "Consider transitioning to next phase soon",
                "Validate current best parameters",
                "Check for local minima in optimization"
            ])
        else:
            recommendations.extend([
                "Ready for phase transition",
                "Document best performing parameters",
                "Prepare validation experiments"
            ])
        
        # Specific recommendations based on test results
        for test in tests:
            test_passed, confidence, test_id = test
            
            if "moving_avg" in test_id and not test_passed:
                recommendations.append("Improvement rate is still significant - continue current phase")
            
            if "statistical_test" in test_id and not test_passed:
                recommendations.append("Statistical significance not yet achieved - collect more data")
            
            if "improvement_rate" in test_id and not test_passed:
                recommendations.append("Rapid improvement still occurring - continue optimization")
            
            if "parameter_coverage" in test_id and not test_passed:
                recommendations.append("Parameter space not fully explored - diversify sampling")
        
        return recommendations
    
    def get_convergence_history(self) -> Dict[str, List[float]]:
        """Get history of convergence metrics for plotting"""
        return {
            "detection_time": self.detection_time_history.copy(),
            "kappa": self.kappa_history.copy(),
            "success_rate": self.success_rate_history.copy(),
            "parameter_diversity": self.parameter_diversity_history.copy()
        }
    
    def reset(self) -> None:
        """Reset the convergence detector"""
        self.detection_time_history.clear()
        self.kappa_history.clear()
        self.success_rate_history.clear()
        self.parameter_diversity_history.clear()


class MultiPhaseConvergenceManager:
    """Manages convergence across multiple experiment phases"""
    
    def __init__(self):
        self.phase_detectors: Dict[str, AdvancedConvergenceDetector] = {}
        self.phase_transition_history: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(__name__)
    
    def add_phase(self, phase_name: str, detector: AdvancedConvergenceDetector) -> None:
        """Add a phase to convergence management"""
        self.phase_detectors[phase_name] = detector
    
    def check_phase_transition(self, current_phase: str, next_phase: str) -> Tuple[bool, List[str]]:
        """Check if transition to next phase is warranted"""
        if current_phase not in self.phase_detectors:
            return False, ["Current phase not found in convergence management"]
        
        detector = self.phase_detectors[current_phase]
        convergence_result = detector.check_convergence()
        
        should_transition = convergence_result.is_converged
        recommendations = convergence_result.recommendations
        
        # Log transition decision
        transition_record = {
            "timestamp": logging.getLogger().handlers[0].formatter.formatTime(
                logging.LogRecord(__name__, logging.INFO, "", 0, "", (), None)
            ) if logging.getLogger().handlers else "unknown",
            "from_phase": current_phase,
            "to_phase": next_phase,
            "should_transition": should_transition,
            "convergence_score": convergence_result.metrics.overall_convergence_score,
            "confidence": convergence_result.confidence,
            "recommendations": recommendations
        }
        
        self.phase_transition_history.append(transition_record)
        
        if should_transition:
            self.logger.info(f"Phase transition recommended: {current_phase} -> {next_phase}")
            self.logger.info(f"Convergence score: {convergence_result.metrics.overall_convergence_score:.3f}")
        else:
            self.logger.info(f"Phase transition not yet recommended for {current_phase}")
        
        return should_transition, recommendations
    
    def get_transition_history(self) -> List[Dict[str, Any]]:
        """Get history of phase transition decisions"""
        return self.phase_transition_history.copy()


def main():
    """Demo and testing for convergence detection"""
    import random
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create detector
    detector = AdvancedConvergenceDetector(window_size=5, min_samples=10)
    
    # Simulate improving results (exponential decay)
    print("Testing convergence detection with simulated data...")
    
    true_value = 1000.0
    improvement_rate = 0.1
    
    for i in range(50):
        # Simulate results with exponential improvement
        current_value = true_value * math.exp(-improvement_rate * i)
        noise = random.gauss(0, current_value * 0.1)  # 10% noise
        
        simulated_results = [{
            "simulation_success": True,
            "t5sigma_s": max(current_value + noise, 1.0),
            "kappa": [current_value * 1e10]  # Arbitrary scaling
        } for _ in range(3)]  # 3 results per iteration
        
        detector.add_results(simulated_results)
        
        if i >= 10:  # Start checking after some data
            result = detector.check_convergence()
            print(f"Iteration {i:2d}: Score={result.metrics.overall_convergence_score:.3f}, "
                  f"Converged={result.is_converged}, Level={result.convergence_level}")
            
            if result.is_converged:
                print("Convergence achieved! Stopping simulation.")
                break
    
    # Print final recommendations
    final_result = detector.check_convergence()
    print("\nFinal Convergence Analysis:")
    print(f"Overall Score: {final_result.metrics.overall_convergence_score:.3f}")
    print(f"Convergence Level: {final_result.convergence_level}")
    print(f"Confidence: {final_result.confidence:.3f}")
    print("Recommendations:")
    for rec in final_result.recommendations:
        print(f"  - {rec}")


if __name__ == "__main__":
    main()
