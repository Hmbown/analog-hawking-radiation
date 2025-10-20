#!/usr/bin/env python3
"""
Synthesis Engine for Analog Hawking Radiation Experiments

Performs cross-phase result aggregation, trend identification, pattern recognition,
optimization trajectory synthesis, and statistical meta-analysis for comprehensive
experiment understanding and insight generation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats, optimize
from scipy.cluster import hierarchy
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Add project paths to Python path
import sys
# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.result_aggregator import ResultAggregator, ExperimentAggregate
from scripts.validation.validation_framework import ValidationFramework


@dataclass
class TrendAnalysis:
    """Analysis of trends across experimental phases"""
    phase_trends: Dict[str, Dict[str, Any]]
    parameter_evolution: Dict[str, List[float]]
    performance_trajectory: Dict[str, List[float]]
    convergence_patterns: List[str]
    optimization_efficiency: float
    improvement_rates: Dict[str, float]


@dataclass
class PatternRecognition:
    """Patterns identified in experiment results"""
    clustering_results: Dict[str, Any]
    parameter_interactions: List[Tuple[str, str, float]]
    optimal_regions: List[Dict[str, Any]]
    anomaly_detections: List[Dict[str, Any]]
    phase_transition_patterns: List[Dict[str, Any]]


@dataclass
class MetaAnalysis:
    """Statistical meta-analysis across phases"""
    combined_statistics: Dict[str, Any]
    effect_sizes: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]
    heterogeneity_measures: Dict[str, float]
    publication_bias_assessment: Dict[str, Any]


@dataclass
class SynthesisReport:
    """Comprehensive synthesis of experiment insights"""
    experiment_id: str
    trend_analysis: TrendAnalysis
    pattern_recognition: PatternRecognition
    meta_analysis: MetaAnalysis
    key_insights: List[str]
    recommendations: List[str]
    future_directions: List[str]
    synthesis_metadata: Dict[str, Any]


class SynthesisEngine:
    """Advanced synthesis engine for multi-phase experiment analysis"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.synthesis_dir = self.experiment_dir / "synthesis"
        self.synthesis_dir.mkdir(parents=True, exist_ok=True)
        
        # Integration components
        self.aggregator = ResultAggregator(experiment_id)
        self.validator = ValidationFramework(experiment_id)
        
        # Analysis settings
        self.analysis_settings = {
            "clustering_method": "hierarchical",
            "trend_detection_threshold": 0.1,
            "pattern_significance_level": 0.05,
            "meta_analysis_confidence": 0.95,
            "anomaly_detection_sigma": 3.0
        }
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Load experiment data
        self.aggregate: Optional[ExperimentAggregate] = None
        self.experiment_manifest: Optional[Dict[str, Any]] = None
        self.phase_results: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger.info(f"Initialized synthesis engine for experiment {experiment_id}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'synthesis_engine.log'),
                logging.StreamHandler()
            ]
        )
    
    def load_experiment_data(self) -> bool:
        """Load all experiment data for synthesis"""
        try:
            # Load result aggregation
            if not self.aggregator.load_experiment_data():
                self.logger.error("Failed to load experiment data")
                return False
            
            self.aggregate = self.aggregator.aggregate_results()
            self.phase_results = self.aggregator.results
            
            # Load manifest
            manifest_file = self.experiment_dir / "experiment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, 'r') as f:
                    self.experiment_manifest = json.load(f)
            
            self.logger.info("Successfully loaded experiment data for synthesis")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment data: {e}")
            return False
    
    def perform_comprehensive_synthesis(self) -> SynthesisReport:
        """Perform comprehensive synthesis analysis"""
        if not self.aggregate:
            self.load_experiment_data()
        
        self.logger.info("Starting comprehensive synthesis analysis")
        
        # Perform all analysis components
        trend_analysis = self._analyze_trends()
        pattern_recognition = self._recognize_patterns()
        meta_analysis = self._perform_meta_analysis()
        
        # Generate insights and recommendations
        key_insights = self._extract_key_insights(trend_analysis, pattern_recognition, meta_analysis)
        recommendations = self._generate_recommendations(trend_analysis, pattern_recognition, meta_analysis)
        future_directions = self._suggest_future_directions(trend_analysis, pattern_recognition, meta_analysis)
        
        report = SynthesisReport(
            experiment_id=self.experiment_id,
            trend_analysis=trend_analysis,
            pattern_recognition=pattern_recognition,
            meta_analysis=meta_analysis,
            key_insights=key_insights,
            recommendations=recommendations,
            future_directions=future_directions,
            synthesis_metadata=self._generate_synthesis_metadata()
        )
        
        # Save synthesis report
        self._save_synthesis_report(report)
        
        self.logger.info("Completed comprehensive synthesis analysis")
        return report
    
    def _analyze_trends(self) -> TrendAnalysis:
        """Analyze trends across experimental phases"""
        self.logger.info("Analyzing trends across phases")
        
        phase_trends = self._calculate_phase_trends()
        parameter_evolution = self._track_parameter_evolution()
        performance_trajectory = self._analyze_performance_trajectory()
        convergence_patterns = self._identify_convergence_patterns()
        optimization_efficiency = self._calculate_optimization_efficiency()
        improvement_rates = self._calculate_improvement_rates()
        
        return TrendAnalysis(
            phase_trends=phase_trends,
            parameter_evolution=parameter_evolution,
            performance_trajectory=performance_trajectory,
            convergence_patterns=convergence_patterns,
            optimization_efficiency=optimization_efficiency,
            improvement_rates=improvement_rates
        )
    
    def _calculate_phase_trends(self) -> Dict[str, Dict[str, Any]]:
        """Calculate trends for each phase"""
        trends = {}
        
        if not self.aggregate:
            return trends
        
        for phase_name, phase_summary in self.aggregate.phase_summary.items():
            phase_trend = {
                "detection_time_trend": self._analyze_metric_trend(phase_name, "detection_time"),
                "kappa_trend": self._analyze_metric_trend(phase_name, "kappa"),
                "success_rate_trend": self._analyze_success_trend(phase_name),
                "parameter_stability": self._assess_parameter_stability(phase_name),
                "convergence_speed": self._calculate_convergence_speed(phase_name)
            }
            trends[phase_name] = phase_trend
        
        return trends
    
    def _analyze_metric_trend(self, phase_name: str, metric: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric in a phase"""
        results = self.phase_results.get(phase_name, [])
        if not results:
            return {"error": "No data available"}
        
        # Extract metric values over time
        metric_values = []
        for result in results:
            if result.get("simulation_success"):
                if metric == "detection_time" and result.get("t5sigma_s"):
                    metric_values.append(result["t5sigma_s"])
                elif metric == "kappa" and result.get("kappa"):
                    kappa_list = result["kappa"]
                    if kappa_list:
                        metric_values.append(max(kappa_list))
        
        if len(metric_values) < 5:
            return {"error": "Insufficient data for trend analysis"}
        
        # Calculate trend
        x = np.arange(len(metric_values))
        if metric == "detection_time":
            y = np.log10(metric_values)  # Log scale for detection time
        else:
            y = np.log10(metric_values)  # Log scale for kappa
        
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            
            return {
                "trend_direction": "improving" if slope < 0 else "deteriorating",
                "trend_magnitude": abs(slope),
                "correlation_strength": abs(r_value),
                "statistical_significance": p_value,
                "confidence_interval": (slope - 1.96*std_err, slope + 1.96*std_err)
            }
        except:
            return {"error": "Trend calculation failed"}
    
    def _analyze_success_trend(self, phase_name: str) -> Dict[str, Any]:
        """Analyze success rate trend in a phase"""
        results = self.phase_results.get(phase_name, [])
        if not results:
            return {"error": "No data available"}
        
        # Calculate running success rate
        window_size = min(10, len(results) // 5)
        if window_size < 3:
            return {"error": "Insufficient data for success trend"}
        
        success_rates = []
        for i in range(0, len(results) - window_size + 1, window_size):
            window = results[i:i + window_size]
            successful = sum(1 for r in window if r.get("simulation_success"))
            success_rates.append(successful / len(window))
        
        if len(success_rates) < 3:
            return {"error": "Insufficient windows for trend analysis"}
        
        # Calculate trend
        x = np.arange(len(success_rates))
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, success_rates)
            
            return {
                "trend_direction": "improving" if slope > 0 else "deteriorating",
                "trend_magnitude": abs(slope),
                "correlation_strength": abs(r_value),
                "statistical_significance": p_value,
                "final_success_rate": success_rates[-1] if success_rates else 0.0
            }
        except:
            return {"error": "Success trend calculation failed"}
    
    def _assess_parameter_stability(self, phase_name: str) -> Dict[str, Any]:
        """Assess parameter stability in a phase"""
        results = self.phase_results.get(phase_name, [])
        if not results:
            return {"error": "No data available"}
        
        # Extract parameter values from successful simulations
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        param_values = {param: [] for param in key_params}
        
        for result in results:
            if result.get("simulation_success"):
                params = result.get("parameters_used", {})
                for param in key_params:
                    value = params.get(param)
                    if value is not None:
                        param_values[param].append(value)
        
        # Calculate coefficient of variation for each parameter
        stability_metrics = {}
        for param, values in param_values.items():
            if len(values) > 5:
                if param in ["laser_intensity", "plasma_density"]:
                    # Use log scale for large dynamic ranges
                    log_values = np.log10(values)
                    cv = np.std(log_values) / np.mean(log_values)
                else:
                    cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0.0
                
                stability_metrics[param] = {
                    "coefficient_of_variation": cv,
                    "stability_level": "high" if cv < 0.1 else "medium" if cv < 0.3 else "low",
                    "sample_size": len(values)
                }
        
        return stability_metrics
    
    def _calculate_convergence_speed(self, phase_name: str) -> Dict[str, Any]:
        """Calculate convergence speed for a phase"""
        results = self.phase_results.get(phase_name, [])
        if not results:
            return {"error": "No data available"}
        
        # Find when best result was achieved
        best_detection_time = float('inf')
        convergence_iteration = None
        
        for i, result in enumerate(results):
            if result.get("simulation_success") and result.get("t5sigma_s"):
                if result["t5sigma_s"] < best_detection_time:
                    best_detection_time = result["t5sigma_s"]
                    convergence_iteration = i
        
        if convergence_iteration is None:
            return {"error": "No successful simulations"}
        
        convergence_speed = convergence_iteration / len(results) if len(results) > 0 else 1.0
        
        return {
            "convergence_iteration": convergence_iteration,
            "total_iterations": len(results),
            "convergence_speed": 1.0 - convergence_speed,  # Higher is faster
            "convergence_efficiency": convergence_iteration / len(results) if len(results) > 0 else 1.0
        }
    
    def _track_parameter_evolution(self) -> Dict[str, List[float]]:
        """Track evolution of key parameters across phases"""
        evolution = {}
        
        if not self.aggregate:
            return evolution
        
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        
        for param in key_params:
            param_values = []
            for phase_name in self.aggregate.phase_summary.keys():
                # Extract parameter values from best result in each phase
                best_result = self._find_best_result_in_phase(phase_name)
                if best_result and best_result.get("parameters_used"):
                    value = best_result["parameters_used"].get(param)
                    if value is not None:
                        param_values.append(value)
            
            if param_values:
                evolution[param] = param_values
        
        return evolution
    
    def _find_best_result_in_phase(self, phase_name: str) -> Optional[Dict[str, Any]]:
        """Find the best result in a phase based on detection time"""
        results = self.phase_results.get(phase_name, [])
        if not results:
            return None
        
        best_result = None
        best_detection_time = float('inf')
        
        for result in results:
            if result.get("simulation_success") and result.get("t5sigma_s"):
                if result["t5sigma_s"] < best_detection_time:
                    best_detection_time = result["t5sigma_s"]
                    best_result = result
        
        return best_result
    
    def _analyze_performance_trajectory(self) -> Dict[str, List[float]]:
        """Analyze performance trajectory across phases"""
        trajectory = {}
        
        if not self.aggregate:
            return trajectory
        
        # Detection time trajectory
        detection_times = []
        for phase_name in self.aggregate.phase_summary.keys():
            best_time = self.aggregate.phase_summary[phase_name].get("best_detection_time")
            if best_time:
                detection_times.append(best_time)
        
        if detection_times:
            trajectory["detection_time"] = detection_times
        
        # Success rate trajectory
        success_rates = []
        for phase_name in self.aggregate.phase_summary.keys():
            success_rate = self.aggregate.phase_summary[phase_name].get("success_rate")
            if success_rate is not None:
                success_rates.append(success_rate)
        
        if success_rates:
            trajectory["success_rate"] = success_rates
        
        # Kappa trajectory
        kappa_values = []
        for phase_name in self.aggregate.phase_summary.keys():
            best_kappa = self.aggregate.phase_summary[phase_name].get("best_kappa")
            if best_kappa:
                kappa_values.append(best_kappa)
        
        if kappa_values:
            trajectory["kappa"] = kappa_values
        
        return trajectory
    
    def _identify_convergence_patterns(self) -> List[str]:
        """Identify patterns in convergence behavior"""
        patterns = []
        
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return patterns
        
        # Analyze convergence across phases
        detection_times = []
        for phase_name in self.aggregate.phase_summary.keys():
            best_time = self.aggregate.phase_summary[phase_name].get("best_detection_time")
            if best_time:
                detection_times.append(best_time)
        
        if len(detection_times) >= 3:
            # Check for monotonic improvement
            is_monotonic = all(detection_times[i] >= detection_times[i+1] for i in range(len(detection_times)-1))
            if is_monotonic:
                patterns.append("monotonic_improvement")
            
            # Check for convergence plateau
            improvements = [detection_times[i] - detection_times[i+1] for i in range(len(detection_times)-1)]
            if len(improvements) >= 2 and improvements[-1] / improvements[0] < 0.1:
                patterns.append("convergence_plateau")
            
            # Check for optimization saturation
            if detection_times[-1] < 1e3:  # Very fast detection times
                patterns.append("optimization_saturation")
        
        return patterns
    
    def _calculate_optimization_efficiency(self) -> float:
        """Calculate overall optimization efficiency"""
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return 0.0
        
        phases = list(self.aggregate.phase_summary.keys())
        initial_performance = self.aggregate.phase_summary[phases[0]].get("best_detection_time")
        final_performance = self.aggregate.phase_summary[phases[-1]].get("best_detection_time")
        
        if not initial_performance or not final_performance:
            return 0.0
        
        # Efficiency: improvement per phase
        improvement = (initial_performance - final_performance) / initial_performance
        efficiency = improvement / len(phases)
        
        return max(efficiency, 0.0)
    
    def _calculate_improvement_rates(self) -> Dict[str, float]:
        """Calculate improvement rates for key metrics"""
        improvement_rates = {}
        
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return improvement_rates
        
        phases = list(self.aggregate.phase_summary.keys())
        
        # Detection time improvement rate
        detection_times = []
        for phase in phases:
            best_time = self.aggregate.phase_summary[phase].get("best_detection_time")
            if best_time:
                detection_times.append(best_time)
        
        if len(detection_times) >= 2:
            total_improvement = (detection_times[0] - detection_times[-1]) / detection_times[0]
            improvement_rates["detection_time"] = total_improvement / (len(detection_times) - 1)
        
        # Success rate improvement rate
        success_rates = []
        for phase in phases:
            success_rate = self.aggregate.phase_summary[phase].get("success_rate")
            if success_rate is not None:
                success_rates.append(success_rate)
        
        if len(success_rates) >= 2:
            total_improvement = success_rates[-1] - success_rates[0]
            improvement_rates["success_rate"] = total_improvement / (len(success_rates) - 1)
        
        return improvement_rates
    
    def _recognize_patterns(self) -> PatternRecognition:
        """Recognize patterns in experiment results"""
        self.logger.info("Recognizing patterns in experiment data")
        
        clustering_results = self._perform_clustering_analysis()
        parameter_interactions = self._analyze_parameter_interactions()
        optimal_regions = self._identify_optimal_regions()
        anomaly_detections = self._detect_anomalies()
        phase_transition_patterns = self._analyze_phase_transitions()
        
        return PatternRecognition(
            clustering_results=clustering_results,
            parameter_interactions=parameter_interactions,
            optimal_regions=optimal_regions,
            anomaly_detections=anomaly_detections,
            phase_transition_patterns=phase_transition_patterns
        )
    
    def _perform_clustering_analysis(self) -> Dict[str, Any]:
        """Perform clustering analysis on successful simulations"""
        successful_results = self._extract_all_successful_results()
        if len(successful_results) < 10:
            return {"error": "Insufficient data for clustering"}
        
        # Extract features for clustering
        features = []
        for result in successful_results:
            feature_vector = self._extract_feature_vector(result)
            if feature_vector:
                features.append(feature_vector)
        
        if len(features) < 10:
            return {"error": "Insufficient feature vectors"}
        
        features_array = np.array(features)
        
        # Perform hierarchical clustering
        try:
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            # Perform clustering
            linkage_matrix = hierarchy.linkage(features_scaled, method='ward')
            clusters = hierarchy.fcluster(linkage_matrix, t=3, criterion='maxclust')
            
            # Analyze clusters
            cluster_analysis = {}
            for cluster_id in np.unique(clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                cluster_results = [successful_results[i] for i in cluster_indices]
                
                # Calculate cluster statistics
                detection_times = [r.get("t5sigma_s", 0) for r in cluster_results if r.get("t5sigma_s")]
                kappa_values = []
                for r in cluster_results:
                    kappa_list = r.get("kappa", [])
                    if kappa_list:
                        kappa_values.append(max(kappa_list))
                
                cluster_analysis[cluster_id] = {
                    "size": len(cluster_indices),
                    "avg_detection_time": np.mean(detection_times) if detection_times else None,
                    "avg_kappa": np.mean(kappa_values) if kappa_values else None,
                    "success_rate": len(cluster_indices) / len(successful_results)
                }
            
            return {
                "method": "hierarchical",
                "number_of_clusters": len(np.unique(clusters)),
                "cluster_sizes": [np.sum(clusters == i) for i in np.unique(clusters)],
                "cluster_analysis": cluster_analysis,
                "silhouette_score": self._calculate_silhouette_score(features_scaled, clusters)
            }
        except Exception as e:
            return {"error": f"Clustering failed: {e}"}
    
    def _extract_feature_vector(self, result: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector from result for clustering"""
        params = result.get("parameters_used", {})
        feature_vector = []
        
        # Key parameters
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        for param in key_params:
            value = params.get(param)
            if value is None:
                return None
            
            if param in ["laser_intensity", "plasma_density"]:
                feature_vector.append(np.log10(value))
            else:
                feature_vector.append(value)
        
        # Performance metrics
        if result.get("t5sigma_s"):
            feature_vector.append(np.log10(result["t5sigma_s"]))
        else:
            return None
        
        kappa_list = result.get("kappa", [])
        if kappa_list:
            feature_vector.append(np.log10(max(kappa_list)))
        else:
            feature_vector.append(0.0)  # Default value
        
        return feature_vector
    
    def _calculate_silhouette_score(self, features: np.ndarray, clusters: np.ndarray) -> float:
        """Calculate silhouette score for clustering quality"""
        try:
            from sklearn.metrics import silhouette_score
            return silhouette_score(features, clusters)
        except:
            return 0.0
    
    def _analyze_parameter_interactions(self) -> List[Tuple[str, str, float]]:
        """Analyze interactions between parameters"""
        successful_results = self._extract_all_successful_results()
        if len(successful_results) < 10:
            return []
        
        # Extract parameter pairs and their joint effect on detection time
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        interactions = []
        
        for i, param1 in enumerate(key_params):
            for j, param2 in enumerate(key_params[i+1:], i+1):
                # Calculate interaction effect
                interaction_strength = self._calculate_interaction_strength(
                    successful_results, param1, param2
                )
                if interaction_strength > 0.1:  # Significant interaction
                    interactions.append((param1, param2, interaction_strength))
        
        # Sort by interaction strength
        interactions.sort(key=lambda x: x[2], reverse=True)
        return interactions
    
    def _calculate_interaction_strength(self, results: List[Dict[str, Any]], param1: str, param2: str) -> float:
        """Calculate interaction strength between two parameters"""
        param1_values = []
        param2_values = []
        detection_times = []
        
        for result in results:
            params = result.get("parameters_used", {})
            detection_time = result.get("t5sigma_s")
            
            if (params.get(param1) is not None and 
                params.get(param2) is not None and 
                detection_time is not None):
                
                # Use log scale for appropriate parameters
                if param1 in ["laser_intensity", "plasma_density"]:
                    param1_values.append(np.log10(params[param1]))
                else:
                    param1_values.append(params[param1])
                
                if param2 in ["laser_intensity", "plasma_density"]:
                    param2_values.append(np.log10(params[param2]))
                else:
                    param2_values.append(params[param2])
                
                detection_times.append(np.log10(detection_time))
        
        if len(param1_values) < 10:
            return 0.0
        
        # Calculate interaction using multiple regression
        try:
            X = np.column_stack([param1_values, param2_values, np.multiply(param1_values, param2_values)])
            X = np.column_stack([np.ones(len(X)), X])  # Add intercept
            
            beta = np.linalg.lstsq(X, detection_times, rcond=None)[0]
            interaction_coef = beta[3]  # Interaction term coefficient
            
            return abs(interaction_coef)
        except:
            return 0.0
    
    def _identify_optimal_regions(self) -> List[Dict[str, Any]]:
        """Identify optimal parameter regions"""
        successful_results = self._extract_all_successful_results()
        if len(successful_results) < 10:
            return []
        
        # Group results by performance
        excellent_results = [r for r in successful_results if r.get("t5sigma_s", float('inf')) < 1e4]
        good_results = [r for r in successful_results if 1e4 <= r.get("t5sigma_s", float('inf')) < 1e6]
        
        optimal_regions = []
        
        if excellent_results:
            excellent_region = self._characterize_parameter_region(excellent_results, "excellent")
            optimal_regions.append(excellent_region)
        
        if good_results:
            good_region = self._characterize_parameter_region(good_results, "good")
            optimal_regions.append(good_region)
        
        return optimal_regions
    
    def _characterize_parameter_region(self, results: List[Dict[str, Any]], performance_level: str) -> Dict[str, Any]:
        """Characterize a parameter region"""
        key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
        characterization = {
            "performance_level": performance_level,
            "sample_size": len(results),
            "parameter_ranges": {},
            "performance_metrics": {}
        }
        
        # Calculate parameter ranges
        for param in key_params:
            values = []
            for result in results:
                value = result.get("parameters_used", {}).get(param)
                if value is not None:
                    values.append(value)
            
            if values:
                characterization["parameter_ranges"][param] = {
                    "min": np.min(values),
                    "max": np.max(values),
                    "mean": np.mean(values),
                    "median": np.median(values)
                }
        
        # Calculate performance metrics
        detection_times = [r.get("t5sigma_s") for r in results if r.get("t5sigma_s")]
        kappa_values = []
        for r in results:
            kappa_list = r.get("kappa", [])
            if kappa_list:
                kappa_values.append(max(kappa_list))
        
        if detection_times:
            characterization["performance_metrics"]["detection_time"] = {
                "min": np.min(detection_times),
                "max": np.max(detection_times),
                "mean": np.mean(detection_times),
                "median": np.median(detection_times)
            }
        
        if kappa_values:
            characterization["performance_metrics"]["kappa"] = {
                "min": np.min(kappa_values),
                "max": np.max(kappa_values),
                "mean": np.mean(kappa_values),
                "median": np.median(kappa_values)
            }
        
        return characterization
    
    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalous results"""
        successful_results = self._extract_all_successful_results()
        if len(successful_results) < 10:
            return []
        
        anomalies = []
        
        # Detect outliers in detection times
        detection_times = [r.get("t5sigma_s") for r in successful_results if r.get("t5sigma_s")]
        if detection_times:
            Q1 = np.percentile(detection_times, 25)
            Q3 = np.percentile(detection_times, 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            for result in successful_results:
                detection_time = result.get("t5sigma_s")
                if detection_time and (detection_time < lower_bound or detection_time > upper_bound):
                    anomalies.append({
                        "type": "detection_time_outlier",
                        "detection_time": detection_time,
                        "bounds": [lower_bound, upper_bound],
                        "parameters": result.get("parameters_used", {})
                    })
        
        # Detect parameter combinations that rarely succeed
        param_combinations = {}
        for result in successful_results:
            params = result.get("parameters_used", {})
            param_key = tuple(sorted(params.items()))
            if param_key not in param_combinations:
                param_combinations[param_key] = 0
            param_combinations[param_key] += 1
        
        rare_combinations = [key for key, count in param_combinations.items() if count == 1]
        for combination in rare_combinations:
            # Find the result with this combination
            for result in successful_results:
                params = result.get("parameters_used", {})
                if tuple(sorted(params.items())) == combination:
                    anomalies.append({
                        "type": "rare_parameter_combination",
                        "parameters": params,
                        "detection_time": result.get("t5sigma_s"),
                        "occurrences": 1
                    })
                    break
        
        return anomalies
    
    def _analyze_phase_transitions(self) -> List[Dict[str, Any]]:
        """Analyze patterns in phase transitions"""
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return []
        
        phase_transitions = []
        phases = list(self.aggregate.phase_summary.keys())
        
        for i in range(len(phases) - 1):
            phase1 = phases[i]
            phase2 = phases[i + 1]
            
            transition = {
                "from_phase": phase1,
                "to_phase": phase2,
                "performance_change": self._calculate_performance_change(phase1, phase2),
                "parameter_shift": self._analyze_parameter_shift(phase1, phase2),
                "success_rate_change": self._calculate_success_rate_change(phase1, phase2)
            }
            
            phase_transitions.append(transition)
        
        return phase_transitions
    
    def _calculate_performance_change(self, phase1: str, phase2: str) -> float:
        """Calculate performance change between phases"""
        best1 = self.aggregate.phase_summary[phase1].get("best_detection_time")
        best2 = self.aggregate.phase_summary[phase2].get("best_detection_time")
        
        if best1 and best2 and best1 > 0:
            return (best1 - best2) / best1  # Improvement ratio
        
        return 0.0
    
    def _analyze_parameter_shift(self, phase1: str, phase2: str) -> Dict[str, Any]:
        """Analyze parameter shifts between phases"""
        best1 = self._find_best_result_in_phase(phase1)
        best2 = self._find_best_result_in_phase(phase2)
        
        if not best1 or not best2:
            return {"error": "Missing best results"}
        
        params1 = best1.get("parameters_used", {})
        params2 = best2.get("parameters_used", {})
        
        shifts = {}
        for param in ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]:
            value1 = params1.get(param)
            value2 = params2.get(param)
            
            if value1 is not None and value2 is not None:
                if param in ["laser_intensity", "plasma_density"]:
                    # Log scale for relative change
                    shift = (np.log10(value2) - np.log10(value1)) / np.log10(value1)
                else:
                    shift = (value2 - value1) / value1 if value1 != 0 else 0.0
                
                shifts[param] = shift
        
        return shifts
    
    def _calculate_success_rate_change(self, phase1: str, phase2: str) -> float:
        """Calculate success rate change between phases"""
        rate1 = self.aggregate.phase_summary[phase1].get("success_rate", 0)
        rate2 = self.aggregate.phase_summary[phase2].get("success_rate", 0)
        
        return rate2 - rate1
    
    def _perform_meta_analysis(self) -> MetaAnalysis:
        """Perform statistical meta-analysis across phases"""
        self.logger.info("Performing meta-analysis across phases")
        
        combined_statistics = self._combine_phase_statistics()
        effect_sizes = self._calculate_effect_sizes()
        confidence_intervals = self._calculate_confidence_intervals()
        heterogeneity_measures = self._assess_heterogeneity()
        publication_bias_assessment = self._assess_publication_bias()
        
        return MetaAnalysis(
            combined_statistics=combined_statistics,
            effect_sizes=effect_sizes,
            confidence_intervals=confidence_intervals,
            heterogeneity_measures=heterogeneity_measures,
            publication_bias_assessment=publication_bias_assessment
        )
    
    def _combine_phase_statistics(self) -> Dict[str, Any]:
        """Combine statistics across phases"""
        if not self.aggregate:
            return {}
        
        # Combine detection times across all phases
        all_detection_times = self._extract_all_detection_times()
        all_kappa_values = self._extract_all_kappa_values()
        
        combined_stats = {
            "total_simulations": self.aggregate.total_simulations,
            "successful_simulations": self.aggregate.successful_simulations,
            "overall_success_rate": self.aggregate.success_rate
        }
        
        if all_detection_times:
            combined_stats["detection_time"] = {
                "mean": np.mean(all_detection_times),
                "median": np.median(all_detection_times),
                "std": np.std(all_detection_times),
                "min": np.min(all_detection_times),
                "max": np.max(all_detection_times),
                "q1": np.percentile(all_detection_times, 25),
                "q3": np.percentile(all_detection_times, 75)
            }
        
        if all_kappa_values:
            combined_stats["kappa"] = {
                "mean": np.mean(all_kappa_values),
                "median": np.median(all_kappa_values),
                "std": np.std(all_kappa_values),
                "min": np.min(all_kappa_values),
                "max": np.max(all_kappa_values),
                "q1": np.percentile(all_kappa_values, 25),
                "q3": np.percentile(all_kappa_values, 75)
            }
        
        return combined_stats
    
    def _calculate_effect_sizes(self) -> Dict[str, float]:
        """Calculate effect sizes for key metrics"""
        effect_sizes = {}
        
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return effect_sizes
        
        phases = list(self.aggregate.phase_summary.keys())
        
        # Effect size for detection time improvement
        initial_times = [r.get("t5sigma_s") for r in self.phase_results[phases[0]] if r.get("t5sigma_s")]
        final_times = [r.get("t5sigma_s") for r in self.phase_results[phases[-1]] if r.get("t5sigma_s")]
        
        if initial_times and final_times:
            # Cohen's d for effect size
            d = (np.mean(initial_times) - np.mean(final_times)) / np.sqrt(
                (np.std(initial_times)**2 + np.std(final_times)**2) / 2
            )
            effect_sizes["detection_time_improvement"] = d
        
        # Effect size for success rate improvement
        initial_success = self.aggregate.phase_summary[phases[0]].get("success_rate", 0)
        final_success = self.aggregate.phase_summary[phases[-1]].get("success_rate", 0)
        
        if initial_success > 0 and final_success > 0:
            # Log odds ratio for effect size
            odds_ratio = (final_success / (1 - final_success)) / (initial_success / (1 - initial_success))
            effect_sizes["success_rate_improvement"] = np.log(odds_ratio)
        
        return effect_sizes
    
    def _calculate_confidence_intervals(self) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for key metrics"""
        confidence_intervals = {}
        
        all_detection_times = self._extract_all_detection_times()
        if all_detection_times:
            mean = np.mean(all_detection_times)
            std_err = stats.sem(all_detection_times)
            ci = stats.t.interval(0.95, len(all_detection_times)-1, loc=mean, scale=std_err)
            confidence_intervals["detection_time_mean"] = ci
        
        all_kappa_values = self._extract_all_kappa_values()
        if all_kappa_values:
            mean = np.mean(all_kappa_values)
            std_err = stats.sem(all_kappa_values)
            ci = stats.t.interval(0.95, len(all_kappa_values)-1, loc=mean, scale=std_err)
            confidence_intervals["kappa_mean"] = ci
        
        return confidence_intervals
    
    def _assess_heterogeneity(self) -> Dict[str, float]:
        """Assess heterogeneity across phases"""
        heterogeneity = {}
        
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return heterogeneity
        
        # Calculate I² statistic for detection times
        detection_time_means = []
        detection_time_vars = []
        sample_sizes = []
        
        for phase_name, phase_summary in self.aggregate.phase_summary.items():
            phase_times = [r.get("t5sigma_s") for r in self.phase_results[phase_name] if r.get("t5sigma_s")]
            if phase_times:
                detection_time_means.append(np.mean(phase_times))
                detection_time_vars.append(np.var(phase_times))
                sample_sizes.append(len(phase_times))
        
        if len(detection_time_means) >= 2:
            # Calculate Q statistic (Cochran's Q)
            overall_mean = np.average(detection_time_means, weights=sample_sizes)
            Q = sum(s * (m - overall_mean)**2 for s, m in zip(sample_sizes, detection_time_means))
            
            # Calculate I²
            df = len(detection_time_means) - 1
            if Q > df:
                I2 = (Q - df) / Q * 100
            else:
                I2 = 0
            
            heterogeneity["detection_time_I2"] = I2
        
        return heterogeneity
    
    def _assess_publication_bias(self) -> Dict[str, Any]:
        """Assess potential publication bias"""
        # This is a simplified assessment for internal use
        assessment = {
            "method": "funnel_plot_symmetry",
            "risk_level": "low",  # Internal analysis, no publication bias
            "notes": "Internal experiment analysis - publication bias not applicable"
        }
        
        return assessment
    
    def _extract_all_successful_results(self) -> List[Dict[str, Any]]:
        """Extract all successful results across phases"""
        successful_results = []
        
        for phase_results in self.phase_results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    successful_results.append(result)
        
        return successful_results
    
    def _extract_all_detection_times(self) -> List[float]:
        """Extract all detection times from results"""
        detection_times = []
        
        for phase_results in self.phase_results.values():
            for result in phase_results:
                if result.get("simulation_success") and result.get("t5sigma_s"):
                    detection_times.append(result["t5sigma_s"])
        
        return detection_times
    
    def _extract_all_kappa_values(self) -> List[float]:
        """Extract all kappa values from results"""
        kappa_values = []
        
        for phase_results in self.phase_results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        kappa_values.extend(kappa_list)
        
        return kappa_values
    
    def _extract_key_insights(self, trend_analysis: TrendAnalysis, 
                            pattern_recognition: PatternRecognition,
                            meta_analysis: MetaAnalysis) -> List[str]:
        """Extract key insights from synthesis analysis"""
        insights = []
        
        # Insights from trend analysis
        if trend_analysis.optimization_efficiency > 0.1:
            insights.append(f"High optimization efficiency ({trend_analysis.optimization_efficiency:.1%}) achieved through phased approach")
        
        if trend_analysis.improvement_rates.get("detection_time", 0) > 0.1:
            insights.append("Significant detection time improvement observed across optimization phases")
        
        if "monotonic_improvement" in trend_analysis.convergence_patterns:
            insights.append("Consistent monotonic improvement demonstrates effective optimization strategy")
        
        # Insights from pattern recognition
        if pattern_recognition.parameter_interactions:
            strongest_interaction = pattern_recognition.parameter_interactions[0]
            insights.append(f"Strong parameter interaction detected between {strongest_interaction[0]} and {strongest_interaction[1]}")
        
        if pattern_recognition.optimal_regions:
            excellent_regions = [r for r in pattern_recognition.optimal_regions if r["performance_level"] == "excellent"]
            if excellent_regions:
                insights.append(f"Identified {len(excellent_regions)} optimal parameter regions for rapid detection")
        
        if pattern_recognition.anomaly_detections:
            insights.append(f"Detected {len(pattern_recognition.anomaly_detections)} anomalous results requiring investigation")
        
        # Insights from meta-analysis
        if meta_analysis.effect_sizes.get("detection_time_improvement", 0) > 0.5:
            insights.append("Large effect size confirms substantial detection time improvement")
        
        return insights
    
    def _generate_recommendations(self, trend_analysis: TrendAnalysis,
                               pattern_recognition: PatternRecognition,
                               meta_analysis: MetaAnalysis) -> List[str]:
        """Generate recommendations based on synthesis analysis"""
        recommendations = []
        
        # Recommendations from trend analysis
        if trend_analysis.optimization_efficiency < 0.05:
            recommendations.append("Consider alternative optimization strategies to improve efficiency")
        
        if not trend_analysis.convergence_patterns:
            recommendations.append("Extend optimization phases to achieve convergence")
        
        # Recommendations from pattern recognition
        if pattern_recognition.parameter_interactions:
            recommendations.append("Leverage identified parameter interactions for more efficient optimization")
        
        if pattern_recognition.optimal_regions:
            recommendations.append("Focus future experiments on identified optimal parameter regions")
        
        if pattern_recognition.anomaly_detections:
            recommendations.append("Investigate anomalous results to understand failure modes")
        
        # General recommendations
        recommendations.extend([
            "Implement real-time synthesis during future experiments",
            "Use identified patterns to guide experimental design",
            "Validate synthesis insights with targeted follow-up experiments"
        ])
        
        return recommendations
    
    def _suggest_future_directions(self, trend_analysis: TrendAnalysis,
                                 pattern_recognition: PatternRecognition,
                                 meta_analysis: MetaAnalysis) -> List[str]:
        """Suggest future research directions"""
        directions = [
            "Extend parameter space exploration to higher-dimensional optimization",
            "Investigate identified parameter interactions with controlled experiments",
            "Develop adaptive optimization strategies based on synthesis patterns",
            "Explore machine learning approaches for pattern recognition and prediction",
            "Validate optimal parameter regions with experimental implementations"
        ]
        
        # Add data-driven directions
        if pattern_recognition.optimal_regions:
            directions.append("Systematically explore boundaries of optimal parameter regions")
        
        if trend_analysis.improvement_rates.get("detection_time", 0) > 0.2:
            directions.append("Apply successful optimization strategy to related physical systems")
        
        return directions
    
    def _generate_synthesis_metadata(self) -> Dict[str, Any]:
        """Generate metadata for synthesis report"""
        return {
            "experiment_id": self.experiment_id,
            "synthesis_timestamp": self._get_current_timestamp(),
            "analysis_methods": [
                "trend_analysis",
                "pattern_recognition", 
                "meta_analysis",
                "clustering_analysis",
                "parameter_interaction_analysis"
            ],
            "data_sources": list(self.phase_results.keys()),
            "settings_used": self.analysis_settings
        }
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _save_synthesis_report(self, report: SynthesisReport) -> None:
        """Save synthesis report to disk"""
        report_file = self.synthesis_dir / "synthesis_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        # Also save as text summary
        summary_file = self.synthesis_dir / "synthesis_summary.txt"
        self._write_synthesis_summary(report, summary_file)
        
        self.logger.info(f"Saved synthesis report to {report_file}")
    
    def _write_synthesis_summary(self, report: SynthesisReport, file_path: Path) -> None:
        """Write synthesis summary as formatted text"""
        with open(file_path, 'w') as f:
            f.write("SYNTHESIS SUMMARY REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment ID: {report.experiment_id}\n")
            f.write(f"Synthesis Timestamp: {report.synthesis_metadata.get('synthesis_timestamp', 'N/A')}\n\n")
            
            f.write("KEY INSIGHTS\n")
            f.write("-" * 30 + "\n")
            for insight in report.key_insights:
                f.write(f"• {insight}\n")
            f.write("\n")
            
            f.write("RECOMMENDATIONS\n")
            f.write("-" * 30 + "\n")
            for recommendation in report.recommendations:
                f.write(f"• {recommendation}\n")
            f.write("\n")
            
            f.write("FUTURE DIRECTIONS\n")
            f.write("-" * 30 + "\n")
            for direction in report.future_directions:
                f.write(f"• {direction}\n")


def main():
    """Main entry point for synthesis engine"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthesis Engine")
    parser.add_argument("experiment_id", help="Experiment ID to synthesize")
    parser.add_argument("--output", help="Output directory for synthesis reports")
    
    args = parser.parse_args()
    
    # Perform synthesis
    engine = SynthesisEngine(args.experiment_id)
    
    if not engine.load_experiment_data():
        print(f"Failed to load experiment data for {args.experiment_id}")
        return 1
    
    try:
        report = engine.perform_comprehensive_synthesis()
        print("Generated comprehensive synthesis report")
        print(f"Key insights: {len(report.key_insights)}")
        print(f"Recommendations: {len(report.recommendations)}")
        print(f"Synthesis saved to: {engine.synthesis_dir}")
        return 0
        
    except Exception as e:
        print(f"Synthesis failed: {e}")
        return 1


if __name__ == "__main__":
    main()
