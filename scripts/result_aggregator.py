#!/usr/bin/env python3
"""
Result Aggregation Framework for Analog Hawking Radiation Experiments

Aggregates, analyzes, and visualizes results from multi-phase experiments
with cross-phase correlation and automated reporting.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns  # type: ignore
except Exception:
    sns = None  # seaborn is optional; visualizations disabled if missing
from scipy import stats

# Add project paths to Python path
import sys
# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.analyze_significance import StatisticalAnalyzer


@dataclass
class CrossPhaseCorrelation:
    """Cross-phase correlation analysis results"""
    phase_pairs: List[Tuple[str, str]]
    correlation_coefficients: Dict[Tuple[str, str], float]
    p_values: Dict[Tuple[str, str], float]
    significant_correlations: List[Tuple[str, str, float, float]]


@dataclass
class ExperimentAggregate:
    """Aggregated experiment results across all phases"""
    experiment_id: str
    total_simulations: int
    successful_simulations: int
    success_rate: float
    best_detection_time: Optional[float]
    best_kappa: Optional[float]
    best_snr: Optional[float]
    phase_summary: Dict[str, Dict[str, Any]]
    parameter_sensitivity: Dict[str, float]
    cross_phase_correlation: CrossPhaseCorrelation
    statistical_significance: Dict[str, Any]


class ResultAggregator:
    """Aggregates and analyzes results from multi-phase experiments"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.results: Dict[str, List[Dict[str, Any]]] = {}
        self.manifest: Optional[Dict[str, Any]] = None
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'aggregation.log'),
                logging.StreamHandler()
            ]
        )
        if sns is None:
            logging.getLogger(__name__).info("Seaborn not installed; visualizations will be skipped.")
    
    def load_experiment_data(self) -> bool:
        """Load all experiment data from disk"""
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
    
    def aggregate_results(self) -> ExperimentAggregate:
        """Aggregate results from all phases"""
        if not self.results:
            self.load_experiment_data()
        
        # Calculate basic statistics
        total_simulations = sum(len(results) for results in self.results.values())
        successful_simulations = sum(
            sum(1 for r in results if r.get("simulation_success"))
            for results in self.results.values()
        )
        success_rate = successful_simulations / total_simulations if total_simulations > 0 else 0.0
        
        # Find best results
        best_detection_time, best_kappa, best_snr = self._find_best_results()
        
        # Generate phase summary
        phase_summary = self._generate_phase_summary()
        
        # Analyze parameter sensitivity
        parameter_sensitivity = self._analyze_parameter_sensitivity()
        
        # Cross-phase correlation
        cross_phase_correlation = self._analyze_cross_phase_correlation()
        
        # Statistical significance
        statistical_significance = self._analyze_statistical_significance()
        
        return ExperimentAggregate(
            experiment_id=self.experiment_id,
            total_simulations=total_simulations,
            successful_simulations=successful_simulations,
            success_rate=success_rate,
            best_detection_time=best_detection_time,
            best_kappa=best_kappa,
            best_snr=best_snr,
            phase_summary=phase_summary,
            parameter_sensitivity=parameter_sensitivity,
            cross_phase_correlation=cross_phase_correlation,
            statistical_significance=statistical_significance
        )
    
    def _find_best_results(self) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """Find the best results across all phases"""
        best_detection_time = None
        best_kappa = None
        best_snr = None
        
        for phase_results in self.results.values():
            for result in phase_results:
                if not result.get("simulation_success"):
                    continue
                
                # Check detection time
                t5sigma = result.get("t5sigma_s")
                if t5sigma is not None and t5sigma > 0:
                    if best_detection_time is None or t5sigma < best_detection_time:
                        best_detection_time = t5sigma
                
                # Check kappa
                kappa_list = result.get("kappa", [])
                if kappa_list:
                    max_kappa = max(kappa_list)
                    if best_kappa is None or max_kappa > best_kappa:
                        best_kappa = max_kappa
                
                # Check SNR
                T_sig = result.get("T_sig_K")
                if T_sig is not None and T_sig > 0:
                    snr = T_sig / 30.0  # Assuming T_sys = 30K
                    if best_snr is None or snr > best_snr:
                        best_snr = snr
        
        return best_detection_time, best_kappa, best_snr
    
    def _generate_phase_summary(self) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics for each phase"""
        phase_summary = {}
        
        for phase_name, results in self.results.items():
            successful_results = [r for r in results if r.get("simulation_success")]
            
            if not successful_results:
                phase_summary[phase_name] = {
                    "total_simulations": len(results),
                    "successful_simulations": 0,
                    "success_rate": 0.0,
                    "avg_detection_time": None,
                    "avg_kappa": None,
                    "best_detection_time": None,
                    "best_kappa": None
                }
                continue
            
            # Calculate statistics
            detection_times = [r.get("t5sigma_s") for r in successful_results if r.get("t5sigma_s") is not None]
            kappa_values = []
            for r in successful_results:
                kappa_list = r.get("kappa", [])
                if kappa_list:
                    kappa_values.append(max(kappa_list))
            
            phase_summary[phase_name] = {
                "total_simulations": len(results),
                "successful_simulations": len(successful_results),
                "success_rate": len(successful_results) / len(results),
                "avg_detection_time": np.mean(detection_times) if detection_times else None,
                "avg_kappa": np.mean(kappa_values) if kappa_values else None,
                "best_detection_time": min(detection_times) if detection_times else None,
                "best_kappa": max(kappa_values) if kappa_values else None,
                "detection_time_std": np.std(detection_times) if detection_times else None,
                "kappa_std": np.std(kappa_values) if kappa_values else None
            }
        
        return phase_summary
    
    def _analyze_parameter_sensitivity(self) -> Dict[str, float]:
        """Analyze sensitivity of results to parameter variations"""
        # Collect all successful results with parameters
        all_results = []
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success") and result.get("t5sigma_s") is not None:
                    all_results.append(result)
        
        if not all_results:
            return {}
        
        # Extract parameters and detection times
        parameters_data = []
        detection_times = []
        
        for result in all_results:
            params = result.get("parameters_used", {})
            detection_time = result.get("t5sigma_s")
            
            if detection_time is None:
                continue
            
            param_vector = []
            # Key parameters to analyze
            key_params = ["laser_intensity", "plasma_density", "temperature_constant", "magnetic_field"]
            
            for param_name in key_params:
                value = params.get(param_name)
                if value is not None:
                    # Use log scale for large dynamic ranges
                    if param_name in ["laser_intensity", "plasma_density"]:
                        param_vector.append(np.log10(value))
                    else:
                        param_vector.append(value)
                else:
                    param_vector.append(0.0)  # Default for missing values
            
            parameters_data.append(param_vector)
            detection_times.append(np.log10(detection_time))  # Log scale for detection time
        
        if len(parameters_data) < 10:  # Need sufficient data
            return {}
        
        # Calculate correlation coefficients
        param_names = ["log_intensity", "log_density", "temperature", "magnetic_field"]
        sensitivity = {}
        
        for i, param_name in enumerate(param_names):
            param_values = [vec[i] for vec in parameters_data]
            if len(set(param_values)) > 1:  # Check for variation
                corr_coef, p_value = stats.pearsonr(param_values, detection_times)
                sensitivity[param_name] = abs(corr_coef)  # Use absolute value for importance
        
        return sensitivity
    
    def _analyze_cross_phase_correlation(self) -> CrossPhaseCorrelation:
        """Analyze correlations between phases"""
        phase_names = list(self.results.keys())
        phase_pairs = []
        correlation_coefficients = {}
        p_values = {}
        significant_correlations = []
        
        # Compare each pair of phases
        for i, phase1 in enumerate(phase_names):
            for j, phase2 in enumerate(phase_names[i+1:], i+1):
                phase_pairs.append((phase1, phase2))
                
                # Extract common metrics for comparison
                metrics1 = self._extract_phase_metrics(phase1)
                metrics2 = self._extract_phase_metrics(phase2)
                
                if not metrics1 or not metrics2:
                    continue
                
                # Find common metrics
                common_metrics = set(metrics1.keys()) & set(metrics2.keys())
                
                if not common_metrics:
                    continue
                
                # Calculate average correlation across common metrics
                correlations = []
                p_vals = []
                
                for metric in common_metrics:
                    if len(metrics1[metric]) > 5 and len(metrics2[metric]) > 5:  # Need sufficient data
                        try:
                            corr, p_val = stats.pearsonr(metrics1[metric], metrics2[metric])
                            correlations.append(abs(corr))
                            p_vals.append(p_val)
                        except:
                            continue
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    avg_p_val = np.mean(p_vals)
                    correlation_coefficients[(phase1, phase2)] = avg_corr
                    p_values[(phase1, phase2)] = avg_p_val
                    
                    if avg_p_val < 0.05:  # Statistically significant
                        significant_correlations.append((phase1, phase2, avg_corr, avg_p_val))
        
        return CrossPhaseCorrelation(
            phase_pairs=phase_pairs,
            correlation_coefficients=correlation_coefficients,
            p_values=p_values,
            significant_correlations=significant_correlations
        )
    
    def _extract_phase_metrics(self, phase_name: str) -> Dict[str, List[float]]:
        """Extract key metrics from a phase's results"""
        results = self.results.get(phase_name, [])
        metrics = {
            "detection_time": [],
            "kappa": [],
            "signal_temperature": []
        }
        
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
                
                # Signal temperature
                T_sig = result.get("T_sig_K")
                if T_sig is not None:
                    metrics["signal_temperature"].append(T_sig)
        
        return metrics
    
    def _analyze_statistical_significance(self) -> Dict[str, Any]:
        """Analyze statistical significance of detection results"""
        # Collect all successful results
        all_results = []
        for phase_results in self.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    all_results.append(result)
        
        if not all_results:
            return {"error": "No successful results found"}
        
        # Use the existing StatisticalAnalyzer
        analyzer = StatisticalAnalyzer()
        enhanced_results = analyzer.calculate_signal_to_noise(all_results)
        
        # Calculate detection probabilities
        detection_stats_1h = analyzer.calculate_detection_probability(enhanced_results, 3600)
        detection_stats_1d = analyzer.calculate_detection_probability(enhanced_results, 86400)
        detection_stats_1w = analyzer.calculate_detection_probability(enhanced_results, 604800)
        
        # Scaling analysis
        scaling_analysis = analyzer.analyze_scaling_requirements(enhanced_results)
        
        return {
            "detection_probability_1h": detection_stats_1h,
            "detection_probability_1d": detection_stats_1d,
            "detection_probability_1w": detection_stats_1w,
            "scaling_analysis": scaling_analysis,
            "total_analyzed_results": len(enhanced_results)
        }
    
    def generate_visualizations(self, aggregate: ExperimentAggregate) -> None:
        """Generate comprehensive visualizations for the experiment"""
        viz_dir = self.experiment_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Phase progression plot
        self._plot_phase_progression(aggregate, viz_dir)
        
        # 2. Parameter sensitivity plot
        self._plot_parameter_sensitivity(aggregate, viz_dir)
        
        # 3. Cross-phase correlation heatmap
        self._plot_cross_phase_correlation(aggregate, viz_dir)
        
        # 4. Detection time distribution
        self._plot_detection_time_distribution(viz_dir)
        
        # 5. Success rate by phase
        self._plot_success_rates(aggregate, viz_dir)
        
        self.logger.info(f"Generated visualizations in {viz_dir}")
    
    def _plot_phase_progression(self, aggregate: ExperimentAggregate, output_dir: Path) -> None:
        """Plot progression of key metrics across phases"""
        phases = list(aggregate.phase_summary.keys())
        best_detection_times = []
        best_kappas = []
        success_rates = []
        
        for phase in phases:
            summary = aggregate.phase_summary[phase]
            best_detection_times.append(summary.get("best_detection_time") or np.nan)
            best_kappas.append(summary.get("best_kappa") or np.nan)
            success_rates.append(summary.get("success_rate") or 0.0)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Best detection time progression
        ax1.plot(range(len(phases)), best_detection_times, 'o-', linewidth=2, markersize=8)
        ax1.set_yscale('log')
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Best Detection Time (s)')
        ax1.set_title('Best Detection Time Progression')
        ax1.set_xticks(range(len(phases)))
        ax1.set_xticklabels([p.replace('phase_', '').replace('_', ' ').title() for p in phases])
        ax1.grid(True, alpha=0.3)
        
        # Best kappa progression
        ax2.plot(range(len(phases)), best_kappas, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.set_yscale('log')
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Best Surface Gravity κ (s⁻¹)')
        ax2.set_title('Best Surface Gravity Progression')
        ax2.set_xticks(range(len(phases)))
        ax2.set_xticklabels([p.replace('phase_', '').replace('_', ' ').title() for p in phases])
        ax2.grid(True, alpha=0.3)
        
        # Success rate progression
        ax3.bar(range(len(phases)), success_rates, color='green', alpha=0.7)
        ax3.set_xlabel('Phase')
        ax3.set_ylabel('Success Rate')
        ax3.set_title('Success Rate by Phase')
        ax3.set_xticks(range(len(phases)))
        ax3.set_xticklabels([p.replace('phase_', '').replace('_', ' ').title() for p in phases])
        ax3.grid(True, alpha=0.3)
        
        # Parameter sensitivity
        sensitivities = list(aggregate.parameter_sensitivity.values())
        param_names = list(aggregate.parameter_sensitivity.keys())
        if sensitivities:
            ax4.bar(range(len(sensitivities)), sensitivities, color='red', alpha=0.7)
            ax4.set_xlabel('Parameter')
            ax4.set_ylabel('Sensitivity (|correlation|)')
            ax4.set_title('Parameter Sensitivity Analysis')
            ax4.set_xticks(range(len(sensitivities)))
            ax4.set_xticklabels(param_names, rotation=45)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'phase_progression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_parameter_sensitivity(self, aggregate: ExperimentAggregate, output_dir: Path) -> None:
        """Plot parameter sensitivity analysis"""
        if not aggregate.parameter_sensitivity:
            return
        
        param_names = list(aggregate.parameter_sensitivity.keys())
        sensitivities = list(aggregate.parameter_sensitivity.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(param_names, sensitivities, color='skyblue', alpha=0.7)
        
        # Add value labels
        for bar, value in zip(bars, sensitivities):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{value:.3f}', va='center')
        
        plt.xlabel('Sensitivity (Absolute Correlation Coefficient)')
        plt.title('Parameter Sensitivity Analysis')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cross_phase_correlation(self, aggregate: ExperimentAggregate, output_dir: Path) -> None:
        """Plot cross-phase correlation heatmap"""
        if not aggregate.cross_phase_correlation.correlation_coefficients:
            return
        
        phase_names = list(self.results.keys())
        n_phases = len(phase_names)
        
        # Create correlation matrix
        corr_matrix = np.ones((n_phases, n_phases))
        for (phase1, phase2), corr in aggregate.cross_phase_correlation.correlation_coefficients.items():
            i = phase_names.index(phase1)
            j = phase_names.index(phase2)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr
        
        # Create heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, 
                   xticklabels=[p.replace('phase_', '').replace('_', '\n').title() for p in phase_names],
                   yticklabels=[p.replace('phase_', '').replace('_', '\n').title() for p in phase_names],
                   annot=True, fmt='.3f', cmap='coolwarm', center=0,
                   cbar_kws={'label': 'Correlation Coefficient'})
        plt.title('Cross-Phase Correlation Matrix')
        plt.tight_layout()
        plt.savefig(output_dir / 'cross_phase_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_detection_time_distribution(self, output_dir: Path) -> None:
        """Plot distribution of detection times across all phases"""
        all_detection_times = []
        phase_labels = []
        
        for phase_name, results in self.results.items():
            detection_times = []
            for result in results:
                if result.get("simulation_success") and result.get("t5sigma_s") is not None:
                    detection_times.append(result["t5sigma_s"])
            
            if detection_times:
                all_detection_times.extend(detection_times)
                phase_labels.extend([phase_name] * len(detection_times))
        
        if not all_detection_times:
            return
        
        # Create DataFrame for seaborn
        df = pd.DataFrame({
            'detection_time': all_detection_times,
            'phase': phase_labels
        })
        
        plt.figure(figsize=(12, 6))
        
        # Box plot by phase
        plt.subplot(1, 2, 1)
        sns.boxplot(data=df, x='phase', y='detection_time')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.title('Detection Time Distribution by Phase')
        plt.ylabel('Detection Time (s)')
        
        # Overall distribution
        plt.subplot(1, 2, 2)
        plt.hist(np.log10(all_detection_times), bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('log10(Detection Time [s])')
        plt.ylabel('Frequency')
        plt.title('Overall Detection Time Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'detection_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_success_rates(self, aggregate: ExperimentAggregate, output_dir: Path) -> None:
        """Plot success rates and simulation counts by phase"""
        phases = list(aggregate.phase_summary.keys())
        success_rates = [aggregate.phase_summary[p]["success_rate"] for p in phases]
        total_simulations = [aggregate.phase_summary[p]["total_simulations"] for p in phases]
        successful_simulations = [aggregate.phase_summary[p]["successful_simulations"] for p in phases]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Success rates
        bars1 = ax1.bar(range(len(phases)), success_rates, color='green', alpha=0.7)
        ax1.set_xlabel('Phase')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('Success Rate by Phase')
        ax1.set_xticks(range(len(phases)))
        ax1.set_xticklabels([p.replace('phase_', '').replace('_', ' ').title() for p in phases], rotation=45)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{rate:.1%}', ha='center', va='bottom')
        
        # Simulation counts
        x = np.arange(len(phases))
        width = 0.35
        bars2 = ax2.bar(x - width/2, total_simulations, width, label='Total', alpha=0.7)
        bars3 = ax2.bar(x + width/2, successful_simulations, width, label='Successful', alpha=0.7)
        ax2.set_xlabel('Phase')
        ax2.set_ylabel('Number of Simulations')
        ax2.set_title('Simulation Counts by Phase')
        ax2.set_xticks(x)
        ax2.set_xticklabels([p.replace('phase_', '').replace('_', ' ').title() for p in phases], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'success_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_comprehensive_report(self, aggregate: ExperimentAggregate) -> None:
        """Generate a comprehensive experiment report"""
        report_file = self.experiment_dir / "comprehensive_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("COMPREHENSIVE EXPERIMENT REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment ID: {self.experiment_id}\n")
            if self.manifest:
                f.write(f"Experiment Name: {self.manifest.get('name', 'N/A')}\n")
                f.write(f"Description: {self.manifest.get('description', 'N/A')}\n")
                f.write(f"Start Time: {self.manifest.get('start_time', 'N/A')}\n")
                f.write(f"End Time: {self.manifest.get('end_time', 'N/A')}\n\n")
            
            f.write("EXPERIMENT SUMMARY\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Simulations: {aggregate.total_simulations}\n")
            f.write(f"Successful Simulations: {aggregate.successful_simulations}\n")
            f.write(f"Overall Success Rate: {aggregate.success_rate:.1%}\n")
            f.write(f"Best Detection Time: {aggregate.best_detection_time:.2e} s\n" if aggregate.best_detection_time is not None else "Best Detection Time: N/A\n")
            f.write(f"Best Surface Gravity: {aggregate.best_kappa:.2e} s⁻¹\n" if aggregate.best_kappa is not None else "Best Surface Gravity: N/A\n")
            f.write(f"Best Signal-to-Noise Ratio: {aggregate.best_snr:.2f}\n\n" if aggregate.best_snr is not None else "Best Signal-to-Noise Ratio: N/A\n\n")
            
            f.write("PHASE SUMMARY\n")
            f.write("-" * 30 + "\n")
            for phase_name, summary in aggregate.phase_summary.items():
                f.write(f"\n{phase_name.replace('_', ' ').title()}:\n")
                f.write(f"  Total Simulations: {summary['total_simulations']}\n")
                f.write(f"  Successful: {summary['successful_simulations']}\n")
                f.write(f"  Success Rate: {summary['success_rate']:.1%}\n")
                f.write(f"  Best Detection Time: {summary['best_detection_time']:.2e} s\n" if summary['best_detection_time'] is not None else "  Best Detection Time: N/A\n")
                f.write(f"  Best Surface Gravity: {summary['best_kappa']:.2e} s⁻¹\n" if summary['best_kappa'] is not None else "  Best Surface Gravity: N/A\n")
            
            f.write("\nPARAMETER SENSITIVITY\n")
            f.write("-" * 30 + "\n")
            for param, sensitivity in aggregate.parameter_sensitivity.items():
                f.write(f"{param}: {sensitivity:.3f}\n")
            
            f.write("\nCROSS-PHASE CORRELATION\n")
            f.write("-" * 30 + "\n")
            for phase1, phase2, corr, p_val in aggregate.cross_phase_correlation.significant_correlations:
                f.write(f"{phase1} ↔ {phase2}: r={corr:.3f}, p={p_val:.3f}\n")
            
            f.write("\nSTATISTICAL SIGNIFICANCE\n")
            f.write("-" * 30 + "\n")
            stats = aggregate.statistical_significance
            if 'detection_probability_1h' in stats:
                prob_1h = stats['detection_probability_1h']
                f.write(f"1-hour observation:\n")
                f.write(f"  3σ detection probability: {prob_1h['detection_probability_3sigma']:.1%}\n")
                f.write(f"  5σ detection probability: {prob_1h['detection_probability_5sigma']:.1%}\n")
                f.write(f"  6σ detection probability: {prob_1h['detection_probability_6sigma']:.1%}\n")
        
        self.logger.info(f"Comprehensive report saved to {report_file}")


def main():
    """Main entry point for result aggregation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Result Aggregation Framework")
    parser.add_argument("experiment_id", help="Experiment ID to aggregate")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Initialize aggregator
    aggregator = ResultAggregator(args.experiment_id)
    
    if not aggregator.load_experiment_data():
        print(f"Failed to load experiment data for {args.experiment_id}")
        return 1
    
    # Aggregate results
    aggregate = aggregator.aggregate_results()
    
    # Generate report
    aggregator.generate_comprehensive_report(aggregate)
    
    # Generate visualizations if requested
    if args.visualize:
        aggregator.generate_visualizations(aggregate)
    
    print(f"Successfully aggregated results for experiment {args.experiment_id}")
    print(f"Total simulations: {aggregate.total_simulations}")
    print(f"Success rate: {aggregate.success_rate:.1%}")
    if aggregate.best_detection_time is not None:
        print(f"Best detection time: {aggregate.best_detection_time:.2e} s")
    else:
        print("Best detection time: N/A")
    
    return 0


if __name__ == "__main__":
    main()
