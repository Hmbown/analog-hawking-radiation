#!/usr/bin/env python3
"""
Statistical significance analysis for analog Hawking radiation detection.
Evaluates signal-to-noise ratios, detection thresholds, and statistical
significance across parameter space.
"""

import numpy as np
import json
import os
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any, Optional
import argparse
from dataclasses import dataclass
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@dataclass
class DetectionThresholds:
    """Detection significance thresholds"""
    sigma_3: float = 3.0  # 3-sigma (99.7% confidence)
    sigma_5: float = 5.0  # 5-sigma (99.99994% confidence)
    sigma_6: float = 6.0  # 6-sigma (discovery level)

class StatisticalAnalyzer:
    """Analyzes statistical significance of Hawking radiation detection"""
    
    def __init__(self, thresholds: DetectionThresholds = DetectionThresholds()):
        self.thresholds = thresholds
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_dir = Path("results/analysis")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'statistical_analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def load_results(self, data_path: str) -> List[Dict[str, Any]]:
        """Load simulation results for analysis"""
        
        data_path = Path(data_path)
        
        if data_path.is_file():
            files = [data_path]
        else:
            files = list(data_path.glob("*.json"))
        
        all_results = []
        
        for file in files:
            try:
                with open(file) as f:
                    data = json.load(f)
                
                if 'results' in data:
                    all_results.extend(data['results'])
                else:
                    all_results.append(data)
            
            except Exception as e:
                self.logger.warning(f"Error loading {file}: {e}")
                continue
        
        # Filter valid results
        valid_results = []
        for result in all_results:
            if 'error' not in result and result.get('kappa') and result.get('t5sigma_s') is not None:
                valid_results.append(result)
        
        self.logger.info(f"Loaded {len(valid_results)} valid results")
        return valid_results
    
    def calculate_signal_to_noise(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate signal-to-noise ratios for detection"""
        
        enhanced_results = []
        
        for result in results:
            enhanced = result.copy()
            
            # Extract key parameters
            T_sig = result.get('T_sig_K')
            T_H = result.get('T_H_K')
            t5sigma = result.get('t5sigma_s')
            klist = result.get('kappa', [])
            kappa = klist[0] if klist else None
            
            if T_sig is None or t5sigma is None:
                continue
            
            # System parameters
            T_sys = 30.0  # K (system temperature)
            B = 1e8  # Hz (bandwidth)
            
            # Signal-to-noise ratio
            snr = T_sig / T_sys if T_sig > 0 else 0.0
            
            # Detection rate (inverse of detection time)
            detection_rate = 1.0 / t5sigma if t5sigma > 0 else 0.0
            
            # Statistical significance
            # For thermal noise, significance scales as sqrt(B * t)
            significance = snr * np.sqrt(B * t5sigma) if t5sigma > 0 else 0.0
            
            # Hawking temperature significance
            hawking_significance = kappa * 1e-12 if kappa else 0.0  # Scale factor
            
            enhanced.update({
                'signal_to_noise_ratio': float(snr),
                'detection_rate': float(detection_rate),
                'statistical_significance': float(significance),
                'hawking_significance': float(hawking_significance),
                'is_detectable_3sigma': significance >= self.thresholds.sigma_3,
                'is_detectable_5sigma': significance >= self.thresholds.sigma_5,
                'is_detectable_6sigma': significance >= self.thresholds.sigma_6
            })
            
            enhanced_results.append(enhanced)
        
        return enhanced_results
    
    def analyze_parameter_sensitivity(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sensitivity of detection to parameter variations"""
        
        if not results:
            return {'error': 'No results to analyze'}
        
        # Extract parameters and metrics
        parameters = []
        detection_times = []
        snr_values = []
        
        feature_names = [
            'laser_intensity', 'plasma_density', 'magnetic_field',
            'temperature_constant', 'laser_wavelength', 'grid_max'
        ]
        
        for result in results:
            params = result.get('parameters', {})
            
            # Extract features
            features = []
            for name in feature_names:
                if name in params:
                    features.append(float(params[name]))
                elif name == 'magnetic_field' and 'magnetic_field' not in params:
                    features.append(0.0)
                else:
                    features.append(1.0)  # Default
            
            t5sigma = result.get('t5sigma_s')
            snr = result.get('signal_to_noise_ratio', 0.0)
            
            if t5sigma is not None and np.isfinite(t5sigma):
                parameters.append(features)
                detection_times.append(float(np.log10(np.maximum(t5sigma, 1e-10))))
                snr_values.append(float(snr))
        
        if not parameters:
            return {'error': 'No valid parameter data'}
        
        parameters = np.array(parameters)
        detection_times = np.array(detection_times)
        snr_values = np.array(snr_values)
        
        # Calculate correlations
        correlations = {}
        for i, name in enumerate(feature_names):
            if len(detection_times) > 1:
                corr_t = np.corrcoef(parameters[:, i], detection_times)[0, 1]
                corr_snr = np.corrcoef(parameters[:, i], snr_values)[0, 1]
                correlations[name] = {
                    'detection_time_correlation': float(corr_t) if not np.isnan(corr_t) else 0.0,
                    'snr_correlation': float(corr_snr) if not np.isnan(corr_snr) else 0.0
                }
        
        # Calculate relative importance
        total_variance = np.var(detection_times)
        importances = {}
        
        for i, name in enumerate(feature_names):
            if len(detection_times) > 1:
                # Simple linear regression importance
                slope, _, _, _, _ = stats.linregress(parameters[:, i], detection_times)
                importance = abs(slope) * np.std(parameters[:, i]) / total_variance
                importances[name] = float(importance)
        
        return {
            'correlations': correlations,
            'importances': importances,
            'parameter_names': feature_names,
            'total_samples': len(results)
        }
    
    def calculate_detection_probability(self, results: List[Dict[str, Any]], 
                                      observation_time: float = 3600.0) -> Dict[str, Any]:
        """Calculate detection probability for given observation time"""
        
        enhanced_results = self.calculate_signal_to_noise(results)
        
        detection_stats = {
            'total_cases': len(enhanced_results),
            'detectable_3sigma': 0,
            'detectable_5sigma': 0,
            'detectable_6sigma': 0,
            'detection_probability_3sigma': 0.0,
            'detection_probability_5sigma': 0.0,
            'detection_probability_6sigma': 0.0
        }
        
        for result in enhanced_results:
            t5sigma = result.get('t5sigma_s')
            if t5sigma is None:
                continue
            
            # Check if observation time is sufficient
            if t5sigma <= observation_time:
                if result.get('is_detectable_3sigma', False):
                    detection_stats['detectable_3sigma'] += 1
                if result.get('is_detectable_5sigma', False):
                    detection_stats['detectable_5sigma'] += 1
                if result.get('is_detectable_6sigma', False):
                    detection_stats['detectable_6sigma'] += 1
        
        # Calculate probabilities
        if detection_stats['total_cases'] > 0:
            detection_stats['detection_probability_3sigma'] = (
                detection_stats['detectable_3sigma'] / detection_stats['total_cases']
            )
            detection_stats['detection_probability_5sigma'] = (
                detection_stats['detectable_5sigma'] / detection_stats['total_cases']
            )
            detection_stats['detection_probability_6sigma'] = (
                detection_stats['detectable_6sigma'] / detection_stats['total_cases']
            )
        
        return detection_stats
    
    def analyze_scaling_requirements(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze requirements for achieving statistical significance"""
        
        enhanced_results = self.calculate_signal_to_noise(results)
        
        # Filter results with valid detection times
        valid_results = [r for r in enhanced_results if r.get('t5sigma_s') is not None]
        
        if not valid_results:
            return {'error': 'No valid detection times'}
        
        # Extract parameters and detection times
        intensities = []
        densities = []
        detection_times = []
        snr_values = []
        
        for result in valid_results:
            params = result.get('parameters', {})
            t5sigma = result.get('t5sigma_s')
            snr = result.get('signal_to_noise_ratio', 0.0)
            
            if t5sigma is not None and np.isfinite(t5sigma):
                intensities.append(float(params.get('laser_intensity', 0.0)))
                densities.append(float(params.get('plasma_density', 0.0)))
                detection_times.append(float(t5sigma))
                snr_values.append(float(snr))
        
        if not detection_times:
            return {'error': 'No valid data for scaling analysis'}
        
        # Convert to arrays
        intensities = np.array(intensities)
        densities = np.array(densities)
        detection_times = np.array(detection_times)
        snr_values = np.array(snr_values)
        
        # Power law scaling analysis
        # Assume t ~ (I * n)^(-alpha)
        combined_params = intensities * densities
        log_params = np.log10(np.maximum(combined_params, 1e-30))
        log_times = np.log10(np.maximum(detection_times, 1e-30))
        
        # Fit power law
        if len(log_params) > 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_params, log_times)
            
            scaling_analysis = {
                'power_law_exponent': float(-slope),  # Negative because t ~ param^(-alpha)
                'correlation_coefficient': float(r_value),
                'p_value': float(p_value),
                'scaling_formula': f't_5sigma ∝ (I·n)^{-slope:.2f}',
                'current_median_time': float(np.median(detection_times)),
                'current_min_time': float(np.min(detection_times)),
                'current_max_time': float(np.max(detection_times))
            }
        else:
            scaling_analysis = {
                'error': 'Insufficient data for power law fitting'
            }
        
        # Calculate scaling requirements
        target_times = [3600, 86400, 604800]  # 1 hour, 1 day, 1 week
        scaling_requirements = {}
        
        current_median = np.median(detection_times)
        
        for target in target_times:
            if current_median > 0:
                scaling_factor = current_median / target
                
                if 'power_law_exponent' in scaling_analysis:
                    exponent = scaling_analysis['power_law_exponent']
                    required_intensity_increase = scaling_factor ** (1/exponent)
                    
                    scaling_requirements[f'target_{target}s'] = {
                        'required_scaling_factor': float(scaling_factor),
                        'required_intensity_density_product': float(required_intensity_increase),
                        'feasibility': 'achievable' if required_intensity_increase < 1e4 else 'challenging'
                    }
        
        return {
            'scaling_analysis': scaling_analysis,
            'scaling_requirements': scaling_requirements,
            'parameter_ranges': {
                'intensity_range': [float(np.min(intensities)), float(np.max(intensities))],
                'density_range': [float(np.min(densities)), float(np.max(densities))],
                'detection_time_range': [float(np.min(detection_times)), float(np.max(detection_times))]
            }
        }
    
    def generate_visualizations(self, results: List[Dict[str, Any]], output_dir: str):
        """Generate statistical analysis visualizations"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Calculate enhanced metrics
        enhanced_results = self.calculate_signal_to_noise(results)
        
        # Extract data for plotting
        snr_values = [r.get('signal_to_noise_ratio', 0.0) for r in enhanced_results]
        detection_times = [r.get('t5sigma_s', 1e6) for r in enhanced_results]
        
        # Filter valid data
        valid_mask = [(s > 0 and t > 0) for s, t in zip(snr_values, detection_times)]
        snr_values = np.array(snr_values)[valid_mask]
        detection_times = np.array(detection_times)[valid_mask]
        
        if len(snr_values) == 0:
            self.logger.warning("No valid data for visualization")
            return
        
        # 1. SNR vs Detection Time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.scatter(snr_values, detection_times, alpha=0.6, s=20)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Signal-to-Noise Ratio (T_sig/T_sys)')
        plt.ylabel('5σ Detection Time (s)')
        plt.title('Detection Time vs Signal-to-Noise Ratio')
        plt.grid(True, alpha=0.3)
        
        # Add threshold lines
        plt.axhline(y=3600, color='r', linestyle='--', alpha=0.7, label='1 hour')
        plt.axhline(y=86400, color='g', linestyle='--', alpha=0.7, label='1 day')
        plt.axhline(y=604800, color='b', linestyle='--', alpha=0.7, label='1 week')
        plt.legend()
        
        # 2. Detection Probability Distribution
        plt.subplot(2, 2, 2)
        
        # Calculate detection probabilities for different observation times
        times = [3600, 86400, 604800]  # 1h, 1d, 1w
        probabilities = []
        
        for t in times:
            stats = self.calculate_detection_probability(enhanced_results, t)
            probabilities.append([
                stats['detection_probability_3sigma'],
                stats['detection_probability_5sigma'],
                stats['detection_probability_6sigma']
            ])
        
        x = ['3σ', '5σ', '6σ']
        width = 0.25
        
        for i, (t, probs) in enumerate(zip(times, probabilities)):
            plt.bar([j + i*width for j in range(3)], probs, width, 
                   label=f'{t/3600:.0f}h' if t < 86400 else f'{t/86400:.0f}d')
        
        plt.xlabel('Significance Level')
        plt.ylabel('Detection Probability')
        plt.title('Detection Probability vs Observation Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Parameter Sensitivity
        sensitivity = self.analyze_parameter_sensitivity(enhanced_results)
        if 'ranked_features' in sensitivity:
            plt.subplot(2, 2, 3)
            
            features, importances = zip(*sensitivity['ranked_features'])
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Relative Importance')
            plt.title('Parameter Sensitivity Analysis')
            plt.grid(True, alpha=0.3)
        
        # 4. Detection Time Distribution
        plt.subplot(2, 2, 4)
        
        log_times = np.log10(np.maximum(detection_times, 1e-10))
        plt.hist(log_times, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(x=np.log10(3600), color='r', linestyle='--', alpha=0.7, label='1 hour')
        plt.axvline(x=np.log10(86400), color='g', linestyle='--', alpha=0.7, label='1 day')
        plt.xlabel('log10(Detection Time [s])')
        plt.ylabel('Frequency')
        plt.title('Distribution of Detection Times')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'statistical_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Visualizations saved to {output_path}")
    
    def generate_report(self, results: List[Dict[str, Any]]) -> str:
        """Generate comprehensive statistical analysis report"""
        
        enhanced_results = self.calculate_signal_to_noise(results)
        
        # Basic statistics
        detection_times = [r.get('t5sigma_s', 1e6) for r in enhanced_results if r.get('t5sigma_s') is not None]
        snr_values = [r.get('signal_to_noise_ratio', 0.0) for r in enhanced_results if r.get('signal_to_noise_ratio') > 0]
        
        if not detection_times:
            return "No valid detection data found"
        
        detection_times = np.array(detection_times)
        snr_values = np.array(snr_values)
        
        # Detection probabilities
        prob_1h = self.calculate_detection_probability(enhanced_results, 3600)
        prob_1d = self.calculate_detection_probability(enhanced_results, 86400)
        prob_1w = self.calculate_detection_probability(enhanced_results, 604800)
        
        # Scaling analysis
        scaling = self.analyze_scaling_requirements(enhanced_results)
        
        report = f"""
Statistical Analysis Report for Analog Hawking Radiation Detection
{'='*70}

Dataset Summary:
- Total valid cases: {len(enhanced_results)}
- Detection time range: {np.min(detection_times):.2e} - {np.max(detection_times):.2e} seconds
- Median detection time: {np.median(detection_times):.2e} seconds
- Signal-to-noise range: {np.min(snr_values):.2e} - {np.max(snr_values):.2e}

Detection Probabilities:
- 1 hour observation:
  * 3σ significance: {prob_1h['detection_probability_3sigma']:.2%}
  * 5σ significance: {prob_1h['detection_probability_5sigma']:.2%}
  * 6σ significance: {prob_1h['detection_probability_6sigma']:.2%}

- 1 day observation:
  * 3σ significance: {prob_1d['detection_probability_3sigma']:.2%}
  * 5σ significance: {prob_1d['detection_probability_5sigma']:.2%}
  * 6σ significance: {prob_1d['detection_probability_6sigma']:.2%}

- 1 week observation:
  * 3σ significance: {prob_1w['detection_probability_3sigma']:.2%}
  * 5σ significance: {prob_1w['detection_probability_5sigma']:.2%}
  * 6σ significance: {prob_1w['detection_probability_6sigma']:.2%}

Scaling Requirements for 1-hour Detection:
"""
        
        if 'scaling_requirements' in scaling:
            for target, req in scaling['scaling_requirements'].items():
                if isinstance(req, dict):
                    report += f"- {target}: {req['required_scaling_factor']:.1f}x intensity-density product "
                    report += f"({req['feasibility']})\n"
        
        return report

def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for Hawking radiation detection")
    parser.add_argument("--data", required=True, help="Path to simulation results")
    parser.add_argument("--output", default="results/analysis", help="Output directory")
    parser.add_argument("--observation-time", type=float, default=3600, help="Observation time in seconds")
    parser.add_argument("--visualize", action="store_true", help="Generate visualizations")
    parser.add_argument("--report", action="store_true", help="Generate detailed report")
    
    args = parser.parse_args()
    
    analyzer = StatisticalAnalyzer()
    
    # Load data
    results = analyzer.load_results(args.data)
    
    # Perform analysis
    enhanced_results = analyzer.calculate_signal_to_noise(results)
    
    # Detection probability
    detection_stats = analyzer.calculate_detection_probability(enhanced_results, args.observation_time)
    
    # Sensitivity analysis
    sensitivity = analyzer.analyze_parameter_sensitivity(enhanced_results)
    
    # Scaling analysis
    scaling = analyzer.analyze_scaling_requirements(enhanced_results)
    
    # Save results
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {
        'detection_statistics': detection_stats,
        'parameter_sensitivity': sensitivity,
        'scaling_analysis': scaling,
        'enhanced_results': enhanced_results,
        'observation_time': args.observation_time
    }
    
    with open(output_path / 'statistical_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    # Generate visualizations
    if args.visualize:
        analyzer.generate_visualizations(enhanced_results, str(output_path))
    
    # Generate report
    if args.report:
        report = analyzer.generate_report(results)
        
        with open(output_path / 'statistical_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
    
    print(f"\nAnalysis complete. Results saved to {output_path}")

if __name__ == "__main__":
    main()
