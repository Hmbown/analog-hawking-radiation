#!/usr/bin/env python3
"""
Comprehensive Detection Feasibility Analysis for Analog Hawking Radiation

This script performs realistic detection feasibility assessment including:
1. Analysis of existing dataset configurations
2. Comprehensive noise modeling and SNR calculations
3. Detection strategy assessment for ELI facilities
4. Near-term achievable goals identification
5. Recommendations for experimental planning

Author: Claude Analysis Assistant
Date: November 2025
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.analog_hawking.detection.detection_feasibility import (
    DetectionFeasibilityAnalyzer,
    SignalCharacteristics,
)
from src.analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory


def load_existing_dataset() -> pd.DataFrame:
    """Load existing dataset for analysis"""

    # Try to load the hybrid sweep data
    hybrid_sweep_path = Path("results/hybrid_sweep.csv")
    if hybrid_sweep_path.exists():
        df = pd.read_csv(hybrid_sweep_path)
        print(f"✅ Loaded hybrid sweep dataset: {len(df)} configurations")
        return df

    # Try alternative locations
    alt_paths = [
        "results/comprehensive_analysis_results.csv",
        "results/parameter_sweep_results.csv"
    ]

    for path in alt_paths:
        if Path(path).exists():
            df = pd.read_csv(path)
            print(f"✅ Loaded dataset from {path}: {len(df)} configurations")
            return df

    # Create synthetic dataset if none found
    print("⚠️ No existing dataset found, creating synthetic parameter sweep")
    return create_synthetic_dataset()


def create_synthetic_dataset() -> pd.DataFrame:
    """Create synthetic dataset parameter sweep for analysis"""

    print("🔄 Generating synthetic parameter sweep...")

    # Parameter ranges based on realistic ELI capabilities
    n_samples = 100
    a0_range = np.linspace(0.5, 5.0, n_samples)  # Normalized vector potential
    ne_range = np.logspace(19, 22, n_samples)  # Electron density (m^-3)

    data = []
    for i in range(n_samples):
        a0 = a0_range[i]
        ne = ne_range[i]

        # Calculate derived parameters
        intensity_W_m2 = (a0 * 9.11e-31 * 3e8 * 2 * np.pi * 3e8 / (800e-9 * 1.6e-19))**2 * 3e8 * 8.85e-12 / 2

        # Estimate surface gravity
        wavelength = 800e-9
        velocity_char = 3e8 * a0 / np.sqrt(1 + a0**2)
        surface_gravity = velocity_char / wavelength

        # Hawking temperature
        h_bar = 1.054e-34
        k_B = 1.381e-23
        hawking_temp = h_bar * surface_gravity / (2 * np.pi * k_B)

        # Peak frequency (estimate)
        peak_freq = k_B * hawking_temp / h_bar / (2 * np.pi)

        data.append({
            'config_id': f'synth_{i:03d}',
            'a0': a0,
            'plasma_density_m3': ne,
            'laser_intensity_W_m2': intensity_W_m2,
            'surface_gravity_s': surface_gravity,
            'hawking_temperature_K': hawking_temp,
            'peak_frequency_Hz': peak_freq,
            'wavelength_nm': 800
        })

    df = pd.DataFrame(data)

    # Save synthetic dataset
    df.to_csv("results/synthetic_detection_dataset.csv", index=False)
    print(f"✅ Created synthetic dataset: {len(df)} configurations")

    return df


def analyze_configuration_detection_feasibility(row: pd.Series,
                                              analyzer: DetectionFeasibilityAnalyzer) -> Dict[str, Any]:
    """Analyze detection feasibility for a single configuration"""

    try:
        # Handle different dataset structures
        if 'config_id' in row:
            config_id = row['config_id']
        else:
            config_id = f"config_{row.name}"

        # Extract parameters from dataset or calculate them
        if 'kappa_mirror' in row:
            surface_gravity = row['kappa_mirror']
        elif 'surface_gravity_s' in row:
            surface_gravity = row['surface_gravity_s']
        else:
            # Estimate from coupling strength
            surface_gravity = 1e12 * row.get('coupling_strength', 0.1)

        # Calculate Hawking temperature
        h_bar = 1.054e-34
        k_B = 1.381e-23
        hawking_temp = h_bar * surface_gravity / (2 * np.pi * k_B)

        # Estimate peak frequency
        if 'peak_frequency_Hz' in row:
            peak_frequency = row['peak_frequency_Hz']
        else:
            peak_frequency = k_B * hawking_temp / h_bar

        # Calculate signal power from hybrid data if available
        if 'T_sig_hybrid' in row:
            signal_temperature = row['T_sig_hybrid']
        elif 'T_sig_fluid' in row:
            signal_temperature = row['T_sig_fluid']
        else:
            signal_temperature = hawking_temp * 0.1

        # Estimate total power
        bandwidth = peak_frequency / 10 if peak_frequency > 0 else 1e12
        total_power = k_B * signal_temperature * bandwidth

        # Create signal characteristics
        signal = SignalCharacteristics(
            hawking_temperature=hawking_temp,
            surface_gravity=surface_gravity,
            peak_frequency=peak_frequency,
            bandwidth=bandwidth,
            total_power=total_power,
            power_density=total_power / bandwidth if bandwidth > 0 else 1e-30,
            signal_temperature=signal_temperature,
            pulse_duration=1e-12,
            rise_time=1e-13,
            repetition_rate=1.0,
            emitting_area=1e-12,  # 1 μm²
            angular_distribution="isotropic"
        )

        # Refine signal power calculation using QFT
        try:
            qft = QuantumFieldTheory(
                surface_gravity=row['surface_gravity_s'],
                emitting_area_m2=1e-12,
                solid_angle_sr=4*np.pi,
                coupling_efficiency=0.1
            )

            omega_peak = 2 * np.pi * row['peak_frequency_Hz']
            frequencies = np.logspace(np.log10(row['peak_frequency_Hz']/10),
                                    np.log10(row['peak_frequency_Hz']*10), 100)
            omega_array = 2 * np.pi * frequencies

            spectrum = qft.hawking_spectrum(omega_array)
            total_power = np.trapz(spectrum, omega_array)
            peak_power_density = np.max(spectrum)

            signal.total_power = total_power
            signal.power_density = peak_power_density
            signal.signal_temperature = total_power / (1.381e-23 * signal.bandwidth)

        except Exception as e:
            print(f"⚠️ QFT calculation failed for {row['config_id']}: {e}")
            # Use rough estimates
            signal.total_power = 1e-25 * (row['hawking_temperature_K'] / 0.01)**4
            signal.power_density = signal.total_power / signal.bandwidth
            signal.signal_temperature = signal.total_power / (1.381e-23 * signal.bandwidth)

        # Plasma parameters
        plasma_params = {
            'temperature': 1e6,  # 1 MK typical
            'density': row['plasma_density_m3']
        }

        # Assess detection feasibility
        assessments = analyzer.assess_detection_feasibility(signal, plasma_params)

        # Extract top assessment
        best_assessment = assessments[0] if assessments else None

        results = {
            'config_id': config_id,
            'coupling_strength': row.get('coupling_strength', 0.1),
            'plasma_density_m3': row.get('plasma_density_m3', 1e24),
            'laser_intensity_W_m2': row.get('laser_intensity_W_m2', 1e20),
            'hawking_temperature_K': hawking_temp,
            'surface_gravity_s': surface_gravity,
            'peak_frequency_Hz': peak_frequency,
            'signal_power_W': signal.total_power,
            'signal_temperature_K': signal.signal_temperature,
            'num_methods_assessed': len(assessments),
        }

        if best_assessment:
            results.update({
                'best_method': best_assessment.detection_method.value,
                'best_detector': best_assessment.detector_type.value,
                'best_snr': best_assessment.snr_optimal,
                'feasibility_level': best_assessment.feasibility_level.value,
                'detection_probability': best_assessment.detection_probability,
                'integration_time_s': best_assessment.optimal_integration_time,
                'required_shots': best_assessment.required_shots,
                'experiment_time_hours': best_assessment.total_experiment_time,
                'cost_estimate': best_assessment.cost_estimate,
                'dominant_noise': best_assessment.dominant_noise_source,
                'recommendations': best_assessment.recommendations[:3]  # Top 3
            })
        else:
            results.update({
                'best_method': 'None',
                'best_detector': 'None',
                'best_snr': 0.0,
                'feasibility_level': 'Impossible',
                'detection_probability': 0.0,
                'integration_time_s': float('inf'),
                'required_shots': 0,
                'experiment_time_hours': float('inf'),
                'cost_estimate': 'N/A',
                'dominant_noise': 'N/A',
                'recommendations': ['No feasible detection method']
            })

        return results

    except Exception as e:
        print(f"❌ Error analyzing configuration {row['config_id']}: {e}")
        config_id = row.get('config_id', f"config_{row.name}")
        return {
            'config_id': config_id,
            'error': str(e),
            'best_snr': 0.0,
            'feasibility_level': 'Error'
        }


def generate_detection_feasibility_plots(results_df: pd.DataFrame, output_dir: str):
    """Generate comprehensive detection feasibility plots"""

    print("🔄 Generating detection feasibility plots...")

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Set up plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # 1. SNR vs Configuration Parameters
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # SNR vs coupling strength
    valid_data = results_df[results_df['best_snr'] > 0]
    if len(valid_data) > 0:
        if 'a0' in valid_data.columns:
            axes[0, 0].scatter(valid_data['a0'], valid_data['best_snr'], alpha=0.6, s=50)
            axes[0, 0].set_xlabel('Normalized Vector Potential a₀')
            axes[0, 0].set_title('SNR vs Laser Intensity Parameter')
            axes[0, 0].set_xscale('log')
        else:
            axes[0, 0].scatter(valid_data['coupling_strength'], valid_data['best_snr'], alpha=0.6, s=50)
            axes[0, 0].set_xlabel('Coupling Strength')
            axes[0, 0].set_title('SNR vs Coupling Strength')

        axes[0, 0].set_ylabel('Best SNR')
        axes[0, 0].set_yscale('log')
        axes[0, 0].grid(True, alpha=0.3)

    # SNR vs Plasma Density
    if len(valid_data) > 0:
        axes[0, 1].scatter(valid_data['plasma_density_m3'], valid_data['best_snr'], alpha=0.6, s=50)
        axes[0, 1].set_xlabel('Plasma Density (m⁻³)')
        axes[0, 1].set_ylabel('Best SNR')
        axes[0, 1].set_title('SNR vs Plasma Density')
        axes[0, 1].set_xscale('log')
        axes[0, 1].set_yscale('log')
        axes[0, 1].grid(True, alpha=0.3)

    # SNR vs Hawking Temperature
    if len(valid_data) > 0:
        axes[1, 0].scatter(valid_data['hawking_temperature_K'], valid_data['best_snr'], alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Hawking Temperature (K)')
        axes[1, 0].set_ylabel('Best SNR')
        axes[1, 0].set_title('SNR vs Hawking Temperature')
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)

        # Add SNR = 5 reference line
        axes[1, 0].axhline(y=5, color='red', linestyle='--', alpha=0.7, label='5σ detection threshold')
        axes[1, 0].legend()

    # Feasibility Level Distribution
    feasibility_counts = results_df['feasibility_level'].value_counts()
    axes[1, 1].bar(range(len(feasibility_counts)), feasibility_counts.values)
    axes[1, 1].set_xlabel('Feasibility Level')
    axes[1, 1].set_ylabel('Number of Configurations')
    axes[1, 1].set_title('Detection Feasibility Distribution')
    axes[1, 1].set_xticks(range(len(feasibility_counts)))
    axes[1, 1].set_xticklabels(feasibility_counts.index, rotation=45)

    plt.tight_layout()
    plt.savefig(output_path / 'detection_feasibility_overview.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Detection Method Performance Comparison
    if len(valid_data) > 0:
        plt.figure(figsize=(12, 8))

        method_performance = valid_data.groupby('best_method').agg({
            'best_snr': ['mean', 'std', 'count'],
            'detection_probability': 'mean',
            'integration_time_s': 'mean'
        }).round(3)

        method_performance.columns = ['Mean SNR', 'SNR Std', 'Count', 'Mean Detection Prob', 'Mean Integration Time']
        method_performance = method_performance.sort_values('Mean SNR', ascending=False)

        # Create subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Mean SNR by method
        method_performance['Mean SNR'].plot(kind='bar', ax=ax1, yerr=method_performance['SNR Std'])
        ax1.set_title('Mean SNR by Detection Method')
        ax1.set_ylabel('SNR')
        ax1.tick_params(axis='x', rotation=45)

        # Detection probability by method
        method_performance['Mean Detection Prob'].plot(kind='bar', ax=ax2)
        ax2.set_title('Detection Probability by Method')
        ax2.set_ylabel('Detection Probability')
        ax2.tick_params(axis='x', rotation=45)

        # Integration time by method
        method_performance['Mean Integration Time'].plot(kind='bar', ax=ax3, logy=True)
        ax3.set_title('Integration Time by Method')
        ax3.set_ylabel('Integration Time (s)')
        ax3.tick_params(axis='x', rotation=45)

        # Count of feasible configurations
        method_performance['Count'].plot(kind='bar', ax=ax4)
        ax4.set_title('Number of Feasible Configurations')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(output_path / 'detection_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

    # 3. Parameter Space Feasibility Map
    if len(valid_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Create feasibility map
        feasibility_map = valid_data.pivot_table(
            values='best_snr',
            index='a0',
            columns='plasma_density_m3',
            aggfunc='mean'
        )

        # SNR heatmap
        im1 = axes[0].imshow(np.log10(feasibility_map.values),
                            aspect='auto', origin='lower', cmap='viridis')
        axes[0].set_title('Log₁₀(SNR) Parameter Space')
        axes[0].set_xlabel('Plasma Density Index')
        axes[0].set_ylabel('a₀ Index')
        plt.colorbar(im1, ax=axes[0], label='Log₁₀(SNR)')

        # Feasibility level heatmap
        feasibility_numeric = valid_data['feasibility_level'].map({
            'Impossible': 0, 'Highly Challenging': 1, 'Challenging': 2,
            'Feasible': 3, 'Straightforward': 4
        })

        feasibility_numeric_map = valid_data.pivot_table(
            values=feasibility_numeric,
            index='a0',
            columns='plasma_density_m3',
            aggfunc='mean'
        )

        im2 = axes[1].imshow(feasibility_numeric_map.values,
                            aspect='auto', origin='lower', cmap='RdYlGn')
        axes[1].set_title('Feasibility Level Parameter Space')
        axes[1].set_xlabel('Plasma Density Index')
        axes[1].set_ylabel('a₀ Index')
        plt.colorbar(im2, ax=axes[1], label='Feasibility Level')

        plt.tight_layout()
        plt.savefig(output_path / 'parameter_space_feasibility.png', dpi=300, bbox_inches='tight')
        plt.close()

    print(f"✅ Plots saved to {output_path}")


def generate_summary_statistics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive summary statistics"""

    print("🔄 Generating summary statistics...")

    # Overall statistics
    total_configs = len(results_df)
    feasible_configs = len(results_df[results_df['best_snr'] >= 5])
    challenging_configs = len(results_df[(results_df['best_snr'] >= 1) & (results_df['best_snr'] < 5)])
    impossible_configs = len(results_df[results_df['best_snr'] < 1])

    summary = {
        'total_configurations': total_configs,
        'feasible_configurations': feasible_configs,
        'challenging_configurations': challenging_configs,
        'impossible_configurations': impossible_configs,
        'feasibility_percentage': 100 * feasible_configs / total_configs if total_configs > 0 else 0,

        'best_overall_snr': results_df['best_snr'].max(),
        'worst_overall_snr': results_df['best_snr'].min(),
        'mean_snr': results_df['best_snr'].mean(),
        'median_snr': results_df['best_snr'].median(),
    }

    # Add detection probability if available
    if 'detection_probability' in results_df.columns:
        summary.update({
            'best_detection_probability': results_df['detection_probability'].max(),
            'mean_detection_probability': results_df['detection_probability'].mean(),
        })
    else:
        summary.update({
            'best_detection_probability': 0.0,
            'mean_detection_probability': 0.0,
        })

    # Parameter ranges
    summary['parameter_ranges'] = {}

    if 'a0' in results_df.columns:
        summary['parameter_ranges']['a0'] = {
            'min': results_df['a0'].min(),
            'max': results_df['a0'].max(),
            'optimal_range': [results_df.loc[results_df['best_snr'].idxmax(), 'a0']]
        }
    elif 'coupling_strength' in results_df.columns:
        summary['parameter_ranges']['coupling_strength'] = {
            'min': results_df['coupling_strength'].min(),
            'max': results_df['coupling_strength'].max(),
            'optimal_range': [results_df.loc[results_df['best_snr'].idxmax(), 'coupling_strength']]
        }

    if 'plasma_density_m3' in results_df.columns:
        summary['parameter_ranges']['plasma_density'] = {
            'min': results_df['plasma_density_m3'].min(),
            'max': results_df['plasma_density_m3'].max(),
            'optimal_range': [results_df.loc[results_df['best_snr'].idxmax(), 'plasma_density_m3']]
        }

    if 'hawking_temperature_K' in results_df.columns:
        summary['parameter_ranges']['hawking_temperature'] = {
            'min': results_df['hawking_temperature_K'].min(),
            'max': results_df['hawking_temperature_K'].max(),
            'optimal_range': [results_df.loc[results_df['best_snr'].idxmax(), 'hawking_temperature_K']]
        }

    # Detection method statistics
    method_stats = results_df.groupby('best_method').agg({
        'best_snr': ['count', 'mean', 'max'],
        'detection_probability': 'mean'
    }).round(3)

    summary['detection_method_performance'] = method_stats.to_dict()

    # Feasibility level distribution
    feasibility_dist = results_df['feasibility_level'].value_counts().to_dict()
    summary['feasibility_distribution'] = feasibility_dist

    # Noise source analysis
    noise_analysis = results_df['dominant_noise'].value_counts().to_dict()
    summary['dominant_noise_sources'] = noise_analysis

    return summary


def generate_eli_facility_recommendations(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Generate ELI facility-specific recommendations"""

    print("🔄 Generating ELI facility recommendations...")

    # Find best performing configurations
    top_configs = results_df.nlargest(10, 'best_snr')

    recommendations = {
        'facility_requirements': {
            'laser_systems': {
                'intensity_range': [results_df['laser_intensity_W_m2'].min(), results_df['laser_intensity_W_m2'].max()],
                'optimal_intensity': top_configs['laser_intensity_W_m2'].mean(),
                'repetition_rate': '1-10 Hz for adequate statistics',
                'pulse_duration': '30-150 fs for optimal plasma formation',
                'wavelength': '800 nm standard, 1030 nm for higher power'
            },
            'target_systems': {
                'plasma_density_range': [results_df['plasma_density_m3'].min(), results_df['plasma_density_m3'].max()],
                'optimal_density': top_configs['plasma_density_m3'].mean(),
                'target_materials': 'Solid targets (foils, tapes) for reproducibility',
                'target_positioning': 'Micron precision for consistent conditions'
            },
            'diagnostics': {
                'primary_methods': top_configs['best_method'].unique().tolist(),
                'timing_resolution': '< 100 fs for transient phenomena',
                'spectral_resolution': 'High resolution for narrow spectral features',
                'spatial_resolution': 'Micron-scale for emission region mapping'
            }
        },

        'near_term_goals': {
            'horizon_detection': {
                'feasibility': 'High',
                'timeline': '6-12 months',
                'requirements': 'Standard interferometry and optical diagnostics',
                'confidence': '80%'
            },
            'temperature_measurement': {
                'feasibility': 'Medium',
                'timeline': '1-2 years',
                'requirements': 'Cryogenic detectors and signal averaging',
                'confidence': '60%'
            },
            'quantum_correlation': {
                'feasibility': 'Low',
                'timeline': '3-5 years',
                'requirements': 'Advanced correlation diagnostics and theoretical development',
                'confidence': '30%'
            }
        },

        'facility_integration': {
            'eli_beamlines': {
                'compatibility': 'High',
                'advantages': 'High repetition rate, established diagnostic suite',
                'challenges': 'Limited maximum intensity compared to ELI-NP'
            },
            'eli_np': {
                'compatibility': 'Medium',
                'advantages': 'Highest available intensity, gamma-ray diagnostics',
                'challenges': 'Lower repetition rate, complex radiation environment'
            },
            'eli_alps': {
                'compatibility': 'Low',
                'advantages': 'High average power, long interaction length',
                'challenges': 'Different parameter regime, less suitable for horizon formation'
            }
        }
    }

    return recommendations


def main():
    """Main analysis pipeline"""

    parser = argparse.ArgumentParser(description='Comprehensive Detection Feasibility Analysis')
    parser.add_argument('--input', type=str, help='Input dataset file (CSV)')
    parser.add_argument('--output', type=str, default='results/detection_feasibility', help='Output directory')
    parser.add_argument('--n-configs', type=int, default=50, help='Number of configurations to analyze')
    parser.add_argument('--generate-report', action='store_true', help='Generate comprehensive report')

    args = parser.parse_args()

    print("🚀 Starting Comprehensive Detection Feasibility Analysis")
    print("=" * 60)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Load dataset
    if args.input:
        df = pd.read_csv(args.input)
        print(f"✅ Loaded input dataset: {len(df)} configurations")
    else:
        df = load_existing_dataset()

    # Limit number of configurations if specified
    if args.n_configs and len(df) > args.n_configs:
        df = df.head(args.n_configs)
        print(f"📊 Analyzing {len(df)} configurations")

    # Initialize detection feasibility analyzer
    print("\n🔧 Initializing detection feasibility analyzer...")
    analyzer = DetectionFeasibilityAnalyzer()

    # Analyze each configuration
    print(f"\n📈 Analyzing detection feasibility for {len(df)} configurations...")

    results = []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"  Analyzing configuration {i+1}/{len(df)}: {row.get('config_id', f'config_{i}')}")

        try:
            result = analyze_configuration_detection_feasibility(row, analyzer)
            results.append(result)
        except Exception as e:
            print(f"    ❌ Error: {e}")
            results.append({
                'config_id': row.get('config_id', f'config_{i}'),
                'error': str(e),
                'best_snr': 0.0,
                'feasibility_level': 'Error'
            })

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    results_df.to_csv(output_dir / 'detection_feasibility_results.csv', index=False)
    print(f"✅ Results saved to {output_dir / 'detection_feasibility_results.csv'}")

    # Generate summary statistics
    summary_stats = generate_summary_statistics(results_df)

    with open(output_dir / 'detection_feasibility_summary.json', 'w') as f:
        json.dump(summary_stats, f, indent=2, default=str)

    print("✅ Summary statistics generated")

    # Generate ELI facility recommendations
    eli_recommendations = generate_eli_facility_recommendations(results_df)

    with open(output_dir / 'eli_facility_recommendations.json', 'w') as f:
        json.dump(eli_recommendations, f, indent=2, default=str)

    print("✅ ELI facility recommendations generated")

    # Generate plots
    generate_detection_feasibility_plots(results_df, output_dir)

    # Generate comprehensive report if requested
    if args.generate_report:
        print("\n📄 Generating comprehensive report...")

        # Get top assessments for detailed analysis
        top_results = results_df[results_df['best_snr'] > 0].nlargest(5, 'best_snr')

        detailed_assessments = []
        for _, row in top_results.iterrows():
            try:
                signal = SignalCharacteristics(
                    hawking_temperature=row['hawking_temperature_K'],
                    surface_gravity=row['surface_gravity_s'],
                    peak_frequency=row['peak_frequency_Hz'],
                    bandwidth=row['peak_frequency_Hz'] / 10,
                    total_power=row.get('signal_power_W', 1e-25),
                    power_density=row.get('signal_power_W', 1e-25) / (row['peak_frequency_Hz'] / 10),
                    signal_temperature=row.get('signal_temperature_K', row['hawking_temperature_K'] * 0.1),
                    pulse_duration=1e-12,
                    rise_time=1e-13,
                    repetition_rate=1.0,
                    emitting_area=1e-12,
                    angular_distribution="isotropic"
                )

                plasma_params = {
                    'temperature': 1e6,
                    'density': row['plasma_density_m3']
                }

                assessments = analyzer.assess_detection_feasibility(signal, plasma_params)
                detailed_assessments.extend(assessments)

            except Exception as e:
                print(f"⚠️ Could not generate detailed assessment for {row['config_id']}: {e}")

        # Generate report
        if detailed_assessments:
            report = analyzer.generate_feasibility_report(
                detailed_assessments,
                output_dir / 'comprehensive_detection_feasibility_report.md'
            )
            print("✅ Comprehensive report generated")

    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)

    if len(results_df) > 0:
        feasible_count = len(results_df[results_df['best_snr'] >= 5])
        total_count = len(results_df[results_df['best_snr'] > 0])

        print(f"Total configurations analyzed: {len(results_df)}")
        print(f"Configurations with measurable signal: {total_count}")
        print(f"Feasible detections (SNR ≥ 5): {feasible_count}")
        print(f"Feasibility rate: {100*feasible_count/total_count:.1f}%" if total_count > 0 else "0%")

        if total_count > 0:
            print(f"Best SNR achieved: {results_df['best_snr'].max():.2f}")
            print(f"Mean detection probability: {results_df['detection_probability'].mean():.1%}")

            best_config = results_df.loc[results_df['best_snr'].idxmax()]
            print("\n🎯 BEST CONFIGURATION:")
            print(f"  Config ID: {best_config['config_id']}")
            print(f"  Method: {best_config['best_method']}")
            print(f"  SNR: {best_config['best_snr']:.2f}")
            print(f"  Detection Probability: {best_config['detection_probability']:.1%}")
            print(f"  Integration Time: {best_config['integration_time_s']:.2e} s")
            print(f"  Required Shots: {int(best_config['required_shots'])}")

    print(f"\n📁 Results saved to: {output_dir}")
    print("🎉 Analysis complete!")


if __name__ == "__main__":
    main()