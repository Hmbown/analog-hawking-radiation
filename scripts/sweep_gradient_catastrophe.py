#!/usr/bin/env python3
"""
Gradient Catastrophe Hunt: Finding the Maximum Achievable κ Before Physics Breakdown

This script explores the boundaries of valid physics in laser-plasma analog Hawking 
radiation systems by systematically increasing gradients until catastrophic failure.

Discovery Goal: Map the fundamental limits κ_max(a0, n_e, gradient) and identify 
the physics mechanisms that constrain achievable surface gravity in laboratory systems.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.constants import c, e, m_e, epsilon_0, k
import scipy.constants as const

# Ensure package imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.physics_engine.plasma_models.laser_plasma_interaction import MaxwellFluidModel
from analog_hawking.physics_engine.plasma_models.validation_protocols import PhysicsValidationFramework
from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.nonlinear_plasma import NonlinearPlasmaSolver


class GradientCatastropheDetector:
    """
    Detector for physics breakdown in extreme gradient regimes
    """
    
    def __init__(self, validation_tolerance: float = 0.1):
        self.validation_tolerance = validation_tolerance
        self.validator = PhysicsValidationFramework(validation_tolerance)
        
    def compute_gradient_metrics(self, x_grid: np.ndarray, velocity: np.ndarray, 
                                sound_speed: np.ndarray) -> Dict[str, float]:
        """Compute gradient steepness metrics"""
        dx = np.diff(x_grid)
        dv_dx = np.gradient(velocity, x_grid)
        dc_dx = np.gradient(sound_speed, x_grid)
        
        # Maximum gradient steepness
        max_velocity_gradient = np.max(np.abs(dv_dx))
        max_sound_gradient = np.max(np.abs(dc_dx))
        
        # Characteristic length scales
        horizon_length = np.mean(sound_speed) / np.mean(np.abs(dv_dx)) if np.any(dv_dx != 0) else np.inf
        
        # Relativistic parameter
        max_mach = np.max(np.abs(velocity / sound_speed)) if np.any(sound_speed > 0) else 0
        
        return {
            'max_velocity_gradient': max_velocity_gradient,
            'max_sound_gradient': max_sound_gradient, 
            'horizon_length_scale': horizon_length,
            'max_mach_number': max_mach,
            'gradient_steepness': max_velocity_gradient * horizon_length
        }
    
    def detect_physics_breakdown(self, simulation_data: Dict) -> Dict[str, any]:
        """
        Detect various physics breakdown modes
        
        Returns:
            Dict with breakdown flags and severity scores
        """
        breakdown_modes = {
            'relativistic_breakdown': False,
            'ionization_breakdown': False,
            'wave_breaking': False,
            'gradient_catastrophe': False,
            'intensity_breakdown': False,
            'numerical_instability': False,
            'validity_score': 1.0
        }
        
        # Check relativistic breakdown (v > 0.5c)
        if 'velocity' in simulation_data:
            velocity = simulation_data['velocity']
            max_v = np.max(np.abs(velocity)) if hasattr(velocity, '__iter__') else abs(velocity)
            if max_v > 0.5 * c:
                breakdown_modes['relativistic_breakdown'] = True
                breakdown_modes['validity_score'] *= 0.3
        
        # Check density spikes (negative or extremely high density)
        if 'density' in simulation_data:
            density = simulation_data['density']
            if hasattr(density, '__iter__'):
                if np.any(density <= 0) or np.any(density > 1e25):  # Beyond solid density
                    breakdown_modes['ionization_breakdown'] = True
                    breakdown_modes['validity_score'] *= 0.1
        
        # Check wave breaking (sound speed collapse)
        if 'sound_speed' in simulation_data:
            cs = simulation_data['sound_speed']
            if hasattr(cs, '__iter__') and np.any(cs <= 0):
                breakdown_modes['wave_breaking'] = True
                breakdown_modes['validity_score'] *= 0.2
        
        # Check gradient breakdown (dv/dx threshold) and catastrophic gradients
        if 'velocity' in simulation_data and 'space_grid' in simulation_data:
            x = simulation_data['space_grid']
            v = simulation_data['velocity']
            if hasattr(v, '__iter__') and len(x) > 1:
                dv_dx = np.gradient(v, x)
                # Relativistic gradient wall per documentation
                if np.any(np.abs(dv_dx) > 4e12):  # s^-1
                    breakdown_modes['gradient_catastrophe'] = True
                    breakdown_modes['validity_score'] *= 0.3
                # Catastrophic (fallback) for absurd gradients
                if np.any(np.abs(dv_dx) > 1e20):
                    breakdown_modes['validity_score'] *= 0.3
        
        # Check intensity breakdown if provided (I > 6e50 W/m^2)
        I = simulation_data.get('intensity', None)
        try:
            if I is not None and float(I) > 6e50:
                breakdown_modes['intensity_breakdown'] = True
                breakdown_modes['validity_score'] *= 0.3
        except Exception:
            pass

        # Run numerical stability check
        validation_results = self.validator.validate_numerical_stability(simulation_data)
        if not validation_results['numerically_stable']:
            breakdown_modes['numerical_instability'] = True
            breakdown_modes['validity_score'] *= 0.2
            
        return breakdown_modes


def run_single_configuration(a0: float, n_e: float, gradient_factor: float) -> Dict[str, any]:
    """
    Run a single configuration and extract physics metrics
    
    Args:
        a0: Normalized laser amplitude
        n_e: Plasma density (m^-3)  
        gradient_factor: Gradient steepness multiplier
        
    Returns:
        Dictionary with results including κ, validity, breakdown modes
    """
    
    # Calculate derived parameters
    lambda_l = 800e-9  # Standard Ti:Sapphire wavelength
    omega_l = 2 * np.pi * c / lambda_l
    
    # Intensity from a0: E0 = a0 * m_e * omega_l * c / e; I = 0.5 * ε0 * c * E0^2
    I_0 = 0.5 * epsilon_0 * c * (a0**2) * (m_e**2 * omega_l**2 * c**2) / (const.e**2)
    
    # Initialize laser-plasma model
    try:
        model = MaxwellFluidModel(
            plasma_density=n_e,
            laser_wavelength=lambda_l,
            laser_intensity=I_0
        )
    except Exception as e:
        return {
            'a0': a0, 'n_e': n_e, 'gradient_factor': gradient_factor,
            'kappa': 0.0, 'validity_score': 0.0, 'error': str(e),
            'breakdown_modes': {'numerical_instability': True}
        }
    
    # Create spatial grid with controlled gradient steepness
    L = 100e-6  # Domain length (100 μm)
    x = np.linspace(-L/2, L/2, 500)
    
    # Create velocity profile with tunable gradient that creates realistic horizons
    x_transition = 0.0
    sigma = L / (20 * gradient_factor)  # Steeper gradients for higher gradient_factor
    
    # Sound speed profile (varies with temperature/density)
    omega_pe = np.sqrt(const.e**2 * n_e / (epsilon_0 * m_e))
    cs_thermal = np.sqrt(k * 10000 / m_e)  # Thermal sound speed at 10000K
    
    # Create a sound speed that varies to enable horizon formation
    cs_base = cs_thermal * (1 + 0.2 * np.exp(-(x/sigma)**2))  # Gaussian bump
    sound_speed = cs_base
    
    # Create velocity that crosses sound speed (creates horizons)
    # Use a steeper profile that can exceed sound speed
    v_scale = cs_thermal * 1.5 * a0 * gradient_factor  # Scale with parameters
    velocity = v_scale * np.tanh((x - x_transition) / sigma)
    
    # Ensure we have horizons by adjusting profiles if needed
    max_v = np.max(np.abs(velocity))
    max_cs = np.max(sound_speed)
    if max_v < max_cs * 0.5:  # If velocity too small, scale it up
        velocity *= 2.0
    
    # Create simulation data dictionary
    simulation_data = {
        'space_grid': x,
        'velocity': velocity,
        'sound_speed': sound_speed,
        'density': np.full_like(x, n_e),
        'electric_field': np.zeros_like(x),  # Simplified
        'a0': a0,
        'plasma_density': n_e,
        'gradient_steepness': gradient_factor,
        'intensity': I_0,
    }
    
    # Initialize breakdown detector
    detector = GradientCatastropheDetector()
    
    # Check for physics breakdown
    breakdown_analysis = detector.detect_physics_breakdown(simulation_data)
    
    # Calculate horizon properties if physics is still valid
    if breakdown_analysis['validity_score'] > 0.1:
        try:
            horizons = find_horizons_with_uncertainty(x, velocity, sound_speed, kappa_method="acoustic_exact")
            kappa = np.mean(horizons.kappa) if len(horizons.kappa) > 0 else 0.0
            
            # Compute gradient metrics
            gradient_metrics = detector.compute_gradient_metrics(x, velocity, sound_speed)
            
        except Exception as e:
            kappa = 0.0
            gradient_metrics = {'error': str(e)}
            breakdown_analysis['validity_score'] = 0.0
    else:
        kappa = 0.0
        gradient_metrics = {}
    
    return {
        'a0': a0,
        'n_e': n_e, 
        'gradient_factor': gradient_factor,
        'kappa': kappa,
        'validity_score': breakdown_analysis['validity_score'],
        'breakdown_modes': breakdown_analysis,
        'gradient_metrics': gradient_metrics,
        'intensity': I_0,
        'max_velocity': np.max(np.abs(velocity)),
        'gradient_steepness': np.max(np.abs(np.gradient(velocity, x)))
    }


def run_gradient_catastrophe_sweep(n_samples: int = 500, output_dir: str = "results/gradient_limits") -> Dict:
    """
    Run comprehensive gradient catastrophe parameter sweep
    
    Args:
        n_samples: Number of parameter combinations to test
        output_dir: Directory to save results
        
    Returns:
        Dictionary with complete sweep results
    """
    
    print(f"GRADIENT CATASTROPHE HUNT: Mapping the Edge of Valid Physics")
    print(f"Testing {n_samples} configurations to find kappa_max...")
    print("="*70)
    
    # Define parameter ranges (logarithmic spacing for wide exploration)
    a0_range = np.logspace(0, 2, 20)  # 1 to 100 (dimensionless laser amplitude)
    n_e_range = np.logspace(18, 22, 15)  # 10^18 to 10^22 m^-3 (underdense to overcritical)
    gradient_range = np.logspace(0, 3, 10)  # 1 to 1000 (gradient steepness factor)
    
    # Generate parameter combinations (stratified sampling)
    results = []
    param_combinations = []
    
    # Create grid of parameters
    total_combinations = len(a0_range) * len(n_e_range) * len(gradient_range)
    if total_combinations > n_samples:
        # Random sampling from parameter space
        np.random.seed(42)  # Reproducible results
        for _ in range(n_samples):
            a0 = np.random.choice(a0_range)
            n_e = np.random.choice(n_e_range)
            gradient_factor = np.random.choice(gradient_range)
            param_combinations.append((a0, n_e, gradient_factor))
    else:
        # Full grid if small enough
        for a0 in a0_range:
            for n_e in n_e_range:
                for gradient_factor in gradient_range:
                    param_combinations.append((a0, n_e, gradient_factor))
    
    # Run sweep with progress bar
    with tqdm(total=len(param_combinations), desc="Exploring parameter space") as pbar:
        for a0, n_e, gradient_factor in param_combinations:
            result = run_single_configuration(a0, n_e, gradient_factor)
            results.append(result)
            
            # Update progress bar with current max kappa
            valid_kappas = [r['kappa'] for r in results if r['validity_score'] > 0.5]
            max_kappa = max(valid_kappas) if valid_kappas else 0
            pbar.set_postfix({
                'max_kappa': f"{max_kappa:.2e}",
                'valid': f"{len(valid_kappas)}/{len(results)}"
            })
            pbar.update(1)
    
    # Analyze results
    analysis = analyze_catastrophe_boundaries(results)
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    sweep_data = {
        'results': results,
        'analysis': analysis,
        'n_samples': len(results),
        'parameter_ranges': {
            'a0_range': a0_range.tolist(),
            'n_e_range': n_e_range.tolist(),
            'gradient_range': gradient_range.tolist()
        }
    }
    
    with open(output_path / "gradient_catastrophe_sweep.json", "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    
    print(f"\nSWEEP COMPLETE!")
    print(f"Results saved to: {output_path}")
    print(f"Configurations tested: {len(results)}")
    print(f"Valid configurations: {len([r for r in results if r['validity_score'] > 0.5])}")
    print(f"Maximum kappa achieved: {analysis['max_kappa']:.2e} Hz")
    print(f"Breakdown rate: {analysis['breakdown_statistics']['total_breakdown_rate']:.1%}")
    
    return sweep_data


def analyze_catastrophe_boundaries(results: List[Dict]) -> Dict:
    """
    Analyze the boundaries where physics breaks down
    
    Args:
        results: List of simulation results
        
    Returns:
        Dictionary with boundary analysis
    """
    
    # Filter valid vs invalid results
    valid_results = [r for r in results if r['validity_score'] > 0.5]
    invalid_results = [r for r in results if r['validity_score'] <= 0.5]
    
    # Find maximum achievable κ
    kappas = [r['kappa'] for r in valid_results]
    max_kappa = max(kappas) if kappas else 0.0
    max_kappa_config = max(valid_results, key=lambda r: r['kappa']) if valid_results else None
    
    # Analyze breakdown modes
    breakdown_stats = {}
    for mode in ['relativistic_breakdown', 'ionization_breakdown', 'wave_breaking', 
                 'gradient_catastrophe', 'intensity_breakdown', 'numerical_instability']:
        count = sum(1 for r in results if r.get('breakdown_modes', {}).get(mode, False))
        breakdown_stats[mode] = {
            'count': count,
            'rate': count / len(results) if results else 0
        }
    
    breakdown_stats['total_breakdown_rate'] = len(invalid_results) / len(results) if results else 0
    
    # Find scaling relationships for valid regime
    valid_a0 = [r['a0'] for r in valid_results]
    valid_n_e = [r['n_e'] for r in valid_results]
    valid_kappa = [r['kappa'] for r in valid_results]
    
    # Fit power law relationships (log-log regression) with simple 95% CI
    if len(valid_results) > 10:
        try:
            # κ vs a0 scaling
            log_a0 = np.log10(valid_a0)
            log_kappa = np.log10([k for k in valid_kappa if k > 0])
            log_a0_clean = [log_a0[i] for i, k in enumerate(valid_kappa) if k > 0]
            
            if len(log_kappa) > 5:
                # slope and intercept
                coeffs = np.polyfit(log_a0_clean, log_kappa, 1)
                kappa_a0_slope = coeffs[0]
                # standard error and 95% CI for slope
                x = np.asarray(log_a0_clean)
                y = np.asarray(log_kappa)
                yhat = coeffs[0] * x + coeffs[1]
                resid = y - yhat
                dof = max(len(x) - 2, 1)
                s_yx = float(np.sqrt(np.sum(resid**2) / dof))
                s_xx = float(np.sum((x - x.mean())**2)) if len(x) > 1 else 1.0
                se_slope = s_yx / np.sqrt(max(s_xx, 1e-30))
                a0_slope_ci95 = (kappa_a0_slope - 1.96 * se_slope, kappa_a0_slope + 1.96 * se_slope)
            else:
                kappa_a0_slope = np.nan
                a0_slope_ci95 = (np.nan, np.nan)
                
            # κ vs n_e scaling  
            log_n_e = np.log10(valid_n_e)
            log_n_e_clean = [log_n_e[i] for i, k in enumerate(valid_kappa) if k > 0]
            
            if len(log_kappa) > 5:
                coeffs_ne = np.polyfit(log_n_e_clean, log_kappa, 1)
                kappa_ne_slope = coeffs_ne[0]
                xn = np.asarray(log_n_e_clean)
                yn = np.asarray(log_kappa)
                yhatn = coeffs_ne[0] * xn + coeffs_ne[1]
                residn = yn - yhatn
                dofn = max(len(xn) - 2, 1)
                s_yxn = float(np.sqrt(np.sum(residn**2) / dofn))
                s_xxn = float(np.sum((xn - xn.mean())**2)) if len(xn) > 1 else 1.0
                se_slopen = s_yxn / np.sqrt(max(s_xxn, 1e-30))
                ne_slope_ci95 = (kappa_ne_slope - 1.96 * se_slopen, kappa_ne_slope + 1.96 * se_slopen)
            else:
                kappa_ne_slope = np.nan
                ne_slope_ci95 = (np.nan, np.nan)
                
        except Exception:
            kappa_a0_slope = np.nan
            kappa_ne_slope = np.nan
            a0_slope_ci95 = (np.nan, np.nan)
            ne_slope_ci95 = (np.nan, np.nan)
    else:
        kappa_a0_slope = np.nan
        kappa_ne_slope = np.nan
        a0_slope_ci95 = (np.nan, np.nan)
        ne_slope_ci95 = (np.nan, np.nan)
    
    return {
        'max_kappa': max_kappa,
        'max_kappa_config': max_kappa_config,
        'valid_configurations': len(valid_results),
        'invalid_configurations': len(invalid_results),
        'breakdown_statistics': breakdown_stats,
        'scaling_relationships': {
            'kappa_vs_a0_exponent': kappa_a0_slope,
            'kappa_vs_ne_exponent': kappa_ne_slope,
            'kappa_vs_a0_exponent_ci95': a0_slope_ci95,
            'kappa_vs_ne_exponent_ci95': ne_slope_ci95,
        },
        'parameter_boundaries': {
            'max_valid_a0': max([r['a0'] for r in valid_results]) if valid_results else 0,
            'max_valid_ne': max([r['n_e'] for r in valid_results]) if valid_results else 0,
            'max_valid_gradient': max([r['gradient_factor'] for r in valid_results]) if valid_results else 0
        }
    }


def generate_catastrophe_plots(sweep_data: Dict, output_dir: str = "results/gradient_limits"):
    """Generate publication-quality plots of gradient catastrophe boundaries"""
    
    results = sweep_data['results']
    analysis = sweep_data['analysis']
    
    # Set up plotting style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gradient Catastrophe: Mapping the Limits of Analog Hawking Physics', fontsize=16)
    
    # Plot 1: κ vs a0 with validity coloring
    ax1 = axes[0, 0]
    valid_results = [r for r in results if r['validity_score'] > 0.5]
    invalid_results = [r for r in results if r['validity_score'] <= 0.5]
    
    if valid_results:
        ax1.scatter([r['a0'] for r in valid_results], [r['kappa'] for r in valid_results],
                   c='green', alpha=0.6, label='Valid Physics', s=30)
    if invalid_results:
        ax1.scatter([r['a0'] for r in invalid_results], [max(r['kappa'], 1e6) for r in invalid_results],
                   c='red', alpha=0.6, label='Physics Breakdown', s=30, marker='x')
    
    ax1.set_xlabel('Normalized Laser Amplitude (a₀)')
    ax1.set_ylabel('Surface Gravity κ (Hz)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Breakdown mode statistics
    ax2 = axes[0, 1]
    breakdown_modes = ['relativistic_breakdown', 'ionization_breakdown', 'wave_breaking', 
                      'gradient_catastrophe', 'numerical_instability']
    breakdown_rates = [analysis['breakdown_statistics'][mode]['rate'] for mode in breakdown_modes]
    mode_labels = ['Relativistic', 'Ionization', 'Wave Breaking', 'Gradient', 'Numerical']
    
    bars = ax2.bar(mode_labels, breakdown_rates, alpha=0.7)
    ax2.set_ylabel('Breakdown Rate')
    ax2.set_title('Physics Breakdown Modes')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Add percentage labels on bars
    for bar, rate in zip(bars, breakdown_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{rate:.1%}', ha='center', va='bottom')
    
    # Plot 3: κ_max vs parameter space
    ax3 = axes[1, 0]
    if valid_results:
        # Create heatmap of maximum κ in (a0, n_e) space
        a0_vals = np.array([r['a0'] for r in valid_results])
        ne_vals = np.array([r['n_e'] for r in valid_results])
        kappa_vals = np.array([r['kappa'] for r in valid_results])
        
        scatter = ax3.scatter(a0_vals, ne_vals, c=kappa_vals, s=50, alpha=0.7, 
                            cmap='viridis', norm=plt.Normalize(vmin=kappa_vals.min(), vmax=kappa_vals.max()))
        ax3.set_xlabel('Normalized Laser Amplitude (a₀)')
        ax3.set_ylabel('Plasma Density (m⁻³)')
        ax3.set_xscale('log')
        ax3.set_yscale('log')
        
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('κ (Hz)')
    
    # Plot 4: Validity score distribution
    ax4 = axes[1, 1]
    validity_scores = [r['validity_score'] for r in results]
    ax4.hist(validity_scores, bins=20, alpha=0.7, edgecolor='black')
    ax4.axvline(x=0.5, color='red', linestyle='--', label='Validity Threshold')
    ax4.set_xlabel('Validity Score')
    ax4.set_ylabel('Number of Configurations')
    ax4.set_title('Distribution of Physics Validity')
    ax4.legend()
    
    plt.tight_layout()
    
    # Save plots
    output_path = Path(output_dir)
    plt.savefig(output_path / "gradient_catastrophe_analysis.png", dpi=300, bbox_inches='tight')
    plt.savefig(output_path / "gradient_catastrophe_analysis.pdf", bbox_inches='tight')
    
    print(f"Plots saved to {output_path}")
    
    return fig


def main():
    """Main entry point for gradient catastrophe hunt"""
    
    import argparse
    parser = argparse.ArgumentParser(description="Hunt for gradient catastrophe boundaries")
    parser.add_argument("--n-samples", type=int, default=500, 
                       help="Number of parameter combinations to test")
    parser.add_argument("--output", type=str, default="results/gradient_limits",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run the sweep
    sweep_data = run_gradient_catastrophe_sweep(args.n_samples, args.output)
    
    # Generate plots
    generate_catastrophe_plots(sweep_data, args.output)
    
    # Print key findings
    analysis = sweep_data['analysis']
    print(f"\nKEY FINDINGS:")
    print(f"Maximum achievable kappa: {analysis['max_kappa']:.2e} Hz")
    
    if analysis['max_kappa_config']:
        config = analysis['max_kappa_config']
        print(f"Optimal configuration:")
        print(f"  - a0 = {config['a0']:.2f}")  
        print(f"  - n_e = {config['n_e']:.2e} m^-3")
        print(f"  - Gradient factor = {config['gradient_factor']:.1f}")
        print(f"  - Required intensity = {config['intensity']:.2e} W/m^2")
    
    print(f"Scaling relationships:")
    if not np.isnan(analysis['scaling_relationships']['kappa_vs_a0_exponent']):
        print(f"  - kappa ∝ a0^{analysis['scaling_relationships']['kappa_vs_a0_exponent']:.2f}")
    if not np.isnan(analysis['scaling_relationships']['kappa_vs_ne_exponent']):
        print(f"  - kappa ∝ n_e^{analysis['scaling_relationships']['kappa_vs_ne_exponent']:.2f}")
    
    print(f"\nMost common breakdown: {max(analysis['breakdown_statistics'].items(), key=lambda x: x[1]['rate'])[0]}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
