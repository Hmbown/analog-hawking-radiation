#!/usr/bin/env python3
"""
MOE-Orchestrated Gradient Catastrophe Sweep

This script runs the gradient catastrophe sweep through the MOE system,
using the HawkingRadiationExpert to preserve spatial coupling information.

The key enhancement: Using MOE's expert system to orchestrate physics
computations with spatially resolved kappa values.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "moeoe"))

from analog_hawking.config.thresholds import Thresholds
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from moeoe.core.unified_moe import UnifiedMOE


class MOEOrchestratedSweep:
    """Gradient catastrophe sweep orchestrated through MOE"""
    
    def __init__(self, output_dir: str = "results/moe_orchestrated"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize MOE and register physics expert
        self.moe = UnifiedMOE()
        self.moe.register_expert(
            'hawking_physics', 
            'moeoe.core.experts.physics.hawking_radiation_expert.HawkingRadiationExpert'
        )
        
        self.thresholds = Thresholds()
        self.results = []
        
    def build_plasma_profile(self, a0: float, n_e: float, gradient_factor: float) -> Dict[str, np.ndarray]:
        """Build a plasma profile for MOE processing"""
        # Setup grid
        x_max = 50e-6
        x = np.linspace(0, x_max, 1024)
        
        # Calculate laser intensity from a0
        from scipy.constants import c, m_e, e, epsilon_0
        lambda_laser = 800e-9
        omega = 2 * np.pi * c / lambda_laser
        E0 = a0 * m_e * c * omega / e
        I_0 = 0.5 * epsilon_0 * c * E0**2
        
        # Use fluid backend for reliable plasma state
        backend = FluidBackend()
        backend.configure({
            "plasma_density": max(n_e, 1e18),
            "laser_wavelength": lambda_laser,
            "laser_intensity": min(I_0, 1e18),
            "grid": x,
            "temperature_settings": {"constant": 1e4},
            "use_fast_magnetosonic": False,
            "scale_with_intensity": True,
        })
        
        try:
            state = backend.step(0.0)
            velocity = state.velocity
            sound_speed = state.sound_speed
            
            # Apply gradient factor to increase steepness
            if gradient_factor > 1.0:
                x_compressed = np.linspace(x[0], x[-1] / gradient_factor, len(x))
                velocity = np.interp(x_compressed, x, velocity)
                sound_speed = np.interp(x_compressed, x, sound_speed)
                x = x_compressed
                
        except Exception as e:
            print(f"Warning: Fluid backend failed: {e}, using synthetic profile")
            # Fallback to synthetic profile
            velocity = np.zeros_like(x)
            velocity[len(x)//3:len(x)//3+100] = np.linspace(0, 1.5e6 * min(a0/10, 1), 100)
            sound_speed = np.ones_like(x) * 1e6
            
        return {
            'x': x,
            'v': velocity,
            'c_s': sound_speed,
            'intensity': I_0
        }
    
    def evaluate_configuration(self, a0: float, n_e: float, gradient_factor: float) -> Dict[str, Any]:
        """Evaluate a configuration through MOE"""
        
        # Build plasma profile
        profile = self.build_plasma_profile(a0, n_e, gradient_factor)
        
        # Query MOE with spatial coupling
        response = self.moe.process(
            query='Compute Hawking spectrum with spatial coupling',
            plasma_profile=profile,
            method='spatial_coupling',
            return_per_patch=True
        )
        
        # Extract results
        if response.result and 'error' not in response.result:
            result = response.result
            kappa_max = result['kappa_max']
            kappa_mean = result['kappa_mean']
            kappa_std = result['kappa_std']
            n_horizons = result['n_horizons']
            
            # Also run legacy method for comparison
            legacy_response = self.moe.process(
                query='Compute Hawking spectrum with averaged method',
                plasma_profile=profile,
                method='averaged',
                return_per_patch=True
            )
            
            legacy_kappa_max = 0
            if legacy_response.result and 'error' not in legacy_response.result:
                legacy_kappa_max = legacy_response.result['kappa_max']
            
            # Calculate enhancement
            enhancement = kappa_max / legacy_kappa_max if legacy_kappa_max > 0 else 1.0
            
        else:
            # Failed case
            kappa_max = 0.0
            kappa_mean = 0.0
            kappa_std = 0.0
            n_horizons = 0
            enhancement = 1.0
            
        return {
            'a0': a0,
            'n_e': n_e,
            'gradient_factor': gradient_factor,
            'kappa_max': kappa_max,
            'kappa_mean': kappa_mean,
            'kappa_std': kappa_std,
            'n_horizons': n_horizons,
            'enhancement_factor': enhancement,
            'max_velocity': np.max(np.abs(profile['v'])),
            'intensity': profile['intensity'],
            'moe_confidence': response.confidence,
            'moe_experts_invoked': response.metadata.get('experts_invoked', []),
            'moe_processing_time_ms': response.processing_time_ms
        }
    
    def run_sweep(self, n_samples: int = 100) -> Dict[str, Any]:
        """Run the MOE-orchestrated parameter sweep"""
        
        print("MOE-ORCHESTRATED GRADIENT CATASTROPHE SWEEP")
        print("=" * 60)
        print(f"Testing {n_samples} configurations...")
        print(f"MOE registered experts: {len(self.moe.experts)}")
        print()
        
        # Define parameter ranges
        a0_range = np.logspace(0, 2, 15)  # 1 to 100
        n_e_range = np.logspace(18, 22, 12)  # 10^18 to 10^22
        gradient_range = np.logspace(0, 2.5, 10)  # 1 to ~316
        
        # Sample parameter space
        results = []
        
        for i in tqdm(range(n_samples), desc="MOE Sweep Progress"):
            a0 = np.random.choice(a0_range)
            n_e = np.random.choice(n_e_range)
            gradient_factor = np.random.choice(gradient_range)
            
            result = self.evaluate_configuration(a0, n_e, gradient_factor)
            results.append(result)
            
            # Print progress for significant results
            if result['kappa_max'] > 1e12:  # Interesting threshold
                print(f"  Sample {i}: κ_max={result['kappa_max']:.2e} Hz, "
                      f"enhancement={result['enhancement_factor']:.2f}x")
        
        self.results = results
        
        # Analyze results
        analysis = self.analyze_results()
        
        # Save results
        self.save_results(analysis)
        
        return analysis
    
    def analyze_results(self) -> Dict[str, Any]:
        """Analyze sweep results"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Convert to arrays for analysis
        kappa_max_values = np.array([r['kappa_max'] for r in self.results])
        enhancement_values = np.array([r['enhancement_factor'] for r in self.results])
        valid_mask = kappa_max_values > 0
        
        analysis = {
            'total_samples': len(self.results),
            'valid_samples': int(np.sum(valid_mask)),
            'kappa_max_mean': float(np.mean(kappa_max_values[valid_mask])),
            'kappa_max_std': float(np.std(kappa_max_values[valid_mask])),
            'kappa_max_max': float(np.max(kappa_max_values[valid_mask])),
            'enhancement_mean': float(np.mean(enhancement_values[valid_mask])),
            'enhancement_max': float(np.max(enhancement_values[valid_mask])),
            'moe_avg_confidence': float(np.mean([r['moe_confidence'] for r in self.results])),
            'moe_avg_time_ms': float(np.mean([r['moe_processing_time_ms'] for r in self.results]))
        }
        
        # Find best configuration
        if np.any(valid_mask):
            best_idx = np.argmax(kappa_max_values[valid_mask])
            best_result = self.results[best_idx]
            analysis['best_configuration'] = best_result
            
            print(f"\nSWEEP SUMMARY:")
            print(f"  Valid samples: {analysis['valid_samples']}/{analysis['total_samples']}")
            print(f"  Peak κ_max: {analysis['kappa_max_max']:.2e} Hz")
            print(f"  Mean enhancement: {analysis['enhancement_mean']:.2f}x")
            print(f"  Max enhancement: {analysis['enhancement_max']:.2f}x")
            print(f"  MOE avg confidence: {analysis['moe_avg_confidence']:.3f}")
            print(f"  MOE avg time: {analysis['moe_avg_time_ms']:.1f} ms")
        
        return analysis
    
    def save_results(self, analysis: Dict[str, Any]):
        """Save results and create visualizations"""
        
        # Save JSON results
        results_file = self.output_dir / "moe_sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'analysis': analysis,
                'results': self.results
            }, f, indent=2)
        
        # Create comparison plot
        self.create_comparison_plot()
        
        print(f"\nResults saved to: {self.output_dir}")
        
    def create_comparison_plot(self):
        """Create visualization comparing spatial vs averaged methods"""
        if not self.results:
            return
            
        # Extract data for plotting
        kappa_max_spatial = [r['kappa_max'] for r in self.results]
        kappa_max_legacy = [r['kappa_max'] / r['enhancement_factor'] for r in self.results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: kappa_max comparison
        ax1.scatter(kappa_max_legacy, kappa_max_spatial, alpha=0.6)
        ax1.plot([1e10, 1e14], [1e10, 1e14], 'r--', label='1:1 line')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.set_xlabel('Legacy κ_max (averaged) [Hz]')
        ax1.set_ylabel('Spatial κ_max (coupled) [Hz]')
        ax1.set_title('Spatial Coupling Enhancement')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Enhancement distribution
        enhancements = [r['enhancement_factor'] for r in self.results if r['enhancement_factor'] > 1]
        ax2.hist(enhancements, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Enhancement Factor (spatial/legacy)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Enhancement Factors')
        ax2.grid(True, alpha=0.3)
        
        # Add mean enhancement line
        if enhancements:
            mean_enhancement = np.mean(enhancements)
            ax2.axvline(mean_enhancement, color='red', linestyle='--', 
                       label=f'Mean: {mean_enhancement:.2f}x')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'moe_enhancement_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run MOE-orchestrated gradient catastrophe sweep')
    parser.add_argument('--n-samples', type=int, default=50, help='Number of samples to test')
    parser.add_argument('--output-dir', type=str, default='results/moe_orchestrated', 
                       help='Output directory for results')
    parser.add_argument('--compare-legacy', action='store_true', 
                       help='Include legacy method comparison')
    
    args = parser.parse_args()
    
    # Run sweep
    sweep = MOEOrchestratedSweep(output_dir=args.output_dir)
    analysis = sweep.run_sweep(n_samples=args.n_samples)
    
    print(f"\n✅ MOE-orchestrated sweep complete!")
    print(f"   Results saved to: {args.output_dir}")
    
    return analysis


if __name__ == "__main__":
    main()