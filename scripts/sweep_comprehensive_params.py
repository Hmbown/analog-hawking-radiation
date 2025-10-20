#!/usr/bin/env python3
"""
Comprehensive parameter sweep for analog Hawking radiation detection.
Systematically explores laser intensity, plasma density, magnetic field, 
and geometry parameters to identify optimal detection conditions.
"""

import json
import os
import numpy as np
from pathlib import Path
import concurrent.futures
from typing import Dict, List, Tuple, Any
import argparse
import logging
from dataclasses import dataclass, asdict

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.run_full_pipeline import run_full_pipeline, FullPipelineSummary

@dataclass
class SweepConfig:
    """Configuration for parameter sweeps"""
    laser_intensity_range: Tuple[float, float, float]  # (min, max, step)
    plasma_density_range: Tuple[float, float, float]
    magnetic_field_range: Tuple[float, float, float]
    temperature_range: Tuple[float, float, float]
    grid_size_range: Tuple[float, float, float]
    hybrid_D_range: Tuple[float, float, float]
    hybrid_eta_range: Tuple[float, float, float]
    n_samples: int = 1000
    n_workers: int = 4
    output_dir: str = "results/sweeps"

class ParameterSweep:
    """Handles systematic parameter exploration"""
    
    def __init__(self, config: SweepConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'sweep.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def generate_parameter_grid(self) -> List[Dict[str, float]]:
        """Generate parameter combinations for systematic exploration"""
        
        # Create linear spaces for each parameter
        intensity_vals = np.logspace(
            np.log10(self.config.laser_intensity_range[0]),
            np.log10(self.config.laser_intensity_range[1]),
            int((self.config.laser_intensity_range[1] / self.config.laser_intensity_range[0]) ** 0.5 * 10)
        )
        
        density_vals = np.logspace(
            np.log10(self.config.plasma_density_range[0]),
            np.log10(self.config.plasma_density_range[1]),
            int((self.config.plasma_density_range[1] / self.config.plasma_density_range[0]) ** 0.5 * 10)
        )
        
        magnetic_vals = np.linspace(
            self.config.magnetic_field_range[0],
            self.config.magnetic_field_range[1],
            max(5, int((self.config.magnetic_field_range[1] - self.config.magnetic_field_range[0]) / 20) + 1)
        )
        
        temperature_vals = np.logspace(
            np.log10(self.config.temperature_range[0]),
            np.log10(self.config.temperature_range[1]),
            10
        )
        
        grid_size_vals = np.linspace(
            self.config.grid_size_range[0],
            self.config.grid_size_range[1],
            8
        )
        
        hybrid_D_vals = np.logspace(
            np.log10(self.config.hybrid_D_range[0]),
            np.log10(self.config.hybrid_D_range[1]),
            6
        )
        
        hybrid_eta_vals = np.linspace(
            self.config.hybrid_eta_range[0],
            self.config.hybrid_eta_range[1],
            5
        )
        
        # Generate combinations (reduced for computational efficiency)
        combinations = []
        for intensity in intensity_vals[:8]:  # Limit to reduce computation
            for density in density_vals[:8]:
                for magnetic in [0, 50, 100]:  # Fewer magnetic field values
                    for temperature in [1e4, 5e4]:  # Fewer temperature values
                        for grid_size in [50e-6]:  # Fixed grid size for now
                            combinations.append({
                                'laser_intensity': float(intensity),
                                'plasma_density': float(density),
                                'magnetic_field': float(magnetic) if magnetic > 0 else None,
                                'temperature_constant': float(temperature),
                                'grid_max': float(grid_size),
                                'grid_points': 512,
                            })
        
        # Add hybrid configurations
        hybrid_combinations = []
        for intensity in [5e17, 1e18, 5e18]:  # Focus on high-intensity regimes
            for density in [5e17, 1e18, 5e18]:
                for D in hybrid_D_vals:
                    for eta in hybrid_eta_vals:
                        hybrid_combinations.append({
                            'laser_intensity': float(intensity),
                            'plasma_density': float(density),
                            'magnetic_field': None,
                            'temperature_constant': 1e4,
                            'enable_hybrid': True,
                            'mirror_D': float(D),
                            'mirror_eta': float(eta),
                            'grid_max': 50e-6,
                            'grid_points': 512,
                        })
        
        return combinations + hybrid_combinations
    
    def run_single_parameter_set(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run pipeline for a single parameter set"""
        try:
            result = run_full_pipeline(**params)
            
            # Convert to serializable format
            serializable_result = asdict(result)
            
            # Add parameter set
            serializable_result['parameters'] = params
            
            return serializable_result
            
        except Exception as e:
            self.logger.error(f"Error processing parameters {params}: {e}")
            return {
                'parameters': params,
                'error': str(e),
                'kappa': [],
                't5sigma_s': None,
                'T_sig_K': None
            }
    
    def run_sweep(self, use_hybrid: bool = False) -> List[Dict[str, Any]]:
        """Execute the parameter sweep"""
        
        self.logger.info("Generating parameter combinations...")
        parameter_sets = self.generate_parameter_grid()
        
        if use_hybrid:
            # Filter for hybrid configurations
            parameter_sets = [p for p in parameter_sets if p.get('enable_hybrid', False)]
        else:
            # Filter for non-hybrid configurations
            parameter_sets = [p for p in parameter_sets if not p.get('enable_hybrid', False)]
        
        self.logger.info(f"Running sweep with {len(parameter_sets)} parameter combinations")
        
        results = []
        
        # Use parallel processing for efficiency
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_params = {
                executor.submit(self.run_single_parameter_set, params): params
                for params in parameter_sets
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(future_to_params)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if i % 10 == 0:
                        self.logger.info(f"Completed {i+1}/{len(parameter_sets)} runs")
                        
                        # Save intermediate results
                        self.save_results(results, f"intermediate_{i+1}.json")
                        
                except Exception as e:
                    params = future_to_params[future]
                    self.logger.error(f"Failed to process {params}: {e}")
        
        return results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = None):
        """Save sweep results to JSON file"""
        if filename is None:
            filename = f"sweep_results_{len(results)}.json"
        
        output_file = self.output_dir / filename
        
        with open(output_file, 'w') as f:
            json.dump({
                'config': asdict(self.config),
                'results': results,
                'summary': self.generate_summary(results)
            }, f, indent=2, default=str)
        
        self.logger.info(f"Results saved to {output_file}")
    
    def generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate summary statistics from sweep results"""
        
        valid_results = [r for r in results if 'error' not in r and r.get('kappa')]
        
        if not valid_results:
            return {'error': 'No valid results found'}
        
        # Extract key metrics
        kappa_values = [r['kappa'][0] for r in valid_results if r.get('kappa')]
        t5sigma_values = [r['t5sigma_s'] for r in valid_results if r.get('t5sigma_s') is not None]
        T_sig_values = [r['T_sig_K'] for r in valid_results if r.get('T_sig_K') is not None]
        
        summary = {
            'total_runs': len(results),
            'valid_runs': len(valid_results),
            'kappa_stats': {
                'min': float(np.min(kappa_values)) if kappa_values else None,
                'max': float(np.max(kappa_values)) if kappa_values else None,
                'mean': float(np.mean(kappa_values)) if kappa_values else None,
                'median': float(np.median(kappa_values)) if kappa_values else None,
            },
            't5sigma_stats': {
                'min': float(np.min(t5sigma_values)) if t5sigma_values else None,
                'max': float(np.max(t5sigma_values)) if t5sigma_values else None,
                'mean': float(np.mean(t5sigma_values)) if t5sigma_values else None,
                'median': float(np.median(t5sigma_values)) if t5sigma_values else None,
            },
            'best_parameters': None,
            'worst_parameters': None
        }
        
        # Find best and worst cases
        if t5sigma_values:
            best_idx = np.argmin(t5sigma_values)
            worst_idx = np.argmax(t5sigma_values)
            
            summary['best_parameters'] = {
                't5sigma_s': float(t5sigma_values[best_idx]),
                'parameters': valid_results[best_idx]['parameters']
            }
            
            summary['worst_parameters'] = {
                't5sigma_s': float(t5sigma_values[worst_idx]),
                'parameters': valid_results[worst_idx]['parameters']
            }
        
        return summary
    
    def generate_heatmap_data(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data for detection time heatmaps"""
        
        valid_results = [r for r in results if 'error' not in r and r.get('t5sigma_s') is not None]
        
        if not valid_results:
            return {'error': 'No valid results for heatmap'}
        
        # Extract 2D parameter space (intensity vs density)
        intensity_density_data = []
        for result in valid_results:
            params = result['parameters']
            if 'magnetic_field' not in params or params['magnetic_field'] is None:
                intensity_density_data.append({
                    'intensity': params['laser_intensity'],
                    'density': params['plasma_density'],
                    't5sigma': result['t5sigma_s'],
                    'kappa': result['kappa'][0] if result.get('kappa') else None
                })
        
        return {
            'intensity_density': intensity_density_data,
            'hybrid_data': [
                {
                    'intensity': r['parameters']['laser_intensity'],
                    'density': r['parameters']['plasma_density'],
                    'mirror_D': r['parameters']['mirror_D'],
                    'mirror_eta': r['parameters']['mirror_eta'],
                    't5sigma': r['t5sigma_s'],
                    'kappa': r['kappa'][0] if r.get('kappa') else None
                }
                for r in valid_results
                if r['parameters'].get('enable_hybrid', False)
            ]
        }

def main():
    parser = argparse.ArgumentParser(description="Parameter sweep for analog Hawking radiation")
    parser.add_argument("--coarse", action="store_true", help="Run coarse parameter sweep")
    parser.add_argument("--fine", action="store_true", help="Run fine parameter sweep")
    parser.add_argument("--hybrid", action="store_true", help="Run hybrid parameter sweep")
    parser.add_argument("--output", default="results/sweeps", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples")
    
    args = parser.parse_args()
    
    # Default configuration
    config = SweepConfig(
        laser_intensity_range=(1e16, 1e20, 2e17),
        plasma_density_range=(1e16, 1e20, 2e17),
        magnetic_field_range=(0, 100, 20),
        temperature_range=(1e3, 1e5, 5e3),
        grid_size_range=(10e-6, 100e-6, 10e-6),
        hybrid_D_range=(1e-6, 100e-6, 20e-6),
        hybrid_eta_range=(0.1, 10, 1),
        n_samples=args.samples,
        n_workers=args.workers,
        output_dir=args.output
    )
    
    sweeper = ParameterSweep(config)
    
    if args.coarse:
        print("Running coarse parameter sweep...")
        results = sweeper.run_sweep(use_hybrid=False)
        sweeper.save_results(results, "coarse_sweep_results.json")
        
        # Generate heatmap data
        heatmap_data = sweeper.generate_heatmap_data(results)
        with open(Path(args.output) / "coarse_heatmap_data.json", 'w') as f:
            json.dump(heatmap_data, f, indent=2)
    
    if args.fine:
        print("Running fine parameter sweep...")
        # Focus on promising regions from coarse sweep
        config.laser_intensity_range = (1e17, 1e19, 1e17)
        config.plasma_density_range = (1e17, 1e19, 1e17)
        sweeper = ParameterSweep(config)
        results = sweeper.run_sweep(use_hybrid=False)
        sweeper.save_results(results, "fine_sweep_results.json")
    
    if args.hybrid:
        print("Running hybrid parameter sweep...")
        results = sweeper.run_sweep(use_hybrid=True)
        sweeper.save_results(results, "hybrid_sweep_results.json")
        
        # Generate hybrid heatmap data
        heatmap_data = sweeper.generate_heatmap_data(results)
        with open(Path(args.output) / "hybrid_heatmap_data.json", 'w') as f:
            json.dump(heatmap_data, f, indent=2)
    
    if not any([args.coarse, args.fine, args.hybrid]):
        print("Running default parameter sweep...")
        results = sweeper.run_sweep(use_hybrid=False)
        sweeper.save_results(results, "default_sweep_results.json")

if __name__ == "__main__":
    main()
