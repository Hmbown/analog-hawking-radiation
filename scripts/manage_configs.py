#!/usr/bin/env python3
"""
Configuration management for the analog Hawking radiation experimental protocol.
Manages parameter sweeps, optimization settings, and experimental configurations.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import argparse

class ConfigManager:
    """Manages configuration files for the experimental protocol"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def create_sweep_configs(self):
        """Create parameter sweep configuration files"""
        
        # Coarse sweep configuration
        coarse_config = {
            'sweep_type': 'coarse',
            'parameters': {
                'laser_intensity': {
                    'min': 1e16,
                    'max': 1e20,
                    'log_scale': True,
                    'points': 8
                },
                'plasma_density': {
                    'min': 1e16,
                    'max': 1e20,
                    'log_scale': True,
                    'points': 8
                },
                'magnetic_field': {
                    'min': 0,
                    'max': 100,
                    'log_scale': False,
                    'points': 5
                },
                'temperature_constant': {
                    'min': 1e3,
                    'max': 1e5,
                    'log_scale': True,
                    'points': 6
                }
            },
            'fixed_parameters': {
                'laser_wavelength': 800e-9,
                'grid_max': 50e-6,
                'grid_points': 512,
                'kappa_method': 'acoustic_exact',
                'graybody': 'acoustic_wkb',
                'alpha_gray': 0.8
            },
            'output_dir': 'results/sweeps/coarse'
        }
        
        # Fine sweep configuration
        fine_config = {
            'sweep_type': 'fine',
            'parameters': {
                'laser_intensity': {
                    'min': 1e17,
                    'max': 1e19,
                    'log_scale': True,
                    'points': 12
                },
                'plasma_density': {
                    'min': 1e17,
                    'max': 1e19,
                    'log_scale': True,
                    'points': 12
                },
                'temperature_constant': {
                    'min': 5e3,
                    'max': 5e4,
                    'log_scale': True,
                    'points': 8
                }
            },
            'fixed_parameters': {
                'magnetic_field': None,
                'laser_wavelength': 800e-9,
                'grid_max': 50e-6,
                'grid_points': 512,
                'kappa_method': 'acoustic_exact',
                'graybody': 'acoustic_wkb',
                'alpha_gray': 0.8
            },
            'output_dir': 'results/sweeps/fine'
        }
        
        # Hybrid sweep configuration
        hybrid_config = {
            'sweep_type': 'hybrid',
            'parameters': {
                'laser_intensity': {
                    'min': 5e17,
                    'max': 1e20,
                    'log_scale': True,
                    'points': 10
                },
                'plasma_density': {
                    'min': 5e17,
                    'max': 1e20,
                    'log_scale': True,
                    'points': 10
                },
                'mirror_D': {
                    'min': 1e-6,
                    'max': 100e-6,
                    'log_scale': True,
                    'points': 8
                },
                'mirror_eta': {
                    'min': 0.1,
                    'max': 10.0,
                    'log_scale': False,
                    'points': 6
                }
            },
            'fixed_parameters': {
                'temperature_constant': 1e4,
                'laser_wavelength': 800e-9,
                'grid_max': 50e-6,
                'grid_points': 512,
                'enable_hybrid': True,
                'hybrid_model': 'anabhel',
                'kappa_method': 'acoustic_exact',
                'graybody': 'acoustic_wkb',
                'alpha_gray': 0.8
            },
            'output_dir': 'results/sweeps/hybrid'
        }
        
        # Save configurations
        configs = {
            'coarse': coarse_config,
            'fine': fine_config,
            'hybrid': hybrid_config
        }
        
        for name, config in configs.items():
            config_file = self.config_dir / f"sweep_{name}.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def create_optimization_configs(self):
        """Create Bayesian optimization configuration files"""
        
        # Standard optimization
        standard_opt = {
            'optimization_type': 'bayesian',
            'model_type': 'standard',
            'parameters': {
                'laser_intensity': {
                    'type': 'real',
                    'bounds': [1e16, 1e20],
                    'log_scale': True
                },
                'plasma_density': {
                    'type': 'real',
                    'bounds': [1e16, 1e20],
                    'log_scale': True
                },
                'magnetic_field': {
                    'type': 'real',
                    'bounds': [0, 100],
                    'log_scale': False
                },
                'temperature_constant': {
                    'type': 'real',
                    'bounds': [1e3, 1e5],
                    'log_scale': True
                },
                'laser_wavelength': {
                    'type': 'real',
                    'bounds': [400e-9, 1200e-9],
                    'log_scale': False
                },
                'grid_max': {
                    'type': 'real',
                    'bounds': [10e-6, 100e-6],
                    'log_scale': False
                }
            },
            'optimization_settings': {
                'n_calls': 100,
                'n_initial_points': 20,
                'random_state': 42,
                'acq_func': 'EI',
                'kappa': 1.96
            },
            'fixed_parameters': {
                'grid_points': 512,
                'use_fast_magnetosonic': False,
                'kappa_method': 'acoustic_exact',
                'graybody': 'acoustic_wkb',
                'alpha_gray': 0.8,
                'enable_hybrid': False
            },
            'output_dir': 'results/optimization/standard'
        }
        
        # Hybrid optimization
        hybrid_opt = {
            'optimization_type': 'bayesian',
            'model_type': 'hybrid',
            'parameters': {
                'laser_intensity': {
                    'type': 'real',
                    'bounds': [1e17, 1e20],
                    'log_scale': True
                },
                'plasma_density': {
                    'type': 'real',
                    'bounds': [1e17, 1e20],
                    'log_scale': True
                },
                'temperature_constant': {
                    'type': 'real',
                    'bounds': [1e3, 1e5],
                    'log_scale': True
                },
                'mirror_D': {
                    'type': 'real',
                    'bounds': [1e-6, 100e-6],
                    'log_scale': True
                },
                'mirror_eta': {
                    'type': 'real',
                    'bounds': [0.1, 10.0],
                    'log_scale': False
                }
            },
            'optimization_settings': {
                'n_calls': 80,
                'n_initial_points': 15,
                'random_state': 42,
                'acq_func': 'EI',
                'kappa': 1.96
            },
            'fixed_parameters': {
                'magnetic_field': None,
                'laser_wavelength': 800e-9,
                'grid_max': 50e-6,
                'grid_points': 512,
                'enable_hybrid': True,
                'hybrid_model': 'anabhel',
                'kappa_method': 'acoustic_exact',
                'graybody': 'acoustic_wkb',
                'alpha_gray': 0.8
            },
            'output_dir': 'results/optimization/hybrid'
        }
        
        # Save configurations
        configs = {
            'standard': standard_opt,
            'hybrid': hybrid_opt
        }
        
        for name, config in configs.items():
            config_file = self.config_dir / f"optimization_{name}.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def create_surrogate_configs(self):
        """Create machine learning surrogate configuration files"""
        
        surrogate_configs = {
            'gaussian_process': {
                'model_type': 'gp',
                'kernel': 'matern',
                'kernel_params': {
                    'nu': 1.5,
                    'length_scale': 1.0,
                    'length_scale_bounds': [1e-2, 1e2]
                },
                'training': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'n_restarts_optimizer': 10
                }
            },
            'random_forest': {
                'model_type': 'rf',
                'model_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'random_state': 42
                },
                'training': {
                    'test_size': 0.2,
                    'random_state': 42,
                    'cross_validation': 5
                }
            },
            'neural_network': {
                'model_type': 'nn',
                'architecture': {
                    'hidden_layers': [64, 128, 64],
                    'activation': 'relu',
                    'dropout_rate': 0.1
                },
                'training': {
                    'epochs': 1000,
                    'batch_size': 32,
                    'learning_rate': 0.001,
                    'optimizer': 'adam',
                    'scheduler': 'reduce_on_plateau'
                }
            },
            'ensemble': {
                'model_type': 'ensemble',
                'models': ['gp', 'rf', 'nn'],
                'selection_criteria': 'cross_validation_score',
                'training': {
                    'test_size': 0.2,
                    'random_state': 42
                }
            }
        }
        
        for name, config in surrogate_configs.items():
            config_file = self.config_dir / f"surrogate_{name}.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def create_experimental_configs(self):
        """Create experimental configuration files"""
        
        experimental_configs = {
            'baseline': {
                'experiment_name': 'baseline_detection',
                'description': 'Baseline configuration for analog Hawking radiation detection',
                'plasma': {
                    'density': 5e17,
                    'temperature': 1e4,
                    'composition': 'hydrogen'
                },
                'laser': {
                    'intensity': 5e17,
                    'wavelength': 800e-9,
                    'pulse_duration': 30e-15,
                    'beam_waist': 10e-6
                },
                'geometry': {
                    'grid_size': 50e-6,
                    'resolution': 512,
                    'boundary_conditions': 'absorbing'
                },
                'detection': {
                    'bandwidth': 1e8,
                    'system_temperature': 30.0,
                    'coupling_efficiency': 0.1,
                    'solid_angle': 0.05,
                    'observation_time': 3600
                },
                'analysis': {
                    'kappa_method': 'acoustic_exact',
                    'graybody_model': 'acoustic_wkb',
                    'uncertainty_quantification': True,
                    'confidence_level': 0.95
                }
            },
            'enhanced': {
                'experiment_name': 'enhanced_detection',
                'description': 'Enhanced configuration with improved parameters',
                'plasma': {
                    'density': 1e19,
                    'temperature': 5e4,
                    'composition': 'hydrogen'
                },
                'laser': {
                    'intensity': 1e19,
                    'wavelength': 800e-9,
                    'pulse_duration': 30e-15,
                    'beam_waist': 5e-6
                },
                'geometry': {
                    'grid_size': 100e-6,
                    'resolution': 1024,
                    'boundary_conditions': 'absorbing'
                },
                'detection': {
                    'bandwidth': 1e9,
                    'system_temperature': 10.0,
                    'coupling_efficiency': 0.5,
                    'solid_angle': 0.1,
                    'observation_time': 3600
                },
                'analysis': {
                    'kappa_method': 'acoustic_exact',
                    'graybody_model': 'acoustic_wkb',
                    'uncertainty_quantification': True,
                    'confidence_level': 0.95
                }
            },
            'breakthrough': {
                'experiment_name': 'breakthrough_detection',
                'description': 'Breakthrough configuration for rapid detection',
                'plasma': {
                    'density': 1e20,
                    'temperature': 1e5,
                    'composition': 'hydrogen'
                },
                'laser': {
                    'intensity': 1e20,
                    'wavelength': 400e-9,
                    'pulse_duration': 10e-15,
                    'beam_waist': 3e-6
                },
                'geometry': {
                    'grid_size': 200e-6,
                    'resolution': 2048,
                    'boundary_conditions': 'absorbing'
                },
                'detection': {
                    'bandwidth': 5e9,
                    'system_temperature': 3.0,
                    'coupling_efficiency': 0.8,
                    'solid_angle': 0.2,
                    'observation_time': 3600
                },
                'analysis': {
                    'kappa_method': 'acoustic_exact',
                    'graybody_model': 'acoustic_wkb',
                    'uncertainty_quantification': True,
                    'confidence_level': 0.95
                }
            }
        }
        
        for name, config in experimental_configs.items():
            config_file = self.config_dir / f"experimental_{name}.yml"
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
    
    def create_scaling_roadmap(self):
        """Create scaling roadmap configuration"""
        
        roadmap = {
            'roadmap_name': 'analog_hawking_scaling',
            'description': 'Technology development roadmap for analog Hawking radiation detection',
            'current_baseline': {
                'laser_intensity': 5e17,
                'plasma_density': 5e17,
                'system_temperature': 30.0,
                'bandwidth': 1e8,
                'coupling_efficiency': 0.1,
                'median_detection_time': 86400  # 1 day
            },
            'targets': {
                'phase1': {
                    'name': 'Parameter Optimization',
                    'duration_years': 1,
                    'target_detection_time': 86400,  # 1 day
                    'required_improvements': {
                        'laser_intensity': 2.0,
                        'plasma_density': 2.0,
                        'system_temperature': 0.8,
                        'bandwidth': 2.0,
                        'coupling_efficiency': 1.5
                    },
                    'technology_level': 'existing'
                },
                'phase2': {
                    'name': 'Enhanced Systems',
                    'duration_years': 2,
                    'target_detection_time': 3600,  # 1 hour
                    'required_improvements': {
                        'laser_intensity': 10.0,
                        'plasma_density': 10.0,
                        'system_temperature': 0.3,
                        'bandwidth': 10.0,
                        'coupling_efficiency': 5.0
                    },
                    'technology_level': 'petawatt_laser'
                },
                'phase3': {
                    'name': 'Advanced Configurations',
                    'duration_years': 3,
                    'target_detection_time': 600,  # 10 minutes
                    'required_improvements': {
                        'laser_intensity': 100.0,
                        'plasma_density': 100.0,
                        'system_temperature': 0.1,
                        'bandwidth': 50.0,
                        'coupling_efficiency': 8.0
                    },
                    'technology_level': 'exawatt_laser'
                },
                'phase4': {
                    'name': 'Breakthrough Detection',
                    'duration_years': 5,
                    'target_detection_time': 60,  # 1 minute
                    'required_improvements': {
                        'laser_intensity': 1000.0,
                        'plasma_density': 1000.0,
                        'system_temperature': 0.03,
                        'bandwidth': 100.0,
                        'coupling_efficiency': 10.0
                    },
                    'technology_level': 'quantum_enhanced'
                }
            },
            'technology_requirements': {
                'laser_systems': [
                    'Petawatt-class Ti:sapphire systems',
                    'Exawatt-class facilities',
                    'Quantum-enhanced amplifiers'
                ],
                'detection_systems': [
                    'Cryogenic receivers (10K)',
                    'Quantum-limited amplifiers',
                    'Wideband coherent detection',
                    'High-efficiency coupling systems'
                ],
                'plasma_systems': [
                    'High-density gas jets',
                    'Laser-wakefield acceleration',
                    'Magnetized plasma configurations'
                ]
            },
            'validation_milestones': [
                '3σ detection within 1 week',
                '5σ detection within 1 day',
                '6σ detection within 1 hour',
                'Discovery-level detection within 10 minutes'
            ]
        }
        
        roadmap_file = self.config_dir / "scaling_roadmap.yml"
        with open(roadmap_file, 'w') as f:
            yaml.dump(roadmap, f, default_flow_style=False)
    
    def create_all_configs(self):
        """Create all configuration files"""
        
        print("Creating configuration files...")
        
        self.create_sweep_configs()
        print("✓ Parameter sweep configurations created")
        
        self.create_optimization_configs()
        print("✓ Optimization configurations created")
        
        self.create_surrogate_configs()
        print("✓ Surrogate model configurations created")
        
        self.create_experimental_configs()
        print("✓ Experimental configurations created")
        
        self.create_scaling_roadmap()
        print("✓ Scaling roadmap created")
        
        print(f"\nAll configurations saved to: {self.config_dir}")

def main():
    parser = argparse.ArgumentParser(description="Configuration management for Hawking radiation protocol")
    parser.add_argument("--config-dir", default="configs", help="Configuration directory")
    parser.add_argument("--create-all", action="store_true", help="Create all configurations")
    parser.add_argument("--create-sweeps", action="store_true", help="Create sweep configurations")
    parser.add_argument("--create-optimizations", action="store_true", help="Create optimization configurations")
    parser.add_argument("--create-surrogates", action="store_true", help="Create surrogate configurations")
    parser.add_argument("--create-experiments", action="store_true", help="Create experimental configurations")
    parser.add_argument("--create-roadmap", action="store_true", help="Create scaling roadmap")
    
    args = parser.parse_args()
    
    manager = ConfigManager(args.config_dir)
    
    if args.create_all:
        manager.create_all_configs()
    else:
        if args.create_sweeps:
            manager.create_sweep_configs()
        if args.create_optimizations:
            manager.create_optimization_configs()
        if args.create_surrogates:
            manager.create_surrogate_configs()
        if args.create_experiments:
            manager.create_experimental_configs()
        if args.create_roadmap:
            manager.create_scaling_roadmap()

if __name__ == "__main__":
    main()
