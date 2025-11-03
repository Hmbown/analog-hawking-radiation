#!/usr/bin/env python3
"""
Enhanced Parameter Generator for Analog Hawking Radiation Analysis

This script generates diverse, physically realistic parameter configurations
with proper space-filling designs to achieve adequate statistical power.

Author: Claude Analysis Assistant
Date: November 2025
"""

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings('ignore')

# Import existing analysis modules
import sys

import numpy as np
import pandas as pd
from scipy.constants import c, e, epsilon_0, k, m_e, m_p
from scipy.stats import qmc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.analysis.gradient_sweep import run_single_configuration
from analog_hawking.config.thresholds import Thresholds


@dataclass
class PhysicalParameterRanges:
    """Define physically realistic parameter ranges for laser-plasma interactions"""

    # Laser parameters
    a0_min: float = 0.1      # Minimum normalized vector potential
    a0_max: float = 100.0    # Maximum normalized vector potential
    lambda_l_min: float = 400e-9   # Minimum laser wavelength (m) - UV
    lambda_l_max: float = 10.6e-6   # Maximum laser wavelength (m) - CO2 laser

    # Plasma parameters
    n_e_min: float = 1e17    # Minimum electron density (m^-3) - underdense
    n_e_max: float = 1e24    # Maximum electron density (m^-3) - solid density
    T_e_min: float = 1000    # Minimum electron temperature (K)
    T_e_max: float = 1e6     # Maximum electron temperature (K) - relativistic

    # Plasma composition
    Z_min: float = 1.0       # Minimum ionization state
    Z_max: float = 10.0      # Maximum ionization state
    A_min: float = 1.0       # Minimum atomic mass number
    A_max: float = 20.0      # Maximum atomic mass number

    # Gradient and flow parameters
    gradient_factor_min: float = 0.1    # Minimum gradient steepness
    gradient_factor_max: float = 100.0  # Maximum gradient steepness
    flow_velocity_min: float = 0.01    # Minimum flow velocity (fraction of c)
    flow_velocity_max: float = 0.5     # Maximum flow velocity (fraction of c)

    # Magnetic field parameters
    B_min: float = 0.0      # Minimum magnetic field (T)
    B_max: float = 100.0    # Maximum magnetic field (T)

    # Pulse duration parameters
    tau_min: float = 10e-15  # Minimum pulse duration (s) - 10 fs
    tau_max: float = 1e-12   # Maximum pulse duration (s) - 1 ps

class EnhancedParameterGenerator:
    """Generate diverse, physically realistic parameter configurations"""

    def __init__(self, ranges: Optional[PhysicalParameterRanges] = None, seed: int = 42):
        self.ranges = ranges or PhysicalParameterRanges()
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _validate_physical_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply physical constraints to parameter set"""

        # Calculate derived quantities
        lambda_l = params['lambda_l']
        omega_l = 2 * np.pi * c / lambda_l
        n_e = params['n_e']
        a0 = params['a0']

        # Plasma frequency
        omega_pe = np.sqrt(e**2 * n_e / (epsilon_0 * m_e))

        # Critical density
        n_crit = epsilon_0 * m_e * omega_l**2 / e**2

        # Relativistic corrections
        gamma = np.sqrt(1 + a0**2 / 2)

        # Ensure physical consistency
        # 1. Density should be in appropriate range relative to critical density
        if n_e > 10 * n_crit:  # Too overdamped
            params['n_e'] = min(params['n_e'], 10 * n_crit)

        # 2. Temperature should be consistent with a0
        # Relativistic temperature estimate
        T_rel = m_e * c**2 * (gamma - 1) / k
        params['T_e'] = min(params['T_e'], T_rel)

        # 3. Flow velocity should be sub-relativistic
        params['flow_velocity'] = min(params['flow_velocity'], 0.9)

        # 4. Gradient scale length should be reasonable
        lambda_p = 2 * np.pi * c / omega_pe  # Plasma wavelength
        L_scale = lambda_p / params['gradient_factor']
        params['gradient_scale_length'] = L_scale

        # 5. Magnetic field should be reasonable for plasma conditions
        B_cyclotron = m_e * omega_pe / e
        params['B'] = min(params['B'], 10 * B_cyclotron)

        # 6. Pulse duration should allow for gradient formation
        params['tau'] = max(params['tau'], L_scale / c)

        return params

    def _calculate_derived_parameters(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate derived parameters for the analysis"""

        lambda_l = params['lambda_l']
        omega_l = 2 * np.pi * c / lambda_l
        n_e = params['n_e']
        a0 = params['a0']
        T_e = params['T_e']

        # Laser intensity
        I_0 = 0.5 * epsilon_0 * c * (a0**2) * (m_e**2 * omega_l**2 * c**2) / (e**2)

        # Plasma frequency and wavelength
        omega_pe = np.sqrt(e**2 * n_e / (epsilon_0 * m_e))
        lambda_p = 2 * np.pi * c / omega_pe

        # Critical density
        n_crit = epsilon_0 * m_e * omega_l**2 / e**2

        # Debye length
        lambda_D = np.sqrt(epsilon_0 * k * T_e / (n_e * e**2))

        # Thermal velocity
        v_th = np.sqrt(k * T_e / m_e)

        # Sound speed (including ion contributions)
        Z = params['Z']
        A = params['A']
        m_i = A * m_p
        c_s = np.sqrt(k * T_e / (Z * m_e + m_i))

        # Effective coupling strength
        coupling_strength = a0 * np.sqrt(n_crit / n_e)

        # Diffusion coefficient
        D = v_th * lambda_p  # Simplified diffusion estimate

        return {
            'intensity': I_0,
            'omega_l': omega_l,
            'omega_pe': omega_pe,
            'lambda_p': lambda_p,
            'n_crit': n_crit,
            'lambda_D': lambda_D,
            'v_th': v_th,
            'c_s': c_s,
            'coupling_strength': coupling_strength,
            'D': D,
            'normalized_density': n_e / n_crit
        }

    def generate_latin_hypercube_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate samples using Latin Hypercube Sampling for optimal space-filling"""

        # Define parameter bounds for LHS
        param_bounds = [
            ('a0', np.log10(self.ranges.a0_min), np.log10(self.ranges.a0_max)),           # log scale
            ('lambda_l', self.ranges.lambda_l_min, self.ranges.lambda_l_max),            # linear scale
            ('n_e', np.log10(self.ranges.n_e_min), np.log10(self.ranges.n_e_max)),         # log scale
            ('T_e', np.log10(self.ranges.T_e_min), np.log10(self.ranges.T_e_max)),         # log scale
            ('Z', self.ranges.Z_min, self.ranges.Z_max),                                   # linear scale
            ('A', self.ranges.A_min, self.ranges.A_max),                                   # linear scale
            ('gradient_factor', np.log10(self.ranges.gradient_factor_min),
             np.log10(self.ranges.gradient_factor_max)),                                   # log scale
            ('flow_velocity', self.ranges.flow_velocity_min, self.ranges.flow_velocity_max), # linear scale
            ('B', self.ranges.B_min, self.ranges.B_max),                                   # linear scale
            ('tau', np.log10(self.ranges.tau_min), np.log10(self.ranges.tau_max)),         # log scale
        ]

        # Create Latin Hypercube sampler
        sampler = qmc.LatinHypercube(d=len(param_bounds), seed=self.seed)
        lhs_samples = sampler.random(n=n_samples)

        # Scale samples to parameter ranges
        samples = []
        for i in range(n_samples):
            params = {}
            for j, (name, lower, upper) in enumerate(param_bounds):
                if name in ['a0', 'n_e', 'T_e', 'gradient_factor', 'tau']:
                    # Logarithmic parameters
                    params[name] = 10**lhs_samples[i, j] * (upper / lower)**(lhs_samples[i, j] - 0.5) * lower
                else:
                    # Linear parameters
                    params[name] = lhs_samples[i, j] * (upper - lower) + lower

            # Apply physical constraints
            params = self._validate_physical_constraints(params)

            # Calculate derived parameters
            derived_params = self._calculate_derived_parameters(params)

            # Combine all parameters
            full_params = {**params, **derived_params}
            samples.append(full_params)

        return samples

    def generate_sobol_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate samples using Sobol sequence for quasi-random sampling"""

        # Define parameter bounds (same as LHS)
        param_bounds = [
            ('a0', np.log10(self.ranges.a0_min), np.log10(self.ranges.a0_max)),
            ('lambda_l', self.ranges.lambda_l_min, self.ranges.lambda_l_max),
            ('n_e', np.log10(self.ranges.n_e_min), np.log10(self.ranges.n_e_max)),
            ('T_e', np.log10(self.ranges.T_e_min), np.log10(self.ranges.T_e_max)),
            ('Z', self.ranges.Z_min, self.ranges.Z_max),
            ('A', self.ranges.A_min, self.ranges.A_max),
            ('gradient_factor', np.log10(self.ranges.gradient_factor_min),
             np.log10(self.ranges.gradient_factor_max)),
            ('flow_velocity', self.ranges.flow_velocity_min, self.ranges.flow_velocity_max),
            ('B', self.ranges.B_min, self.ranges.B_max),
            ('tau', np.log10(self.ranges.tau_min), np.log10(self.ranges.tau_max)),
        ]

        # Create Sobol sampler
        sampler = qmc.Sobol(d=len(param_bounds), seed=self.seed)
        sobol_samples = sampler.random(n=n_samples)

        # Scale samples to parameter ranges
        samples = []
        for i in range(n_samples):
            params = {}
            for j, (name, lower, upper) in enumerate(param_bounds):
                if name in ['a0', 'n_e', 'T_e', 'gradient_factor', 'tau']:
                    # Logarithmic parameters
                    params[name] = 10**sobol_samples[i, j] * (upper / lower)**(sobol_samples[i, j] - 0.5) * lower
                else:
                    # Linear parameters
                    params[name] = sobol_samples[i, j] * (upper - lower) + lower

            # Apply physical constraints
            params = self._validate_physical_constraints(params)

            # Calculate derived parameters
            derived_params = self._calculate_derived_parameters(params)

            # Combine all parameters
            full_params = {**params, **derived_params}
            samples.append(full_params)

        return samples

    def generate_stratified_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate stratified samples ensuring coverage of different regimes"""

        # Define physical regimes to ensure coverage
        regimes = [
            # (a0_range, n_e_range, regime_name)
            ((0.1, 1.0), (1e17, 1e19), "weakly_nonlinear_underdense"),
            ((1.0, 10.0), (1e17, 1e19), "moderately_nonlinear_underdense"),
            ((10.0, 100.0), (1e17, 1e19), "strongly_nonlinear_underdense"),
            ((0.1, 1.0), (1e21, 1e24), "weakly_nonlinear_overdense"),
            ((1.0, 10.0), (1e21, 1e24), "moderately_nonlinear_overdense"),
            ((10.0, 100.0), (1e21, 1e24), "strongly_nonlinear_overdense"),
        ]

        samples_per_regime = n_samples // len(regimes)
        remaining_samples = n_samples % len(regimes)

        all_samples = []

        for i, (a0_range, n_e_range, regime_name) in enumerate(regimes):
            n_regime_samples = samples_per_regime + (1 if i < remaining_samples else 0)

            for _ in range(n_regime_samples):
                params = {
                    'a0': 10**self.rng.uniform(np.log10(a0_range[0]), np.log10(a0_range[1])),
                    'lambda_l': self.rng.uniform(self.ranges.lambda_l_min, self.ranges.lambda_l_max),
                    'n_e': 10**self.rng.uniform(np.log10(n_e_range[0]), np.log10(n_e_range[1])),
                    'T_e': 10**self.rng.uniform(np.log10(self.ranges.T_e_min), np.log10(self.ranges.T_e_max)),
                    'Z': self.rng.uniform(self.ranges.Z_min, self.ranges.Z_max),
                    'A': self.rng.uniform(self.ranges.A_min, self.ranges.A_max),
                    'gradient_factor': 10**self.rng.uniform(np.log10(self.ranges.gradient_factor_min),
                                                         np.log10(self.ranges.gradient_factor_max)),
                    'flow_velocity': self.rng.uniform(self.ranges.flow_velocity_min, self.ranges.flow_velocity_max),
                    'B': self.rng.uniform(self.ranges.B_min, self.ranges.B_max),
                    'tau': 10**self.rng.uniform(np.log10(self.ranges.tau_min), np.log10(self.ranges.tau_max)),
                }

                # Apply physical constraints
                params = self._validate_physical_constraints(params)

                # Calculate derived parameters
                derived_params = self._calculate_derived_parameters(params)

                # Combine all parameters and add regime information
                full_params = {**params, **derived_params, 'regime': regime_name}
                all_samples.append(full_params)

        return all_samples

    def generate_mixed_samples(self, n_samples: int) -> List[Dict[str, Any]]:
        """Generate mixed samples combining different sampling strategies"""

        # Allocate samples between different strategies
        n_lhs = n_samples // 3
        n_sobol = n_samples // 3
        n_stratified = n_samples - n_lhs - n_sobol

        # Generate samples using different strategies
        lhs_samples = self.generate_latin_hypercube_samples(n_lhs)
        sobol_samples = self.generate_sobol_samples(n_sobol)
        stratified_samples = self.generate_stratified_samples(n_stratified)

        # Combine all samples
        all_samples = lhs_samples + sobol_samples + stratified_samples

        # Shuffle to mix strategies
        self.rng.shuffle(all_samples)

        return all_samples

def run_configuration_analysis(params: Dict[str, Any], thresholds: Thresholds) -> Dict[str, Any]:
    """Run analysis for a single parameter configuration"""

    try:
        # Convert to gradient sweep parameters
        a0 = params['a0']
        n_e = params['n_e']
        gradient_factor = params['gradient_factor']

        # Run single configuration
        result = run_single_configuration(a0, n_e, gradient_factor, thresholds=thresholds)

        # Add physical parameters to result
        result.update(params)

        return result

    except Exception as e:
        return {
            'error': str(e),
            'validity_score': 0.0,
            'kappa': 0.0,
            **params
        }

def generate_expanded_dataset(
    n_samples: int = 120,
    sampling_strategy: str = 'mixed',
    output_file: str = 'results/enhanced_hawking_dataset.csv',
    seed: int = 42
) -> None:
    """Generate expanded dataset with diverse parameter configurations"""

    print(f"Generating enhanced dataset with {n_samples} configurations...")
    print(f"Sampling strategy: {sampling_strategy}")
    print(f"Random seed: {seed}")

    # Initialize generator
    generator = EnhancedParameterGenerator(seed=seed)

    # Generate parameter samples
    if sampling_strategy == 'lhs':
        samples = generator.generate_latin_hypercube_samples(n_samples)
    elif sampling_strategy == 'sobol':
        samples = generator.generate_sobol_samples(n_samples)
    elif sampling_strategy == 'stratified':
        samples = generator.generate_stratified_samples(n_samples)
    elif sampling_strategy == 'mixed':
        samples = generator.generate_mixed_samples(n_samples)
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling_strategy}")

    print(f"Generated {len(samples)} parameter configurations")

    # Load thresholds
    thresholds = Thresholds()

    # Run analysis for each configuration
    print("Running configuration analysis...")
    results = []
    valid_count = 0

    for i, params in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"  Processing configuration {i+1}/{len(samples)} (valid: {valid_count})")

        result = run_configuration_analysis(params, thresholds)
        results.append(result)

        if result.get('validity_score', 0) > 0.5:
            valid_count += 1

    print(f"Analysis complete. Valid configurations: {valid_count}/{len(results)}")

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Create output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Dataset saved to {output_file}")

    # Generate summary statistics
    print("\nDataset Summary:")
    print(f"  Total configurations: {len(df)}")
    print(f"  Valid configurations: {valid_count} ({100*valid_count/len(df):.1f}%)")
    print(f"  Parameter columns: {len(df.columns)}")

    # Parameter ranges
    if 'coupling_strength' in df.columns:
        print(f"  Coupling strength range: {df['coupling_strength'].min():.3f} - {df['coupling_strength'].max():.3f}")
    if 'D' in df.columns:
        print(f"  Diffusion coefficient range: {df['D'].min():.2e} - {df['D'].max():.2e}")
    if 'kappa' in df.columns:
        valid_kappas = df[df['validity_score'] > 0.5]['kappa']
        if len(valid_kappas) > 0:
            print(f"  Kappa range (valid): {valid_kappas.min():.2e} - {valid_kappas.max():.2e}")

    # Regime distribution (if stratified sampling was used)
    if 'regime' in df.columns:
        print("\nRegime distribution:")
        regime_counts = df['regime'].value_counts()
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} configurations")

    # Save metadata
    metadata = {
        'n_samples': len(df),
        'n_valid': valid_count,
        'sampling_strategy': sampling_strategy,
        'seed': seed,
        'parameter_ranges': asdict(generator.ranges),
        'generation_date': pd.Timestamp.now().isoformat(),
        'columns': list(df.columns)
    }

    metadata_file = output_file.replace('.csv', '_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Metadata saved to {metadata_file}")

def main():
    """Main function for enhanced parameter generation"""

    parser = argparse.ArgumentParser(description='Generate enhanced parameter dataset for Hawking radiation analysis')
    parser.add_argument('--n-samples', type=int, default=120, help='Number of configurations to generate')
    parser.add_argument('--strategy', type=str, default='mixed',
                       choices=['lhs', 'sobol', 'stratified', 'mixed'],
                       help='Sampling strategy')
    parser.add_argument('--output', type=str, default='results/enhanced_hawking_dataset.csv',
                       help='Output file path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    # Generate dataset
    generate_expanded_dataset(
        n_samples=args.n_samples,
        sampling_strategy=args.strategy,
        output_file=args.output,
        seed=args.seed
    )

    print("\nEnhanced parameter generation complete!")
    print(f"Dataset: {args.output}")
    print(f"Configurations: {args.n_samples}")
    print(f"Strategy: {args.strategy}")

if __name__ == "__main__":
    main()