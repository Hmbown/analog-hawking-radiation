"""
Uncertainty Quantification for Spatial Coupling Enhancement

Implements bootstrap and Monte Carlo methods to establish confidence intervals
on κ predictions and enhancement ratios.

Author: AHR Validation Team
Date: 2025-11-06
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable
from scipy import stats
import warnings

from analog_hawking.physics_engine.enhanced_coupling import (
    SpatialCouplingProfile,
    compute_patchwise_effective_kappa
)


@dataclass
class UncertaintyResult:
    """Container for uncertainty quantification results."""
    
    kappa_spatial: np.ndarray
    kappa_averaged: float
    enhancement_ratio: float
    
    # Bootstrap statistics
    kappa_spatial_bootstrap: np.ndarray
    kappa_averaged_bootstrap: np.ndarray
    enhancement_bootstrap: np.ndarray
    
    # Confidence intervals
    ci_kappa_spatial: Tuple[float, float]
    ci_kappa_averaged: Tuple[float, float]
    ci_enhancement: Tuple[float, float]
    
    # Additional metrics
    n_bootstrap_samples: int
    spatial_std: float
    averaged_std: float
    enhancement_std: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            'kappa_spatial_mean': float(np.mean(self.kappa_spatial)),
            'kappa_spatial_std': float(self.spatial_std),
            'kappa_spatial_ci_lower': float(self.ci_kappa_spatial[0]),
            'kappa_spatial_ci_upper': float(self.ci_kappa_spatial[1]),
            'kappa_averaged_mean': float(self.kappa_averaged),
            'kappa_averaged_std': float(self.averaged_std),
            'kappa_averaged_ci_lower': float(self.ci_kappa_averaged[0]),
            'kappa_averaged_ci_upper': float(self.ci_kappa_averaged[1]),
            'enhancement_ratio': float(self.enhancement_ratio),
            'enhancement_std': float(self.enhancement_std),
            'enhancement_ci_lower': float(self.ci_enhancement[0]),
            'enhancement_ci_upper': float(self.ci_enhancement[1]),
            'n_bootstrap_samples': self.n_bootstrap_samples,
            'significant': bool(self.ci_enhancement[0] > 1.0)  # Enhancement > 1
        }


class BootstrapUncertaintyQuantifier:
    """
    Bootstrap uncertainty quantification for spatial coupling enhancement.
    """
    
    def __init__(self, n_bootstrap_samples: int = 10000, confidence_level: float = 0.95,
                 random_seed: Optional[int] = None):
        """
        Initialize bootstrap uncertainty quantifier.
        
        Args:
            n_bootstrap_samples: Number of bootstrap samples to generate
            confidence_level: Confidence level for intervals (e.g., 0.95 for 95% CI)
            random_seed: Random seed for reproducibility
        """
        self.n_bootstrap_samples = n_bootstrap_samples
        self.confidence_level = confidence_level
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def quantify_uncertainty(self, profile: SpatialCouplingProfile,
                           measurement_errors: Optional[Dict[str, np.ndarray]] = None,
                           systematic_errors: Optional[Dict[str, float]] = None) -> UncertaintyResult:
        """
        Perform comprehensive uncertainty quantification.
        
        Args:
            profile: Spatial coupling profile
            measurement_errors: Optional measurement errors for each field
            systematic_errors: Optional systematic error estimates
            
        Returns:
            UncertaintyResult with confidence intervals and statistics
        """
        # Base calculation
        kappa_per_patch = compute_patchwise_effective_kappa(profile)
        kappa_spatial = kappa_per_patch
        kappa_averaged = float(np.mean(kappa_per_patch))
        enhancement_ratio = float(np.max(kappa_per_patch) / kappa_averaged)
        
        # Bootstrap sampling
        (kappa_spatial_boot, kappa_averaged_boot, enhancement_boot) = self._bootstrap_analysis(
            profile, measurement_errors, systematic_errors
        )
        
        # Compute confidence intervals
        alpha = 1.0 - self.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        ci_kappa_spatial = tuple(np.percentile(kappa_spatial_boot, [lower_percentile, upper_percentile]))
        ci_kappa_averaged = tuple(np.percentile(kappa_averaged_boot, [lower_percentile, upper_percentile]))
        ci_enhancement = tuple(np.percentile(enhancement_boot, [lower_percentile, upper_percentile]))
        
        # Compute standard deviations
        spatial_std = float(np.std(kappa_spatial_boot))
        averaged_std = float(np.std(kappa_averaged_boot))
        enhancement_std = float(np.std(enhancement_boot))
        
        return UncertaintyResult(
            kappa_spatial=kappa_spatial,
            kappa_averaged=kappa_averaged,
            enhancement_ratio=enhancement_ratio,
            kappa_spatial_bootstrap=kappa_spatial_boot,
            kappa_averaged_bootstrap=kappa_averaged_boot,
            enhancement_bootstrap=enhancement_boot,
            ci_kappa_spatial=ci_kappa_spatial,
            ci_kappa_averaged=ci_kappa_averaged,
            ci_enhancement=ci_enhancement,
            n_bootstrap_samples=self.n_bootstrap_samples,
            spatial_std=spatial_std,
            averaged_std=averaged_std,
            enhancement_std=enhancement_std
        )
    
    def _bootstrap_analysis(self, profile: SpatialCouplingProfile,
                          measurement_errors: Optional[Dict[str, np.ndarray]],
                          systematic_errors: Optional[Dict[str, float]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform bootstrap resampling analysis.
        
        Args:
            profile: Spatial coupling profile
            measurement_errors: Measurement errors for resampling
            systematic_errors: Systematic errors to include
            
        Returns:
            Tuple of bootstrap arrays: (kappa_spatial_samples, kappa_averaged_samples, enhancement_samples)
        """
        n_patches = len(profile.positions)
        
        # Initialize arrays for bootstrap results
        kappa_spatial_samples = np.zeros((self.n_bootstrap_samples, n_patches))
        kappa_averaged_samples = np.zeros(self.n_bootstrap_samples)
        enhancement_samples = np.zeros(self.n_bootstrap_samples)
        
        for i in range(self.n_bootstrap_samples):
            # Generate bootstrap sample
            boot_profile = self._generate_bootstrap_sample(profile, measurement_errors, systematic_errors)
            
            # Calculate kappa for this sample
            kappa_boot = compute_patchwise_effective_kappa(boot_profile)
            
            # Store results
            kappa_spatial_samples[i, :] = kappa_boot
            kappa_averaged_samples[i] = float(np.mean(kappa_boot))
            enhancement_samples[i] = float(np.max(kappa_boot) / np.mean(kappa_boot))
        
        return kappa_spatial_samples, kappa_averaged_samples, enhancement_samples
    
    def _generate_bootstrap_sample(self, profile: SpatialCouplingProfile,
                                 measurement_errors: Optional[Dict[str, np.ndarray]],
                                 systematic_errors: Optional[Dict[str, float]]) -> SpatialCouplingProfile:
        """
        Generate a single bootstrap sample by resampling with noise.
        
        Args:
            profile: Original spatial coupling profile
            measurement_errors: Measurement errors for adding noise
            systematic_errors: Systematic errors to include
            
        Returns:
            Bootstrap sample profile
        """
        # Start with original data
        positions = profile.positions.copy()
        fluid_kappa = profile.fluid_kappa.copy()
        mirror_kappa = profile.mirror_kappa
        coupling_weights = profile.coupling_weights.copy()
        alignment = profile.alignment.copy()
        
        # Add measurement noise if errors provided
        if measurement_errors is not None:
            if 'fluid_kappa' in measurement_errors:
                fluid_kappa += np.random.normal(0, measurement_errors['fluid_kappa'])
            if 'mirror_kappa' in measurement_errors and 'mirror_kappa' in measurement_errors:
                mirror_kappa += float(np.random.normal(0, measurement_errors['mirror_kappa']))
            if 'coupling_weights' in measurement_errors:
                coupling_weights += np.random.normal(0, measurement_errors['coupling_weights'])
        
        # Add systematic errors if provided
        if systematic_errors is not None:
            if 'fluid_kappa' in systematic_errors:
                fluid_kappa += np.random.normal(0, systematic_errors['fluid_kappa'], size=fluid_kappa.shape)
            if 'mirror_kappa' in systematic_errors:
                mirror_kappa += float(np.random.normal(0, systematic_errors['mirror_kappa']))
            if 'coupling_weights' in systematic_errors:
                coupling_weights += np.random.normal(0, systematic_errors['coupling_weights'], size=coupling_weights.shape)
        
        # Ensure physical constraints (kappa > 0, weights in [0,1])
        fluid_kappa = np.maximum(fluid_kappa, 1e-6)
        mirror_kappa = max(mirror_kappa, 1e-6)
        coupling_weights = np.clip(coupling_weights, 0.0, 1.0)
        
        return SpatialCouplingProfile(
            positions=positions,
            fluid_kappa=fluid_kappa,
            mirror_kappa=mirror_kappa,
            coupling_weights=coupling_weights,
            alignment=alignment
        )


class MonteCarloUncertaintyQuantifier:
    """
    Monte Carlo uncertainty quantification for spatial coupling enhancement.
    """
    
    def __init__(self, n_samples: int = 10000, random_seed: Optional[int] = None):
        """
        Initialize Monte Carlo uncertainty quantifier.
        
        Args:
            n_samples: Number of Monte Carlo samples
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.random_seed = random_seed
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def parameter_sweep_uncertainty(self, profile: SpatialCouplingProfile,
                                  parameter_ranges: Dict[str, Tuple[float, float]]) -> Dict:
        """
        Perform Monte Carlo parameter sweep to map uncertainty across parameter space.
        
        Args:
            profile: Base spatial coupling profile
            parameter_ranges: Dictionary mapping parameter names to (min, max) ranges
            
        Returns:
            Dictionary with parameter sweep results
        """
        n_patches = len(profile.positions)
        
        # Initialize results arrays
        enhancement_samples = np.zeros(self.n_samples)
        max_kappa_samples = np.zeros(self.n_samples)
        mean_kappa_samples = np.zeros(self.n_samples)
        
        # Sample parameter space
        for i in range(self.n_samples):
            # Generate random parameters
            params = {}
            for param_name, (min_val, max_val) in parameter_ranges.items():
                params[param_name] = np.random.uniform(min_val, max_val)
            
            # Create modified profile
            modified_profile = self._modify_profile(profile, params)
            
            # Calculate enhancement
            kappa_per_patch = compute_patchwise_effective_kappa(modified_profile)
            
            enhancement_samples[i] = float(np.max(kappa_per_patch) / np.mean(kappa_per_patch))
            max_kappa_samples[i] = float(np.max(kappa_per_patch))
            mean_kappa_samples[i] = float(np.mean(kappa_per_patch))
        
        # Analyze results
        results = {
            'enhancement_mean': float(np.mean(enhancement_samples)),
            'enhancement_std': float(np.std(enhancement_samples)),
            'enhancement_min': float(np.min(enhancement_samples)),
            'enhancement_max': float(np.max(enhancement_samples)),
            'kappa_max_mean': float(np.mean(max_kappa_samples)),
            'kappa_max_std': float(np.std(max_kappa_samples)),
            'kappa_mean_mean': float(np.mean(mean_kappa_samples)),
            'kappa_mean_std': float(np.std(mean_kappa_samples)),
            'significant_fraction': float(np.mean(enhancement_samples > 1.5)),
            'n_samples': self.n_samples
        }
        
        return results
    
    def _modify_profile(self, profile: SpatialCouplingProfile,
                       modifications: Dict[str, float]) -> SpatialCouplingProfile:
        """
        Modify profile according to parameter changes.
        
        Args:
            profile: Original profile
            modifications: Dictionary of parameter modifications
            
        Returns:
            Modified profile
        """
        positions = profile.positions.copy()
        fluid_kappa = profile.fluid_kappa.copy()
        mirror_kappa = profile.mirror_kappa
        coupling_weights = profile.coupling_weights.copy()
        alignment = profile.alignment.copy()
        
        # Apply modifications
        if 'fluid_kappa_scale' in modifications:
            scale = modifications['fluid_kappa_scale']
            fluid_kappa *= scale
        
        if 'mirror_kappa_scale' in modifications:
            scale = modifications['mirror_kappa_scale']
            mirror_kappa *= scale
        
        if 'coupling_weight_scale' in modifications:
            scale = modifications['coupling_weight_scale']
            coupling_weights *= scale
            coupling_weights = np.clip(coupling_weights, 0.0, 1.0)
        
        if 'coupling_weight_variation' in modifications:
            # Add variation to coupling weights
            variation = modifications['coupling_weight_variation']
            coupling_weights += np.random.normal(0, variation, size=coupling_weights.shape)
            coupling_weights = np.clip(coupling_weights, 0.0, 1.0)
        
        return SpatialCouplingProfile(
            positions=positions,
            fluid_kappa=fluid_kappa,
            mirror_kappa=mirror_kappa,
            coupling_weights=coupling_weights,
            alignment=alignment
        )


def analyze_uncertainty_significance(result: UncertaintyResult, threshold: float = 1.5) -> Dict:
    """
    Analyze statistical significance of enhancement.
    
    Args:
        result: UncertaintyResult from bootstrap analysis
        threshold: Enhancement threshold for significance
        
    Returns:
        Dictionary with significance analysis
    """
    # Calculate probability that enhancement > threshold
    p_enhancement = float(np.mean(result.enhancement_bootstrap > threshold))
    
    # Calculate effect size (Cohen's d approximation)
    effect_size = float((result.enhancement_ratio - 1.0) / result.enhancement_std)
    
    # Determine significance level
    if p_enhancement > 0.99:
        significance = 'high'
    elif p_enhancement > 0.95:
        significance = 'moderate'
    elif p_enhancement > 0.90:
        significance = 'low'
    else:
        significance = 'not_significant'
    
    return {
        'p_enhancement': p_enhancement,
        'effect_size': effect_size,
        'significance_level': significance,
        'significant': p_enhancement > 0.95,
        'ci_above_threshold': result.ci_enhancement[0] > threshold,
        'threshold': threshold
    }


# Example usage and testing
if __name__ == "__main__":
    import numpy as np
    from analog_hawking.physics_engine.enhanced_coupling import SpatialCouplingProfile
    
    # Create example profile
    n_patches = 16
    positions = np.linspace(0, 100e-6, n_patches)
    fluid_kappa = np.linspace(1e12, 3e12, n_patches)
    mirror_kappa = 5e12
    coupling_weights = np.linspace(0.1, 0.5, n_patches)
    alignment = np.ones(n_patches)
    
    profile = SpatialCouplingProfile(
        positions=positions,
        fluid_kappa=fluid_kappa,
        mirror_kappa=mirror_kappa,
        coupling_weights=coupling_weights,
        alignment=alignment
    )
    
    # Define measurement errors
    measurement_errors = {
        'fluid_kappa': np.ones(n_patches) * 0.1e12,  # 10% error
        'mirror_kappa': 0.5e12,  # 10% error
        'coupling_weights': np.ones(n_patches) * 0.05  # 5% error
    }
    
    # Run bootstrap uncertainty quantification
    print("Running bootstrap uncertainty quantification...")
    bootstrap = BootstrapUncertaintyQuantifier(n_bootstrap_samples=1000, random_seed=42)
    
    result = bootstrap.quantify_uncertainty(profile, measurement_errors)
    
    print(f"\nResults:")
    print(f"Mean κ (spatial): {np.mean(result.kappa_spatial):.2e} ± {result.spatial_std:.2e} Hz")
    print(f"Mean κ (averaged): {result.kappa_averaged:.2e} ± {result.averaged_std:.2e} Hz")
    print(f"Enhancement ratio: {result.enhancement_ratio:.2f} ± {result.enhancement_std:.2f}x")
    print(f"\n95% Confidence Intervals:")
    print(f"  κ spatial: [{result.ci_kappa_spatial[0]:.2e}, {result.ci_kappa_spatial[1]:.2e}] Hz")
    print(f"  κ averaged: [{result.ci_kappa_averaged[0]:.2e}, {result.ci_kappa_averaged[1]:.2e}] Hz")
    print(f"  Enhancement: [{result.ci_enhancement[0]:.2f}, {result.ci_enhancement[1]:.2f}]x")
    
    # Analyze significance
    significance = analyze_uncertainty_significance(result, threshold=1.5)
    print(f"\nSignificance Analysis:")
    print(f"  P(enhancement > 1.5x): {significance['p_enhancement']:.3f}")
    print(f"  Effect size: {significance['effect_size']:.2f}")
    print(f"  Significance level: {significance['significance_level']}")
    print(f"  Statistically significant: {significance['significant']}")
    
    # Run Monte Carlo parameter sweep
    print(f"\n\nRunning Monte Carlo parameter sweep...")
    monte_carlo = MonteCarloUncertaintyQuantifier(n_samples=1000, random_seed=42)
    
    parameter_ranges = {
        'fluid_kappa_scale': (0.5, 2.0),  # Factor of 0.5x to 2x
        'mirror_kappa_scale': (0.5, 2.0),
        'coupling_weight_scale': (0.5, 1.5),
        'coupling_weight_variation': (0.0, 0.2)
    }
    
    sweep_results = monte_carlo.parameter_sweep_uncertainty(profile, parameter_ranges)
    
    print(f"\nParameter Sweep Results:")
    print(f"  Enhancement range: {sweep_results['enhancement_min']:.2f}x to {sweep_results['enhancement_max']:.2f}x")
    print(f"  Mean enhancement: {sweep_results['enhancement_mean']:.2f} ± {sweep_results['enhancement_std']:.2f}x")
    print(f"  Fraction > 1.5x: {sweep_results['significant_fraction']:.1%}")
    
    print(f"\n✓ Uncertainty quantification complete!")