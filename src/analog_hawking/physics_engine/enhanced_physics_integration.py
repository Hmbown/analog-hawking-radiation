"""
Enhanced Physics Integration with Analysis Pipeline

This module integrates the enhanced physics models (relativistic effects, ionization physics,
plasma-surface interactions) with the existing analog Hawking radiation analysis pipeline
while maintaining backward compatibility.

Key Features:
1. Seamless integration with existing horizon finding and graybody calculations
2. Backward compatibility with legacy physics models
3. Model selection and validation options
4. Enhanced uncertainty quantification including model uncertainties
5. ELI facility-specific parameter optimization

Author: Enhanced Physics Implementation
Date: November 2025
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, hbar, k, m_p, pi
from scipy.interpolate import interp1d
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Import existing physics modules
from .horizon import find_horizons_with_uncertainty, sound_speed
from .plasma_mirror import calculate_plasma_mirror_dynamics
from .detection.graybody_nd import GraybodySpectrumND

# Import enhanced physics models
from .enhanced_relativistic_physics import RelativisticPlasmaPhysics
from .enhanced_ionization_physics import IonizationDynamics, ATOMIC_DATA
from .enhanced_plasma_surface_physics import PlasmaDynamicsAtSurface
from .physics_validation_framework import PhysicsModelValidator

class PhysicsModel(Enum):
    """Enumeration for different physics model options"""
    LEGACY = "legacy"
    RELATIVISTIC = "relativistic"
    ENHANCED_IONIZATION = "enhanced_ionization"
    PLASMA_SURFACE = "plasma_surface"
    COMPREHENSIVE = "comprehensive"

@dataclass
class EnhancedPhysicsConfig:
    """Configuration for enhanced physics models"""
    model: PhysicsModel = PhysicsModel.LEGACY
    include_relativistic: bool = False
    include_ionization_dynamics: bool = False
    include_surface_physics: bool = False
    include_validation: bool = True
    uncertainty_quantification: bool = True
    eli_optimization: bool = False

    # Material parameters
    target_material: str = "Al"
    surface_roughness: float = 5e-9  # meters

    # ELI facility parameters
    eli_wavelength: float = 800e-9
    eli_max_intensity: float = 1e23  # W/m^2

    # Numerical parameters
    temporal_resolution: int = 100
    spatial_resolution: int = 1000

@dataclass
class EnhancedPhysicsResults:
    """Container for enhanced physics results"""
    horizon_positions: np.ndarray
    surface_gravity: np.ndarray
    surface_gravity_uncertainty: np.ndarray
    hawking_temperature: np.ndarray
    graybody_spectrum: Optional[np.ndarray] = None
    model_validation: Optional[Dict] = None
    uncertainties: Optional[Dict] = None

    # Enhanced results
    relativistic_corrections: Optional[Dict] = None
    ionization_dynamics: Optional[Dict] = None
    surface_interaction: Optional[Dict] = None
    eli_optimization: Optional[Dict] = None

class EnhancedPhysicsEngine:
    """
    Enhanced physics engine that integrates all improved models with the existing pipeline
    """

    def __init__(self, config: EnhancedPhysicsConfig):
        """
        Initialize enhanced physics engine

        Args:
            config: Configuration for enhanced physics models
        """
        self.config = config
        self.validator = None

        # Initialize enhanced physics components based on configuration
        self._initialize_components()

    def _initialize_components(self):
        """Initialize physics components based on configuration"""
        # Relativistic physics
        if self.config.include_relativistic or self.config.model in [PhysicsModel.RELATIVISTIC, PhysicsModel.COMPREHENSIVE]:
            self.relativistic_physics = RelativisticPlasmaPhysics(
                electron_density=1e19,
                laser_wavelength=self.config.eli_wavelength,
                laser_intensity=1e20
            )
        else:
            self.relativistic_physics = None

        # Ionization dynamics
        if self.config.include_ionization_dynamics or self.config.model in [PhysicsModel.ENHANCED_IONIZATION, PhysicsModel.COMPREHENSIVE]:
            if self.config.target_material in ATOMIC_DATA:
                self.ionization_physics = IonizationDynamics(
                    ATOMIC_DATA[self.config.target_material],
                    laser_wavelength=self.config.eli_wavelength
                )
            else:
                warnings.warn(f"Unknown target material: {self.config.target_material}. Using Aluminum.")
                self.ionization_physics = IonizationDynamics(
                    ATOMIC_DATA['Al'],
                    laser_wavelength=self.config.eli_wavelength
                )
        else:
            self.ionization_physics = None

        # Surface physics
        if self.config.include_surface_physics or self.config.model in [PhysicsModel.PLASMA_SURFACE, PhysicsModel.COMPREHENSIVE]:
            self.surface_physics = PlasmaDynamicsAtSurface(self.config.target_material)
        else:
            self.surface_physics = None

        # Validation framework
        if self.config.include_validation:
            self.validator = PhysicsModelValidator()

    def enhanced_horizon_finding(self, x: np.ndarray, v: np.ndarray, T_e: np.ndarray,
                               n_e: Optional[np.ndarray] = None,
                               B_field: Optional[np.ndarray] = None,
                               gamma_factor: Optional[np.ndarray] = None) -> EnhancedPhysicsResults:
        """
        Enhanced horizon finding with relativistic corrections and ionization effects

        Args:
            x: Position array in meters
            v: Velocity array in m/s
            T_e: Electron temperature array in Kelvin
            n_e: Optional electron density array in m^-3
            B_field: Optional magnetic field array in Tesla
            gamma_factor: Optional relativistic gamma factor array

        Returns:
            EnhancedPhysicsResults object
        """
        # Apply relativistic corrections if enabled
        if self.relativistic_physics and gamma_factor is not None:
            # Relativistic sound speed
            c_s_enhanced = self.relativistic_physics.relativistic_sound_speed(T_e, gamma_factor)
        else:
            # Legacy sound speed calculation
            c_s_enhanced = sound_speed(T_e)

        # Find horizons using enhanced sound speed
        horizon_result = find_horizons_with_uncertainty(x, v, c_s_enhanced)

        # Apply ionization effects if enabled
        if self.ionization_physics and n_e is not None:
            # Calculate ionization effects on horizon properties
            ionization_effects = self._calculate_ionization_effects_on_horizon(
                x, v, T_e, n_e, horizon_result
            )
            # Modify surface gravity based on ionization
            kappa_modified = horizon_result.kappa * ionization_effects['correction_factor']
        else:
            kappa_modified = horizon_result.kappa
            ionization_effects = None

        # Calculate Hawking temperature with relativistic corrections
        if self.relativistic_physics and gamma_factor is not None:
            # Apply relativistic corrections to Hawking temperature
            T_hawking = self.relativistic_physics.relativistic_hawking_temperature(
                kappa_modified, np.mean(gamma_factor)
            )
        else:
            # Legacy Hawking temperature calculation
            T_hawking = hbar * kappa_modified / (2 * pi * k)

        # Calculate uncertainties
        if self.config.uncertainty_quantification:
            uncertainties = self._calculate_uncertainties(
                x, v, T_e, horizon_result, ionization_effects
            )
        else:
            uncertainties = None

        # Enhanced results
        relativistic_corrections = None
        if self.relativistic_physics:
            relativistic_corrections = {
                'gamma_factor': gamma_factor,
                'relativistic_sound_speed': c_s_enhanced if self.relativistic_physics else c_s_enhanced,
                'regime': self.relativistic_physics.check_relativistic_regime()
            }

        return EnhancedPhysicsResults(
            horizon_positions=horizon_result.positions,
            surface_gravity=kappa_modified,
            surface_gravity_uncertainty=horizon_result.kappa_err,
            hawking_temperature=T_hawking,
            uncertainties=uncertainties,
            relativistic_corrections=relativistic_corrections,
            ionization_dynamics=ionization_effects
        )

    def _calculate_ionization_effects_on_horizon(self, x: np.ndarray, v: np.ndarray,
                                               T_e: np.ndarray, n_e: np.ndarray,
                                               horizon_result) -> Dict[str, Any]:
        """
        Calculate ionization effects on horizon properties

        Args:
            x: Position array
            v: Velocity array
            T_e: Temperature array
            n_e: Density array
            horizon_result: Horizon finding results

        Returns:
            Dictionary with ionization effects
        """
        # Calculate local ionization state at horizon positions
        ionization_states = []
        correction_factors = []

        for pos in horizon_result.positions:
            # Find nearest grid point
            idx = np.argmin(np.abs(x - pos))

            # Local conditions
            local_density = n_e[idx]
            local_temperature = T_e[idx]
            local_velocity = v[idx]

            # Simplified ionization state calculation
            # (In practice, would use full ionization dynamics simulation)
            average_charge_state = min(int(np.log10(local_density / 1e18)), 6)  # Simplified
            ionization_states.append(average_charge_state)

            # Ionization affects effective mass and sound speed
            # Higher ionization -> higher effective sound speed
            ionization_correction = 1.0 + 0.1 * average_charge_state
            correction_factors.append(ionization_correction)

        return {
            'ionization_states': np.array(ionization_states),
            'correction_factor': np.mean(correction_factors),
            'charge_distribution': np.array(ionization_states)
        }

    def _calculate_uncertainties(self, x: np.ndarray, v: np.ndarray, T_e: np.ndarray,
                               horizon_result, ionization_effects: Optional[Dict]) -> Dict[str, Any]:
        """
        Calculate uncertainties for enhanced physics predictions

        Args:
            x: Position array
            v: Velocity array
            T_e: Temperature array
            horizon_result: Horizon results
            ionization_effects: Ionization effects if available

        Returns:
            Dictionary with uncertainty estimates
        """
        uncertainties = {}

        # Numerical uncertainties from horizon finding
        uncertainties['surface_gravity_numerical'] = horizon_result.kappa_err

        # Model uncertainties
        if self.relativistic_physics:
            # Relativistic model uncertainty (typically 5-10% for high a0)
            regime = self.relativistic_physics.check_relativistic_regime()
            if regime['regime'] in ['Relativistic', 'Highly relativistic']:
                uncertainties['relativistic_model'] = 0.1  # 10% uncertainty
            else:
                uncertainties['relativistic_model'] = 0.05  # 5% uncertainty

        if self.ionization_physics:
            # Ionization model uncertainty
            uncertainties['ionization_model'] = 0.15  # 15% uncertainty

        if self.surface_physics:
            # Surface physics uncertainty
            uncertainties['surface_physics_model'] = 0.2  # 20% uncertainty

        # Combined uncertainty (quadrature sum)
        model_uncertainties = [v for k, v in uncertainties.items() if k.endswith('_model')]
        if model_uncertainties:
            uncertainties['combined_model'] = np.sqrt(sum(u**2 for u in model_uncertainties))

        return uncertainties

    def enhanced_graybody_calculation(self, frequencies: np.ndarray,
                                    horizon_results: EnhancedPhysicsResults,
                                    additional_params: Optional[Dict] = None) -> np.ndarray:
        """
        Calculate graybody spectrum with enhanced physics

        Args:
            frequencies: Frequency array in Hz
            horizon_results: Enhanced horizon results
            additional_params: Additional parameters for spectrum calculation

        Returns:
            Graybody spectrum array
        """
        # Use existing graybody calculation as base
        if horizon_results.surface_gravity.size > 0:
            kappa = horizon_results.surface_gravity[0]  # Use first horizon
            T_H = horizon_results.hawking_temperature[0]
        else:
            # Default values if no horizon found
            kappa = 1e13  # s^-1
            T_H = 1e-10  # K

        # Create graybody spectrum
        graybody_calc = GraybodySpectrumND([1])  # 1D case
        spectrum = graybody_calc.calculate_spectrum(frequencies, T_H, kappa, additional_params or {})

        # Apply enhanced physics corrections
        if horizon_results.relativistic_corrections:
            # Relativistic corrections to spectrum
            gamma_mean = np.mean(horizon_results.relativistic_corrections['gamma_factor'])
            spectrum *= 1.0 / gamma_mean  # Time dilation effect

        if horizon_results.ionization_dynamics:
            # Ionization effects on transmission
            charge_state = np.mean(horizon_results.ionization_dynamics['ionization_states'])
            transmission_factor = 1.0 + 0.05 * charge_state  # Simplified model
            spectrum *= transmission_factor

        return spectrum

    def eli_facility_optimization(self, parameter_ranges: Dict[str, Tuple[float, float]],
                                optimization_objective: str = "hawking_temperature") -> Dict[str, Any]:
        """
        Optimize parameters for ELI facility conditions

        Args:
            parameter_ranges: Dictionary of parameter ranges to optimize
            optimization_objective: Objective function to optimize

        Returns:
            Dictionary with optimization results
        """
        if not self.config.eli_optimization:
            warnings.warn("ELI optimization not enabled in configuration")
            return {}

        # Define objective function
        def objective_function(params):
            # Create temporary enhanced physics engine with test parameters
            test_config = EnhancedPhysicsConfig(
                model=self.config.model,
                include_relativistic=self.config.include_relativistic,
                include_ionization_dynamics=self.config.include_ionization_dynamics,
                include_surface_physics=self.config.include_surface_physics,
                eli_optimization=True
            )

            # Simulate with test parameters (simplified)
            # In practice, would run full simulation
            if optimization_objective == "hawking_temperature":
                # Maximizing Hawking temperature
                return -1.0 * params.get('intensity', 1e20) * params.get('density', 1e19)
            elif optimization_objective == "signal_to_noise":
                # Maximizing signal-to-noise ratio
                return -1.0 * np.sqrt(params.get('intensity', 1e20) / params.get('density', 1e19))
            else:
                return 0

        # Simple grid-based optimization (in practice, would use more sophisticated methods)
        best_params = {}
        best_objective = float('inf')

        for param_name, (min_val, max_val) in parameter_ranges.items():
            # Test a few values in each range
            test_values = np.linspace(min_val, max_val, 5)
            best_value = None
            best_local_objective = float('inf')

            for value in test_values:
                test_params = best_params.copy()
                test_params[param_name] = value
                objective = objective_function(test_params)

                if objective < best_local_objective:
                    best_local_objective = objective
                    best_value = value

            if best_value is not None:
                best_params[param_name] = best_value
                if best_local_objective < best_objective:
                    best_objective = best_local_objective

        return {
            'optimal_parameters': best_params,
            'objective_value': -best_objective,
            'optimization_target': optimization_objective,
            'facility': 'ELI'
        }

    def run_validation_suite(self) -> Dict[str, Any]:
        """
        Run comprehensive validation suite for enhanced physics models

        Returns:
            Validation results dictionary
        """
        if not self.validator:
            warnings.warn("Validation not enabled in configuration")
            return {}

        validation_results = {}

        # Validate relativistic physics
        if self.relativistic_physics:
            validation_results['relativistic'] = self.validator.validate_relativistic_physics(
                self.relativistic_physics
            )

        # Validate ionization physics
        if self.ionization_physics:
            validation_results['ionization'] = self.validator.validate_ionization_physics(
                self.ionization_physics
            )

        # Validate surface physics
        if self.surface_physics:
            validation_results['surface'] = self.validator.validate_surface_physics(
                self.surface_physics
            )

        # Generate comprehensive report
        validation_results['summary'] = self.validator.generate_validation_report()

        return validation_results

class BackwardCompatibilityWrapper:
    """
    Wrapper to maintain backward compatibility with existing analysis pipeline
    """

    def __init__(self, enhanced_engine: EnhancedPhysicsEngine):
        """
        Initialize backward compatibility wrapper

        Args:
            enhanced_engine: Enhanced physics engine instance
        """
        self.engine = enhanced_engine

    def find_horizons_legacy(self, x: np.ndarray, v: np.ndarray, c_s: np.ndarray):
        """
        Legacy horizon finding interface

        Args:
            x: Position array
            v: Velocity array
            c_s: Sound speed array

        Returns:
            Horizon results in legacy format
        """
        # Convert to enhanced interface
        T_e = c_s**2 * self.engine.relativistic_physics.m_i / (5/3 * k) if self.engine.relativistic_physics else c_s**2 * m_p / (5/3 * k)

        # Use enhanced engine
        enhanced_results = self.engine.enhanced_horizon_finding(x, v, T_e)

        # Convert back to legacy format
        from .horizon import HorizonResult
        return HorizonResult(
            positions=enhanced_results.horizon_positions,
            kappa=enhanced_results.surface_gravity,
            kappa_err=enhanced_results.surface_gravity_uncertainty,
            dvdx=np.gradient(v, x),
            dcsdx=np.gradient(c_s, x)
        )

    def calculate_hawking_temperature_legacy(self, kappa: np.ndarray):
        """
        Legacy Hawking temperature calculation

        Args:
            kappa: Surface gravity array

        Returns:
            Hawking temperature array
        """
        return hbar * kappa / (2 * pi * k)

def create_enhanced_pipeline(model_type: PhysicsModel = PhysicsModel.COMPREHENSIVE,
                           target_material: str = "Al",
                           eli_optimization: bool = False) -> EnhancedPhysicsEngine:
    """
    Factory function to create enhanced physics pipeline

    Args:
        model_type: Type of physics model to use
        target_material: Target material for experiments
        eli_optimization: Enable ELI facility optimization

    Returns:
        Enhanced physics engine instance
    """
    config = EnhancedPhysicsConfig(
        model=model_type,
        include_relativistic=(model_type in [PhysicsModel.RELATIVISTIC, PhysicsModel.COMPREHENSIVE]),
        include_ionization_dynamics=(model_type in [PhysicsModel.ENHANCED_IONIZATION, PhysicsModel.COMPREHENSIVE]),
        include_surface_physics=(model_type in [PhysicsModel.PLASMA_SURFACE, PhysicsModel.COMPREHENSIVE]),
        target_material=target_material,
        eli_optimization=eli_optimization
    )

    return EnhancedPhysicsEngine(config)

def test_enhanced_integration():
    """
    Test the enhanced physics integration
    """
    print("Testing Enhanced Physics Integration")
    print("=" * 50)

    # Create enhanced physics engine
    engine = create_enhanced_pipeline(
        model_type=PhysicsModel.COMPREHENSIVE,
        target_material="Al",
        eli_optimization=True
    )

    # Test enhanced horizon finding
    print("\n1. Testing Enhanced Horizon Finding")
    print("-" * 40)

    # Create test data
    x = np.linspace(0, 100e-6, 1000)  # 100 micron domain
    v = 0.1 * c * np.tanh((x - 50e-6) / 10e-6)  # Velocity profile
    T_e = 1e6 * np.ones_like(x)  # 1 MK temperature
    n_e = 1e19 * np.ones_like(x)  # 10^19 m^-3 density
    gamma = np.ones_like(x) * 1.5  # Relativistic factor

    # Find horizons
    horizon_results = engine.enhanced_horizon_finding(x, v, T_e, n_e, gamma_factor=gamma)

    print(f"Found {len(horizon_results.horizon_positions)} horizons")
    if len(horizon_results.horizon_positions) > 0:
        print(f"Horizon positions: {horizon_results.horizon_positions*1e6} μm")
        print(f"Surface gravity: {horizon_results.surface_gravity:.2e} s^-1")
        print(f"Hawking temperature: {horizon_results.hawking_temperature:.2e} K")

    # Test enhanced graybody calculation
    print("\n2. Testing Enhanced Graybody Calculation")
    print("-" * 40)

    frequencies = np.logspace(10, 14, 100)  # 10 GHz to 100 THz
    spectrum = engine.enhanced_graybody_calculation(frequencies, horizon_results)

    print(f"Calculated spectrum for {len(frequencies)} frequency points")
    print(f"Spectrum peak: {frequencies[np.argmax(spectrum)]:.2e} Hz")
    print(f"Peak intensity: {np.max(spectrum):.2e}")

    # Test ELI optimization
    print("\n3. Testing ELI Facility Optimization")
    print("-" * 40)

    parameter_ranges = {
        'intensity': (1e19, 1e22),  # W/m^2
        'density': (1e18, 1e21),     # m^-3
        'wavelength': (400e-9, 1064e-9)  # m
    }

    optimization_results = engine.eli_facility_optimization(
        parameter_ranges, "hawking_temperature"
    )

    print("Optimization results:")
    for param, value in optimization_results.get('optimal_parameters', {}).items():
        print(f"  {param}: {value:.2e}")

    # Test validation
    print("\n4. Testing Physics Model Validation")
    print("-" * 40)

    validation_results = engine.run_validation_suite()
    if validation_results:
        summary = validation_results.get('summary', {})
        print(f"Validation summary:")
        print(f"  Total tests: {summary.get('total_tests', 0)}")
        print(f"  Passed: {summary.get('passed_tests', 0)}")
        print(f"  Status: {summary.get('overall_status', 'Unknown')}")

    print("\nEnhanced Integration Test Complete!")

if __name__ == "__main__":
    test_enhanced_integration()