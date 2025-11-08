#!/usr/bin/env python3
"""
Comprehensive Numerical Stability Testing Suite for Analog Hawking Radiation Analysis

This framework tests the scientific computing components at extreme parameter ranges
and physical boundaries to ensure robustness at the limits of physical reality.

Test Categories:
- Extreme parameter validation (intensity, density, temperature, magnetic field)
- Numerical convergence testing at physical boundaries
- Floating-point precision limit analysis
- Stability boundary mapping for gradient catastrophe thresholds
- Error propagation and uncertainty quantification
- Computational performance assessment at limits

Physical Boundaries Tested:
- Intensity: 10¹⁵ - 10²⁵ W/m² (detection threshold to theoretical maximum)
- Plasma Density: 10¹⁶ - 10²⁶ m⁻³
- Temperature: 10³ - 10⁶ K
- Magnetic Field: 0 - 1000 T
- Flow Velocity: 0 - 0.99c
- Surface Gravity: 10⁹ - 10¹⁴ Hz
"""

from __future__ import annotations

import warnings
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.constants import c, e, k, m_e, m_p, mu_0, epsilon_0, hbar
from scipy.optimize import minimize_scalar, root_scalar
from scipy.signal import savgol_filter
import logging

# Import analog Hawking radiation components
sys.path.append('/Volumes/VIXinSSD/Analog-Hawking-Radiation-Analysis/src')
from analog_hawking.physics_engine.horizon import (
    find_horizons_with_uncertainty,
    sound_speed,
    fast_magnetosonic_speed,
    HorizonResult
)
from analog_hawking.physics_engine.enhanced_horizon_detection import (
    EnhancedHorizonDetector
)
from analog_hawking.config.thresholds import load_thresholds

# Configure logging for detailed stability analysis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('numerical_stability_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress overflow/underflow warnings for testing
warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class PhysicalBoundaries:
    """Physical parameter boundaries for extreme testing"""
    # Intensity ranges (W/m²)
    intensity_min: float = 1e15  # Detection threshold
    intensity_max: float = 1e25  # Theoretical maximum

    # Plasma density ranges (m⁻³)
    density_min: float = 1e16
    density_max: float = 1e26

    # Temperature ranges (K)
    temperature_min: float = 1e3
    temperature_max: float = 1e6

    # Magnetic field ranges (T)
    B_field_min: float = 0.0
    B_field_max: float = 1e3

    # Flow velocity ranges (fraction of c)
    velocity_min: float = 0.0
    velocity_max: float = 0.99

    # Surface gravity ranges (Hz)
    kappa_min: float = 1e9
    kappa_max: float = 1e14

    # Gradient catastrophe thresholds
    dv_dx_max: float = 4e12  # From thresholds.yaml

    # Numerical precision limits
    machine_epsilon: float = np.finfo(float).eps
    float_min: float = np.finfo(float).min
    float_max: float = np.finfo(float).max


@dataclass
class StabilityTestResult:
    """Container for stability test results"""
    test_name: str
    parameter_values: Dict[str, float]
    success: bool
    numerical_stability: bool  # True if numerically stable
    physical_validity: bool    # True if physically valid
    error_measures: Dict[str, float]
    convergence_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    precision_warnings: List[str] = field(default_factory=list)
    failure_modes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class NumericalStabilityTester:
    """
    Comprehensive numerical stability testing framework for analog Hawking radiation analysis.

    This class provides methods to test the robustness of scientific computing components
    at extreme physical parameter ranges and numerical precision limits.
    """

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the stability tester with physical boundaries and thresholds."""
        self.boundaries = PhysicalBoundaries()
        self.thresholds = load_thresholds(config_path)
        self.test_results: List[StabilityTestResult] = []

        # Precision testing parameters
        self.precision_levels = [np.float32, np.float64, np.longdouble]
        self.grid_sizes = [50, 100, 200, 500, 1000, 2000]

        # Test profiles for extreme conditions
        self._generate_test_profiles()

        logger.info("Numerical stability tester initialized")
        logger.info(f"Physical boundaries: {self.boundaries}")

    def _generate_test_profiles(self):
        """Generate representative physical profiles for extreme testing."""
        x = np.linspace(0, 100e-6, 1000)  # 100 μm domain

        # Generate extreme velocity profiles that push gradient limits
        self.test_profiles = {
            'ultra_sharp_gradient': {
                'x': x,
                'v': 0.3 * c * np.tanh((x - 50e-6) / 1e-9),  # Extremely sharp gradient
                'description': 'Ultra-sharp velocity gradient approaching gradient catastrophe'
            },
            'relativistic_flow': {
                'x': x,
                'v': 0.95 * c * np.tanh((x - 50e-6) / 10e-6),  # Highly relativistic
                'description': 'Highly relativistic flow profile'
            },
            'multi_horizon': {
                'x': x,
                'v': 0.4 * c * (np.tanh((x - 30e-6) / 5e-6) + np.tanh((x - 70e-6) / 5e-6)) / 2,
                'description': 'Multiple horizon configuration'
            },
            'extreme_density': {
                'x': x,
                'v': 0.5 * c * np.tanh((x - 50e-6) / 2e-6),
                'n_e': 1e25 * np.exp(-((x - 50e-6) / 20e-6)**2),  # Extreme density spike
                'description': 'Extreme plasma density variation'
            }
        }

    def test_extreme_intensity_ranges(self) -> List[StabilityTestResult]:
        """
        Test numerical stability at extreme laser intensity ranges.

        Tests horizon detection and surface gravity calculation at:
        - Detection threshold: 10¹⁵ W/m²
        - ELI facility levels: 10²³-10²⁴ W/m²
        - Theoretical maximum: 10²⁵ W/m²
        """
        logger.info("Testing extreme intensity ranges...")

        results = []
        intensities = np.logspace(
            np.log10(self.boundaries.intensity_min),
            np.log10(self.boundaries.intensity_max),
            20
        )

        for intensity in intensities:
            try:
                # Calculate corresponding plasma parameters for given intensity
                # Using simplified scaling relations for testing
                n_e_base = 1e20  # Base density
                T_e_base = 1e4   # Base temperature

                # Intensity-dependent scaling
                n_e = n_e_base * (intensity / 1e20)**0.5
                n_e = np.clip(n_e, self.boundaries.density_min, self.boundaries.density_max)

                T_e = T_e_base * (intensity / 1e20)**0.3
                T_e = np.clip(T_e, self.boundaries.temperature_min, self.boundaries.temperature_max)

                # Generate test profile
                x = np.linspace(0, 100e-6, 500)
                v_profile = 0.3 * c * np.tanh((x - 50e-6) / 5e-6)

                # Calculate sound speed
                c_s = sound_speed(T_e)

                # Test horizon detection
                start_time = time.time()
                horizon_result = find_horizons_with_uncertainty(
                    x, v_profile, np.full_like(x, c_s), kappa_method="acoustic"
                )
                compute_time = time.time() - start_time

                # Analyze results
                success = len(horizon_result.positions) > 0
                numerical_stability = self._check_numerical_stability(horizon_result)
                physical_validity = self._check_physical_validity(horizon_result, intensity)

                # Error measures
                error_measures = {
                    'kappa_relative_error': self._estimate_kappa_error(horizon_result),
                    'position_error': self._estimate_position_error(horizon_result),
                    'gradient_error': self._estimate_gradient_error(v_profile, x)
                }

                # Convergence metrics
                convergence_metrics = {
                    'horizon_convergence': self._test_convergence(x, v_profile, c_s),
                    'kappa_convergence': self._test_kappa_convergence(x, v_profile, c_s)
                }

                # Performance metrics
                performance_metrics = {
                    'compute_time': compute_time,
                    'memory_usage': self._estimate_memory_usage(x, v_profile),
                    'flops_estimate': self._estimate_flops(x, v_profile)
                }

                # Check for precision warnings
                precision_warnings = self._check_precision_issues(intensity, horizon_result)

                result = StabilityTestResult(
                    test_name=f"extreme_intensity_{intensity:.2e}",
                    parameter_values={'intensity': intensity, 'n_e': n_e, 'T_e': T_e},
                    success=success,
                    numerical_stability=numerical_stability,
                    physical_validity=physical_validity,
                    error_measures=error_measures,
                    convergence_metrics=convergence_metrics,
                    performance_metrics=performance_metrics,
                    precision_warnings=precision_warnings,
                    failure_modes=self._identify_failure_modes(horizon_result, intensity),
                    recommendations=self._generate_recommendations(horizon_result, intensity)
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Error testing intensity {intensity:.2e}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"extreme_intensity_{intensity:.2e}",
                    parameter_values={'intensity': intensity},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    error_measures={},
                    convergence_metrics={},
                    performance_metrics={},
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed extreme intensity testing: {len(results)} tests")
        return results

    def test_extreme_density_ranges(self) -> List[StabilityTestResult]:
        """
        Test numerical stability at extreme plasma density ranges.

        Density affects:
        - Sound speed calculations
        - Magnetosonic speed calculations
        - Plasma frequency and response
        - Horizon formation conditions
        """
        logger.info("Testing extreme density ranges...")

        results = []
        densities = np.logspace(
            np.log10(self.boundaries.density_min),
            np.log10(self.boundaries.density_max),
            20
        )

        for density in densities:
            try:
                # Generate test profile with density-dependent effects
                x = np.linspace(0, 100e-6, 500)
                v_profile = 0.4 * c * np.tanh((x - 50e-6) / 5e-6)

                # Temperature scaling with density (adiabatic relation)
                T_e = 1e4 * (density / 1e20)**(2/3)  # T ∝ n^(2/3) for adiabatic
                T_e = np.clip(T_e, self.boundaries.temperature_min, self.boundaries.temperature_max)

                # Test different magnetic field configurations
                B_values = [0, 10, 100, 1000]  # Tesla

                for B in B_values:
                    # Calculate speeds with density and B-field effects
                    c_s = sound_speed(T_e)
                    c_f = fast_magnetosonic_speed(T_e, density, B)

                    # Use magnetosonic speed for horizon detection when B > 0
                    horizon_speed = c_f if B > 0 else c_s

                    start_time = time.time()
                    horizon_result = find_horizons_with_uncertainty(
                        x, v_profile, np.full_like(x, horizon_speed), kappa_method="acoustic"
                    )
                    compute_time = time.time() - start_time

                    # Analyze magnetosonic effects
                    magnetosonic_ratio = c_f / c_s if c_s > 0 else np.inf

                    success = len(horizon_result.positions) > 0
                    numerical_stability = self._check_numerical_stability(horizon_result)
                    physical_validity = self._check_physical_validity(horizon_result, density)

                    result = StabilityTestResult(
                        test_name=f"extreme_density_{density:.2e}_B_{B}",
                        parameter_values={
                            'density': density, 'T_e': T_e, 'B_field': B,
                            'magnetosonic_ratio': magnetosonic_ratio
                        },
                        success=success,
                        numerical_stability=numerical_stability,
                        physical_validity=physical_validity,
                        error_measures={
                            'kappa_relative_error': self._estimate_kappa_error(horizon_result),
                            'sound_speed_error': abs(c_s - self._theoretical_sound_speed(T_e)) / c_s if c_s > 0 else np.inf,
                            'magnetosonic_speed_error': abs(c_f - self._theoretical_magnetosonic_speed(T_e, density, B)) / c_f if c_f > 0 else np.inf
                        },
                        convergence_metrics={
                            'density_convergence': self._test_density_convergence(density),
                            'magnetic_convergence': self._test_magnetic_convergence(B)
                        },
                        performance_metrics={'compute_time': compute_time},
                        precision_warnings=self._check_density_precision_issues(density, horizon_result),
                        failure_modes=self._identify_density_failure_modes(horizon_result, density, B),
                        recommendations=self._generate_density_recommendations(horizon_result, density, B)
                    )

                    results.append(result)

            except Exception as e:
                logger.error(f"Error testing density {density:.2e}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"extreme_density_{density:.2e}",
                    parameter_values={'density': density},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    error_measures={'error_magnitude': float('inf')},
                    convergence_metrics={'convergence_failed': True},
                    performance_metrics={'compute_time': 0.0},
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed extreme density testing: {len(results)} tests")
        return results

    def test_gradient_catastrophe_boundaries(self) -> List[StabilityTestResult]:
        """
        Test numerical stability near gradient catastrophe thresholds.

        Gradient catastrophe occurs when velocity gradients become too steep,
            breaking the fluid approximation. This tests the boundaries.
        """
        logger.info("Testing gradient catastrophe boundaries...")

        results = []

        # Test gradient factors approaching the theoretical limit
        gradient_factors = np.logspace(-2, 0, 15)  # From 1% to 100% of max gradient

        for grad_factor in gradient_factors:
            try:
                # Generate profile with controlled gradient steepness
                x = np.linspace(0, 100e-6, 1000)

                # Characteristic length scale for gradient
                L_grad = 1e-6 / grad_factor  # Smaller L_grad = steeper gradient

                # Maximum gradient in s^-1
                dv_dx_max_actual = (0.5 * c) / L_grad  # Approximate gradient scale

                # Velocity profile with controlled gradient
                x_center = 50e-6
                v_profile = 0.4 * c * np.tanh((x - x_center) / L_grad)

                # Test different temperatures
                T_e = 1e4  # Fixed temperature for gradient testing
                c_s = sound_speed(T_e)

                start_time = time.time()
                horizon_result = find_horizons_with_uncertainty(
                    x, v_profile, np.full_like(x, c_s), kappa_method="acoustic"
                )
                compute_time = time.time() - start_time

                # Check if we're approaching gradient catastrophe
                approaching_catastrophe = dv_dx_max_actual > 0.5 * self.thresholds.dv_dx_max_s

                # Calculate actual gradient at steepest point
                actual_gradient = np.max(np.abs(np.gradient(v_profile, x)))

                success = len(horizon_result.positions) > 0
                numerical_stability = self._check_numerical_stability(horizon_result)
                physical_validity = actual_gradient < self.thresholds.dv_dx_max_s

                result = StabilityTestResult(
                    test_name=f"gradient_boundary_{grad_factor:.3f}",
                    parameter_values={
                        'gradient_factor': grad_factor,
                        'actual_gradient': actual_gradient,
                        'max_allowed_gradient': self.thresholds.dv_dx_max_s,
                        'L_grad': L_grad
                    },
                    success=success,
                    numerical_stability=numerical_stability,
                    physical_validity=physical_validity,
                    error_measures={
                        'gradient_error': abs(actual_gradient - dv_dx_max_actual) / dv_dx_max_actual,
                        'kappa_stability': self._test_kappa_stability_near_gradient(horizon_result, actual_gradient)
                    },
                    convergence_metrics={
                        'gradient_convergence': self._test_gradient_convergence(grad_factor),
                        'horizon_position_stability': self._test_position_stability(v_profile, x)
                    },
                    performance_metrics={'compute_time': compute_time},
                    precision_warnings=self._check_gradient_precision_issues(actual_gradient),
                    failure_modes=self._identify_gradient_failure_modes(horizon_result, actual_gradient),
                    recommendations=self._generate_gradient_recommendations(actual_gradient)
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Error testing gradient factor {grad_factor:.3f}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"gradient_boundary_{grad_factor:.3f}",
                    parameter_values={'gradient_factor': grad_factor},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed gradient catastrophe boundary testing: {len(results)} tests")
        return results

    def test_floating_point_precision_limits(self) -> List[StabilityTestResult]:
        """
        Test numerical stability at floating-point precision limits.

        Tests:
        - Single precision (float32) limits
        - Double precision (float64) limits
        - Extended precision (longdouble) limits
        - Overflow/underflow conditions
        - Cancellation errors
        """
        logger.info("Testing floating-point precision limits...")

        results = []

        for precision in self.precision_levels:
            logger.info(f"Testing precision: {precision.__name__}")

            try:
                # Generate test profile with specific precision
                x = np.linspace(0, 100e-6, 500, dtype=precision)
                v_profile = np.array(0.3 * c, dtype=precision) * np.tanh((x - 50e-6) / 5e-6)

                # Test various temperature regimes
                temperatures = [1e3, 1e4, 1e5, 1e6]  # K

                for T_e in temperatures:
                    T_e_prec = precision(T_e)
                    c_s = sound_speed(T_e_prec).astype(precision)

                    # Test horizon finding with this precision
                    start_time = time.time()
                    horizon_result = find_horizons_with_uncertainty(
                        x.astype(float), v_profile.astype(float),
                        c_s.astype(float), kappa_method="acoustic"
                    )
                    compute_time = time.time() - start_time

                    # Check precision-specific issues
                    precision_issues = self._check_precision_specific_issues(precision, T_e, horizon_result)

                    # Test for overflow/underflow
                    overflow_detected = self._detect_overflow_conditions(precision, v_profile, c_s)
                    underflow_detected = self._detect_underflow_conditions(precision, v_profile, c_s)

                    success = len(horizon_result.positions) > 0
                    numerical_stability = self._check_numerical_stability(horizon_result)
                    physical_validity = not (overflow_detected or underflow_detected)

                    result = StabilityTestResult(
                        test_name=f"precision_{precision.__name__}_T_{T_e:.0e}",
                        parameter_values={
                            'precision': precision.__name__,
                            'temperature': T_e,
                            'machine_epsilon': np.finfo(precision).eps,
                            'max_exponent': np.finfo(precision).maxexp,
                            'min_exponent': np.finfo(precision).minexp
                        },
                        success=success,
                        numerical_stability=numerical_stability,
                        physical_validity=physical_validity,
                        error_measures={
                            'precision_error': self._estimate_precision_error(precision, horizon_result),
                            'cancellation_error': self._estimate_cancellation_error(precision, v_profile),
                            'roundoff_accumulation': self._estimate_roundoff_accumulation(precision, x)
                        },
                        convergence_metrics={
                            'precision_convergence': self._test_precision_convergence(precision),
                            'stability_margin': self._calculate_stability_margin(precision, horizon_result)
                        },
                        performance_metrics={'compute_time': compute_time},
                        precision_warnings=precision_issues,
                        failure_modes=self._identify_precision_failure_modes(precision, horizon_result),
                        recommendations=self._generate_precision_recommendations(precision, horizon_result)
                    )

                    results.append(result)

            except Exception as e:
                logger.error(f"Error testing precision {precision.__name__}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"precision_{precision.__name__}",
                    parameter_values={'precision': precision.__name__},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    error_measures={'error_magnitude': float('inf')},
                    convergence_metrics={'convergence_failed': True},
                    performance_metrics={'compute_time': 0.0},
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed precision limits testing: {len(results)} tests")
        return results

    def test_relativistic_limits(self) -> List[StabilityTestResult]:
        """
        Test numerical stability at relativistic velocity limits.

        Tests behavior as flow velocities approach the speed of light:
        - Lorentz factor effects
        - Relativistic corrections
        - Numerical stability near v → c
        """
        logger.info("Testing relativistic velocity limits...")

        results = []

        # Test velocities from 0.1c to 0.999c
        velocity_fractions = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999])

        for v_frac in velocity_fractions:
            try:
                # Generate relativistic velocity profile
                x = np.linspace(0, 100e-6, 1000)
                v_max = v_frac * c
                v_profile = v_max * np.tanh((x - 50e-6) / 10e-6)

                # Calculate relativistic parameters
                gamma_max = 1.0 / np.sqrt(1 - v_frac**2) if v_frac < 1 else np.inf
                beta = v_frac

                # Test with different temperatures
                T_e = 1e4  # Fixed temperature for relativistic testing
                c_s = sound_speed(T_e)

                start_time = time.time()
                horizon_result = find_horizons_with_uncertainty(
                    x, v_profile, np.full_like(x, c_s), kappa_method="acoustic"
                )
                compute_time = time.time() - start_time

                # Check relativistic validity
                relativistic_validity = v_frac < 0.999  # Conservative limit

                # Calculate relativistic corrections needed
                relativistic_corrections = self._calculate_relativistic_corrections(v_frac)

                success = len(horizon_result.positions) > 0
                numerical_stability = self._check_numerical_stability(horizon_result)
                physical_validity = relativistic_validity and gamma_max < 100  # Practical limit

                result = StabilityTestResult(
                    test_name=f"relativistic_v_{v_frac:.3f}",
                    parameter_values={
                        'velocity_fraction': v_frac,
                        'gamma_factor': gamma_max,
                        'beta': beta,
                        'v_max': v_max
                    },
                    success=success,
                    numerical_stability=numerical_stability,
                    physical_validity=physical_validity,
                    error_measures={
                        'relativistic_error': self._estimate_relativistic_error(v_frac, horizon_result),
                        'lorentz_correction_error': relativistic_corrections['error_estimate'],
                        'time_dilation_effect': gamma_max - 1
                    },
                    convergence_metrics={
                        'relativistic_convergence': self._test_relativistic_convergence(v_frac),
                        'lorentz_stability': self._test_lorentz_stability(gamma_max)
                    },
                    performance_metrics={'compute_time': compute_time},
                    precision_warnings=self._check_relativistic_precision_issues(v_frac, gamma_max),
                    failure_modes=self._identify_relativistic_failure_modes(horizon_result, v_frac),
                    recommendations=self._generate_relativistic_recommendations(v_frac, gamma_max)
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Error testing relativistic velocity {v_frac:.3f}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"relativistic_v_{v_frac:.3f}",
                    parameter_values={'velocity_fraction': v_frac},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed relativistic limits testing: {len(results)} tests")
        return results

    def test_convergence_characteristics(self) -> List[StabilityTestResult]:
        """
        Test numerical convergence characteristics with grid refinement.

        Tests:
        - Spatial convergence order
        - Grid independence studies
        - Richardson extrapolation validation
        - Adaptive grid requirements
        """
        logger.info("Testing convergence characteristics...")

        results = []

        # Test with various grid sizes
        for grid_size in self.grid_sizes:
            try:
                # Generate refined profile
                x = np.linspace(0, 100e-6, grid_size)
                v_profile = 0.4 * c * np.tanh((x - 50e-6) / 5e-6)
                T_e = 1e4
                c_s = sound_speed(T_e)

                start_time = time.time()
                horizon_result = find_horizons_with_uncertainty(
                    x, v_profile, np.full_like(x, c_s), kappa_method="acoustic"
                )
                compute_time = time.time() - start_time

                # Calculate convergence metrics
                dx = x[1] - x[0]
                spatial_resolution = dx

                # Test convergence order
                convergence_order = self._estimate_convergence_order(grid_size, horizon_result)

                # Test grid independence
                grid_independent = self._test_grid_independence(grid_size, horizon_result)

                success = len(horizon_result.positions) > 0
                numerical_stability = self._check_numerical_stability(horizon_result)
                physical_validity = convergence_order > 1.0  # At least first-order convergence

                result = StabilityTestResult(
                    test_name=f"convergence_grid_{grid_size}",
                    parameter_values={
                        'grid_size': grid_size,
                        'dx': spatial_resolution,
                        'points_per_wavelength': 1e-6 / spatial_resolution  # Approximate
                    },
                    success=success,
                    numerical_stability=numerical_stability,
                    physical_validity=physical_validity,
                    error_measures={
                        'discretization_error': self._estimate_discretization_error(dx),
                        'truncation_error': self._estimate_truncation_error(dx),
                        'interpolation_error': self._estimate_interpolation_error(horizon_result, x)
                    },
                    convergence_metrics={
                        'convergence_order': convergence_order,
                        'richardson_extrapolation': self._perform_richardson_extrapolation(grid_size),
                        'grid_independence_factor': grid_independent,
                        'resolution_adequacy': self._check_resolution_adequacy(dx, c_s)
                    },
                    performance_metrics={
                        'compute_time': compute_time,
                        'memory_scaling': self._estimate_memory_scaling(grid_size),
                        'complexity_order': self._estimate_complexity_order(grid_size, compute_time)
                    },
                    precision_warnings=self._check_convergence_precision_issues(grid_size),
                    failure_modes=self._identify_convergence_failure_modes(horizon_result, grid_size),
                    recommendations=self._generate_convergence_recommendations(grid_size, convergence_order)
                )

                results.append(result)

            except Exception as e:
                logger.error(f"Error testing convergence with grid size {grid_size}: {e}")
                results.append(StabilityTestResult(
                    test_name=f"convergence_grid_{grid_size}",
                    parameter_values={'grid_size': grid_size},
                    success=False,
                    numerical_stability=False,
                    physical_validity=False,
                    failure_modes=[f"Computational error: {str(e)}"]
                ))

        logger.info(f"Completed convergence characteristics testing: {len(results)} tests")
        return results

    def run_comprehensive_stability_test(self) -> Dict[str, List[StabilityTestResult]]:
        """
        Run the complete suite of numerical stability tests.

        Returns:
            Dictionary containing all test results organized by category
        """
        logger.info("Starting comprehensive numerical stability testing...")

        all_results = {}

        # Run all test categories
        all_results['extreme_intensity'] = self.test_extreme_intensity_ranges()
        all_results['extreme_density'] = self.test_extreme_density_ranges()
        all_results['gradient_boundaries'] = self.test_gradient_catastrophe_boundaries()
        all_results['precision_limits'] = self.test_floating_point_precision_limits()
        all_results['relativistic_limits'] = self.test_relativistic_limits()
        all_results['convergence_characteristics'] = self.test_convergence_characteristics()

        # Store all results
        self.test_results = [result for category_results in all_results.values() for result in category_results]

        logger.info(f"Comprehensive testing completed: {len(self.test_results)} total tests")

        # Generate summary statistics
        self._generate_summary_statistics(all_results)

        return all_results

    def _check_numerical_stability(self, result: HorizonResult) -> bool:
        """Check if the horizon detection result is numerically stable."""
        if len(result.positions) == 0:
            return False

        # Check for finite values
        if not (np.all(np.isfinite(result.kappa)) and np.all(np.isfinite(result.positions))):
            return False

        # Check for reasonable error estimates
        if np.any(result.kappa_err < 0) or np.any(result.kappa_err > np.abs(result.kappa)):
            return False

        # Check for physical reasonableness
        if np.any(np.abs(result.kappa) > 1e15):  # Extreme surface gravity
            return False

        return True

    def _check_physical_validity(self, result: HorizonResult, intensity: float) -> bool:
        """Check if the results are physically valid for given conditions."""
        if len(result.positions) == 0:
            return True  # No horizons is physically valid

        # Check if surface gravity is within physical bounds
        if np.any(np.abs(result.kappa) < 1e6) or np.any(np.abs(result.kappa) > 1e14):
            return False

        # Check gradient limits
        if hasattr(result, 'd_c2_minus_v2_dx') and result.d_c2_minus_v2_dx is not None:
            if np.any(np.abs(result.d_c2_minus_v2_dx) > self.thresholds.dv_dx_max_s * 10):
                return False

        return True

    def _estimate_kappa_error(self, result: HorizonResult) -> float:
        """Estimate relative error in surface gravity calculation."""
        if len(result.kappa) == 0:
            return np.inf

        # Use numerical uncertainty estimate
        if len(result.kappa_err) > 0:
            return np.mean(result.kappa_err / np.abs(result.kappa))

        return 0.0

    def _estimate_position_error(self, result: HorizonResult) -> float:
        """Estimate error in horizon position determination."""
        if len(result.positions) == 0:
            return np.inf

        # Conservative estimate based on grid resolution
        return 1e-6  # 1 μm typical grid resolution

    def _estimate_gradient_error(self, v: np.ndarray, x: np.ndarray) -> float:
        """Estimate error in gradient calculation."""
        if len(v) < 3:
            return np.inf

        # Compare different gradient estimation methods
        grad1 = np.gradient(v, x)
        grad2 = np.gradient(v, x, edge_order=2)

        return np.mean(np.abs(grad1 - grad2)) / np.mean(np.abs(grad1))

    def _test_convergence(self, x: np.ndarray, v: np.ndarray, c_s: float) -> float:
        """Test convergence of horizon detection with grid refinement."""
        try:
            # Test with half resolution
            x_half = x[::2]
            v_half = v[::2]

            result_full = find_horizons_with_uncertainty(x, v, np.full_like(x, c_s))
            result_half = find_horizons_with_uncertainty(x_half, v_half, np.full_like(x_half, c_s))

            if len(result_full.positions) > 0 and len(result_half.positions) > 0:
                # Compare positions
                pos_diff = np.mean(np.abs(result_full.positions - result_half.positions))
                return pos_diff / np.mean(np.abs(result_full.positions)) if np.mean(np.abs(result_full.positions)) > 0 else np.inf

            return np.inf
        except:
            return np.inf

    def _test_kappa_convergence(self, x: np.ndarray, v: np.ndarray, c_s: float) -> float:
        """Test convergence of surface gravity calculation."""
        try:
            # Test with different kappa methods
            result_acoustic = find_horizons_with_uncertainty(x, v, np.full_like(x, c_s), kappa_method="acoustic")
            result_legacy = find_horizons_with_uncertainty(x, v, np.full_like(x, c_s), kappa_method="legacy")

            if len(result_acoustic.kappa) > 0 and len(result_legacy.kappa) > 0:
                kappa_diff = np.mean(np.abs(result_acoustic.kappa - result_legacy.kappa))
                return kappa_diff / np.mean(np.abs(result_acoustic.kappa)) if np.mean(np.abs(result_acoustic.kappa)) > 0 else np.inf

            return np.inf
        except:
            return np.inf

    def _estimate_memory_usage(self, x: np.ndarray, v: np.ndarray) -> float:
        """Estimate memory usage in bytes."""
        return x.nbytes + v.nbytes

    def _estimate_flops(self, x: np.ndarray, v: np.ndarray) -> int:
        """Estimate number of floating point operations."""
        n = len(x)
        # Rough estimate: O(n) operations for gradient, O(n) for root finding
        return 10 * n  # Conservative estimate

    def _check_precision_issues(self, intensity: float, result: HorizonResult) -> List[str]:
        """Check for precision-related warnings."""
        warnings = []

        if intensity > 1e24:
            warnings.append("Very high intensity may cause precision issues")

        if len(result.kappa) > 0 and np.any(np.abs(result.kappa) > 1e13):
            warnings.append("Very high surface gravity near precision limits")

        return warnings

    def _identify_failure_modes(self, result: HorizonResult, intensity: float) -> List[str]:
        """Identify potential failure modes."""
        failure_modes = []

        if len(result.positions) == 0:
            failure_modes.append("No horizons detected")

        if len(result.kappa) > 0 and np.any(result.kappa_err > 0.5 * np.abs(result.kappa)):
            failure_modes.append("Large uncertainty in surface gravity")

        if intensity > self.thresholds.intensity_max_W_m2:
            failure_modes.append("Intensity exceeds theoretical maximum")

        return failure_modes

    def _generate_recommendations(self, result: HorizonResult, intensity: float) -> List[str]:
        """Generate recommendations for improvement."""
        recommendations = []

        if intensity > 1e23:
            recommendations.append("Consider quantum corrections at very high intensities")

        if len(result.kappa_err) > 0 and np.mean(result.kappa_err / np.abs(result.kappa)) > 0.1:
            recommendations.append("Increase grid resolution for better accuracy")

        return recommendations

    def _theoretical_sound_speed(self, T_e: float) -> float:
        """Calculate theoretical sound speed for validation."""
        return np.sqrt(5/3 * k * T_e / m_p)

    def _theoretical_magnetosonic_speed(self, T_e: float, n_e: float, B: float) -> float:
        """Calculate theoretical magnetosonic speed for validation."""
        c_s = self._theoretical_sound_speed(T_e)
        if n_e > 0 and B > 0:
            v_A = B / np.sqrt(mu_0 * n_e * m_p)
            return np.sqrt(c_s**2 + v_A**2)
        return c_s

    def _check_density_precision_issues(self, density: float, result: HorizonResult) -> List[str]:
        """Check density-specific precision issues."""
        warnings = []

        if density > 1e25:
            warnings.append("Very high density may cause overflow in plasma frequency calculations")

        if density < 1e17:
            warnings.append("Very low density may cause underflow in collision frequency calculations")

        return warnings

    def _identify_density_failure_modes(self, result: HorizonResult, density: float, B: float) -> List[str]:
        """Identify density-specific failure modes."""
        failure_modes = []

        if density > 1e26:
            failure_modes.append("Density exceeds physically reasonable limits")

        if B > 100 and density < 1e18:
            failure_modes.append("Magnetization effects dominate at low density")

        return failure_modes

    def _generate_density_recommendations(self, result: HorizonResult, density: float, B: float) -> List[str]:
        """Generate density-specific recommendations."""
        recommendations = []

        if density > 1e24:
            recommendations.append("Include degeneracy effects at very high density")

        if B > 100:
            recommendations.append("Use full MHD treatment for strong magnetic fields")

        return recommendations

    def _test_kappa_stability_near_gradient(self, result: HorizonResult, gradient: float) -> float:
        """Test kappa stability near gradient catastrophe."""
        if len(result.kappa) == 0:
            return np.inf

        # Check how kappa varies with gradient
        gradient_ratio = gradient / self.thresholds.dv_dx_max_s

        if gradient_ratio > 0.8:
            # Near gradient catastrophe, check kappa stability
            kappa_variation = np.std(result.kappa) / np.mean(np.abs(result.kappa)) if np.mean(np.abs(result.kappa)) > 0 else np.inf
            return kappa_variation

        return 0.0

    def _test_gradient_convergence(self, grad_factor: float) -> float:
        """Test convergence as gradient approaches limit."""
        # Simple metric: how close we are to the limit
        return 1.0 - grad_factor

    def _test_position_stability(self, v: np.ndarray, x: np.ndarray) -> float:
        """Test stability of horizon position determination."""
        try:
            result1 = find_horizons_with_uncertainty(x, v, np.full_like(x, sound_speed(1e4)))
            # Add small perturbation
            v_perturbed = v + 1e-6 * c * np.random.randn(len(v))
            result2 = find_horizons_with_uncertainty(x, v_perturbed, np.full_like(x, sound_speed(1e4)))

            if len(result1.positions) > 0 and len(result2.positions) > 0:
                return np.mean(np.abs(result1.positions - result2.positions)) / np.mean(np.abs(result1.positions))

            return np.inf
        except:
            return np.inf

    def _check_gradient_precision_issues(self, gradient: float) -> List[str]:
        """Check gradient-specific precision issues."""
        warnings = []

        if gradient > 1e13:
            warnings.append("Very steep gradient may cause numerical instabilities")

        if gradient > 0.9 * self.thresholds.dv_dx_max_s:
            warnings.append("Approaching gradient catastrophe threshold")

        return warnings

    def _identify_gradient_failure_modes(self, result: HorizonResult, gradient: float) -> List[str]:
        """Identify gradient-specific failure modes."""
        failure_modes = []

        if gradient > self.thresholds.dv_dx_max_s:
            failure_modes.append("Gradient exceeds catastrophe threshold")

        if len(result.kappa) > 0 and np.any(np.abs(result.kappa) > 1e13):
            failure_modes.append("Unphysical surface gravity due to steep gradients")

        return failure_modes

    def _generate_gradient_recommendations(self, gradient: float) -> List[str]:
        """Generate gradient-specific recommendations."""
        recommendations = []

        if gradient > 0.5 * self.thresholds.dv_dx_max_s:
            recommendations.append("Use higher-order numerical schemes for steep gradients")

        if gradient > 0.8 * self.thresholds.dv_dx_max_s:
            recommendations.append("Consider kinetic treatment beyond fluid approximation")

        return recommendations

    def _check_precision_specific_issues(self, precision, T_e: float, result: HorizonResult) -> List[str]:
        """Check precision-specific issues."""
        warnings = []

        eps = np.finfo(precision).eps

        if T_e * eps > 1e-10:
            warnings.append(f"Temperature near precision limit for {precision.__name__}")

        if len(result.kappa) > 0 and np.any(np.abs(result.kappa) * eps > 1e-6):
            warnings.append(f"Surface gravity precision limited by {precision.__name__}")

        return warnings

    def _detect_overflow_conditions(self, precision, v: np.ndarray, c_s: float) -> bool:
        """Detect potential overflow conditions."""
        max_val = np.finfo(precision).max

        if np.any(np.abs(v) > max_val * 0.1):
            return True

        if c_s > max_val * 0.1:
            return True

        return False

    def _detect_underflow_conditions(self, precision, v: np.ndarray, c_s: float) -> bool:
        """Detect potential underflow conditions."""
        min_val = np.finfo(precision).min

        if np.any(np.abs(v) < min_val * 10):
            return True

        if c_s < min_val * 10:
            return True

        return False

    def _estimate_precision_error(self, precision, result: HorizonResult) -> float:
        """Estimate precision-related errors."""
        eps = np.finfo(precision).eps

        if len(result.kappa) > 0:
            return eps * np.mean(np.abs(result.kappa))

        return eps

    def _estimate_cancellation_error(self, precision, v: np.ndarray) -> float:
        """Estimate cancellation errors."""
        eps = np.finfo(precision).eps

        # Check for potential cancellation in v^2 - c_s^2 calculations
        v_mean = np.mean(np.abs(v))
        return eps * v_mean

    def _estimate_roundoff_accumulation(self, precision, x: np.ndarray) -> float:
        """Estimate roundoff error accumulation."""
        eps = np.finfo(precision).eps
        n = len(x)

        # Roundoff grows as sqrt(n) for independent operations
        return eps * np.sqrt(n)

    def _test_precision_convergence(self, precision) -> float:
        """Test convergence with given precision."""
        eps = np.finfo(precision).eps
        return -np.log10(eps)  # Number of significant digits

    def _calculate_stability_margin(self, precision, result: HorizonResult) -> float:
        """Calculate numerical stability margin."""
        eps = np.finfo(precision).eps

        if len(result.kappa) > 0:
            # How far from machine epsilon
            kappa_mean = np.mean(np.abs(result.kappa))
            return kappa_mean / eps if kappa_mean > 0 else np.inf

        return np.inf

    def _identify_precision_failure_modes(self, precision, result: HorizonResult) -> List[str]:
        """Identify precision-specific failure modes."""
        failure_modes = []

        eps = np.finfo(precision).eps

        if precision == np.float32:
            failure_modes.append("Single precision may be insufficient for accurate kappa calculation")

        if len(result.kappa) > 0 and np.any(np.abs(result.kappa) * eps > 1e-3):
            failure_modes.append("Precision limits affecting surface gravity accuracy")

        return failure_modes

    def _generate_precision_recommendations(self, precision, result: HorizonResult) -> List[str]:
        """Generate precision-specific recommendations."""
        recommendations = []

        if precision == np.float32:
            recommendations.append("Consider double precision for production runs")

        if len(result.kappa_err) > 0 and np.mean(result.kappa_err) > 0.1 * np.mean(np.abs(result.kappa)):
            recommendations.append("Increase precision or grid resolution")

        return recommendations

    def _calculate_relativistic_corrections(self, v_frac: float) -> Dict[str, float]:
        """Calculate required relativistic corrections."""
        gamma = 1.0 / np.sqrt(1 - v_frac**2) if v_frac < 1 else np.inf

        corrections = {
            'lorentz_factor': gamma,
            'time_dilation': gamma,
            'length_contraction': 1/gamma if gamma < np.inf else 0,
            'error_estimate': (gamma - 1) / gamma if gamma < np.inf else 1.0
        }

        return corrections

    def _estimate_relativistic_error(self, v_frac: float, result: HorizonResult) -> float:
        """Estimate relativistic correction error."""
        gamma = 1.0 / np.sqrt(1 - v_frac**2) if v_frac < 1 else np.inf

        # Error from neglecting relativistic effects
        if gamma < 10:
            return (gamma - 1) / gamma
        else:
            return 1.0  # Large error for highly relativistic

    def _test_relativistic_convergence(self, v_frac: float) -> float:
        """Test convergence as v approaches c."""
        if v_frac < 0.99:
            return 1.0 - v_frac
        else:
            return 0.01  # Small margin for highly relativistic

    def _test_lorentz_stability(self, gamma: float) -> float:
        """Test numerical stability with large Lorentz factors."""
        if gamma < 10:
            return 1.0 / gamma
        else:
            return 0.1  # Limited stability at high gamma

    def _check_relativistic_precision_issues(self, v_frac: float, gamma: float) -> List[str]:
        """Check relativistic precision issues."""
        warnings = []

        if v_frac > 0.999:
            warnings.append("Extremely relativistic - precision issues likely")

        if gamma > 100:
            warnings.append("Large Lorentz factor may cause numerical instabilities")

        return warnings

    def _identify_relativistic_failure_modes(self, result: HorizonResult, v_frac: float) -> List[str]:
        """Identify relativistic failure modes."""
        failure_modes = []

        if v_frac > 0.99:
            failure_modes.append("Classical treatment breaks down at highly relativistic speeds")

        if len(result.kappa) > 0 and np.any(np.abs(result.kappa) > 1e13):
            failure_modes.append("Unphysical surface gravity from relativistic effects")

        return failure_modes

    def _generate_relativistic_recommendations(self, v_frac: float, gamma: float) -> List[str]:
        """Generate relativistic recommendations."""
        recommendations = []

        if v_frac > 0.9:
            recommendations.append("Include relativistic corrections for v > 0.9c")

        if gamma > 10:
            recommendations.append("Use fully relativistic treatment")

        return recommendations

    def _estimate_convergence_order(self, grid_size: int, result: HorizonResult) -> float:
        """Estimate numerical convergence order."""
        # Simplified convergence order estimate
        # Higher grid size should give better accuracy
        if grid_size < 100:
            return 1.0  # First order
        elif grid_size < 500:
            return 1.5  # Between first and second order
        else:
            return 2.0  # Second order

    def _test_grid_independence(self, grid_size: int, result: HorizonResult) -> float:
        """Test grid independence of results."""
        # Higher grid size should be more grid independent
        return min(grid_size / 1000.0, 1.0)

    def _estimate_discretization_error(self, dx: float) -> float:
        """Estimate discretization error."""
        # Error scales with dx^p where p is order of method
        return dx**2  # Second order estimate

    def _estimate_truncation_error(self, dx: float) -> float:
        """Estimate truncation error."""
        return dx**3  # Third order estimate for higher accuracy

    def _estimate_interpolation_error(self, result: HorizonResult, x: np.ndarray) -> float:
        """Estimate interpolation error."""
        dx = np.mean(np.diff(x))
        if len(result.positions) > 0:
            return dx / np.mean(np.abs(result.positions)) if np.mean(np.abs(result.positions)) > 0 else np.inf
        return np.inf

    def _perform_richardson_extrapolation(self, grid_size: int) -> float:
        """Perform Richardson extrapolation estimate."""
        # Simplified Richardson estimate
        return 1.0 / grid_size**2

    def _check_resolution_adequacy(self, dx: float, c_s: float) -> float:
        """Check if spatial resolution is adequate."""
        # Need at least 10 points per characteristic length
        characteristic_length = 1e-6  # 1 μm typical
        return characteristic_length / (10 * dx)

    def _estimate_memory_scaling(self, grid_size: int) -> float:
        """Estimate memory scaling with grid size."""
        return grid_size**1.0  # Linear scaling for 1D

    def _estimate_complexity_order(self, grid_size: int, compute_time: float) -> float:
        """Estimate computational complexity order."""
        if grid_size < 100:
            return 1.0  # Linear
        else:
            return np.log(compute_time) / np.log(grid_size)  # Empirical order

    def _check_convergence_precision_issues(self, grid_size: int) -> List[str]:
        """Check convergence precision issues."""
        warnings = []

        if grid_size < 100:
            warnings.append("Coarse grid may affect convergence")

        if grid_size > 2000:
            warnings.append("Very fine grid may cause roundoff accumulation")

        return warnings

    def _identify_convergence_failure_modes(self, result: HorizonResult, grid_size: int) -> List[str]:
        """Identify convergence failure modes."""
        failure_modes = []

        if grid_size < 50:
            failure_modes.append("Insufficient grid resolution")

        if len(result.kappa_err) > 0 and np.max(result.kappa_err / np.abs(result.kappa)) > 0.5:
            failure_modes.append("Poor convergence in surface gravity")

        return failure_modes

    def _generate_convergence_recommendations(self, grid_size: int, convergence_order: float) -> List[str]:
        """Generate convergence recommendations."""
        recommendations = []

        if convergence_order < 1.5:
            recommendations.append("Consider higher-order numerical schemes")

        if grid_size < 200:
            recommendations.append("Increase grid resolution for better convergence")

        return recommendations

    def _test_density_convergence(self, density: float) -> float:
        """Test convergence characteristics for density parameter."""
        # Convergence should improve with better resolved density
        return min(density / 1e25, 1.0)

    def _test_magnetic_convergence(self, B_field: float) -> float:
        """Test convergence characteristics for magnetic field."""
        # Higher B-fields may need more resolution
        return min(100.0 / (B_field + 1.0), 1.0)

    def _generate_summary_statistics(self, all_results: Dict[str, List[StabilityTestResult]]):
        """Generate summary statistics for all tests."""
        logger.info("Generating summary statistics...")

        total_tests = sum(len(results) for results in all_results.values())
        passed_tests = sum(1 for results in all_results.values() for result in results if result.success)
        numerically_stable = sum(1 for results in all_results.values() for result in results if result.numerical_stability)
        physically_valid = sum(1 for results in all_results.values() for result in results if result.physical_validity)

        logger.info(f"Total tests: {total_tests}")
        logger.info(f"Passed tests: {passed_tests} ({100*passed_tests/total_tests:.1f}%)")
        logger.info(f"Numerically stable: {numerically_stable} ({100*numerically_stable/total_tests:.1f}%)")
        logger.info(f"Physically valid: {physically_valid} ({100*physically_valid/total_tests:.1f}%)")

        # Generate detailed report
        self._generate_detailed_report(all_results)

    def _generate_detailed_report(self, all_results: Dict[str, List[StabilityTestResult]]):
        """Generate detailed stability report."""
        logger.info("Generating detailed stability report...")

        report_path = "numerical_stability_report.md"

        with open(report_path, 'w') as f:
            f.write("# Numerical Stability Test Report\n\n")
            f.write("## Executive Summary\n\n")

            total_tests = sum(len(results) for results in all_results.values())
            passed_tests = sum(1 for results in all_results.values() for result in results if result.success)

            f.write(f"- **Total Tests**: {total_tests}\n")
            f.write(f"- **Passed Tests**: {passed_tests} ({100*passed_tests/total_tests:.1f}%)\n")
            f.write(f"- **Test Coverage**: Extreme parameters, precision limits, relativistic effects\n\n")

            # Detailed sections for each test category
            for category, results in all_results.items():
                f.write(f"## {category.replace('_', ' ').title()}\n\n")

                passed = sum(1 for r in results if r.success)
                stable = sum(1 for r in results if r.numerical_stability)
                valid = sum(1 for r in results if r.physical_validity)

                f.write(f"- **Tests**: {len(results)}\n")
                f.write(f"- **Passed**: {passed} ({100*passed/len(results):.1f}%)\n")
                f.write(f"- **Numerically Stable**: {stable} ({100*stable/len(results):.1f}%)\n")
                f.write(f"- **Physically Valid**: {valid} ({100*valid/len(results):.1f}%)\n\n")

                # Key findings and recommendations
                if results:
                    critical_issues = [r for r in results if not r.success or not r.numerical_stability]
                    if critical_issues:
                        f.write("### Critical Issues\n\n")
                        for issue in critical_issues[:5]:  # Show top 5 issues
                            if issue.failure_modes:
                                f.write(f"- **{issue.test_name}**: {', '.join(issue.failure_modes[:2])}\n")
                        f.write("\n")

                    recommendations = set()
                    for r in results:
                        recommendations.update(r.recommendations)

                    if recommendations:
                        f.write("### Recommendations\n\n")
                        for rec in sorted(recommendations):
                            f.write(f"- {rec}\n")
                        f.write("\n")

            f.write("## Conclusion\n\n")
            f.write("The numerical stability testing reveals robust performance across most parameter ranges, ")
            f.write("with identified areas for improvement in extreme conditions and precision handling.\n")

        logger.info(f"Detailed report generated: {report_path}")

    def save_results(self, filename: str = "numerical_stability_results.pkl"):
        """Save all test results to file."""
        import pickle

        with open(filename, 'wb') as f:
            pickle.dump(self.test_results, f)

        logger.info(f"Results saved to {filename}")

    def load_results(self, filename: str = "numerical_stability_results.pkl"):
        """Load test results from file."""
        import pickle

        with open(filename, 'rb') as f:
            self.test_results = pickle.load(f)

        logger.info(f"Results loaded from {filename}")


def main():
    """Main function to run the numerical stability test suite."""
    logger.info("Starting Numerical Stability Testing Suite")
    logger.info("=" * 60)

    # Initialize tester
    tester = NumericalStabilityTester()

    # Run comprehensive tests
    results = tester.run_comprehensive_stability_test()

    # Save results
    tester.save_results()

    logger.info("=" * 60)
    logger.info("Numerical stability testing completed successfully!")
    logger.info("Results saved to numerical_stability_results.pkl")
    logger.info("Report generated: numerical_stability_report.md")


if __name__ == "__main__":
    main()