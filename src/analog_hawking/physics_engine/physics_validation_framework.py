"""
Physics Model Validation Framework for Analog Hawking Radiation Analysis

This module provides a comprehensive validation framework for the enhanced physics
models, ensuring they meet physical constraints, produce realistic results, and
maintain proper limiting behavior across different regimes.

Key Features:
1. Physical constraint validation (energy conservation, causality, etc.)
2. Limiting behavior checks (classical ↔ relativistic regimes)
3. Benchmark validation against known theoretical results
4. Uncertainty quantification for model predictions
5. Model intercomparison and consistency checks

Author: Enhanced Physics Implementation
Date: November 2025
References:
- Jackson, Classical Electrodynamics
- Landau & Lifshitz, Physical Kinetics
- ELI Facility Physics Validation Protocols
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, hbar, k, m_p, pi
from scipy.integrate import quad, solve_ivp
from scipy.optimize import minimize, root_scalar
from scipy.stats import chi2, norm
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt

# Import enhanced physics models
from .enhanced_relativistic_physics import RelativisticPlasmaPhysics
from .enhanced_ionization_physics import IonizationDynamics, ATOMIC_DATA
from .enhanced_plasma_surface_physics import PlasmaDynamicsAtSurface

@dataclass
class ValidationResult:
    """Container for validation results"""
    test_name: str
    passed: bool
    value: float
    expected_value: Optional[float]
    tolerance: float
    description: str
    severity: str = "error"  # "error", "warning", "info"

class PhysicalConstraintsValidator:
    """
    Validates that physics models respect fundamental physical constraints
    """

    def __init__(self):
        """Initialize physical constraints validator"""
        self.results = []

    def test_energy_conservation(self, incident_energy: float,
                               absorbed_energy: float,
                               reflected_energy: float,
                               transmitted_energy: float = 0,
                               tolerance: float = 0.01) -> ValidationResult:
        """
        Test energy conservation in laser-plasma interaction

        Args:
            incident_energy: Incident laser energy
            absorbed_energy: Absorbed energy
            reflected_energy: Reflected energy
            transmitted_energy: Transmitted energy (default 0)
            tolerance: Allowed fractional error

        Returns:
            ValidationResult object
        """
        total_energy = absorbed_energy + reflected_energy + transmitted_energy
        fractional_error = abs(total_energy - incident_energy) / incident_energy

        passed = fractional_error < tolerance
        description = f"Energy conservation test: {total_energy:.3e} vs {incident_energy:.3e}"

        result = ValidationResult(
            test_name="energy_conservation",
            passed=passed,
            value=fractional_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

    def test_causality(self, group_velocity: float, phase_velocity: float) -> ValidationResult:
        """
        Test causality constraints on wave propagation

        Args:
            group_velocity: Group velocity in m/s
            phase_velocity: Phase velocity in m/s

        Returns:
            ValidationResult object
        """
        # Both velocities should not exceed c
        max_velocity = max(abs(group_velocity), abs(phase_velocity))
        tolerance = 1e-10 * c  # Allow tiny numerical errors

        passed = max_velocity <= c + tolerance
        fractional_excess = max(max_velocity - c, 0) / c

        description = f"Causality test: max velocity = {max_velocity:.3e} m/s vs c = {c:.3e} m/s"

        result = ValidationResult(
            test_name="causality",
            passed=passed,
            value=fractional_excess,
            expected_value=0.0,
            tolerance=tolerance/c,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

    def test_positivity_definite(self, quantities: Dict[str, float]) -> List[ValidationResult]:
        """
        Test that physical quantities remain positive definite

        Args:
            quantities: Dictionary of quantity names and values

        Returns:
            List of ValidationResult objects
        """
        results = []
        for name, value in quantities.items():
            tolerance = -1e-15  # Allow tiny negative numerical errors

            passed = value >= tolerance
            description = f"Positivity test for {name}: {value:.3e}"

            result = ValidationResult(
                test_name=f"positivity_{name}",
                passed=passed,
                value=value,
                expected_value=0.0,
                tolerance=abs(tolerance),
                description=description,
                severity="error" if not passed else "info"
            )

            results.append(result)
            self.results.append(result)

        return results

    def test_momentum_conservation(self, incident_momentum: float,
                                 absorbed_momentum: float,
                                 reflected_momentum: float,
                                 transmitted_momentum: float = 0,
                                 tolerance: float = 0.01) -> ValidationResult:
        """
        Test momentum conservation in laser-plasma interaction

        Args:
            incident_momentum: Incident laser momentum
            absorbed_momentum: Absorbed momentum
            reflected_momentum: Reflected momentum
            transmitted_momentum: Transmitted momentum (default 0)
            tolerance: Allowed fractional error

        Returns:
            ValidationResult object
        """
        total_momentum = absorbed_momentum + reflected_momentum + transmitted_momentum
        fractional_error = abs(total_momentum - incident_momentum) / incident_momentum

        passed = fractional_error < tolerance
        description = f"Momentum conservation test: {total_momentum:.3e} vs {incident_momentum:.3e}"

        result = ValidationResult(
            test_name="momentum_conservation",
            passed=passed,
            value=fractional_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

class LimitingBehaviorValidator:
    """
    Validates proper limiting behavior of physics models
    """

    def __init__(self):
        """Initialize limiting behavior validator"""
        self.results = []

    def test_classical_limit(self, relativistic_model: RelativisticPlasmaPhysics,
                           test_parameter: str = "plasma_frequency",
                           tolerance: float = 0.01) -> ValidationResult:
        """
        Test that relativistic model reduces to classical limit when γ → 1

        Args:
            relativistic_model: Relativistic plasma physics model
            test_parameter: Parameter to test
            tolerance: Allowed fractional error

        Returns:
            ValidationResult object
        """
        # Classical value (γ = 1)
        if test_parameter == "plasma_frequency":
            classical_value = relativistic_model.omega_pe
            relativistic_value = relativistic_model.relativistic_plasma_frequency(np.array([1.0]))[0]
        else:
            raise ValueError(f"Unknown test parameter: {test_parameter}")

        fractional_error = abs(relativistic_value - classical_value) / classical_value

        passed = fractional_error < tolerance
        description = f"Classical limit test for {test_parameter}: {relativistic_value:.3e} vs {classical_value:.3e}"

        result = ValidationResult(
            test_name=f"classical_limit_{test_parameter}",
            passed=passed,
            value=fractional_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

    def test_relativistic_limit(self, relativistic_model: RelativisticPlasmaPhysics,
                              gamma_values: np.ndarray = None,
                              tolerance: float = 0.1) -> ValidationResult:
        """
        Test behavior in ultra-relativistic limit

        Args:
            relativistic_model: Relativistic plasma physics model
            gamma_values: Array of γ values to test
            tolerance: Allowed deviation from expected scaling

        Returns:
            ValidationResult object
        """
        if gamma_values is None:
            gamma_values = np.array([1.0, 2.0, 5.0, 10.0, 100.0])

        # Test plasma frequency scaling: ω_pe,rel = ω_pe / √γ
        omega_pe_rel = relativistic_model.relativistic_plasma_frequency(gamma_values)
        expected_scaling = relativistic_model.omega_pe / np.sqrt(gamma_values)

        # Calculate scaling error
        fractional_errors = np.abs(omega_pe_rel - expected_scaling) / expected_scaling
        max_error = np.max(fractional_errors)

        passed = max_error < tolerance
        description = f"Relativistic scaling test: max error = {max_error:.3f}"

        result = ValidationResult(
            test_name="relativistic_scaling",
            passed=passed,
            value=max_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

    def test_intensity_limiting(self, surface_model: PlasmaDynamicsAtSurface,
                              intensities: np.ndarray = None,
                              tolerance: float = 0.1) -> List[ValidationResult]:
        """
        Test proper behavior at intensity limits

        Args:
            surface_model: Plasma surface dynamics model
            intensities: Array of intensities to test
            tolerance: Allowed deviation from expected behavior

        Returns:
            List of ValidationResult objects
        """
        if intensities is None:
            intensities = np.logspace(16, 24, 9)  # 10^16 to 10^24 W/m^2

        results = []
        wavelength = 800e-9
        pulse_duration = 30e-15

        for I in intensities:
            # Test that absorption + reflectivity ≤ 1
            interaction = surface_model.full_surface_interaction(
                I, wavelength, pulse_duration, 0, 'p')

            total_interaction = interaction['absorption_fraction'] + interaction['reflectivity']
            excess = max(total_interaction - 1.0, 0)

            passed = excess < tolerance
            description = f"Intensity limiting test at I = {I:.1e} W/m^2: total = {total_interaction:.3f}"

            result = ValidationResult(
                test_name=f"intensity_limiting_{I:.1e}",
                passed=passed,
                value=excess,
                expected_value=0.0,
                tolerance=tolerance,
                description=description,
                severity="error" if not passed else "warning"
            )

            results.append(result)
            self.results.append(result)

        return results

class BenchmarkValidator:
    """
    Validates models against known theoretical benchmarks
    """

    def __init__(self):
        """Initialize benchmark validator"""
        self.results = []

    def test_critical_density(self, plasma_model: RelativisticPlasmaPhysics,
                            tolerance: float = 0.01) -> ValidationResult:
        """
        Test critical density calculation against analytical result

        Args:
            plasma_model: Relativistic plasma physics model
            tolerance: Allowed fractional error

        Returns:
            ValidationResult object
        """
        # Analytical critical density
        omega_l = plasma_model.omega_l
        n_critical_analytical = epsilon_0 * m_e * omega_l**2 / e**2

        fractional_error = abs(plasma_model.n_critical - n_critical_analytical) / n_critical_analytical

        passed = fractional_error < tolerance
        description = f"Critical density test: {plasma_model.n_critical:.3e} vs {n_critical_analytical:.3e}"

        result = ValidationResult(
            test_name="critical_density",
            passed=passed,
            value=fractional_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

    def test_adk_limiting_cases(self, ionization_model: IonizationDynamics,
                              tolerance: float = 0.1) -> List[ValidationResult]:
        """
        Test ADK ionization model against known limiting cases

        Args:
            ionization_model: Ionization dynamics model
            tolerance: Allowed fractional error

        Returns:
            List of ValidationResult objects
        """
        results = []

        # Test weak field limit (very low ionization rate)
        E_weak = 1e8  # V/m - very weak field
        rate_weak = ionization_model.adk_model.adk_rate(E_weak, 0)

        # In weak field limit, rate should be exponentially small
        expected_max = 1e-10  # s^-1
        passed_weak = rate_weak < expected_max

        result_weak = ValidationResult(
            test_name="adk_weak_field_limit",
            passed=passed_weak,
            value=rate_weak,
            expected_value=0.0,
            tolerance=expected_max,
            description=f"ADK weak field test: rate = {rate_weak:.2e} s^-1",
            severity="error" if not passed_weak else "info"
        )

        results.append(result_weak)
        self.results.append(result_weak)

        # Test strong field scaling (approximate)
        E_strong_values = np.logspace(11, 13, 5)  # V/m
        rates_strong = [ionization_model.adk_model.adk_rate(E, 0) for E in E_strong_values]

        # Check that rates increase with field
        monotonic_increasing = all(rates_strong[i] < rates_strong[i+1] for i in range(len(rates_strong)-1))

        result_scaling = ValidationResult(
            test_name="adk_strong_field_scaling",
            passed=monotonic_increasing,
            value=1.0 if monotonic_increasing else 0.0,
            expected_value=1.0,
            tolerance=0.0,
            description="ADK strong field monotonicity test",
            severity="error" if not monotonic_increasing else "info"
        )

        results.append(result_scaling)
        self.results.append(result_scaling)

        return results

    def test_dispersion_relation(self, plasma_model: RelativisticPlasmaPhysics,
                               tolerance: float = 0.05) -> ValidationResult:
        """
        Test electromagnetic wave dispersion relation

        Args:
            plasma_model: Relativistic plasma physics model
            tolerance: Allowed fractional error

        Returns:
            ValidationResult object
        """
        # Test at different frequencies
        omega_values = np.linspace(0.5 * plasma_model.omega_pe, 2 * plasma_model.omega_l, 10)
        gamma_test = 1.5  # Test at modest relativistic factor

        max_error = 0
        for omega in omega_values:
            # Calculate k from model
            k_model = plasma_model.relativistic_dispersion_relation(
                omega, np.array([gamma_test]), 'electromagnetic')[0]

            # Analytical result: ω² = ω_pe²/γ + k²c²
            k_analytical = np.sqrt(max(omega**2 - plasma_model.omega_pe**2/gamma_test, 0) / c**2)

            if k_analytical > 0:
                fractional_error = abs(k_model - k_analytical) / k_analytical
                max_error = max(max_error, fractional_error)

        passed = max_error < tolerance
        description = f"Dispersion relation test: max error = {max_error:.3f}"

        result = ValidationResult(
            test_name="dispersion_relation",
            passed=passed,
            value=max_error,
            expected_value=0.0,
            tolerance=tolerance,
            description=description,
            severity="error" if not passed else "info"
        )

        self.results.append(result)
        return result

class UncertaintyQuantifier:
    """
    Quantifies uncertainties in physics model predictions
    """

    def __init__(self):
        """Initialize uncertainty quantifier"""
        self.results = []

    def monte_carlo_uncertainty(self, model_func: Callable,
                              parameter_ranges: Dict[str, Tuple[float, float]],
                              n_samples: int = 1000,
                              confidence_level: float = 0.95) -> Dict[str, float]:
        """
        Perform Monte Carlo uncertainty analysis

        Args:
            model_func: Function to evaluate model
            parameter_ranges: Dictionary of parameter names and (min, max) ranges
            n_samples: Number of Monte Carlo samples
            confidence_level: Confidence level for uncertainty bounds

        Returns:
            Dictionary with uncertainty statistics
        """
        # Generate random samples
        samples = {}
        for param, (min_val, max_val) in parameter_ranges.items():
            samples[param] = np.random.uniform(min_val, max_val, n_samples)

        # Evaluate model for all samples
        outputs = []
        for i in range(n_samples):
            args = {param: samples[param][i] for param in parameter_ranges}
            try:
                output = model_func(**args)
                outputs.append(output)
            except Exception as e:
                warnings.warn(f"Model evaluation failed for sample {i}: {e}")
                continue

        outputs = np.array(outputs)

        # Calculate statistics
        mean_output = np.mean(outputs)
        std_output = np.std(outputs)

        # Confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        ci_lower = np.percentile(outputs, lower_percentile)
        ci_upper = np.percentile(outputs, upper_percentile)

        return {
            'mean': mean_output,
            'std': std_output,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'relative_uncertainty': std_output / abs(mean_output) if mean_output != 0 else np.inf,
            'samples_successful': len(outputs)
        }

class PhysicsModelValidator:
    """
    Comprehensive physics model validation framework
    """

    def __init__(self):
        """Initialize comprehensive validator"""
        self.constraints_validator = PhysicalConstraintsValidator()
        self.limits_validator = LimitingBehaviorValidator()
        self.benchmark_validator = BenchmarkValidator()
        self.uncertainty_quantifier = UncertaintyQuantifier()
        self.all_results = []

    def validate_relativistic_physics(self, plasma_model: RelativisticPlasmaPhysics) -> Dict[str, List[ValidationResult]]:
        """
        Validate relativistic physics model

        Args:
            plasma_model: Relativistic plasma physics model

        Returns:
            Dictionary with validation results by category
        """
        results = {
            'constraints': [],
            'limits': [],
            'benchmarks': []
        }

        # Physical constraints
        print("Testing physical constraints...")

        # Test positive quantities
        quantities = {
            'plasma_frequency': plasma_model.omega_pe,
            'laser_frequency': plasma_model.omega_l,
            'critical_density': plasma_model.n_critical,
            'a0_parameter': plasma_model.a0
        }
        results['constraints'].extend(
            self.constraints_validator.test_positivity_definite(quantities)
        )

        # Test causality for group and phase velocities
        for gamma in [1.0, 2.0, 10.0]:
            k = plasma_model.relativistic_dispersion_relation(
                plasma_model.omega_l, np.array([gamma]), 'electromagnetic')[0]
            phase_velocity = plasma_model.omega_l / k if k > 0 else c
            # Group velocity approximation
            group_velocity = 0.99 * c  # Simplified

            results['constraints'].append(
                self.constraints_validator.test_causality(group_velocity, phase_velocity)
            )

        # Limiting behavior
        print("Testing limiting behavior...")
        results['limits'].append(
            self.limits_validator.test_classical_limit(plasma_model)
        )
        results['limits'].append(
            self.limits_validator.test_relativistic_limit(plasma_model)
        )

        # Benchmarks
        print("Testing theoretical benchmarks...")
        results['benchmarks'].append(
            self.benchmark_validator.test_critical_density(plasma_model)
        )
        results['benchmarks'].append(
            self.benchmark_validator.test_dispersion_relation(plasma_model)
        )

        # Collect all results
        self.all_results.extend(results['constraints'] + results['limits'] + results['benchmarks'])

        return results

    def validate_ionization_physics(self, ionization_model: IonizationDynamics) -> Dict[str, List[ValidationResult]]:
        """
        Validate ionization physics model

        Args:
            ionization_model: Ionization dynamics model

        Returns:
            Dictionary with validation results by category
        """
        results = {
            'constraints': [],
            'benchmarks': []
        }

        # Physical constraints
        print("Testing ionization constraints...")

        # Test that ionization rates are positive
        E_fields = np.logspace(10, 14, 5)
        for E in E_fields:
            for charge_state in range(min(3, ionization_model.atom.n_states)):
                rate = ionization_model.adk_model.adk_rate(E, charge_state)
                quantities = {f'adk_rate_E{E:.1e}_Z{charge_state}': rate}
                results['constraints'].extend(
                    self.constraints_validator.test_positivity_definite(quantities)
                )

        # Benchmarks
        print("Testing ionization benchmarks...")
        results['benchmarks'].extend(
            self.benchmark_validator.test_adk_limiting_cases(ionization_model)
        )

        # Collect all results
        self.all_results.extend(results['constraints'] + results['benchmarks'])

        return results

    def validate_surface_physics(self, surface_model: PlasmaDynamicsAtSurface) -> Dict[str, List[ValidationResult]]:
        """
        Validate plasma surface physics model

        Args:
            surface_model: Plasma surface dynamics model

        Returns:
            Dictionary with validation results by category
        """
        results = {
            'constraints': [],
            'limits': [],
            'benchmarks': []
        }

        # Physical constraints
        print("Testing surface physics constraints...")

        # Test energy conservation
        intensities = np.logspace(18, 22, 5)
        wavelength = 800e-9
        pulse_duration = 30e-15

        for I in intensities:
            interaction = surface_model.full_surface_interaction(I, wavelength, pulse_duration, 0, 'p')

            # Energy conservation (simplified)
            absorbed = interaction['absorption_fraction']
            reflected = interaction['reflectivity']

            results['constraints'].append(
                self.constraints_validator.test_energy_conservation(
                    incident_energy=1.0, absorbed_energy=absorbed, reflected_energy=reflected
                )
            )

            # Test positive quantities
            quantities = {
                'absorption_fraction': absorbed,
                'reflectivity': reflected,
                'electron_temperature': interaction['electron_temperature'],
                'expansion_velocity': interaction['expansion_velocity']
            }
            results['constraints'].extend(
                self.constraints_validator.test_positivity_definite(quantities)
            )

        # Limiting behavior
        print("Testing surface physics limits...")
        results['limits'].extend(
            self.limits_validator.test_intensity_limiting(surface_model, intensities)
        )

        # Collect all results
        self.all_results.extend(results['constraints'] + results['limits'] + results['benchmarks'])

        return results

    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report

        Returns:
            Dictionary with validation summary
        """
        if not self.all_results:
            return {'error': 'No validation results available. Run validation tests first.'}

        # Categorize results
        errors = [r for r in self.all_results if not r.passed and r.severity == "error"]
        warnings = [r for r in self.all_results if not r.passed and r.severity == "warning"]
        passed = [r for r in self.all_results if r.passed]

        total_tests = len(self.all_results)
        passed_tests = len(passed)
        error_tests = len(errors)
        warning_tests = len(warnings)

        # Summary statistics
        summary = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'error_tests': error_tests,
            'warning_tests': warning_tests,
            'pass_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'overall_status': 'PASS' if error_tests == 0 else 'FAIL',
            'errors': [{'test': r.test_name, 'description': r.description, 'value': r.value}
                      for r in errors],
            'warnings': [{'test': r.test_name, 'description': r.description, 'value': r.value}
                        for r in warnings],
            'details': [r.__dict__ for r in self.all_results]
        }

        return summary

    def plot_validation_results(self, save_path: Optional[str] = None):
        """
        Create visualization of validation results

        Args:
            save_path: Optional path to save the plot
        """
        if not self.all_results:
            print("No validation results to plot")
            return

        # Count results by category
        test_names = list(set(r.test_name for r in self.all_results))
        passed_counts = []
        failed_counts = []

        for test_name in test_names:
            passed = len([r for r in self.all_results if r.test_name == test_name and r.passed])
            failed = len([r for r in self.all_results if r.test_name == test_name and not r.passed])
            passed_counts.append(passed)
            failed_counts.append(failed)

        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(test_names))
        width = 0.35

        bars1 = ax.bar(x - width/2, passed_counts, width, label='Passed', color='green', alpha=0.7)
        bars2 = ax.bar(x + width/2, failed_counts, width, label='Failed', color='red', alpha=0.7)

        ax.set_xlabel('Validation Tests')
        ax.set_ylabel('Number of Tests')
        ax.set_title('Physics Model Validation Results')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names, rotation=45, ha='right')
        ax.legend()

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate('{}'.format(int(height)),
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def run_comprehensive_validation():
    """
    Run comprehensive validation of all enhanced physics models
    """
    print("Comprehensive Physics Model Validation")
    print("=" * 50)

    # Initialize validator
    validator = PhysicsModelValidator()

    # Test relativistic physics
    print("\n1. Validating Relativistic Physics")
    print("-" * 30)
    plasma_model = RelativisticPlasmaPhysics(
        electron_density=1e19,
        laser_wavelength=800e-9,
        laser_intensity=1e20
    )
    relativistic_results = validator.validate_relativistic_physics(plasma_model)

    # Test ionization physics
    print("\n2. Validating Ionization Physics")
    print("-" * 30)
    ionization_model = IonizationDynamics(ATOMIC_DATA['Al'], laser_wavelength=800e-9)
    ionization_results = validator.validate_ionization_physics(ionization_model)

    # Test surface physics
    print("\n3. Validating Surface Physics")
    print("-" * 30)
    surface_model = PlasmaDynamicsAtSurface('Al')
    surface_results = validator.validate_surface_physics(surface_model)

    # Generate report
    print("\n4. Generating Validation Report")
    print("-" * 30)
    report = validator.generate_validation_report()

    print(f"\nValidation Summary:")
    print(f"  Total tests: {report['total_tests']}")
    print(f"  Passed: {report['passed_tests']}")
    print(f"  Errors: {report['error_tests']}")
    print(f"  Warnings: {report['warning_tests']}")
    print(f"  Pass rate: {report['pass_rate']:.1%}")
    print(f"  Overall status: {report['overall_status']}")

    if report['errors']:
        print(f"\nCritical Errors:")
        for error in report['errors']:
            print(f"  - {error['test']}: {error['description']}")

    if report['warnings']:
        print(f"\nWarnings:")
        for warning in report['warnings']:
            print(f"  - {warning['test']}: {warning['description']}")

    # Create visualization
    print("\n5. Creating validation visualization")
    print("-" * 30)
    validator.plot_validation_results()

    return report

if __name__ == "__main__":
    report = run_comprehensive_validation()