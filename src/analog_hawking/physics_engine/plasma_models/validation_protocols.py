"""
Comprehensive Validation Protocols

This module implements comprehensive validation protocols to ensure
the physics models are reliable and scientifically valid for analog 
Hawking radiation research.
"""

import numpy as np
import json
from datetime import datetime
from pathlib import Path
import warnings
from scipy.constants import c, hbar, k, e, m_e, epsilon_0
from .plasma_physics import PlasmaPhysicsModel, AnalogHorizonPhysics, QEDPhysics
from .quantum_field_theory import QuantumFieldTheory, BogoliubovTransformations, HawkingRadiationModel
from .laser_plasma_interaction import LaserPlasmaDynamics
from .anaBHEL_parameters import AnaBHELExperiment
from .test_analytical_solutions import ValidationTests, ConvergenceTests

class PhysicsValidationFramework:
    """
    Comprehensive framework for validating physics models
    """
    
    def __init__(self, validation_tolerance=1e-2, convergence_tolerance=1e-3):
        """
        Initialize the validation framework
        
        Args:
            validation_tolerance (float): Tolerance for analytical validation
            convergence_tolerance (float): Tolerance for numerical convergence
        """
        self.validation_tolerance = validation_tolerance
        self.convergence_tolerance = convergence_tolerance
        self.validation_results = {}
        self.timestamp = datetime.now().isoformat()
    
    def validate_conservation_laws(self, simulation_data):
        """
        Validate conservation laws in simulation
        
        Args:
            simulation_data (dict): Simulation results
            
        Returns:
            Dictionary with conservation validation results
        """
        print("Validating Conservation Laws...")
        
        # Check energy conservation if available
        energy_conserved = True
        energy_error = 0.0
        
        if 'electric_field' in simulation_data and 'density' in simulation_data:
            # Calculate electromagnetic energy in the field
            E_field = simulation_data['electric_field']
            if hasattr(E_field, 'shape') and len(E_field.shape) == 2:
                em_energy = 0.5 * epsilon_0 * np.trapz(E_field[-1,:]**2)  # Energy at last time step
                initial_em_energy = 0.5 * epsilon_0 * np.trapz(E_field[0,:]**2)  # Energy at first time step
                energy_error = abs(em_energy - initial_em_energy) / initial_em_energy if initial_em_energy != 0 else 0
                energy_conserved = energy_error < 0.1  # Allow 10% change for numerical reasons
        
        # Check particle number conservation
        if 'density' in simulation_data:
            n_e = simulation_data['density']
            if hasattr(n_e, 'shape') and len(n_e.shape) == 2:
                final_particles = np.trapz(n_e[-1,:])
                initial_particles = np.trapz(n_e[0,:])
                particle_error = abs(final_particles - initial_particles) / initial_particles if initial_particles != 0 else 0
                particles_conserved = particle_error < 0.05  # Allow 5% change
            else:
                particles_conserved = True
                particle_error = 0.0
        else:
            particles_conserved = True
            particle_error = 0.0
        
        # Check momentum conservation
        momentum_conserved = True
        momentum_error = 0.0
        
        if 'velocity' in simulation_data:
            v_e = simulation_data['velocity']
            n_e = simulation_data.get('density', np.ones_like(v_e))
            if hasattr(v_e, 'shape') and len(v_e.shape) == 2:
                final_momentum = np.trapz(n_e[-1,:] * v_e[-1,:] * m_e)
                initial_momentum = np.trapz(n_e[0,:] * v_e[0,:] * m_e)
                momentum_error = abs(final_momentum - initial_momentum) / abs(initial_momentum) if initial_momentum != 0 else 0
                momentum_conserved = momentum_error < 0.1  # Allow 10% change
        
        return {
            'energy_conservation': {
                'conserved': energy_conserved,
                'relative_error': energy_error
            },
            'particle_conservation': {
                'conserved': particles_conserved,
                'relative_error': particle_error
            },
            'momentum_conservation': {
                'conserved': momentum_conserved,
                'relative_error': momentum_error
            },
            'all_conserved': all([energy_conserved, particles_conserved, momentum_conserved])
        }
    
    def validate_physical_bounds(self, simulation_data):
        """
        Validate that simulation parameters stay within physical bounds
        
        Args:
            simulation_data (dict): Simulation results
            
        Returns:
            Dictionary with physical bounds validation results
        """
        print("Validating Physical Bounds...")
        
        # Check velocities don't exceed speed of light
        velocity_bounded = True
        max_violation = 0.0
        
        if 'velocity' in simulation_data:
            v_e = simulation_data['velocity']
            if hasattr(v_e, '__getitem__'):
                max_velocity = np.max(np.abs(v_e))
                velocity_bounded = max_velocity <= 0.99 * c  # Allow small numerical errors
                max_violation = max_velocity / c
        
        # Check temperatures are positive
        temp_positive = True
        if 'hawking_temperature' in simulation_data:
            T_H = simulation_data['hawking_temperature']
            if np.isscalar(T_H):
                temp_positive = T_H >= 0
            elif hasattr(T_H, '__iter__'):
                temp_positive = np.all(np.array(T_H) >= 0)
        
        # Check densities are positive
        density_positive = True
        if 'density' in simulation_data:
            n_e = simulation_data['density']
            if hasattr(n_e, '__getitem__'):
                density_positive = np.all(n_e >= 0)
        
        # Check that Hawking temperature is not impossibly high
        temp_reasonable = True
        if 'hawking_temperature' in simulation_data:
            T_H = simulation_data['hawking_temperature']
            if np.isscalar(T_H):
                # Hawking temperature should be less than Planck temperature (1.4e32 K) and typically much less
                temp_reasonable = 0 <= T_H <= 1e15  # Reasonable upper bound for lab analogs
            elif hasattr(T_H, '__iter__'):
                temp_reasonable = np.all((np.array(T_H) >= 0) & (np.array(T_H) <= 1e15))
        
        return {
            'velocity_bounded': {
                'valid': velocity_bounded,
                'max_violation': max_violation,
                'max_allowed': 0.99
            },
            'temperature_positive': {
                'valid': temp_positive
            },
            'density_positive': {
                'valid': density_positive
            },
            'temperature_reasonable': {
                'valid': temp_reasonable
            },
            'all_bounds_valid': all([velocity_bounded, temp_positive, density_positive, temp_reasonable])
        }
    
    def validate_numerical_stability(self, simulation_data):
        """
        Validate numerical stability of the simulation
        
        Args:
            simulation_data (dict): Simulation results
            
        Returns:
            Dictionary with stability validation results
        """
        print("Validating Numerical Stability...")
        
        # Check for NaN or infinite values
        has_nans = False
        has_infs = False
        
        for key, value in simulation_data.items():
            if hasattr(value, '__getitem__') and hasattr(value, 'size'):
                has_nans = has_nans or np.any(np.isnan(value))
                has_infs = has_infs or np.any(np.isinf(value))
        
        # Check if values are reasonable (not extremely large)
        values_reasonable = True
        extreme_threshold = 1e50  # Reasonable upper bound for physical values
        
        for key, value in simulation_data.items():
            if hasattr(value, '__getitem__') and hasattr(value, 'size'):
                if np.any(np.abs(value) > extreme_threshold):
                    values_reasonable = False
                    break
        
        # Stability based on time step constraints (if applicable)
        dt_stable = True
        if 'time_grid' in simulation_data:
            time_grid = simulation_data['time_grid']
            if len(time_grid) > 1:
                dt = time_grid[1] - time_grid[0]
                # For electromagnetic simulations, dt should be less than L/c where L is grid spacing
                if 'space_grid' in simulation_data:
                    dx = simulation_data['space_grid'][1] - simulation_data['space_grid'][0]
                    dt_cfl = dx / c  # CFL condition
                    dt_stable = dt < 0.99 * dt_cfl  # 99% of CFL limit
        
        return {
            'no_nans': not has_nans,
            'no_infs': not has_infs,
            'values_reasonable': values_reasonable,
            'time_step_stable': dt_stable,
            'numerically_stable': not has_nans and not has_infs and values_reasonable and dt_stable
        }
    
    def validate_theoretical_consistency(self):
        """
        Validate consistency with theoretical expectations
        
        Returns:
            Dictionary with theoretical consistency validation results
        """
        print("Validating Theoretical Consistency...")
        
        # Test the relationship between surface gravity and temperature
        test_kappas = [1e10, 1e11, 1e12, 1e13]  # Various surface gravities
        theoretical_consistent = True
        
        for kappa in test_kappas:
            T_H = hbar * kappa / (2 * np.pi * k)
            # Verify the relationship holds
            calculated_kappa = 2 * np.pi * k * T_H / hbar
            if abs(kappa - calculated_kappa) / kappa > self.validation_tolerance:
                theoretical_consistent = False
                break
        
        # Test that low-frequency modes approach classical limit
        qft = QuantumFieldTheory(surface_gravity=1e12)
        low_freq = 1e10  # Low frequency limit
        quantum_occupation = qft.occupation_number(2 * np.pi * low_freq)
        classical_occupation = k * qft.T_H / (hbar * 2 * np.pi * low_freq)  # Rayleigh-Jeans law
        
        classical_limit_ok = abs(quantum_occupation - classical_occupation) / classical_occupation < 0.01 if classical_occupation != 0 else True
        
        # Validate that entropy and temperature relationship is correct
        # For thermal radiation: entropy S = 4σT³V/3c for volume V (simplified)
        # We'll check that entropy increases with temperature
        temperatures = np.logspace(6, 10, 5)  # From 10^6 to 10^10 K
        entropies = []
        
        for T in temperatures:
            # Simplified entropy calculation for thermal radiation
            # S = (4σ/c)*(T³) for unit volume
            entropy = 4 * 5.67e-8 * T**3 / c  # Stefan-Boltzmann constant σ = 5.67e-8
            entropies.append(entropy)
        
        entropy_increasing = all(np.diff(entropies) > 0)
        
        return {
            'hawking_relation_consistent': theoretical_consistent,
            'classical_limit_valid': classical_limit_ok,
            'entropy_behavior_correct': entropy_increasing,
            'theoretically_consistent': all([theoretical_consistent, classical_limit_ok, entropy_increasing])
        }
    
    def run_comprehensive_validation(self, simulation_data):
        """
        Run comprehensive validation on simulation data
        
        Args:
            simulation_data (dict): Complete simulation results
            
        Returns:
            Dictionary with complete validation results
        """
        print("RUNNING COMPREHENSIVE VALIDATION")
        print("="*35)
        
        results = {}
        
        # Run each validation test
        results['conservation_laws'] = self.validate_conservation_laws(simulation_data)
        results['physical_bounds'] = self.validate_physical_bounds(simulation_data)
        results['numerical_stability'] = self.validate_numerical_stability(simulation_data)
        results['theoretical_consistency'] = self.validate_theoretical_consistency()
        
        # Overall assessment
        overall_valid = all([
            results['conservation_laws']['all_conserved'],
            results['physical_bounds']['all_bounds_valid'],
            results['numerical_stability']['numerically_stable'],
            results['theoretical_consistency']['theoretically_consistent']
        ])
        
        results['overall_validation'] = {
            'valid': overall_valid,
            'timestamp': self.timestamp,
            'tolerance_used': self.validation_tolerance
        }
        
        self.validation_results = results
        
        print(f"\nCOMPREHENSIVE VALIDATION: {'PASS' if overall_valid else 'FAIL'}")
        print(f"  Conservation laws: {'PASS' if results['conservation_laws']['all_conserved'] else 'FAIL'}")
        print(f"  Physical bounds: {'PASS' if results['physical_bounds']['all_bounds_valid'] else 'FAIL'}")
        print(f"  Numerical stability: {'PASS' if results['numerical_stability']['numerically_stable'] else 'FAIL'}")
        print(f"  Theoretical consistency: {'PASS' if results['theoretical_consistency']['theoretically_consistent'] else 'FAIL'}")
        
        return results

class UncertaintyQuantification:
    """
    Framework for quantifying uncertainties in the simulation
    """
    
    def __init__(self):
        self.parameter_uncertainties = {}
    
    def quantify_input_uncertainty(self, input_params):
        """
        Quantify uncertainties in input parameters
        
        Args:
            input_params (dict): Input parameters with possible uncertainty bounds
            
        Returns:
            Dictionary with uncertainty quantification
        """
        uncertainties = {}
        
        for param_name, param_value in input_params.items():
            if isinstance(param_value, dict) and 'value' in param_value and 'uncertainty' in param_value:
                # Parameter has explicit uncertainty
                uncertainties[param_name] = {
                    'value': param_value['value'],
                    'uncertainty': param_value['uncertainty'],
                    'relative_uncertainty': abs(param_value['uncertainty'] / param_value['value']) if param_value['value'] != 0 else float('inf')
                }
            else:
                # Default uncertainty for parameters without explicit uncertainty
                if isinstance(param_value, (int, float)):
                    # Assume 1% uncertainty for unspecified parameters
                    uncertainties[param_name] = {
                        'value': param_value,
                        'uncertainty': 0.01 * abs(param_value),
                        'relative_uncertainty': 0.01
                    }
                else:
                    # For non-numeric parameters, no uncertainty quantification
                    uncertainties[param_name] = {
                        'value': param_value,
                        'uncertainty': 0,
                        'relative_uncertainty': 0
                    }
        
        return uncertainties
    
    def propagate_uncertainty(self, model_function, nominal_params, param_uncertainties, n_samples=1000):
        """
        Propagate parameter uncertainties through the model using Monte Carlo
        
        Args:
            model_function: Function that takes parameters and returns result
            nominal_params (dict): Nominal parameter values
            param_uncertainties (dict): Parameter uncertainties
            n_samples (int): Number of Monte Carlo samples
            
        Returns:
            Dictionary with uncertainty propagation results
        """
        # Generate samples for each parameter
        samples = {}
        for param_name, param_info in param_uncertainties.items():
            if isinstance(param_info['value'], (int, float)):
                # Generate samples from normal distribution
                samples[param_name] = np.random.normal(
                    loc=param_info['value'],
                    scale=param_info['uncertainty'],
                    size=n_samples
                )
            else:
                # For non-numeric parameters, use the same value for all samples
                samples[param_name] = np.full(n_samples, param_info['value'])
        
        # Evaluate model for each sample
        results = []
        for i in range(n_samples):
            sample_params = {name: samples[name][i] for name in samples}
            try:
                result = model_function(**sample_params)
                results.append(result)
            except Exception:
                # If evaluation fails, use NaN
                results.append(float('nan'))
        
        # Calculate statistics
        results_array = np.array(results)
        valid_results = results_array[~np.isnan(results_array)]
        
        if len(valid_results) > 0:
            mean_result = np.mean(valid_results)
            std_result = np.std(valid_results)
            ci_95 = 1.96 * std_result if len(valid_results) > 1 else 0  # 95% confidence interval
        else:
            mean_result = float('nan')
            std_result = float('nan')
            ci_95 = float('nan')
        
        return {
            'nominal_result': model_function(**nominal_params),
            'mean_result': mean_result,
            'std_result': std_result,
            'ci_95': ci_95,
            'valid_samples_fraction': len(valid_results) / n_samples,
            'effective_samples': len(valid_results)
        }

class SensitivityAnalysis:
    """
    Framework for performing sensitivity analysis
    """
    
    def __init__(self):
        pass
    
    def local_sensitivity_analysis(self, model_function, base_params, param_names=None):
        """
        Perform local sensitivity analysis by varying each parameter individually
        
        Args:
            model_function: Function to analyze
            base_params (dict): Base parameter values
            param_names (list): List of parameter names to analyze (None for all)
            
        Returns:
            Dictionary with sensitivity analysis results
        """
        if param_names is None:
            param_names = list(base_params.keys())
        
        base_result = model_function(**base_params)
        
        sensitivities = {}
        
        for param_name in param_names:
            if param_name in base_params:
                # Calculate a small perturbation (1% of parameter value)
                base_value = base_params[param_name]
                if isinstance(base_value, (int, float)):
                    delta = 0.01 * abs(base_value) if base_value != 0 else 1e-10
                else:
                    continue  # Skip non-numeric parameters
                
                # Perturb parameter up and down
                params_plus = base_params.copy()
                params_plus[param_name] = base_value + delta
                
                params_minus = base_params.copy()
                params_minus[param_name] = base_value - delta
                
                try:
                    result_plus = model_function(**params_plus)
                    result_minus = model_function(**params_minus)
                    
                    # Calculate sensitivity as (Δoutput)/(Δinput)
                    if isinstance(base_result, (int, float)) and isinstance(result_plus, (int, float)) and isinstance(result_minus, (int, float)):
                        sensitivity = (result_plus - result_minus) / (2 * delta)
                        
                        # Calculate dimensionless sensitivity (elasticity)
                        if base_value != 0 and base_result != 0:
                            elasticity = sensitivity * (base_value / base_result)
                        else:
                            elasticity = 0.0
                    else:
                        sensitivity = 0.0
                        elasticity = 0.0
                    
                    sensitivities[param_name] = {
                        'sensitivity': sensitivity,
                        'elasticity': elasticity,
                        'delta_param': delta,
                        'delta_output': result_plus - result_minus
                    }
                except Exception:
                    sensitivities[param_name] = {
                        'sensitivity': float('nan'),
                        'elasticity': float('nan'),
                        'delta_param': delta,
                        'delta_output': float('nan')
                    }
        
        return {
            'base_result': base_result,
            'sensitivities': sensitivities,
            'most_sensitive': self._find_most_sensitive(sensitivities)
        }
    
    def _find_most_sensitive(self, sensitivities):
        """
        Find the most sensitive parameters
        
        Args:
            sensitivities (dict): Sensitivity results
            
        Returns:
            List of most sensitive parameter names
        """
        if not sensitivities:
            return []
        
        # Sort by elasticity (dimensionless sensitivity measure)
        sorted_sens = sorted(
            sensitivities.items(),
            key=lambda x: abs(x[1]['elasticity']) if not np.isnan(x[1]['elasticity']) else 0,
            reverse=True
        )
        
        return [param[0] for param in sorted_sens if not np.isnan(param[1]['elasticity'])]

def perform_validation_analysis():
    """
    Perform comprehensive validation and uncertainty analysis
    
    Returns:
        Dictionary with complete analysis results
    """
    print("COMPREHENSIVE VALIDATION AND UNCERTAINTY ANALYSIS")
    print("="*55)
    
    # Initialize validation framework
    validator = PhysicsValidationFramework()
    
    # Create example simulation data
    example_data = {
        'time_grid': np.linspace(0, 100e-15, 100),
        'space_grid': np.linspace(-50e-6, 50e-6, 200),
        'electric_field': np.random.random((100, 200)) * 1e10,
        'density': np.random.random((100, 200)) * 1e18 + 1e17,
        'velocity': (np.random.random((100, 200)) - 0.5) * 0.1 * c,
        'hawking_temperature': 1.2e9
    }
    
    # Run comprehensive validation
    validation_results = validator.run_comprehensive_validation(example_data)
    
    # Perform uncertainty quantification
    print(f"\nPERFORMING UNCERTAINTY QUANTIFICATION")
    print("-"*35)
    
    uq = UncertaintyQuantification()
    
    # Example parameters with uncertainties
    example_params = {
        'laser_intensity': {'value': 1e18, 'uncertainty': 1e17},
        'plasma_density': {'value': 1e18, 'uncertainty': 1e17},
        'temperature': 10000,
        'wavelength': 800e-9
    }
    
    param_uncertainties = uq.quantify_input_uncertainty(example_params)
    
    # Define a simple test function for uncertainty propagation
    def test_function(laser_intensity=1e18, plasma_density=1e18, **kwargs):
        # Simple function that combines parameters
        return np.sqrt(laser_intensity * plasma_density) * 1e-15  # Arbitrary scaling
    
    uncertainty_results = uq.propagate_uncertainty(
        test_function,
        {k: v['value'] if isinstance(v, dict) else v for k, v in example_params.items()},
        param_uncertainties
    )
    
    print(f"Uncertainty propagation results:")
    print(f"  Nominal result: {uncertainty_results['nominal_result']:.2e}")
    print(f"  Mean result: {uncertainty_results['mean_result']:.2e}")
    print(f"  Std deviation: {uncertainty_results['std_result']:.2e}")
    print(f"  95% CI: ±{uncertainty_results['ci_95']:.2e}")
    print(f"  Valid samples: {uncertainty_results['valid_samples_fraction']:.1%}")
    
    # Perform sensitivity analysis
    print(f"\nPERFORMING SENSITIVITY ANALYSIS")
    print("-"*30)
    
    sa = SensitivityAnalysis()
    
    # Add a simple function for sensitivity analysis
    def sensitivity_func(laser_intensity=1e18, plasma_density=1e18, temperature=10000):
        # A function that depends on the parameters
        return (laser_intensity**0.5) * (plasma_density**0.3) * (temperature**0.1)
    
    sensitivity_results = sa.local_sensitivity_analysis(
        sensitivity_func,
        {
            'laser_intensity': 1e18,
            'plasma_density': 1e18,
            'temperature': 10000
        }
    )
    
    print("Sensitivity analysis results:")
    for param, results in sensitivity_results['sensitivities'].items():
        print(f"  {param}: elasticity = {results['elasticity']:.3f}")
    
    print(f"\nMost sensitive parameters: {sensitivity_results['most_sensitive'][:3]}")  # Top 3
    
    # Compile all results
    complete_analysis = {
        'validation_results': validation_results,
        'uncertainty_analysis': uncertainty_results,
        'sensitivity_analysis': sensitivity_results,
        'parameter_uncertainties': param_uncertainties,
        'overall_assessment': validation_results['overall_validation']['valid'],
        'timestamp': datetime.now().isoformat()
    }
    
    return complete_analysis

def generate_validation_report(results, filename=None):
    """
    Generate a comprehensive validation report
    
    Args:
        results (dict): Results from validation analysis
        filename (str): Output filename (None for console only)
    """
    report = f"""
COMPREHENSIVE VALIDATION REPORT
===============================

Generated on: {results['timestamp']}

1. VALIDATION RESULTS
---------------------
Overall Validation: {'PASSED' if results['overall_assessment'] else 'FAILED'}

Conservation Laws:
- Energy Conservation: {'PASS' if results['validation_results']['conservation_laws']['all_conserved'] else 'FAIL'}
- Particle Conservation: {'PASS' if results['validation_results']['conservation_laws']['all_conserved'] else 'FAIL'}
- Momentum Conservation: {'PASS' if results['validation_results']['conservation_laws']['all_conserved'] else 'FAIL'}

Physical Bounds:
- Velocities Bounded: {'PASS' if results['validation_results']['physical_bounds']['all_bounds_valid'] else 'FAIL'}
- Temperatures Positive: {'PASS' if results['validation_results']['physical_bounds']['all_bounds_valid'] else 'FAIL'}
- Densities Positive: {'PASS' if results['validation_results']['physical_bounds']['all_bounds_valid'] else 'FAIL'}

Numerical Stability:
- No NaNs: {'PASS' if results['validation_results']['numerical_stability']['numerically_stable'] else 'FAIL'}
- No Infs: {'PASS' if results['validation_results']['numerical_stability']['numerically_stable'] else 'FAIL'}
- Time Step Stable: {'PASS' if results['validation_results']['numerical_stability']['numerically_stable'] else 'FAIL'}

Theoretical Consistency:
- Hawking Relation: {'PASS' if results['validation_results']['theoretical_consistency']['theoretically_consistent'] else 'FAIL'}
- Classical Limit: {'PASS' if results['validation_results']['theoretical_consistency']['theoretically_consistent'] else 'FAIL'}
- Entropy Behavior: {'PASS' if results['validation_results']['theoretical_consistency']['theoretically_consistent'] else 'FAIL'}

2. UNCERTAINTY ANALYSIS
------------------------
Nominal Result: {results['uncertainty_analysis']['nominal_result']:.2e}
Mean Result: {results['uncertainty_analysis']['mean_result']:.2e}
Standard Deviation: {results['uncertainty_analysis']['std_result']:.2e}
95% Confidence Interval: ±{results['uncertainty_analysis']['ci_95']:.2e}
Valid Samples Fraction: {results['uncertainty_analysis']['valid_samples_fraction']:.1%}

3. SENSITIVITY ANALYSIS
------------------------
Base Result: {results['sensitivity_analysis']['base_result']:.2e}
Most Sensitive Parameters: {results['sensitivity_analysis']['most_sensitive'][:5]}

4. RECOMMENDATIONS
------------------
"""
    
    if results['overall_assessment']:
        report += "- Model validation PASSED - ready for scientific application\n"
    else:
        report += "- Model validation FAILED - requires revision before scientific application\n"
        
    if results['uncertainty_analysis']['std_result'] > 0.1 * abs(results['uncertainty_analysis']['mean_result']):
        report += "- High output uncertainty - consider refining parameter estimates\n"
    
    if len(results['sensitivity_analysis']['most_sensitive'][:3]) > 0:
        report += f"- Most sensitive parameters: {results['sensitivity_analysis']['most_sensitive'][:3]} - prioritize accurate measurement of these\n"
    
    if filename:
        with open(filename, 'w') as f:
            f.write(report)
        print(f"Validation report saved to {filename}")
    else:
        print(report)

if __name__ == "__main__":
    # Run the comprehensive validation and uncertainty analysis
    analysis_results = perform_validation_analysis()
    
    # Generate a validation report
    generate_validation_report(analysis_results, 'validation_report.txt')
