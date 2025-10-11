#!/usr/bin/env python3
"""
Einstein-Inspired Parameter Optimization for Micro-Singularity Factory

This script systematically explores parameter space to optimize Hawking radiation detection,
following Einstein's methodical approach of varying fundamental parameters to understand
the underlying physics.
"""

import numpy as np
import json
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
import warnings

# Add subdirectories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'pulse-designer'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hawking-hunter'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'horizon-builder'))

from pulse_optimizer import PulseOptimizer
from spectrum_analyzer import SpectrumAnalyzer
from blackbody_fitter import BlackbodyFitter
import pulse_utils as pu

class ParameterOptimizer:
    """
    Einstein-inspired parameter optimization for analog Hawking radiation detection.

    Systematically varies fundamental parameters to understand their impact on
    horizon formation and radiation detection confidence.
    """

    def __init__(self):
        # Base parameters (from current working setup)
        self.base_wavelength = 800e-9  # 800 nm
        self.base_energy = 1e-3  # 1 mJ
        self.base_focus = 1e-6  # 1 µm
        self.base_pressure = 1e-5  # Torr
        self.base_gas = 'H2'

        # Parameter ranges for optimization
        self.param_ranges = {
            'wavelength': (400e-9, 1200e-9),  # 400-1200 nm
            'pulse_duration': (1e-15, 100e-15),  # 1-100 fs
            'focus_diameter': (0.5e-6, 5e-6),  # 0.5-5 µm
            'pressure': (1e-6, 1e-3),  # 10^-6 to 10^-3 Torr
            'gas_types': ['H2', 'He', 'Ne', 'Ar']
        }

        # Results storage
        self.optimization_results = []

        # Initialize analyzers
        self.pulse_optimizer = None
        self.spectrum_analyzer = SpectrumAnalyzer()
        self.blackbody_fitter = BlackbodyFitter()

    def systematic_parameter_sweep(self, n_points=10):
        """
        Systematic parameter sweep across all dimensions.

        Einstein-inspired: Methodically vary each parameter while holding others constant
        to understand individual contributions to horizon formation and radiation detection.
        """
        print("=" * 80)
        print("SYSTEMATIC PARAMETER SWEEP - EINSTEIN-INSPIRED ANALYSIS")
        print("=" * 80)

        results = {}

        # 1. Wavelength sweep
        print("\n1. WAVELENGTH OPTIMIZATION")
        print("-" * 40)
        wavelength_range = np.linspace(self.param_ranges['wavelength'][0],
                                     self.param_ranges['wavelength'][1], n_points)
        wavelength_results = []

        for wavelength in wavelength_range:
            print(".1f")
            result = self._evaluate_single_configuration(
                wavelength=wavelength, pulse_duration=25e-15,  # 25 fs
                focus_diameter=self.base_focus, pressure=self.base_pressure
            )
            if result['success']:
                wavelength_results.append(result)
                self.optimization_results.append(result)

        results['wavelength_sweep'] = wavelength_results

        # 2. Pulse duration sweep
        print("\n2. PULSE DURATION OPTIMIZATION")
        print("-" * 40)
        duration_range = np.logspace(np.log10(self.param_ranges['pulse_duration'][0]),
                                   np.log10(self.param_ranges['pulse_duration'][1]), n_points)
        duration_results = []

        for duration in duration_range:
            print(".1f")
            result = self._evaluate_single_configuration(
                wavelength=self.base_wavelength, pulse_duration=duration,
                focus_diameter=self.base_focus, pressure=self.base_pressure
            )
            if result['success']:
                duration_results.append(result)
                self.optimization_results.append(result)

        results['duration_sweep'] = duration_results

        # 3. Focus diameter sweep
        print("\n3. FOCUS DIAMETER OPTIMIZATION")
        print("-" * 40)
        focus_range = np.linspace(self.param_ranges['focus_diameter'][0],
                                self.param_ranges['focus_diameter'][1], n_points)
        focus_results = []

        for focus in focus_range:
            print(".1f")
            result = self._evaluate_single_configuration(
                wavelength=self.base_wavelength, pulse_duration=25e-15,
                focus_diameter=focus, pressure=self.base_pressure
            )
            if result['success']:
                focus_results.append(result)
                self.optimization_results.append(result)

        results['focus_sweep'] = focus_results

        # 4. Pressure sweep
        print("\n4. PRESSURE OPTIMIZATION")
        print("-" * 40)
        pressure_range = np.logspace(np.log10(self.param_ranges['pressure'][0]),
                                   np.log10(self.param_ranges['pressure'][1]), n_points)
        pressure_results = []

        for pressure in pressure_range:
            print(".2e")
            result = self._evaluate_single_configuration(
                wavelength=self.base_wavelength, pulse_duration=25e-15,
                focus_diameter=self.base_focus, pressure=pressure
            )
            if result['success']:
                pressure_results.append(result)
                self.optimization_results.append(result)

        results['pressure_sweep'] = pressure_results

        # 5. Gas species comparison
        print("\n5. GAS SPECIES OPTIMIZATION")
        print("-" * 40)
        gas_results = []

        for gas in self.param_ranges['gas_types']:
            print(f"Evaluating gas: {gas}")
            result = self._evaluate_single_configuration(
                wavelength=self.base_wavelength, pulse_duration=25e-15,
                focus_diameter=self.base_focus, pressure=self.base_pressure,
                gas_type=gas
            )
            if result['success']:
                gas_results.append(result)
                self.optimization_results.append(result)

        results['gas_sweep'] = gas_results

        return results

    def global_optimization(self, max_evaluations=50):
        """
        Global optimization using evolutionary algorithms.

        Einstein-inspired: Use sophisticated optimization to find the best
        combination of parameters for maximum Hawking radiation detection confidence.
        """
        print("\n" + "=" * 80)
        print("GLOBAL PARAMETER OPTIMIZATION")
        print("=" * 80)

        # Define bounds for optimization
        bounds = [
            self.param_ranges['wavelength'],      # wavelength (m)
            self.param_ranges['pulse_duration'],  # pulse duration (s)
            (0.5e-6, 5e-6),                      # focus diameter (m)
            self.param_ranges['pressure']         # pressure (Torr)
        ]

        def objective_function(params):
            wavelength, pulse_duration, focus_diameter, pressure = params

            result = self._evaluate_single_configuration(
                wavelength=wavelength, pulse_duration=pulse_duration,
                focus_diameter=focus_diameter, pressure=pressure
            )

            if result['success']:
                # Objective: maximize confidence, minimize chi-squared
                confidence = result.get('confidence', 0)
                chi_squared = result.get('chi_squared', float('inf'))
                if chi_squared > 0:
                    fitness = confidence - 0.1 * np.log(chi_squared)
                else:
                    fitness = confidence
                return -fitness  # Negative for minimization
            else:
                return 100  # Large penalty for failure

        # Run differential evolution optimization
        result = differential_evolution(
            objective_function,
            bounds,
            maxiter=max_evaluations // 10,
            popsize=10,
            tol=0.01,
            seed=42
        )

        # Evaluate best solution
        best_params = result.x
        best_result = self._evaluate_single_configuration(
            wavelength=best_params[0], pulse_duration=best_params[1],
            focus_diameter=best_params[2], pressure=best_params[3]
        )

        optimization_result = {
            'method': 'differential_evolution',
            'best_params': {
                'wavelength': best_params[0],
                'pulse_duration': best_params[1],
                'focus_diameter': best_params[2],
                'pressure': best_params[3]
            },
            'best_result': best_result,
            'optimization_success': result.success,
            'n_evaluations': result.nfev
        }

        return optimization_result

    def _evaluate_single_configuration(self, wavelength, pulse_duration, focus_diameter,
                                     pressure, gas_type='H2'):
        """
        Evaluate a single parameter configuration.

        Returns comprehensive metrics for horizon formation and radiation detection.
        """
        try:
            # Create plasma parameters
            plasma_params = {
                'gas_type': gas_type,
                'pressure': pressure,
                'laser_intensity': 1e17,  # W/cm² (placeholder)
                'laser_duration': pulse_duration
            }

            # Initialize pulse optimizer for this configuration
            self.pulse_optimizer = PulseOptimizer(wavelength, self.base_energy, focus_diameter)

            # Optimize pulse (simplified - use fixed duration for now)
            time = np.linspace(-5*pulse_duration, 5*pulse_duration, 1000)
            # Create Gaussian pulse with specified duration
            pulse = np.exp(-time**2 / (2 * (pulse_duration/2.355)**2))
            pulse = pulse / np.max(np.abs(pulse))  # Normalize

            # Calculate electric field
            e_field = pu.calculate_e_field(pulse, wavelength, self.base_energy, focus_diameter, time)

            # Generate spectrum
            energies, spectrum = self.spectrum_analyzer.generate_spectrum(
                {'pair_density': np.random.rand(10, 10) * 1e15,  # Mock data
                 'electron_density': np.random.rand(10, 10) * 1e15,
                 'positron_density': np.random.rand(10, 10) * 1e15,
                 'e_field': np.ones(10) * e_field,
                 't': np.linspace(0, 1e-12, 10),
                 'x': np.linspace(-1e-6, 1e-6, 10),
                 'y': np.linspace(-1e-6, 1e-6, 10)},
                plasma_params
            )

            # Try multiple fitting methods
            fit_results = {}

            # Chi-squared fit
            chi_fit = self.blackbody_fitter.fit_blackbody(energies, spectrum,
                                                        energy_range=(300, 400))
            if chi_fit['success']:
                chi_confidence = self.blackbody_fitter.calculate_fit_confidence(chi_fit)
                fit_results['chi_squared'] = {**chi_fit, **chi_confidence}

            # Bayesian fit
            try:
                bayes_fit = self.blackbody_fitter.bayesian_fit_blackbody(
                    energies, spectrum, energy_range=(300, 400), n_samples=200
                )
                if bayes_fit['success']:
                    bayes_confidence = self.blackbody_fitter.calculate_fit_confidence(bayes_fit)
                    fit_results['bayesian'] = {**bayes_fit, **bayes_confidence}
            except:
                pass

            # Maximum likelihood fit
            try:
                ml_fit = self.blackbody_fitter.maximum_likelihood_fit(
                    energies, spectrum, energy_range=(300, 400)
                )
                if ml_fit['success']:
                    ml_confidence = self.blackbody_fitter.calculate_fit_confidence(ml_fit)
                    fit_results['maximum_likelihood'] = {**ml_fit, **ml_confidence}
            except:
                pass

            # Find best fit method
            best_confidence = 0
            best_method = None
            best_fit = None

            for method, result in fit_results.items():
                confidence = result.get('overall_confidence', 0)
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_method = method
                    best_fit = result

            # Calculate derived metrics
            horizon_strength = self._estimate_horizon_strength(e_field, plasma_params)
            pair_production_rate = self._estimate_pair_production(e_field, plasma_params)

            return {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'parameters': {
                    'wavelength': wavelength,
                    'pulse_duration': pulse_duration,
                    'focus_diameter': focus_diameter,
                    'pressure': pressure,
                    'gas_type': gas_type,
                    'e_field': e_field
                },
                'derived_metrics': {
                    'horizon_strength': horizon_strength,
                    'pair_production_rate': pair_production_rate
                },
                'fit_results': fit_results,
                'best_fit': {
                    'method': best_method,
                    'confidence': best_confidence,
                    'temperature': best_fit.get('T_fit') if best_fit else None,
                    'chi_squared': best_fit.get('reduced_chi_squared') if best_fit else None
                } if best_fit else None
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'parameters': {
                    'wavelength': wavelength,
                    'pulse_duration': pulse_duration,
                    'focus_diameter': focus_diameter,
                    'pressure': pressure,
                    'gas_type': gas_type
                }
            }

    def _estimate_horizon_strength(self, e_field, plasma_params):
        """
        Estimate analog horizon strength based on plasma parameters.

        Einstein-inspired: Consider the fundamental relationship between field strength
        and horizon formation in analog gravity systems.
        """
        # Simplified model: horizon strength proportional to E-field relative to Schwinger limit
        schwinger_limit = 1.3e18  # V/m
        relative_field = e_field / schwinger_limit

        # Plasma frequency correction
        pressure = plasma_params['pressure']
        plasma_freq = np.sqrt(pressure * 1e-5) * 1e12  # Rough estimate in Hz

        # Horizon strength (c/(dU/dx) in analog systems)
        horizon_strength = relative_field * (1 + plasma_freq / 1e15)

        return min(horizon_strength, 10)  # Cap at reasonable value

    def _estimate_pair_production(self, e_field, plasma_params):
        """
        Estimate pair production rate.

        Based on Schwinger mechanism in strong fields.
        """
        schwinger_limit = 1.3e18  # V/m
        relative_field = e_field / schwinger_limit

        # Exponential dependence on field strength
        pair_rate = 1e10 * np.exp(10 * (relative_field - 1)) if relative_field > 0.8 else 0

        return pair_rate

    def analyze_optimization_results(self, results):
        """
        Analyze optimization results and extract insights.

        Einstein-inspired: Synthesize findings to understand the fundamental
        relationships between parameters and Hawking radiation detection.
        """
        print("\n" + "=" * 80)
        print("OPTIMIZATION ANALYSIS - FUNDAMENTAL INSIGHTS")
        print("=" * 80)

        # Extract successful results
        successful_results = [r for r in self.optimization_results if r['success']]

        if not successful_results:
            print("No successful parameter configurations found.")
            return

        # Analyze parameter correlations with confidence
        print(f"\nAnalyzed {len(successful_results)} successful configurations")

        # Find best configuration
        best_result = max(successful_results,
                         key=lambda x: x.get('best_fit', {}).get('confidence', 0))

        print("\nBEST CONFIGURATION FOUND:")
        print("-" * 40)
        params = best_result['parameters']
        best_fit = best_result.get('best_fit', {})
        derived = best_result.get('derived_metrics', {})

        print(f"Wavelength: {params['wavelength']*1e9:.1f} nm")
        print(f"Pulse duration: {params['pulse_duration']*1e15:.1f} fs")
        print(f"Focus diameter: {params['focus_diameter']*1e6:.1f} µm")
        print(f"Pressure: {params['pressure']:.2e} Torr")
        print(f"Electric field: {params['e_field']:.2e} V/m")
        print(f"Horizon strength: {derived.get('horizon_strength', 'N/A'):.2f}")
        print(f"Best fit method: {best_fit.get('method', 'N/A')}")
        print(f"Confidence: {best_fit.get('confidence', 0):.3f}")
        print(f"Temperature: {best_fit.get('temperature', 0):.2e} K")
        print(f"Reduced χ²: {best_fit.get('chi_squared', 'N/A'):.3f}")

        # Parameter sensitivity analysis
        self._analyze_parameter_sensitivity(successful_results)

        return best_result

    def _analyze_parameter_sensitivity(self, results):
        """
        Analyze how different parameters affect the detection confidence.
        """
        print("\nPARAMETER SENSITIVITY ANALYSIS:")
        print("-" * 40)

        # Extract parameter arrays and confidence scores
        wavelengths = []
        durations = []
        focuses = []
        pressures = []
        confidences = []

        for result in results:
            if result.get('best_fit'):
                params = result['parameters']
                conf = result['best_fit']['confidence']

                wavelengths.append(params['wavelength'] * 1e9)  # nm
                durations.append(params['pulse_duration'] * 1e15)  # fs
                focuses.append(params['focus_diameter'] * 1e6)  # µm
                pressures.append(params['pressure'])
                confidences.append(conf)

        if not confidences:
            print("Insufficient data for sensitivity analysis.")
            return

        # Calculate correlations
        params_data = [wavelengths, durations, focuses, pressures]
        param_names = ['Wavelength (nm)', 'Pulse Duration (fs)',
                      'Focus Diameter (µm)', 'Pressure (Torr)']

        print("Parameter correlations with detection confidence:")
        for name, param_data in zip(param_names, params_data):
            if len(param_data) > 5:
                correlation = np.corrcoef(param_data, confidences)[0, 1]
                print("10s")

    def save_results(self, results, filename=None):
        """
        Save optimization results to file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"parameter_optimization_{timestamp}.json"

        output_path = os.path.join(os.path.dirname(__file__), 'results', filename)

        # Convert numpy arrays and other non-JSON types to serializable forms
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_to_serializable(item) for item in obj)
            else:
                return obj

        serializable_results = convert_to_serializable(results)

        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"\nResults saved to: {output_path}")
        return output_path

def main():
    """
    Main function to run the Einstein-inspired parameter optimization.
    """
    print("MICRO-SINGULARITY FACTORY - EINSTEIN-INSPIRED PARAMETER OPTIMIZATION")
    print("=" * 80)

    optimizer = ParameterOptimizer()

    # Run systematic parameter sweep
    sweep_results = optimizer.systematic_parameter_sweep(n_points=8)

    # Run global optimization
    global_results = optimizer.global_optimization(max_evaluations=30)

    # Analyze all results
    analysis = optimizer.analyze_optimization_results(sweep_results)

    # Save comprehensive results
    all_results = {
        'sweep_results': sweep_results,
        'global_optimization': global_results,
        'analysis': analysis,
        'total_evaluations': len(optimizer.optimization_results)
    }

    saved_path = optimizer.save_results(all_results)

    print("\n" + "=" * 80)
    print("PARAMETER OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"Total parameter configurations evaluated: {len(optimizer.optimization_results)}")
    print(f"Results saved to: {saved_path}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
