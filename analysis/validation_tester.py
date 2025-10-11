#!/usr/bin/env python3
"""
Synthetic Data Validation for Enhanced Fitting Methods

Einstein-inspired: Test fitting methods with known synthetic data to validate
their reliability and quantify systematic uncertainties.
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings

# Add subdirectories to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'hawking-hunter'))

from blackbody_fitter import BlackbodyFitter
from spectrum_analyzer import SpectrumAnalyzer

class ValidationTester:
    """
    Tests fitting methods with synthetic data to validate reliability.

    Einstein-inspired: Use controlled synthetic experiments to understand
    method limitations and systematic uncertainties.
    """

    def __init__(self):
        self.blackbody_fitter = BlackbodyFitter()
        self.spectrum_analyzer = SpectrumAnalyzer()

        # Test scenarios
        self.test_scenarios = {
            'perfect_blackbody': {
                'description': 'Perfect Planck spectrum with known temperature',
                'noise_level': 0.001,
                'plasma_correction': 0.0
            },
            'noisy_blackbody': {
                'description': 'Planck spectrum with realistic noise',
                'noise_level': 0.05,
                'plasma_correction': 0.0
            },
            'plasma_modified': {
                'description': 'Black-body with plasma frequency corrections',
                'noise_level': 0.02,
                'plasma_correction': 0.2
            },
            'systematic_errors': {
                'description': 'Black-body with systematic calibration errors',
                'noise_level': 0.03,
                'systematic_error': 0.1
            },
            'no_signal': {
                'description': 'Background only (no Hawking signal)',
                'noise_level': 0.02,
                'signal_strength': 0.0
            }
        }

    def run_validation_tests(self, n_tests=10):
        """
        Run comprehensive validation tests with synthetic data.
        """
        print("=" * 80)
        print("SYNTHETIC DATA VALIDATION - FITTING METHOD RELIABILITY")
        print("=" * 80)

        all_results = {}

        for scenario_name, scenario_config in self.test_scenarios.items():
            print(f"\nTesting scenario: {scenario_name}")
            print(f"Description: {scenario_config['description']}")
            print("-" * 60)

            scenario_results = []
            success_count = 0

            for i in range(n_tests):
                result = self._run_single_test(scenario_name, scenario_config)
                scenario_results.append(result)

                if result['fit_success']:
                    success_count += 1

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{n_tests} tests...")

            # Analyze scenario results
            analysis = self._analyze_scenario_results(scenario_results, scenario_config)

            all_results[scenario_name] = {
                'config': scenario_config,
                'results': scenario_results,
                'analysis': analysis,
                'success_rate': success_count / n_tests
            }

            print(f"  Success rate: {success_count}/{n_tests} ({success_count/n_tests:.1%})")
            print(f"  Mean temperature error: {analysis['mean_temp_error']:.2e} K")
            print(f"  Temperature accuracy: {analysis['temp_accuracy']:.2%}")

        # Overall analysis
        overall_analysis = self._analyze_all_results(all_results)

        return {
            'scenarios': all_results,
            'overall_analysis': overall_analysis,
            'timestamp': datetime.now().isoformat()
        }

    def _run_single_test(self, scenario_name, config):
        """
        Run a single validation test with synthetic data.
        """
        # Generate synthetic spectrum
        energies = np.linspace(50, 1000, 500)
        true_temperature = self.spectrum_analyzer.T_H_target  # 1.2e9 K

        # Generate true spectrum
        if scenario_name == 'no_signal':
            # Background only
            spectrum_true = self._generate_background_spectrum(energies)
        else:
            # Black-body signal
            spectrum_true = self.blackbody_fitter.modified_blackbody_function(
                energies, true_temperature, 1.0, 0.1, config.get('plasma_correction', 0.0)
            )

        # Add noise
        noise_level = config['noise_level']
        noise = np.random.normal(0, noise_level * np.max(spectrum_true), len(energies))
        spectrum_noisy = spectrum_true + noise

        # Add systematic errors if specified
        if 'systematic_error' in config:
            systematic_factor = 1 + config['systematic_error'] * np.sin(energies / 100)
            spectrum_noisy *= systematic_factor

        # Try fitting with all methods
        fit_results = {}

        # Chi-squared fit
        try:
            chi_fit = self.blackbody_fitter.fit_blackbody(
                energies, spectrum_noisy, energy_range=(300, 400)
            )
            if chi_fit['success']:
                chi_confidence = self.blackbody_fitter.calculate_fit_confidence(chi_fit)
                fit_results['chi_squared'] = {**chi_fit, **chi_confidence}
        except:
            pass

        # Bayesian fit
        try:
            bayes_fit = self.blackbody_fitter.bayesian_fit_blackbody(
                energies, spectrum_noisy, energy_range=(300, 400), n_samples=300
            )
            if bayes_fit['success']:
                bayes_confidence = self.blackbody_fitter.calculate_fit_confidence(bayes_fit)
                fit_results['bayesian'] = {**bayes_fit, **bayes_confidence}
        except:
            pass

        # Maximum likelihood fit
        try:
            ml_fit = self.blackbody_fitter.maximum_likelihood_fit(
                energies, spectrum_noisy, energy_range=(300, 400)
            )
            if ml_fit['success']:
                ml_confidence = self.blackbody_fitter.calculate_fit_confidence(ml_fit)
                fit_results['maximum_likelihood'] = {**ml_fit, **ml_confidence}
        except:
            pass

        # Find best fit
        best_confidence = 0
        best_method = None
        best_fit = None

        for method, result in fit_results.items():
            confidence = result.get('overall_confidence', 0)
            if confidence > best_confidence:
                best_confidence = confidence
                best_method = method
                best_fit = result

        # Calculate metrics
        fit_success = best_fit is not None
        temp_error = 0
        confidence_level = 0

        if fit_success:
            fitted_temp = best_fit['T_fit']
            temp_error = abs(fitted_temp - true_temperature) / true_temperature
            confidence_level = best_confidence

        return {
            'scenario': scenario_name,
            'true_temperature': true_temperature,
            'fit_success': fit_success,
            'best_method': best_method,
            'temperature_error': temp_error,
            'confidence_level': confidence_level,
            'fit_results': fit_results,
            'energies': energies,
            'spectrum_true': spectrum_true,
            'spectrum_noisy': spectrum_noisy
        }

    def _generate_background_spectrum(self, energies):
        """
        Generate a realistic background spectrum (no Hawking signal).
        """
        # Thermal background similar to what the analyzer generates
        k_B = 8.617e-5
        T_plasma = 1e7  # Plasma temperature
        background = np.exp(-energies / (k_B * T_plasma))

        # Add some structure to make it realistic
        background *= (1 + 0.1 * np.sin(energies / 200))
        background += 0.01 * np.random.random(len(energies))

        return background

    def _analyze_scenario_results(self, results, config):
        """
        Analyze results from a single test scenario.
        """
        successful_fits = [r for r in results if r['fit_success']]

        if not successful_fits:
            return {
                'mean_temp_error': float('inf'),
                'std_temp_error': float('inf'),
                'temp_accuracy': 0.0,
                'mean_confidence': 0.0,
                'method_distribution': {}
            }

        temp_errors = [r['temperature_error'] for r in successful_fits]
        confidences = [r['confidence_level'] for r in successful_fits]
        methods = [r['best_method'] for r in successful_fits]

        # Method distribution
        method_counts = {}
        for method in methods:
            method_counts[method] = method_counts.get(method, 0) + 1

        return {
            'mean_temp_error': np.mean(temp_errors),
            'std_temp_error': np.std(temp_errors),
            'temp_accuracy': 1.0 - np.mean(temp_errors),  # 1 - relative error
            'mean_confidence': np.mean(confidences),
            'method_distribution': method_counts,
            'n_successful': len(successful_fits)
        }

    def _analyze_all_results(self, all_results):
        """
        Analyze results across all test scenarios.
        """
        print("\n" + "=" * 80)
        print("OVERALL VALIDATION ANALYSIS")
        print("=" * 80)

        # Aggregate statistics
        total_tests = 0
        total_successes = 0
        all_temp_errors = []
        method_performance = {}

        for scenario_name, scenario_data in all_results.items():
            results = scenario_data['results']
            success_rate = scenario_data['success_rate']
            analysis = scenario_data['analysis']

            total_tests += len(results)
            total_successes += int(success_rate * len(results))

            if analysis['temp_accuracy'] != 0.0:
                all_temp_errors.extend([r['temperature_error'] for r in results if r['fit_success']])

            # Track method performance
            for result in results:
                if result['fit_success']:
                    method = result['best_method']
                    if method not in method_performance:
                        method_performance[method] = []
                    method_performance[method].append(result['temperature_error'])

            print(f"\n{scenario_name.upper()}:")
            print(f"  Success rate: {success_rate:.1%}")
            print(f"  Temperature accuracy: {analysis['temp_accuracy']:.1%}")
            if analysis['method_distribution']:
                print(f"  Best methods: {analysis['method_distribution']}")

        overall_success_rate = total_successes / total_tests if total_tests > 0 else 0
        mean_temp_error = np.mean(all_temp_errors) if all_temp_errors else float('inf')

        print(f"\nOVERALL RESULTS:")
        print(f"  Total success rate: {overall_success_rate:.1%}")
        print(f"  Mean temperature error: {mean_temp_error:.2e}")
        print(f"  Methods used: {list(method_performance.keys())}")

        # Method comparison
        print(f"\nMETHOD PERFORMANCE:")
        for method, errors in method_performance.items():
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors)
                success_rate = len(errors) / total_successes
                print(f"  {method}: {mean_error:.2e} ± {std_error:.2e} (used in {success_rate:.1%} of fits)")

        return {
            'overall_success_rate': overall_success_rate,
            'mean_temperature_error': mean_temp_error,
            'method_performance': method_performance,
            'recommendations': self._generate_recommendations(all_results)
        }

    def _generate_recommendations(self, all_results):
        """
        Generate recommendations based on validation results.
        """
        recommendations = []

        # Find most reliable method
        method_scores = {}
        for scenario_data in all_results.values():
            for result in scenario_data['results']:
                if result['fit_success']:
                    method = result['best_method']
                    score = 1.0 - result['temperature_error']  # Higher score for lower error
                    if method not in method_scores:
                        method_scores[method] = []
                    method_scores[method].append(score)

        if method_scores:
            best_method = max(method_scores.keys(),
                            key=lambda m: np.mean(method_scores[m]))
            recommendations.append(f"Use {best_method} as primary fitting method")

        # Check for systematic issues
        no_signal_results = all_results.get('no_signal', {}).get('results', [])
        false_positives = sum(1 for r in no_signal_results if r['fit_success'])
        if false_positives > 0:
            recommendations.append("Caution: fitting methods may detect false positives in background-only data")

        return recommendations

    def plot_validation_results(self, results, save_path=None):
        """
        Create comprehensive plots of validation results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Success rates by scenario
        scenarios = list(results['scenarios'].keys())
        success_rates = [results['scenarios'][s]['success_rate'] for s in scenarios]

        axes[0, 0].bar(scenarios, success_rates)
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Fitting Success Rate by Scenario')
        axes[0, 0].tick_params(axis='x', rotation=45)

        # Plot 2: Temperature accuracy by scenario
        accuracies = []
        for s in scenarios:
            analysis = results['scenarios'][s]['analysis']
            accuracies.append(analysis['temp_accuracy'])

        axes[0, 1].bar(scenarios, accuracies)
        axes[0, 1].set_ylabel('Temperature Accuracy (1 - rel_error)')
        axes[0, 1].set_title('Temperature Accuracy by Scenario')
        axes[0, 1].tick_params(axis='x', rotation=45)

        # Plot 3: Method distribution
        method_counts = {}
        for scenario_data in results['scenarios'].values():
            for result in scenario_data['results']:
                if result['fit_success']:
                    method = result['best_method']
                    method_counts[method] = method_counts.get(method, 0) + 1

        if method_counts:
            methods = list(method_counts.keys())
            counts = list(method_counts.values())

            axes[1, 0].pie(counts, labels=methods, autopct='%1.1f%%')
            axes[1, 0].set_title('Method Usage Distribution')

        # Plot 4: Temperature error distribution
        all_errors = []
        for scenario_data in results['scenarios'].values():
            for result in scenario_data['results']:
                if result['fit_success']:
                    all_errors.append(result['temperature_error'])

        if all_errors:
            axes[1, 1].hist(all_errors, bins=20, alpha=0.7)
            axes[1, 1].set_xlabel('Relative Temperature Error')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Temperature Error Distribution')
            axes[1, 1].axvline(np.mean(all_errors), color='r', linestyle='--',
                              label=f'Mean: {np.mean(all_errors):.2e}')
            axes[1, 1].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            plt.show()
            plt.close()
            return None

def main():
    """
    Main function to run synthetic data validation.
    """
    print("MICRO-SINGULARITY FACTORY - SYNTHETIC DATA VALIDATION")
    print("=" * 80)

    validator = ValidationTester()

    # Run validation tests
    results = validator.run_validation_tests(n_tests=5)  # Reduced for faster testing

    # Generate plots
    plot_path = validator.plot_validation_results(results,
        os.path.join(os.path.dirname(__file__), 'results', 'validation_results.png'))

    # Save results
    import json
    results_file = os.path.join(os.path.dirname(__file__), 'results',
                               f'validation_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')

    # Convert numpy arrays and other non-JSON types to serializable forms
    def make_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(make_serializable(item) for item in obj)
        else:
            return obj

    with open(results_file, 'w') as f:
        json.dump(make_serializable(results), f, indent=2)

    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {results_file}")
    if plot_path:
        print(f"Plots saved to: {plot_path}")

    # Print recommendations
    recommendations = results['overall_analysis'].get('recommendations', [])
    if recommendations:
        print("\nRECOMMENDATIONS:")
        for rec in recommendations:
            print(f"  • {rec}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
