"""
Numerical Validation Framework for Enhanced Methods

This module provides comprehensive validation of the enhanced numerical methods
against analytical solutions, including:

- Analytical test cases with known solutions
- Convergence order verification
- Error analysis and accuracy assessment
- Performance benchmarking
- Grid independence studies
- Comparative analysis with existing methods

Author: Claude Scientific Computing Expert
Date: November 2025
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .enhanced_horizon_detection import EnhancedHorizonDetector, HorizonDetectionConfig
from .enhanced_numerical_methods import (
    EnhancedInterpolation,
    FourthOrderFiniteDifferences,
)


@dataclass
class ValidationTestCase:
    """Definition of a validation test case"""

    name: str
    description: str
    analytical_solution: Callable[[np.ndarray], np.ndarray]
    analytical_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None
    analytical_second_derivative: Optional[Callable[[np.ndarray], np.ndarray]] = None
    domain: Tuple[float, float] = (0.0, 1.0)
    expected_convergence_order: float = 2.0
    tolerance: float = 1e-10
    difficulty: str = "easy"  # 'easy', 'medium', 'hard'


@dataclass
class ValidationResults:
    """Results of numerical validation tests"""

    test_name: str
    method: str
    grid_sizes: List[int]
    errors: List[float]
    convergence_orders: List[float]
    execution_times: List[float]
    passed_convergence_test: bool
    achieved_order: float
    recommendations: List[str] = field(default_factory=list)
    analytical_comparison: Dict[str, float] = field(default_factory=dict)


@dataclass
class PerformanceBenchmark:
    """Performance benchmarking results"""

    method_name: str
    grid_size: int
    execution_time: float
    memory_usage: float
    accuracy_metric: float
    efficiency_score: float  # accuracy per unit time


class AnalyticalTestCases:
    """Collection of analytical test cases for validation"""

    @staticmethod
    def polynomial_test_case() -> ValidationTestCase:
        """Polynomial function with known derivatives"""

        def f(x):
            return x**3 - 2 * x**2 + x + 1

        def df(x):
            return 3 * x**2 - 4 * x + 1

        def d2f(x):
            return 6 * x - 4

        return ValidationTestCase(
            name="polynomial_cubic",
            description="Cubic polynomial f(x) = x³ - 2x² + x + 1",
            analytical_solution=f,
            analytical_derivative=df,
            analytical_second_derivative=d2f,
            domain=(0.0, 2.0),
            expected_convergence_order=4.0,  # Should be exact for polynomials
            tolerance=1e-12,
        )

    @staticmethod
    def trigonometric_test_case() -> ValidationTestCase:
        """Trigonometric function for smooth derivative testing"""

        def f(x):
            return np.sin(2 * np.pi * x) + 0.5 * np.cos(4 * np.pi * x)

        def df(x):
            return 2 * np.pi * np.cos(2 * np.pi * x) - 2 * np.pi * np.sin(4 * np.pi * x)

        def d2f(x):
            return -4 * np.pi**2 * np.sin(2 * np.pi * x) - 8 * np.pi**2 * np.cos(4 * np.pi * x)

        return ValidationTestCase(
            name="trigonometric_smooth",
            description="Smooth trigonometric function",
            analytical_solution=f,
            analytical_derivative=df,
            analytical_second_derivative=d2f,
            domain=(0.0, 1.0),
            expected_convergence_order=4.0,
            tolerance=1e-8,
        )

    @staticmethod
    def exponential_test_case() -> ValidationTestCase:
        """Exponential function for convergence testing"""

        def f(x):
            return np.exp(-(x**2)) * np.cos(5 * x)

        def df(x):
            return np.exp(-(x**2)) * (-2 * x * np.cos(5 * x) - 5 * np.sin(5 * x))

        def d2f(x):
            return np.exp(-(x**2)) * ((4 * x**2 - 25) * np.cos(5 * x) + 20 * x * np.sin(5 * x))

        return ValidationTestCase(
            name="exponential_oscillatory",
            description="Exponential times oscillatory function",
            analytical_solution=f,
            analytical_derivative=df,
            analytical_second_derivative=d2f,
            domain=(-2.0, 2.0),
            expected_convergence_order=2.0,  # Reduced due to oscillations
            tolerance=1e-6,
            difficulty="medium",
        )

    @staticmethod
    def discontinuous_test_case() -> ValidationTestCase:
        """Function with discontinuous derivative for robustness testing"""

        def f(x):
            return np.where(x < 0.5, x**2, (x - 0.5) ** 2 + 0.25)

        def df(x):
            return np.where(x < 0.5, 2 * x, 2 * (x - 0.5))

        return ValidationTestCase(
            name="discontinuous_derivative",
            description="Function with discontinuous derivative at x=0.5",
            analytical_solution=f,
            analytical_derivative=df,
            domain=(0.0, 1.0),
            expected_convergence_order=1.0,  # Reduced due to discontinuity
            tolerance=1e-2,
            difficulty="hard",
        )

    @staticmethod
    def horizon_analytical_test_case() -> ValidationTestCase:
        """Analytical horizon-like function for horizon detection testing"""

        def f(x):
            # Function mimicking |v| - c_s with a clear horizon
            return np.tanh(10 * (x - 0.5))

        def df(x):
            return 10 / np.cosh(10 * (x - 0.5)) ** 2

        return ValidationTestCase(
            name="horizon_function",
            description="Analytical horizon-like function",
            analytical_solution=f,
            analytical_derivative=df,
            domain=(0.0, 1.0),
            expected_convergence_order=4.0,
            tolerance=1e-8,
            difficulty="medium",
        )


class ConvergenceAnalysis:
    """Tools for convergence analysis and order verification"""

    @staticmethod
    def compute_convergence_order(errors: List[float], grid_ratios: List[float]) -> List[float]:
        """
        Compute observed order of accuracy from error sequence

        Args:
            errors: List of errors for different grid resolutions
            grid_ratios: List of grid refinement ratios (usually 2.0)

        Returns:
            List of observed convergence orders
        """
        orders = []
        for i in range(1, len(errors)):
            if errors[i - 1] > 1e-15 and errors[i] > 1e-15:  # Avoid division by zero
                r = grid_ratios[i - 1] if i - 1 < len(grid_ratios) else 2.0
                order = np.log(errors[i - 1] / errors[i]) / np.log(r)
                orders.append(order)
            else:
                orders.append(np.nan)

        return orders

    @staticmethod
    def richardson_extrapolation_error(
        errors: List[float], orders: List[float], refinement_ratio: float = 2.0
    ) -> float:
        """
        Estimate Richardson extrapolated error

        Args:
            errors: Error sequence
            orders: Observed convergence orders
            refinement_ratio: Grid refinement ratio

        Returns:
            Richardson extrapolated error estimate
        """
        if len(errors) >= 2 and len(orders) >= 1:
            r = refinement_ratio
            p = orders[-1] if not np.isnan(orders[-1]) else 2.0
            richardson_error = errors[-1] / (r**p - 1)
            return richardson_error
        return errors[-1] if errors else 0.0


class NumericalValidator:
    """Main validation framework class"""

    def __init__(self):
        """Initialize numerical validator"""
        self.test_cases = self._get_standard_test_cases()
        self.validation_results = {}
        self.performance_benchmarks = {}

    def _get_standard_test_cases(self) -> List[ValidationTestCase]:
        """Get standard test cases for validation"""
        return [
            AnalyticalTestCases.polynomial_test_case(),
            AnalyticalTestCases.trigonometric_test_case(),
            AnalyticalTestCases.exponential_test_case(),
            AnalyticalTestCases.discontinuous_test_case(),
            AnalyticalTestCases.horizon_analytical_test_case(),
        ]

    def validate_gradient_methods(
        self, grid_sizes: List[int] = None, test_cases: List[ValidationTestCase] = None
    ) -> Dict[str, ValidationResults]:
        """
        Validate gradient computation methods against analytical solutions

        Args:
            grid_sizes: List of grid sizes to test
            test_cases: Specific test cases to validate (default: all standard cases)

        Returns:
            Dictionary of validation results for each test case
        """
        if grid_sizes is None:
            grid_sizes = [10, 20, 40, 80, 160]

        if test_cases is None:
            test_cases = self.test_cases

        results = {}
        fd_solver = FourthOrderFiniteDifferences()

        for test_case in test_cases:
            print(f"\nValidating gradient methods for {test_case.name}")
            print(f"Description: {test_case.description}")

            test_results = []

            # Test 2nd-order method (numpy gradient)
            result_2nd = self._test_gradient_method(
                test_case, grid_sizes, "numpy_gradient", lambda x, y: np.gradient(y, x), "2nd_order"
            )
            test_results.append(result_2nd)

            # Test 4th-order method
            if max(grid_sizes) >= 5:  # 4th-order needs at least 5 points
                result_4th = self._test_gradient_method(
                    test_case,
                    grid_sizes,
                    "4th_order_central",
                    lambda x, y: fd_solver.gradient_central_4th(x, y),
                    "4th_order",
                )
                test_results.append(result_4th)

            results[test_case.name] = test_results

        self.validation_results["gradient_methods"] = results
        return results

    def _test_gradient_method(
        self,
        test_case: ValidationTestCase,
        grid_sizes: List[int],
        method_name: str,
        method_function: Callable,
        order_label: str,
    ) -> ValidationResults:
        """Test a specific gradient computation method"""
        errors = []
        execution_times = []
        analytical_errors = []

        for n in grid_sizes:
            # Skip if grid too small for method
            if order_label == "4th_order" and n < 5:
                continue

            # Create grid
            x = np.linspace(test_case.domain[0], test_case.domain[1], n)
            y = test_case.analytical_solution(x)

            # Compute gradient
            start_time = time.time()
            computed_gradient = method_function(x, y)
            execution_time = time.time() - start_time

            # Compare with analytical solution
            if test_case.analytical_derivative is not None:
                analytical_gradient = test_case.analytical_derivative(x)

                # Compute error (L2 norm)
                error = np.linalg.norm(computed_gradient - analytical_gradient) / np.linalg.norm(
                    analytical_gradient
                )
                errors.append(error)

                # Compute analytical error for comparison
                analytical_error = np.linalg.norm(
                    np.gradient(y, x) - analytical_gradient
                ) / np.linalg.norm(analytical_gradient)
                analytical_errors.append(analytical_error)
            else:
                errors.append(0.0)
                analytical_errors.append(0.0)

            execution_times.append(execution_time)

        # Compute convergence orders
        grid_ratios = [grid_sizes[i + 1] / grid_sizes[i] for i in range(len(grid_sizes) - 1)]
        convergence_orders = ConvergenceAnalysis.compute_convergence_order(errors, grid_ratios)

        # Check if convergence test passed
        if len(convergence_orders) > 0:
            last_order = convergence_orders[-1]
            passed_convergence = abs(last_order - test_case.expected_convergence_order) < 0.5
            achieved_order = last_order
        else:
            passed_convergence = False
            achieved_order = 0.0

        # Generate recommendations
        recommendations = []
        if not passed_convergence:
            recommendations.append(
                f"Expected order {test_case.expected_convergence_order}, got {achieved_order:.2f}"
            )
        if max(execution_times) > 0.1:
            recommendations.append("Method may be computationally expensive for large grids")

        return ValidationResults(
            test_name=test_case.name,
            method=method_name,
            grid_sizes=[n for n in grid_sizes if n >= 5 or order_label == "2nd_order"],
            errors=errors,
            convergence_orders=convergence_orders,
            execution_times=execution_times,
            passed_convergence_test=passed_convergence,
            achieved_order=achieved_order,
            recommendations=recommendations,
            analytical_comparison={
                "baseline_error": analytical_errors[-1] if analytical_errors else 0.0
            },
        )

    def validate_interpolation_methods(
        self, grid_sizes: List[int] = None, test_cases: List[ValidationTestCase] = None
    ) -> Dict[str, ValidationResults]:
        """Validate interpolation methods"""
        if grid_sizes is None:
            grid_sizes = [10, 20, 40, 80]

        if test_cases is None:
            test_cases = self.test_cases[:3]  # Use smooth test cases for interpolation

        results = {}
        interpolator = EnhancedInterpolation()

        for test_case in test_cases:
            print(f"\nValidating interpolation methods for {test_case.name}")

            test_results = []

            # Test linear interpolation
            result_linear = self._test_interpolation_method(
                test_case, grid_sizes, "linear", lambda x, y, xi: np.interp(xi, x, y), "linear"
            )
            test_results.append(result_linear)

            # Test cubic spline interpolation
            result_cubic = self._test_interpolation_method(
                test_case,
                grid_sizes,
                "cubic_spline",
                lambda x, y, xi: interpolator.cubic_spline_interpolation(x, y)(xi),
                "cubic",
            )
            test_results.append(result_cubic)

            # Test monotonic cubic interpolation
            result_monotonic = self._test_interpolation_method(
                test_case,
                grid_sizes,
                "pchip",
                lambda x, y, xi: interpolator.cubic_spline_interpolation(x, y, monotonic=True)(xi),
                "monotonic",
            )
            test_results.append(result_monotonic)

            results[test_case.name] = test_results

        self.validation_results["interpolation_methods"] = results
        return results

    def _test_interpolation_method(
        self,
        test_case: ValidationTestCase,
        grid_sizes: List[int],
        method_name: str,
        method_function: Callable,
        order_label: str,
    ) -> ValidationResults:
        """Test a specific interpolation method"""
        errors = []
        execution_times = []

        # Create high-resolution test grid
        x_test = np.linspace(test_case.domain[0], test_case.domain[1], 1000)
        y_test_analytical = test_case.analytical_solution(x_test)

        for n in grid_sizes:
            # Create training grid
            x = np.linspace(test_case.domain[0], test_case.domain[1], n)
            y = test_case.analytical_solution(x)

            # Perform interpolation
            start_time = time.time()
            y_test_interpolated = method_function(x, y, x_test)
            execution_time = time.time() - start_time

            # Compute error
            error = np.linalg.norm(y_test_interpolated - y_test_analytical) / np.linalg.norm(
                y_test_analytical
            )
            errors.append(error)
            execution_times.append(execution_time)

        # Compute convergence orders
        grid_ratios = [grid_sizes[i + 1] / grid_sizes[i] for i in range(len(grid_sizes) - 1)]
        convergence_orders = ConvergenceAnalysis.compute_convergence_order(errors, grid_ratios)

        # Check convergence
        if len(convergence_orders) > 0:
            last_order = convergence_orders[-1]
            passed_convergence = last_order > 1.5  # Interpolation should converge reasonably fast
            achieved_order = last_order
        else:
            passed_convergence = False
            achieved_order = 0.0

        return ValidationResults(
            test_name=test_case.name,
            method=method_name,
            grid_sizes=grid_sizes,
            errors=errors,
            convergence_orders=convergence_orders,
            execution_times=execution_times,
            passed_convergence_test=passed_convergence,
            achieved_order=achieved_order,
        )

    def validate_horizon_detection(
        self, test_cases: List[ValidationTestCase] = None
    ) -> Dict[str, ValidationResults]:
        """Validate horizon detection methods"""
        if test_cases is None:
            test_cases = [AnalyticalTestCases.horizon_analytical_test_case()]

        results = {}

        for test_case in test_cases:
            print(f"\nValidating horizon detection for {test_case.name}")

            grid_sizes = [50, 100, 200, 400]
            test_results = []

            # Test 2nd-order horizon detection
            config_2nd = HorizonDetectionConfig(gradient_order=2)
            detector_2nd = EnhancedHorizonDetector(config_2nd)
            result_2nd = self._test_horizon_detection_method(
                test_case, grid_sizes, detector_2nd, "2nd_order"
            )
            test_results.append(result_2nd)

            # Test 4th-order horizon detection
            config_4th = HorizonDetectionConfig(gradient_order=4)
            detector_4th = EnhancedHorizonDetector(config_4th)
            result_4th = self._test_horizon_detection_method(
                test_case, grid_sizes, detector_4th, "4th_order"
            )
            test_results.append(result_4th)

            results[test_case.name] = test_results

        self.validation_results["horizon_detection"] = results
        return results

    def _test_horizon_detection_method(
        self,
        test_case: ValidationTestCase,
        grid_sizes: List[int],
        detector: EnhancedHorizonDetector,
        method_name: str,
    ) -> ValidationResults:
        """Test horizon detection method"""
        errors = []
        execution_times = []

        # Analytical horizon location for test case
        analytical_horizon = 0.5  # For the standard horizon test case

        for n in grid_sizes:
            # Create grid
            x = np.linspace(test_case.domain[0], test_case.domain[1], n)
            y = test_case.analytical_solution(x)
            c_s = np.ones_like(x)  # Constant sound speed for simplicity

            # Detect horizons
            start_time = time.time()
            result = detector.find_horizons_enhanced(x, y, c_s)
            execution_time = time.time() - start_time

            # Compute error in horizon position
            if len(result.positions) > 0:
                # Find closest horizon to analytical location
                distances = np.abs(result.positions - analytical_horizon)
                min_distance = np.min(distances)
                errors.append(min_distance)
            else:
                errors.append(1.0)  # Large error if no horizon found

            execution_times.append(execution_time)

        # Expected convergence order for root finding
        expected_order = 4.0 if "4th" in method_name else 2.0

        # Compute convergence orders
        grid_ratios = [grid_sizes[i + 1] / grid_sizes[i] for i in range(len(grid_sizes) - 1)]
        convergence_orders = ConvergenceAnalysis.compute_convergence_order(errors, grid_ratios)

        # Check convergence
        if len(convergence_orders) > 0:
            last_order = convergence_orders[-1]
            passed_convergence = abs(last_order - expected_order) < 1.0
            achieved_order = last_order
        else:
            passed_convergence = False
            achieved_order = 0.0

        return ValidationResults(
            test_name=test_case.name,
            method=method_name,
            grid_sizes=grid_sizes,
            errors=errors,
            convergence_orders=convergence_orders,
            execution_times=execution_times,
            passed_convergence_test=passed_convergence,
            achieved_order=achieved_order,
        )

    def performance_benchmark(self, grid_size: int = 1000) -> Dict[str, PerformanceBenchmark]:
        """Benchmark performance of different methods"""
        print(f"\nRunning performance benchmarks with grid size {grid_size}")

        benchmarks = {}
        x = np.linspace(0, 1, grid_size)

        # Test function
        y = np.sin(10 * np.pi * x) * np.exp(-(x**2))

        # Benchmark 2nd-order gradient
        start_time = time.time()
        grad_2nd = np.gradient(y, x)
        time_2nd = time.time() - start_time
        accuracy_2nd = np.linalg.norm(
            grad_2nd
            - (
                10 * np.pi * np.cos(10 * np.pi * x) * np.exp(-(x**2))
                - 2 * x * np.sin(10 * np.pi * x) * np.exp(-(x**2))
            )
        )
        benchmarks["2nd_order"] = PerformanceBenchmark(
            method_name="2nd_order_gradient",
            grid_size=grid_size,
            execution_time=time_2nd,
            memory_usage=0.0,  # Could be measured with memory_profiler
            accuracy_metric=accuracy_2nd,
            efficiency_score=accuracy_2nd / time_2nd if time_2nd > 0 else 0.0,
        )

        # Benchmark 4th-order gradient
        if grid_size >= 5:
            fd_solver = FourthOrderFiniteDifferences()
            start_time = time.time()
            grad_4th = fd_solver.gradient_central_4th(x, y)
            time_4th = time.time() - start_time
            accuracy_4th = np.linalg.norm(
                grad_4th
                - (
                    10 * np.pi * np.cos(10 * np.pi * x) * np.exp(-(x**2))
                    - 2 * x * np.sin(10 * np.pi * x) * np.exp(-(x**2))
                )
            )
            benchmarks["4th_order"] = PerformanceBenchmark(
                method_name="4th_order_gradient",
                grid_size=grid_size,
                execution_time=time_4th,
                memory_usage=0.0,
                accuracy_metric=accuracy_4th,
                efficiency_score=accuracy_4th / time_4th if time_4th > 0 else 0.0,
            )

        self.performance_benchmarks[grid_size] = benchmarks
        return benchmarks

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("NUMERICAL VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Gradient methods validation
        if "gradient_methods" in self.validation_results:
            report.append("GRADIENT METHODS VALIDATION")
            report.append("-" * 40)

            for test_name, test_results in self.validation_results["gradient_methods"].items():
                report.append(f"\nTest Case: {test_name}")
                for result in test_results:
                    report.append(f"  Method: {result.method}")
                    report.append(f"    Achieved order: {result.achieved_order:.2f}")
                    report.append(
                        f"    Convergence test: {'PASSED' if result.passed_convergence_test else 'FAILED'}"
                    )
                    report.append(f"    Final error: {result.errors[-1]:.2e}")
                    report.append(f"    Execution time: {result.execution_times[-1]:.4f}s")
                    if result.recommendations:
                        report.append(f"    Recommendations: {', '.join(result.recommendations)}")

        # Interpolation methods validation
        if "interpolation_methods" in self.validation_results:
            report.append("\nINTERPOLATION METHODS VALIDATION")
            report.append("-" * 40)

            for test_name, test_results in self.validation_results["interpolation_methods"].items():
                report.append(f"\nTest Case: {test_name}")
                for result in test_results:
                    report.append(f"  Method: {result.method}")
                    report.append(f"    Achieved order: {result.achieved_order:.2f}")
                    report.append(f"    Final error: {result.errors[-1]:.2e}")

        # Performance benchmarks
        if self.performance_benchmarks:
            report.append("\nPERFORMANCE BENCHMARKS")
            report.append("-" * 40)

            for grid_size, benchmarks in self.performance_benchmarks.items():
                report.append(f"\nGrid size: {grid_size}")
                for method_name, benchmark in benchmarks.items():
                    report.append(f"  {method_name}:")
                    report.append(f"    Time: {benchmark.execution_time:.4f}s")
                    report.append(f"    Accuracy: {benchmark.accuracy_metric:.2e}")
                    report.append(f"    Efficiency: {benchmark.efficiency_score:.2e}")

        # Summary and recommendations
        report.append("\nSUMMARY AND RECOMMENDATIONS")
        report.append("-" * 40)

        # Count passed tests
        total_tests = 0
        passed_tests = 0

        for category, results in self.validation_results.items():
            for test_name, test_results in results.items():
                for result in test_results:
                    total_tests += 1
                    if result.passed_convergence_test:
                        passed_tests += 1

        report.append(f"Total tests: {total_tests}")
        report.append(f"Passed tests: {passed_tests}")
        report.append(f"Success rate: {100*passed_tests/total_tests:.1f}%")

        if passed_tests / total_tests > 0.8:
            report.append("✅ OVERALL VALIDATION: PASSED")
        else:
            report.append("❌ OVERALL VALIDATION: NEEDS IMPROVEMENT")

        return "\n".join(report)


# Convenience function for running full validation
def run_comprehensive_validation() -> str:
    """Run comprehensive validation of all enhanced numerical methods"""
    validator = NumericalValidator()

    print("Starting comprehensive numerical validation...")

    # Validate gradient methods
    validator.validate_gradient_methods()

    # Validate interpolation methods
    validator.validate_interpolation_methods()

    # Validate horizon detection
    validator.validate_horizon_detection()

    # Run performance benchmarks
    validator.performance_benchmark()

    # Generate report
    report = validator.generate_validation_report()
    print(report)

    return report
