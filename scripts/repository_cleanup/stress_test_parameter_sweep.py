#!/usr/bin/env python3
"""
Large-Scale Parameter Sweep Stress Testing

This script performs comprehensive stress testing through large-scale parameter sweeps,
validating system performance, memory usage, and scalability under realistic research workloads.

Features:
- 100+ parameter configurations with systematic variation
- Memory profiling and performance benchmarking
- Concurrent execution testing
- Detailed reporting and visualization
- Regression detection and performance analysis
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from scripts.run_full_pipeline import run_full_pipeline, FullPipelineSummary

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')

# Configure seaborn
sns.set_style("whitegrid")
sns.set_palette("husl")


@dataclass
class StressTestConfig:
    """Configuration for stress testing parameters"""
    sweep_size: int = 100
    concurrent_workers: int = 4
    memory_threshold_mb: float = 8192  # 8GB
    timeout_per_config: float = 300.0  # 5 minutes
    enable_profiling: bool = True
    test_concurrent_execution: bool = True
    generate_visualizations: bool = True
    output_dir: str = "results/stress_testing"
    seed: int = 42


@dataclass
class ParameterConfiguration:
    """Single parameter configuration for testing"""
    config_id: str
    plasma_density: float
    laser_wavelength: float
    laser_intensity: float
    temperature_constant: float
    magnetic_field: Optional[float]
    grid_points: int
    kappa_method: str
    graybody_method: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


@dataclass
class ConfigurationResult:
    """Results from running a single configuration"""
    config_id: str
    success: bool
    execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    error_message: Optional[str] = None
    summary: Optional[FullPipelineSummary] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestSummary:
    """Summary of stress testing results"""
    test_timestamp: str
    total_configurations: int
    successful_configurations: int
    failed_configurations: int
    success_rate: float
    total_execution_time: float
    average_execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    concurrent_workers_used: int
    performance_regression_detected: bool
    scalability_metrics: Dict[str, float]
    recommendations: List[str]
    critical_issues: List[str]


class MemoryProfiler:
    """Memory profiling for stress testing"""

    def __init__(self):
        self.process = psutil.Process()
        self.memory_samples: List[Tuple[float, float]] = []  # (timestamp, memory_mb)
        self.peak_memory = 0.0
        self.avg_memory = 0.0

    def start_profiling(self):
        """Start memory profiling"""
        self.memory_samples.clear()
        self.peak_memory = 0.0

    def sample_memory(self):
        """Sample current memory usage"""
        try:
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            timestamp = time.time()
            self.memory_samples.append((timestamp, memory_mb))
            self.peak_memory = max(self.peak_memory, memory_mb)
            return memory_mb
        except Exception:
            return 0.0

    def stop_profiling(self) -> Tuple[float, float]:
        """Stop profiling and return peak and average memory"""
        if self.memory_samples:
            self.avg_memory = np.mean([mem for _, mem in self.memory_samples])
        else:
            self.avg_memory = 0.0
        return self.peak_memory, self.avg_memory


class ParameterGenerator:
    """Generate parameter configurations for stress testing"""

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_configurations(self, n_configs: int) -> List[ParameterConfiguration]:
        """Generate diverse parameter configurations"""
        configs = []

        # Define parameter ranges based on physical constraints
        density_range = np.logspace(16, 18, n_configs // 4)  # 1e16 to 1e18 cm^-3
        intensity_range = np.logspace(16, 18, n_configs // 4)  # 1e16 to 1e18 W/cm^2
        wavelength_range = np.linspace(400e-9, 1064e-9, n_configs // 4)  # Visible to NIR
        temperature_range = np.logspace(3, 5, n_configs // 4)  # 1e3 to 1e5 K

        # Grid points variations
        grid_points_options = [256, 512, 1024, 2048]

        # Method options
        kappa_methods = ["acoustic", "acoustic_exact", "legacy"]
        graybody_methods = ["dimensionless", "wkb", "acoustic_wkb"]

        for i in range(n_configs):
            config_id = f"config_{i:04d}"

            # Sample parameters with some structure to ensure coverage
            if i < n_configs // 3:
                # Low-intensity regime
                plasma_density = float(self.rng.choice(density_range[:len(density_range)//2]))
                laser_intensity = float(self.rng.choice(intensity_range[:len(intensity_range)//2]))
            elif i < 2 * n_configs // 3:
                # Mid-intensity regime
                plasma_density = float(self.rng.choice(density_range))
                laser_intensity = float(self.rng.choice(intensity_range))
            else:
                # High-intensity regime
                plasma_density = float(self.rng.choice(density_range[len(density_range)//2:]))
                laser_intensity = float(self.rng.choice(intensity_range[len(intensity_range)//2:]))

            laser_wavelength = float(self.rng.choice(wavelength_range))
            temperature_constant = float(self.rng.choice(temperature_range))

            # Occasionally add magnetic field
            magnetic_field = None
            if self.rng.random() < 0.3:  # 30% chance
                magnetic_field = float(self.rng.uniform(1e6, 1e9))  # 1T to 1000T

            grid_points = int(self.rng.choice(grid_points_options))
            kappa_method = self.rng.choice(kappa_methods)
            graybody_method = self.rng.choice(graybody_methods)

            config = ParameterConfiguration(
                config_id=config_id,
                plasma_density=plasma_density,
                laser_wavelength=laser_wavelength,
                laser_intensity=laser_intensity,
                temperature_constant=temperature_constant,
                magnetic_field=magnetic_field,
                grid_points=grid_points,
                kappa_method=kappa_method,
                graybody_method=graybody_method
            )

            configs.append(config)

        return configs


class StressTestRunner:
    """Main stress testing execution engine"""

    def __init__(self, config: StressTestConfig):
        self.config = config
        self.results: List[ConfigurationResult] = []
        self.profiler = MemoryProfiler()
        self.param_generator = ParameterGenerator(config.seed)

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.start_time = 0.0
        self.total_execution_time = 0.0

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"stress_test_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def run_configuration(self, config: ParameterConfiguration) -> ConfigurationResult:
        """Run a single configuration with profiling"""
        config_id = config.config_id
        self.logger.info(f"Running configuration: {config_id}")

        # Start profiling
        if self.config.enable_profiling:
            self.profiler.start_profiling()

        result = ConfigurationResult(
            config_id=config_id,
            success=False,
            execution_time=0.0,
            memory_peak_mb=0.0,
            memory_avg_mb=0.0
        )

        try:
            start_time = time.time()

            # Sample memory periodically during execution
            def memory_monitor():
                while time.time() - start_time < self.config.timeout_per_config:
                    if self.config.enable_profiling:
                        self.profiler.sample_memory()
                    time.sleep(0.1)  # Sample every 100ms

            # Start memory monitor in background
            import threading
            monitor_thread = threading.Thread(target=memory_monitor, daemon=True)
            monitor_thread.start()

            # Run the pipeline
            summary = run_full_pipeline(
                plasma_density=config.plasma_density,
                laser_wavelength=config.laser_wavelength,
                laser_intensity=config.laser_intensity,
                temperature_constant=config.temperature_constant,
                magnetic_field=config.magnetic_field,
                grid_points=config.grid_points,
                kappa_method=config.kappa_method,
                graybody=config.graybody_method,
                save_graybody_figure=False,  # Don't save figures during stress test
                respect_parametric_bounds=True
            )

            execution_time = time.time() - start_time

            # Get memory metrics
            if self.config.enable_profiling:
                peak_memory, avg_memory = self.profiler.stop_profiling()
            else:
                peak_memory, avg_memory = 0.0, 0.0

            # Check if execution was successful
            success = (summary.kappa is not None and len(summary.kappa) > 0 and
                      execution_time < self.config.timeout_per_config)

            # Extract performance metrics
            performance_metrics = {}
            if success and summary.kappa:
                performance_metrics = {
                    'kappa_value': summary.kappa[0],
                    'horizon_count': len(summary.horizon_positions),
                    'peak_frequency': summary.spectrum_peak_frequency or 0.0,
                    'detection_time': summary.t5sigma_s or float('inf'),
                    'hawking_temp': summary.T_H_K or 0.0
                }

            result.success = success
            result.execution_time = execution_time
            result.memory_peak_mb = peak_memory
            result.memory_avg_mb = avg_memory
            result.summary = summary
            result.performance_metrics = performance_metrics

            self.logger.info(f"Configuration {config_id} completed in {execution_time:.2f}s, "
                           f"success: {success}, peak memory: {peak_memory:.1f}MB")

        except Exception as e:
            execution_time = time.time() - start_time
            result.success = False
            result.execution_time = execution_time
            result.error_message = str(e)

            if self.config.enable_profiling:
                result.memory_peak_mb, result.memory_avg_mb = self.profiler.stop_profiling()

            self.logger.error(f"Configuration {config_id} failed after {execution_time:.2f}s: {e}")

        return result

    def run_stress_test(self) -> StressTestSummary:
        """Run the complete stress test suite"""
        self.logger.info(f"Starting stress test with {self.config.sweep_size} configurations")
        self.start_time = time.time()

        # Generate parameter configurations
        self.logger.info("Generating parameter configurations...")
        configurations = self.param_generator.generate_configurations(self.config.sweep_size)

        # Save configurations
        configs_file = self.output_dir / f"configurations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(configs_file, 'w') as f:
            json.dump([config.to_dict() for config in configurations], f, indent=2)

        # Run configurations
        if self.config.test_concurrent_execution:
            self.logger.info(f"Running {self.config.sweep_size} configurations with {self.config.concurrent_workers} workers")
            self._run_concurrent_configurations(configurations)
        else:
            self.logger.info(f"Running {self.config.sweep_size} configurations sequentially")
            self._run_sequential_configurations(configurations)

        self.total_execution_time = time.time() - self.start_time

        # Generate summary and analysis
        summary = self._generate_summary()

        # Save results
        self._save_results(summary)

        # Generate visualizations
        if self.config.generate_visualizations:
            self._generate_visualizations()

        self.logger.info(f"Stress test completed in {self.total_execution_time:.2f}s")
        return summary

    def _run_sequential_configurations(self, configurations: List[ParameterConfiguration]):
        """Run configurations sequentially"""
        for i, config in enumerate(configurations):
            self.logger.info(f"Progress: {i+1}/{len(configurations)}")
            result = self.run_configuration(config)
            self.results.append(result)

            # Check memory threshold
            if result.memory_peak_mb > self.config.memory_threshold_mb:
                self.logger.warning(f"Memory threshold exceeded: {result.memory_peak_mb:.1f}MB > {self.config.memory_threshold_mb}MB")

    def _run_concurrent_configurations(self, configurations: List[ParameterConfiguration]):
        """Run configurations concurrently"""
        with ThreadPoolExecutor(max_workers=self.config.concurrent_workers) as executor:
            # Submit all configurations
            future_to_config = {
                executor.submit(self.run_configuration, config): config
                for config in configurations
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_config):
                result = future.result()
                self.results.append(result)
                completed += 1

                self.logger.info(f"Progress: {completed}/{len(configurations)} completed")

                # Check memory threshold
                if result.memory_peak_mb > self.config.memory_threshold_mb:
                    self.logger.warning(f"Memory threshold exceeded for {result.config_id}: "
                                      f"{result.memory_peak_mb:.1f}MB > {self.config.memory_threshold_mb}MB")

    def _generate_summary(self) -> StressTestSummary:
        """Generate comprehensive stress test summary"""
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        success_rate = len(successful) / len(self.results) if self.results else 0.0

        # Calculate performance metrics
        if successful:
            avg_execution_time = np.mean([r.execution_time for r in successful])
            memory_peak = np.max([r.memory_peak_mb for r in self.results])
            memory_avg = np.mean([r.memory_avg_mb for r in self.results])
        else:
            avg_execution_time = 0.0
            memory_peak = 0.0
            memory_avg = 0.0

        # Scalability metrics
        scalability_metrics = self._calculate_scalability_metrics()

        # Performance regression detection
        regression_detected = self._detect_performance_regressions()

        # Generate recommendations and identify issues
        recommendations = self._generate_recommendations(successful, failed)
        critical_issues = self._identify_critical_issues(failed)

        return StressTestSummary(
            test_timestamp=datetime.now().isoformat(),
            total_configurations=len(self.results),
            successful_configurations=len(successful),
            failed_configurations=len(failed),
            success_rate=success_rate,
            total_execution_time=self.total_execution_time,
            average_execution_time=avg_execution_time,
            memory_peak_mb=memory_peak,
            memory_avg_mb=memory_avg,
            concurrent_workers_used=self.config.concurrent_workers,
            performance_regression_detected=regression_detected,
            scalability_metrics=scalability_metrics,
            recommendations=recommendations,
            critical_issues=critical_issues
        )

    def _calculate_scalability_metrics(self) -> Dict[str, float]:
        """Calculate scalability and performance metrics"""
        if not self.results:
            return {}

        successful = [r for r in self.results if r.success]
        if not successful:
            return {}

        # Execution time statistics
        exec_times = [r.execution_time for r in successful]
        memory_usage = [r.memory_peak_mb for r in self.results]

        # Calculate throughput (configurations per hour)
        throughput = len(self.results) / (self.total_execution_time / 3600) if self.total_execution_time > 0 else 0

        # Memory efficiency
        max_memory = np.max(memory_usage)
        memory_per_config = max_memory / len(self.results) if len(self.results) > 0 else 0

        # Performance consistency (coefficient of variation)
        performance_consistency = np.std(exec_times) / np.mean(exec_times) if np.mean(exec_times) > 0 else float('inf')

        return {
            'throughput_configs_per_hour': throughput,
            'max_memory_mb': max_memory,
            'memory_per_config_mb': memory_per_config,
            'performance_consistency_cv': performance_consistency,
            'execution_time_std': np.std(exec_times),
            'execution_time_range': np.max(exec_times) - np.min(exec_times)
        }

    def _detect_performance_regressions(self) -> bool:
        """Detect performance regressions compared to baselines"""
        if not self.results:
            return False

        # Simple regression detection based on success rate and execution times
        successful = [r for r in self.results if r.success]

        # Check for low success rate (regression if < 80%)
        success_rate = len(successful) / len(self.results)
        if success_rate < 0.8:
            return True

        # Check for unusually long execution times
        if successful:
            exec_times = [r.execution_time for r in successful]
            if np.mean(exec_times) > 60:  # Average > 60 seconds indicates regression
                return True

        # Check for memory issues
        memory_usage = [r.memory_peak_mb for r in self.results]
        if np.max(memory_usage) > self.config.memory_threshold_mb:
            return True

        return False

    def _generate_recommendations(self, successful: List[ConfigurationResult],
                                failed: List[ConfigurationResult]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Success rate recommendations
        success_rate = len(successful) / len(self.results) if self.results else 0
        if success_rate < 0.9:
            recommendations.append(f"Success rate ({success_rate:.1%}) is below target (>90%). "
                                 "Consider parameter space restrictions or error handling improvements.")

        # Performance recommendations
        if successful:
            exec_times = [r.execution_time for r in successful]
            avg_time = np.mean(exec_times)
            if avg_time > 30:
                recommendations.append(f"Average execution time ({avg_time:.1f}s) is high. "
                                     "Consider optimization of physics calculations or parameter bounds.")

        # Memory recommendations
        if self.results:
            memory_usage = [r.memory_peak_mb for r in self.results]
            max_memory = np.max(memory_usage)
            if max_memory > self.config.memory_threshold_mb * 0.8:  # 80% of threshold
                recommendations.append(f"Peak memory usage ({max_memory:.1f}MB) is approaching limits. "
                                     "Consider memory optimization or reducing concurrent workers.")

        # Concurrency recommendations
        if self.config.concurrent_workers > 1 and successful:
            # Compare sequential vs concurrent performance
            avg_time_concurrent = np.mean([r.execution_time for r in successful])
            estimated_sequential_time = avg_time_concurrent * len(successful)
            actual_total_time = self.total_execution_time

            if actual_total_time > estimated_sequential_time * 0.8:
                recommendations.append("Concurrent execution may not be providing expected speedup. "
                                     "Consider reducing worker count or optimizing for parallelization.")

        # Failure analysis recommendations
        if failed:
            error_types = {}
            for result in failed:
                error_type = "timeout" if result.execution_time >= self.config.timeout_per_config else "other"
                error_types[error_type] = error_types.get(error_type, 0) + 1

            if error_types.get("timeout", 0) > len(failed) * 0.5:
                recommendations.append("Many configurations are timing out. "
                                     "Consider increasing timeout or reducing parameter complexity.")

        return recommendations

    def _identify_critical_issues(self, failed: List[ConfigurationResult]) -> List[str]:
        """Identify critical issues from failed configurations"""
        issues = []

        if not failed:
            return issues

        # Categorize failures
        timeouts = [r for r in failed if r.execution_time >= self.config.timeout_per_config]
        memory_issues = [r for r in failed if r.memory_peak_mb > self.config.memory_threshold_mb]
        other_failures = [r for r in failed if r not in timeouts and r not in memory_issues]

        if timeouts:
            issues.append(f"{len(timeouts)} configurations timed out ({self.config.timeout_per_config}s limit)")

        if memory_issues:
            issues.append(f"{len(memory_issues)} configurations exceeded memory threshold "
                         f"({self.config.memory_threshold_mb}MB)")

        if other_failures:
            issues.append(f"{len(other_failures)} configurations failed with errors")

        # Check for systemic issues
        if len(failed) > len(self.results) * 0.5:  # >50% failure rate
            issues.append("High failure rate indicates potential systemic issues")

        return issues

    def _save_results(self, summary: StressTestSummary):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"stress_test_results_{timestamp}.json"
        results_data = {
            'summary': asdict(summary),
            'configurations': [asdict(result) for result in self.results],
            'test_config': asdict(self.config)
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save summary separately
        summary_file = self.output_dir / f"stress_test_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(asdict(summary), f, indent=2)

        self.logger.info(f"Results saved to {results_file}")

    def _generate_visualizations(self):
        """Generate comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Prepare data
        successful = [r for r in self.results if r.success]
        failed = [r for r in self.results if not r.success]

        if not self.results:
            self.logger.warning("No results to visualize")
            return

        # 1. Success rate pie chart
        plt.figure(figsize=(10, 6))
        success_counts = {'Successful': len(successful), 'Failed': len(failed)}
        plt.pie(success_counts.values(), labels=success_counts.keys(), autopct='%1.1f%%',
                colors=['#2ecc71', '#e74c3c'])
        plt.title('Stress Test Success Rate')
        plt.savefig(viz_dir / f"success_rate_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Execution time distribution
        if successful:
            plt.figure(figsize=(12, 6))
            exec_times = [r.execution_time for r in successful]

            plt.subplot(1, 2, 1)
            plt.hist(exec_times, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            plt.xlabel('Execution Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('Execution Time Distribution')

            plt.subplot(1, 2, 2)
            plt.boxplot(exec_times)
            plt.ylabel('Execution Time (seconds)')
            plt.title('Execution Time Box Plot')

            plt.tight_layout()
            plt.savefig(viz_dir / f"execution_times_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 3. Memory usage analysis
        if self.results:
            plt.figure(figsize=(12, 6))
            memory_usage = [r.memory_peak_mb for r in self.results]

            plt.subplot(1, 2, 1)
            plt.hist(memory_usage, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
            plt.xlabel('Peak Memory Usage (MB)')
            plt.ylabel('Frequency')
            plt.title('Memory Usage Distribution')

            # Add threshold line
            plt.axvline(x=self.config.memory_threshold_mb, color='r', linestyle='--',
                       label=f'Threshold ({self.config.memory_threshold_mb}MB)')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.scatter(range(len(self.results)), memory_usage, alpha=0.6, s=20)
            plt.axhline(y=self.config.memory_threshold_mb, color='r', linestyle='--',
                       label=f'Threshold ({self.config.memory_threshold_mb}MB)')
            plt.xlabel('Configuration Index')
            plt.ylabel('Peak Memory Usage (MB)')
            plt.title('Memory Usage by Configuration')
            plt.legend()

            plt.tight_layout()
            plt.savefig(viz_dir / f"memory_usage_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()

        # 4. Performance metrics scatter plots
        if successful:
            performance_data = []
            for r in successful:
                if r.performance_metrics:
                    performance_data.append({
                        'execution_time': r.execution_time,
                        'memory_mb': r.memory_peak_mb,
                        'kappa': r.performance_metrics.get('kappa_value', 0),
                        'detection_time': r.performance_metrics.get('detection_time', float('inf')),
                        'horizon_count': r.performance_metrics.get('horizon_count', 0)
                    })

            if performance_data:
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))

                # Execution time vs Memory
                exec_times = [d['execution_time'] for d in performance_data]
                memory_vals = [d['memory_mb'] for d in performance_data]
                axes[0, 0].scatter(exec_times, memory_vals, alpha=0.6)
                axes[0, 0].set_xlabel('Execution Time (s)')
                axes[0, 0].set_ylabel('Memory Usage (MB)')
                axes[0, 0].set_title('Execution Time vs Memory Usage')

                # Kappa values
                kappas = [d['kappa'] for d in performance_data if d['kappa'] > 0]
                if kappas:
                    axes[0, 1].hist(kappas, bins=20, alpha=0.7, color='#e67e22')
                    axes[0, 1].set_xlabel('Kappa Value (1/s)')
                    axes[0, 1].set_ylabel('Frequency')
                    axes[0, 1].set_title('Kappa Value Distribution')

                # Detection times
                detection_times = [d['detection_time'] for d in performance_data
                                 if d['detection_time'] != float('inf')]
                if detection_times:
                    axes[1, 0].hist(detection_times, bins=20, alpha=0.7, color='#27ae60')
                    axes[1, 0].set_xlabel('Detection Time (s)')
                    axes[1, 0].set_ylabel('Frequency')
                    axes[1, 0].set_title('Detection Time Distribution')

                # Horizon counts
                horizon_counts = [d['horizon_count'] for d in performance_data]
                axes[1, 1].hist(horizon_counts, bins=range(max(horizon_counts)+2), alpha=0.7, color='#f39c12')
                axes[1, 1].set_xlabel('Number of Horizons')
                axes[1, 1].set_ylabel('Frequency')
                axes[1, 1].set_title('Horizon Count Distribution')

                plt.tight_layout()
                plt.savefig(viz_dir / f"performance_metrics_{timestamp}.png", dpi=300, bbox_inches='tight')
                plt.close()

        # 5. Scalability analysis
        if self.config.test_concurrent_execution and successful:
            plt.figure(figsize=(10, 6))

            # Calculate theoretical vs actual speedup
            sequential_time_estimate = np.mean([r.execution_time for r in successful]) * len(successful)
            actual_time = self.total_execution_time
            theoretical_speedup = len(self.config.concurrent_workers)
            actual_speedup = sequential_time_estimate / actual_time if actual_time > 0 else 0

            categories = ['Theoretical', 'Actual']
            speedups = [theoretical_speedup, actual_speedup]

            bars = plt.bar(categories, speedups, color=['#3498db', '#e74c3c'], alpha=0.7)
            plt.ylabel('Speedup Factor')
            plt.title(f'Concurrency Analysis ({self.config.concurrent_workers} workers)')
            plt.ylim(0, max(speedups) * 1.2)

            # Add value labels on bars
            for bar, speedup in zip(bars, speedups):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{speedup:.2f}x', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(viz_dir / f"scalability_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info(f"Visualizations saved to {viz_dir}")


def main():
    """Main function for running stress tests"""
    import argparse

    parser = argparse.ArgumentParser(description="Large-Scale Parameter Sweep Stress Testing")
    parser.add_argument("--sweep-size", type=int, default=100, help="Number of configurations to test")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    parser.add_argument("--memory-threshold", type=float, default=8192, help="Memory threshold in MB")
    parser.add_argument("--timeout", type=float, default=300, help="Timeout per configuration in seconds")
    parser.add_argument("--no-profiling", action="store_true", help="Disable memory profiling")
    parser.add_argument("--no-concurrent", action="store_true", help="Disable concurrent execution")
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization generation")
    parser.add_argument("--output-dir", type=str, default="results/stress_testing", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Create configuration
    config = StressTestConfig(
        sweep_size=args.sweep_size,
        concurrent_workers=args.workers,
        memory_threshold_mb=args.memory_threshold,
        timeout_per_config=args.timeout,
        enable_profiling=not args.no_profiling,
        test_concurrent_execution=not args.no_concurrent,
        generate_visualizations=not args.no_viz,
        output_dir=args.output_dir,
        seed=args.seed
    )

    print("🚀 Starting Large-Scale Parameter Sweep Stress Testing")
    print(f"Configuration: {config.sweep_size} configurations, {config.concurrent_workers} workers")
    print(f"Memory threshold: {config.memory_threshold_mb}MB, Timeout: {config.timeout_per_config}s")
    print(f"Output directory: {config.output_dir}")

    # Run stress test
    runner = StressTestRunner(config)
    summary = runner.run_stress_test()

    # Print results
    print(f"\n{'='*60}")
    print("STRESS TEST RESULTS")
    print(f"{'='*60}")
    print(f"Total Configurations: {summary.total_configurations}")
    print(f"Successful: {summary.successful_configurations} ({summary.success_rate:.1%})")
    print(f"Failed: {summary.failed_configurations}")
    print(f"Total Execution Time: {summary.total_execution_time:.2f}s")
    print(f"Average Execution Time: {summary.average_execution_time:.2f}s")
    print(f"Peak Memory Usage: {summary.memory_peak_mb:.1f}MB")
    print(f"Performance Regression Detected: {summary.performance_regression_detected}")

    print(f"\nScalability Metrics:")
    for metric, value in summary.scalability_metrics.items():
        print(f"  {metric}: {value:.3f}")

    if summary.recommendations:
        print(f"\nRecommendations:")
        for rec in summary.recommendations:
            print(f"  • {rec}")

    if summary.critical_issues:
        print(f"\nCritical Issues:")
        for issue in summary.critical_issues:
            print(f"  ❌ {issue}")

    print(f"\nDetailed results saved to: {config.output_dir}")
    print(f"{'='*60}")

    # Exit with appropriate code
    if summary.success_rate < 0.8:
        print("❌ Stress test indicates performance issues")
        sys.exit(1)
    elif summary.performance_regression_detected:
        print("⚠️  Performance regressions detected")
        sys.exit(2)
    else:
        print("✅ Stress test passed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()