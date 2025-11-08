#!/usr/bin/env python3
"""
Advanced Memory Profiling and Performance Benchmarking System

This module provides comprehensive memory profiling, performance benchmarking,
and regression detection for the Analog Hawking Radiation Analysis project.

Features:
- Real-time memory monitoring and profiling
- Performance benchmarking with baselines
- Regression detection and alerting
- Detailed memory leak detection
- Resource usage analysis and optimization recommendations
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import threading
import traceback
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np
import psutil
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import tracemalloc

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')
sns.set_style("whitegrid")


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot"""
    timestamp: float
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    percent: float  # Memory usage percentage
    available_mb: float
    heap_size_mb: float = 0.0  # Python heap size
    object_count: int = 0  # Number of Python objects
    gc_collections: int = 0  # Number of garbage collections


@dataclass
class PerformanceBenchmark:
    """Performance benchmark results"""
    name: str
    execution_time: float
    memory_peak_mb: float
    memory_avg_mb: float
    memory_leaked_mb: float
    cpu_time: float
    sys_time: float
    success: bool
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegressionAlert:
    """Performance regression alert"""
    timestamp: float
    benchmark_name: str
    metric_name: str
    current_value: float
    baseline_value: float
    regression_percentage: float
    severity: str  # "low", "medium", "high", "critical"
    description: str


class MemoryProfiler:
    """Advanced memory profiling system"""

    def __init__(self, sample_interval: float = 0.1, max_samples: int = 10000):
        self.sample_interval = sample_interval
        self.max_samples = max_samples
        self.process = psutil.Process()

        # Profiling state
        self.is_profiling = False
        self.profiler_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Memory data
        self.snapshots: deque = deque(maxlen=max_samples)
        self.peak_memory = 0.0
        self.memory_growth_rate = 0.0

        # Python-specific tracking
        self.tracemalloc_enabled = False
        self.gc_stats_before: Optional[Dict[str, int]] = None

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Setup logging for memory profiler"""
        log_dir = Path("results/memory_profiling")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"memory_profiler_{timestamp}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger = logging.getLogger(f"{__name__}.MemoryProfiler")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    def start_profiling(self, enable_tracemalloc: bool = True):
        """Start memory profiling"""
        if self.is_profiling:
            self.logger.warning("Memory profiling already active")
            return

        self.is_profiling = True
        self.stop_event.clear()
        self.snapshots.clear()
        self.peak_memory = 0.0

        # Enable tracemalloc for detailed tracking
        if enable_tracemalloc:
            tracemalloc.start()
            self.tracemalloc_enabled = True
            self.logger.info("Enabled tracemalloc for detailed memory tracking")

        # Record initial state
        self.gc_stats_before = gc.get_stats()
        gc.collect()  # Force garbage collection before profiling

        # Start profiling thread
        self.profiler_thread = threading.Thread(target=self._profiling_loop, daemon=True)
        self.profiler_thread.start()

        self.logger.info("Started memory profiling")

    def stop_profiling(self) -> Tuple[float, float, float]:
        """Stop profiling and return memory metrics"""
        if not self.is_profiling:
            return 0.0, 0.0, 0.0

        self.is_profiling = False
        self.stop_event.set()

        if self.profiler_thread:
            self.profiler_thread.join(timeout=5.0)

        # Get final statistics
        gc_stats_after = gc.get_stats()
        gc_collections = sum(
            stats.get('collections', 0) - self.gc_stats_before[i].get('collections', 0)
            for i, stats in enumerate(gc_stats_after)
        ) if self.gc_stats_before else 0

        # Calculate memory metrics
        if self.snapshots:
            memory_values = [s.rss_mb for s in self.snapshots]
            peak_memory = max(memory_values)
            avg_memory = np.mean(memory_values)

            # Calculate memory growth rate (MB per second)
            if len(self.snapshots) > 1:
                time_span = self.snapshots[-1].timestamp - self.snapshots[0].timestamp
                memory_growth = (self.snapshots[-1].rss_mb - self.snapshots[0].rss_mb)
                self.memory_growth_rate = memory_growth / time_span if time_span > 0 else 0.0
        else:
            peak_memory = avg_memory = 0.0

        # Disable tracemalloc
        if self.tracemalloc_enabled:
            current, peak = tracemalloc.get_traced_memory()
            self.logger.info(f"Tracemalloc: current={current/1024/1024:.1f}MB, peak={peak/1024/1024:.1f}MB")
            tracemalloc.stop()
            self.tracemalloc_enabled = False

        self.logger.info(f"Stopped profiling: peak={peak_memory:.1f}MB, avg={avg_memory:.1f}MB, "
                        f"growth_rate={self.memory_growth_rate:.3f}MB/s")

        return peak_memory, avg_memory, self.memory_growth_rate

    def _profiling_loop(self):
        """Main profiling loop"""
        while not self.stop_event.is_set():
            try:
                snapshot = self._take_snapshot()
                self.snapshots.append(snapshot)
                self.peak_memory = max(self.peak_memory, snapshot.rss_mb)

                # Check for memory leaks
                if len(self.snapshots) > 100:  # Check after enough samples
                    self._check_memory_leaks()

            except Exception as e:
                self.logger.error(f"Error in profiling loop: {e}")

            self.stop_event.wait(self.sample_interval)

    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        try:
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()

            # Get system memory info
            sys_memory = psutil.virtual_memory()

            # Get Python heap size if tracemalloc is enabled
            heap_size = 0
            object_count = 0
            if self.tracemalloc_enabled:
                current, peak = tracemalloc.get_traced_memory()
                heap_size = current / (1024 * 1024)

            # Get object count
            object_count = len(gc.get_objects())

            # Get GC collection count
            gc_collections = sum(stats.get('collections', 0) for stats in gc.get_stats())

            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=memory_info.rss / (1024 * 1024),
                vms_mb=memory_info.vms / (1024 * 1024),
                percent=memory_percent,
                available_mb=sys_memory.available / (1024 * 1024),
                heap_size_mb=heap_size,
                object_count=object_count,
                gc_collections=gc_collections
            )

        except Exception as e:
            self.logger.error(f"Error taking memory snapshot: {e}")
            return MemorySnapshot(
                timestamp=time.time(),
                rss_mb=0, vms_mb=0, percent=0, available_mb=0
            )

    def _check_memory_leaks(self):
        """Check for potential memory leaks"""
        if len(self.snapshots) < 100:
            return

        # Get recent memory samples
        recent_samples = list(self.snapshots)[-100:]
        memory_values = [s.rss_mb for s in recent_samples]

        # Check for consistent memory growth
        if len(memory_values) > 20:
            # Calculate trend using linear regression
            x = np.arange(len(memory_values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, memory_values)

            # If slope is significantly positive and R² is high, potential leak
            if slope > 0.1 and r_value**2 > 0.7:  # Growth > 0.1MB/sample and strong correlation
                growth_per_second = slope / self.sample_interval
                if growth_per_second > 1.0:  # More than 1MB/second growth
                    self.logger.warning(f"Potential memory leak detected: growth rate = {growth_per_second:.3f}MB/s")

    def get_memory_timeline(self) -> List[Tuple[float, float]]:
        """Get memory usage timeline"""
        return [(s.timestamp, s.rss_mb) for s in self.snapshots]

    def get_memory_statistics(self) -> Dict[str, float]:
        """Get comprehensive memory statistics"""
        if not self.snapshots:
            return {}

        memory_values = [s.rss_mb for s in self.snapshots]
        heap_values = [s.heap_size_mb for s in self.snapshots if s.heap_size_mb > 0]

        stats_dict = {
            'peak_memory_mb': max(memory_values),
            'avg_memory_mb': np.mean(memory_values),
            'min_memory_mb': min(memory_values),
            'std_memory_mb': np.std(memory_values),
            'memory_growth_rate_mb_per_s': self.memory_growth_rate,
            'total_samples': len(self.snapshots),
            'profiling_duration_s': self.snapshots[-1].timestamp - self.snapshots[0].timestamp if len(self.snapshots) > 1 else 0
        }

        if heap_values:
            stats_dict.update({
                'peak_heap_mb': max(heap_values),
                'avg_heap_mb': np.mean(heap_values),
                'heap_efficiency': np.mean(heap_values) / np.mean(memory_values) if np.mean(memory_values) > 0 else 0
            })

        return stats_dict


class PerformanceBenchmarkRunner:
    """Performance benchmarking system with regression detection"""

    def __init__(self, baseline_file: Optional[str] = None):
        self.baseline_file = baseline_file or "results/baselines/performance_baselines.json"
        self.benchmarks: Dict[str, PerformanceBenchmark] = {}
        self.baselines: Dict[str, Dict[str, float]] = {}
        self.regression_alerts: List[RegressionAlert] = []

        # Load existing baselines
        self._load_baselines()

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

    def _setup_logging(self):
        """Setup logging for benchmarking"""
        log_dir = Path("results/benchmarking")
        log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"benchmark_{timestamp}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger = logging.getLogger(f"{__name__}.BenchmarkRunner")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    def _load_baselines(self):
        """Load performance baselines from file"""
        try:
            if Path(self.baseline_file).exists():
                with open(self.baseline_file, 'r') as f:
                    self.baselines = json.load(f)
                self.logger.info(f"Loaded {len(self.baselines)} baseline benchmarks")
        except Exception as e:
            self.logger.warning(f"Could not load baselines: {e}")
            self.baselines = {}

    def _save_baselines(self):
        """Save performance baselines to file"""
        try:
            Path(self.baseline_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.baseline_file, 'w') as f:
                json.dump(self.baselines, f, indent=2)
            self.logger.info(f"Saved {len(self.baselines)} baseline benchmarks")
        except Exception as e:
            self.logger.error(f"Could not save baselines: {e}")

    def run_benchmark(self, name: str, func: Callable, *args, **kwargs) -> PerformanceBenchmark:
        """Run a performance benchmark"""
        self.logger.info(f"Running benchmark: {name}")

        # Initialize memory profiler
        profiler = MemoryProfiler()
        profiler.start_profiling()

        # Track time and resources
        start_time = time.perf_counter()
        start_cpu = time.process_time()

        try:
            # Run the benchmark function
            result = func(*args, **kwargs)

            # Force garbage collection
            gc.collect()

            execution_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_cpu
            sys_time = 0.0  # Could be tracked separately if needed

            # Stop profiling and get memory metrics
            peak_memory, avg_memory, growth_rate = profiler.stop_profiling()

            # Calculate memory leak (growth during execution)
            memory_leaked = max(0, growth_rate * execution_time)

            # Extract performance metrics from result if available
            metrics = {}
            if isinstance(result, dict):
                metrics = {k: v for k, v in result.items() if isinstance(v, (int, float))}

            benchmark = PerformanceBenchmark(
                name=name,
                execution_time=execution_time,
                memory_peak_mb=peak_memory,
                memory_avg_mb=avg_memory,
                memory_leaked_mb=memory_leaked,
                cpu_time=cpu_time,
                sys_time=sys_time,
                success=True,
                metrics=metrics
            )

            self.logger.info(f"Benchmark {name} completed: {execution_time:.3f}s, "
                           f"peak={peak_memory:.1f}MB, leaked={memory_leaked:.1f}MB")

        except Exception as e:
            execution_time = time.perf_counter() - start_time
            cpu_time = time.process_time() - start_cpu

            peak_memory, avg_memory, growth_rate = profiler.stop_profiling()
            memory_leaked = max(0, growth_rate * execution_time)

            benchmark = PerformanceBenchmark(
                name=name,
                execution_time=execution_time,
                memory_peak_mb=peak_memory,
                memory_avg_mb=avg_memory,
                memory_leaked_mb=memory_leaked,
                cpu_time=cpu_time,
                sys_time=0.0,
                success=False,
                error_message=str(e)
            )

            self.logger.error(f"Benchmark {name} failed after {execution_time:.3f}s: {e}")

        # Store benchmark
        self.benchmarks[name] = benchmark

        # Check for regressions
        self._check_regression(name, benchmark)

        return benchmark

    def _check_regression(self, name: str, benchmark: PerformanceBenchmark):
        """Check for performance regressions compared to baselines"""
        if name not in self.baselines:
            # No baseline available, save current as baseline
            self.baselines[name] = {
                'execution_time': benchmark.execution_time,
                'memory_peak_mb': benchmark.memory_peak_mb,
                'memory_avg_mb': benchmark.memory_avg_mb,
                'memory_leaked_mb': benchmark.memory_leaked_mb,
                'cpu_time': benchmark.cpu_time
            }
            self.logger.info(f"Established baseline for {name}")
            return

        baseline = self.baselines[name]
        regressions = []

        # Check execution time regression
        if benchmark.execution_time > baseline['execution_time']:
            regression_pct = ((benchmark.execution_time - baseline['execution_time']) /
                            baseline['execution_time']) * 100
            if regression_pct > 10:  # 10% threshold
                severity = self._calculate_severity(regression_pct)
                regressions.append(RegressionAlert(
                    timestamp=time.time(),
                    benchmark_name=name,
                    metric_name='execution_time',
                    current_value=benchmark.execution_time,
                    baseline_value=baseline['execution_time'],
                    regression_percentage=regression_pct,
                    severity=severity,
                    description=f"Execution time increased by {regression_pct:.1f}%"
                ))

        # Check memory regression
        for metric in ['memory_peak_mb', 'memory_avg_mb', 'memory_leaked_mb']:
            if getattr(benchmark, metric) > baseline[metric]:
                regression_pct = ((getattr(benchmark, metric) - baseline[metric]) /
                                max(baseline[metric], 0.1)) * 100
                if regression_pct > 20:  # 20% threshold for memory
                    severity = self._calculate_severity(regression_pct)
                    regressions.append(RegressionAlert(
                        timestamp=time.time(),
                        benchmark_name=name,
                        metric_name=metric,
                        current_value=getattr(benchmark, metric),
                        baseline_value=baseline[metric],
                        regression_percentage=regression_pct,
                        severity=severity,
                        description=f"{metric} increased by {regression_pct:.1f}%"
                    ))

        # Add alerts and log
        for alert in regressions:
            self.regression_alerts.append(alert)
            level = {
                'low': logging.WARNING,
                'medium': logging.ERROR,
                'high': logging.ERROR,
                'critical': logging.CRITICAL
            }.get(alert.severity, logging.WARNING)

            self.logger.log(level, f"PERFORMANCE REGRESSION: {alert.description}")

    def _calculate_severity(self, regression_pct: float) -> str:
        """Calculate regression severity based on percentage"""
        if regression_pct > 100:
            return "critical"
        elif regression_pct > 50:
            return "high"
        elif regression_pct > 25:
            return "medium"
        else:
            return "low"

    def update_baselines(self, benchmark_names: Optional[List[str]] = None):
        """Update baselines with current benchmark results"""
        names = benchmark_names or list(self.benchmarks.keys())

        for name in names:
            if name in self.benchmarks and self.benchmarks[name].success:
                benchmark = self.benchmarks[name]
                self.baselines[name] = {
                    'execution_time': benchmark.execution_time,
                    'memory_peak_mb': benchmark.memory_peak_mb,
                    'memory_avg_mb': benchmark.memory_avg_mb,
                    'memory_leaked_mb': benchmark.memory_leaked_mb,
                    'cpu_time': benchmark.cpu_time,
                    'timestamp': time.time()
                }
                self.logger.info(f"Updated baseline for {name}")

        self._save_baselines()

    def generate_report(self, output_dir: str = "results/benchmarking") -> str:
        """Generate comprehensive benchmark report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = Path(output_dir) / f"benchmark_report_{timestamp}.html"

        report_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate HTML report
        html_content = self._generate_html_report()

        with open(report_file, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Generated benchmark report: {report_file}")
        return str(report_file)

    def _generate_html_report(self) -> str:
        """Generate HTML benchmark report"""
        html = """
<!DOCTYPE html>
<html>
<head>
    <title>Performance Benchmark Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .benchmark { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .success { border-left: 5px solid #28a745; }
        .failure { border-left: 5px solid #dc3545; }
        .regression { border-left: 5px solid #ffc107; }
        table { width: 100%; border-collapse: collapse; margin: 10px 0; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .alert { padding: 10px; margin: 5px 0; border-radius: 3px; }
        .alert-critical { background-color: #f8d7da; border: 1px solid #f5c6cb; }
        .alert-high { background-color: #fff3cd; border: 1px solid #ffeaa7; }
        .alert-medium { background-color: #d1ecf1; border: 1px solid #bee5eb; }
        .alert-low { background-color: #d4edda; border: 1px solid #c3e6cb; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Performance Benchmark Report</h1>
        <p>Generated: {timestamp}</p>
    </div>
""".format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        # Add summary statistics
        total_benchmarks = len(self.benchmarks)
        successful = sum(1 for b in self.benchmarks.values() if b.success)
        failed = total_benchmarks - successful
        regressions = len(self.regression_alerts)

        html += f"""
    <div class="benchmark">
        <h2>Summary</h2>
        <table>
            <tr><th>Total Benchmarks</th><td>{total_benchmarks}</td></tr>
            <tr><th>Successful</th><td>{successful}</td></tr>
            <tr><th>Failed</th><td>{failed}</td></tr>
            <tr><th>Regressions</th><td>{regressions}</td></tr>
        </table>
    </div>
"""

        # Add regression alerts
        if self.regression_alerts:
            html += """
    <div class="benchmark regression">
        <h2>Performance Regressions</h2>
""".format()
            for alert in self.regression_alerts:
                html += f"""
        <div class="alert alert-{alert.severity}">
            <strong>{alert.benchmark_name}.{alert.metric_name}</strong>: {alert.description}<br>
            Current: {alert.current_value:.3f}, Baseline: {alert.baseline_value:.3f},
            Regression: {alert.regression_percentage:.1f}%
        </div>
"""
            html += "    </div>"

        # Add individual benchmarks
        for name, benchmark in self.benchmarks.items():
            css_class = "success" if benchmark.success else "failure"
            html += f"""
    <div class="benchmark {css_class}">
        <h3>{name}</h3>
        <table>
            <tr><th>Status</th><td>{"✅ Success" if benchmark.success else "❌ Failed"}</td></tr>
            <tr><th>Execution Time</th><td>{benchmark.execution_time:.3f}s</td></tr>
            <tr><th>Peak Memory</th><td>{benchmark.memory_peak_mb:.1f}MB</td></tr>
            <tr><th>Average Memory</th><td>{benchmark.memory_avg_mb:.1f}MB</td></tr>
            <tr><th>Memory Leaked</th><td>{benchmark.memory_leaked_mb:.1f}MB</td></tr>
            <tr><th>CPU Time</th><td>{benchmark.cpu_time:.3f}s</td></tr>
"""
            if benchmark.error_message:
                html += f'            <tr><th>Error</th><td>{benchmark.error_message}</td></tr>'

            html += "        </table>    </div>"

        html += """
</body>
</html>
"""
        return html

    def save_benchmarks(self, output_file: str):
        """Save benchmark results to file"""
        benchmark_data = {
            'timestamp': datetime.now().isoformat(),
            'benchmarks': {name: asdict(benchmark) for name, benchmark in self.benchmarks.items()},
            'regressions': [asdict(alert) for alert in self.regression_alerts],
            'baselines_used': self.baselines
        }

        with open(output_file, 'w') as f:
            json.dump(benchmark_data, f, indent=2)

        self.logger.info(f"Saved benchmark results to {output_file}")


def run_comprehensive_benchmarks():
    """Run comprehensive performance benchmarks"""
    print("🚀 Starting Comprehensive Performance Benchmarks")

    # Initialize benchmark runner
    runner = PerformanceBenchmarkRunner()

    # Test functions
    def test_basic_pipeline():
        """Benchmark basic pipeline execution"""
        from scripts.run_full_pipeline import run_full_pipeline
        return run_full_pipeline(
            demo=True,
            kappa_method="acoustic",
            graybody="dimensionless",
            save_graybody_figure=False
        )

    def test_memory_intensive_operation():
        """Benchmark memory-intensive operations"""
        # Create large arrays and perform computations
        arrays = []
        for i in range(10):
            arrays.append(np.random.random((1000, 1000)))

        # Perform matrix operations
        result = []
        for arr in arrays:
            result.append(np.linalg.svd(arr, compute_uv=False))

        return {"arrays_created": len(arrays), "svd_computed": len(result)}

    def test_parameter_sweep_small():
        """Benchmark small parameter sweep"""
        from stress_test_parameter_sweep import ParameterGenerator, StressTestRunner

        config = type('Config', (), {
            'sweep_size': 5,
            'concurrent_workers': 1,
            'memory_threshold_mb': 4096,
            'timeout_per_config': 60.0,
            'enable_profiling': True,
            'test_concurrent_execution': False,
            'generate_visualizations': False,
            'output_dir': 'results/benchmarking/temp',
            'seed': 42
        })()

        generator = ParameterGenerator(42)
        configurations = generator.generate_configurations(5)

        results = []
        for config in configurations:
            from scripts.run_full_pipeline import run_full_pipeline
            summary = run_full_pipeline(
                plasma_density=config.plasma_density,
                laser_wavelength=config.laser_wavelength,
                laser_intensity=config.laser_intensity,
                temperature_constant=config.temperature_constant,
                magnetic_field=config.magnetic_field,
                grid_points=config.grid_points,
                kappa_method=config.kappa_method,
                graybody=config.graybody_method,
                save_graybody_figure=False
            )
            results.append(summary)

        return {"configurations_processed": len(results), "successful": len([r for r in results if r.kappa])}

    def test_concurrent_execution():
        """Benchmark concurrent execution"""
        from concurrent.futures import ThreadPoolExecutor
        import time

        def cpu_bound_task():
            # Simulate CPU-bound work
            result = 0
            for i in range(1000000):
                result += i * i
            return result

        start_time = time.time()
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(cpu_bound_task) for _ in range(8)]
            results = [f.result() for f in futures]

        execution_time = time.time() - start_time
        return {"tasks_completed": len(results), "execution_time": execution_time}

    # Run benchmarks
    benchmarks = [
        ("basic_pipeline", test_basic_pipeline),
        ("memory_intensive", test_memory_intensive_operation),
        ("parameter_sweep_small", test_parameter_sweep_small),
        ("concurrent_execution", test_concurrent_execution)
    ]

    results = {}
    for name, func in benchmarks:
        print(f"Running benchmark: {name}")
        benchmark = runner.run_benchmark(name, func)
        results[name] = benchmark

        if benchmark.success:
            print(f"  ✅ {benchmark.execution_time:.3f}s, {benchmark.memory_peak_mb:.1f}MB peak")
        else:
            print(f"  ❌ Failed: {benchmark.error_message}")

    # Generate report
    output_dir = "results/benchmarking"
    report_file = runner.generate_report(output_dir)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(output_dir) / f"benchmark_results_{timestamp}.json"
    runner.save_benchmarks(str(results_file))

    # Print summary
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS SUMMARY")
    print(f"{'='*60}")

    successful = sum(1 for b in results.values() if b.success)
    total = len(results)
    regressions = len(runner.regression_alerts)

    print(f"Total Benchmarks: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {total - successful}")
    print(f"Regressions Detected: {regressions}")

    if runner.regression_alerts:
        print(f"\nRegressions:")
        for alert in runner.regression_alerts:
            print(f"  ❌ {alert.benchmark_name}.{alert.metric_name}: {alert.description}")

    print(f"\nReport saved to: {report_file}")
    print(f"Results saved to: {results_file}")
    print(f"{'='*60}")

    return results, runner.regression_alerts


def main():
    """Main function for memory profiling and benchmarking"""
    import argparse

    parser = argparse.ArgumentParser(description="Memory Profiling and Performance Benchmarking")
    parser.add_argument("--benchmark", action="store_true", help="Run comprehensive benchmarks")
    parser.add_argument("--update-baselines", action="store_true", help="Update performance baselines")
    parser.add_argument("--output-dir", type=str, default="results/benchmarking", help="Output directory")

    args = parser.parse_args()

    if args.benchmark:
        results, regressions = run_comprehensive_benchmarks()

        if args.update_baselines:
            runner = PerformanceBenchmarkRunner()
            runner.update_baselines()
            print("Updated performance baselines")

        # Exit with appropriate code
        if regressions:
            print("❌ Performance regressions detected")
            sys.exit(1)
        else:
            print("✅ All benchmarks passed")
            sys.exit(0)
    else:
        print("Memory Profiling and Performance Benchmarking System")
        print("Use --benchmark to run comprehensive benchmarks")
        print("Use --update-baselines to update performance baselines")


if __name__ == "__main__":
    main()