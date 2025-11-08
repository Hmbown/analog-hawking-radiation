#!/usr/bin/env python3
"""
Concurrent Execution Testing Framework

This framework tests the system's ability to handle concurrent pipeline executions,
validates thread safety, measures scalability, and identifies resource contention issues.

Features:
- Multi-threaded and multi-process execution testing
- Thread safety validation
- Scalability analysis and speedup measurement
- Resource contention detection
- Deadlock and race condition detection
- Performance impact analysis under concurrent load
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')
sns.set_style("whitegrid")


@dataclass
class ConcurrentTestConfig:
    """Configuration for concurrent execution tests"""
    max_workers: int = 8
    test_durations: List[int] = field(default_factory=lambda: [30, 60, 120])  # seconds
    worker_counts: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    thread_mode: bool = True  # True for threads, False for processes
    enable_monitoring: bool = True
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        'memory_limit_mb': 16384,  # 16GB
        'cpu_limit_percent': 95.0,
        'timeout_per_task': 300.0  # 5 minutes
    })
    output_dir: str = "results/concurrent_testing"


@dataclass
class WorkerResult:
    """Result from a single worker execution"""
    worker_id: int
    start_time: float
    end_time: float
    execution_time: float
    success: bool
    memory_peak_mb: float
    cpu_time: float
    error_message: Optional[str] = None
    task_results: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcurrentTestSummary:
    """Summary of concurrent execution test results"""
    test_type: str  # "thread" or "process"
    worker_count: int
    test_duration: float
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    success_rate: float
    throughput_tasks_per_second: float
    speedup_factor: float
    efficiency_percent: float
    memory_usage_stats: Dict[str, float]
    cpu_usage_stats: Dict[str, float]
    resource_contention_events: int
    deadlock_events: int
    race_condition_warnings: int
    scalability_metrics: Dict[str, float]
    recommendations: List[str]


class ResourceMonitor:
    """Monitor system resources during concurrent execution"""

    def __init__(self, sample_interval: float = 0.5):
        self.sample_interval = sample_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()

        # Resource data
        self.cpu_samples: List[float] = []
        self.memory_samples: List[float] = []
        self.timestamps: List[float] = []
        self.process_counts: List[int] = []
        self.thread_counts: List[int] = []

        # Contention detection
        self.contention_events = 0
        self.high_cpu_threshold = 90.0
        self.high_memory_threshold = 85.0

        # Setup logging
        self.logger = logging.getLogger(f"{__name__}.ResourceMonitor")

    def start_monitoring(self):
        """Start resource monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.stop_event.clear()
        self.cpu_samples.clear()
        self.memory_samples.clear()
        self.timestamps.clear()
        self.process_counts.clear()
        self.thread_counts.clear()
        self.contention_events = 0

        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()

        self.logger.info("Started resource monitoring")

    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return statistics"""
        if not self.is_monitoring:
            return {}

        self.is_monitoring = False
        self.stop_event.set()

        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)

        # Calculate statistics
        stats = {}
        if self.cpu_samples:
            stats.update({
                'cpu_avg': np.mean(self.cpu_samples),
                'cpu_max': np.max(self.cpu_samples),
                'cpu_std': np.std(self.cpu_samples),
                'cpu_samples': len(self.cpu_samples)
            })

        if self.memory_samples:
            stats.update({
                'memory_avg': np.mean(self.memory_samples),
                'memory_max': np.max(self.memory_samples),
                'memory_std': np.std(self.memory_samples),
                'memory_samples': len(self.memory_samples)
            })

        if self.timestamps and len(self.timestamps) > 1:
            duration = self.timestamps[-1] - self.timestamps[0]
            stats.update({
                'monitoring_duration': duration,
                'contention_events': self.contention_events
            })

        self.logger.info(f"Stopped monitoring: {stats}")
        return stats

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self.stop_event.is_set():
            try:
                timestamp = time.time()
                cpu_percent = psutil.cpu_percent(interval=None)
                memory_percent = psutil.virtual_memory().percent
                process_count = len(psutil.pids())
                thread_count = threading.active_count()

                self.timestamps.append(timestamp)
                self.cpu_samples.append(cpu_percent)
                self.memory_samples.append(memory_percent)
                self.process_counts.append(process_count)
                self.thread_counts.append(thread_count)

                # Check for resource contention
                if cpu_percent > self.high_cpu_threshold or memory_percent > self.high_memory_threshold:
                    self.contention_events += 1
                    if self.contention_events == 1:  # Log first occurrence
                        self.logger.warning(f"Resource contention detected: CPU={cpu_percent:.1f}%, "
                                          f"Memory={memory_percent:.1f}%")

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")

            self.stop_event.wait(self.sample_interval)


class TaskQueue:
    """Thread-safe task queue for concurrent execution testing"""

    def __init__(self, maxsize: int = 0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.lock = threading.Lock()
        self.logger = logging.getLogger(f"{__name__}.TaskQueue")

    def put_task(self, task_id: int, task_func: Callable, *args, **kwargs):
        """Add a task to the queue"""
        self.queue.put((task_id, task_func, args, kwargs))

    def get_task(self, timeout: Optional[float] = None) -> Tuple[int, Callable, tuple, dict]:
        """Get a task from the queue"""
        return self.queue.get(timeout=timeout)

    def task_done(self, success: bool = True):
        """Mark a task as completed"""
        with self.lock:
            if success:
                self.completed_tasks += 1
            else:
                self.failed_tasks += 1

    def get_stats(self) -> Dict[str, int]:
        """Get queue statistics"""
        with self.lock:
            return {
                'completed': self.completed_tasks,
                'failed': self.failed_tasks,
                'pending': self.queue.qsize()
            }


class ConcurrentExecutionTester:
    """Main concurrent execution testing framework"""

    def __init__(self, config: ConcurrentTestConfig):
        self.config = config
        self.resource_monitor = ResourceMonitor()
        self.results: List[WorkerResult] = []
        self.baseline_results: Dict[str, float] = {}

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Race condition detection
        self.shared_counter = 0
        self.counter_lock = threading.Lock()
        self.race_warnings = 0

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"concurrent_test_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def run_worker_task(self, worker_id: int, task_config: Dict[str, Any]) -> WorkerResult:
        """Execute a task in a worker thread/process"""
        start_time = time.time()
        start_cpu = time.process_time()

        # Get memory baseline
        process = psutil.Process()
        memory_baseline = process.memory_info().rss / (1024 * 1024)

        result = WorkerResult(
            worker_id=worker_id,
            start_time=start_time,
            end_time=0.0,
            execution_time=0.0,
            success=False,
            memory_peak_mb=0.0,
            cpu_time=0.0
        )

        try:
            # Execute the actual task
            if task_config.get('type') == 'pipeline':
                task_result = self._run_pipeline_task(task_config)
            elif task_config.get('type') == 'memory_test':
                task_result = self._run_memory_test_task(task_config)
            elif task_config.get('type') == 'cpu_test':
                task_result = self._run_cpu_test_task(task_config)
            elif task_config.get('type') == 'race_condition_test':
                task_result = self._run_race_condition_test(task_config)
            else:
                raise ValueError(f"Unknown task type: {task_config.get('type')}")

            result.success = True
            result.task_results = task_result

            # Test race condition detection
            if task_config.get('type') == 'race_condition_test':
                self._test_race_condition(worker_id)

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            self.logger.error(f"Worker {worker_id} failed: {e}")

        # Calculate metrics
        end_time = time.time()
        end_cpu = time.process_time()

        result.end_time = end_time
        result.execution_time = end_time - start_time
        result.cpu_time = end_cpu - start_cpu

        # Get peak memory
        memory_peak = process.memory_info().rss / (1024 * 1024)
        result.memory_peak_mb = max(0, memory_peak - memory_baseline)

        return result

    def _run_pipeline_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a pipeline task"""
        from scripts.run_full_pipeline import run_full_pipeline

        # Get parameters from config or use defaults
        params = task_config.get('params', {})

        # Set conservative defaults for concurrent execution
        default_params = {
            'demo': True,
            'kappa_method': 'acoustic',
            'graybody': 'dimensionless',
            'save_graybody_figure': False,
            'grid_points': 256,  # Smaller grid for concurrent testing
            'respect_parametric_bounds': True
        }

        # Merge with provided params
        params = {**default_params, **params}

        summary = run_full_pipeline(**params)

        return {
            'kappa_values': summary.kappa,
            'horizon_count': len(summary.horizon_positions),
            'spectrum_peak': summary.spectrum_peak_frequency,
            'detection_time': summary.t5sigma_s
        }

    def _run_memory_test_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a memory-intensive task"""
        array_size = task_config.get('array_size', 1000)
        num_arrays = task_config.get('num_arrays', 10)

        arrays = []
        for i in range(num_arrays):
            # Create large arrays
            arr = np.random.random((array_size, array_size))
            arrays.append(arr)

            # Perform some computation
            result = np.sum(arr * arr.T)

        # Memory allocation pattern
        memory_usage = []
        for i in range(5):
            big_array = np.random.random((array_size * 2, array_size * 2))
            memory_usage.append(big_array.nbytes)
            del big_array

        return {
            'arrays_created': len(arrays),
            'memory_usage_bytes': memory_usage,
            'computation_result': result if 'result' in locals() else 0
        }

    def _run_cpu_test_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a CPU-intensive task"""
        iterations = task_config.get('iterations', 1000000)

        # CPU-bound computation
        result = 0
        for i in range(iterations):
            result += i * i * np.sin(i * 0.001)

        # Matrix operations
        matrix_size = task_config.get('matrix_size', 100)
        matrices = []
        for i in range(5):
            matrix = np.random.random((matrix_size, matrix_size))
            eigenvals = np.linalg.eigvals(matrix)
            matrices.append(eigenvals)

        return {
            'iterations': iterations,
            'computation_result': result,
            'matrix_eigenvalues_computed': len(matrices)
        }

    def _run_race_condition_test_task(self, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a task to test race conditions"""
        iterations = task_config.get('iterations', 1000)

        # Simulate shared resource access
        local_counter = 0
        for i in range(iterations):
            # Non-atomic increment (potential race condition)
            current_value = self.shared_counter
            time.sleep(0.0001)  # Small delay to increase chance of race condition
            self.shared_counter = current_value + 1
            local_counter += 1

        return {
            'iterations': iterations,
            'local_counter': local_counter,
            'shared_counter_read': self.shared_counter
        }

    def _test_race_condition(self, worker_id: int):
        """Test for race conditions"""
        # Atomic increment for comparison
        with self.counter_lock:
            expected_value = self.shared_counter

        # Check if there's a discrepancy
        if self.shared_counter != expected_value:
            self.race_warnings += 1
            self.logger.warning(f"Potential race condition detected by worker {worker_id}: "
                             f"expected {expected_value}, got {self.shared_counter}")

    def run_scalability_test(self) -> List[ConcurrentTestSummary]:
        """Run scalability tests with different worker counts"""
        self.logger.info("Starting scalability tests")

        summaries = []

        # First, establish baseline with single worker
        baseline_time = self._run_single_worker_baseline()
        self.baseline_results['single_worker_time'] = baseline_time

        # Test with different worker counts
        for worker_count in self.config.worker_counts:
            self.logger.info(f"Testing with {worker_count} workers")

            summary = self._run_concurrent_test(worker_count, self.config.test_durations[0])
            summaries.append(summary)

            # Calculate speedup and efficiency
            if baseline_time > 0:
                summary.speedup_factor = baseline_time / summary.average_execution_time
                summary.efficiency_percent = (summary.speedup_factor / worker_count) * 100

        return summaries

    def _run_single_worker_baseline(self) -> float:
        """Run baseline test with single worker"""
        self.logger.info("Running single worker baseline")

        task_config = {
            'type': 'pipeline',
            'params': {
                'demo': True,
                'kappa_method': 'acoustic',
                'graybody': 'dimensionless',
                'save_graybody_figure': False
            }
        }

        start_time = time.time()
        result = self.run_worker_task(0, task_config)
        end_time = time.time()

        if result.success:
            self.logger.info(f"Baseline completed in {end_time - start_time:.3f}s")
            return end_time - start_time
        else:
            self.logger.error(f"Baseline failed: {result.error_message}")
            return 0.0

    def _run_concurrent_test(self, worker_count: int, duration: float) -> ConcurrentTestSummary:
        """Run concurrent test with specified worker count"""
        test_type = "thread" if self.config.thread_mode else "process"
        self.logger.info(f"Running {test_type} test with {worker_count} workers for {duration}s")

        # Start resource monitoring
        if self.config.enable_monitoring:
            self.resource_monitor.start_monitoring()

        # Prepare tasks
        task_configs = self._generate_task_configs(duration)

        # Execute tasks concurrently
        start_time = time.time()
        if self.config.thread_mode:
            results = self._run_with_threads(worker_count, task_configs)
        else:
            results = self._run_with_processes(worker_count, task_configs)
        end_time = time.time()

        # Stop monitoring
        resource_stats = {}
        if self.config.enable_monitoring:
            resource_stats = self.resource_monitor.stop_monitoring()

        # Calculate summary
        summary = self._calculate_summary(
            test_type=test_type,
            worker_count=worker_count,
            test_duration=end_time - start_time,
            results=results,
            resource_stats=resource_stats
        )

        self.logger.info(f"Test completed: {summary.successful_tasks}/{summary.total_tasks} successful, "
                        f"throughput: {summary.throughput_tasks_per_second:.2f} tasks/s")

        return summary

    def _generate_task_configs(self, duration: float) -> List[Dict[str, Any]]:
        """Generate task configurations for the test"""
        configs = []
        num_tasks = int(duration * 2)  # Aim for 2 tasks per second

        # Mix different task types
        task_types = ['pipeline', 'memory_test', 'cpu_test']
        if self.race_warnings < 5:  # Add race condition tests sparingly
            task_types.append('race_condition_test')

        for i in range(num_tasks):
            task_type = task_types[i % len(task_types)]

            if task_type == 'pipeline':
                config = {
                    'type': 'pipeline',
                    'params': {
                        'demo': True,
                        'kappa_method': np.random.choice(['acoustic', 'acoustic_exact']),
                        'graybody': 'dimensionless',
                        'grid_points': np.random.choice([256, 512]),
                        'save_graybody_figure': False
                    }
                }
            elif task_type == 'memory_test':
                config = {
                    'type': 'memory_test',
                    'array_size': np.random.randint(500, 1000),
                    'num_arrays': np.random.randint(5, 15)
                }
            elif task_type == 'cpu_test':
                config = {
                    'type': 'cpu_test',
                    'iterations': np.random.randint(500000, 1500000),
                    'matrix_size': np.random.randint(50, 150)
                }
            else:  # race_condition_test
                config = {
                    'type': 'race_condition_test',
                    'iterations': 1000
                }

            configs.append(config)

        return configs

    def _run_with_threads(self, worker_count: int, task_configs: List[Dict[str, Any]]) -> List[WorkerResult]:
        """Run tasks using ThreadPoolExecutor"""
        results = []

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self.run_worker_task, i, config): i
                for i, config in enumerate(task_configs)
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                try:
                    result = future.result(timeout=self.config.resource_limits['timeout_per_task'])
                    results.append(result)
                except Exception as e:
                    worker_id = future_to_id[future]
                    error_result = WorkerResult(
                        worker_id=worker_id,
                        start_time=time.time(),
                        end_time=time.time(),
                        execution_time=0.0,
                        success=False,
                        memory_peak_mb=0.0,
                        cpu_time=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)

        return results

    def _run_with_processes(self, worker_count: int, task_configs: List[Dict[str, Any]]) -> List[WorkerResult]:
        """Run tasks using ProcessPoolExecutor"""
        results = []

        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(self._run_worker_task_process, i, config): i
                for i, config in enumerate(task_configs)
            }

            # Collect results as they complete
            for future in as_completed(future_to_id):
                try:
                    result_dict = future.result(timeout=self.config.resource_limits['timeout_per_task'])
                    result = WorkerResult(**result_dict)
                    results.append(result)
                except Exception as e:
                    worker_id = future_to_id[future]
                    error_result = WorkerResult(
                        worker_id=worker_id,
                        start_time=time.time(),
                        end_time=time.time(),
                        execution_time=0.0,
                        success=False,
                        memory_peak_mb=0.0,
                        cpu_time=0.0,
                        error_message=str(e)
                    )
                    results.append(error_result)

        return results

    @staticmethod
    def _run_worker_task_process(worker_id: int, task_config: Dict[str, Any]) -> Dict[str, Any]:
        """Static method for running tasks in separate processes"""
        # This method needs to be static for pickle compatibility
        tester = ConcurrentExecutionTester.__new__(ConcurrentExecutionTester)
        return asdict(tester.run_worker_task(worker_id, task_config))

    def _calculate_summary(self, test_type: str, worker_count: int, test_duration: float,
                          results: List[WorkerResult], resource_stats: Dict[str, float]) -> ConcurrentTestSummary:
        """Calculate test summary statistics"""
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]

        # Basic metrics
        total_tasks = len(results)
        successful_tasks = len(successful)
        failed_tasks = len(failed)
        success_rate = successful_tasks / total_tasks if total_tasks > 0 else 0.0

        # Performance metrics
        if successful:
            execution_times = [r.execution_time for r in successful]
            average_execution_time = np.mean(execution_times)
        else:
            average_execution_time = 0.0

        throughput = successful_tasks / test_duration if test_duration > 0 else 0.0

        # Memory and CPU stats
        memory_stats = {}
        cpu_stats = {}

        if successful:
            memory_usage = [r.memory_peak_mb for r in results]
            cpu_times = [r.cpu_time for r in successful]

            memory_stats = {
                'peak': np.max(memory_usage),
                'average': np.mean(memory_usage),
                'total': np.sum(memory_usage)
            }

            cpu_stats = {
                'total': np.sum(cpu_times),
                'average': np.mean(cpu_times),
                'efficiency': np.sum(cpu_times) / (worker_count * test_duration) if test_duration > 0 else 0
            }

        # Scalability metrics
        scalability_metrics = {
            'tasks_per_worker_per_second': throughput / worker_count if worker_count > 0 else 0,
            'average_task_time': average_execution_time,
            'parallel_efficiency': (throughput * average_execution_time) / worker_count if worker_count > 0 else 0
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(success_rate, throughput, memory_stats, cpu_stats)

        return ConcurrentTestSummary(
            test_type=test_type,
            worker_count=worker_count,
            test_duration=test_duration,
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            success_rate=success_rate,
            throughput_tasks_per_second=throughput,
            speedup_factor=0.0,  # Will be calculated by caller
            efficiency_percent=0.0,  # Will be calculated by caller
            memory_usage_stats=memory_stats,
            cpu_usage_stats=cpu_stats,
            resource_contention_events=resource_stats.get('contention_events', 0),
            deadlock_events=0,  # TODO: Implement deadlock detection
            race_condition_warnings=self.race_warnings,
            scalability_metrics=scalability_metrics,
            recommendations=recommendations
        )

    def _generate_recommendations(self, success_rate: float, throughput: float,
                                memory_stats: Dict[str, float], cpu_stats: Dict[str, float]) -> List[str]:
        """Generate performance recommendations"""
        recommendations = []

        # Success rate recommendations
        if success_rate < 0.9:
            recommendations.append(f"Success rate ({success_rate:.1%}) is below target (>90%). "
                                 "Consider reducing task complexity or increasing timeouts.")

        # Throughput recommendations
        if throughput < 1.0:
            recommendations.append(f"Low throughput ({throughput:.2f} tasks/sec). "
                                 "Consider optimizing tasks or increasing worker count.")

        # Memory recommendations
        if memory_stats.get('peak', 0) > self.config.resource_limits['memory_limit_mb'] * 0.8:
            recommendations.append(f"High memory usage ({memory_stats['peak']:.1f}MB). "
                                 "Consider reducing concurrent workers or optimizing memory usage.")

        # CPU efficiency recommendations
        if cpu_stats.get('efficiency', 0) < 0.7:
            recommendations.append(f"Low CPU efficiency ({cpu_stats['efficiency']:.1%}). "
                                 "Tasks may be I/O bound or experiencing contention.")

        # Race condition warnings
        if self.race_warnings > 0:
            recommendations.append(f"Race conditions detected ({self.race_warnings} warnings). "
                                 "Review shared resource access and implement proper synchronization.")

        return recommendations

    def generate_visualizations(self, summaries: List[ConcurrentTestSummary]):
        """Generate comprehensive visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        if not summaries:
            self.logger.warning("No data to visualize")
            return

        # 1. Scalability analysis
        plt.figure(figsize=(15, 10))

        # Speedup plot
        plt.subplot(2, 3, 1)
        worker_counts = [s.worker_count for s in summaries]
        speedups = [s.speedup_factor for s in summaries]

        plt.plot(worker_counts, speedups, 'bo-', label='Actual Speedup')
        plt.plot(worker_counts, worker_counts, 'r--', label='Ideal Speedup')
        plt.xlabel('Number of Workers')
        plt.ylabel('Speedup Factor')
        plt.title('Scalability: Speedup vs Workers')
        plt.legend()
        plt.grid(True)

        # Efficiency plot
        plt.subplot(2, 3, 2)
        efficiencies = [s.efficiency_percent for s in summaries]
        plt.plot(worker_counts, efficiencies, 'go-', label='Parallel Efficiency')
        plt.xlabel('Number of Workers')
        plt.ylabel('Efficiency (%)')
        plt.title('Parallel Efficiency vs Workers')
        plt.grid(True)
        plt.ylim(0, 120)

        # Throughput plot
        plt.subplot(2, 3, 3)
        throughputs = [s.throughput_tasks_per_second for s in summaries]
        plt.plot(worker_counts, throughputs, 'mo-', label='Throughput')
        plt.xlabel('Number of Workers')
        plt.ylabel('Tasks per Second')
        plt.title('Throughput vs Workers')
        plt.grid(True)

        # Success rate plot
        plt.subplot(2, 3, 4)
        success_rates = [s.success_rate for s in summaries]
        plt.plot(worker_counts, success_rates, 'co-', label='Success Rate')
        plt.xlabel('Number of Workers')
        plt.ylabel('Success Rate')
        plt.title('Success Rate vs Workers')
        plt.grid(True)
        plt.ylim(0, 1.1)

        # Memory usage plot
        plt.subplot(2, 3, 5)
        memory_peaks = [s.memory_usage_stats.get('peak', 0) for s in summaries]
        memory_avgs = [s.memory_usage_stats.get('average', 0) for s in summaries]

        plt.plot(worker_counts, memory_peaks, 'r^-', label='Peak Memory')
        plt.plot(worker_counts, memory_avgs, 'b^-', label='Average Memory')
        plt.xlabel('Number of Workers')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Workers')
        plt.legend()
        plt.grid(True)

        # Resource contention events
        plt.subplot(2, 3, 6)
        contention_events = [s.resource_contention_events for s in summaries]
        plt.plot(worker_counts, contention_events, 'ko-', label='Contention Events')
        plt.xlabel('Number of Workers')
        plt.ylabel('Contention Events')
        plt.title('Resource Contention vs Workers')
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(viz_dir / f"scalability_analysis_{timestamp}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Performance comparison heatmap
        if len(summaries) > 1:
            metrics_data = []
            metric_names = ['Success Rate', 'Throughput', 'Efficiency', 'Speedup']

            for summary in summaries:
                metrics_data.append([
                    summary.success_rate,
                    summary.throughput_tasks_per_second,
                    summary.efficiency_percent / 100,
                    summary.speedup_factor
                ])

            # Normalize data for heatmap
            metrics_array = np.array(metrics_data)
            normalized_data = metrics_array / np.max(metrics_array, axis=0)

            plt.figure(figsize=(10, 6))
            sns.heatmap(normalized_data.T,
                       xticklabels=[f"{w} workers" for w in worker_counts],
                       yticklabels=metric_names,
                       annot=True,
                       cmap='YlOrRd',
                       cbar_kws={'label': 'Normalized Performance'})
            plt.title('Performance Metrics Heatmap')
            plt.tight_layout()
            plt.savefig(viz_dir / f"performance_heatmap_{timestamp}.png", dpi=300, bbox_inches='tight')
            plt.close()

        self.logger.info(f"Visualizations saved to {viz_dir}")

    def save_results(self, summaries: List[ConcurrentTestSummary]):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed results
        results_file = self.output_dir / f"concurrent_test_results_{timestamp}.json"
        results_data = {
            'timestamp': datetime.now().isoformat(),
            'test_config': asdict(self.config),
            'summaries': [asdict(summary) for summary in summaries],
            'baseline_results': self.baseline_results,
            'race_condition_warnings': self.race_warnings
        }

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        # Save summary CSV
        csv_file = self.output_dir / f"concurrent_test_summary_{timestamp}.csv"
        with open(csv_file, 'w') as f:
            f.write("test_type,worker_count,success_rate,throughput,speedup,efficiency,memory_peak,cpu_efficiency\n")
            for summary in summaries:
                f.write(f"{summary.test_type},{summary.worker_count},{summary.success_rate:.3f},"
                       f"{summary.throughput_tasks_per_second:.3f},{summary.speedup_factor:.3f},"
                       f"{summary.efficiency_percent:.1f},{summary.memory_usage_stats.get('peak', 0):.1f},"
                       f"{summary.cpu_usage_stats.get('efficiency', 0):.3f}\n")

        self.logger.info(f"Results saved to {results_file}")


def run_comprehensive_concurrent_tests():
    """Run comprehensive concurrent execution tests"""
    print("🚀 Starting Comprehensive Concurrent Execution Tests")

    # Test configuration
    config = ConcurrentTestConfig(
        max_workers=8,
        worker_counts=[1, 2, 4, 8, 16],
        thread_mode=True,  # Test threads first
        enable_monitoring=True,
        output_dir="results/concurrent_testing"
    )

    # Run thread-based tests
    print("Running thread-based concurrent tests...")
    tester = ConcurrentExecutionTester(config)
    thread_summaries = tester.run_scalability_test()

    # Run process-based tests
    print("Running process-based concurrent tests...")
    config.thread_mode = False
    process_tester = ConcurrentExecutionTester(config)
    process_summaries = process_tester.run_scalability_test()

    # Combine results
    all_summaries = thread_summaries + process_summaries

    # Generate visualizations and save results
    tester.generate_visualizations(all_summaries)
    tester.save_results(all_summaries)

    # Print summary
    print(f"\n{'='*60}")
    print("CONCURRENT EXECUTION TEST SUMMARY")
    print(f"{'='*60}")

    for summary in all_summaries:
        print(f"\n{summary.test_type.title()} Test - {summary.worker_count} Workers:")
        print(f"  Success Rate: {summary.success_rate:.1%}")
        print(f"  Throughput: {summary.throughput_tasks_per_second:.2f} tasks/sec")
        print(f"  Speedup: {summary.speedup_factor:.2f}x")
        print(f"  Efficiency: {summary.efficiency_percent:.1f}%")
        print(f"  Peak Memory: {summary.memory_usage_stats.get('peak', 0):.1f}MB")
        print(f"  Contentions: {summary.resource_contention_events}")

        if summary.recommendations:
            print(f"  Recommendations:")
            for rec in summary.recommendations[:2]:  # Show top 2
                print(f"    • {rec}")

    print(f"\nRace Condition Warnings: {tester.race_warnings}")
    print(f"Results saved to: {config.output_dir}")
    print(f"{'='*60}")

    return all_summaries


def main():
    """Main function for concurrent execution testing"""
    import argparse

    parser = argparse.ArgumentParser(description="Concurrent Execution Testing Framework")
    parser.add_argument("--threads", action="store_true", help="Test thread-based execution")
    parser.add_argument("--processes", action="store_true", help="Test process-based execution")
    parser.add_argument("--workers", type=int, nargs='+', default=[1, 2, 4, 8, 16], help="Worker counts to test")
    parser.add_argument("--duration", type=int, default=60, help="Test duration in seconds")
    parser.add_argument("--output-dir", type=str, default="results/concurrent_testing", help="Output directory")

    args = parser.parse_args()

    if args.threads or args.processes:
        config = ConcurrentTestConfig(
            worker_counts=args.workers,
            test_durations=[args.duration],
            thread_mode=args.threads if args.threads != args.processes else True,
            enable_monitoring=True,
            output_dir=args.output_dir
        )

        tester = ConcurrentExecutionTester(config)
        summaries = tester.run_scalability_test()

        tester.generate_visualizations(summaries)
        tester.save_results(summaries)

        # Print results
        print(f"\nConcurrent Execution Test Results:")
        for summary in summaries:
            print(f"{summary.worker_count} workers: {summary.success_rate:.1%} success, "
                  f"{summary.throughput_tasks_per_second:.2f} tasks/sec, "
                  f"{summary.speedup_factor:.2f}x speedup")
    else:
        # Run comprehensive tests
        summaries = run_comprehensive_concurrent_tests()

        # Evaluate overall success
        avg_success_rate = np.mean([s.success_rate for s in summaries])
        avg_efficiency = np.mean([s.efficiency_percent for s in summaries])

        if avg_success_rate < 0.8:
            print("❌ Concurrent execution tests indicate significant issues")
            sys.exit(1)
        elif avg_efficiency < 50:
            print("⚠️  Concurrent execution shows poor scalability")
            sys.exit(2)
        else:
            print("✅ Concurrent execution tests passed successfully")
            sys.exit(0)


if __name__ == "__main__":
    main()