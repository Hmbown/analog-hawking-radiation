#!/usr/bin/env python3
"""
Automated Stress Test Orchestration System

This orchestrator coordinates comprehensive stress testing across all components,
automates test execution, manages resources, and provides integrated reporting.

Features:
- Automated test pipeline orchestration
- Resource management and scheduling
- Progressive test complexity scaling
- Automatic failure recovery and retry logic
- Integrated reporting and alerting
- CI/CD integration capabilities
- Multi-environment testing support
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import schedule
import yaml
from dataclasses import dataclass, field

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

# Import stress testing modules
from stress_test_parameter_sweep import StressTestRunner, StressTestConfig
from memory_profiler_benchmark import PerformanceBenchmarkRunner, run_comprehensive_benchmarks
from concurrent_execution_test import ConcurrentExecutionTester, ConcurrentTestConfig, run_comprehensive_concurrent_tests
from stress_test_visualizer import StressTestVisualizer


class TestStatus(Enum):
    """Test execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


@dataclass
class TestSuite:
    """Test suite configuration"""
    name: str
    description: str
    test_type: str  # "parameter_sweep", "memory_benchmark", "concurrent", "custom"
    config: Dict[str, Any]
    enabled: bool = True
    priority: int = 0  # Lower number = higher priority
    dependencies: List[str] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    timeout_minutes: int = 60


@dataclass
class TestExecution:
    """Test execution record"""
    suite_name: str
    execution_id: str
    status: TestStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: float = 0.0
    success: bool = False
    results_file: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    retry_count: int = 0


@dataclass
class OrchestrationConfig:
    """Orchestration system configuration"""
    max_concurrent_suites: int = 2
    resource_limits: Dict[str, float] = field(default_factory=lambda: {
        'memory_limit_gb': 16.0,
        'cpu_limit_percent': 80.0,
        'disk_space_gb': 10.0
    })
    notification_settings: Dict[str, Any] = field(default_factory=lambda: {
        'email_enabled': False,
        'slack_enabled': False,
        'success_threshold': 0.8,
        'alert_on_failure': True
    })
    schedule_settings: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': False,
        'cron_expression': '0 2 * * *',  # Daily at 2 AM
        'timezone': 'UTC'
    })
    output_directory: str = "results/stress_testing"
    retention_days: int = 30


class StressTestOrchestrator:
    """Main stress test orchestration system"""

    def __init__(self, config: OrchestrationConfig):
        self.config = config
        self.test_suites: List[TestSuite] = []
        self.execution_history: List[TestExecution] = []
        self.active_executions: Dict[str, TestExecution] = {}

        # Setup directories
        self.output_dir = Path(config.output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "executions").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "configs").mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize visualizer
        self.visualizer = StressTestVisualizer(str(self.output_dir / "visualizations"))

        # Load default test suites
        self._load_default_test_suites()

    def _setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"orchestrator_{timestamp}.log"

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )

    def _load_default_test_suites(self):
        """Load default test suite configurations"""
        default_suites = [
            TestSuite(
                name="basic_parameter_sweep",
                description="Basic parameter sweep with 50 configurations",
                test_type="parameter_sweep",
                config={
                    'sweep_size': 50,
                    'concurrent_workers': 2,
                    'memory_threshold_mb': 4096,
                    'timeout_per_config': 120,
                    'enable_profiling': True,
                    'test_concurrent_execution': True,
                    'generate_visualizations': True
                },
                priority=1,
                max_retries=2,
                timeout_minutes=30
            ),
            TestSuite(
                name="comprehensive_parameter_sweep",
                description="Comprehensive parameter sweep with 100+ configurations",
                test_type="parameter_sweep",
                config={
                    'sweep_size': 100,
                    'concurrent_workers': 4,
                    'memory_threshold_mb': 8192,
                    'timeout_per_config': 300,
                    'enable_profiling': True,
                    'test_concurrent_execution': True,
                    'generate_visualizations': True
                },
                priority=2,
                dependencies=["basic_parameter_sweep"],
                max_retries=2,
                timeout_minutes=60
            ),
            TestSuite(
                name="memory_profiling",
                description="Memory profiling and performance benchmarking",
                test_type="memory_benchmark",
                config={},
                priority=1,
                max_retries=1,
                timeout_minutes=20
            ),
            TestSuite(
                name="concurrent_execution",
                description="Concurrent execution testing (threads and processes)",
                test_type="concurrent",
                config={
                    'worker_counts': [1, 2, 4, 8],
                    'test_durations': [60],
                    'thread_mode': True,
                    'enable_monitoring': True
                },
                priority=2,
                max_retries=1,
                timeout_minutes=30
            ),
            TestSuite(
                name="extreme_stress_test",
                description="Extreme stress test with maximum configurations",
                test_type="parameter_sweep",
                config={
                    'sweep_size': 200,
                    'concurrent_workers': 8,
                    'memory_threshold_mb': 16384,
                    'timeout_per_config': 600,
                    'enable_profiling': True,
                    'test_concurrent_execution': True,
                    'generate_visualizations': True
                },
                priority=3,
                dependencies=["comprehensive_parameter_sweep"],
                max_retries=1,
                timeout_minutes=120
            )
        ]

        self.test_suites.extend(default_suites)
        self.logger.info(f"Loaded {len(default_suites)} default test suites")

    def add_test_suite(self, suite: TestSuite):
        """Add a custom test suite"""
        self.test_suites.append(suite)
        self.logger.info(f"Added test suite: {suite.name}")

    def load_test_suites_from_config(self, config_file: str):
        """Load test suites from configuration file"""
        try:
            with open(config_file, 'r') as f:
                data = yaml.safe_load(f)

            for suite_data in data.get('test_suites', []):
                suite = TestSuite(**suite_data)
                self.add_test_suite(suite)

            self.logger.info(f"Loaded test suites from {config_file}")
        except Exception as e:
            self.logger.error(f"Error loading test suites from {config_file}: {e}")

    def validate_environment(self) -> bool:
        """Validate that the environment can handle stress testing"""
        self.logger.info("Validating environment for stress testing")

        try:
            # Check system resources
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()

            if memory_gb < self.config.resource_limits['memory_limit_gb']:
                self.logger.warning(f"Available memory ({memory_gb:.1f}GB) is less than recommended "
                                  f"({self.config.resource_limits['memory_limit_gb']}GB)")

            if cpu_count < 4:
                self.logger.warning(f"Available CPU cores ({cpu_count}) may limit concurrent testing")

            # Check disk space
            disk_usage = psutil.disk_usage(str(self.output_dir))
            available_gb = disk_usage.free / (1024**3)

            if available_gb < self.config.resource_limits['disk_space_gb']:
                self.logger.error(f"Insufficient disk space: {available_gb:.1f}GB available, "
                                f"{self.config.resource_limits['disk_space_gb']}GB required")
                return False

            # Check Python dependencies
            required_modules = ['numpy', 'matplotlib', 'seaborn', 'scipy', 'psutil']
            missing_modules = []

            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    missing_modules.append(module)

            if missing_modules:
                self.logger.error(f"Missing required modules: {missing_modules}")
                return False

            self.logger.info("Environment validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Environment validation failed: {e}")
            return False

    def execute_test_suite(self, suite: TestSuite) -> TestExecution:
        """Execute a single test suite"""
        execution_id = f"{suite.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        execution = TestExecution(
            suite_name=suite.name,
            execution_id=execution_id,
            status=TestStatus.RUNNING,
            start_time=datetime.now()
        )

        self.active_executions[execution_id] = execution
        self.logger.info(f"Starting test suite execution: {suite.name} (ID: {execution_id})")

        try:
            if suite.test_type == "parameter_sweep":
                results = self._execute_parameter_sweep(suite)
            elif suite.test_type == "memory_benchmark":
                results = self._execute_memory_benchmark(suite)
            elif suite.test_type == "concurrent":
                results = self._execute_concurrent_test(suite)
            else:
                raise ValueError(f"Unknown test type: {suite.test_type}")

            # Save results
            results_file = self._save_execution_results(execution_id, suite, results)
            execution.results_file = results_file
            execution.success = True
            execution.status = TestStatus.COMPLETED

            # Extract key metrics
            if suite.test_type == "parameter_sweep":
                execution.metrics = {
                    'success_rate': results.get('summary', {}).get('success_rate', 0),
                    'total_time': results.get('summary', {}).get('total_execution_time', 0),
                    'peak_memory': results.get('summary', {}).get('memory_peak_mb', 0)
                }
            elif suite.test_type == "memory_benchmark":
                execution.metrics = {
                    'benchmarks_run': len(results.get('benchmarks', {})),
                    'regressions': len(results.get('regressions', []))
                }

            self.logger.info(f"Test suite {suite.name} completed successfully")

        except Exception as e:
            execution.success = False
            execution.error_message = str(e)
            execution.status = TestStatus.FAILED
            self.logger.error(f"Test suite {suite.name} failed: {e}")

        finally:
            execution.end_time = datetime.now()
            execution.duration_seconds = (execution.end_time - execution.start_time).total_seconds()
            self.execution_history.append(execution)
            del self.active_executions[execution_id]

        return execution

    def _execute_parameter_sweep(self, suite: TestSuite) -> Dict[str, Any]:
        """Execute parameter sweep test suite"""
        config = StressTestConfig(**suite.config)
        runner = StressTestRunner(config)
        summary = runner.run_stress_test()

        # Return results in standard format
        return {
            'summary': summary.__dict__,
            'config': config.__dict__,
            'detailed_results': [asdict(result) for result in runner.results]
        }

    def _execute_memory_benchmark(self, suite: TestSuite) -> Dict[str, Any]:
        """Execute memory profiling and benchmarking"""
        results, regressions = run_comprehensive_benchmarks()

        return {
            'benchmarks': {name: asdict(benchmark) for name, benchmark in results.items()},
            'regressions': [asdict(regression) for regression in regressions]
        }

    def _execute_concurrent_test(self, suite: TestSuite) -> Dict[str, Any]:
        """Execute concurrent execution testing"""
        config = ConcurrentTestConfig(**suite.config)
        tester = ConcurrentExecutionTester(config)
        summaries = tester.run_scalability_test()

        return {
            'summaries': [asdict(summary) for summary in summaries],
            'config': config.__dict__
        }

    def _save_execution_results(self, execution_id: str, suite: TestSuite, results: Dict[str, Any]) -> str:
        """Save execution results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / "executions" / f"{execution_id}_results.json"

        execution_data = {
            'execution_id': execution_id,
            'suite_name': suite.name,
            'suite_config': suite.__dict__,
            'timestamp': datetime.now().isoformat(),
            'results': results
        }

        with open(results_file, 'w') as f:
            json.dump(execution_data, f, indent=2, default=str)

        self.logger.info(f"Saved execution results to {results_file}")
        return str(results_file)

    def run_orchestrated_test_cycle(self, suite_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run a complete orchestrated test cycle"""
        self.logger.info("Starting orchestrated stress test cycle")

        # Validate environment
        if not self.validate_environment():
            raise RuntimeError("Environment validation failed")

        # Filter test suites
        if suite_names:
            suites_to_run = [s for s in self.test_suites if s.name in suite_names and s.enabled]
        else:
            suites_to_run = [s for s in self.test_suites if s.enabled]

        # Sort by priority and dependencies
        suites_to_run = self._sort_suites_by_dependencies(suites_to_run)

        cycle_start_time = datetime.now()
        executed_suites = []
        failed_suites = []

        for suite in suites_to_run:
            self.logger.info(f"Executing suite: {suite.name}")

            # Check dependencies
            if not self._check_dependencies(suite, executed_suites):
                self.logger.warning(f"Skipping {suite.name} due to unmet dependencies")
                continue

            # Check resource availability
            if not self._check_resource_availability():
                self.logger.warning(f"Skipping {suite.name} due to insufficient resources")
                continue

            # Execute suite with retry logic
            execution = None
            for attempt in range(suite.max_retries + 1):
                if attempt > 0:
                    self.logger.info(f"Retrying {suite.name} (attempt {attempt + 1})")
                    time.sleep(60)  # Wait before retry

                try:
                    execution = self.execute_test_suite(suite)
                    if execution.success:
                        break
                    else:
                        suite.retry_count += 1
                except Exception as e:
                    self.logger.error(f"Suite {suite.name} execution failed: {e}")
                    suite.retry_count += 1

            if execution and execution.success:
                executed_suites.append(suite)
                self.logger.info(f"Suite {suite.name} completed successfully")
            else:
                failed_suites.append(suite)
                self.logger.error(f"Suite {suite.name} failed after all retries")

                # Check if we should continue
                if suite.priority <= 1:  # High priority suite failed
                    self.logger.error("High priority suite failed, aborting cycle")
                    break

        cycle_end_time = datetime.now()
        cycle_duration = (cycle_end_time - cycle_start_time).total_seconds()

        # Generate cycle report
        cycle_summary = {
            'cycle_id': f"cycle_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': cycle_start_time.isoformat(),
            'end_time': cycle_end_time.isoformat(),
            'duration_seconds': cycle_duration,
            'total_suites': len(suites_to_run),
            'executed_suites': len(executed_suites),
            'failed_suites': len(failed_suites),
            'success_rate': len(executed_suites) / len(suites_to_run) if suites_to_run else 0,
            'executed_suite_names': [s.name for s in executed_suites],
            'failed_suite_names': [s.name for s in failed_suites],
            'executions': [asdict(exec) for exec in self.execution_history[-len(executed_suites):]]
        }

        # Save cycle summary
        self._save_cycle_summary(cycle_summary)

        # Generate comprehensive report
        self._generate_cycle_report(cycle_summary)

        # Send notifications if configured
        if self.config.notification_settings.get('alert_on_failure', True) and failed_suites:
            self._send_failure_notification(failed_suites)

        self.logger.info(f"Orchestrated test cycle completed: {len(executed_suites)}/{len(suites_to_run)} suites successful")

        return cycle_summary

    def _sort_suites_by_dependencies(self, suites: List[TestSuite]) -> List[TestSuite]:
        """Sort suites by priority and dependencies (topological sort)"""
        # Simple implementation: sort by priority first, then handle dependencies
        sorted_suites = sorted(suites, key=lambda s: s.priority)

        # Reorder based on dependencies
        final_order = []
        remaining = sorted_suites.copy()

        while remaining:
            # Find suites with no unmet dependencies
            ready = []
            for suite in remaining:
                dependencies_met = all(
                    dep.suite_name in [s.name for s in final_order]
                    for dep in self.test_suites if dep.name in suite.dependencies
                )
                if dependencies_met:
                    ready.append(suite)

            if not ready:
                # Circular dependency or missing dependency
                self.logger.warning("Circular or missing dependency detected, adding remaining suites")
                ready.extend(remaining)
                remaining.clear()

            for suite in ready:
                final_order.append(suite)
                if suite in remaining:
                    remaining.remove(suite)

        return final_order

    def _check_dependencies(self, suite: TestSuite, executed_suites: List[TestSuite]) -> bool:
        """Check if all dependencies for a suite have been executed successfully"""
        executed_names = {s.name for s in executed_suites}
        return all(dep in executed_names for dep in suite.dependencies)

    def _check_resource_availability(self) -> bool:
        """Check if system resources are available for testing"""
        try:
            import psutil

            # Check memory
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.config.resource_limits['cpu_limit_percent']:
                return False

            # Check CPU load
            cpu_percent = psutil.cpu_percent(interval=1)
            if cpu_percent > self.config.resource_limits['cpu_limit_percent']:
                return False

            return True

        except Exception:
            return True  # Assume available if we can't check

    def _save_cycle_summary(self, summary: Dict[str, Any]):
        """Save cycle summary to file"""
        summary_file = self.output_dir / "reports" / f"{summary['cycle_id']}_summary.json"

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        self.logger.info(f"Saved cycle summary to {summary_file}")

    def _generate_cycle_report(self, cycle_summary: Dict[str, Any]):
        """Generate comprehensive cycle report"""
        # Collect results from executed suites
        stress_data = {}
        concurrent_data = {}
        benchmark_data = {}

        for execution in self.execution_history[-cycle_summary['executed_suites']:]:
            if execution.results_file and Path(execution.results_file).exists():
                with open(execution.results_file, 'r') as f:
                    data = json.load(f)

                suite_name = execution.suite_name
                if 'parameter_sweep' in suite_name:
                    stress_data = data
                elif 'concurrent' in suite_name:
                    concurrent_data = data
                elif 'memory' in suite_name or 'benchmark' in suite_name:
                    benchmark_data = data

        # Generate comprehensive report
        report_file = self.visualizer.generate_comprehensive_report(
            stress_data=stress_data,
            concurrent_data={'thread_data': concurrent_data, 'process_data': {}} if concurrent_data else None,
            benchmark_data=benchmark_data
        )

        self.logger.info(f"Generated comprehensive cycle report: {report_file}")

    def _send_failure_notification(self, failed_suites: List[TestSuite]):
        """Send failure notification (placeholder)"""
        # This would integrate with email, Slack, etc.
        self.logger.warning(f"Notification: {len(failed_suites)} test suites failed: "
                          f"{[s.name for s in failed_suites]}")

    def cleanup_old_results(self):
        """Clean up old test results based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=self.config.retention_days)

        # Clean up execution files
        executions_dir = self.output_dir / "executions"
        if executions_dir.exists():
            for file_path in executions_dir.glob("*.json"):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old execution file: {file_path}")

        # Clean up logs
        logs_dir = self.output_dir / "logs"
        if logs_dir.exists():
            for file_path in logs_dir.glob("*.log"):
                file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_time < cutoff_date:
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old log file: {file_path}")

    def start_scheduled_execution(self):
        """Start scheduled execution daemon"""
        if not self.config.schedule_settings.get('enabled', False):
            self.logger.info("Scheduled execution is disabled")
            return

        cron_expr = self.config.schedule_settings.get('cron_expression', '0 2 * * *')
        self.logger.info(f"Starting scheduled execution with cron: {cron_expr}")

        # Schedule daily execution
        schedule.every().day.at("02:00").do(self.run_orchestrated_test_cycle)

        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


def create_default_config() -> OrchestrationConfig:
    """Create default orchestration configuration"""
    return OrchestrationConfig(
        max_concurrent_suites=2,
        resource_limits={
            'memory_limit_gb': 16.0,
            'cpu_limit_percent': 80.0,
            'disk_space_gb': 10.0
        },
        notification_settings={
            'email_enabled': False,
            'slack_enabled': False,
            'success_threshold': 0.8,
            'alert_on_failure': True
        },
        schedule_settings={
            'enabled': False,
            'cron_expression': '0 2 * * *',
            'timezone': 'UTC'
        },
        output_directory="results/stress_testing",
        retention_days=30
    )


def main():
    """Main function for stress test orchestration"""
    import argparse

    parser = argparse.ArgumentParser(description="Automated Stress Test Orchestration")
    parser.add_argument("--config", type=str, help="Orchestration configuration file")
    parser.add_argument("--suites", nargs='+', help="Specific test suites to run")
    parser.add_argument("--validate-only", action="store_true", help="Only validate environment")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old results")
    parser.add_argument("--schedule", action="store_true", help="Start scheduled execution daemon")
    parser.add_argument("--output-dir", type=str, default="results/stress_testing", help="Output directory")

    args = parser.parse_args()

    # Load or create configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_data = yaml.safe_load(f)
        config = OrchestrationConfig(**config_data)
    else:
        config = create_default_config()
        config.output_directory = args.output_dir

    # Initialize orchestrator
    orchestrator = StressTestOrchestrator(config)

    # Load custom test suites if config file provided
    if args.config:
        orchestrator.load_test_suites_from_config(args.config)

    if args.validate_only:
        success = orchestrator.validate_environment()
        sys.exit(0 if success else 1)

    if args.cleanup:
        orchestrator.cleanup_old_results()
        print("✅ Cleanup completed")
        sys.exit(0)

    if args.schedule:
        print("🚀 Starting scheduled stress test execution daemon")
        try:
            orchestrator.start_scheduled_execution()
        except KeyboardInterrupt:
            print("\n⏹️  Scheduled execution stopped")
        sys.exit(0)

    # Run orchestrated test cycle
    print("🚀 Starting orchestrated stress test cycle")
    print(f"Output directory: {config.output_directory}")

    try:
        cycle_summary = orchestrator.run_orchestrated_test_cycle(args.suites)

        print(f"\n{'='*60}")
        print("STRESS TEST CYCLE SUMMARY")
        print(f"{'='*60}")
        print(f"Cycle ID: {cycle_summary['cycle_id']}")
        print(f"Duration: {cycle_summary['duration_seconds']:.1f} seconds")
        print(f"Success Rate: {cycle_summary['success_rate']:.1%}")
        print(f"Executed Suites: {', '.join(cycle_summary['executed_suite_names'])}")

        if cycle_summary['failed_suite_names']:
            print(f"Failed Suites: {', '.join(cycle_summary['failed_suite_names'])}")

        print(f"\nDetailed results saved to: {config.output_directory}")
        print(f"{'='*60}")

        # Determine exit code
        if cycle_summary['success_rate'] < 0.8:
            print("❌ Stress test cycle indicates significant issues")
            sys.exit(1)
        else:
            print("✅ Stress test cycle completed successfully")
            sys.exit(0)

    except Exception as e:
        print(f"❌ Stress test cycle failed: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()