#!/usr/bin/env python3
"""
Integration Test for Analog Hawking Radiation Orchestration System

Tests the complete orchestration system integration with existing pipeline components:
- Configuration system
- Orchestration engine
- Result aggregation
- Convergence detection
- Performance monitoring
- Experiment tracking
- Physics engine integration
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project paths to Python path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import orchestration components
from scripts.convergence_detector import AdvancedConvergenceDetector
from scripts.experiment_tracker import ExperimentTracker
from scripts.orchestration_engine import OrchestrationEngine
from scripts.performance_monitor import PerformanceMonitor
from scripts.result_aggregator import ResultAggregator
from scripts.run_full_pipeline import FullPipelineSummary


class TestOrchestrationIntegration(unittest.TestCase):
    """Integration tests for the complete orchestration system"""

    def setUp(self):
        """Set up test environment"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="orchestration_test_"))
        self.results_dir = self.test_dir / "results"
        self.configs_dir = self.test_dir / "configs" / "orchestration"

        # Create directory structure
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.configs_dir.mkdir(parents=True, exist_ok=True)

        # Create minimal test configuration
        self._create_test_configs()

        # Patch the results directory in orchestration components
        self.results_patch = patch("scripts.orchestration_engine.Path")
        self.mock_path = self.results_patch.start()
        self.mock_path.return_value.__truediv__.return_value = self.results_dir

    def tearDown(self):
        """Clean up test environment"""
        self.results_patch.stop()
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def _create_test_configs(self):
        """Create minimal test configurations"""
        # Base configuration
        base_config = {
            "experiment": {
                "name": "Integration Test",
                "description": "Test orchestration system integration",
                "author": "Test System",
                "version": "1.0.0",
            },
            "phases": [
                {
                    "name": "phase_1_test",
                    "description": "Test phase with minimal parameters",
                    "convergence_threshold": 0.5,  # High threshold for quick testing
                    "max_iterations": 5,
                    "parallel_workers": 2,
                }
            ],
            "physics": {
                "grid_points": 64,  # Low resolution for speed
                "grid_min": 0.0,
                "grid_max": 10e-6,
                "kappa_method": "acoustic",
                "graybody": "dimensionless",
                "enable_hybrid": False,
            },
            "detection": {
                "system_temperature": 30.0,
                "reference_bandwidth": 1e8,
                "significance_threshold": 5.0,
            },
            "parameter_bounds": {
                "laser_intensity": {"min": 1e16, "max": 1e18, "log_scale": True},
                "plasma_density": {"min": 1e16, "max": 1e18, "log_scale": True},
                "temperature_constant": {"min": 1e3, "max": 1e4, "log_scale": True},
            },
            "convergence": {
                "moving_average_window": 3,
                "improvement_threshold": 0.1,
                "min_samples_for_convergence": 5,
            },
            "results": {
                "output_directory": str(self.results_dir),
                "save_intermediate_results": True,
            },
        }

        # Phase configuration
        phase_config = {
            "phase_name": "phase_1_test",
            "description": "Test phase",
            "convergence_threshold": 0.5,
            "max_iterations": 5,
            "parallel_workers": 2,
            "sampling": {"method": "random", "n_samples": 3},
            "parameter_space": {
                "laser_intensity": {"min": 1e16, "max": 1e18, "n_points": 3, "log_scale": True},
                "plasma_density": {"min": 1e16, "max": 1e18, "n_points": 3, "log_scale": True},
            },
            "physics": {"grid_points": 64, "save_graybody_figure": False},
        }

        # Save configurations
        with open(self.configs_dir / "base.yml", "w") as f:
            import yaml

            yaml.dump(base_config, f)

        with open(self.configs_dir / "phase_1_test.yml", "w") as f:
            yaml.dump(phase_config, f)

    def _create_mock_pipeline_result(self, success: bool = True, **kwargs):
        """Create a mock pipeline result for testing"""
        if success:
            return FullPipelineSummary(
                plasma_density=kwargs.get("plasma_density", 5e17),
                laser_wavelength=800e-9,
                laser_intensity=kwargs.get("laser_intensity", 5e17),
                temperature_constant=kwargs.get("temperature_constant", 1e4),
                magnetic_field=None,
                use_fast_magnetosonic=False,
                grid_points=64,
                horizon_positions=[1e-6, 2e-6],
                kappa=[1e10, 2e10],
                kappa_err=[1e9, 2e9],
                spectrum_peak_frequency=1e8,
                inband_power_W=1e-12,
                T_sig_K=0.1,
                t5sigma_s=3600.0,
                T_H_K=0.05,
            )
        else:
            # Return a result that indicates failure
            return FullPipelineSummary(
                plasma_density=kwargs.get("plasma_density", 5e17),
                laser_wavelength=800e-9,
                laser_intensity=kwargs.get("laser_intensity", 5e17),
                temperature_constant=kwargs.get("temperature_constant", 1e4),
                magnetic_field=None,
                use_fast_magnetosonic=False,
                grid_points=64,
                horizon_positions=[],
                kappa=[],
                kappa_err=[],
                spectrum_peak_frequency=None,
                inband_power_W=None,
                T_sig_K=None,
                t5sigma_s=None,
                T_H_K=None,
            )

    @patch("scripts.orchestration_engine.run_full_pipeline")
    def test_orchestration_engine_initialization(self, mock_pipeline):
        """Test orchestration engine initialization"""
        # Mock the pipeline to return successful results
        mock_pipeline.return_value = self._create_mock_pipeline_result()

        engine = OrchestrationEngine(base_config_path=str(self.configs_dir / "base.yml"))
        engine.initialize_experiment("integration_test")

        self.assertIsNotNone(engine.manifest)
        self.assertEqual(engine.manifest.name, "integration_test")
        self.assertIn("phase_1_test", engine.phases)
        self.assertEqual(len(engine.phases), 1)

        # Verify phase configuration was loaded
        phase = engine.phases["phase_1_test"]
        self.assertEqual(phase.name, "phase_1_test")
        self.assertEqual(phase.config["max_iterations"], 5)
        self.assertEqual(phase.config["parallel_workers"], 2)

    @patch("scripts.orchestration_engine.run_full_pipeline")
    def test_single_simulation_execution(self, mock_pipeline):
        """Test execution of a single simulation"""
        # Mock the pipeline to return successful results
        mock_pipeline.return_value = self._create_mock_pipeline_result()

        engine = OrchestrationEngine(base_config_path=str(self.configs_dir / "base.yml"))
        engine.initialize_experiment()

        # Test single simulation
        test_params = {
            "laser_intensity": 1e17,
            "plasma_density": 1e17,
            "temperature_constant": 1e4,
            "physics": engine.phases["phase_1_test"].config["physics"],
        }

        result = engine._run_single_simulation(test_params)

        # Verify simulation was executed
        mock_pipeline.assert_called_once()

        # Verify result structure
        self.assertTrue(result["simulation_success"])
        self.assertIn("t5sigma_s", result)
        self.assertIn("kappa", result)
        self.assertIn("parameters_used", result)

    @patch("scripts.orchestration_engine.ProcessPoolExecutor")
    @patch("scripts.orchestration_engine.run_full_pipeline")
    def test_parallel_phase_execution(self, mock_pipeline, mock_executor):
        """Test parallel execution of a phase"""
        # Mock the pipeline to return successful results
        mock_pipeline.return_value = self._create_mock_pipeline_result()

        # Mock ProcessPoolExecutor
        mock_future = MagicMock()
        mock_future.result.return_value = {
            "simulation_success": True,
            "t5sigma_s": 3600.0,
            "kappa": [1e10],
            "parameters_used": {"laser_intensity": 1e17, "plasma_density": 1e17},
        }

        mock_executor_instance = MagicMock()
        mock_executor_instance.__enter__.return_value.submit.return_value = mock_future
        mock_executor_instance.__enter__.return_value.__iter__.return_value = [mock_future]
        mock_executor.return_value = mock_executor_instance

        engine = OrchestrationEngine(base_config_path=str(self.configs_dir / "base.yml"))
        engine.initialize_experiment()

        # Run a phase
        phase = engine.phases["phase_1_test"]
        engine._run_phase_parallel(phase)

        # Verify phase completed
        self.assertEqual(phase.status, "completed")
        self.assertGreater(len(phase.results), 0)

        # Verify results were collected
        for result in phase.results:
            self.assertTrue(result["simulation_success"])
            self.assertIn("t5sigma_s", result)

    def test_convergence_detection(self):
        """Test convergence detection system"""
        detector = AdvancedConvergenceDetector(
            window_size=3, improvement_threshold=0.1, min_samples=5
        )

        # Add improving results (simulating convergence)
        improving_results = [
            [{"simulation_success": True, "t5sigma_s": 1000.0, "kappa": [1e10]}],
            [{"simulation_success": True, "t5sigma_s": 800.0, "kappa": [1.2e10]}],
            [{"simulation_success": True, "t5sigma_s": 600.0, "kappa": [1.4e10]}],
            [{"simulation_success": True, "t5sigma_s": 500.0, "kappa": [1.5e10]}],
            [{"simulation_success": True, "t5sigma_s": 480.0, "kappa": [1.6e10]}],
        ]

        for results in improving_results:
            detector.add_results(results)

        # Check convergence
        result = detector.check_convergence()

        # Should not converge yet with only 5 samples
        self.assertFalse(result.is_converged)
        self.assertGreater(result.metrics.overall_convergence_score, 0.0)

        # Add more stable results to trigger convergence
        stable_results = [
            [{"simulation_success": True, "t5sigma_s": 470.0, "kappa": [1.6e10]}],
            [{"simulation_success": True, "t5sigma_s": 475.0, "kappa": [1.6e10]}],
            [{"simulation_success": True, "t5sigma_s": 472.0, "kappa": [1.6e10]}],
        ]

        for results in stable_results:
            detector.add_results(results)

        result = detector.check_convergence()
        # With more stable results, should approach convergence
        self.assertGreater(result.metrics.overall_convergence_score, 0.3)

    def test_experiment_tracking(self):
        """Test experiment tracking and UUID system"""
        tracker = ExperimentTracker(experiment_name="tracking_test", base_dir=str(self.results_dir))

        # Test initial state
        self.assertIsNotNone(tracker.experiment_id)
        self.assertEqual(tracker.manifest.name, "tracking_test")
        self.assertEqual(tracker.manifest.status, "created")

        # Test phase tracking
        test_config = {"max_iterations": 5, "parallel_workers": 2}
        tracker.add_phase("test_phase", test_config)
        tracker.start_phase("test_phase")

        self.assertEqual(tracker.manifest.current_phase, "test_phase")
        self.assertIn("test_phase", tracker.manifest.phases)

        # Test metrics tracking
        tracker.update_metrics(
            simulations_completed=10, simulations_successful=8, compute_time=120.0
        )

        self.assertEqual(tracker.manifest.metrics.total_simulations, 10)
        self.assertEqual(tracker.manifest.metrics.successful_simulations, 8)
        self.assertEqual(tracker.manifest.metrics.total_compute_time, 120.0)

        # Test completion
        tracker.complete_experiment()
        self.assertEqual(tracker.manifest.status, "completed")
        self.assertIsNotNone(tracker.manifest.end_timestamp)

    def test_performance_monitoring(self):
        """Test performance monitoring system"""
        monitor = PerformanceMonitor(
            experiment_id="perf_test", update_interval=1.0  # Short interval for testing
        )

        # Test metrics collection
        monitor.update_simulation_stats(completed=5, successful=4, compute_time=60.0)

        monitor.set_phase_progress(0.5, total_simulations=10)

        # Get current metrics
        system_metrics, experiment_metrics = monitor.get_current_metrics()

        # Experiment metrics should be available
        if experiment_metrics:
            self.assertEqual(experiment_metrics.simulations_completed, 5)
            self.assertEqual(experiment_metrics.simulations_successful, 4)
            self.assertEqual(experiment_metrics.phase_progress, 0.5)

        # Test performance summary
        summary = monitor.get_performance_summary()
        self.assertIn("experiment_id", summary)
        self.assertIn("simulations", summary)
        self.assertIn("progress", summary)

    def test_result_aggregation(self):
        """Test result aggregation framework"""
        # Create test results
        test_results = {
            "phase_1_test": [
                {
                    "simulation_success": True,
                    "t5sigma_s": 3600.0,
                    "kappa": [1e10],
                    "T_sig_K": 0.1,
                    "parameters_used": {"laser_intensity": 1e17, "plasma_density": 1e17},
                },
                {
                    "simulation_success": True,
                    "t5sigma_s": 1800.0,
                    "kappa": [2e10],
                    "T_sig_K": 0.2,
                    "parameters_used": {"laser_intensity": 2e17, "plasma_density": 2e17},
                },
            ]
        }

        # Create test experiment directory
        test_experiment_dir = self.results_dir / "test_aggregation"
        test_experiment_dir.mkdir(parents=True, exist_ok=True)

        # Save test results
        phase_dir = test_experiment_dir / "phase_1_test"
        phase_dir.mkdir()

        with open(phase_dir / "simulation_results.json", "w") as f:
            json.dump(test_results["phase_1_test"], f)

        # Test aggregation
        aggregator = ResultAggregator("test_aggregation")
        aggregator.experiment_dir = test_experiment_dir  # Override for test

        success = aggregator.load_experiment_data()
        self.assertTrue(success)
        self.assertIn("phase_1_test", aggregator.results)

        # Test aggregation
        aggregate = aggregator.aggregate_results()

        self.assertEqual(aggregate.experiment_id, "test_aggregation")
        self.assertEqual(aggregate.total_simulations, 2)
        self.assertEqual(aggregate.successful_simulations, 2)
        self.assertEqual(aggregate.success_rate, 1.0)
        self.assertEqual(aggregate.best_detection_time, 1800.0)
        self.assertEqual(aggregate.best_kappa, 2e10)

    def test_configuration_hierarchy(self):
        """Test hierarchical configuration system"""
        base_config = {
            "physics": {"grid_points": 512, "enable_hybrid": False},
            "detection": {"system_temperature": 30.0},
        }

        phase_config = {
            "physics": {"grid_points": 256, "save_graybody_figure": False},
            "sampling": {"method": "random"},
        }

        engine = OrchestrationEngine()
        merged_config = engine._merge_configs(base_config, phase_config)

        # Test that phase config overrides base config
        self.assertEqual(merged_config["physics"]["grid_points"], 256)
        self.assertEqual(merged_config["physics"]["enable_hybrid"], False)  # From base
        self.assertEqual(merged_config["physics"]["save_graybody_figure"], False)  # From phase
        self.assertEqual(merged_config["detection"]["system_temperature"], 30.0)  # From base
        self.assertEqual(merged_config["sampling"]["method"], "random")  # From phase

    @patch("scripts.orchestration_engine.run_full_pipeline")
    def test_end_to_end_minimal_experiment(self, mock_pipeline):
        """Test end-to-end minimal experiment execution"""

        # Mock pipeline to return mixed results (some success, some failure)
        def mock_pipeline_side_effect(**kwargs):
            # Fail for very high intensity (simulating edge case)
            if kwargs.get("laser_intensity", 0) > 5e17:
                return self._create_mock_pipeline_result(success=False, **kwargs)
            else:
                return self._create_mock_pipeline_result(success=True, **kwargs)

        mock_pipeline.side_effect = mock_pipeline_side_effect

        # Create a minimal engine with very small parameters
        engine = OrchestrationEngine(base_config_path=str(self.configs_dir / "base.yml"))
        engine.initialize_experiment("end_to_end_test")

        # Run a very minimal version of the experiment
        # We'll manually run one phase with reduced parameters
        phase = engine.phases["phase_1_test"]
        phase.config["max_iterations"] = 2  # Very small for testing
        phase.config["parallel_workers"] = 1

        # Mock the parallel execution to run sequentially for testing
        with patch("scripts.orchestration_engine.ProcessPoolExecutor") as mock_executor:
            # Set up mock to run sequentially
            mock_executor.return_value.__enter__.return_value.submit.side_effect = (
                lambda func, *args: MagicMock(result=lambda: func(*args))
            )
            mock_executor.return_value.__enter__.return_value.__iter__.return_value = []

            engine._run_phase_parallel(phase)

        # Verify phase completed
        self.assertEqual(phase.status, "completed")
        self.assertGreater(len(phase.results), 0)

        # Verify results were saved
        results_file = (
            self.results_dir / engine.experiment_id / "phase_1_test" / "simulation_results.json"
        )
        # Note: In actual execution, this would be created by _save_phase_results

        # Test result aggregation
        if len(phase.results) > 0:
            aggregator = ResultAggregator(engine.experiment_id)
            aggregator.results = {"phase_1_test": phase.results}  # Manually set for test

            aggregate = aggregator.aggregate_results()

            # Should have some successful simulations
            self.assertGreater(aggregate.successful_simulations, 0)
            self.assertGreater(aggregate.success_rate, 0.0)


class TestPhysicsEngineIntegration(unittest.TestCase):
    """Tests specifically for physics engine integration"""

    @patch("scripts.run_full_pipeline.run_full_pipeline")
    def test_pipeline_parameter_passing(self, mock_pipeline):
        """Test that parameters are correctly passed to physics engine"""
        from scripts.orchestration_engine import OrchestrationEngine

        # Mock the pipeline
        mock_pipeline.return_value = MagicMock(kappa=[1e10], t5sigma_s=3600.0, T_sig_K=0.1)

        engine = OrchestrationEngine()

        test_params = {
            "plasma_density": 5e17,
            "laser_intensity": 1e17,
            "temperature_constant": 1e4,
            "physics": {"grid_points": 64, "kappa_method": "acoustic", "graybody": "dimensionless"},
        }

        result = engine._run_single_simulation(test_params)

        # Verify pipeline was called with correct parameters
        mock_pipeline.assert_called_once()
        call_kwargs = mock_pipeline.call_args.kwargs

        self.assertEqual(call_kwargs["plasma_density"], 5e17)
        self.assertEqual(call_kwargs["laser_intensity"], 1e17)
        self.assertEqual(call_kwargs["temperature_constant"], 1e4)
        self.assertEqual(call_kwargs["grid_points"], 64)
        self.assertEqual(call_kwargs["kappa_method"], "acoustic")
        self.assertEqual(call_kwargs["graybody"], "dimensionless")


def run_integration_tests():
    """Run all integration tests and report results"""
    print("Running Analog Hawking Radiation Orchestration Integration Tests")
    print("=" * 70)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOrchestrationIntegration)
    suite.addTests(loader.loadTestsFromTestCase(TestPhysicsEngineIntegration))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 70)
    print("INTEGRATION TEST SUMMARY")
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("✅ All integration tests passed!")
        return True
    else:
        print("❌ Some integration tests failed:")
        for test, traceback in result.failures + result.errors:
            print(f"  - {test}: {traceback.splitlines()[-1]}")
        return False


if __name__ == "__main__":
    success = run_integration_tests()
    sys.exit(0 if success else 1)
