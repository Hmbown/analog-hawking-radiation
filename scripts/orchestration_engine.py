#!/usr/bin/env python3
"""
Core Orchestration Engine for Analog Hawking Radiation Detection Experiments

Coordinates multi-phase experiments with automated phase transitions,
parallel processing, and convergence detection.
"""

from __future__ import annotations

import json
import logging
import os

# Add project paths to Python path
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))


# Import monitoring and validation components
# Heavy dependencies are imported lazily in their initialization methods to
# keep CLI help and basic imports lightweight in environments without full deps.


@dataclass
class ExperimentPhase:
    """Represents a single phase in the multi-phase experiment"""

    name: str
    description: str
    config: Dict[str, Any]
    results: List[Dict[str, Any]] = field(default_factory=list)
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class ExperimentManifest:
    """Experiment metadata and tracking information"""

    experiment_id: str
    name: str
    description: str
    start_time: float
    end_time: Optional[float] = None
    phases: List[str] = field(default_factory=list)
    current_phase: Optional[str] = None
    git_commit: Optional[str] = None
    total_simulations: int = 0
    successful_simulations: int = 0
    failed_simulations: int = 0


class ConvergenceDetector:
    """Detects convergence in experiment phases"""

    def __init__(self, window_size: int = 10, improvement_threshold: float = 0.01):
        self.window_size = window_size
        self.improvement_threshold = improvement_threshold
        self.metric_history: List[float] = []

    def add_metric(self, metric: float) -> None:
        """Add a new metric value to the history"""
        self.metric_history.append(metric)
        if len(self.metric_history) > self.window_size:
            self.metric_history.pop(0)

    def has_converged(self, min_samples: int = 20) -> Tuple[bool, float]:
        """Check if the metric has converged"""
        if len(self.metric_history) < min_samples:
            return False, 0.0

        recent_avg = np.mean(self.metric_history[-self.window_size :])
        overall_avg = np.mean(self.metric_history)

        improvement = abs(recent_avg - overall_avg) / (overall_avg + 1e-10)
        return improvement < self.improvement_threshold, improvement


def run_simulation_worker(params: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker function to run a single simulation.

    Defined at module scope so it is picklable and usable with
    ProcessPoolExecutor on 'spawn' platforms (e.g., macOS, Windows).
    """
    try:
        physics_config = params.get("physics", {})
        simulation_params = {
            "plasma_density": params.get("plasma_density", 5e17),
            "laser_intensity": params.get("laser_intensity", 5e17),
            "temperature_constant": params.get("temperature_constant", 1e4),
            "magnetic_field": params.get("magnetic_field"),
            "grid_points": int(physics_config.get("grid_points", 512) or 512),
            "grid_min": float(physics_config.get("grid_min", 0.0) or 0.0),
            "grid_max": float(physics_config.get("grid_max", 50e-6) or 50e-6),
            "kappa_method": physics_config.get("kappa_method", "acoustic_exact"),
            "graybody": physics_config.get("graybody", "acoustic_wkb"),
            "alpha_gray": physics_config.get("alpha_gray", 0.8),
            "enable_hybrid": physics_config.get("enable_hybrid", False),
            "hybrid_model": physics_config.get("hybrid_model", "anabhel"),
            "mirror_D": params.get("mirror_D", 10e-6),
            "mirror_eta": params.get("mirror_eta", 1.0),
            "save_graybody_figure": physics_config.get("save_graybody_figure", False),
            "perform_kappa_inference": bool(physics_config.get("perform_kappa_inference", False)),
            "inference_calls": int(physics_config.get("inference_calls", 40) or 40),
        }
        inference_bounds = physics_config.get("inference_bounds")
        if inference_bounds:
            try:
                b0, b1 = float(inference_bounds[0]), float(inference_bounds[1])
                if b1 > b0 > 0:
                    simulation_params["inference_bounds"] = (b0, b1)
            except Exception:
                pass

        # Import inside the worker so tests can patch either
        # 'scripts.run_full_pipeline.run_full_pipeline' or
        # 'scripts.orchestration_engine.run_full_pipeline'.
        from scripts.run_full_pipeline import run_full_pipeline as _run

        result = _run(**simulation_params)
        result_dict = asdict(result)
        result_dict["simulation_success"] = True
        result_dict["parameters_used"] = simulation_params
        return result_dict

    except Exception as e:
        return {
            "simulation_success": False,
            "error": str(e),
            "parameters_used": params,
        }


class OrchestrationEngine:
    """Enhanced orchestration engine with comprehensive monitoring and validation"""

    def __init__(self, base_config_path: str = "configs/orchestration/base.yml"):
        self.base_config_path = base_config_path
        self.base_config = self._load_config(base_config_path)
        self.experiment_id = str(uuid.uuid4())[:8]
        self.phases: Dict[str, ExperimentPhase] = {}
        self.manifest: Optional[ExperimentManifest] = None
        self.convergence_detectors: Dict[str, ConvergenceDetector] = {}

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self.start_time = time.time()
        self.simulation_count = 0
        self.successful_simulations = 0

        # Initialize monitoring and validation components
        self._initialize_monitoring_components()
        self._initialize_validation_components()
        self._initialize_reporting_components()

    def _initialize_monitoring_components(self) -> None:
        """Initialize monitoring and performance tracking components"""
        from scripts.monitoring.dashboard import ExperimentDashboard
        from scripts.performance_monitor import PerformanceMonitor, ResourceManager

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor(
            experiment_id=self.experiment_id, update_interval=30.0
        )

        # Resource management
        self.resource_manager = ResourceManager(
            max_workers=self.base_config.get("parallel_workers", 4)
        )

        # Real-time dashboard
        self.dashboard = ExperimentDashboard(experiment_id=self.experiment_id, update_interval=5.0)

        self.logger.info("Initialized monitoring components")

    def _initialize_validation_components(self) -> None:
        """Initialize validation framework components"""
        from scripts.validation.benchmark_validator import BenchmarkValidator
        from scripts.validation.cross_phase_validator import CrossPhaseValidator
        from scripts.validation.physics_model_validator import PhysicsModelValidator
        from scripts.validation.quality_assurance import QualityAssuranceSystem
        from scripts.validation.validation_framework import ValidationFramework

        # Core validation framework
        self.validation_framework = ValidationFramework(self.experiment_id)

        # Quality assurance system
        self.quality_assurance = QualityAssuranceSystem(self.experiment_id)

        # Benchmark validation
        self.benchmark_validator = BenchmarkValidator(self.experiment_id)

        # Cross-phase validation
        self.cross_phase_validator = CrossPhaseValidator(self.experiment_id)

        # Physics model validation
        self.physics_validator = PhysicsModelValidator(self.experiment_id)

        self.logger.info("Initialized validation components")

    def _initialize_reporting_components(self) -> None:
        """Initialize reporting system components"""
        try:
            from scripts.reporting.integration import ReportingIntegration
            from scripts.reporting.publication_formatter import PublicationFormatter
            from scripts.reporting.report_generator import ReportGenerator
            from scripts.reporting.synthesis_engine import SynthesisEngine
            from scripts.reporting.visualization_pipeline import VisualizationPipeline

            # Core reporting components
            self.report_generator = ReportGenerator(self.experiment_id)
            self.visualization_pipeline = VisualizationPipeline(self.experiment_id)
            self.synthesis_engine = SynthesisEngine(self.experiment_id)
            self.publication_formatter = PublicationFormatter(self.experiment_id)

            # Integration system
            self.reporting_integration = ReportingIntegration(self.experiment_id)
            self.reporting_available = True

            self.logger.info("Initialized reporting components")
        except Exception as e:
            # Make reporting optional so missing plotting deps (e.g., seaborn) don't block experiments
            self.reporting_available = False
            self.report_generator = None
            self.visualization_pipeline = None
            self.synthesis_engine = None
            self.publication_formatter = None
            self.reporting_integration = None
            self.logger.warning(
                f"Reporting components unavailable ({e}); will fall back to legacy reporting."
            )

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = Path("results/orchestration/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers = [logging.StreamHandler()]
        try:
            log_file = log_dir / f"orchestration_{self.experiment_id}.log"
            # If a mocking setup replaced Path operations and yielded a directory,
            # guard against passing a directory to FileHandler.
            if hasattr(log_file, "is_dir") and callable(getattr(log_file, "is_dir")):
                if not log_file.is_dir():
                    handlers.insert(0, logging.FileHandler(log_file))
                # else: fall back to stream only
            else:
                handlers.insert(0, logging.FileHandler(log_file))
        except Exception:
            # Fall back to stream-only logging in constrained environments
            pass

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=handlers,
        )

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load YAML configuration file"""
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load configuration from {config_path}: {e}")

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries recursively"""
        result = base.copy()
        for key, value in override.items():
            if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result

    def initialize_experiment(self, experiment_name: Optional[str] = None) -> None:
        """Initialize the experiment with all phases"""
        if experiment_name is None:
            experiment_name = f"hawking_experiment_{self.experiment_id}"

        self.manifest = ExperimentManifest(
            experiment_id=self.experiment_id,
            name=experiment_name,
            description="Multi-phase analog Hawking radiation detection experiment",
            start_time=time.time(),
        )

        # Load phase configurations
        # 1) Prefer phase_*.yml next to the provided base config (test-friendly)
        import glob

        cfg_dir = os.path.dirname(self.base_config_path)
        discovered = sorted(glob.glob(os.path.join(cfg_dir, "phase_*.yml")))
        if discovered:
            # Use absolute paths to avoid Path monkeypatching in tests
            phase_paths = discovered[:]
        else:
            # 2) Fall back to repository default phases
            default_files = [
                "phase_1_initial_exploration.yml",
                "phase_2_refinement.yml",
                "phase_3_optimization.yml",
                "phase_4_validation.yml",
            ]
            phase_paths = [os.path.join("configs/orchestration", fname) for fname in default_files]

        for phase in phase_paths:
            phase_path = phase
            # Guard against test environments that monkeypatch Path to a directory
            # (e.g., pointing at a results dir). Skip if the resolved target is a directory.
            if not os.path.isdir(str(phase_path)) and os.path.exists(str(phase_path)):
                phase_config = self._load_config(str(phase_path))
                phase_name = phase_config.get(
                    "phase_name", os.path.basename(phase_path).replace(".yml", "")
                )

                # Merge with base configuration
                merged_config = self._merge_configs(self.base_config, phase_config)

                self.phases[phase_name] = ExperimentPhase(
                    name=phase_name,
                    description=phase_config.get("description", ""),
                    config=merged_config,
                )

                self.convergence_detectors[phase_name] = ConvergenceDetector(
                    window_size=merged_config.get("convergence", {}).get(
                        "moving_average_window", 10
                    ),
                    improvement_threshold=merged_config.get("convergence", {}).get(
                        "improvement_threshold", 0.01
                    ),
                )

                self.manifest.phases.append(phase_name)

        self.logger.info(
            f"Initialized experiment {self.experiment_id} with {len(self.phases)} phases"
        )

    def _run_single_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Compatibility shim retained for sequential execution/testing."""
        return run_simulation_worker(params)

    def _generate_parameter_samples(
        self, phase: ExperimentPhase, n_samples: int
    ) -> List[Dict[str, Any]]:
        """Generate parameter samples for a given phase"""
        param_space = phase.config.get("parameter_space", {})
        samples = []

        # Simple Latin Hypercube sampling for initial implementation
        # In a full implementation, this would use scipy.stats.qmc.LatinHypercube
        for i in range(n_samples):
            sample = {}
            for param_name, param_config in param_space.items():
                try:
                    if param_config.get("log_scale", False):
                        vmin = float(param_config["min"])  # may raise
                        vmax = float(param_config["max"])  # may raise
                        min_val = np.log10(vmin)
                        max_val = np.log10(vmax)
                        value = 10 ** (min_val + (max_val - min_val) * np.random.random())
                    else:
                        vmin = float(param_config["min"])  # may raise
                        vmax = float(param_config["max"])  # may raise
                        value = vmin + (vmax - vmin) * np.random.random()
                except Exception:
                    # Fallback: simple [0,1] sample if bounds invalid
                    value = float(np.random.random())
                sample[param_name] = value

            sample["physics"] = phase.config.get("physics", {})
            samples.append(sample)

        return samples

    def _check_phase_convergence(self, phase_name: str) -> Tuple[bool, float]:
        """Check if a phase has converged"""
        if phase_name not in self.convergence_detectors:
            return False, 0.0

        phase = self.phases[phase_name]
        if len(phase.results) < phase.config.get("convergence", {}).get(
            "min_samples_for_convergence", 20
        ):
            return False, 0.0

        # Calculate convergence metric (detection time improvement)
        detection_times = []
        for result in phase.results[-20:]:  # Last 20 results
            if result.get("simulation_success") and result.get("t5sigma_s"):
                detection_times.append(result["t5sigma_s"])

        if not detection_times:
            return False, 0.0

        current_metric = np.median(detection_times)
        self.convergence_detectors[phase_name].add_metric(current_metric)

        return self.convergence_detectors[phase_name].has_converged()

    def _run_phase_parallel(self, phase: ExperimentPhase) -> None:
        """Run a phase using parallel processing"""
        phase_config = phase.config
        n_workers = phase_config.get("parallel_workers", 4)
        max_iterations = phase_config.get("max_iterations", 50)

        self.logger.info(f"Starting phase {phase.name} with {n_workers} workers")
        phase.status = "running"
        try:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                iteration = 0

                while iteration < max_iterations:
                    # Generate parameter samples for this iteration
                    n_samples = min(10, max_iterations - iteration)  # Batch size
                    parameter_samples = self._generate_parameter_samples(phase, n_samples)

                    # Submit simulations
                    future_to_params = {
                        executor.submit(run_simulation_worker, params): params
                        for params in parameter_samples
                    }

                    # Collect results
                    for future in as_completed(future_to_params):
                        try:
                            result = future.result()
                            phase.results.append(result)
                            self.simulation_count += 1

                            if result.get("simulation_success"):
                                self.successful_simulations += 1
                                if result.get("t5sigma_s"):
                                    # Update convergence metric
                                    current_metric = result["t5sigma_s"]
                                    self.convergence_detectors[phase.name].add_metric(
                                        current_metric
                                    )

                            # Log progress
                            if self.simulation_count % 10 == 0:
                                success_rate = self.successful_simulations / self.simulation_count
                                self.logger.info(
                                    f"Phase {phase.name}: {self.simulation_count} simulations, "
                                    f"success rate: {success_rate:.2f}"
                                )

                        except Exception as e:
                            self.logger.error(f"Simulation failed: {e}")
                            phase.results.append({"simulation_success": False, "error": str(e)})
                            self.simulation_count += 1

                    iteration += n_samples

                    # Check convergence
                    converged, improvement = self._check_phase_convergence(phase.name)
                    if converged:
                        self.logger.info(
                            f"Phase {phase.name} converged with improvement {improvement:.4f}"
                        )
                        break
        except Exception as e:
            # Fallback to sequential execution in restricted environments
            self.logger.warning(
                f"Parallel execution unavailable ({e}); falling back to sequential runs."
            )
            iteration = 0

            while iteration < max_iterations:
                # Generate parameter samples for this iteration
                n_samples = min(10, max_iterations - iteration)  # Batch size
                parameter_samples = self._generate_parameter_samples(phase, n_samples)

                # Sequential execution
                for params in parameter_samples:
                    try:
                        result = run_simulation_worker(params)
                        phase.results.append(result)
                        self.simulation_count += 1
                        if result.get("simulation_success"):
                            self.successful_simulations += 1
                            if result.get("t5sigma_s"):
                                self.convergence_detectors[phase.name].add_metric(
                                    result["t5sigma_s"]
                                )
                    except Exception as e2:
                        self.logger.error(f"Simulation failed: {e2}")
                        phase.results.append({"simulation_success": False, "error": str(e2)})
                        self.simulation_count += 1

                iteration += n_samples

                # Check convergence
                converged, improvement = self._check_phase_convergence(phase.name)
                if converged:
                    self.logger.info(
                        f"Phase {phase.name} converged with improvement {improvement:.4f}"
                    )
                    break

        phase.status = "completed"
        self.logger.info(f"Completed phase {phase.name} with {len(phase.results)} simulations")

    def _run_bayesian_optimization_phase(self, phase: ExperimentPhase) -> None:
        """Run Bayesian optimization phase"""
        self.logger.info(f"Starting Bayesian optimization phase: {phase.name}")
        phase.status = "running"

        try:
            # Lazy import to avoid requiring scikit-optimize when not needed
            try:
                from scripts.bayesian_optimization import HawkingOptimization, OptimizationConfig
            except Exception:
                self.logger.warning(
                    "Bayesian optimization dependencies not available; "
                    "falling back to randomized search for this phase."
                )
                # Fallback: randomized parallel sampling using the same machinery
                try:
                    n_workers = phase.config.get("parallel_workers", 4)
                    n_calls = phase.config.get("bayesian_optimization", {}).get("n_calls", 50)
                    self.logger.info(
                        f"Fallback randomized search with {n_calls} samples and {n_workers} workers"
                    )
                    from concurrent.futures import ProcessPoolExecutor, as_completed

                    try:
                        with ProcessPoolExecutor(max_workers=n_workers) as executor:
                            parameter_samples = self._generate_parameter_samples(phase, n_calls)
                            future_to_params = {
                                executor.submit(run_simulation_worker, p): p
                                for p in parameter_samples
                            }
                            for future in as_completed(future_to_params):
                                try:
                                    result = future.result()
                                    phase.results.append(result)
                                    self.simulation_count += 1
                                    if result.get("simulation_success"):
                                        self.successful_simulations += 1
                                except Exception as e:
                                    self.logger.error(f"Fallback optimization sample failed: {e}")
                                    phase.results.append(
                                        {"simulation_success": False, "error": str(e)}
                                    )
                                    self.simulation_count += 1
                    except Exception as e_par:
                        # Sequential fallback
                        self.logger.warning(
                            f"Parallel fallback unavailable ({e_par}); using sequential optimization samples."
                        )
                        parameter_samples = self._generate_parameter_samples(phase, n_calls)
                        for p in parameter_samples:
                            try:
                                result = run_simulation_worker(p)
                                phase.results.append(result)
                                self.simulation_count += 1
                                if result.get("simulation_success"):
                                    self.successful_simulations += 1
                            except Exception as e_seq:
                                self.logger.error(f"Fallback optimization sample failed: {e_seq}")
                                phase.results.append(
                                    {"simulation_success": False, "error": str(e_seq)}
                                )
                                self.simulation_count += 1
                    phase.status = "completed"
                    return
                except Exception as fallback_err:
                    raise RuntimeError("Fallback randomized search failed") from fallback_err
            # Configure optimization
            opt_config = OptimizationConfig(
                n_calls=phase.config.get("bayesian_optimization", {}).get("n_calls", 50),
                n_initial_points=phase.config.get("bayesian_optimization", {}).get(
                    "n_initial_points", 10
                ),
            )

            optimizer = HawkingOptimization(
                opt_config, use_hybrid=phase.config.get("physics", {}).get("enable_hybrid", False)
            )

            # Run optimization
            results = optimizer.run_optimization()

            # Convert optimization results to standard format
            for opt_result in results.get("all_results", []):
                phase.results.append(
                    {
                        "simulation_success": True,
                        "parameters_used": opt_result.get("parameters", {}),
                        "t5sigma_s": opt_result.get("result", {}).get("t5sigma_s"),
                        "kappa": (
                            [opt_result.get("result", {}).get("kappa")]
                            if opt_result.get("result", {}).get("kappa")
                            else []
                        ),
                        "T_sig_K": opt_result.get("result", {}).get("T_sig_K"),
                        "optimization_objective": opt_result.get("objective_value"),
                    }
                )

            phase.status = "completed"
            self.logger.info(
                f"Completed Bayesian optimization phase with {len(phase.results)} results"
            )

        except Exception as e:
            self.logger.error(f"Bayesian optimization failed: {e}")
            phase.status = "failed"

    def run_experiment(self) -> None:
        """Run the complete multi-phase experiment with comprehensive monitoring and validation"""
        if not self.manifest:
            self.initialize_experiment()

        self.logger.info(f"Starting experiment {self.experiment_id}")

        # Start monitoring systems
        self._start_monitoring_systems()

        try:
            for phase_name in self.manifest.phases:
                phase = self.phases[phase_name]
                self.manifest.current_phase = phase_name

                # Update monitoring systems for phase transition
                self._notify_phase_transition(phase_name)

                self.logger.info(f"Transitioning to phase: {phase_name}")

                if "optimization" in phase_name.lower():
                    self._run_bayesian_optimization_phase(phase)
                else:
                    self._run_phase_parallel(phase)

                # Run phase-level validation
                self._run_phase_validation(phase)

                # Save intermediate results
                self._save_phase_results(phase)

                # Check if we should continue to next phase
                if phase.status == "failed":
                    self.logger.warning(f"Phase {phase_name} failed, stopping experiment")
                    break

            # Run comprehensive validation
            self._run_comprehensive_validation()

            # Finalize experiment with comprehensive reporting
            self.manifest.end_time = time.time()
            self._save_experiment_manifest()
            self._generate_comprehensive_reports()

            self.logger.info(
                f"Experiment completed in {self.manifest.end_time - self.manifest.start_time:.2f} seconds"
            )

        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            if self.manifest:
                self.manifest.end_time = time.time()
                self._save_experiment_manifest()

        finally:
            # Stop monitoring systems
            self._stop_monitoring_systems()

    def _start_monitoring_systems(self) -> None:
        """Start all monitoring and dashboard systems"""
        try:
            # Start performance monitoring
            self.performance_monitor.start_monitoring()

            # Start real-time dashboard
            self.dashboard.start_dashboard()

            # Initialize validation systems
            self.validation_framework.initialize_validation()
            self.quality_assurance.initialize_qa_system()

            self.logger.info("Started monitoring and validation systems")

        except Exception as e:
            self.logger.error(f"Failed to start monitoring systems: {e}")

    def _stop_monitoring_systems(self) -> None:
        """Stop all monitoring systems"""
        try:
            # Stop performance monitoring
            self.performance_monitor.stop_monitoring()

            # Stop dashboard
            self.dashboard.stop_dashboard()

            # Save final performance metrics
            self.performance_monitor.save_metrics()

            self.logger.info("Stopped monitoring systems")

        except Exception as e:
            self.logger.error(f"Error stopping monitoring systems: {e}")

    def _notify_phase_transition(self, phase_name: str) -> None:
        """Notify monitoring systems of phase transition"""
        try:
            # Update performance monitor
            self.performance_monitor.set_phase_progress(0.0)

            # Update optimization tracking
            self.performance_monitor.set_optimization_phase(phase_name)

            # Update dashboard
            self.dashboard.update_phase(phase_name)

            self.logger.debug(f"Notified monitoring systems of phase transition to {phase_name}")

        except Exception as e:
            self.logger.error(f"Error notifying phase transition: {e}")

    def _run_phase_validation(self, phase: ExperimentPhase) -> None:
        """Run comprehensive validation for a completed phase"""
        try:
            self.logger.info(f"Running validation for phase {phase.name}")

            # Run core validation framework
            validation_results = self.validation_framework.validate_phase(phase.name, phase.results)

            # Run quality assurance checks
            qa_results = self.quality_assurance.run_quality_checks(phase.results)

            # Run benchmark validation
            benchmark_results = self.benchmark_validator.validate_against_benchmarks(phase.results)

            # Log validation summary
            validation_passed = (
                validation_results.overall_validation_status == "passed"
                and qa_results.overall_quality_status == "passed"
                and benchmark_results.overall_benchmark_status == "passed"
            )

            if validation_passed:
                self.logger.info(f"Phase {phase.name} validation: PASSED")
            else:
                self.logger.warning(f"Phase {phase.name} validation: ISSUES DETECTED")

                # Log specific issues
                for issue in validation_results.validation_issues:
                    self.logger.warning(f"Validation issue: {issue}")
                for issue in qa_results.quality_issues:
                    self.logger.warning(f"QA issue: {issue}")
                for issue in benchmark_results.benchmark_issues:
                    self.logger.warning(f"Benchmark issue: {issue}")

            # Save validation reports
            self._save_validation_reports(
                phase.name, validation_results, qa_results, benchmark_results
            )

        except Exception as e:
            self.logger.error(f"Phase validation failed: {e}")

    def _run_comprehensive_validation(self) -> None:
        """Run comprehensive cross-phase and physics validation"""
        try:
            self.logger.info("Running comprehensive experiment validation")

            # Collect all phase results
            all_phase_results = {}
            for phase_name in self.manifest.phases:
                phase = self.phases[phase_name]
                all_phase_results[phase_name] = phase.results

            # Run cross-phase validation
            cross_phase_results = self.cross_phase_validator.validate_cross_phase_consistency(
                all_phase_results
            )

            # Run physics model validation
            physics_results = self.physics_validator.run_comprehensive_physics_validation()

            # Generate overall validation report
            overall_validation = self._generate_overall_validation_report(
                cross_phase_results, physics_results
            )

            # Save comprehensive validation
            self._save_comprehensive_validation_reports(
                cross_phase_results, physics_results, overall_validation
            )

            self.logger.info("Comprehensive validation completed")

        except Exception as e:
            self.logger.error(f"Comprehensive validation failed: {e}")

    def _save_validation_reports(
        self, phase_name: str, validation_results, qa_results, benchmark_results
    ) -> None:
        """Save phase validation reports to disk"""
        try:
            validation_dir = (
                Path("results/orchestration") / self.experiment_id / "validation" / phase_name
            )
            validation_dir.mkdir(parents=True, exist_ok=True)

            # Save validation framework results
            validation_file = validation_dir / "validation_framework_report.json"
            with open(validation_file, "w") as f:
                json.dump(asdict(validation_results), f, indent=2)

            # Save QA results
            qa_file = validation_dir / "quality_assurance_report.json"
            with open(qa_file, "w") as f:
                json.dump(asdict(qa_results), f, indent=2)

            # Save benchmark validation results
            benchmark_file = validation_dir / "benchmark_validation_report.json"
            with open(benchmark_file, "w") as f:
                json.dump(asdict(benchmark_results), f, indent=2)

            self.logger.debug(f"Saved validation reports for phase {phase_name}")

        except Exception as e:
            self.logger.error(f"Failed to save validation reports: {e}")

    def _save_comprehensive_validation_reports(
        self, cross_phase_results, physics_results, overall_validation
    ) -> None:
        """Save comprehensive validation reports"""
        try:
            validation_dir = Path("results/orchestration") / self.experiment_id / "validation"
            validation_dir.mkdir(parents=True, exist_ok=True)

            # Save cross-phase validation
            cross_phase_file = validation_dir / "cross_phase_validation_report.json"
            with open(cross_phase_file, "w") as f:
                json.dump(asdict(cross_phase_results), f, indent=2)

            # Save physics validation
            physics_file = validation_dir / "physics_validation_report.json"
            with open(physics_file, "w") as f:
                json.dump(asdict(physics_results), f, indent=2)

            # Save overall validation summary
            overall_file = validation_dir / "overall_validation_summary.json"
            with open(overall_file, "w") as f:
                json.dump(overall_validation, f, indent=2)

            self.logger.info("Saved comprehensive validation reports")

        except Exception as e:
            self.logger.error(f"Failed to save comprehensive validation reports: {e}")

    def _generate_overall_validation_report(
        self, cross_phase_results, physics_results
    ) -> Dict[str, Any]:
        """Generate overall validation report"""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": time.time(),
            "cross_phase_consistency": cross_phase_results.overall_consistency_score,
            "physics_model_consistency": physics_results.overall_physical_consistency,
            "total_validation_checks": (
                len(cross_phase_results.validation_results)
                + len(physics_results.validation_results)
            ),
            "passed_validation_checks": (
                sum(1 for r in cross_phase_results.validation_results if r.passed)
                + sum(1 for r in physics_results.validation_results if r.passed)
            ),
            "overall_validation_status": (
                "passed"
                if (
                    cross_phase_results.overall_consistency_score > 0.8
                    and physics_results.overall_physical_consistency > 0.8
                )
                else "needs_review"
            ),
            "critical_issues": (
                cross_phase_results.critical_consistency_issues
                + physics_results.critical_physics_issues
            ),
            "recommendations": (
                cross_phase_results.recommendations + physics_results.recommendations
            ),
        }

    def _save_phase_results(self, phase: ExperimentPhase) -> None:
        """Save phase results to disk"""
        results_dir = Path("results/orchestration") / self.experiment_id / phase.name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        results_file = results_dir / "simulation_results.json"
        with open(results_file, "w") as f:
            json.dump(phase.results, f, indent=2, default=str)

        # Save configuration
        config_file = results_dir / "phase_config.json"
        with open(config_file, "w") as f:
            json.dump(phase.config, f, indent=2, default=str)

        self.logger.info(f"Saved {len(phase.results)} results for phase {phase.name}")

    def _save_experiment_manifest(self) -> None:
        """Save experiment manifest"""
        if not self.manifest:
            return

        manifest_dir = Path("results/orchestration") / self.experiment_id
        manifest_dir.mkdir(parents=True, exist_ok=True)

        manifest_file = manifest_dir / "experiment_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(asdict(self.manifest), f, indent=2, default=str)

    def _generate_comprehensive_reports(self) -> None:
        """Generate comprehensive reports using the reporting system"""
        if not self.manifest:
            return

        # Skip if reporting was not initialized (e.g., seaborn not installed)
        if not getattr(self, "reporting_available", False):
            self.logger.info("Reporting not available; generating legacy final report only.")
            self._generate_legacy_final_report()
            return

        self.logger.info("Generating comprehensive reports and publications")

        try:
            # Integrate reporting system with orchestration components
            integration_success = (
                self.reporting_integration.perform_complete_integration()
                if self.reporting_integration
                else False
            )

            if not integration_success:
                self.logger.warning("Reporting integration completed with some failures")

            # Load experiment data for reporting
            if not (self.report_generator and self.report_generator.load_experiment_data()):
                self.logger.error("Failed to load experiment data for reporting")
                return

            # Generate comprehensive scientific report
            scientific_report = (
                self.report_generator.generate_scientific_report()
                if self.report_generator
                else None
            )
            self.logger.info("Generated scientific report")

            # Generate executive summary
            executive_summary = (
                self.report_generator.generate_executive_summary()
                if self.report_generator
                else None
            )
            self.logger.info("Generated executive summary")

            # Generate technical report
            technical_report = (
                self.report_generator.generate_technical_report() if self.report_generator else None
            )
            self.logger.info("Generated technical report")

            # Generate comprehensive visualizations
            visualization_bundle = (
                self.visualization_pipeline.generate_comprehensive_figures()
                if self.visualization_pipeline
                else None
            )
            self.logger.info("Generated comprehensive visualizations")

            # Perform cross-phase synthesis
            synthesis_report = (
                self.synthesis_engine.perform_comprehensive_synthesis()
                if self.synthesis_engine
                else None
            )
            self.logger.info("Generated synthesis analysis")

            # Generate publication materials
            publication_materials = (
                self.publication_formatter.generate_complete_publication_package()
                if self.publication_formatter
                else None
            )
            self.logger.info("Generated publication materials")

            # Save legacy final report for compatibility
            self._generate_legacy_final_report()

            self.logger.info("Comprehensive reporting package generated successfully")

        except Exception as e:
            self.logger.error(f"Comprehensive reporting failed: {e}")
            # Fall back to legacy reporting
            self._generate_legacy_final_report()

    def _generate_legacy_final_report(self) -> None:
        """Generate legacy final experiment report for compatibility"""
        if not self.manifest:
            return

        report_dir = Path("results/orchestration") / self.experiment_id
        report_file = report_dir / "final_report.txt"

        with open(report_file, "w") as f:
            f.write("Analog Hawking Radiation Detection Experiment Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Experiment ID: {self.experiment_id}\n")
            f.write(f"Name: {self.manifest.name}\n")
            f.write(f"Description: {self.manifest.description}\n")
            f.write(f"Start Time: {time.ctime(self.manifest.start_time)}\n")
            f.write(
                f"End Time: {time.ctime(self.manifest.end_time) if self.manifest.end_time else 'N/A'}\n"
            )
            f.write(
                f"Duration: {self.manifest.end_time - self.manifest.start_time:.2f} seconds\n\n"
            )

            f.write("Phase Summary:\n")
            f.write("-" * 40 + "\n")
            for phase_name in self.manifest.phases:
                phase = self.phases[phase_name]
                successful = sum(1 for r in phase.results if r.get("simulation_success"))
                total = len(phase.results)
                f.write(
                    f"{phase_name}: {successful}/{total} successful ({successful/total*100:.1f}%)\n"
                )

            f.write(f"\nTotal Simulations: {self.simulation_count}\n")
            f.write(f"Successful Simulations: {self.successful_simulations}\n")
            f.write(f"Success Rate: {self.successful_simulations/self.simulation_count*100:.1f}%\n")

        self.logger.info(f"Legacy final report saved to {report_file}")


def main():
    """Main entry point for the orchestration engine"""
    import argparse

    parser = argparse.ArgumentParser(description="Analog Hawking Radiation Orchestration Engine")
    parser.add_argument(
        "--config", default="configs/orchestration/base.yml", help="Base configuration file"
    )
    parser.add_argument("--name", help="Experiment name")
    parser.add_argument("--phases", nargs="+", help="Specific phases to run")

    args = parser.parse_args()

    # Initialize and run orchestration engine
    engine = OrchestrationEngine(base_config_path=args.config)
    engine.initialize_experiment(experiment_name=args.name)

    # Filter phases if specified
    if args.phases:
        engine.manifest.phases = [p for p in engine.manifest.phases if p in args.phases]

    # Run the experiment
    engine.run_experiment()


if __name__ == "__main__":
    main()
