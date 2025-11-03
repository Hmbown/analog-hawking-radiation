#!/usr/bin/env python3
"""
Comprehensive Performance Monitoring and Optimization Tracking System

Monitors experiment performance, system resources, optimization progress,
and provides real-time tracking with automated performance reporting.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil


@dataclass
class OptimizationMetrics:
    """Optimization progress and performance metrics"""
    timestamp: float
    phase_name: str
    best_kappa: float
    best_detection_time: float
    best_snr: float
    improvement_rate: float
    parameter_convergence: float
    optimization_efficiency: float
    optimization_iterations: int
    convergence_status: str  # "converged", "improving", "stalled", "diverging"


@dataclass
class PerformanceReport:
    """Comprehensive performance report with analysis and recommendations"""
    timestamp: float
    experiment_id: str
    report_period: Tuple[float, float]  # start and end timestamps
    system_performance_summary: Dict[str, Any]
    experiment_performance_summary: Dict[str, Any]
    optimization_progress_summary: Dict[str, Any]
    resource_utilization_analysis: Dict[str, Any]
    performance_recommendations: List[str]
    critical_issues: List[str]
    overall_health_score: float


@dataclass
class SystemMetrics:
    """System resource usage metrics"""
    timestamp: float
    cpu_percent: float
    memory_used_mb: float
    memory_total_mb: float
    memory_percent: float
    disk_used_gb: float
    disk_total_gb: float
    disk_percent: float
    active_processes: int
    load_average: Tuple[float, float, float]


@dataclass
class ExperimentMetrics:
    """Experiment-specific performance metrics"""
    timestamp: float
    simulations_completed: int
    simulations_successful: int
    simulations_failed: int
    success_rate: float
    average_simulation_time: float
    total_compute_time: float
    phase_progress: float  # 0.0 to 1.0
    estimated_phase_completion: Optional[float] = None
    estimated_experiment_completion: Optional[float] = None
    performance_issues: List[str] = field(default_factory=list)


@dataclass
class PerformanceAlert:
    """Performance alert with severity and recommendation"""
    timestamp: float
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "cpu", "memory", "disk", "performance", "stability"
    message: str
    metric: str
    value: float
    threshold: float
    recommendation: str


class PerformanceMonitor:
    """Comprehensive performance monitoring and optimization tracking"""
    
    def __init__(self, experiment_id: str, update_interval: float = 30.0):
        self.experiment_id = experiment_id
        self.update_interval = update_interval
        self.is_monitoring = False
        self.monitor_thread: Optional[Thread] = None
        self.stop_event = Event()
        
        # Metrics history
        self.system_metrics_history: List[SystemMetrics] = []
        self.experiment_metrics_history: List[ExperimentMetrics] = []
        self.optimization_metrics_history: List[OptimizationMetrics] = []
        self.alerts: List[PerformanceAlert] = []
        
        # Performance thresholds
        self.thresholds = {
            "cpu_warning": 80.0,      # CPU usage percentage
            "cpu_critical": 95.0,
            "memory_warning": 85.0,   # Memory usage percentage
            "memory_critical": 95.0,
            "disk_warning": 90.0,     # Disk usage percentage
            "disk_critical": 98.0,
            "success_rate_warning": 0.3,  # Minimum success rate
            "success_rate_critical": 0.1,
            "simulation_time_warning": 300.0,  # Maximum average simulation time (seconds)
            "optimization_stall_threshold": 0.01,  # Minimum improvement rate
            "convergence_threshold": 0.95,         # Parameter convergence threshold
        }
        
        # Experiment state tracking
        self.current_simulations = 0
        self.successful_simulations = 0
        self.failed_simulations = 0
        self.total_compute_time = 0.0
        self.phase_start_time = time.time()
        self.experiment_start_time = time.time()
        self.phase_progress = 0.0
        self.phase_total_simulations = 0
        
        # Optimization tracking
        self.current_phase = "initial_exploration"
        self.best_kappa = 0.0
        self.best_detection_time = float('inf')
        self.best_snr = 0.0
        self.optimization_iterations = 0
        self.improvement_history: List[float] = []
        self.parameter_convergence_history: List[float] = []
        
        # Performance reporting
        self.last_report_time = time.time()
        self.report_interval = 3600.0  # 1 hour
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized comprehensive performance monitor for experiment {experiment_id}")
    
    def _setup_logging(self) -> None:
        """Setup performance monitoring logging"""
        log_dir = Path("results/orchestration") / self.experiment_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'performance_monitor.log'),
                logging.StreamHandler()
            ]
        )
    
    def start_monitoring(self) -> None:
        """Start the performance monitoring thread"""
        if self.is_monitoring:
            self.logger.warning("Performance monitoring already running")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        self.monitor_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Started performance monitoring")
    
    def stop_monitoring(self) -> None:
        """Stop the performance monitoring thread"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped performance monitoring")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop with optimization tracking and automated reporting"""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                system_metrics = self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect experiment metrics
                experiment_metrics = self._collect_experiment_metrics()
                self.experiment_metrics_history.append(experiment_metrics)
                
                # Collect optimization metrics
                optimization_metrics = self._collect_optimization_metrics()
                self.optimization_metrics_history.append(optimization_metrics)
                
                # Check for performance issues
                self._check_performance_issues(system_metrics, experiment_metrics)
                
                # Check optimization performance
                self._check_optimization_performance(optimization_metrics)
                
                # Check for automated reporting
                current_time = time.time()
                if current_time - self.last_report_time >= self.report_interval:
                    self._generate_automated_performance_report()
                    self.last_report_time = current_time
                
                # Limit history size
                max_history = 1000
                if len(self.system_metrics_history) > max_history:
                    self.system_metrics_history.pop(0)
                if len(self.experiment_metrics_history) > max_history:
                    self.experiment_metrics_history.pop(0)
                if len(self.optimization_metrics_history) > max_history:
                    self.optimization_metrics_history.pop(0)
                if len(self.alerts) > 100:
                    self.alerts.pop(0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
            
            # Wait for next update
            self.stop_event.wait(self.update_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system resource metrics"""
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1.0)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_used_mb = memory.used / (1024 * 1024)
        memory_total_mb = memory.total / (1024 * 1024)
        memory_percent = memory.percent
        
        # Disk usage (current working directory)
        disk = psutil.disk_usage('.')
        disk_used_gb = disk.used / (1024 * 1024 * 1024)
        disk_total_gb = disk.total / (1024 * 1024 * 1024)
        disk_percent = disk.percent
        
        # Process count
        active_processes = len(psutil.pids())
        
        # Load average (Unix-like systems)
        if hasattr(os, 'getloadavg'):
            load_avg = os.getloadavg()
        else:
            load_avg = (0.0, 0.0, 0.0)
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_used_mb=memory_used_mb,
            memory_total_mb=memory_total_mb,
            memory_percent=memory_percent,
            disk_used_gb=disk_used_gb,
            disk_total_gb=disk_total_gb,
            disk_percent=disk_percent,
            active_processes=active_processes,
            load_average=load_avg
        )
    
    def _collect_experiment_metrics(self) -> ExperimentMetrics:
        """Collect current experiment performance metrics"""
        current_time = time.time()
        
        # Calculate success rate
        total_simulations = self.current_simulations
        if total_simulations > 0:
            success_rate = self.successful_simulations / total_simulations
        else:
            success_rate = 0.0
        
        # Calculate average simulation time
        if self.successful_simulations > 0:
            average_simulation_time = self.total_compute_time / self.successful_simulations
        else:
            average_simulation_time = 0.0
        
        # Estimate completion times
        estimated_phase_completion = None
        estimated_experiment_completion = None
        
        if self.phase_progress > 0 and self.phase_progress < 1.0:
            phase_elapsed = current_time - self.phase_start_time
            estimated_total_phase_time = phase_elapsed / self.phase_progress
            estimated_phase_completion = current_time + (estimated_total_phase_time - phase_elapsed)
        
        # Check for performance issues
        performance_issues = self._detect_performance_issues(
            success_rate, average_simulation_time
        )
        
        return ExperimentMetrics(
            timestamp=current_time,
            simulations_completed=total_simulations,
            simulations_successful=self.successful_simulations,
            simulations_failed=self.failed_simulations,
            success_rate=success_rate,
            average_simulation_time=average_simulation_time,
            total_compute_time=self.total_compute_time,
            phase_progress=self.phase_progress,
            estimated_phase_completion=estimated_phase_completion,
            estimated_experiment_completion=estimated_experiment_completion,
            performance_issues=performance_issues
        )
    
    def _detect_performance_issues(self, success_rate: float, avg_simulation_time: float) -> List[str]:
        """Detect performance issues based on metrics"""
        issues = []
        
        if success_rate < self.thresholds["success_rate_critical"]:
            issues.append(f"Critical: Very low success rate ({success_rate:.1%})")
        elif success_rate < self.thresholds["success_rate_warning"]:
            issues.append(f"Warning: Low success rate ({success_rate:.1%})")
        
        if avg_simulation_time > self.thresholds["simulation_time_warning"]:
            issues.append(f"Warning: Long simulation time ({avg_simulation_time:.1f}s)")
        
        return issues
    
    def _check_performance_issues(self, system_metrics: SystemMetrics, 
                                experiment_metrics: ExperimentMetrics) -> None:
        """Check for performance issues and generate alerts"""
        # CPU usage alerts
        if system_metrics.cpu_percent > self.thresholds["cpu_critical"]:
            self._create_alert(
                severity="critical",
                category="cpu",
                message=f"CPU usage critically high: {system_metrics.cpu_percent:.1f}%",
                metric="cpu_percent",
                value=system_metrics.cpu_percent,
                threshold=self.thresholds["cpu_critical"],
                recommendation="Consider reducing parallel workers or pausing other processes"
            )
        elif system_metrics.cpu_percent > self.thresholds["cpu_warning"]:
            self._create_alert(
                severity="high",
                category="cpu",
                message=f"CPU usage high: {system_metrics.cpu_percent:.1f}%",
                metric="cpu_percent",
                value=system_metrics.cpu_percent,
                threshold=self.thresholds["cpu_warning"],
                recommendation="Monitor CPU usage and consider optimization"
            )
        
        # Memory usage alerts
        if system_metrics.memory_percent > self.thresholds["memory_critical"]:
            self._create_alert(
                severity="critical",
                category="memory",
                message=f"Memory usage critically high: {system_metrics.memory_percent:.1f}%",
                metric="memory_percent",
                value=system_metrics.memory_percent,
                threshold=self.thresholds["memory_critical"],
                recommendation="Reduce memory usage or add swap space"
            )
        elif system_metrics.memory_percent > self.thresholds["memory_warning"]:
            self._create_alert(
                severity="high",
                category="memory",
                message=f"Memory usage high: {system_metrics.memory_percent:.1f}%",
                metric="memory_percent",
                value=system_metrics.memory_percent,
                threshold=self.thresholds["memory_warning"],
                recommendation="Monitor memory usage and consider optimization"
            )
        
        # Disk usage alerts
        if system_metrics.disk_percent > self.thresholds["disk_critical"]:
            self._create_alert(
                severity="critical",
                category="disk",
                message=f"Disk usage critically high: {system_metrics.disk_percent:.1f}%",
                metric="disk_percent",
                value=system_metrics.disk_percent,
                threshold=self.thresholds["disk_critical"],
                recommendation="Free up disk space immediately"
            )
        elif system_metrics.disk_percent > self.thresholds["disk_warning"]:
            self._create_alert(
                severity="medium",
                category="disk",
                message=f"Disk usage high: {system_metrics.disk_percent:.1f}%",
                metric="disk_percent",
                value=system_metrics.disk_percent,
                threshold=self.thresholds["disk_warning"],
                recommendation="Monitor disk usage and consider cleanup"
            )
        
        # Experiment performance alerts
        if experiment_metrics.success_rate < self.thresholds["success_rate_critical"]:
            self._create_alert(
                severity="critical",
                category="performance",
                message=f"Experiment success rate critically low: {experiment_metrics.success_rate:.1%}",
                metric="success_rate",
                value=experiment_metrics.success_rate,
                threshold=self.thresholds["success_rate_critical"],
                recommendation="Check simulation parameters and configuration"
            )
    
    def _create_alert(self, severity: str, category: str, message: str,
                     metric: str, value: float, threshold: float, recommendation: str) -> None:
        """Create a new performance alert"""
        alert = PerformanceAlert(
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            metric=metric,
            value=value,
            threshold=threshold,
            recommendation=recommendation
        )
        
        self.alerts.append(alert)
        
        # Log based on severity
        if severity == "critical":
            self.logger.critical(f"PERFORMANCE ALERT: {message} - {recommendation}")
        elif severity == "high":
            self.logger.error(f"Performance warning: {message} - {recommendation}")
        elif severity == "medium":
            self.logger.warning(f"Performance notice: {message} - {recommendation}")
        else:
            self.logger.info(f"Performance info: {message}")
    
    def _collect_optimization_metrics(self) -> OptimizationMetrics:
        """Collect optimization progress and performance metrics"""
        current_time = time.time()
        
        # Calculate improvement rate (average improvement over last 10 iterations)
        if len(self.improvement_history) >= 2:
            improvement_rate = np.mean(self.improvement_history[-10:]) if self.improvement_history else 0.0
        else:
            improvement_rate = 0.0
        
        # Calculate parameter convergence (how stable parameters are)
        if len(self.parameter_convergence_history) >= 2:
            parameter_convergence = np.mean(self.parameter_convergence_history[-10:]) if self.parameter_convergence_history else 0.0
        else:
            parameter_convergence = 0.0
        
        # Calculate optimization efficiency (improvement per iteration)
        if self.optimization_iterations > 0:
            optimization_efficiency = improvement_rate / self.optimization_iterations
        else:
            optimization_efficiency = 0.0
        
        # Determine convergence status
        if parameter_convergence > self.thresholds["convergence_threshold"]:
            convergence_status = "converged"
        elif improvement_rate > self.thresholds["optimization_stall_threshold"]:
            convergence_status = "improving"
        elif improvement_rate > 0:
            convergence_status = "stalled"
        else:
            convergence_status = "diverging"
        
        return OptimizationMetrics(
            timestamp=current_time,
            phase_name=self.current_phase,
            best_kappa=self.best_kappa,
            best_detection_time=self.best_detection_time,
            best_snr=self.best_snr,
            improvement_rate=improvement_rate,
            parameter_convergence=parameter_convergence,
            optimization_efficiency=optimization_efficiency,
            optimization_iterations=self.optimization_iterations,
            convergence_status=convergence_status
        )
    
    def _check_optimization_performance(self, optimization_metrics: OptimizationMetrics) -> None:
        """Check optimization performance and generate alerts"""
        # Check for optimization stall
        if (optimization_metrics.convergence_status == "stalled" and
            self.optimization_iterations > 10):
            self._create_alert(
                severity="medium",
                category="optimization",
                message=f"Optimization stalled: improvement rate {optimization_metrics.improvement_rate:.3f}",
                metric="improvement_rate",
                value=optimization_metrics.improvement_rate,
                threshold=self.thresholds["optimization_stall_threshold"],
                recommendation="Consider adjusting optimization parameters or exploring different regions"
            )
        
        # Check for divergence
        if optimization_metrics.convergence_status == "diverging":
            self._create_alert(
                severity="high",
                category="optimization",
                message=f"Optimization diverging: negative improvement rate {optimization_metrics.improvement_rate:.3f}",
                metric="improvement_rate",
                value=optimization_metrics.improvement_rate,
                threshold=0.0,
                recommendation="Review optimization strategy and parameter bounds"
            )
        
        # Check for slow convergence
        if (optimization_metrics.optimization_efficiency < 0.001 and
            self.optimization_iterations > 20):
            self._create_alert(
                severity="low",
                category="optimization",
                message=f"Slow optimization progress: efficiency {optimization_metrics.optimization_efficiency:.6f}",
                metric="optimization_efficiency",
                value=optimization_metrics.optimization_efficiency,
                threshold=0.001,
                recommendation="Consider increasing exploration or adjusting convergence criteria"
            )
    
    def update_optimization_progress(self, kappa: float, detection_time: float, snr: float,
                                   improvement: float, convergence: float) -> None:
        """Update optimization progress metrics"""
        # Update best values if improved
        if kappa > self.best_kappa:
            self.best_kappa = kappa
        if detection_time < self.best_detection_time:
            self.best_detection_time = detection_time
        if snr > self.best_snr:
            self.best_snr = snr
        
        # Track improvement and convergence
        self.improvement_history.append(improvement)
        self.parameter_convergence_history.append(convergence)
        self.optimization_iterations += 1
        
        # Limit history size
        max_history = 100
        if len(self.improvement_history) > max_history:
            self.improvement_history.pop(0)
        if len(self.parameter_convergence_history) > max_history:
            self.parameter_convergence_history.pop(0)
    
    def set_optimization_phase(self, phase_name: str) -> None:
        """Set current optimization phase and reset tracking if needed"""
        if phase_name != self.current_phase:
            self.current_phase = phase_name
            # Reset some metrics for new phase
            self.improvement_history.clear()
            self.parameter_convergence_history.clear()
            self.optimization_iterations = 0
            self.logger.info(f"Switched to optimization phase: {phase_name}")
    
    def _generate_automated_performance_report(self) -> None:
        """Generate automated performance report and save to disk"""
        try:
            report = self._create_comprehensive_performance_report()
            
            # Save report to disk
            output_dir = Path("results/orchestration") / self.experiment_id / "performance_reports"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = output_dir / f"performance_report_{timestamp}.json"
            
            with open(report_file, 'w') as f:
                json.dump(asdict(report), f, indent=2)
            
            self.logger.info(f"Generated automated performance report: {report_file}")
            
            # Also generate text summary
            text_report = self._generate_comprehensive_text_report(report)
            text_file = output_dir / f"performance_summary_{timestamp}.txt"
            
            with open(text_file, 'w') as f:
                f.write(text_report)
            
        except Exception as e:
            self.logger.error(f"Failed to generate automated performance report: {e}")
    
    def _create_comprehensive_performance_report(self) -> PerformanceReport:
        """Create comprehensive performance report with analysis and recommendations"""
        current_time = time.time()
        
        # System performance analysis
        system_summary = self._analyze_system_performance()
        
        # Experiment performance analysis
        experiment_summary = self._analyze_experiment_performance()
        
        # Optimization progress analysis
        optimization_summary = self._analyze_optimization_progress()
        
        # Resource utilization analysis
        resource_analysis = self._analyze_resource_utilization()
        
        # Generate recommendations
        recommendations = self._generate_performance_recommendations(
            system_summary, experiment_summary, optimization_summary
        )
        
        # Identify critical issues
        critical_issues = self._identify_critical_issues()
        
        # Calculate overall health score (0.0 to 1.0)
        health_score = self._calculate_health_score(
            system_summary, experiment_summary, optimization_summary
        )
        
        return PerformanceReport(
            timestamp=current_time,
            experiment_id=self.experiment_id,
            report_period=(self.last_report_time, current_time),
            system_performance_summary=system_summary,
            experiment_performance_summary=experiment_summary,
            optimization_progress_summary=optimization_summary,
            resource_utilization_analysis=resource_analysis,
            performance_recommendations=recommendations,
            critical_issues=critical_issues,
            overall_health_score=health_score
        )
    
    def _analyze_system_performance(self) -> Dict[str, Any]:
        """Analyze system resource usage and performance"""
        if not self.system_metrics_history:
            return {"error": "No system metrics available"}
        
        recent_metrics = self.system_metrics_history[-10:]  # Last 10 samples
        
        cpu_usage = [m.cpu_percent for m in recent_metrics]
        memory_usage = [m.memory_percent for m in recent_metrics]
        disk_usage = [m.disk_percent for m in recent_metrics]
        
        return {
            "cpu_usage": {
                "average": np.mean(cpu_usage),
                "max": np.max(cpu_usage),
                "trend": "increasing" if cpu_usage[-1] > cpu_usage[0] else "decreasing",
                "status": "normal" if np.mean(cpu_usage) < 80 else "high"
            },
            "memory_usage": {
                "average": np.mean(memory_usage),
                "max": np.max(memory_usage),
                "trend": "increasing" if memory_usage[-1] > memory_usage[0] else "decreasing",
                "status": "normal" if np.mean(memory_usage) < 85 else "high"
            },
            "disk_usage": {
                "average": np.mean(disk_usage),
                "max": np.max(disk_usage),
                "trend": "increasing" if disk_usage[-1] > disk_usage[0] else "decreasing",
                "status": "normal" if np.mean(disk_usage) < 90 else "high"
            },
            "system_stability": "stable" if len([a for a in self.alerts if a.severity == "critical"]) == 0 else "unstable"
        }
    
    def _analyze_experiment_performance(self) -> Dict[str, Any]:
        """Analyze experiment performance and progress"""
        if not self.experiment_metrics_history:
            return {"error": "No experiment metrics available"}
        
        recent_metrics = self.experiment_metrics_history[-10:]
        
        success_rates = [m.success_rate for m in recent_metrics]
        simulation_times = [m.average_simulation_time for m in recent_metrics]
        
        return {
            "success_rate": {
                "current": success_rates[-1],
                "average": np.mean(success_rates),
                "trend": "improving" if success_rates[-1] > success_rates[0] else "declining",
                "status": "good" if np.mean(success_rates) > 0.7 else "needs_attention"
            },
            "simulation_efficiency": {
                "average_time": np.mean(simulation_times),
                "trend": "improving" if simulation_times[-1] < simulation_times[0] else "declining",
                "status": "efficient" if np.mean(simulation_times) < 60 else "slow"
            },
            "progress": {
                "phase_progress": self.phase_progress,
                "estimated_completion": self.experiment_metrics_history[-1].estimated_phase_completion,
                "status": "on_track" if self.phase_progress > 0.5 else "early_stage"
            }
        }
    
    def _analyze_optimization_progress(self) -> Dict[str, Any]:
        """Analyze optimization progress and efficiency"""
        if not self.optimization_metrics_history:
            return {"error": "No optimization metrics available"}
        
        recent_metrics = self.optimization_metrics_history[-10:]
        
        improvement_rates = [m.improvement_rate for m in recent_metrics]
        convergence_scores = [m.parameter_convergence for m in recent_metrics]
        
        return {
            "current_phase": self.current_phase,
            "best_metrics": {
                "kappa": self.best_kappa,
                "detection_time": self.best_detection_time,
                "snr": self.best_snr
            },
            "optimization_efficiency": {
                "improvement_rate": np.mean(improvement_rates),
                "convergence": np.mean(convergence_scores),
                "iterations": self.optimization_iterations,
                "status": self.optimization_metrics_history[-1].convergence_status
            },
            "progress_assessment": self._assess_optimization_progress()
        }
    
    def _assess_optimization_progress(self) -> str:
        """Assess overall optimization progress"""
        if not self.optimization_metrics_history:
            return "not_started"
        
        latest = self.optimization_metrics_history[-1]
        
        if latest.convergence_status == "converged":
            return "completed"
        elif latest.convergence_status == "improving":
            return "making_good_progress"
        elif latest.convergence_status == "stalled":
            return "stalled_needs_attention"
        else:  # diverging
            return "struggling_needs_intervention"
    
    def _analyze_resource_utilization(self) -> Dict[str, Any]:
        """Analyze resource utilization efficiency"""
        if not self.system_metrics_history or not self.experiment_metrics_history:
            return {"error": "Insufficient data for resource analysis"}
        
        recent_system = self.system_metrics_history[-10:]
        recent_experiment = self.experiment_metrics_history[-10:]
        
        avg_cpu = np.mean([m.cpu_percent for m in recent_system])
        avg_memory = np.mean([m.memory_percent for m in recent_system])
        success_rate = np.mean([m.success_rate for m in recent_experiment])
        
        # Calculate resource efficiency score
        cpu_efficiency = min(avg_cpu / 80.0, 1.0)  # Target 80% utilization
        memory_efficiency = min(avg_memory / 85.0, 1.0)  # Target 85% utilization
        compute_efficiency = success_rate  # Higher success rate = better compute usage
        
        overall_efficiency = (cpu_efficiency + memory_efficiency + compute_efficiency) / 3.0
        
        return {
            "cpu_utilization": avg_cpu,
            "memory_utilization": avg_memory,
            "compute_efficiency": success_rate,
            "overall_efficiency_score": overall_efficiency,
            "efficiency_rating": "excellent" if overall_efficiency > 0.8 else "good" if overall_efficiency > 0.6 else "needs_improvement"
        }
    
    def _generate_performance_recommendations(self, system_summary: Dict[str, Any],
                                           experiment_summary: Dict[str, Any],
                                           optimization_summary: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []
        
        # System recommendations
        if "error" not in system_summary:
            cpu_status = system_summary["cpu_usage"]["status"]
            memory_status = system_summary["memory_usage"]["status"]
            disk_status = system_summary["disk_usage"]["status"]
            
            if cpu_status == "high":
                recommendations.append("Consider reducing parallel workers to lower CPU usage")
            if memory_status == "high":
                recommendations.append("Optimize memory usage by reducing batch sizes or using more efficient data structures")
            if disk_status == "high":
                recommendations.append("Clean up temporary files and consider archiving old results")
        
        # Experiment recommendations
        if "error" not in experiment_summary:
            success_status = experiment_summary["success_rate"]["status"]
            efficiency_status = experiment_summary["simulation_efficiency"]["status"]
            
            if success_status == "needs_attention":
                recommendations.append("Review simulation parameters to improve success rate")
            if efficiency_status == "slow":
                recommendations.append("Optimize physics parameters to reduce simulation times")
        
        # Optimization recommendations
        if "error" not in optimization_summary:
            progress_assessment = optimization_summary["progress_assessment"]
            
            if progress_assessment in ["stalled_needs_attention", "struggling_needs_intervention"]:
                recommendations.append("Adjust optimization strategy or explore different parameter regions")
            elif progress_assessment == "making_good_progress":
                recommendations.append("Continue current optimization approach - progress is good")
        
        return recommendations
    
    def _identify_critical_issues(self) -> List[str]:
        """Identify critical performance issues"""
        critical_issues = []
        
        # Check for recent critical alerts
        recent_critical_alerts = [
            alert for alert in self.alerts[-20:]
            if alert.severity == "critical"
        ]
        
        for alert in recent_critical_alerts:
            critical_issues.append(f"{alert.category}: {alert.message}")
        
        # Check for system resource exhaustion
        if self.system_metrics_history:
            latest_system = self.system_metrics_history[-1]
            if latest_system.memory_percent > 95:
                critical_issues.append("Memory usage critically high - risk of system instability")
            if latest_system.disk_percent > 98:
                critical_issues.append("Disk space critically low - risk of data loss")
        
        # Check for experiment failure
        if self.experiment_metrics_history:
            latest_experiment = self.experiment_metrics_history[-1]
            if latest_experiment.success_rate < 0.1:
                critical_issues.append("Experiment success rate critically low")
        
        return critical_issues
    
    def _calculate_health_score(self, system_summary: Dict[str, Any],
                              experiment_summary: Dict[str, Any],
                              optimization_summary: Dict[str, Any]) -> float:
        """Calculate overall system health score (0.0 to 1.0)"""
        scores = []
        
        # System health (40% weight)
        if "error" not in system_summary:
            system_score = 0.0
            system_score += 0.4 if system_summary["cpu_usage"]["status"] == "normal" else 0.1
            system_score += 0.3 if system_summary["memory_usage"]["status"] == "normal" else 0.1
            system_score += 0.3 if system_summary["disk_usage"]["status"] == "normal" else 0.1
            scores.append(system_score)
        else:
            scores.append(0.5)  # Default score if no data
        
        # Experiment health (35% weight)
        if "error" not in experiment_summary:
            experiment_score = 0.0
            experiment_score += 0.5 if experiment_summary["success_rate"]["status"] == "good" else 0.2
            experiment_score += 0.3 if experiment_summary["simulation_efficiency"]["status"] == "efficient" else 0.1
            experiment_score += 0.2 if experiment_summary["progress"]["status"] == "on_track" else 0.1
            scores.append(experiment_score)
        else:
            scores.append(0.5)
        
        # Optimization health (25% weight)
        if "error" not in optimization_summary:
            optimization_score = 0.0
            progress = optimization_summary["progress_assessment"]
            if progress == "completed":
                optimization_score = 1.0
            elif progress == "making_good_progress":
                optimization_score = 0.8
            elif progress == "stalled_needs_attention":
                optimization_score = 0.4
            else:  # struggling or not started
                optimization_score = 0.2
            scores.append(optimization_score)
        else:
            scores.append(0.5)
        
        # Weighted average
        weights = [0.4, 0.35, 0.25]
        weighted_score = sum(score * weight for score, weight in zip(scores, weights))
        
        return weighted_score
    
    def _generate_comprehensive_text_report(self, report: PerformanceReport) -> str:
        """Generate comprehensive text performance report"""
        text = f"COMPREHENSIVE PERFORMANCE REPORT - Experiment {report.experiment_id}\n"
        text += "=" * 70 + "\n\n"
        
        text += f"Report Period: {datetime.fromtimestamp(report.report_period[0]).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(report.report_period[1]).strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += f"Overall Health Score: {report.overall_health_score:.3f}/1.0\n\n"
        
        # System Performance
        text += "SYSTEM PERFORMANCE\n"
        text += "-" * 30 + "\n"
        if "error" not in report.system_performance_summary:
            sys = report.system_performance_summary
            text += f"CPU Usage: {sys['cpu_usage']['average']:.1f}% ({sys['cpu_usage']['status']}, {sys['cpu_usage']['trend']})\n"
            text += f"Memory Usage: {sys['memory_usage']['average']:.1f}% ({sys['memory_usage']['status']}, {sys['memory_usage']['trend']})\n"
            text += f"Disk Usage: {sys['disk_usage']['average']:.1f}% ({sys['disk_usage']['status']}, {sys['disk_usage']['trend']})\n"
            text += f"System Stability: {sys['system_stability']}\n"
        else:
            text += "No system performance data available\n"
        text += "\n"
        
        # Experiment Performance
        text += "EXPERIMENT PERFORMANCE\n"
        text += "-" * 30 + "\n"
        if "error" not in report.experiment_performance_summary:
            exp = report.experiment_performance_summary
            text += f"Success Rate: {exp['success_rate']['current']:.1%} ({exp['success_rate']['status']}, {exp['success_rate']['trend']})\n"
            text += f"Simulation Efficiency: {exp['simulation_efficiency']['average_time']:.2f}s ({exp['simulation_efficiency']['status']})\n"
            text += f"Progress: {exp['progress']['phase_progress']:.1%} ({exp['progress']['status']})\n"
        else:
            text += "No experiment performance data available\n"
        text += "\n"
        
        # Optimization Progress
        text += "OPTIMIZATION PROGRESS\n"
        text += "-" * 30 + "\n"
        if "error" not in report.optimization_progress_summary:
            opt = report.optimization_progress_summary
            text += f"Current Phase: {opt['current_phase']}\n"
            text += f"Best Kappa: {opt['best_metrics']['kappa']:.2e}\n"
            text += f"Best Detection Time: {opt['best_metrics']['detection_time']:.2e}s\n"
            text += f"Best SNR: {opt['best_metrics']['snr']:.2f}\n"
            text += f"Improvement Rate: {opt['optimization_efficiency']['improvement_rate']:.4f}\n"
            text += f"Convergence: {opt['optimization_efficiency']['convergence']:.3f}\n"
            text += f"Status: {opt['optimization_efficiency']['status']}\n"
            text += f"Progress Assessment: {opt['progress_assessment']}\n"
        else:
            text += "No optimization progress data available\n"
        text += "\n"
        
        # Resource Utilization
        text += "RESOURCE UTILIZATION\n"
        text += "-" * 30 + "\n"
        if "error" not in report.resource_utilization_analysis:
            res = report.resource_utilization_analysis
            text += f"CPU Utilization: {res['cpu_utilization']:.1f}%\n"
            text += f"Memory Utilization: {res['memory_utilization']:.1f}%\n"
            text += f"Compute Efficiency: {res['compute_efficiency']:.1%}\n"
            text += f"Overall Efficiency Score: {res['overall_efficiency_score']:.3f} ({res['efficiency_rating']})\n"
        else:
            text += "No resource utilization data available\n"
        text += "\n"
        
        # Critical Issues
        text += "CRITICAL ISSUES\n"
        text += "-" * 30 + "\n"
        if report.critical_issues:
            for issue in report.critical_issues:
                text += f"• {issue}\n"
        else:
            text += "No critical issues detected\n"
        text += "\n"
        
        # Recommendations
        text += "PERFORMANCE RECOMMENDATIONS\n"
        text += "-" * 30 + "\n"
        if report.performance_recommendations:
            for rec in report.performance_recommendations:
                text += f"• {rec}\n"
        else:
            text += "No specific recommendations at this time\n"
        
        return text
    
    def update_simulation_stats(self, completed: int, successful: int, 
                              compute_time: float) -> None:
        """Update simulation statistics"""
        self.current_simulations += completed
        self.successful_simulations += successful
        self.failed_simulations += (completed - successful)
        self.total_compute_time += compute_time
        # Make a metrics snapshot available even if background monitoring
        # is not running, so callers can still get a summary.
        try:
            snapshot = self._collect_experiment_metrics()
            self.experiment_metrics_history.append(snapshot)
        except Exception:
            pass
    
    def set_phase_progress(self, progress: float, total_simulations: int = 0) -> None:
        """Update phase progress and reset phase timer if starting new phase"""
        if progress == 0.0:  # Starting new phase
            self.phase_start_time = time.time()
            self.phase_total_simulations = total_simulations
        
        self.phase_progress = progress
        # Persist a snapshot if not running background monitor
        try:
            snapshot = self._collect_experiment_metrics()
            self.experiment_metrics_history.append(snapshot)
        except Exception:
            pass
    
    def get_current_metrics(self) -> Tuple[SystemMetrics, ExperimentMetrics]:
        """Get the most recent metrics"""
        system_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
        experiment_metrics = self.experiment_metrics_history[-1] if self.experiment_metrics_history else None
        
        return system_metrics, experiment_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance metrics"""
        if not self.experiment_metrics_history:
            return {"error": "No metrics collected yet"}
        
        latest_exp_metrics = self.experiment_metrics_history[-1]
        
        # Calculate trends
        success_rate_trend = self._calculate_trend("success_rate")
        simulation_time_trend = self._calculate_trend("average_simulation_time")
        
        # Get recent alerts
        recent_alerts = [alert for alert in self.alerts[-10:]]  # Last 10 alerts
        
        return {
            "experiment_id": self.experiment_id,
            "current_time": time.time(),
            "simulations": {
                "total": latest_exp_metrics.simulations_completed,
                "successful": latest_exp_metrics.simulations_successful,
                "failed": latest_exp_metrics.simulations_failed,
                "success_rate": latest_exp_metrics.success_rate,
                "success_rate_trend": success_rate_trend,
                "average_time_per_simulation": latest_exp_metrics.average_simulation_time,
                "simulation_time_trend": simulation_time_trend,
                "total_compute_time": latest_exp_metrics.total_compute_time
            },
            "progress": {
                "phase_progress": latest_exp_metrics.phase_progress,
                "estimated_phase_completion": latest_exp_metrics.estimated_phase_completion,
                "estimated_experiment_completion": latest_exp_metrics.estimated_experiment_completion
            },
            "performance_issues": latest_exp_metrics.performance_issues,
            "recent_alerts": [asdict(alert) for alert in recent_alerts],
            "metrics_history_size": {
                "system_metrics": len(self.system_metrics_history),
                "experiment_metrics": len(self.experiment_metrics_history),
                "alerts": len(self.alerts)
            }
        }
    
    def _calculate_trend(self, metric_name: str, window: int = 10) -> str:
        """Calculate trend for a specific metric"""
        if len(self.experiment_metrics_history) < window:
            return "insufficient_data"
        
        # Extract metric values
        metric_values = []
        for metrics in self.experiment_metrics_history[-window:]:
            value = getattr(metrics, metric_name, None)
            if value is not None:
                metric_values.append(value)
        
        if len(metric_values) < 2:
            return "insufficient_data"
        
        # Simple linear trend
        x = np.arange(len(metric_values))
        slope, _, r_value, _, _ = linregress(x, metric_values)
        
        if abs(slope) < 1e-6:
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def generate_performance_report(self) -> str:
        """Generate a human-readable performance report"""
        summary = self.get_performance_summary()
        
        report = f"Performance Report - Experiment {self.experiment_id}\n"
        report += "=" * 50 + "\n\n"
        
        report += "Simulation Statistics:\n"
        report += f"  Total Simulations: {summary['simulations']['total']}\n"
        report += f"  Successful: {summary['simulations']['successful']}\n"
        report += f"  Failed: {summary['simulations']['failed']}\n"
        report += f"  Success Rate: {summary['simulations']['success_rate']:.1%}\n"
        report += f"  Average Time per Simulation: {summary['simulations']['average_time_per_simulation']:.2f}s\n"
        report += f"  Total Compute Time: {summary['simulations']['total_compute_time']:.2f}s\n\n"
        
        report += "Progress:\n"
        report += f"  Phase Progress: {summary['progress']['phase_progress']:.1%}\n"
        
        if summary['progress']['estimated_phase_completion']:
            eta = datetime.fromtimestamp(summary['progress']['estimated_phase_completion'])
            report += f"  Estimated Phase Completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        if summary['progress']['estimated_experiment_completion']:
            eta = datetime.fromtimestamp(summary['progress']['estimated_experiment_completion'])
            report += f"  Estimated Experiment Completion: {eta.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        report += "\nPerformance Issues:\n"
        if summary['performance_issues']:
            for issue in summary['performance_issues']:
                report += f"  - {issue}\n"
        else:
            report += "  No significant issues detected\n"
        
        report += "\nRecent Alerts:\n"
        if summary['recent_alerts']:
            for alert in summary['recent_alerts']:
                report += f"  [{alert['severity'].upper()}] {alert['message']}\n"
        else:
            report += "  No recent alerts\n"
        
        return report
    
    def save_metrics(self) -> None:
        """Save metrics to disk for later analysis"""
        output_dir = Path("results/orchestration") / self.experiment_id / "performance"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save system metrics
        system_file = output_dir / f"system_metrics_{timestamp}.json"
        with open(system_file, 'w') as f:
            json.dump([asdict(m) for m in self.system_metrics_history], f, indent=2)
        
        # Save experiment metrics
        experiment_file = output_dir / f"experiment_metrics_{timestamp}.json"
        with open(experiment_file, 'w') as f:
            json.dump([asdict(m) for m in self.experiment_metrics_history], f, indent=2)
        
        # Save alerts
        alerts_file = output_dir / f"alerts_{timestamp}.json"
        with open(alerts_file, 'w') as f:
            json.dump([asdict(a) for a in self.alerts], f, indent=2)
        
        self.logger.info(f"Saved performance metrics to {output_dir}")


class ResourceManager:
    """Manages computational resources for optimal performance"""
    
    def __init__(self, max_workers: int = None):
        self.max_workers = max_workers or os.cpu_count()
        self.available_workers = self.max_workers
        self.worker_allocations: Dict[str, int] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def allocate_workers(self, phase_name: str, requested_workers: int) -> int:
        """Allocate workers for a phase, returning actual allocated count"""
        if requested_workers <= self.available_workers:
            allocated = requested_workers
        else:
            allocated = self.available_workers
            self.logger.warning(
                f"Requested {requested_workers} workers for {phase_name}, "
                f"but only {allocated} available"
            )
        
        self.worker_allocations[phase_name] = allocated
        self.available_workers -= allocated
        
        self.logger.info(f"Allocated {allocated} workers to {phase_name}")
        return allocated
    
    def release_workers(self, phase_name: str) -> None:
        """Release workers from a completed phase"""
        if phase_name in self.worker_allocations:
            released = self.worker_allocations[phase_name]
            self.available_workers += released
            del self.worker_allocations[phase_name]
            
            self.logger.info(f"Released {released} workers from {phase_name}")
    
    def get_resource_recommendations(self, monitor: PerformanceMonitor) -> List[str]:
        """Get resource optimization recommendations"""
        recommendations = []
        
        system_metrics, experiment_metrics = monitor.get_current_metrics()
        
        if not system_metrics or not experiment_metrics:
            return ["Insufficient data for recommendations"]
        
        # CPU-based recommendations
        if system_metrics.cpu_percent > 90:
            recommendations.append("CPU usage very high - consider reducing parallel workers")
        elif system_metrics.cpu_percent < 50 and self.available_workers > 0:
            recommendations.append("CPU usage low - could increase parallel workers")
        
        # Memory-based recommendations
        if system_metrics.memory_percent > 90:
            recommendations.append("Memory usage very high - consider reducing batch sizes")
        
        # Performance-based recommendations
        if experiment_metrics.success_rate < 0.5:
            recommendations.append("Low success rate - consider parameter space adjustment")
        
        if experiment_metrics.average_simulation_time > 300:
            recommendations.append("Long simulation times - consider optimizing physics parameters")
        
        return recommendations


def main():
    """Demo and testing for performance monitoring"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitoring System")
    parser.add_argument("--demo", action="store_true", help="Run demo monitoring")
    parser.add_argument("--duration", type=float, default=60.0, help="Demo duration in seconds")
    
    args = parser.parse_args()
    
    if args.demo:
        print("Starting performance monitoring demo...")
        
        monitor = PerformanceMonitor("demo_experiment", update_interval=5.0)
        monitor.start_monitoring()
        
        # Simulate some activity
        start_time = time.time()
        simulation_count = 0
        
        try:
            while time.time() - start_time < args.duration:
                # Simulate completing simulations
                completed = np.random.randint(1, 5)
                successful = np.random.randint(0, completed + 1)
                compute_time = np.random.uniform(1.0, 10.0) * completed
                
                monitor.update_simulation_stats(completed, successful, compute_time)
                
                # Update progress
                progress = min(1.0, simulation_count / 100.0)
                monitor.set_phase_progress(progress)
                
                simulation_count += completed
                
                # Print status every 10 seconds
                if int(time.time() - start_time) % 10 == 0:
                    summary = monitor.get_performance_summary()
                    print(f"\nProgress: {progress:.1%}, "
                          f"Simulations: {summary['simulations']['total']}, "
                          f"Success Rate: {summary['simulations']['success_rate']:.1%}")
                
                time.sleep(2.0)
        
        except KeyboardInterrupt:
            print("\nDemo interrupted by user")
        
        finally:
            monitor.stop_monitoring()
            print("\nFinal Performance Report:")
            print(monitor.generate_performance_report())
    
    else:
        print("Performance Monitoring System")
        print("Use --demo to run a demonstration")
        print("Use --duration to set demo duration (default: 60 seconds)")


# Import required for trend calculation
try:
    from scipy.stats import linregress
except ImportError:
    # Fallback implementation
    def linregress(x, y):
        """Simple linear regression fallback"""
        n = len(x)
        if n < 2:
            return 0, 0, 0, 0, 0
        
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        
        if denominator == 0:
            return 0, 0, 0, 0, 0
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # Calculate R-squared
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y_mean) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return slope, intercept, r_squared, 0, 0


if __name__ == "__main__":
    main()
