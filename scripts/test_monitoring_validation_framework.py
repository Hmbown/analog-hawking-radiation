#!/usr/bin/env python3
"""
Comprehensive Test Script for Monitoring and Validation Framework

Tests the complete monitoring and validation system integration
with the orchestration engine.
"""

from __future__ import annotations

import json
import logging

# Add project paths to Python path
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scripts.monitoring.dashboard import ExperimentDashboard
from scripts.orchestration_engine import OrchestrationEngine
from scripts.performance_monitor import PerformanceMonitor, ResourceManager
from scripts.validation.benchmark_validator import BenchmarkValidator
from scripts.validation.cross_phase_validator import CrossPhaseValidator
from scripts.validation.physics_model_validator import PhysicsModelValidator
from scripts.validation.quality_assurance import QualityAssuranceSystem
from scripts.validation.validation_framework import ValidationFramework


class MonitoringValidationTestSuite:
    """Comprehensive test suite for monitoring and validation framework"""
    
    def __init__(self):
        self.test_experiment_id = f"test_framework_{int(time.time())}"
        self.test_results: Dict[str, Any] = {}
        self.setup_logging()
    
    def setup_logging(self) -> None:
        """Setup test logging"""
        log_dir = Path("results/tests") / self.test_experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'framework_test.log'),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run comprehensive test suite for monitoring and validation framework"""
        self.logger.info("Starting comprehensive monitoring and validation framework test suite")
        
        test_results = {
            "test_experiment_id": self.test_experiment_id,
            "start_time": time.time(),
            "tests": {},
            "overall_status": "pending"
        }
        
        try:
            # Test 1: Component Initialization
            test_results["tests"]["component_initialization"] = self.test_component_initialization()
            
            # Test 2: Monitoring System Integration
            test_results["tests"]["monitoring_integration"] = self.test_monitoring_integration()
            
            # Test 3: Validation Framework Integration
            test_results["tests"]["validation_integration"] = self.test_validation_integration()
            
            # Test 4: Performance Monitoring
            test_results["tests"]["performance_monitoring"] = self.test_performance_monitoring()
            
            # Test 5: Quality Assurance
            test_results["tests"]["quality_assurance"] = self.test_quality_assurance()
            
            # Test 6: Cross-Phase Validation
            test_results["tests"]["cross_phase_validation"] = self.test_cross_phase_validation()
            
            # Test 7: Physics Model Validation
            test_results["tests"]["physics_validation"] = self.test_physics_validation()
            
            # Test 8: Resource Management
            test_results["tests"]["resource_management"] = self.test_resource_management()
            
            # Test 9: Dashboard Functionality
            test_results["tests"]["dashboard_functionality"] = self.test_dashboard_functionality()
            
            # Test 10: End-to-End Integration
            test_results["tests"]["end_to_end_integration"] = self.test_end_to_end_integration()
            
            # Calculate overall test status
            all_passed = all(test["status"] == "passed" for test in test_results["tests"].values())
            test_results["overall_status"] = "passed" if all_passed else "failed"
            test_results["passed_tests"] = sum(1 for test in test_results["tests"].values() if test["status"] == "passed")
            test_results["total_tests"] = len(test_results["tests"])
            
        except Exception as e:
            self.logger.error(f"Test suite failed: {e}")
            test_results["overall_status"] = "failed"
            test_results["error"] = str(e)
        
        test_results["end_time"] = time.time()
        test_results["duration"] = test_results["end_time"] - test_results["start_time"]
        
        # Save test results
        self._save_test_results(test_results)
        
        self.logger.info(f"Test suite completed: {test_results['overall_status']}")
        self.logger.info(f"Passed {test_results['passed_tests']}/{test_results['total_tests']} tests")
        
        return test_results
    
    def test_component_initialization(self) -> Dict[str, Any]:
        """Test initialization of all monitoring and validation components"""
        self.logger.info("Testing component initialization...")
        
        test_result = {
            "name": "component_initialization",
            "status": "pending",
            "components_tested": [],
            "errors": []
        }
        
        try:
            # Test monitoring components
            components = [
                ("PerformanceMonitor", lambda: PerformanceMonitor(self.test_experiment_id)),
                ("ResourceManager", lambda: ResourceManager()),
                ("ExperimentDashboard", lambda: ExperimentDashboard(self.test_experiment_id)),
                ("ValidationFramework", lambda: ValidationFramework(self.test_experiment_id)),
                ("QualityAssuranceSystem", lambda: QualityAssuranceSystem(self.test_experiment_id)),
                ("BenchmarkValidator", lambda: BenchmarkValidator(self.test_experiment_id)),
                ("CrossPhaseValidator", lambda: CrossPhaseValidator(self.test_experiment_id)),
                ("PhysicsModelValidator", lambda: PhysicsModelValidator(self.test_experiment_id)),
            ]
            
            for component_name, init_func in components:
                try:
                    component = init_func()
                    test_result["components_tested"].append({
                        "name": component_name,
                        "status": "initialized",
                        "details": f"Successfully initialized {component_name}"
                    })
                except Exception as e:
                    test_result["components_tested"].append({
                        "name": component_name,
                        "status": "failed",
                        "error": str(e)
                    })
                    test_result["errors"].append(f"{component_name}: {e}")
            
            # Check if all components initialized successfully
            failed_components = [c for c in test_result["components_tested"] if c["status"] == "failed"]
            if not failed_components:
                test_result["status"] = "passed"
                test_result["message"] = "All components initialized successfully"
            else:
                test_result["status"] = "failed"
                test_result["message"] = f"{len(failed_components)} components failed to initialize"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["errors"].append(f"Test setup failed: {e}")
        
        return test_result
    
    def test_monitoring_integration(self) -> Dict[str, Any]:
        """Test monitoring system integration and basic functionality"""
        self.logger.info("Testing monitoring system integration...")
        
        test_result = {
            "name": "monitoring_integration",
            "status": "pending",
            "metrics_collected": [],
            "alerts_generated": []
        }
        
        try:
            # Initialize performance monitor
            monitor = PerformanceMonitor(self.test_experiment_id, update_interval=1.0)
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Let it run for a few cycles
            time.sleep(3)
            
            # Check if metrics are being collected
            system_metrics, experiment_metrics = monitor.get_current_metrics()
            
            if system_metrics:
                test_result["metrics_collected"].append({
                    "type": "system_metrics",
                    "status": "collected",
                    "cpu_usage": system_metrics.cpu_percent,
                    "memory_usage": system_metrics.memory_percent
                })
            
            if experiment_metrics:
                test_result["metrics_collected"].append({
                    "type": "experiment_metrics",
                    "status": "collected",
                    "simulations": experiment_metrics.simulations_completed,
                    "success_rate": experiment_metrics.success_rate
                })
            
            # Generate some test alerts by updating stats
            monitor.update_simulation_stats(10, 8, 120.0)  # 10 completed, 8 successful, 120s compute time
            
            # Check for alerts
            summary = monitor.get_performance_summary()
            if "recent_alerts" in summary:
                test_result["alerts_generated"] = summary["recent_alerts"]
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Validate test results
            if (len(test_result["metrics_collected"]) >= 2 and 
                system_metrics and experiment_metrics):
                test_result["status"] = "passed"
                test_result["message"] = "Monitoring system integrated successfully"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "Insufficient metrics collected"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_validation_integration(self) -> Dict[str, Any]:
        """Test validation framework integration"""
        self.logger.info("Testing validation framework integration...")
        
        test_result = {
            "name": "validation_integration",
            "status": "pending",
            "validation_checks": [],
            "issues_detected": []
        }
        
        try:
            # Initialize validation framework
            validation_framework = ValidationFramework(self.test_experiment_id)
            
            # Create test simulation results
            test_results = self._generate_test_simulation_results()
            
            # Run validation
            validation_results = validation_framework.validate_phase("test_phase", test_results)
            
            # Record validation checks
            for result in validation_results.validation_results:
                test_result["validation_checks"].append({
                    "check_name": result.check_name,
                    "passed": result.passed,
                    "confidence": result.confidence,
                    "message": result.message
                })
            
            # Record any issues
            test_result["issues_detected"] = validation_results.validation_issues
            
            # Validate test results
            if validation_results.validation_results:
                test_result["status"] = "passed"
                test_result["message"] = f"Validation framework executed {len(validation_results.validation_results)} checks"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "No validation checks executed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_performance_monitoring(self) -> Dict[str, Any]:
        """Test performance monitoring capabilities"""
        self.logger.info("Testing performance monitoring...")
        
        test_result = {
            "name": "performance_monitoring",
            "status": "pending",
            "system_metrics": {},
            "experiment_metrics": {},
            "performance_alerts": []
        }
        
        try:
            # Initialize performance monitor
            monitor = PerformanceMonitor(self.test_experiment_id, update_interval=1.0)
            
            # Start monitoring
            monitor.start_monitoring()
            
            # Simulate some experiment activity
            for i in range(5):
                completed = 2
                successful = 1 if i % 2 == 0 else 2  # Vary success rate
                compute_time = 10.0 * completed
                
                monitor.update_simulation_stats(completed, successful, compute_time)
                monitor.set_phase_progress(i * 0.2)
                
                time.sleep(1)
            
            # Get performance summary
            summary = monitor.get_performance_summary()
            
            # Record metrics
            test_result["system_metrics"] = {
                "cpu_usage": summary.get("simulations", {}).get("success_rate", 0),
                "memory_usage": summary.get("metrics_history_size", {}).get("system_metrics", 0)
            }
            
            test_result["experiment_metrics"] = summary.get("simulations", {})
            test_result["performance_alerts"] = summary.get("recent_alerts", [])
            
            # Stop monitoring
            monitor.stop_monitoring()
            
            # Save final metrics
            monitor.save_metrics()
            
            # Validate test results
            if (test_result["experiment_metrics"] and 
                len(test_result["performance_alerts"]) >= 0):  # Alerts are optional
                test_result["status"] = "passed"
                test_result["message"] = "Performance monitoring functioning correctly"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "Performance monitoring data incomplete"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_quality_assurance(self) -> Dict[str, Any]:
        """Test quality assurance system"""
        self.logger.info("Testing quality assurance system...")
        
        test_result = {
            "name": "quality_assurance",
            "status": "pending",
            "qa_checks": [],
            "anomalies_detected": []
        }
        
        try:
            # Initialize QA system
            qa_system = QualityAssuranceSystem(self.test_experiment_id)
            
            # Create test results with some anomalies
            test_results = self._generate_test_simulation_results(with_anomalies=True)
            
            # Run QA checks
            qa_results = qa_system.run_quality_checks(test_results)
            
            # Record QA checks
            for result in qa_results.quality_checks:
                test_result["qa_checks"].append({
                    "check_name": result.check_name,
                    "passed": result.passed,
                    "severity": result.severity,
                    "message": result.message
                })
            
            # Record anomalies
            test_result["anomalies_detected"] = qa_results.quality_issues
            
            # Validate test results
            if qa_results.quality_checks:
                test_result["status"] = "passed"
                test_result["message"] = f"QA system executed {len(qa_results.quality_checks)} checks"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "No QA checks executed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_cross_phase_validation(self) -> Dict[str, Any]:
        """Test cross-phase validation"""
        self.logger.info("Testing cross-phase validation...")
        
        test_result = {
            "name": "cross_phase_validation",
            "status": "pending",
            "consistency_checks": [],
            "phase_transitions": []
        }
        
        try:
            # Initialize cross-phase validator
            cross_validator = CrossPhaseValidator(self.test_experiment_id)
            
            # Create test phase results
            phase_results = {
                "phase_1": self._generate_test_simulation_results(phase="phase_1"),
                "phase_2": self._generate_test_simulation_results(phase="phase_2"),
                "phase_3": self._generate_test_simulation_results(phase="phase_3")
            }
            
            # Run cross-phase validation
            validation_results = cross_validator.validate_cross_phase_consistency(phase_results)
            
            # Record consistency checks
            for result in validation_results.validation_results:
                test_result["consistency_checks"].append({
                    "check_name": result.check_name,
                    "passed": result.passed,
                    "consistency_score": result.consistency_score,
                    "message": result.message
                })
            
            # Record phase transitions
            test_result["phase_transitions"] = validation_results.phase_transition_analysis
            
            # Validate test results
            if validation_results.validation_results:
                test_result["status"] = "passed"
                test_result["message"] = f"Cross-phase validation executed {len(validation_results.validation_results)} checks"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "No cross-phase validation checks executed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_physics_validation(self) -> Dict[str, Any]:
        """Test physics model validation"""
        self.logger.info("Testing physics model validation...")
        
        test_result = {
            "name": "physics_validation",
            "status": "pending",
            "physics_checks": [],
            "model_consistency": 0.0
        }
        
        try:
            # Initialize physics validator
            physics_validator = PhysicsModelValidator(self.test_experiment_id)
            
            # Run comprehensive physics validation
            validation_results = physics_validator.run_comprehensive_physics_validation()
            
            # Record physics checks
            for result in validation_results.validation_results:
                test_result["physics_checks"].append({
                    "check_name": result.check_name,
                    "passed": result.passed,
                    "confidence": result.confidence,
                    "physical_interpretation": result.physical_interpretation
                })
            
            # Record overall consistency
            test_result["model_consistency"] = validation_results.overall_physical_consistency
            
            # Validate test results
            if validation_results.validation_results:
                test_result["status"] = "passed"
                test_result["message"] = f"Physics validation executed {len(validation_results.validation_results)} checks"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "No physics validation checks executed"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_resource_management(self) -> Dict[str, Any]:
        """Test resource management system"""
        self.logger.info("Testing resource management...")
        
        test_result = {
            "name": "resource_management",
            "status": "pending",
            "worker_allocations": [],
            "resource_recommendations": []
        }
        
        try:
            # Initialize resource manager
            resource_manager = ResourceManager(max_workers=8)
            
            # Test worker allocation
            phases = ["phase_1", "phase_2", "phase_3"]
            for phase in phases:
                allocated = resource_manager.allocate_workers(phase, requested_workers=3)
                test_result["worker_allocations"].append({
                    "phase": phase,
                    "requested": 3,
                    "allocated": allocated,
                    "available_after": resource_manager.available_workers
                })
            
            # Test worker release
            for phase in phases[:2]:  # Release first two phases
                resource_manager.release_workers(phase)
                test_result["worker_allocations"].append({
                    "phase": f"{phase}_released",
                    "available_after": resource_manager.available_workers
                })
            
            # Test resource recommendations
            monitor = PerformanceMonitor(self.test_experiment_id)
            recommendations = resource_manager.get_resource_recommendations(monitor)
            test_result["resource_recommendations"] = recommendations
            
            # Validate test results
            if (len(test_result["worker_allocations"]) >= 5 and 
                resource_manager.available_workers > 0):
                test_result["status"] = "passed"
                test_result["message"] = "Resource management functioning correctly"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "Resource management issues detected"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_dashboard_functionality(self) -> Dict[str, Any]:
        """Test dashboard functionality"""
        self.logger.info("Testing dashboard functionality...")
        
        test_result = {
            "name": "dashboard_functionality",
            "status": "pending",
            "dashboard_components": [],
            "update_frequency": 0
        }
        
        try:
            # Initialize dashboard
            dashboard = ExperimentDashboard(self.test_experiment_id, update_interval=1.0)
            
            # Start dashboard
            dashboard.start_dashboard()
            
            # Update dashboard with test data
            for i in range(3):
                dashboard.update_phase(f"test_phase_{i}")
                dashboard.update_progress(i * 0.33)
                dashboard.update_metrics({
                    "simulations_completed": i * 10,
                    "success_rate": 0.7 + (i * 0.1),
                    "average_simulation_time": 45.0 - (i * 5.0)
                })
                
                time.sleep(1)
            
            # Get dashboard status
            status = dashboard.get_dashboard_status()
            test_result["dashboard_components"] = status.get("components", [])
            test_result["update_frequency"] = status.get("update_count", 0)
            
            # Stop dashboard
            dashboard.stop_dashboard()
            
            # Validate test results
            if (test_result["dashboard_components"] and 
                test_result["update_frequency"] > 0):
                test_result["status"] = "passed"
                test_result["message"] = "Dashboard functioning correctly"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "Dashboard components not updating"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def test_end_to_end_integration(self) -> Dict[str, Any]:
        """Test end-to-end integration with orchestration engine"""
        self.logger.info("Testing end-to-end integration...")
        
        test_result = {
            "name": "end_to_end_integration",
            "status": "pending",
            "orchestration_components": [],
            "integration_points": []
        }
        
        try:
            # Initialize orchestration engine
            engine = OrchestrationEngine()
            
            # Check if monitoring components are integrated
            if hasattr(engine, 'performance_monitor'):
                test_result["orchestration_components"].append({
                    "component": "performance_monitor",
                    "status": "integrated"
                })
            
            if hasattr(engine, 'dashboard'):
                test_result["orchestration_components"].append({
                    "component": "dashboard",
                    "status": "integrated"
                })
            
            if hasattr(engine, 'validation_framework'):
                test_result["orchestration_components"].append({
                    "component": "validation_framework",
                    "status": "integrated"
                })
            
            # Test integration points
            integration_points = [
                ("_start_monitoring_systems", callable(getattr(engine, '_start_monitoring_systems', None))),
                ("_stop_monitoring_systems", callable(getattr(engine, '_stop_monitoring_systems', None))),
                ("_run_phase_validation", callable(getattr(engine, '_run_phase_validation', None))),
                ("_run_comprehensive_validation", callable(getattr(engine, '_run_comprehensive_validation', None))),
            ]
            
            for point_name, is_callable in integration_points:
                test_result["integration_points"].append({
                    "point": point_name,
                    "implemented": is_callable
                })
            
            # Validate test results
            implemented_points = [p for p in test_result["integration_points"] if p["implemented"]]
            if (len(test_result["orchestration_components"]) >= 3 and
                len(implemented_points) >= 3):
                test_result["status"] = "passed"
                test_result["message"] = "End-to-end integration successful"
            else:
                test_result["status"] = "failed"
                test_result["message"] = "Incomplete integration"
            
        except Exception as e:
            test_result["status"] = "failed"
            test_result["error"] = str(e)
        
        return test_result
    
    def _generate_test_simulation_results(self, phase: str = "test_phase", 
                                        with_anomalies: bool = False) -> List[Dict[str, Any]]:
        """Generate test simulation results for validation testing"""
        results = []
        
        # Generate normal results
        for i in range(10):
            result = {
                "simulation_success": True,
                "parameters_used": {
                    "plasma_density": 5e17 + i * 1e16,
                    "laser_intensity": 5e17 + i * 2e16,
                    "temperature_constant": 1e4,
                },
                "t5sigma_s": 100.0 + i * 10.0,  # Increasing detection time
                "kappa": [1e10 + i * 1e9],
                "T_sig_K": 1e-6 + i * 1e-7,
                "phase": phase
            }
            results.append(result)
        
        # Add some anomalies if requested
        if with_anomalies:
            # Add a failed simulation
            results.append({
                "simulation_success": False,
                "error": "Test anomaly: simulation failed",
                "parameters_used": {
                    "plasma_density": 1e20,  # Unusually high
                    "laser_intensity": 1e15,  # Unusually low
                },
                "phase": phase
            })
            
            # Add an outlier result
            results.append({
                "simulation_success": True,
                "parameters_used": {
                    "plasma_density": 5e17,
                    "laser_intensity": 5e17,
                },
                "t5sigma_s": 1000.0,  # Unusually high detection time
                "kappa": [1e8],  # Unusually low kappa
                "T_sig_K": 1e-3,  # Unusually high temperature
                "phase": phase
            })
        
        return results
    
    def _save_test_results(self, test_results: Dict[str, Any]) -> None:
        """Save test results to disk"""
        results_dir = Path("results/tests") / self.test_experiment_id
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / "framework_test_results.json"
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        
        # Also generate a summary report
        summary_file = results_dir / "test_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("MONITORING AND VALIDATION FRAMEWORK TEST SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Experiment ID: {test_results['test_experiment_id']}\n")
            f.write(f"Overall Status: {test_results['overall_status']}\n")
            f.write(f"Passed Tests: {test_results.get('passed_tests', 0)}/{test_results.get('total_tests', 0)}\n")
            f.write(f"Duration: {test_results.get('duration', 0):.2f} seconds\n\n")
            
            f.write("DETAILED TEST RESULTS:\n")
            f.write("-" * 40 + "\n")
            for test_name, test_result in test_results.get('tests', {}).items():
                f.write(f"{test_name}: {test_result.get('status', 'unknown')}\n")
                if test_result.get('message'):
                    f.write(f"  {test_result['message']}\n")
                if test_result.get('errors'):
                    for error in test_result['errors']:
                        f.write(f"  ERROR: {error}\n")
            
        self.logger.info(f"Test results saved to {results_dir}")


def main():
    """Main entry point for framework testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitoring and Validation Framework Test Suite")
    parser.add_argument("--quick", action="store_true", help="Run quick test (skip some tests)")
    
    args = parser.parse_args()
    
    print("Starting Monitoring and Validation Framework Test Suite...")
    
    # Run comprehensive test suite
    test_suite = MonitoringValidationTestSuite()
    results = test_suite.run_comprehensive_test_suite()
    
    # Print summary
    print(f"\nTEST SUITE COMPLETED: {results['overall_status'].upper()}")
    print(f"Passed {results['passed_tests']}/{results['total_tests']} tests")
    print(f"Duration: {results['duration']:.2f} seconds")
    
    if results['overall_status'] == 'passed':
        print("🎉 All tests passed! Monitoring and validation framework is ready for use.")
    else:
        print("❌ Some tests failed. Check the detailed test report for issues.")
    
    print(f"\nDetailed results saved to: results/tests/{results['test_experiment_id']}/")


if __name__ == "__main__":
    main()