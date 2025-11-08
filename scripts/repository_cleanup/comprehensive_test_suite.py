#!/usr/bin/env python3
"""
Comprehensive Test Suite for Analog Hawking Radiation Simulator
===============================================================

This script performs systematic testing of all major components,
identifies issues, and provides recommendations for improvement.

Usage:
    python comprehensive_test_suite.py [--quick] [--detailed] [--output-report]
"""

import subprocess
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import importlib.util

class ComponentTester:
    def __init__(self, quick_mode=False, detailed_mode=False):
        self.quick_mode = quick_mode
        self.detailed_mode = detailed_mode
        self.results = {}
        self.start_time = time.time()
        
    def run_command(self, cmd, timeout=60, description=""):
        """Run a command and capture results."""
        print(f"Testing: {description}")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=timeout
            )
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            print(f"  {'✅ PASS' if success else '❌ FAIL'}: {description}")
            if not success and self.detailed_mode:
                print(f"  Error: {output[:200]}...")
                
            return success, output
            
        except subprocess.TimeoutExpired:
            print(f"  ⏰ TIMEOUT: {description}")
            return False, "Command timed out"
        except Exception as e:
            print(f"  💥 ERROR: {description} - {str(e)}")
            return False, str(e)

    def test_core_physics(self):
        """Test core physics components."""
        print("\n🔬 Testing Core Physics Components")
        
        tests = [
            ("pytest tests/test_horizon_kappa_analytic.py -v", "Horizon Kappa Analytics"),
            ("pytest tests/test_graybody.py -v", "Graybody Transmission"),
            ("pytest tests/test_physics_validators.py -v", "Physics Validation"),
            ("pytest tests/test_horizon_nd.py -v", "nD Horizon Detection"),
        ]
        
        if not self.quick_mode:
            tests.extend([
                ("pytest tests/test_experiment_universality.py -v", "Universality Experiments"),
                ("pytest tests/test_multi_physics_coupling.py -v", "Multi-physics Coupling"),
            ])
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def test_cli_functionality(self):
        """Test command-line interface."""
        print("\n🖥️  Testing CLI Functionality")
        
        tests = [
            ("ahr --version", "CLI Version"),
            ("ahr gpu-info", "GPU Information"),
            ("ahr quickstart --help", "Quickstart Help"),
            ("ahr regress", "Regression Tests"),
        ]
        
        if not self.quick_mode:
            tests.extend([
                ("ahr validate --help", "Validation Help"),
            ])
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def test_pipeline_execution(self):
        """Test pipeline execution."""
        print("\n⚙️  Testing Pipeline Execution")
        
        tests = [
            ("timeout 30 python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb", 
             "Basic Pipeline Demo"),
        ]
        
        if not self.quick_mode:
            tests.extend([
                ("timeout 30 python scripts/run_horizon_nd_demo.py --dim 2 --nx 20 --ny 10 --sigma 4e-7 --v0 2.0e6 --cs0 1.0e6 --x0 5e-6",
                 "2D Horizon Demo"),
                ("timeout 20 python scripts/validate_physical_configs.py",
                 "Physical Config Validation"),
            ])
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def test_advanced_features(self):
        """Test advanced features."""
        print("\n🚀 Testing Advanced Features")
        
        tests = []
        
        if not self.quick_mode:
            tests.extend([
                ("timeout 30 python enhanced_validation_framework.py --n-configs 3 --seed 42",
                 "Enhanced Validation Framework"),
                ("timeout 15 python scripts/orchestration_engine.py --config configs/orchestration/pic_downramp.yml --name test_orchestration --phases exploration",
                 "Orchestration Engine"),
                ("timeout 30 python scripts/sweep_gradient_catastrophe.py --n-samples 3 --output results/test_gradient_sweep",
                 "Gradient Catastrophe Sweep"),
            ])
        else:
            # Quick tests only
            tests.extend([
                ("python enhanced_validation_framework.py --n-configs 1 --seed 42",
                 "Enhanced Validation (Quick)"),
            ])
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def test_academic_frameworks(self):
        """Test academic collaboration frameworks."""
        print("\n🎓 Testing Academic Frameworks")
        
        tests = [
            ("python academic_collaboration_framework.py",
             "Academic Collaboration Setup"),
        ]
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def test_integration_points(self):
        """Test integration points and dependencies."""
        print("\n🔗 Testing Integration Points")
        
        tests = [
            ("pytest tests/test_doc_sync.py -v", "Documentation Sync"),
            ("pytest tests/test_cli_regress.py -v", "CLI Regression"),
        ]
        
        if not self.quick_mode:
            tests.extend([
                ("pytest tests/test_pipeline_cli_flags.py -v", "Pipeline CLI Flags"),
            ])
        
        results = {}
        for cmd, desc in tests:
            success, output = self.run_command(cmd, description=desc)
            results[desc] = {"success": success, "output": output}
            
        return results

    def generate_report(self):
        """Generate comprehensive test report."""
        print("\n📊 Generating Test Report")
        
        # Run all test suites
        physics_results = self.test_core_physics()
        cli_results = self.test_cli_functionality()
        pipeline_results = self.test_pipeline_execution()
        advanced_results = self.test_advanced_features()
        academic_results = self.test_academic_frameworks()
        integration_results = self.test_integration_points()
        
        # Compile results
        all_results = {
            "core_physics": physics_results,
            "cli_functionality": cli_results,
            "pipeline_execution": pipeline_results,
            "advanced_features": advanced_results,
            "academic_frameworks": academic_results,
            "integration_points": integration_results,
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "quick_mode": self.quick_mode,
                "detailed_mode": self.detailed_mode,
                "total_time": time.time() - self.start_time
            }
        }
        
        # Calculate summary statistics
        total_tests = 0
        passed_tests = 0
        
        for category, results in all_results.items():
            if category == "metadata":
                continue
            for test_name, result in results.items():
                total_tests += 1
                if result["success"]:
                    passed_tests += 1
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "grade": self._calculate_grade(passed_tests / total_tests if total_tests > 0 else 0)
        }
        
        all_results["summary"] = summary
        
        # Save report
        report_path = Path("results/comprehensive_test_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        self._print_summary(all_results, summary)
        
        return all_results

    def _calculate_grade(self, success_rate):
        """Calculate letter grade from success rate."""
        if success_rate >= 0.95:
            return "A+"
        elif success_rate >= 0.90:
            return "A"
        elif success_rate >= 0.85:
            return "A-"
        elif success_rate >= 0.80:
            return "B+"
        elif success_rate >= 0.75:
            return "B"
        elif success_rate >= 0.70:
            return "B-"
        else:
            return "C or lower"

    def _print_summary(self, all_results, summary):
        """Print test summary."""
        print(f"\n{'='*60}")
        print(f"COMPREHENSIVE TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Passed Tests: {summary['passed_tests']}")
        print(f"Success Rate: {summary['success_rate']:.1%}")
        print(f"Grade: {summary['grade']}")
        print(f"Execution Time: {all_results['metadata']['total_time']:.1f}s")
        print(f"Timestamp: {all_results['metadata']['timestamp']}")
        
        # Category breakdown
        print(f"\nCATEGORY BREAKDOWN:")
        for category, results in all_results.items():
            if category in ["metadata", "summary"]:
                continue
            
            category_tests = len(results)
            category_passed = sum(1 for r in results.values() if r["success"])
            category_rate = category_passed / category_tests if category_tests > 0 else 0
            
            print(f"  {category.replace('_', ' ').title()}: {category_passed}/{category_tests} ({category_rate:.1%})")
        
        # Failed tests
        failed_tests = []
        for category, results in all_results.items():
            if category in ["metadata", "summary"]:
                continue
            for test_name, result in results.items():
                if not result["success"]:
                    failed_tests.append(f"{category}: {test_name}")
        
        if failed_tests:
            print(f"\nFAILED TESTS:")
            for test in failed_tests:
                print(f"  ❌ {test}")
        
        print(f"\nReport saved to: results/comprehensive_test_report.json")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Comprehensive test suite for Analog Hawking Radiation Simulator")
    parser.add_argument("--quick", action="store_true", help="Run quick tests only")
    parser.add_argument("--detailed", action="store_true", help="Show detailed error messages")
    parser.add_argument("--output-report", action="store_true", help="Generate detailed HTML report")
    
    args = parser.parse_args()
    
    print("🚀 Starting Comprehensive Test Suite")
    print(f"Mode: {'Quick' if args.quick else 'Full'}")
    print(f"Detailed: {'Yes' if args.detailed else 'No'}")
    
    tester = ComponentTester(quick_mode=args.quick, detailed_mode=args.detailed)
    results = tester.generate_report()
    
    # Exit with appropriate code
    success_rate = results["summary"]["success_rate"]
    if success_rate >= 0.85:
        print("\n✅ Overall assessment: PASSING - Ready for production use")
        sys.exit(0)
    elif success_rate >= 0.70:
        print("\n⚠️  Overall assessment: CAUTION - Some issues detected")
        sys.exit(1)
    else:
        print("\n❌ Overall assessment: FAILING - Significant issues require attention")
        sys.exit(2)

if __name__ == "__main__":
    main()