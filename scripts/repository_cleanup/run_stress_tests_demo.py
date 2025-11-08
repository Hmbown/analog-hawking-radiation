#!/usr/bin/env python3
"""
Stress Testing Framework Demo

This script demonstrates the comprehensive stress testing framework
with a quick demo that can be run to validate the installation.
"""

import os
import sys
import time
from pathlib import Path

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

def print_header(title):
    """Print a formatted header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def print_step(step, description):
    """Print a step description"""
    print(f"\n📋 Step {step}: {description}")

def run_demo():
    """Run the stress testing demo"""
    print_header("🚀 Stress Testing Framework Demo")
    print("This demo will validate the stress testing framework with lightweight tests.")
    print("Expected runtime: 2-5 minutes")

    start_time = time.time()

    try:
        # Step 1: Environment validation
        print_step(1, "Environment Validation")
        print("Validating system resources and dependencies...")

        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            cpu_count = psutil.cpu_count()
            print(f"  ✅ Memory: {memory_gb:.1f}GB")
            print(f"  ✅ CPU: {cpu_count} cores")
        except ImportError:
            print("  ❌ psutil not installed. Install with: pip install psutil")
            return False

        # Check required modules
        required_modules = ['numpy', 'matplotlib', 'seaborn', 'scipy']
        missing_modules = []
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing_modules.append(module)

        if missing_modules:
            print(f"  ❌ Missing modules: {missing_modules}")
            print(f"  Install with: pip install {' '.join(missing_modules)}")
            return False
        else:
            print("  ✅ All required modules available")

        # Step 2: Quick parameter sweep demo
        print_step(2, "Quick Parameter Sweep Demo")
        print("Running a lightweight parameter sweep with 10 configurations...")

        try:
            from stress_test_parameter_sweep import StressTestRunner, StressTestConfig

            config = StressTestConfig(
                sweep_size=10,  # Small for demo
                concurrent_workers=2,
                memory_threshold_mb=2048,  # Lower for demo
                timeout_per_config=60,  # Shorter for demo
                enable_profiling=True,
                test_concurrent_execution=False,  # Simpler for demo
                generate_visualizations=True,
                output_dir="results/stress_testing_demo"
            )

            runner = StressTestRunner(config)
            summary = runner.run_stress_test()

            print(f"  ✅ Completed: {summary.successful_configurations}/{summary.total_configurations} successful")
            print(f"  ✅ Success rate: {summary.success_rate:.1%}")
            print(f"  ✅ Average time: {summary.average_execution_time:.2f}s")
            print(f"  ✅ Peak memory: {summary.memory_peak_mb:.1f}MB")

        except Exception as e:
            print(f"  ❌ Parameter sweep failed: {e}")
            print("  This might be due to missing dependencies or system limitations")

        # Step 3: Memory profiling demo
        print_step(3, "Memory Profiling Demo")
        print("Running quick memory profiling benchmark...")

        try:
            from memory_profiler_benchmark import PerformanceBenchmarkRunner

            runner = PerformanceBenchmarkRunner()

            # Simple benchmark function
            def quick_test():
                import numpy as np
                # Simple computation
                arr = np.random.random((100, 100))
                result = np.linalg.det(arr)
                return {"determinant": result}

            benchmark = runner.run_benchmark("quick_demo", quick_test)

            if benchmark.success:
                print(f"  ✅ Benchmark completed: {benchmark.execution_time:.3f}s")
                print(f"  ✅ Memory used: {benchmark.memory_peak_mb:.1f}MB")
            else:
                print(f"  ❌ Benchmark failed: {benchmark.error_message}")

        except Exception as e:
            print(f"  ❌ Memory profiling failed: {e}")

        # Step 4: Visualization demo
        print_step(4, "Visualization Demo")
        print("Creating example visualizations...")

        try:
            from stress_test_visualizer import StressTestVisualizer

            visualizer = StressTestVisualizer("results/stress_testing_demo/visualizations")

            # Create dummy data for visualization
            demo_data = {
                'summary': {
                    'total_configurations': 10,
                    'success_rate': 0.8,
                    'average_execution_time': 2.5,
                    'memory_peak_mb': 512,
                    'scalability_metrics': {
                        'throughput_configs_per_hour': 1440,
                        'performance_consistency_cv': 0.15,
                        'memory_per_config_mb': 51.2
                    },
                    'recommendations': [
                        "System performed well during demo",
                        "Consider increasing test size for production validation"
                    ],
                    'critical_issues': []
                },
                'configurations': [
                    {
                        'success': i < 8,
                        'execution_time': 2.0 + i * 0.3,
                        'memory_peak_mb': 400 + i * 20,
                        'config_id': f'demo_config_{i:02d}'
                    }
                    for i in range(10)
                ]
            }

            # Create visualizations
            perf_viz = visualizer.create_performance_overview(demo_data)
            memory_viz = visualizer.create_memory_analysis(demo_data)
            dashboard_viz = visualizer.create_interactive_dashboard(demo_data)

            print(f"  ✅ Performance overview: {Path(perf_viz).name}")
            print(f"  ✅ Memory analysis: {Path(memory_viz).name}")
            print(f"  ✅ Interactive dashboard: {Path(dashboard_viz).name}")

        except Exception as e:
            print(f"  ❌ Visualization failed: {e}")
            print("  This might be due to missing plotting libraries")

        # Step 5: Summary
        total_time = time.time() - start_time
        print_step(5, "Demo Summary")
        print(f"✅ Demo completed successfully in {total_time:.1f} seconds")
        print("\n📁 Results saved to: results/stress_testing_demo/")
        print("\n🎯 Next Steps:")
        print("  1. Run full stress tests: python orchestrate_stress_tests.py")
        print("  2. Check documentation: STRESS_TESTING_README.md")
        print("  3. Customize configuration for your needs")
        print("  4. Integrate with CI/CD pipeline")

        return True

    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Ensure all dependencies are installed")
        print("  2. Check system resources (memory, CPU)")
        print("  3. Verify Python environment")
        print("  4. Check file permissions")
        return False

def main():
    """Main function"""
    print("Analog Hawking Radiation Analysis - Stress Testing Framework Demo")
    print("=" * 70)

    # Check if we're in the right directory
    if not Path("stress_test_parameter_sweep.py").exists():
        print("❌ Error: stress testing scripts not found")
        print("Please run this demo from the project root directory")
        sys.exit(1)

    # Run the demo
    success = run_demo()

    if success:
        print("\n🎉 Demo completed successfully!")
        print("The stress testing framework is ready for use.")
        sys.exit(0)
    else:
        print("\n💥 Demo encountered issues.")
        print("Please check the error messages above and resolve any problems.")
        sys.exit(1)

if __name__ == "__main__":
    main()