# Comprehensive Stress Testing Framework

This directory contains a comprehensive stress testing framework for the Analog Hawking Radiation Analysis project. The framework validates system performance, memory usage, scalability, and stability under realistic research workloads.

## 🎯 Overview

The stress testing framework consists of five main components:

1. **Large-Scale Parameter Sweep Validation** (`stress_test_parameter_sweep.py`)
2. **Memory Profiling and Performance Benchmarking** (`memory_profiler_benchmark.py`)
3. **Concurrent Execution Testing** (`concurrent_execution_test.py`)
4. **Visualization and Reporting** (`stress_test_visualizer.py`)
5. **Automated Test Orchestration** (`orchestrate_stress_tests.py`)

## 🚀 Quick Start

### Basic Stress Testing

```bash
# Run a basic parameter sweep (50 configurations)
python stress_test_parameter_sweep.py --sweep-size 50 --workers 2

# Run memory profiling and benchmarking
python memory_profiler_benchmark.py --benchmark

# Run concurrent execution testing
python concurrent_execution_test.py --threads --workers 1 2 4 8

# Generate comprehensive visualizations
python stress_test_visualizer.py --stress-data results/stress_testing/stress_test_results_*.json
```

### Automated Orchestration

```bash
# Run the complete orchestrated stress test cycle
python orchestrate_stress_tests.py

# Run specific test suites
python orchestrate_stress_tests.py --suites basic_parameter_sweep memory_profiling

# Validate environment before testing
python orchestrate_stress_tests.py --validate-only

# Clean up old results
python orchestrate_stress_tests.py --cleanup
```

## 📊 Components

### 1. Parameter Sweep Stress Testing

**File**: `stress_test_parameter_sweep.py`

**Purpose**: Validate system performance with large-scale parameter variations

**Features**:
- 100+ parameter configurations with systematic variation
- Memory profiling and performance tracking
- Concurrent execution testing
- Automatic regression detection
- Performance recommendations

**Usage**:
```bash
python stress_test_parameter_sweep.py [options]
```

**Key Options**:
- `--sweep-size`: Number of configurations to test (default: 100)
- `--workers`: Number of concurrent workers (default: 4)
- `--memory-threshold`: Memory threshold in MB (default: 8192)
- `--timeout`: Timeout per configuration in seconds (default: 300)
- `--no-profiling`: Disable memory profiling
- `--no-concurrent`: Disable concurrent execution
- `--no-viz`: Disable visualization generation

### 2. Memory Profiling and Benchmarking

**File**: `memory_profiler_benchmark.py`

**Purpose**: Track memory usage, detect leaks, and establish performance baselines

**Features**:
- Real-time memory monitoring with tracemalloc
- Performance regression detection
- Memory leak detection
- Baseline establishment and comparison
- Comprehensive HTML reports

**Usage**:
```bash
python memory_profiler_benchmark.py [options]
```

**Key Options**:
- `--benchmark`: Run comprehensive benchmarks
- `--update-baselines`: Update performance baselines
- `--output-dir`: Output directory for results

### 3. Concurrent Execution Testing

**File**: `concurrent_execution_test.py`

**Purpose**: Test thread safety, scalability, and resource contention

**Features**:
- Multi-threaded and multi-process testing
- Scalability analysis and speedup measurement
- Resource contention detection
- Race condition detection
- Deadlock detection

**Usage**:
```bash
python concurrent_execution_test.py [options]
```

**Key Options**:
- `--threads`: Test thread-based execution
- `--processes`: Test process-based execution
- `--workers`: Worker counts to test (default: 1 2 4 8 16)
- `--duration`: Test duration in seconds (default: 60)

### 4. Visualization and Reporting

**File**: `stress_test_visualizer.py`

**Purpose**: Generate comprehensive visualizations and reports

**Features**:
- Interactive matplotlib and seaborn visualizations
- Performance trend analysis
- Interactive Plotly dashboards
- Comprehensive HTML reports
- Statistical analysis with confidence intervals

**Usage**:
```bash
python stress_test_visualizer.py [options]
```

**Key Options**:
- `--stress-data`: Path to stress test results JSON file
- `--concurrent-data`: Path to concurrent test results
- `--benchmark-data`: Path to benchmark results
- `--output-dir`: Output directory for visualizations

### 5. Automated Orchestration

**File**: `orchestrate_stress_tests.py`

**Purpose**: Coordinate comprehensive stress testing with automated scheduling

**Features**:
- Progressive test complexity scaling
- Dependency management
- Automatic failure recovery and retry logic
- Resource management and scheduling
- CI/CD integration capabilities
- Scheduled execution daemon

**Usage**:
```bash
python orchestrate_stress_tests.py [options]
```

**Key Options**:
- `--config`: Orchestration configuration file
- `--suites`: Specific test suites to run
- `--validate-only`: Only validate environment
- `--cleanup`: Clean up old results
- `--schedule`: Start scheduled execution daemon

## 📈 Test Suites

### Default Test Suites

1. **Basic Parameter Sweep** (`basic_parameter_sweep`)
   - 50 configurations
   - 2 concurrent workers
   - 30-minute timeout
   - Memory profiling enabled

2. **Comprehensive Parameter Sweep** (`comprehensive_parameter_sweep`)
   - 100 configurations
   - 4 concurrent workers
   - 60-minute timeout
   - Depends on basic sweep

3. **Memory Profiling** (`memory_profiling`)
   - Memory benchmarking
   - Leak detection
   - Regression analysis

4. **Concurrent Execution** (`concurrent_execution`)
   - Thread and process testing
   - Scalability analysis
   - Resource contention detection

5. **Extreme Stress Test** (`extreme_stress_test`)
   - 200 configurations
   - 8 concurrent workers
   - 120-minute timeout
   - Maximum load testing

### Custom Test Suites

Create a YAML configuration file to define custom test suites:

```yaml
test_suites:
  - name: "custom_lightweight_test"
    description: "Lightweight test for quick validation"
    test_type: "parameter_sweep"
    config:
      sweep_size: 20
      concurrent_workers: 2
      memory_threshold_mb: 2048
      timeout_per_config: 60
    enabled: true
    priority: 1
    max_retries: 2
    timeout_minutes: 15

  - name: "custom_memory_focus"
    description: "Memory-focused testing"
    test_type: "memory_benchmark"
    config: {}
    enabled: true
    priority: 2
    max_retries: 1
    timeout_minutes: 20
```

## 📁 Directory Structure

```
results/stress_testing/
├── executions/           # Individual test execution results
├── reports/             # Cycle summaries and comprehensive reports
├── visualizations/      # Plots, charts, and interactive dashboards
├── configs/             # Test configuration files
├── logs/                # Execution logs
├── baselines/           # Performance baselines
└── benchmarking/        # Benchmark-specific results
```

## 🔧 Configuration

### Orchestration Configuration

Create an orchestration configuration file:

```yaml
max_concurrent_suites: 2
resource_limits:
  memory_limit_gb: 16.0
  cpu_limit_percent: 80.0
  disk_space_gb: 10.0
notification_settings:
  email_enabled: false
  slack_enabled: false
  success_threshold: 0.8
  alert_on_failure: true
schedule_settings:
  enabled: false
  cron_expression: "0 2 * * *"  # Daily at 2 AM
  timezone: "UTC"
output_directory: "results/stress_testing"
retention_days: 30
```

### Environment Requirements

- **Python**: 3.8+
- **Memory**: Minimum 8GB, recommended 16GB+
- **CPU**: Minimum 4 cores, recommended 8+ cores
- **Disk**: Minimum 10GB free space

### Required Packages

```bash
pip install numpy matplotlib seaborn scipy psutil schedule plotly pandas pyyaml
```

## 📊 Interpreting Results

### Success Criteria

- **Success Rate**: >90% of configurations should complete successfully
- **Memory Usage**: Peak memory should stay within configured thresholds
- **Performance**: No significant regressions compared to baselines
- **Scalability**: Concurrent execution should show positive speedup

### Key Metrics

1. **Success Rate**: Percentage of configurations completed successfully
2. **Throughput**: Configurations processed per hour
3. **Memory Efficiency**: Memory usage per configuration
4. **Scalability**: Speedup factor with increasing workers
5. **Performance Consistency**: Coefficient of variation in execution times

### Alert Conditions

- **Critical**: Success rate <80%, memory threshold exceeded, system instability
- **Warning**: Success rate 80-90%, performance regressions >10%, resource contention
- **Info**: General progress notifications, completion summaries

## 🔍 Troubleshooting

### Common Issues

1. **Memory Errors**
   - Reduce `sweep_size` or `concurrent_workers`
   - Increase `memory_threshold_mb`
   - Close other applications

2. **Timeout Errors**
   - Increase `timeout_per_config`
   - Reduce parameter complexity
   - Check system load

3. **Import Errors**
   - Install required packages: `pip install -r requirements.txt`
   - Check Python path configuration

4. **Permission Errors**
   - Ensure write permissions to output directory
   - Check disk space availability

### Debug Mode

Enable debug logging:

```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m logging --level=DEBUG stress_test_parameter_sweep.py --sweep-size 5
```

### Performance Tuning

1. **For Testing Environments**
   - Use smaller `sweep_size` (20-50)
   - Reduce `concurrent_workers` (1-2)
   - Set lower memory thresholds

2. **For Production Validation**
   - Use full `sweep_size` (100+)
   - Maximize `concurrent_workers` (4-8)
   - Set realistic memory thresholds

## 🚀 CI/CD Integration

### GitHub Actions Example

```yaml
name: Stress Testing

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
  workflow_dispatch:

jobs:
  stress-test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt

    - name: Run stress tests
      run: |
        python orchestrate_stress_tests.py --validate-only
        python orchestrate_stress_tests.py --suites basic_parameter_sweep memory_profiling

    - name: Upload results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: stress-test-results
        path: results/stress_testing/
```

## 📈 Performance Baselines

The system automatically establishes and maintains performance baselines:

- **Initial Baseline**: Created on first successful run
- **Baseline Updates**: Manual updates with `--update-baselines`
- **Regression Detection**: Automatic comparison with baselines
- **Trend Analysis**: Performance tracking over time

### Baseline Metrics

- Execution time percentiles (50th, 90th, 95th)
- Memory usage statistics (mean, peak, std)
- Success rate thresholds
- Scalability coefficients

## 🔄 Scheduled Execution

Enable scheduled execution for automated testing:

```bash
# Create configuration with schedule enabled
cat > stress_test_config.yaml << EOF
schedule_settings:
  enabled: true
  cron_expression: "0 2 * * *"
  timezone: "UTC"
EOF

# Start scheduled daemon
python orchestrate_stress_tests.py --config stress_test_config.yaml --schedule
```

## 📚 Advanced Usage

### Custom Test Functions

Create custom test functions for specialized validation:

```python
def custom_physics_test():
    """Custom physics validation test"""
    from scripts.run_full_pipeline import run_full_pipeline

    summary = run_full_pipeline(
        plasma_density=1e17,
        laser_wavelength=800e-9,
        laser_intensity=1e17,
        # ... custom parameters
    )

    return {
        'kappa': summary.kappa[0] if summary.kappa else 0,
        'success': bool(summary.kappa)
    }
```

### Memory Leak Detection

Enable detailed memory leak detection:

```python
from memory_profiler_benchmark import MemoryProfiler

profiler = MemoryProfiler(sample_interval=0.01)  # High-resolution sampling
profiler.start_profiling(enable_tracemalloc=True)

# Run your code here...

peak_memory, avg_memory, growth_rate = profiler.stop_profiling()
print(f"Memory leak rate: {growth_rate:.3f} MB/s")
```

### Performance Profiling

Profile specific functions:

```python
from memory_profiler_benchmark import PerformanceBenchmarkRunner

runner = PerformanceBenchmarkRunner()
benchmark = runner.run_benchmark("custom_test", your_function, args, kwargs)
print(f"Execution time: {benchmark.execution_time:.3f}s")
print(f"Peak memory: {benchmark.memory_peak_mb:.1f}MB")
```

## 🤝 Contributing

When adding new stress tests:

1. Follow existing code patterns and style
2. Add comprehensive error handling
3. Include performance metrics collection
4. Update documentation
5. Add test cases for new functionality

## 📞 Support

For issues with the stress testing framework:

1. Check log files in `results/stress_testing/logs/`
2. Review troubleshooting section
3. Validate environment with `--validate-only`
4. Check system resources and dependencies

---

**Note**: This stress testing framework is designed for research-grade software validation. Adjust parameters and thresholds based on your specific computational resources and requirements.