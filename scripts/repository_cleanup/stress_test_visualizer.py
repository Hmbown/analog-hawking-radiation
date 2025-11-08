#!/usr/bin/env python3
"""
Comprehensive Stress Testing Visualization and Reporting System

This module provides advanced visualization and reporting capabilities for stress testing results,
including interactive dashboards, performance trend analysis, and comprehensive HTML reports.

Features:
- Interactive matplotlib and seaborn visualizations
- Performance trend analysis and forecasting
- Memory leak detection visualization
- Scalability analysis charts
- Comprehensive HTML report generation
- Real-time dashboard creation
- Statistical analysis and confidence intervals
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Add project paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Set up matplotlib
plt.use('Agg')
sns.set_style("whitegrid")
sns.set_palette("husl")

# Configure plotly
px.defaults.template = "plotly_white"


@dataclass
class StressTestReport:
    """Comprehensive stress test report data structure"""
    timestamp: str
    test_type: str
    total_configurations: int
    success_rate: float
    performance_metrics: Dict[str, float]
    memory_metrics: Dict[str, float]
    scalability_metrics: Dict[str, float]
    regression_alerts: List[Dict[str, Any]]
    recommendations: List[str]
    visualizations: Dict[str, str]  # paths to visualization files


class StressTestVisualizer:
    """Advanced visualization system for stress testing results"""

    def __init__(self, output_dir: str = "results/stress_testing/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "dashboards").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Visualization settings
        self.figure_size = (12, 8)
        self.dpi = 300
        self.color_palette = sns.color_palette("husl", 10)

    def _setup_logging(self):
        """Setup logging for visualizer"""
        log_dir = self.output_dir / "logs"
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"visualizer_{timestamp}.log"

        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))

        logger = logging.getLogger(f"{__name__}.Visualizer")
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

    def load_stress_test_results(self, results_file: str) -> Dict[str, Any]:
        """Load stress test results from JSON file"""
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
            self.logger.info(f"Loaded stress test results from {results_file}")
            return data
        except Exception as e:
            self.logger.error(f"Error loading results from {results_file}: {e}")
            return {}

    def create_performance_overview(self, data: Dict[str, Any]) -> str:
        """Create comprehensive performance overview visualization"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "plots" / f"performance_overview_{timestamp}.png"

        # Extract data
        summary = data.get('summary', {})
        configurations = data.get('configurations', [])

        if not configurations:
            self.logger.warning("No configuration data available for performance overview")
            return str(output_file)

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Stress Test Performance Overview', fontsize=16, fontweight='bold')

        # Prepare data
        successful = [c for c in configurations if c.get('success', False)]
        failed = [c for c in configurations if not c.get('success', False)]

        # 1. Success rate pie chart
        success_counts = {'Successful': len(successful), 'Failed': len(failed)}
        axes[0, 0].pie(success_counts.values(), labels=success_counts.keys(), autopct='%1.1f%%',
                      colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[0, 0].set_title('Success Rate Distribution')

        # 2. Execution time distribution
        if successful:
            exec_times = [c.get('execution_time', 0) for c in successful]
            axes[0, 1].hist(exec_times, bins=30, alpha=0.7, color='#3498db', edgecolor='black')
            axes[0, 1].set_xlabel('Execution Time (seconds)')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Execution Time Distribution')
            axes[0, 1].axvline(np.mean(exec_times), color='red', linestyle='--',
                              label=f'Mean: {np.mean(exec_times):.2f}s')
            axes[0, 1].legend()

        # 3. Memory usage analysis
        memory_usage = [c.get('memory_peak_mb', 0) for c in configurations]
        if memory_usage:
            axes[0, 2].hist(memory_usage, bins=30, alpha=0.7, color='#9b59b6', edgecolor='black')
            axes[0, 2].set_xlabel('Peak Memory Usage (MB)')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].set_title('Memory Usage Distribution')

            # Add threshold line if available
            threshold = summary.get('memory_threshold_mb', 8192)
            axes[0, 2].axvline(threshold, color='red', linestyle='--',
                              label=f'Threshold: {threshold}MB')
            axes[0, 2].legend()

        # 4. Performance metrics scatter plot
        if successful:
            exec_times = [c.get('execution_time', 0) for c in successful]
            memory_vals = [c.get('memory_peak_mb', 0) for c in successful]

            scatter = axes[1, 0].scatter(exec_times, memory_vals, alpha=0.6, c=range(len(successful)),
                                       cmap='viridis', s=30)
            axes[1, 0].set_xlabel('Execution Time (seconds)')
            axes[1, 0].set_ylabel('Memory Usage (MB)')
            axes[1, 0].set_title('Performance: Time vs Memory')
            plt.colorbar(scatter, ax=axes[1, 0], label='Configuration Index')

        # 5. Performance timeline (if timestamps available)
        if configurations:
            config_indices = list(range(len(configurations)))
            exec_times = [c.get('execution_time', 0) for c in configurations]

            axes[1, 1].plot(config_indices, exec_times, 'o-', alpha=0.7, markersize=4)
            axes[1, 1].set_xlabel('Configuration Index')
            axes[1, 1].set_ylabel('Execution Time (seconds)')
            axes[1, 1].set_title('Execution Timeline')
            axes[1, 1].grid(True, alpha=0.3)

        # 6. Key metrics summary
        axes[1, 2].axis('off')
        metrics_text = f"""
        Key Performance Metrics:

        Total Configurations: {summary.get('total_configurations', 0)}
        Success Rate: {summary.get('success_rate', 0):.1%}
        Average Execution Time: {summary.get('average_execution_time', 0):.3f}s
        Peak Memory Usage: {summary.get('memory_peak_mb', 0):.1f}MB
        Throughput: {summary.get('scalability_metrics', {}).get('throughput_configs_per_hour', 0):.1f} configs/hour

        Resource Efficiency:
        Performance Consistency: {summary.get('scalability_metrics', {}).get('performance_consistency_cv', 0):.3f}
        Memory per Config: {summary.get('scalability_metrics', {}).get('memory_per_config_mb', 0):.1f}MB
        """

        axes[1, 2].text(0.1, 0.9, metrics_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created performance overview: {output_file}")
        return str(output_file)

    def create_memory_analysis(self, data: Dict[str, Any]) -> str:
        """Create detailed memory analysis visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "plots" / f"memory_analysis_{timestamp}.png"

        configurations = data.get('configurations', [])
        if not configurations:
            self.logger.warning("No configuration data available for memory analysis")
            return str(output_file)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Memory Usage Analysis', fontsize=16, fontweight='bold')

        # Extract memory data
        memory_peak = [c.get('memory_peak_mb', 0) for c in configurations]
        memory_avg = [c.get('memory_avg_mb', 0) for c in configurations]
        exec_times = [c.get('execution_time', 0) for c in configurations]

        # 1. Memory usage distribution comparison
        axes[0, 0].hist(memory_peak, bins=30, alpha=0.7, label='Peak Memory', color='#e74c3c')
        axes[0, 0].hist(memory_avg, bins=30, alpha=0.7, label='Average Memory', color='#3498db')
        axes[0, 0].set_xlabel('Memory Usage (MB)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Memory Usage Distribution')
        axes[0, 0].legend()

        # 2. Memory vs Execution time correlation
        if exec_times and memory_peak:
            valid_indices = [i for i, (t, m) in enumerate(zip(exec_times, memory_peak)) if t > 0 and m > 0]
            if valid_indices:
                times_valid = [exec_times[i] for i in valid_indices]
                memory_valid = [memory_peak[i] for i in valid_indices]

                axes[0, 1].scatter(times_valid, memory_valid, alpha=0.6, s=30)

                # Add trend line
                if len(times_valid) > 1:
                    z = np.polyfit(times_valid, memory_valid, 1)
                    p = np.poly1d(z)
                    axes[0, 1].plot(times_valid, p(times_valid), "r--", alpha=0.8)

                    # Calculate correlation
                    correlation = np.corrcoef(times_valid, memory_valid)[0, 1]
                    axes[0, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                                   transform=axes[0, 1].transAxes)

        axes[0, 1].set_xlabel('Execution Time (seconds)')
        axes[0, 1].set_ylabel('Peak Memory (MB)')
        axes[0, 1].set_title('Memory vs Execution Time')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Memory efficiency analysis
        memory_efficiency = []
        for i, config in enumerate(configurations):
            exec_time = config.get('execution_time', 0)
            memory = config.get('memory_peak_mb', 0)
            if exec_time > 0 and memory > 0:
                efficiency = exec_time / memory  # Time per MB
                memory_efficiency.append(efficiency)

        if memory_efficiency:
            axes[1, 0].hist(memory_efficiency, bins=30, alpha=0.7, color='#27ae60', edgecolor='black')
            axes[1, 0].set_xlabel('Execution Time per MB (s/MB)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Memory Efficiency Distribution')
            axes[1, 0].axvline(np.mean(memory_efficiency), color='red', linestyle='--',
                              label=f'Mean: {np.mean(memory_efficiency):.4f}s/MB')
            axes[1, 0].legend()

        # 4. Memory usage timeline with threshold
        config_indices = list(range(len(configurations)))
        axes[1, 1].fill_between(config_indices, 0, memory_avg, alpha=0.3, label='Average Memory', color='#3498db')
        axes[1, 1].plot(config_indices, memory_peak, 'o-', alpha=0.7, label='Peak Memory', color='#e74c3c', markersize=3)

        # Add threshold line
        threshold = data.get('summary', {}).get('memory_threshold_mb', 8192)
        axes[1, 1].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold}MB')

        axes[1, 1].set_xlabel('Configuration Index')
        axes[1, 1].set_ylabel('Memory Usage (MB)')
        axes[1, 1].set_title('Memory Usage Timeline')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created memory analysis: {output_file}")
        return str(output_file)

    def create_scalability_analysis(self, thread_data: Dict[str, Any], process_data: Dict[str, Any]) -> str:
        """Create scalability comparison analysis"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "plots" / f"scalability_analysis_{timestamp}.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Scalability Analysis: Threads vs Processes', fontsize=16, fontweight='bold')

        def extract_summary_data(data):
            summaries = data.get('summaries', [])
            return {
                'workers': [s.get('worker_count', 0) for s in summaries],
                'success_rates': [s.get('success_rate', 0) for s in summaries],
                'throughputs': [s.get('throughput_tasks_per_second', 0) for s in summaries],
                'speedups': [s.get('speedup_factor', 0) for s in summaries],
                'efficiencies': [s.get('efficiency_percent', 0) for s in summaries],
                'memory_peaks': [s.get('memory_usage_stats', {}).get('peak', 0) for s in summaries]
            }

        thread_summary = extract_summary_data(thread_data)
        process_summary = extract_summary_data(process_data)

        # 1. Speedup comparison
        axes[0, 0].plot(thread_summary['workers'], thread_summary['speedups'], 'bo-', label='Threads', linewidth=2)
        axes[0, 0].plot(process_summary['workers'], process_summary['speedups'], 'ro-', label='Processes', linewidth=2)

        # Add ideal speedup line
        if thread_summary['workers']:
            max_workers = max(max(thread_summary['workers']), max(process_summary['workers']))
            ideal_workers = list(range(1, max_workers + 1))
            axes[0, 0].plot(ideal_workers, ideal_workers, 'g--', alpha=0.7, label='Ideal Speedup')

        axes[0, 0].set_xlabel('Number of Workers')
        axes[0, 0].set_ylabel('Speedup Factor')
        axes[0, 0].set_title('Speedup Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Throughput comparison
        axes[0, 1].plot(thread_summary['workers'], thread_summary['throughputs'], 'bo-', label='Threads', linewidth=2)
        axes[0, 1].plot(process_summary['workers'], process_summary['throughputs'], 'ro-', label='Processes', linewidth=2)
        axes[0, 1].set_xlabel('Number of Workers')
        axes[0, 1].set_ylabel('Throughput (tasks/sec)')
        axes[0, 1].set_title('Throughput Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Efficiency comparison
        axes[0, 2].plot(thread_summary['workers'], thread_summary['efficiencies'], 'bo-', label='Threads', linewidth=2)
        axes[0, 2].plot(process_summary['workers'], process_summary['efficiencies'], 'ro-', label='Processes', linewidth=2)
        axes[0, 2].set_xlabel('Number of Workers')
        axes[0, 2].set_ylabel('Efficiency (%)')
        axes[0, 2].set_title('Parallel Efficiency')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 120)

        # 4. Success rate comparison
        axes[1, 0].plot(thread_summary['workers'], thread_summary['success_rates'], 'bo-', label='Threads', linewidth=2)
        axes[1, 0].plot(process_summary['workers'], process_summary['success_rates'], 'ro-', label='Processes', linewidth=2)
        axes[1, 0].set_xlabel('Number of Workers')
        axes[1, 0].set_ylabel('Success Rate')
        axes[1, 0].set_title('Success Rate Comparison')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(0, 1.1)

        # 5. Memory usage comparison
        axes[1, 1].plot(thread_summary['workers'], thread_summary['memory_peaks'], 'bo-', label='Threads', linewidth=2)
        axes[1, 1].plot(process_summary['workers'], process_summary['memory_peaks'], 'ro-', label='Processes', linewidth=2)
        axes[1, 1].set_xlabel('Number of Workers')
        axes[1, 1].set_ylabel('Peak Memory (MB)')
        axes[1, 1].set_title('Memory Usage Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. Summary statistics table
        axes[1, 2].axis('off')

        # Calculate summary statistics
        thread_max_speedup = max(thread_summary['speedups']) if thread_summary['speedups'] else 0
        process_max_speedup = max(process_summary['speedups']) if process_summary['speedups'] else 0
        thread_avg_efficiency = np.mean(thread_summary['efficiencies']) if thread_summary['efficiencies'] else 0
        process_avg_efficiency = np.mean(process_summary['efficiencies']) if process_summary['efficiencies'] else 0

        summary_text = f"""
        Scalability Summary:

        Threads:
        Max Speedup: {thread_max_speedup:.2f}x
        Avg Efficiency: {thread_avg_efficiency:.1f}%
        Max Throughput: {max(thread_summary['throughputs']) if thread_summary['throughputs'] else 0:.2f} tasks/s

        Processes:
        Max Speedup: {process_max_speedup:.2f}x
        Avg Efficiency: {process_avg_efficiency:.1f}%
        Max Throughput: {max(process_summary['throughputs']) if process_summary['throughputs'] else 0:.2f} tasks/s

        Recommendation:
        {'Threads' if thread_max_speedup > process_max_speedup else 'Processes'}
        show better scalability for this workload.
        """

        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created scalability analysis: {output_file}")
        return str(output_file)

    def create_interactive_dashboard(self, data: Dict[str, Any]) -> str:
        """Create interactive Plotly dashboard"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "dashboards" / f"interactive_dashboard_{timestamp}.html"

        configurations = data.get('configurations', [])
        if not configurations:
            self.logger.warning("No configuration data available for dashboard")
            return str(output_file)

        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Execution Time Distribution', 'Memory Usage Timeline',
                          'Performance Scatter Plot', 'Success Rate Analysis',
                          'Resource Efficiency', 'Performance Metrics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )

        # Extract data
        successful = [c for c in configurations if c.get('success', False)]
        failed = [c for c in configurations if not c.get('success', False)]

        # 1. Execution time distribution
        if successful:
            exec_times = [c.get('execution_time', 0) for c in successful]
            fig.add_trace(
                go.Histogram(x=exec_times, name='Execution Time', nbinsx=30),
                row=1, col=1
            )

        # 2. Memory usage timeline
        config_indices = list(range(len(configurations)))
        memory_peak = [c.get('memory_peak_mb', 0) for c in configurations]
        memory_avg = [c.get('memory_avg_mb', 0) for c in configurations]

        fig.add_trace(
            go.Scatter(x=config_indices, y=memory_peak, mode='lines+markers',
                      name='Peak Memory', line=dict(color='red')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=config_indices, y=memory_avg, mode='lines',
                      name='Average Memory', line=dict(color='blue'), fill='tonexty'),
            row=1, col=2
        )

        # 3. Performance scatter plot
        if successful:
            exec_times = [c.get('execution_time', 0) for c in successful]
            memory_vals = [c.get('memory_peak_mb', 0) for c in successful]

            fig.add_trace(
                go.Scatter(x=exec_times, y=memory_vals, mode='markers',
                          name='Configurations', text=[f"Config {i}" for i in range(len(successful))]),
                row=2, col=1
            )

        # 4. Success rate analysis
        success_data = {
            'Successful': len(successful),
            'Failed': len(failed)
        }

        fig.add_trace(
            go.Pie(labels=list(success_data.keys()), values=list(success_data.values()),
                  name="Success Rate"),
            row=2, col=2
        )

        # 5. Resource efficiency
        if successful:
            efficiency_scores = []
            for c in successful:
                exec_time = c.get('execution_time', 0)
                memory = c.get('memory_peak_mb', 0)
                if exec_time > 0 and memory > 0:
                    # Simple efficiency score: inverse of time*memory
                    efficiency = 1 / (exec_time * memory / 1000)  # Normalize
                    efficiency_scores.append(efficiency)
                else:
                    efficiency_scores.append(0)

            fig.add_trace(
                go.Histogram(x=efficiency_scores, name='Efficiency Score', nbinsx=20),
                row=3, col=1
            )

        # 6. Performance metrics table
        summary = data.get('summary', {})
        table_data = [
            ['Metric', 'Value'],
            ['Total Configurations', str(summary.get('total_configurations', 0))],
            ['Success Rate', f"{summary.get('success_rate', 0):.1%}"],
            ['Average Execution Time', f"{summary.get('average_execution_time', 0):.3f}s"],
            ['Peak Memory Usage', f"{summary.get('memory_peak_mb', 0):.1f}MB"],
            ['Throughput', f"{summary.get('scalability_metrics', {}).get('throughput_configs_per_hour', 0):.1f} configs/hour"],
            ['Performance Consistency', f"{summary.get('scalability_metrics', {}).get('performance_consistency_cv', 0):.3f}"]
        ]

        fig.add_trace(
            go.Table(
                header=dict(values=table_data[0], fill_color='lightblue'),
                cells=dict(values=list(zip(*table_data[1:])), fill_color='lightgrey')
            ),
            row=3, col=2
        )

        # Update layout
        fig.update_layout(
            title_text="Stress Test Interactive Dashboard",
            showlegend=True,
            height=1200,
            width=1400
        )

        # Update axes labels
        fig.update_xaxes(title_text="Execution Time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_xaxes(title_text="Configuration Index", row=1, col=2)
        fig.update_yaxes(title_text="Memory (MB)", row=1, col=2)
        fig.update_xaxes(title_text="Execution Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Memory (MB)", row=2, col=1)
        fig.update_xaxes(title_text="Efficiency Score", row=3, col=1)
        fig.update_yaxes(title_text="Count", row=3, col=1)

        # Save interactive dashboard
        pyo.plot(fig, filename=str(output_file), auto_open=False)

        self.logger.info(f"Created interactive dashboard: {output_file}")
        return str(output_file)

    def generate_comprehensive_report(self, stress_data: Dict[str, Any],
                                   concurrent_data: Optional[Dict[str, Any]] = None,
                                   benchmark_data: Optional[Dict[str, Any]] = None) -> str:
        """Generate comprehensive HTML report"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / "reports" / f"comprehensive_report_{timestamp}.html"

        # Generate all visualizations
        viz_files = {}

        # Stress test visualizations
        if stress_data:
            viz_files['performance_overview'] = self.create_performance_overview(stress_data)
            viz_files['memory_analysis'] = self.create_memory_analysis(stress_data)
            viz_files['interactive_dashboard'] = self.create_interactive_dashboard(stress_data)

        # Concurrent execution visualizations
        if concurrent_data and 'thread_data' in concurrent_data and 'process_data' in concurrent_data:
            viz_files['scalability_analysis'] = self.create_scalability_analysis(
                concurrent_data['thread_data'], concurrent_data['process_data']
            )

        # Generate HTML content
        html_content = self._generate_html_report(stress_data, concurrent_data, benchmark_data, viz_files)

        # Save HTML report
        with open(report_file, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Generated comprehensive report: {report_file}")
        return str(report_file)

    def _generate_html_report(self, stress_data: Dict[str, Any],
                            concurrent_data: Optional[Dict[str, Any]],
                            benchmark_data: Optional[Dict[str, Any]],
                            viz_files: Dict[str, str]) -> str:
        """Generate HTML report content"""

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stress Testing Comprehensive Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card h3 {{
            margin: 0 0 10px 0;
            color: #333;
            font-size: 1.1em;
        }}
        .summary-card .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .summary-card .unit {{
            font-size: 0.8em;
            color: #666;
        }}
        .section {{
            background: white;
            margin: 20px 0;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin: 0 0 20px 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .visualization {{
            text-align: center;
            margin: 20px 0;
        }}
        .visualization img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .recommendations {{
            background: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
        }}
        .recommendations h3 {{
            margin: 0 0 15px 0;
            color: #155724;
        }}
        .recommendations ul {{
            margin: 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
        }}
        .alert {{
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
        }}
        .alert-success {{
            background-color: #d4edda;
            border: 1px solid #c3e6cb;
            color: #155724;
        }}
        .alert-warning {{
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            color: #856404;
        }}
        .alert-danger {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
        }}
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Stress Testing Comprehensive Report</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>

    <div class="summary-grid">
"""

        # Add summary cards
        if stress_data and 'summary' in stress_data:
            summary = stress_data['summary']
            html += f"""
        <div class="summary-card">
            <h3>Total Configurations</h3>
            <div class="value">{summary.get('total_configurations', 0)}</div>
        </div>
        <div class="summary-card">
            <h3>Success Rate</h3>
            <div class="value">{summary.get('success_rate', 0):.1%}</div>
        </div>
        <div class="summary-card">
            <h3>Average Execution Time</h3>
            <div class="value">{summary.get('average_execution_time', 0):.3f}</div>
            <div class="unit">seconds</div>
        </div>
        <div class="summary-card">
            <h3>Peak Memory Usage</h3>
            <div class="value">{summary.get('memory_peak_mb', 0):.1f}</div>
            <div class="unit">MB</div>
        </div>
"""

        html += """
    </div>
"""

        # Add stress test section
        if stress_data:
            html += self._generate_stress_test_section(stress_data, viz_files)

        # Add concurrent execution section
        if concurrent_data:
            html += self._generate_concurrent_section(concurrent_data, viz_files)

        # Add benchmark section
        if benchmark_data:
            html += self._generate_benchmark_section(benchmark_data)

        # Add footer
        html += f"""
    <div class="footer">
        <p>Report generated by Analog Hawking Radiation Analysis Stress Testing Framework</p>
        <p>For questions or issues, please refer to the project documentation</p>
    </div>
</body>
</html>
"""

        return html

    def _generate_stress_test_section(self, stress_data: Dict[str, Any], viz_files: Dict[str, str]) -> str:
        """Generate stress test section HTML"""
        summary = stress_data.get('summary', {})
        recommendations = summary.get('recommendations', [])
        critical_issues = summary.get('critical_issues', [])

        html = f"""
    <div class="section">
        <h2>Stress Test Results</h2>

        <div class="visualization">
            <h3>Performance Overview</h3>
            <img src="{Path(viz_files.get('performance_overview', '')).relative_to(self.output_dir.parent.parent)}"
                 alt="Performance Overview">
        </div>

        <div class="visualization">
            <h3>Memory Analysis</h3>
            <img src="{Path(viz_files.get('memory_analysis', '')).relative_to(self.output_dir.parent.parent)}"
                 alt="Memory Analysis">
        </div>
"""

        # Add recommendations
        if recommendations:
            html += """
        <div class="recommendations">
            <h3>Performance Recommendations</h3>
            <ul>
"""
            for rec in recommendations:
                html += f"                <li>{rec}</li>\n"
            html += """
            </ul>
        </div>
"""

        # Add critical issues
        if critical_issues:
            html += """
        <div class="alert alert-danger">
            <h3>Critical Issues Detected</h3>
            <ul>
"""
            for issue in critical_issues:
                html += f"                <li>{issue}</li>\n"
            html += """
            </ul>
        </div>
"""

        html += "    </div>\n"
        return html

    def _generate_concurrent_section(self, concurrent_data: Dict[str, Any], viz_files: Dict[str, str]) -> str:
        """Generate concurrent execution section HTML"""
        html = f"""
    <div class="section">
        <h2>Concurrent Execution Analysis</h2>

        <div class="visualization">
            <h3>Scalability Analysis</h3>
            <img src="{Path(viz_files.get('scalability_analysis', '')).relative_to(self.output_dir.parent.parent)}"
                 alt="Scalability Analysis">
        </div>
    </div>
"""
        return html

    def _generate_benchmark_section(self, benchmark_data: Dict[str, Any]) -> str:
        """Generate benchmark section HTML"""
        html = """
    <div class="section">
        <h2>Performance Benchmarks</h2>
        <p>Benchmark data analysis will be displayed here.</p>
    </div>
"""
        return html

    def create_trend_analysis(self, historical_data: List[Dict[str, Any]]) -> str:
        """Create performance trend analysis over time"""
        if not historical_data:
            self.logger.warning("No historical data available for trend analysis")
            return ""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / "plots" / f"trend_analysis_{timestamp}.png"

        # Extract time series data
        timestamps = []
        success_rates = []
        avg_execution_times = []
        peak_memories = []

        for data in historical_data:
            summary = data.get('summary', {})
            timestamps.append(datetime.fromisoformat(summary.get('test_timestamp', data.get('timestamp', ''))))
            success_rates.append(summary.get('success_rate', 0))
            avg_execution_times.append(summary.get('average_execution_time', 0))
            peak_memories.append(summary.get('memory_peak_mb', 0))

        if not timestamps:
            return str(output_file)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Performance Trend Analysis Over Time', fontsize=16, fontweight='bold')

        # Convert timestamps for plotting
        timestamp_nums = mdates.date2num(timestamps)

        # 1. Success rate trend
        axes[0, 0].plot_date(timestamp_nums, success_rates, 'o-', label='Success Rate', linewidth=2)
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_title('Success Rate Trend')
        axes[0, 0].grid(True, alpha=0.3)

        # Add trend line
        if len(success_rates) > 1:
            z = np.polyfit(timestamp_nums, success_rates, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(timestamp_nums, p(timestamp_nums), "r--", alpha=0.8, label='Trend')
            axes[0, 0].legend()

        # 2. Execution time trend
        axes[0, 1].plot_date(timestamp_nums, avg_execution_times, 'o-', color='orange',
                            label='Avg Execution Time', linewidth=2)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Execution Time (seconds)')
        axes[0, 1].set_title('Execution Time Trend')
        axes[0, 1].grid(True, alpha=0.3)

        # Add trend line
        if len(avg_execution_times) > 1:
            z = np.polyfit(timestamp_nums, avg_execution_times, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(timestamp_nums, p(timestamp_nums), "r--", alpha=0.8, label='Trend')
            axes[0, 1].legend()

        # 3. Memory usage trend
        axes[1, 0].plot_date(timestamp_nums, peak_memories, 'o-', color='purple',
                            label='Peak Memory', linewidth=2)
        axes[1, 0].set_xlabel('Date')
        axes[1, 0].set_ylabel('Memory Usage (MB)')
        axes[1, 0].set_title('Memory Usage Trend')
        axes[1, 0].grid(True, alpha=0.3)

        # Add trend line
        if len(peak_memories) > 1:
            z = np.polyfit(timestamp_nums, peak_memories, 1)
            p = np.poly1d(z)
            axes[1, 0].plot(timestamp_nums, p(timestamp_nums), "r--", alpha=0.8, label='Trend')
            axes[1, 0].legend()

        # 4. Performance correlation heatmap
        metrics_data = np.array([success_rates, avg_execution_times, peak_memories])
        correlation_matrix = np.corrcoef(metrics_data)

        im = axes[1, 1].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[1, 1].set_xticks(range(3))
        axes[1, 1].set_yticks(range(3))
        axes[1, 1].set_xticklabels(['Success Rate', 'Exec Time', 'Memory'])
        axes[1, 1].set_yticklabels(['Success Rate', 'Exec Time', 'Memory'])
        axes[1, 1].set_title('Performance Metrics Correlation')

        # Add correlation values
        for i in range(3):
            for j in range(3):
                text = axes[1, 1].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                     ha="center", va="center", color="black", fontweight='bold')

        # Format dates on x-axis
        for ax in [axes[0, 0], axes[0, 1], axes[1, 0]]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(timestamps)//5)))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Created trend analysis: {output_file}")
        return str(output_file)


def main():
    """Main function for stress testing visualization"""
    import argparse

    parser = argparse.ArgumentParser(description="Stress Testing Visualization and Reporting")
    parser.add_argument("--stress-data", type=str, help="Path to stress test results JSON file")
    parser.add_argument("--concurrent-data", type=str, help="Path to concurrent test results JSON file")
    parser.add_argument("--benchmark-data", type=str, help="Path to benchmark results JSON file")
    parser.add_argument("--output-dir", type=str, default="results/stress_testing/visualizations",
                       help="Output directory for visualizations")
    parser.add_argument("--report-only", action="store_true", help="Generate only HTML report")

    args = parser.parse_args()

    # Initialize visualizer
    visualizer = StressTestVisualizer(args.output_dir)

    # Load data
    stress_data = {}
    concurrent_data = {}
    benchmark_data = {}

    if args.stress_data:
        stress_data = visualizer.load_stress_test_results(args.stress_data)
    if args.concurrent_data:
        concurrent_data = visualizer.load_stress_test_results(args.concurrent_data)
    if args.benchmark_data:
        benchmark_data = visualizer.load_stress_test_results(args.benchmark_data)

    if not stress_data and not concurrent_data and not benchmark_data:
        print("No data provided. Use --stress-data, --concurrent-data, or --benchmark-data")
        sys.exit(1)

    # Generate report
    report_file = visualizer.generate_comprehensive_report(stress_data, concurrent_data, benchmark_data)

    print(f"✅ Generated comprehensive stress testing report: {report_file}")
    print(f"📊 Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()