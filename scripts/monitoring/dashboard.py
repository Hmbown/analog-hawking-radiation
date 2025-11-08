#!/usr/bin/env python3
"""
Real-time Experiment Dashboard for Analog Hawking Radiation Orchestration System

Provides live monitoring of experiment progress, key metrics visualization,
resource usage tracking, and automated alerting for exceptional results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

# Optional rich TUI dependencies; keep module import-friendly when not installed
RICH_AVAILABLE = True
try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
except Exception:
    RICH_AVAILABLE = False
    Console = Layout = Live = Panel = Progress = SpinnerColumn = TextColumn = BarColumn = TaskProgressColumn = Table = Text = object  # type: ignore

# Add project paths to Python path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Note: Avoid importing orchestration components here to prevent import cycles.
# Any integration should occur via lightweight method parameters and data files.


class ExperimentDashboard:
    """Lightweight programmatic dashboard API used by orchestration/tests.

    This wrapper intentionally provides a minimal, headless interface for
    starting/stopping a dashboard and updating simple status metrics without
    requiring the rich TUI to run. It coexists with RealTimeDashboard below.

    Methods exposed here are the ones used throughout the codebase and tests.
    """

    def __init__(self, experiment_id: str, update_interval: float = 5.0):
        self.experiment_id = experiment_id
        self.update_interval = update_interval
        self._is_running = False
        self._update_count = 0

        # Simple state tracked for status queries
        self._current_phase: str = "Unknown"
        self._phase_progress: float = 0.0
        self._metrics: Dict[str, Any] = {}

        # Where to persist a lightweight status snapshot for other tools
        self._status_path = Path("results/orchestration") / experiment_id / "dashboard_status.json"
        self._status_path.parent.mkdir(parents=True, exist_ok=True)

        # Logger
        self._logger = logging.getLogger(__name__)

        # Optional: attach a performance monitor to enrich status snapshots
        self._perf_monitor: Optional[PerformanceMonitor] = None

    # --- lifecycle ---
    def start_dashboard(self) -> None:
        self._is_running = True
        self._persist_status()
        self._logger.info(f"ExperimentDashboard started for {self.experiment_id}")

    def stop_dashboard(self) -> None:
        self._is_running = False
        self._persist_status()
        self._logger.info("ExperimentDashboard stopped")

    # --- updates ---
    def update_phase(self, phase_name: str) -> None:
        self._current_phase = phase_name
        self._update_count += 1
        self._persist_status()

    def update_progress(self, progress: float) -> None:
        self._phase_progress = max(0.0, min(1.0, float(progress)))
        self._update_count += 1
        self._persist_status()

    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        # Merge and persist
        self._metrics.update(metrics or {})
        self._update_count += 1
        self._persist_status()

    def integrate_with_monitor(self, monitor) -> None:
        self._perf_monitor = monitor
        self._persist_status()

    # --- queries ---
    def get_dashboard_status(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "running": self._is_running,
            "current_phase": self._current_phase,
            "phase_progress": self._phase_progress,
            "metrics": self._metrics,
            "update_count": self._update_count,
            "components": [
                "status_store",
                "perf_monitor" if self._perf_monitor else "no_perf_monitor",
            ],
        }

    # --- internals ---
    def _persist_status(self) -> None:
        try:
            payload = {
                "experiment_id": self.experiment_id,
                "running": self._is_running,
                "current_phase": self._current_phase,
                "phase_progress": self._phase_progress,
                "metrics": self._sanitize_metrics(self._metrics),
                "update_count": self._update_count,
                "timestamp": time.time(),
            }
            with open(self._status_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            # Non-fatal; keep dashboard tolerant if filesystem is unavailable
            self._logger.debug(f"Failed persisting dashboard status: {e}")

    def _sanitize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize metrics to ensure JSON serializable objects only"""
        def _make_serializable(obj):
            """Recursively convert objects to JSON-serializable types"""
            if obj is None or isinstance(obj, (str, int, float, bool)):
                return obj
            elif isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(item) for item in obj]
            elif hasattr(obj, '__call__'):
                # Handle callable objects (methods, functions)
                return f"<callable: {getattr(obj, '__name__', 'unknown')}>"
            elif hasattr(obj, '__dict__'):
                # Handle custom objects by converting to string representation
                return f"<object: {obj.__class__.__name__}>"
            else:
                # Handle any other types by string conversion
                return str(obj)

        try:
            return _make_serializable(metrics)
        except Exception as e:
            self._logger.warning(f"Error sanitizing metrics: {e}")
            # Return safe fallback metrics
            return {"sanitization_error": str(e), "original_keys": list(metrics.keys()) if isinstance(metrics, dict) else []}


@dataclass
class DashboardMetrics:
    """Comprehensive metrics for dashboard display"""

    timestamp: float
    experiment_id: str
    current_phase: str
    phase_progress: float
    total_simulations: int
    successful_simulations: int
    failed_simulations: int
    success_rate: float
    best_detection_time: Optional[float]
    best_kappa: Optional[float]
    best_snr: Optional[float]
    convergence_score: float
    system_cpu_percent: float
    system_memory_percent: float
    system_disk_percent: float
    active_alerts: List[str]
    estimated_completion: Optional[float]
    performance_issues: List[str]


@dataclass
class Alert:
    """Dashboard alert with severity and context"""

    timestamp: float
    severity: str  # "info", "warning", "error", "critical"
    category: str  # "performance", "convergence", "system", "validation"
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


class RealTimeDashboard:
    """Real-time experiment dashboard with live updates"""

    def __init__(self, experiment_id: str, update_interval: float = 5.0):
        if not RICH_AVAILABLE:
            raise RuntimeError(
                "rich is not installed; RealTimeDashboard is unavailable. Use ExperimentDashboard instead."
            )
        self.experiment_id = experiment_id
        self.update_interval = update_interval
        self.console = Console()
        self.layout = Layout()
        self.is_running = False

        # Metrics and state
        self.metrics_history: List[DashboardMetrics] = []
        self.alerts: List[Alert] = []
        self.experiment_data: Optional[Dict[str, Any]] = None

        # Integration components
        self.performance_monitor: Optional[PerformanceMonitor] = None
        self.result_aggregator: Optional[ResultAggregator] = None

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Initialize dashboard layout
        self._setup_layout()

        self.logger.info(f"Initialized real-time dashboard for experiment {experiment_id}")

    def _setup_logging(self) -> None:
        """Setup dashboard logging"""
        log_dir = Path("results/orchestration") / self.experiment_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_dir / "dashboard.log"), logging.StreamHandler()],
        )

    def _setup_layout(self) -> None:
        """Setup the dashboard layout structure"""
        self.layout.split(
            Layout(name="header", size=3), Layout(name="body"), Layout(name="footer", size=3)
        )

        self.layout["body"].split_row(Layout(name="left"), Layout(name="right"))

        self.layout["left"].split(
            Layout(name="progress", size=8), Layout(name="metrics"), Layout(name="alerts")
        )

        self.layout["right"].split(
            Layout(name="system"), Layout(name="convergence"), Layout(name="performance")
        )

    def _collect_metrics(self) -> DashboardMetrics:
        """Collect current metrics from all sources"""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage(".").percent

        # Experiment metrics (placeholder - would integrate with actual engine)
        current_phase = "Unknown"
        phase_progress = 0.0
        total_simulations = 0
        successful_simulations = 0
        failed_simulations = 0
        success_rate = 0.0
        best_detection_time = None
        best_kappa = None
        best_snr = None
        convergence_score = 0.0
        estimated_completion = None

        # Try to load experiment data
        self._load_experiment_data()

        if self.experiment_data:
            current_phase = self.experiment_data.get("current_phase", "Unknown")
            phase_progress = self.experiment_data.get("phase_progress", 0.0)
            total_simulations = self.experiment_data.get("total_simulations", 0)
            successful_simulations = self.experiment_data.get("successful_simulations", 0)
            failed_simulations = self.experiment_data.get("failed_simulations", 0)

            if total_simulations > 0:
                success_rate = successful_simulations / total_simulations

            best_detection_time = self.experiment_data.get("best_detection_time")
            best_kappa = self.experiment_data.get("best_kappa")
            best_snr = self.experiment_data.get("best_snr")
            convergence_score = self.experiment_data.get("convergence_score", 0.0)
            estimated_completion = self.experiment_data.get("estimated_completion")

        # Check for alerts
        active_alerts = self._check_alerts()
        performance_issues = self._check_performance_issues()

        return DashboardMetrics(
            timestamp=time.time(),
            experiment_id=self.experiment_id,
            current_phase=current_phase,
            phase_progress=phase_progress,
            total_simulations=total_simulations,
            successful_simulations=successful_simulations,
            failed_simulations=failed_simulations,
            success_rate=success_rate,
            best_detection_time=best_detection_time,
            best_kappa=best_kappa,
            best_snr=best_snr,
            convergence_score=convergence_score,
            system_cpu_percent=cpu_percent,
            system_memory_percent=memory_percent,
            system_disk_percent=disk_percent,
            active_alerts=active_alerts,
            estimated_completion=estimated_completion,
            performance_issues=performance_issues,
        )

    def _load_experiment_data(self) -> None:
        """Load experiment data from disk"""
        try:
            experiment_dir = Path("results/orchestration") / self.experiment_id

            # Load manifest
            manifest_file = experiment_dir / "experiment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    manifest = json.load(f)
                    self.experiment_data = manifest

            # Load performance data if available
            if self.performance_monitor:
                perf_summary = self.performance_monitor.get_performance_summary()
                if self.experiment_data:
                    self.experiment_data.update(perf_summary)

        except Exception as e:
            self.logger.warning(f"Failed to load experiment data: {e}")

    def _check_alerts(self) -> List[str]:
        """Check for conditions that warrant alerts"""
        alerts = []

        if not self.experiment_data:
            return ["No experiment data available"]

        # Success rate alerts
        success_rate = self.experiment_data.get("success_rate", 0.0)
        if success_rate < 0.1:
            alerts.append("Critical: Very low success rate")
        elif success_rate < 0.3:
            alerts.append("Warning: Low success rate")

        # System resource alerts
        if psutil.cpu_percent() > 90:
            alerts.append("Critical: High CPU usage")
        elif psutil.cpu_percent() > 80:
            alerts.append("Warning: Elevated CPU usage")

        if psutil.virtual_memory().percent > 90:
            alerts.append("Critical: High memory usage")
        elif psutil.virtual_memory().percent > 80:
            alerts.append("Warning: Elevated memory usage")

        if psutil.disk_usage(".").percent > 95:
            alerts.append("Critical: Disk space running low")

        # Convergence alerts
        convergence_score = self.experiment_data.get("convergence_score", 0.0)
        if convergence_score > 0.8:
            alerts.append("Info: Strong convergence detected")
        elif convergence_score < 0.3:
            alerts.append("Warning: Poor convergence")

        return alerts

    def _check_performance_issues(self) -> List[str]:
        """Check for performance issues"""
        issues = []

        if not self.experiment_data:
            return issues

        # Long simulation times
        avg_sim_time = self.experiment_data.get("average_simulation_time", 0.0)
        if avg_sim_time > 300:
            issues.append(f"Long simulation times: {avg_sim_time:.1f}s")

        # High failure rate
        success_rate = self.experiment_data.get("success_rate", 0.0)
        if success_rate < 0.5:
            issues.append(f"Low success rate: {success_rate:.1%}")

        return issues

    def _create_header(self) -> Panel:
        """Create dashboard header"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header_text = Text()
        header_text.append("Analog Hawking Radiation Experiment Dashboard", style="bold blue")
        header_text.append(f" | Experiment: {self.experiment_id}", style="green")
        header_text.append(f" | {current_time}", style="yellow")

        return Panel(header_text, style="bold")

    def _create_progress_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create progress tracking panel"""
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Phase", style="cyan")
        table.add_column("Progress", width=20)
        table.add_column("Simulations", justify="right")
        table.add_column("Success Rate", justify="right")

        # Progress bar
        progress_bar = "█" * int(metrics.phase_progress * 20) + "░" * (
            20 - int(metrics.phase_progress * 20)
        )

        table.add_row(
            metrics.current_phase,
            f"{progress_bar} {metrics.phase_progress:.1%}",
            f"{metrics.total_simulations}",
            f"{metrics.success_rate:.1%}",
        )

        # Best results
        best_results = []
        if metrics.best_detection_time:
            best_results.append(f"Best Detection: {metrics.best_detection_time:.2e}s")
        if metrics.best_kappa:
            best_results.append(f"Best κ: {metrics.best_kappa:.2e}s⁻¹")
        if metrics.best_snr:
            best_results.append(f"Best SNR: {metrics.best_snr:.2f}")

        best_text = " | ".join(best_results) if best_results else "No results yet"

        progress_panel = Panel(
            table, title="Experiment Progress", title_align="left", subtitle=best_text
        )

        return progress_panel

    def _create_metrics_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create key metrics panel"""
        table = Table(show_header=False, box=None)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Convergence Score", f"{metrics.convergence_score:.3f}")
        table.add_row(
            "Total Compute Time",
            f"{metrics.timestamp - (self.experiment_data.get('start_time', metrics.timestamp) if self.experiment_data else metrics.timestamp):.0f}s",
        )

        if metrics.estimated_completion:
            eta = datetime.fromtimestamp(metrics.estimated_completion)
            table.add_row("Estimated Completion", eta.strftime("%Y-%m-%d %H:%M"))
        else:
            table.add_row("Estimated Completion", "Calculating...")

        return Panel(table, title="Key Metrics", title_align="left")

    def _create_alerts_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create alerts panel"""
        if not metrics.active_alerts:
            return Panel("No active alerts", title="Alerts", title_align="left", style="green")

        alert_text = "\n".join([f"• {alert}" for alert in metrics.active_alerts])
        return Panel(alert_text, title="Active Alerts", title_align="left", style="red")

    def _create_system_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create system resources panel"""
        table = Table(show_header=False, box=None)
        table.add_column("Resource", style="cyan")
        table.add_column("Usage", style="white")
        table.add_column("Bar", width=15)

        # CPU
        cpu_bar = "█" * int(metrics.system_cpu_percent / 5) + "░" * (
            20 - int(metrics.system_cpu_percent / 5)
        )
        table.add_row("CPU", f"{metrics.system_cpu_percent:.1f}%", cpu_bar)

        # Memory
        memory_bar = "█" * int(metrics.system_memory_percent / 5) + "░" * (
            20 - int(metrics.system_memory_percent / 5)
        )
        table.add_row("Memory", f"{metrics.system_memory_percent:.1f}%", memory_bar)

        # Disk
        disk_bar = "█" * int(metrics.system_disk_percent / 5) + "░" * (
            20 - int(metrics.system_disk_percent / 5)
        )
        table.add_row("Disk", f"{metrics.system_disk_percent:.1f}%", disk_bar)

        return Panel(table, title="System Resources", title_align="left")

    def _create_convergence_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create convergence analysis panel"""
        convergence_level = "None"
        convergence_style = "red"

        if metrics.convergence_score >= 0.8:
            convergence_level = "Strong"
            convergence_style = "green"
        elif metrics.convergence_score >= 0.6:
            convergence_level = "Good"
            convergence_style = "yellow"
        elif metrics.convergence_score >= 0.4:
            convergence_level = "Partial"
            convergence_style = "orange"

        convergence_bar = "█" * int(metrics.convergence_score * 10) + "░" * (
            10 - int(metrics.convergence_score * 10)
        )

        content = f"""
Level: [{convergence_style}]{convergence_level}[/{convergence_style}]
Score: {metrics.convergence_score:.3f}
Progress: {convergence_bar}
"""

        return Panel(content, title="Convergence Analysis", title_align="left")

    def _create_performance_panel(self, metrics: DashboardMetrics) -> Panel:
        """Create performance issues panel"""
        if not metrics.performance_issues:
            return Panel(
                "No performance issues detected",
                title="Performance",
                title_align="left",
                style="green",
            )

        issues_text = "\n".join([f"• {issue}" for issue in metrics.performance_issues])
        return Panel(issues_text, title="Performance Issues", title_align="left", style="yellow")

    def _create_footer(self) -> Panel:
        """Create dashboard footer"""
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("[Q] Quit", style="cyan")
        footer_text.append(" | ")
        footer_text.append("[R] Refresh", style="cyan")
        footer_text.append(" | ")
        footer_text.append("[L] Logs", style="cyan")
        footer_text.append(" | ")
        footer_text.append("[E] Export", style="cyan")

        return Panel(footer_text, style="dim")

    def update_display(self) -> Layout:
        """Update the entire dashboard display"""
        metrics = self._collect_metrics()
        self.metrics_history.append(metrics)

        # Limit history size
        if len(self.metrics_history) > 1000:
            self.metrics_history.pop(0)

        # Update layout components
        self.layout["header"].update(self._create_header())
        self.layout["progress"].update(self._create_progress_panel(metrics))
        self.layout["metrics"].update(self._create_metrics_panel(metrics))
        self.layout["alerts"].update(self._create_alerts_panel(metrics))
        self.layout["system"].update(self._create_system_panel(metrics))
        self.layout["convergence"].update(self._create_convergence_panel(metrics))
        self.layout["performance"].update(self._create_performance_panel(metrics))
        self.layout["footer"].update(self._create_footer())

        return self.layout

    async def run(self) -> None:
        """Run the dashboard with live updates"""
        self.is_running = True

        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            while self.is_running:
                try:
                    # Update display
                    live.update(self.update_display())
                    # Non-blocking loop: avoid waiting for interactive input
                    # Wait for next update
                    await asyncio.sleep(self.update_interval)

                except KeyboardInterrupt:
                    self.logger.info("Dashboard interrupted by user")
                    break
                except Exception as e:
                    self.logger.error(f"Dashboard error: {e}")
                    await asyncio.sleep(self.update_interval)

        self.logger.info("Dashboard stopped")

    def export_metrics(self, output_path: Optional[Path] = None) -> None:
        """Export dashboard metrics to file"""
        if not output_path:
            output_path = (
                Path("results/orchestration") / self.experiment_id / "dashboard_metrics.json"
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        export_data = {
            "experiment_id": self.experiment_id,
            "export_time": time.time(),
            "metrics_history": [asdict(m) for m in self.metrics_history],
            "alerts": [asdict(a) for a in self.alerts],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)

        self.logger.info(f"Exported dashboard metrics to {output_path}")

    def integrate_with_monitor(self, monitor: PerformanceMonitor) -> None:
        """Integrate with performance monitor"""
        self.performance_monitor = monitor
        self.logger.info("Integrated with performance monitor")

    def integrate_with_aggregator(self, aggregator: ResultAggregator) -> None:
        """Integrate with result aggregator"""
        self.result_aggregator = aggregator
        self.logger.info("Integrated with result aggregator")


def main():
    """Main entry point for dashboard"""
    import argparse

    parser = argparse.ArgumentParser(description="Real-time Experiment Dashboard")
    parser.add_argument("experiment_id", help="Experiment ID to monitor")
    parser.add_argument("--interval", type=float, default=5.0, help="Update interval in seconds")
    parser.add_argument("--export", action="store_true", help="Export metrics and exit")

    args = parser.parse_args()

    # Create and run dashboard
    dashboard = RealTimeDashboard(args.experiment_id, args.interval)

    if args.export:
        dashboard.export_metrics()
        print(f"Exported metrics for experiment {args.experiment_id}")
        return

    # Run dashboard
    try:
        asyncio.run(dashboard.run())
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    except Exception as e:
        print(f"Dashboard error: {e}")


if __name__ == "__main__":
    main()
