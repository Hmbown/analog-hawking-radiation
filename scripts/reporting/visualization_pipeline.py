#!/usr/bin/env python3
"""
Visualization Pipeline for Analog Hawking Radiation Experiments

Generates publication-quality figures, multi-experiment comparisons, performance
visualizations, and statistical significance charts for scientific reports.
"""

from __future__ import annotations

import json
import logging

# Add project paths to Python path
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec
from scipy import stats

# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.result_aggregator import ExperimentAggregate, ResultAggregator
from scripts.validation.validation_framework import ValidationFramework


@dataclass
class FigureSpecification:
    """Specification for a publication-quality figure"""

    figure_id: str
    title: str
    description: str
    figure_type: str  # "multi_panel", "heatmap", "distribution", "correlation", "progression"
    data_sources: List[str]
    style_parameters: Dict[str, Any]
    output_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "svg"])
    dpi: int = 300
    width: float = 8.0  # inches
    height: float = 6.0  # inches


@dataclass
class VisualizationBundle:
    """Complete set of visualizations for an experiment"""

    experiment_id: str
    figures: List[FigureSpecification]
    generated_files: List[Path]
    metadata: Dict[str, Any]


class VisualizationPipeline:
    """Automated visualization generation with publication-quality output"""

    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.viz_dir = self.experiment_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)

        # Integration components
        self.aggregator = ResultAggregator(experiment_id)
        self.validator = ValidationFramework(experiment_id)

        # Visualization settings
        self.style_settings = {
            "color_palette": "husl",
            "font_family": "serif",
            "font_size": 12,
            "line_width": 2.0,
            "marker_size": 6,
            "grid_alpha": 0.3,
            "figure_style": "seaborn-v0_8-whitegrid",
        }

        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)

        # Load experiment data
        self.aggregate: Optional[ExperimentAggregate] = None
        self.experiment_manifest: Optional[Dict[str, Any]] = None

        self.logger.info(f"Initialized visualization pipeline for experiment {experiment_id}")

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_dir / "visualization_pipeline.log"),
                logging.StreamHandler(),
            ],
        )

    def load_experiment_data(self) -> bool:
        """Load all experiment data for visualization"""
        try:
            # Load result aggregation
            if not self.aggregator.load_experiment_data():
                self.logger.error("Failed to load experiment data")
                return False

            self.aggregate = self.aggregator.aggregate_results()

            # Load manifest
            manifest_file = self.experiment_dir / "experiment_manifest.json"
            if manifest_file.exists():
                with open(manifest_file, "r") as f:
                    self.experiment_manifest = json.load(f)

            self.logger.info("Successfully loaded experiment data for visualization")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load experiment data: {e}")
            return False

    def generate_publication_figures(self) -> VisualizationBundle:
        """Generate complete set of publication-quality figures"""
        if not self.aggregate:
            self.load_experiment_data()

        # Setup matplotlib style
        self._setup_matplotlib_style()

        figures = []
        generated_files = []

        # Generate core figures
        fig_specs = [
            self._create_phase_progression_spec(),
            self._create_parameter_sensitivity_spec(),
            self._create_cross_phase_correlation_spec(),
            self._create_detection_time_distribution_spec(),
            self._create_statistical_significance_spec(),
            self._create_success_metrics_spec(),
            self._create_optimization_trajectory_spec(),
            self._create_validation_summary_spec(),
        ]

        # Generate each figure
        for spec in fig_specs:
            try:
                file_paths = self._generate_figure(spec)
                generated_files.extend(file_paths)
                figures.append(spec)
                self.logger.info(f"Generated figure: {spec.figure_id}")
            except Exception as e:
                self.logger.error(f"Failed to generate figure {spec.figure_id}: {e}")

        # Generate multi-experiment comparison if available
        comparison_figures = self._generate_multi_experiment_comparisons()
        figures.extend(comparison_figures)

        bundle = VisualizationBundle(
            experiment_id=self.experiment_id,
            figures=figures,
            generated_files=generated_files,
            metadata=self._generate_visualization_metadata(),
        )

        # Save visualization bundle
        self._save_visualization_bundle(bundle)

        self.logger.info(f"Generated {len(generated_files)} visualization files")
        return bundle

    def _setup_matplotlib_style(self) -> None:
        """Setup matplotlib for publication-quality output"""
        plt.style.use(self.style_settings["figure_style"])

        # Set publication-quality parameters
        plt.rcParams.update(
            {
                "font.family": self.style_settings["font_family"],
                "font.size": self.style_settings["font_size"],
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "figure.figsize": (8, 6),
                "figure.autolayout": True,
                "lines.linewidth": self.style_settings["line_width"],
                "lines.markersize": self.style_settings["marker_size"],
                "axes.grid": True,
                "grid.alpha": self.style_settings["grid_alpha"],
                "legend.frameon": True,
                "legend.fancybox": True,
                "legend.shadow": True,
            }
        )

    def _create_phase_progression_spec(self) -> FigureSpecification:
        """Create specification for phase progression figure"""
        return FigureSpecification(
            figure_id="phase_progression",
            title="Phase Progression of Detection Metrics",
            description="Evolution of key metrics across experimental phases showing optimization effectiveness",
            figure_type="multi_panel",
            data_sources=["phase_summary", "cross_phase_correlation"],
            style_parameters={
                "layout": "2x2",
                "color_scheme": "sequential",
                "annotations": True,
                "trend_lines": True,
            },
            width=12.0,
            height=10.0,
        )

    def _create_parameter_sensitivity_spec(self) -> FigureSpecification:
        """Create specification for parameter sensitivity figure"""
        return FigureSpecification(
            figure_id="parameter_sensitivity",
            title="Parameter Sensitivity Analysis",
            description="Correlation between key parameters and detection efficiency",
            figure_type="bar_chart",
            data_sources=["parameter_sensitivity"],
            style_parameters={
                "orientation": "horizontal",
                "color_gradient": True,
                "value_labels": True,
                "threshold_line": 0.3,
            },
            width=10.0,
            height=6.0,
        )

    def _create_cross_phase_correlation_spec(self) -> FigureSpecification:
        """Create specification for cross-phase correlation figure"""
        return FigureSpecification(
            figure_id="cross_phase_correlation",
            title="Cross-Phase Correlation Matrix",
            description="Statistical consistency between results from different optimization phases",
            figure_type="heatmap",
            data_sources=["cross_phase_correlation"],
            style_parameters={
                "color_map": "coolwarm",
                "annotations": True,
                "dendrogram": False,
                "symmetrical": True,
            },
            width=8.0,
            height=6.0,
        )

    def _create_detection_time_distribution_spec(self) -> FigureSpecification:
        """Create specification for detection time distribution figure"""
        return FigureSpecification(
            figure_id="detection_time_distribution",
            title="Detection Time Distribution Analysis",
            description="Statistical distribution of detection times across all successful simulations",
            figure_type="distribution",
            data_sources=["all_detection_times"],
            style_parameters={
                "log_scale": True,
                "multiple_views": True,
                "outlier_detection": True,
                "distribution_fit": True,
            },
            width=12.0,
            height=8.0,
        )

    def _create_statistical_significance_spec(self) -> FigureSpecification:
        """Create specification for statistical significance figure"""
        return FigureSpecification(
            figure_id="statistical_significance",
            title="Statistical Significance Analysis",
            description="Detection probability as function of observation time for different significance levels",
            figure_type="line_plot",
            data_sources=["statistical_significance"],
            style_parameters={
                "log_scale_x": True,
                "multiple_lines": True,
                "confidence_intervals": True,
                "reference_lines": True,
            },
            width=10.0,
            height=8.0,
        )

    def _create_success_metrics_spec(self) -> FigureSpecification:
        """Create specification for success metrics figure"""
        return FigureSpecification(
            figure_id="success_metrics",
            title="Comprehensive Success Metrics",
            description="Multi-dimensional assessment of experiment success across technical and validation dimensions",
            figure_type="radar_chart",
            data_sources=["success_metrics", "validation_summary"],
            style_parameters={
                "normalized_scales": True,
                "fill_areas": True,
                "grid_lines": True,
                "category_labels": True,
            },
            width=8.0,
            height=8.0,
        )

    def _create_optimization_trajectory_spec(self) -> FigureSpecification:
        """Create specification for optimization trajectory figure"""
        return FigureSpecification(
            figure_id="optimization_trajectory",
            title="Bayesian Optimization Trajectory",
            description="Evolution of objective function and parameter exploration during optimization phases",
            figure_type="scatter_plot",
            data_sources=["optimization_history"],
            style_parameters={
                "color_by_iteration": True,
                "trend_line": True,
                "confidence_region": True,
                "best_point_highlight": True,
            },
            width=10.0,
            height=8.0,
        )

    def _create_validation_summary_spec(self) -> FigureSpecification:
        """Create specification for validation summary figure"""
        return FigureSpecification(
            figure_id="validation_summary",
            title="Comprehensive Validation Summary",
            description="Results from all validation checks with confidence scores and severity levels",
            figure_type="multi_panel",
            data_sources=["validation_results"],
            style_parameters={
                "layout": "1x2",
                "color_by_severity": True,
                "threshold_indicators": True,
                "detailed_annotations": True,
            },
            width=12.0,
            height=6.0,
        )

    def _generate_figure(self, spec: FigureSpecification) -> List[Path]:
        """Generate a single figure based on specification"""
        figure_generators = {
            "multi_panel": self._generate_multi_panel_figure,
            "bar_chart": self._generate_bar_chart,
            "heatmap": self._generate_heatmap,
            "distribution": self._generate_distribution_plot,
            "line_plot": self._generate_line_plot,
            "radar_chart": self._generate_radar_chart,
            "scatter_plot": self._generate_scatter_plot,
        }

        generator = figure_generators.get(spec.figure_type)
        if not generator:
            self.logger.warning(f"No generator for figure type: {spec.figure_type}")
            return []

        return generator(spec)

    def _generate_multi_panel_figure(self, spec: FigureSpecification) -> List[Path]:
        """Generate multi-panel figure for phase progression"""
        if spec.figure_id == "phase_progression":
            return self._generate_phase_progression_figure(spec)
        elif spec.figure_id == "validation_summary":
            return self._generate_validation_summary_figure(spec)
        else:
            return self._generate_generic_multi_panel(spec)

    def _generate_phase_progression_figure(self, spec: FigureSpecification) -> List[Path]:
        """Generate phase progression multi-panel figure"""
        if not self.aggregate:
            return []

        fig = plt.figure(figsize=(spec.width, spec.height))
        gs = gridspec.GridSpec(2, 2, figure=fig)

        phases = list(self.aggregate.phase_summary.keys())
        phase_labels = [p.replace("phase_", "").replace("_", " ").title() for p in phases]

        # Panel 1: Best detection time progression
        ax1 = fig.add_subplot(gs[0, 0])
        best_times = [
            self.aggregate.phase_summary[p].get("best_detection_time") or np.nan for p in phases
        ]
        valid_indices = [i for i, t in enumerate(best_times) if not np.isnan(t)]

        if valid_indices:
            valid_times = [best_times[i] for i in valid_indices]
            valid_phases = [phase_labels[i] for i in valid_indices]

            ax1.plot(
                valid_indices,
                valid_times,
                "o-",
                linewidth=2,
                markersize=8,
                color="blue",
                label="Best Detection Time",
            )
            ax1.set_yscale("log")
            ax1.set_ylabel("Best Detection Time (s)")
            ax1.set_title("Detection Time Optimization")
            ax1.set_xticks(valid_indices)
            ax1.set_xticklabels(valid_phases, rotation=45)
            ax1.grid(True, alpha=0.3)
            ax1.legend()

        # Panel 2: Best kappa progression
        ax2 = fig.add_subplot(gs[0, 1])
        best_kappas = [self.aggregate.phase_summary[p].get("best_kappa") or np.nan for p in phases]
        valid_kappa_indices = [i for i, k in enumerate(best_kappas) if not np.isnan(k)]

        if valid_kappa_indices:
            valid_kappas = [best_kappas[i] for i in valid_kappa_indices]
            valid_kappa_phases = [phase_labels[i] for i in valid_kappa_indices]

            ax2.plot(
                valid_kappa_indices,
                valid_kappas,
                "s-",
                linewidth=2,
                markersize=8,
                color="orange",
                label="Best Surface Gravity",
            )
            ax2.set_yscale("log")
            ax2.set_ylabel("Surface Gravity κ (s⁻¹)")
            ax2.set_title("Surface Gravity Progression")
            ax2.set_xticks(valid_kappa_indices)
            ax2.set_xticklabels(valid_kappa_phases, rotation=45)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        # Panel 3: Success rate progression
        ax3 = fig.add_subplot(gs[1, 0])
        success_rates = [self.aggregate.phase_summary[p].get("success_rate") or 0.0 for p in phases]

        bars = ax3.bar(range(len(phases)), success_rates, color="green", alpha=0.7)
        ax3.set_xlabel("Phase")
        ax3.set_ylabel("Success Rate")
        ax3.set_title("Success Rate by Phase")
        ax3.set_xticks(range(len(phases)))
        ax3.set_xticklabels(phase_labels, rotation=45)
        ax3.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars
        for bar, rate in zip(bars, success_rates):
            ax3.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
            )

        # Panel 4: Parameter sensitivity
        ax4 = fig.add_subplot(gs[1, 1])
        if self.aggregate.parameter_sensitivity:
            param_names = list(self.aggregate.parameter_sensitivity.keys())
            sensitivities = list(self.aggregate.parameter_sensitivity.values())

            bars = ax4.barh(param_names, sensitivities, color="red", alpha=0.7)
            ax4.set_xlabel("Sensitivity (|correlation|)")
            ax4.set_title("Parameter Sensitivity Analysis")
            ax4.grid(True, alpha=0.3, axis="x")

            # Add threshold line
            ax4.axvline(
                x=0.3, color="gray", linestyle="--", alpha=0.7, label="Significance Threshold"
            )
            ax4.legend()

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_bar_chart(self, spec: FigureSpecification) -> List[Path]:
        """Generate bar chart for parameter sensitivity"""
        if not self.aggregate or not self.aggregate.parameter_sensitivity:
            return []

        fig, ax = plt.subplots(figsize=(spec.width, spec.height))

        param_names = list(self.aggregate.parameter_sensitivity.keys())
        sensitivities = list(self.aggregate.parameter_sensitivity.values())

        # Create horizontal bar chart
        bars = ax.barh(
            param_names,
            sensitivities,
            color=plt.cm.viridis(np.linspace(0, 1, len(param_names))),
            alpha=0.7,
        )

        # Add value labels
        for bar, value in zip(bars, sensitivities):
            ax.text(
                bar.get_width() + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.3f}",
                va="center",
                fontsize=10,
            )

        # Add threshold line
        ax.axvline(
            x=0.3, color="red", linestyle="--", alpha=0.7, label="Significance Threshold (0.3)"
        )

        ax.set_xlabel("Sensitivity (Absolute Correlation Coefficient)")
        ax.set_title(spec.title)
        ax.grid(True, alpha=0.3, axis="x")
        ax.legend()

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_heatmap(self, spec: FigureSpecification) -> List[Path]:
        """Generate heatmap for cross-phase correlation"""
        if (
            not self.aggregate
            or not self.aggregate.cross_phase_correlation.correlation_coefficients
        ):
            return []

        fig, ax = plt.subplots(figsize=(spec.width, spec.height))

        phase_names = list(self.aggregate.phase_summary.keys())
        n_phases = len(phase_names)

        # Create correlation matrix
        corr_matrix = np.ones((n_phases, n_phases))
        for (
            phase1,
            phase2,
        ), corr in self.aggregate.cross_phase_correlation.correlation_coefficients.items():
            i = phase_names.index(phase1)
            j = phase_names.index(phase2)
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr

        # Create heatmap
        im = ax.imshow(corr_matrix, cmap="coolwarm", vmin=0, vmax=1)

        # Set labels
        phase_labels = [p.replace("phase_", "").replace("_", "\n").title() for p in phase_names]
        ax.set_xticks(range(n_phases))
        ax.set_yticks(range(n_phases))
        ax.set_xticklabels(phase_labels)
        ax.set_yticklabels(phase_labels)

        # Add annotations
        for i in range(n_phases):
            for j in range(n_phases):
                ax.text(
                    j,
                    i,
                    f"{corr_matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if corr_matrix[i, j] < 0.5 else "black",
                )

        ax.set_title(spec.title)
        plt.colorbar(im, ax=ax, label="Correlation Coefficient")

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_distribution_plot(self, spec: FigureSpecification) -> List[Path]:
        """Generate distribution plot for detection times"""
        detection_times = self._extract_all_detection_times()
        if not detection_times:
            return []

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(spec.width, spec.height))

        # Panel 1: Histogram with distribution fit
        n, bins, patches = ax1.hist(
            np.log10(detection_times), bins=30, alpha=0.7, edgecolor="black", density=True
        )
        ax1.set_xlabel("log10(Detection Time [s])")
        ax1.set_ylabel("Probability Density")
        ax1.set_title("Detection Time Distribution")
        ax1.grid(True, alpha=0.3)

        # Fit normal distribution to log data
        try:
            mu, sigma = stats.norm.fit(np.log10(detection_times))
            x = np.linspace(np.log10(detection_times).min(), np.log10(detection_times).max(), 100)
            pdf = stats.norm.pdf(x, mu, sigma)
            ax1.plot(x, pdf, "r-", linewidth=2, label=f"Normal fit (μ={mu:.2f}, σ={sigma:.2f})")
            ax1.legend()
        except:
            pass

        # Panel 2: Box plot by phase
        phase_detection_times = {}
        for phase_name, results in self.aggregator.results.items():
            times = []
            for result in results:
                if result.get("simulation_success") and result.get("t5sigma_s"):
                    times.append(result["t5sigma_s"])
            if times:
                phase_detection_times[phase_name] = times

        if phase_detection_times:
            phase_labels = [
                p.replace("phase_", "").replace("_", " ").title()
                for p in phase_detection_times.keys()
            ]
            data = [np.log10(times) for times in phase_detection_times.values()]

            box_plot = ax2.boxplot(data, labels=phase_labels, patch_artist=True)
            ax2.set_ylabel("log10(Detection Time [s])")
            ax2.set_title("Detection Time by Phase")
            ax2.grid(True, alpha=0.3)

            # Color the boxes
            colors = plt.cm.Set3(np.linspace(0, 1, len(phase_detection_times)))
            for patch, color in zip(box_plot["boxes"], colors):
                patch.set_facecolor(color)

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_line_plot(self, spec: FigureSpecification) -> List[Path]:
        """Generate line plot for statistical significance"""
        if (
            not self.aggregate
            or "detection_probability_1h" not in self.aggregate.statistical_significance
        ):
            return []

        fig, ax = plt.subplots(figsize=(spec.width, spec.height))

        # Define observation times (seconds)
        observation_times = [3600, 86400, 604800]  # 1h, 1d, 1w
        time_labels = ["1 hour", "1 day", "1 week"]

        # Extract detection probabilities for different sigma levels
        sigma_levels = [3, 5, 6]
        probabilities = {}

        for sigma in sigma_levels:
            probs = []
            for time_key in [
                "detection_probability_1h",
                "detection_probability_1d",
                "detection_probability_1w",
            ]:
                if time_key in self.aggregate.statistical_significance:
                    prob_key = f"detection_probability_{sigma}sigma"
                    prob = self.aggregate.statistical_significance[time_key].get(prob_key, 0)
                    probs.append(prob)
            if probs:
                probabilities[sigma] = probs

        # Plot lines for each sigma level
        colors = ["blue", "green", "red"]
        for i, (sigma, probs) in enumerate(probabilities.items()):
            if len(probs) == len(observation_times):
                ax.plot(
                    observation_times,
                    probs,
                    "o-",
                    linewidth=2,
                    markersize=8,
                    color=colors[i],
                    label=f"{sigma}σ Detection",
                )

        ax.set_xscale("log")
        ax.set_xlabel("Observation Time (s)")
        ax.set_ylabel("Detection Probability")
        ax.set_title(spec.title)
        ax.set_xticks(observation_times)
        ax.set_xticklabels(time_labels)
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1)

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_radar_chart(self, spec: FigureSpecification) -> List[Path]:
        """Generate radar chart for success metrics"""
        # Calculate success metrics
        metrics = self._calculate_success_metrics()
        if not metrics:
            return []

        fig, ax = plt.subplots(
            figsize=(spec.width, spec.height), subplot_kw=dict(projection="polar")
        )

        # Define categories and values
        categories = list(metrics.keys())
        values = list(metrics.values())

        # Complete the circle
        values += values[:1]
        categories_display = [c.replace("_", " ").title() for c in categories]
        categories_display += categories_display[:1]

        # Calculate angles
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Plot radar chart
        ax.plot(angles, values, "o-", linewidth=2, label="Success Metrics")
        ax.fill(angles, values, alpha=0.25)

        # Add category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories_display[:-1])

        # Set y-axis limits and grid
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.grid(True)

        ax.set_title(spec.title)

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_scatter_plot(self, spec: FigureSpecification) -> List[Path]:
        """Generate scatter plot for optimization trajectory"""
        # This would use actual optimization history data
        # For now, create a placeholder visualization
        if not self.aggregate:
            return []

        fig, ax = plt.subplots(figsize=(spec.width, spec.height))

        # Extract detection times and kappa values for successful simulations
        detection_times = []
        kappa_values = []

        for phase_results in self.aggregator.results.values():
            for result in phase_results:
                if result.get("simulation_success") and result.get("t5sigma_s"):
                    detection_times.append(result["t5sigma_s"])
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        kappa_values.append(max(kappa_list))

        if detection_times and kappa_values:
            # Ensure equal length
            min_len = min(len(detection_times), len(kappa_values))
            detection_times = detection_times[:min_len]
            kappa_values = kappa_values[:min_len]

            scatter = ax.scatter(
                np.log10(detection_times),
                np.log10(kappa_values),
                c=np.arange(len(detection_times)),
                cmap="viridis",
                alpha=0.6,
                s=50,
            )

            ax.set_xlabel("log10(Detection Time [s])")
            ax.set_ylabel("log10(Surface Gravity κ [s⁻¹])")
            ax.set_title("Detection Time vs Surface Gravity")
            ax.grid(True, alpha=0.3)

            plt.colorbar(scatter, ax=ax, label="Simulation Index")

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_validation_summary_figure(self, spec: FigureSpecification) -> List[Path]:
        """Generate validation summary multi-panel figure"""
        validation_summary = self.validator.run_comprehensive_validation()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(spec.width, spec.height))

        # Panel 1: Validation results by category
        categories = {}
        for result in validation_summary.results:
            category = result.check_name.split("_")[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(result)

        category_names = list(categories.keys())
        pass_rates = []
        confidences = []

        for category in category_names:
            results = categories[category]
            pass_rate = sum(1 for r in results if r.passed) / len(results)
            avg_confidence = np.mean([r.confidence for r in results])
            pass_rates.append(pass_rate)
            confidences.append(avg_confidence)

        x = np.arange(len(category_names))
        width = 0.35

        bars1 = ax1.bar(x - width / 2, pass_rates, width, label="Pass Rate", alpha=0.7)
        bars2 = ax1.bar(x + width / 2, confidences, width, label="Avg Confidence", alpha=0.7)

        ax1.set_xlabel("Validation Category")
        ax1.set_ylabel("Score")
        ax1.set_title("Validation Results by Category")
        ax1.set_xticks(x)
        ax1.set_xticklabels([c.title() for c in category_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        # Panel 2: Severity distribution
        severities = {}
        for result in validation_summary.results:
            severity = result.severity
            if severity not in severities:
                severities[severity] = 0
            severities[severity] += 1

        if severities:
            severity_names = list(severities.keys())
            severity_counts = list(severities.values())

            colors = {"critical": "red", "error": "orange", "warning": "yellow", "info": "green"}
            bar_colors = [colors.get(sev, "gray") for sev in severity_names]

            bars = ax2.bar(severity_names, severity_counts, color=bar_colors, alpha=0.7)
            ax2.set_xlabel("Severity Level")
            ax2.set_ylabel("Number of Checks")
            ax2.set_title("Validation Check Severity Distribution")
            ax2.grid(True, alpha=0.3, axis="y")

            # Add value labels
            for bar, count in zip(bars, severity_counts):
                ax2.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    str(count),
                    ha="center",
                    va="bottom",
                )

        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_generic_multi_panel(self, spec: FigureSpecification) -> List[Path]:
        """Generate generic multi-panel figure"""
        # Placeholder implementation
        fig, axes = plt.subplots(2, 2, figsize=(spec.width, spec.height))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.plot([0, 1], [0, 1], label=f"Panel {i+1}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.suptitle(spec.title)
        plt.tight_layout()

        # Save in multiple formats
        file_paths = []
        for fmt in spec.output_formats:
            file_path = self.viz_dir / f"{spec.figure_id}.{fmt}"
            plt.savefig(file_path, dpi=spec.dpi, bbox_inches="tight")
            file_paths.append(file_path)

        plt.close()
        return file_paths

    def _generate_multi_experiment_comparisons(self) -> List[FigureSpecification]:
        """Generate multi-experiment comparison figures"""
        # This would compare across multiple experiment directories
        # For now, return empty list as we only have one experiment
        return []

    def _extract_all_detection_times(self) -> List[float]:
        """Extract all detection times from results"""
        detection_times = []
        if not hasattr(self.aggregator, "results") or not self.aggregator.results:
            return detection_times

        for phase_results in self.aggregator.results.values():
            for result in phase_results:
                if result.get("simulation_success") and result.get("t5sigma_s"):
                    detection_times.append(result["t5sigma_s"])

        return detection_times

    def _calculate_success_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive success metrics for radar chart"""
        if not self.aggregate:
            return {}

        metrics = {
            "technical_success": self.aggregate.success_rate,
            "detection_efficiency": self._calculate_detection_efficiency(),
            "optimization_effectiveness": self._calculate_optimization_effectiveness(),
            "validation_confidence": 0.8,  # Placeholder - would use actual validation results
            "physical_plausibility": self._assess_physical_plausibility(),
            "statistical_robustness": self._assess_statistical_robustness(),
        }

        return metrics

    def _calculate_detection_efficiency(self) -> float:
        """Calculate detection efficiency metric"""
        detection_times = self._extract_all_detection_times()
        if not detection_times:
            return 0.0

        # Efficiency metric: inverse of median detection time (higher is better)
        median_time = np.median(detection_times)
        efficiency = 1.0 / median_time if median_time > 0 else 0.0

        # Normalize to [0, 1] range (assuming efficiency < 0.01 is maximum)
        return min(efficiency / 0.01, 1.0)

    def _calculate_optimization_effectiveness(self) -> float:
        """Calculate optimization effectiveness metric"""
        if not self.aggregate or len(self.aggregate.phase_summary) < 2:
            return 0.0

        phases = list(self.aggregate.phase_summary.keys())
        improvements = []

        for i in range(1, len(phases)):
            prev_phase = self.aggregate.phase_summary[phases[i - 1]]
            curr_phase = self.aggregate.phase_summary[phases[i]]

            if prev_phase.get("best_detection_time") and curr_phase.get("best_detection_time"):
                improvement = (
                    prev_phase["best_detection_time"] - curr_phase["best_detection_time"]
                ) / prev_phase["best_detection_time"]
                improvements.append(max(improvement, 0.0))

        return np.mean(improvements) if improvements else 0.0

    def _assess_physical_plausibility(self) -> float:
        """Assess physical plausibility of results"""
        kappa_values = self._extract_all_kappa_values()
        if not kappa_values:
            return 0.0

        # Count kappa values in physically plausible range (1e9 - 1e12 s⁻¹)
        plausible = [k for k in kappa_values if 1e9 <= k <= 1e12]
        return len(plausible) / len(kappa_values)

    def _assess_statistical_robustness(self) -> float:
        """Assess statistical robustness of results"""
        if not self.aggregate:
            return 0.0

        stats_data = self.aggregate.statistical_significance
        if "detection_probability_1d" not in stats_data:
            return 0.0

        prob_5sigma = stats_data["detection_probability_1d"].get("detection_probability_5sigma", 0)
        return prob_5sigma

    def _extract_all_kappa_values(self) -> List[float]:
        """Extract all kappa values from results"""
        kappa_values = []
        if not hasattr(self.aggregator, "results") or not self.aggregator.results:
            return kappa_values

        for phase_results in self.aggregator.results.values():
            for result in phase_results:
                if result.get("simulation_success"):
                    kappa_list = result.get("kappa", [])
                    if kappa_list:
                        kappa_values.extend(kappa_list)

        return kappa_values

    def _generate_visualization_metadata(self) -> Dict[str, Any]:
        """Generate metadata for visualization bundle"""
        return {
            "experiment_id": self.experiment_id,
            "generation_time": self._get_current_timestamp(),
            "matplotlib_version": plt.__version__,
            "seaborn_version": sns.__version__,
            "style_settings": self.style_settings,
            "figure_count": len(self._get_figure_specifications()),
        }

    def _get_figure_specifications(self) -> List[FigureSpecification]:
        """Get all figure specifications"""
        return [
            self._create_phase_progression_spec(),
            self._create_parameter_sensitivity_spec(),
            self._create_cross_phase_correlation_spec(),
            self._create_detection_time_distribution_spec(),
            self._create_statistical_significance_spec(),
            self._create_success_metrics_spec(),
            self._create_optimization_trajectory_spec(),
            self._create_validation_summary_spec(),
        ]

    def _get_current_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime

        return datetime.now().isoformat()

    def _save_visualization_bundle(self, bundle: VisualizationBundle) -> None:
        """Save visualization bundle to disk"""
        bundle_file = self.viz_dir / "visualization_bundle.json"

        with open(bundle_file, "w") as f:
            json.dump(asdict(bundle), f, indent=2, default=str)

        self.logger.info(f"Saved visualization bundle to {bundle_file}")


def main():
    """Main entry point for visualization pipeline"""
    import argparse

    parser = argparse.ArgumentParser(description="Visualization Pipeline")
    parser.add_argument("experiment_id", help="Experiment ID to generate visualizations for")
    parser.add_argument("--figure_type", help="Generate specific figure type")
    parser.add_argument("--list_figures", action="store_true", help="List available figure types")

    args = parser.parse_args()

    if args.list_figures:
        print("Available figure types:")
        print("- phase_progression: Phase progression of detection metrics")
        print("- parameter_sensitivity: Parameter sensitivity analysis")
        print("- cross_phase_correlation: Cross-phase correlation matrix")
        print("- detection_time_distribution: Detection time distribution analysis")
        print("- statistical_significance: Statistical significance analysis")
        print("- success_metrics: Comprehensive success metrics")
        print("- optimization_trajectory: Bayesian optimization trajectory")
        print("- validation_summary: Comprehensive validation summary")
        return

    # Generate visualizations
    pipeline = VisualizationPipeline(args.experiment_id)

    if not pipeline.load_experiment_data():
        print(f"Failed to load experiment data for {args.experiment_id}")
        return 1

    try:
        if args.figure_type:
            # Generate specific figure
            spec = None
            if args.figure_type == "phase_progression":
                spec = pipeline._create_phase_progression_spec()
            elif args.figure_type == "parameter_sensitivity":
                spec = pipeline._create_parameter_sensitivity_spec()
            # Add other figure types as needed

            if spec:
                file_paths = pipeline._generate_figure(spec)
                print(f"Generated {len(file_paths)} files for {args.figure_type}")
            else:
                print(f"Unknown figure type: {args.figure_type}")
                return 1
        else:
            # Generate all figures
            bundle = pipeline.generate_publication_figures()
            print(f"Generated {len(bundle.generated_files)} visualization files")
            print(f"Visualizations saved to: {pipeline.viz_dir}")

        return 0

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        return 1


if __name__ == "__main__":
    main()
