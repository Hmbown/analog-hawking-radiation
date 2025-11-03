#!/usr/bin/env python3
"""
Test Script for Reporting and Synthesis Workflow

Tests the complete reporting system including report generation, visualization,
synthesis analysis, publication formatting, and integration with orchestration components.
"""

import json
import sys
from pathlib import Path

# Add project paths to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from scripts.reporting.integration import ReportingIntegration
from scripts.reporting.publication_formatter import PublicationFormatter
from scripts.reporting.report_generator import ReportGenerator, ScientificReport
from scripts.reporting.synthesis_engine import SynthesisEngine
from scripts.reporting.visualization_pipeline import VisualizationPipeline


class TestReportingWorkflow:
    """Test class for the complete reporting workflow"""

    def __init__(self):
        self.test_experiment_id = "test_reporting_workflow"
        self.test_dir = Path("results/orchestration") / self.test_experiment_id
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Test components
        self.report_generator = None
        self.visualization_pipeline = None
        self.synthesis_engine = None
        self.publication_formatter = None
        self.integration = None

        self.test_results = {}

    def setup_test_data(self):
        """Create mock experiment data for testing"""
        print("Setting up test data...")

        # Create mock experiment structure
        experiment_structure = {
            "experiment_id": self.test_experiment_id,
            "name": "Test Reporting Workflow",
            "description": "Comprehensive test of reporting system",
            "start_time": "2024-01-01T00:00:00Z",
            "end_time": "2024-01-01T01:00:00Z",
            "phases": [
                "phase_1_initial_exploration",
                "phase_2_refinement",
                "phase_3_optimization",
                "phase_4_validation",
            ],
            "current_phase": "phase_4_validation",
            "total_simulations": 150,
            "successful_simulations": 120,
            "failed_simulations": 30,
        }

        # Save experiment manifest
        manifest_file = self.test_dir / "experiment_manifest.json"
        with open(manifest_file, "w") as f:
            json.dump(experiment_structure, f, indent=2)

        # Create mock phase results
        for phase_name in experiment_structure["phases"]:
            phase_dir = self.test_dir / phase_name
            phase_dir.mkdir(parents=True, exist_ok=True)

            # Create mock simulation results
            phase_results = []
            for i in range(25):
                result = {
                    "simulation_success": True,
                    "parameters_used": {
                        "plasma_density": 5e17 + i * 1e16,
                        "laser_intensity": 5e17 + i * 2e16,
                        "temperature_constant": 1e4 + i * 100,
                        "mirror_D": 10e-6,
                        "mirror_eta": 1.0,
                    },
                    "t5sigma_s": 1e-6 / (i + 1),
                    "kappa": [1e12 + i * 1e11],
                    "T_sig_K": 1e6 + i * 1e5,
                    "detection_probability": 0.8 + i * 0.01,
                    "statistical_significance": 3.0 + i * 0.1,
                }
                phase_results.append(result)

            # Save phase results
            results_file = phase_dir / "simulation_results.json"
            with open(results_file, "w") as f:
                json.dump(phase_results, f, indent=2)

            # Create phase config
            phase_config = {
                "phase_name": phase_name,
                "description": f"Test phase {phase_name}",
                "parameter_space": {
                    "plasma_density": {"min": 1e17, "max": 1e18, "log_scale": True},
                    "laser_intensity": {"min": 1e17, "max": 1e18, "log_scale": True},
                    "temperature_constant": {"min": 1e3, "max": 1e5, "log_scale": True},
                },
                "convergence": {
                    "moving_average_window": 10,
                    "improvement_threshold": 0.01,
                    "min_samples_for_convergence": 20,
                },
            }

            config_file = phase_dir / "phase_config.json"
            with open(config_file, "w") as f:
                json.dump(phase_config, f, indent=2)

        print("Test data setup complete")

    def test_report_generator(self):
        """Test automated report generation"""
        print("\n=== Testing Report Generator ===")

        try:
            self.report_generator = ReportGenerator(self.test_experiment_id)

            # Test loading experiment data
            load_success = self.report_generator.load_experiment_data()
            self.test_results["report_generator_load"] = load_success
            print(f"Data loading: {'SUCCESS' if load_success else 'FAILED'}")

            if load_success:
                # Test scientific report generation
                scientific_report = self.report_generator.generate_scientific_report()
                self.test_results["scientific_report"] = isinstance(
                    scientific_report, ScientificReport
                )
                print(
                    f"Scientific report: {'SUCCESS' if self.test_results['scientific_report'] else 'FAILED'}"
                )

                # Test executive summary generation
                executive_summary = self.report_generator.generate_executive_summary()
                self.test_results["executive_summary"] = executive_summary is not None
                print(
                    f"Executive summary: {'SUCCESS' if self.test_results['executive_summary'] else 'FAILED'}"
                )

                # Test technical report generation
                technical_report = self.report_generator.generate_technical_report()
                self.test_results["technical_report"] = technical_report is not None
                print(
                    f"Technical report: {'SUCCESS' if self.test_results['technical_report'] else 'FAILED'}"
                )

                # Save reports for inspection
                reports_dir = self.test_dir / "reports"
                reports_dir.mkdir(exist_ok=True)

                # Save scientific report
                if self.test_results["scientific_report"]:
                    report_file = reports_dir / "scientific_report.json"
                    with open(report_file, "w") as f:
                        json.dump(
                            {
                                "title": scientific_report.title,
                                "abstract": scientific_report.abstract,
                                "sections": list(scientific_report.sections.keys()),
                                "results_summary": "Available",
                            },
                            f,
                            indent=2,
                        )

            return all(
                [
                    self.test_results.get("report_generator_load", False),
                    self.test_results.get("scientific_report", False),
                    self.test_results.get("executive_summary", False),
                    self.test_results.get("technical_report", False),
                ]
            )

        except Exception as e:
            print(f"Report generator test failed: {e}")
            return False

    def test_visualization_pipeline(self):
        """Test visualization pipeline"""
        print("\n=== Testing Visualization Pipeline ===")

        try:
            self.visualization_pipeline = VisualizationPipeline(self.test_experiment_id)

            # Test loading experiment data
            load_success = self.visualization_pipeline.load_experiment_data()
            self.test_results["visualization_load"] = load_success
            print(f"Data loading: {'SUCCESS' if load_success else 'FAILED'}")

            if load_success:
                # Test basic figure generation
                basic_figures = self.visualization_pipeline.generate_basic_figures()
                self.test_results["basic_figures"] = basic_figures is not None
                print(
                    f"Basic figures: {'SUCCESS' if self.test_results['basic_figures'] else 'FAILED'}"
                )

                # Test comprehensive figure generation
                comprehensive_figures = self.visualization_pipeline.generate_comprehensive_figures()
                self.test_results["comprehensive_figures"] = comprehensive_figures is not None
                print(
                    f"Comprehensive figures: {'SUCCESS' if self.test_results['comprehensive_figures'] else 'FAILED'}"
                )

                # Test publication figure generation
                publication_figures = self.visualization_pipeline.generate_publication_figures()
                self.test_results["publication_figures"] = publication_figures is not None
                print(
                    f"Publication figures: {'SUCCESS' if self.test_results['publication_figures'] else 'FAILED'}"
                )

                # Save visualization metadata
                visualizations_dir = self.test_dir / "visualizations"
                visualizations_dir.mkdir(exist_ok=True)

                if self.test_results["comprehensive_figures"]:
                    metadata_file = visualizations_dir / "visualization_metadata.json"
                    with open(metadata_file, "w") as f:
                        json.dump(
                            {
                                "figure_count": (
                                    len(publication_figures.figures) if publication_figures else 0
                                ),
                                "figure_types": (
                                    [spec.figure_id for spec in publication_figures.figures]
                                    if publication_figures
                                    else []
                                ),
                            },
                            f,
                            indent=2,
                        )

            return all(
                [
                    self.test_results.get("visualization_load", False),
                    self.test_results.get("basic_figures", False),
                    self.test_results.get("comprehensive_figures", False),
                    self.test_results.get("publication_figures", False),
                ]
            )

        except Exception as e:
            print(f"Visualization pipeline test failed: {e}")
            return False

    def test_synthesis_engine(self):
        """Test synthesis engine"""
        print("\n=== Testing Synthesis Engine ===")

        try:
            self.synthesis_engine = SynthesisEngine(self.test_experiment_id)

            # Test loading experiment data
            load_success = self.synthesis_engine.load_experiment_data()
            self.test_results["synthesis_load"] = load_success
            print(f"Data loading: {'SUCCESS' if load_success else 'FAILED'}")

            if load_success:
                # Test basic synthesis
                basic_synthesis = self.synthesis_engine.perform_basic_synthesis()
                self.test_results["basic_synthesis"] = basic_synthesis is not None
                print(
                    f"Basic synthesis: {'SUCCESS' if self.test_results['basic_synthesis'] else 'FAILED'}"
                )

                # Test comprehensive synthesis
                comprehensive_synthesis = self.synthesis_engine.perform_comprehensive_synthesis()
                self.test_results["comprehensive_synthesis"] = comprehensive_synthesis is not None
                print(
                    f"Comprehensive synthesis: {'SUCCESS' if self.test_results['comprehensive_synthesis'] else 'FAILED'}"
                )

                # Test cross-phase analysis
                cross_phase_analysis = self.synthesis_engine.perform_cross_phase_analysis()
                self.test_results["cross_phase_analysis"] = cross_phase_analysis is not None
                print(
                    f"Cross-phase analysis: {'SUCCESS' if self.test_results['cross_phase_analysis'] else 'FAILED'}"
                )

                # Save synthesis metadata
                synthesis_dir = self.test_dir / "synthesis"
                synthesis_dir.mkdir(exist_ok=True)

                if self.test_results["comprehensive_synthesis"]:
                    metadata_file = synthesis_dir / "synthesis_metadata.json"
                    with open(metadata_file, "w") as f:
                        json.dump(
                            {
                                "analysis_types": (
                                    list(comprehensive_synthesis.keys())
                                    if comprehensive_synthesis
                                    else []
                                ),
                                "trend_analysis": (
                                    "Available"
                                    if comprehensive_synthesis
                                    and "trend_analysis" in comprehensive_synthesis
                                    else "Missing"
                                ),
                                "pattern_recognition": (
                                    "Available"
                                    if comprehensive_synthesis
                                    and "pattern_recognition" in comprehensive_synthesis
                                    else "Missing"
                                ),
                                "meta_analysis": (
                                    "Available"
                                    if comprehensive_synthesis
                                    and "meta_analysis" in comprehensive_synthesis
                                    else "Missing"
                                ),
                            },
                            f,
                            indent=2,
                        )

            return all(
                [
                    self.test_results.get("synthesis_load", False),
                    self.test_results.get("basic_synthesis", False),
                    self.test_results.get("comprehensive_synthesis", False),
                    self.test_results.get("cross_phase_analysis", False),
                ]
            )

        except Exception as e:
            print(f"Synthesis engine test failed: {e}")
            return False

    def test_publication_formatter(self):
        """Test publication formatter"""
        print("\n=== Testing Publication Formatter ===")

        try:
            self.publication_formatter = PublicationFormatter(self.test_experiment_id)

            # Test loading experiment data
            load_success = self.publication_formatter.load_experiment_data()
            self.test_results["publication_load"] = load_success
            print(f"Data loading: {'SUCCESS' if load_success else 'FAILED'}")

            if load_success:
                # Test LaTeX publication generation
                latex_doc = self.publication_formatter.generate_latex_publication()
                self.test_results["latex_publication"] = latex_doc is not None
                print(
                    f"LaTeX publication: {'SUCCESS' if self.test_results['latex_publication'] else 'FAILED'}"
                )

                # Test markdown documentation generation
                markdown_doc = self.publication_formatter.generate_markdown_documentation()
                self.test_results["markdown_documentation"] = markdown_doc is not None
                print(
                    f"Markdown documentation: {'SUCCESS' if self.test_results['markdown_documentation'] else 'FAILED'}"
                )

                # Test presentation slides generation
                presentation_slides = self.publication_formatter.generate_presentation_slides()
                self.test_results["presentation_slides"] = presentation_slides is not None
                print(
                    f"Presentation slides: {'SUCCESS' if self.test_results['presentation_slides'] else 'FAILED'}"
                )

                # Test data tables generation
                data_tables = self.publication_formatter.generate_data_tables()
                self.test_results["data_tables"] = data_tables is not None
                print(f"Data tables: {'SUCCESS' if self.test_results['data_tables'] else 'FAILED'}")

                # Test complete publication package
                publication_package = (
                    self.publication_formatter.generate_complete_publication_package()
                )
                self.test_results["publication_package"] = publication_package is not None
                print(
                    f"Publication package: {'SUCCESS' if self.test_results['publication_package'] else 'FAILED'}"
                )

            return all(
                [
                    self.test_results.get("publication_load", False),
                    self.test_results.get("latex_publication", False),
                    self.test_results.get("markdown_documentation", False),
                    self.test_results.get("presentation_slides", False),
                    self.test_results.get("data_tables", False),
                    self.test_results.get("publication_package", False),
                ]
            )

        except Exception as e:
            print(f"Publication formatter test failed: {e}")
            return False

    def test_integration(self):
        """Test reporting integration with orchestration components"""
        print("\n=== Testing Reporting Integration ===")

        try:
            self.integration = ReportingIntegration(self.test_experiment_id)

            # Test integration with orchestration engine
            orchestration_integration = self.integration.integrate_with_orchestration_engine()
            self.test_results["orchestration_integration"] = orchestration_integration
            print(
                f"Orchestration integration: {'SUCCESS' if orchestration_integration else 'FAILED'}"
            )

            # Test integration with monitoring dashboard
            dashboard_integration = self.integration.integrate_with_monitoring_dashboard()
            self.test_results["dashboard_integration"] = dashboard_integration
            print(f"Dashboard integration: {'SUCCESS' if dashboard_integration else 'FAILED'}")

            # Test integration with result aggregator
            aggregator_integration = self.integration.integrate_with_result_aggregator()
            self.test_results["aggregator_integration"] = aggregator_integration
            print(f"Aggregator integration: {'SUCCESS' if aggregator_integration else 'FAILED'}")

            # Test integration with validation framework
            validation_integration = self.integration.integrate_with_validation_framework()
            self.test_results["validation_integration"] = validation_integration
            print(f"Validation integration: {'SUCCESS' if validation_integration else 'FAILED'}")

            # Test complete integration
            complete_integration = self.integration.perform_complete_integration()
            self.test_results["complete_integration"] = complete_integration
            print(f"Complete integration: {'SUCCESS' if complete_integration else 'FAILED'}")

            # Get integration status
            integration_status = self.integration.get_integration_status()
            self.test_results["integration_status"] = integration_status is not None
            print(
                f"Integration status: {'SUCCESS' if self.test_results['integration_status'] else 'FAILED'}"
            )

            return all(
                [
                    self.test_results.get("orchestration_integration", False),
                    self.test_results.get("dashboard_integration", False),
                    self.test_results.get("aggregator_integration", False),
                    self.test_results.get("validation_integration", False),
                    self.test_results.get("complete_integration", False),
                    self.test_results.get("integration_status", False),
                ]
            )

        except Exception as e:
            print(f"Integration test failed: {e}")
            return False

    def run_complete_workflow_test(self):
        """Run complete reporting workflow test"""
        print("=" * 60)
        print("COMPREHENSIVE REPORTING WORKFLOW TEST")
        print("=" * 60)

        # Setup test data
        self.setup_test_data()

        # Run individual component tests
        component_results = {
            "report_generator": self.test_report_generator(),
            "visualization_pipeline": self.test_visualization_pipeline(),
            "synthesis_engine": self.test_synthesis_engine(),
            "publication_formatter": self.test_publication_formatter(),
            "integration": self.test_integration(),
        }

        # Generate test summary
        self.generate_test_summary(component_results)

        return all(component_results.values())

    def generate_test_summary(self, component_results):
        """Generate comprehensive test summary"""
        print("\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)

        summary = {
            "test_experiment_id": self.test_experiment_id,
            "test_timestamp": "2024-01-01T00:00:00Z",  # Would use datetime.now() in real scenario
            "component_results": component_results,
            "detailed_results": self.test_results,
            "overall_success": all(component_results.values()),
        }

        # Print summary to console
        for component, success in component_results.items():
            status = "PASS" if success else "FAIL"
            print(f"{component:25} : {status}")

        print(f"\nOverall test result: {'PASS' if summary['overall_success'] else 'FAIL'}")

        # Save detailed test report
        summary_file = self.test_dir / "test_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nDetailed test results saved to: {summary_file}")

        return summary


def main():
    """Main entry point for reporting workflow test"""
    import argparse

    parser = argparse.ArgumentParser(description="Test Reporting Workflow")
    parser.add_argument(
        "--component",
        choices=["all", "reports", "visualization", "synthesis", "publication", "integration"],
        default="all",
        help="Specific component to test",
    )

    args = parser.parse_args()

    # Run the test
    test_runner = TestReportingWorkflow()

    if args.component == "all":
        success = test_runner.run_complete_workflow_test()
    else:
        test_runner.setup_test_data()

        component_tests = {
            "reports": test_runner.test_report_generator,
            "visualization": test_runner.test_visualization_pipeline,
            "synthesis": test_runner.test_synthesis_engine,
            "publication": test_runner.test_publication_formatter,
            "integration": test_runner.test_integration,
        }

        success = component_tests[args.component]()

        # Generate summary for single component test
        test_runner.generate_test_summary({args.component: success})

    return 0 if success else 1


if __name__ == "__main__":
    main()
