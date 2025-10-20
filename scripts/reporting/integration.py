#!/usr/bin/env python3
"""
Integration Module for Reporting System

Provides seamless integration between the reporting framework and existing
orchestration components including the orchestration engine, monitoring dashboard,
result aggregator, and validation framework.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project paths to Python path
import sys
# Ensure repository root and src/ are importable so `from scripts.*` works
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from scripts.reporting.report_generator import ReportGenerator
from scripts.reporting.visualization_pipeline import VisualizationPipeline
from scripts.reporting.synthesis_engine import SynthesisEngine
from scripts.reporting.publication_formatter import PublicationFormatter


@dataclass
class IntegrationStatus:
    """Status tracking for reporting system integration"""
    orchestration_engine: bool = False
    monitoring_dashboard: bool = False
    result_aggregator: bool = False
    validation_framework: bool = False
    overall_status: bool = False
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ReportingIntegration:
    """Main integration class for connecting reporting system with orchestration components"""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.experiment_dir = Path("results/orchestration") / experiment_id
        self.integration_dir = self.experiment_dir / "integration"
        self.integration_dir.mkdir(parents=True, exist_ok=True)
        
        # Integration components
        self.report_generator = ReportGenerator(experiment_id)
        self.visualization_pipeline = VisualizationPipeline(experiment_id)
        self.synthesis_engine = SynthesisEngine(experiment_id)
        self.publication_formatter = PublicationFormatter(experiment_id)
        
        # Integration status
        self.status = IntegrationStatus()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Initialized reporting integration for experiment {experiment_id}")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_dir = self.experiment_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / 'reporting_integration.log'),
                logging.StreamHandler()
            ]
        )
    
    def integrate_with_orchestration_engine(self) -> bool:
        """Integrate reporting system with orchestration engine"""
        try:
            self.logger.info("Integrating with orchestration engine")
            
            # Load orchestration engine
            from scripts.orchestration_engine import OrchestrationEngine
            
            # Create integration hooks
            integration_hooks = {
                "phase_completion": self._on_phase_completion,
                "experiment_completion": self._on_experiment_completion,
                "optimization_update": self._on_optimization_update,
                "validation_result": self._on_validation_result
            }
            
            # Save integration configuration
            integration_config = {
                "experiment_id": self.experiment_id,
                "integration_hooks": list(integration_hooks.keys()),
                "reporting_components": [
                    "report_generator",
                    "visualization_pipeline", 
                    "synthesis_engine",
                    "publication_formatter"
                ],
                "integration_timestamp": datetime.now().isoformat()
            }
            
            # Write integration config
            config_file = self.integration_dir / "orchestration_integration.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(integration_config, f, indent=2)
            
            self.status.orchestration_engine = True
            self._update_overall_status()
            
            self.logger.info("Successfully integrated with orchestration engine")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with orchestration engine: {e}")
            return False
    
    def integrate_with_monitoring_dashboard(self) -> bool:
        """Integrate reporting system with monitoring dashboard"""
        try:
            self.logger.info("Integrating with monitoring dashboard")
            
            # Load monitoring dashboard
            from scripts.monitoring.dashboard import MonitoringDashboard
            
            # Create dashboard integration
            dashboard_integration = {
                "reporting_metrics": {
                    "report_generation_status": self._get_report_generation_status,
                    "visualization_status": self._get_visualization_status,
                    "synthesis_status": self._get_synthesis_status,
                    "publication_status": self._get_publication_status
                },
                "dashboard_widgets": [
                    {
                        "name": "Reporting Status",
                        "type": "status_panel",
                        "data_source": "reporting_integration"
                    },
                    {
                        "name": "Publication Progress", 
                        "type": "progress_bar",
                        "data_source": "publication_formatter"
                    }
                ]
            }
            
            # Save dashboard integration config
            dashboard_config = {
                "experiment_id": self.experiment_id,
                "dashboard_integration": dashboard_integration,
                "integration_timestamp": datetime.now().isoformat()
            }
            
            config_file = self.integration_dir / "dashboard_integration.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(dashboard_config, f, indent=2)
            
            self.status.monitoring_dashboard = True
            self._update_overall_status()
            
            self.logger.info("Successfully integrated with monitoring dashboard")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with monitoring dashboard: {e}")
            return False
    
    def integrate_with_result_aggregator(self) -> bool:
        """Integrate reporting system with result aggregator"""
        try:
            self.logger.info("Integrating with result aggregator")
            
            # Load result aggregator
            from scripts.result_aggregator import ResultAggregator
            
            # Create result aggregator integration
            aggregator_integration = {
                "reporting_data_sources": [
                    "scientific_reports",
                    "executive_summaries", 
                    "technical_reports",
                    "visualization_bundles",
                    "synthesis_reports",
                    "publication_materials"
                ],
                "data_formats": ["json", "csv", "pdf", "html", "tex"],
                "integration_methods": {
                    "direct_access": True,
                    "file_based": True,
                    "api_endpoints": False
                }
            }
            
            # Save aggregator integration config
            aggregator_config = {
                "experiment_id": self.experiment_id,
                "aggregator_integration": aggregator_integration,
                "integration_timestamp": datetime.now().isoformat()
            }
            
            config_file = self.integration_dir / "aggregator_integration.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(aggregator_config, f, indent=2)
            
            self.status.result_aggregator = True
            self._update_overall_status()
            
            self.logger.info("Successfully integrated with result aggregator")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with result aggregator: {e}")
            return False
    
    def integrate_with_validation_framework(self) -> bool:
        """Integrate reporting system with validation framework"""
        try:
            self.logger.info("Integrating with validation framework")
            
            # Load validation framework
            from scripts.validation.validation_framework import ValidationFramework
            
            # Create validation integration
            validation_integration = {
                "validation_checks": [
                    {
                        "name": "report_generation_validation",
                        "description": "Validate scientific report generation",
                        "validation_function": self._validate_report_generation
                    },
                    {
                        "name": "visualization_validation", 
                        "description": "Validate visualization pipeline",
                        "validation_function": self._validate_visualization_pipeline
                    },
                    {
                        "name": "synthesis_validation",
                        "description": "Validate synthesis engine",
                        "validation_function": self._validate_synthesis_engine
                    },
                    {
                        "name": "publication_validation",
                        "description": "Validate publication formatter", 
                        "validation_function": self._validate_publication_formatter
                    }
                ],
                "integration_methods": {
                    "direct_validation": True,
                    "cross_validation": True,
                    "statistical_validation": True
                }
            }
            
            # Save validation integration config
            validation_config = {
                "experiment_id": self.experiment_id,
                "validation_integration": validation_integration,
                "integration_timestamp": datetime.now().isoformat()
            }
            
            config_file = self.integration_dir / "validation_integration.json"
            import json
            with open(config_file, 'w') as f:
                json.dump(validation_config, f, indent=2)
            
            self.status.validation_framework = True
            self._update_overall_status()
            
            self.logger.info("Successfully integrated with validation framework")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to integrate with validation framework: {e}")
            return False
    
    def perform_complete_integration(self) -> bool:
        """Perform complete integration with all components"""
        self.logger.info("Performing complete reporting system integration")
        
        integration_results = {
            "orchestration_engine": self.integrate_with_orchestration_engine(),
            "monitoring_dashboard": self.integrate_with_monitoring_dashboard(),
            "result_aggregator": self.integrate_with_result_aggregator(),
            "validation_framework": self.integrate_with_validation_framework()
        }
        
        # Save integration report
        integration_report = {
            "experiment_id": self.experiment_id,
            "integration_results": integration_results,
            "overall_status": self.status.overall_status,
            "integration_timestamp": datetime.now().isoformat(),
            "components_integrated": [
                component for component, status in integration_results.items() if status
            ],
            "components_failed": [
                component for component, status in integration_results.items() if not status
            ]
        }
        
        report_file = self.integration_dir / "integration_report.json"
        import json
        with open(report_file, 'w') as f:
            json.dump(integration_report, f, indent=2)
        
        if self.status.overall_status:
            self.logger.info("Complete integration successful")
        else:
            self.logger.warning("Integration completed with some failures")
        
        return self.status.overall_status
    
    def _on_phase_completion(self, phase_data: Dict[str, Any]) -> None:
        """Callback for phase completion events"""
        self.logger.info(f"Phase completion event: {phase_data.get('phase_name', 'unknown')}")
        
        # Generate phase-specific reports
        try:
            # Generate phase summary report
            phase_report = self.report_generator.generate_phase_summary(phase_data)
            
            # Generate phase visualizations
            phase_visualizations = self.visualization_pipeline.generate_phase_figures(phase_data)
            
            # Update synthesis with phase data
            self.synthesis_engine.update_with_phase_data(phase_data)
            
            self.logger.info(f"Generated reports for phase {phase_data.get('phase_name')}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate phase reports: {e}")
    
    def _on_experiment_completion(self, experiment_data: Dict[str, Any]) -> None:
        """Callback for experiment completion events"""
        self.logger.info("Experiment completion event")
        
        # Generate comprehensive reports
        try:
            # Load all experiment data
            if not self.report_generator.load_experiment_data():
                self.logger.error("Failed to load experiment data for reporting")
                return
            
            # Generate all report types
            scientific_report = self.report_generator.generate_scientific_report()
            executive_summary = self.report_generator.generate_executive_summary()
            technical_report = self.report_generator.generate_technical_report()
            
            # Generate all visualizations
            visualization_bundle = self.visualization_pipeline.generate_comprehensive_figures()
            
            # Perform comprehensive synthesis
            synthesis_report = self.synthesis_engine.perform_comprehensive_synthesis()
            
            # Generate publication materials
            publication_materials = self.publication_formatter.generate_complete_publication_package()
            
            self.logger.info("Generated comprehensive reporting package for experiment completion")
            
        except Exception as e:
            self.logger.error(f"Failed to generate experiment completion reports: {e}")
    
    def _on_optimization_update(self, optimization_data: Dict[str, Any]) -> None:
        """Callback for optimization update events"""
        self.logger.debug(f"Optimization update: {optimization_data.get('iteration', 'unknown')}")
        
        # Update real-time optimization tracking
        try:
            # Update optimization progress visualization
            self.visualization_pipeline.update_optimization_progress(optimization_data)
            
            # Update synthesis with optimization data
            self.synthesis_engine.update_optimization_trajectory(optimization_data)
            
        except Exception as e:
            self.logger.error(f"Failed to process optimization update: {e}")
    
    def _on_validation_result(self, validation_data: Dict[str, Any]) -> None:
        """Callback for validation result events"""
        self.logger.info(f"Validation result: {validation_data.get('validation_type', 'unknown')}")
        
        # Incorporate validation results into reporting
        try:
            # Update reports with validation findings
            self.report_generator.incorporate_validation_results(validation_data)
            
            # Update visualizations with validation metrics
            self.visualization_pipeline.update_validation_visualizations(validation_data)
            
            # Update synthesis with validation insights
            self.synthesis_engine.incorporate_validation_insights(validation_data)
            
        except Exception as e:
            self.logger.error(f"Failed to incorporate validation results: {e}")
    
    def _get_report_generation_status(self) -> Dict[str, Any]:
        """Get report generation status for dashboard"""
        return {
            "scientific_report": self.report_generator.scientific_report is not None,
            "executive_summary": self.report_generator.executive_summary is not None,
            "technical_report": self.report_generator.technical_report is not None,
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_visualization_status(self) -> Dict[str, Any]:
        """Get visualization status for dashboard"""
        return {
            "phase_figures": len(self.visualization_pipeline.phase_figures) if hasattr(self.visualization_pipeline, 'phase_figures') else 0,
            "publication_figures": len(self.visualization_pipeline.publication_figures) if hasattr(self.visualization_pipeline, 'publication_figures') else 0,
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_synthesis_status(self) -> Dict[str, Any]:
        """Get synthesis status for dashboard"""
        return {
            "trend_analysis": self.synthesis_engine.trend_analysis is not None,
            "pattern_recognition": self.synthesis_engine.pattern_recognition is not None,
            "meta_analysis": self.synthesis_engine.meta_analysis is not None,
            "last_updated": datetime.now().isoformat()
        }
    
    def _get_publication_status(self) -> Dict[str, Any]:
        """Get publication status for dashboard"""
        return {
            "latex_document": hasattr(self.publication_formatter, 'latex_document_generated'),
            "markdown_documentation": hasattr(self.publication_formatter, 'markdown_documentation_generated'),
            "presentation_slides": hasattr(self.publication_formatter, 'presentation_slides_generated'),
            "data_tables": hasattr(self.publication_formatter, 'data_tables_generated'),
            "last_updated": datetime.now().isoformat()
        }
    
    def _validate_report_generation(self) -> Dict[str, Any]:
        """Validate report generation functionality"""
        try:
            # Test report generation
            test_report = self.report_generator.generate_scientific_report()
            
            return {
                "status": "PASS",
                "message": "Report generation validated successfully",
                "details": {
                    "report_type": "scientific_report",
                    "sections_generated": len(test_report.sections) if hasattr(test_report, 'sections') else 0,
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Report generation validation failed: {e}",
                "details": {
                    "error": str(e),
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
    
    def _validate_visualization_pipeline(self) -> Dict[str, Any]:
        """Validate visualization pipeline functionality"""
        try:
            # Test visualization generation
            test_visualizations = self.visualization_pipeline.generate_basic_figures()
            
            return {
                "status": "PASS",
                "message": "Visualization pipeline validated successfully",
                "details": {
                    "figures_generated": len(test_visualizations.figures) if hasattr(test_visualizations, 'figures') else 0,
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Visualization pipeline validation failed: {e}",
                "details": {
                    "error": str(e),
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
    
    def _validate_synthesis_engine(self) -> Dict[str, Any]:
        """Validate synthesis engine functionality"""
        try:
            # Test synthesis analysis
            test_synthesis = self.synthesis_engine.perform_basic_synthesis()
            
            return {
                "status": "PASS",
                "message": "Synthesis engine validated successfully",
                "details": {
                    "analysis_types": len(test_synthesis.analyses) if hasattr(test_synthesis, 'analyses') else 0,
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Synthesis engine validation failed: {e}",
                "details": {
                    "error": str(e),
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
    
    def _validate_publication_formatter(self) -> Dict[str, Any]:
        """Validate publication formatter functionality"""
        try:
            # Test publication formatting
            test_publication = self.publication_formatter.generate_latex_publication()
            
            return {
                "status": "PASS",
                "message": "Publication formatter validated successfully",
                "details": {
                    "document_sections": len(test_publication.sections) if hasattr(test_publication, 'sections') else 0,
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            return {
                "status": "FAIL",
                "message": f"Publication formatter validation failed: {e}",
                "details": {
                    "error": str(e),
                    "validation_timestamp": datetime.now().isoformat()
                }
            }
    
    def _update_overall_status(self) -> None:
        """Update overall integration status"""
        self.status.overall_status = (
            self.status.orchestration_engine and
            self.status.monitoring_dashboard and
            self.status.result_aggregator and
            self.status.validation_framework
        )
        self.status.last_updated = datetime.now().isoformat()
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get comprehensive integration status"""
        return {
            "experiment_id": self.experiment_id,
            "status": asdict(self.status),
            "integration_components": {
                "orchestration_engine": {
                    "integrated": self.status.orchestration_engine,
                    "description": "Integration with main orchestration engine"
                },
                "monitoring_dashboard": {
                    "integrated": self.status.monitoring_dashboard,
                    "description": "Integration with real-time monitoring dashboard"
                },
                "result_aggregator": {
                    "integrated": self.status.result_aggregator,
                    "description": "Integration with result aggregation framework"
                },
                "validation_framework": {
                    "integrated": self.status.validation_framework,
                    "description": "Integration with validation framework"
                }
            },
            "reporting_components": {
                "report_generator": {
                    "status": "READY",
                    "description": "Automated report generation"
                },
                "visualization_pipeline": {
                    "status": "READY", 
                    "description": "Publication-quality visualization"
                },
                "synthesis_engine": {
                    "status": "READY",
                    "description": "Cross-phase synthesis and analysis"
                },
                "publication_formatter": {
                    "status": "READY",
                    "description": "Multi-format publication outputs"
                }
            }
        }


def main():
    """Main entry point for reporting integration"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Reporting System Integration")
    parser.add_argument("experiment_id", help="Experiment ID to integrate reporting for")
    parser.add_argument("--component", choices=["all", "orchestration", "dashboard", "aggregator", "validation"],
                       default="all", help="Specific component to integrate")
    
    args = parser.parse_args()
    
    # Perform integration
    integration = ReportingIntegration(args.experiment_id)
    
    if args.component == "all":
        success = integration.perform_complete_integration()
    else:
        component_map = {
            "orchestration": integration.integrate_with_orchestration_engine,
            "dashboard": integration.integrate_with_monitoring_dashboard,
            "aggregator": integration.integrate_with_result_aggregator,
            "validation": integration.integrate_with_validation_framework
        }
        success = component_map[args.component]()
    
    # Print integration status
    status = integration.get_integration_status()
    import json
    print(json.dumps(status, indent=2))
    
    return 0 if success else 1


if __name__ == "__main__":
    main()
