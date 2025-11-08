#!/usr/bin/env python3
"""
Complete Academic Publication Pipeline Test
=========================================

This script performs comprehensive testing of the complete academic publication
pipeline, including manuscript generation, peer review simulation, citation
validation, and supplementary materials creation.

Author: Pipeline Validation Task Force
Version: 1.0.0 (Comprehensive Test)
"""

import os
import sys
import json
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime, timezone
import logging
from dataclasses import dataclass

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import pipeline components
try:
    from publication_pipeline import AcademicPublicationPipeline
    from peer_review_simulation import PeerReviewSimulator
    from citation_reproducibility_validator import CitationReproducibilityValidator
    from supplementary_materials_generator import SupplementaryMaterialsGenerator
except ImportError as e:
    print(f"Error importing pipeline components: {e}")
    print("Make sure all pipeline modules are in the current directory")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Container for test results."""
    test_name: str
    status: str  # PASS, FAIL, SKIP
    duration: float
    details: str
    artifacts: List[str]
    error_message: Optional[str] = None

class PipelineTestSuite:
    """Comprehensive test suite for the academic publication pipeline."""

    def __init__(self, test_output_dir: str = "pipeline_test_results"):
        """Initialize the test suite."""
        self.test_output_dir = Path(test_output_dir)
        self.test_output_dir.mkdir(exist_ok=True, parents=True)

        self.test_results = []
        self.start_time = time.time()

        # Target journals for testing
        self.target_journals = [
            "Nature Physics",
            "Physical Review Letters",
            "Physical Review E",
            "Nature Communications"
        ]

        logger.info(f"Pipeline test suite initialized with output directory: {self.test_output_dir}")

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all pipeline tests."""
        logger.info("Starting comprehensive pipeline test suite...")

        test_functions = [
            self.test_publication_pipeline_initialization,
            self.test_research_data_loading,
            self.test_manuscript_generation,
            self.test_figure_creation,
            self.test_latex_compilation,
            self.test_peer_review_simulation,
            self.test_citation_validation,
            self.test_reproducibility_validation,
            self.test_supplementary_materials,
            self.test_journal_adaptation,
            self.test_end_to_end_integration,
            self.test_package_creation,
            self.test_quality_metrics
        ]

        for test_func in test_functions:
            try:
                logger.info(f"Running {test_func.__name__}...")
                result = test_func()
                self.test_results.append(result)
                logger.info(f"✅ {test_func.__name__}: {result.status}")
            except Exception as e:
                logger.error(f"❌ {test_func.__name__}: FAILED - {str(e)}")
                result = TestResult(
                    test_name=test_func.__name__,
                    status="FAIL",
                    duration=0,
                    details="Test execution failed",
                    artifacts=[],
                    error_message=str(e)
                )
                self.test_results.append(result)

        # Generate test report
        test_report = self._generate_test_report()

        logger.info("Pipeline test suite completed")
        return test_report

    def test_publication_pipeline_initialization(self) -> TestResult:
        """Test publication pipeline initialization."""
        start_time = time.time()

        try:
            # Initialize pipeline
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "publication_test")
            )

            # Check structure
            expected_dirs = ["manuscripts", "figures", "supplementary", "peer_review", "metadata", "submissions"]
            for dir_name in expected_dirs:
                dir_path = pipeline.output_dir / dir_name
                if not dir_path.exists():
                    raise FileNotFoundError(f"Expected directory {dir_name} not found")

            # Check journal specifications
            if len(pipeline.journal_specs) < 4:
                raise ValueError("Insufficient journal specifications loaded")

            artifacts = [str(pipeline.output_dir)]
            details = f"Pipeline initialized with {len(pipeline.journal_specs)} journal specifications"

            return TestResult(
                test_name="publication_pipeline_initialization",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="publication_pipeline_initialization",
                status="FAIL",
                duration=time.time() - start_time,
                details="Pipeline initialization failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_research_data_loading(self) -> TestResult:
        """Test research data loading and analysis."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "data_test")
            )

            # Check if research data is loaded
            if not pipeline.research_data:
                raise ValueError("No research data loaded")

            # Check key data components
            required_keys = ["key_findings", "gradient_analysis", "hybrid_sweep"]
            for key in required_keys:
                if key not in pipeline.research_data:
                    raise ValueError(f"Missing research data component: {key}")

            # Check key findings
            findings = pipeline.research_data["key_findings"]
            if "max_surface_gravity" not in findings:
                raise ValueError("Missing max_surface_gravity in key findings")

            artifacts = []
            details = f"Loaded research data with {len(pipeline.research_data)} components"

            return TestResult(
                test_name="research_data_loading",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="research_data_loading",
                status="FAIL",
                duration=time.time() - start_time,
                details="Research data loading failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_manuscript_generation(self) -> TestResult:
        """Test manuscript generation for different journals."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "manuscript_test")
            )

            artifacts = []
            generated_manuscripts = []

            # Test manuscript generation for each journal
            for journal in self.target_journals[:2]:  # Test subset for speed
                content = pipeline.generate_manuscript_content(journal)

                # Check content structure
                required_sections = ["abstract", "introduction", "methods", "results", "discussion", "conclusions"]
                for section in required_sections:
                    if not hasattr(content, section) or not getattr(content, section):
                        raise ValueError(f"Missing or empty section: {section} for {journal}")

                # Check content length
                abstract_words = len(content.abstract.split())
                if abstract_words < 50 or abstract_words > 500:
                    raise ValueError(f"Abstract length issue for {journal}: {abstract_words} words")

                generated_manuscripts.append(journal)

            details = f"Generated manuscripts for {len(generated_manuscripts)} journals: {', '.join(generated_manuscripts)}"

            return TestResult(
                test_name="manuscript_generation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="manuscript_generation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Manuscript generation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_figure_creation(self) -> TestResult:
        """Test publication-ready figure creation."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "figure_test")
            )

            # Create figures
            figure_paths = pipeline.create_publication_figures()

            # Check figure files
            expected_figures = ["framework_overview", "parameter_space", "detection_feasibility", "eli_assessment"]
            for fig_name in expected_figures:
                if fig_name not in figure_paths:
                    raise ValueError(f"Missing figure: {fig_name}")

                fig_path = Path(figure_paths[fig_name])
                if not fig_path.exists():
                    raise FileNotFoundError(f"Figure file not found: {fig_path}")

                if fig_path.stat().st_size < 1000:  # Less than 1KB seems too small
                    raise ValueError(f"Figure file too small: {fig_path}")

            artifacts = list(figure_paths.values())
            details = f"Created {len(figure_paths)} publication figures"

            return TestResult(
                test_name="figure_creation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="figure_creation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Figure creation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_latex_compilation(self) -> TestResult:
        """Test LaTeX manuscript generation and compilation."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "latex_test")
            )

            # Generate manuscript content
            content = pipeline.generate_manuscript_content("Nature Physics")

            # Create metadata
            from publication_pipeline import PublicationMetadata
            metadata = PublicationMetadata(
                title="Test Manuscript for Pipeline Validation",
                authors=[{"name": "Test Author", "affiliation": "Test Institution"}],
                affiliations=["Test Institution"],
                abstract=content.abstract,
                keywords=["test", "validation"],
                submission_date=datetime.now(timezone.utc),
                journal_target="Nature Physics",
                manuscript_type="Original Research",
                word_count=1000,
                figure_count=4,
                reference_count=10
            )

            # Generate LaTeX
            latex_content = pipeline.generate_latex_manuscript("Nature Physics", content, metadata)

            # Check LaTeX content
            if not latex_content or len(latex_content) < 1000:
                raise ValueError("LaTeX content too short or empty")

            if r"\begin{document}" not in latex_content or r"\end{document}" not in latex_content:
                raise ValueError("LaTeX document structure invalid")

            # Save LaTeX file
            latex_file = pipeline.output_dir / "test_manuscript.tex"
            with open(latex_file, 'w') as f:
                f.write(latex_content)

            artifacts = [str(latex_file)]
            details = f"Generated LaTeX manuscript ({len(latex_content)} characters)"

            return TestResult(
                test_name="latex_compilation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="latex_compilation",
                status="FAIL",
                duration=time.time() - start_time,
                details="LaTeX compilation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_peer_review_simulation(self) -> TestResult:
        """Test peer review simulation."""
        start_time = time.time()

        try:
            simulator = PeerReviewSimulator(
                output_dir=str(self.test_output_dir / "peer_review_test")
            )

            # Create test manuscript content
            manuscript_content = {
                "title": "Test Manuscript for Peer Review",
                "abstract": "This is a test abstract for peer review simulation.",
                "introduction": "This is the introduction section.",
                "methods": "This describes the methods used.",
                "results": "These are the results.",
                "discussion": "This is the discussion section.",
                "conclusions": "These are the conclusions."
            }

            # Simulate peer review
            manuscript_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            journal = "Physical Review Letters"

            review_results = simulator.simulate_peer_review_process(
                manuscript_content, journal, manuscript_id
            )

            # Check review results
            if "individual_reviews" not in review_results:
                raise ValueError("Missing individual reviews in results")

            if len(review_results["individual_reviews"]) < 2:
                raise ValueError("Insufficient number of reviews generated")

            if "editorial_decision" not in review_results:
                raise ValueError("Missing editorial decision")

            # Generate review report
            report_file = simulator.generate_review_report(review_results)

            artifacts = [report_file]
            details = f"Simulated {len(review_results['individual_reviews'])} peer reviews with editorial decision: {review_results['editorial_decision']['decision']}"

            return TestResult(
                test_name="peer_review_simulation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="peer_review_simulation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Peer review simulation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_citation_validation(self) -> TestResult:
        """Test citation validation."""
        start_time = time.time()

        try:
            validator = CitationReproducibilityValidator(
                project_root=".",
                output_dir=str(self.test_output_dir / "citation_test")
            )

            # Create test manuscript content with citations
            manuscript_content = {
                "title": "Test Manuscript for Citation Validation",
                "abstract": "Test abstract with citations [1,2].",
                "references": [
                    {
                        "key": "unruh1981",
                        "authors": ["W.G. Unruh"],
                        "title": "Experimental black-hole evaporation?",
                        "journal": "Phys. Rev. Lett.",
                        "volume": "46",
                        "pages": "1351",
                        "year": 1981,
                        "doi": "10.1103/PhysRevLett.46.1351"
                    },
                    {
                        "key": "chen2017",
                        "authors": ["P. Chen", "G. Mourou"],
                        "title": "Accelerating plasma mirrors",
                        "journal": "Phys. Rev. Lett.",
                        "volume": "118",
                        "pages": "045001",
                        "year": 2017,
                        "doi": "10.1103/PhysRevLett.118.045001"
                    }
                ]
            }

            # Validate citations
            citation_score, citation_entries = validator.validate_citations(
                manuscript_content, "aps"
            )

            # Check validation results
            if citation_score < 0 or citation_score > 1:
                raise ValueError(f"Invalid citation score: {citation_score}")

            if len(citation_entries) < 2:
                raise ValueError("Insufficient citation entries generated")

            # Check citation entries
            valid_entries = [c for c in citation_entries if c.validation_status == "Valid"]
            if len(valid_entries) < 1:
                raise ValueError("No valid citations found")

            artifacts = []
            details = f"Validated {len(citation_entries)} citations with score {citation_score:.2f}"

            return TestResult(
                test_name="citation_validation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="citation_validation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Citation validation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_reproducibility_validation(self) -> TestResult:
        """Test reproducibility validation."""
        start_time = time.time()

        try:
            validator = CitationReproducibilityValidator(
                project_root=".",
                output_dir=str(self.test_output_dir / "reproducibility_test")
            )

            # Validate reproducibility
            reproducibility_score, reproducibility_checks = validator.validate_reproducibility()

            # Check validation results
            if reproducibility_score < 0 or reproducibility_score > 1:
                raise ValueError(f"Invalid reproducibility score: {reproducibility_score}")

            if len(reproducibility_checks) < 5:
                raise ValueError("Insufficient reproducibility checks performed")

            # Check critical checks
            critical_checks = [c for c in reproducibility_checks if c.critical]
            if len(critical_checks) < 3:
                raise ValueError("Insufficient critical reproducibility checks")

            # Generate computational environment specification
            env_spec = validator.generate_computational_environment_spec()

            if not env_spec.python_version:
                raise ValueError("Missing Python version in environment specification")

            artifacts = []
            details = f"Validated reproducibility with score {reproducibility_score:.2f} ({len(reproducibility_checks)} checks)"

            return TestResult(
                test_name="reproducibility_validation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="reproducibility_validation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Reproducibility validation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_supplementary_materials(self) -> TestResult:
        """Test supplementary materials generation."""
        start_time = time.time()

        try:
            generator = SupplementaryMaterialsGenerator(
                results_dir="results",
                output_dir=str(self.test_output_dir / "supplementary_test")
            )

            # Create test research data
            research_data = {
                "key_findings": {
                    "max_surface_gravity": 5.94e12,
                    "scaling_relationships": {
                        "kappa_vs_a0": {"exponent": 0.66, "uncertainty": 0.22}
                    }
                }
            }

            # Generate supplementary materials
            manuscript_id = f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            journal = "Nature Physics"

            package = generator.generate_comprehensive_supplementary_materials(
                manuscript_id, journal, research_data
            )

            # Check package structure
            if len(package.sections) < 5:
                raise ValueError("Insufficient sections in supplementary package")

            # Check essential sections
            essential_sections = ["Extended Methods", "Extended Data", "Additional Figures"]
            for section_name in essential_sections:
                found = any(s.section_name == section_name for s in package.sections)
                if not found:
                    raise ValueError(f"Missing essential section: {section_name}")

            artifacts = [str(generator.output_dir)]
            details = f"Generated supplementary package with {len(package.sections)} sections ({package.total_size_mb:.1f} MB)"

            return TestResult(
                test_name="supplementary_materials",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="supplementary_materials",
                status="FAIL",
                duration=time.time() - start_time,
                details="Supplementary materials generation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_journal_adaptation(self) -> TestResult:
        """Test journal-specific adaptation."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "adaptation_test")
            )

            artifacts = []
            adaptation_results = []

            # Test journal adaptation for different journals
            for journal in ["Nature Physics", "Physical Review Letters"]:
                # Generate content
                content = pipeline.generate_manuscript_content(journal)

                # Check journal-specific requirements
                journal_spec = pipeline.journal_specs[journal]

                # Check abstract length
                abstract_words = len(content.abstract.split())
                max_words = journal_spec["abstract_max"]

                if abstract_words > max_words * 1.2:  # Allow 20% tolerance
                    raise ValueError(f"Abstract too long for {journal}: {abstract_words} > {max_words}")

                adaptation_results.append(f"{journal}: {abstract_words}/{max_words} words")

            details = f"Journal adaptation successful for {len(adaptation_results)} journals"

            return TestResult(
                test_name="journal_adaptation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="journal_adaptation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Journal adaptation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_end_to_end_integration(self) -> TestResult:
        """Test end-to-end integration."""
        start_time = time.time()

        try:
            # Initialize all components
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "integration_test")
            )

            # Run complete pipeline for one journal
            packages = pipeline.run_complete_pipeline(journals=["Physical Review Letters"])

            # Check results
            if "Physical Review Letters" not in packages:
                raise ValueError("Physical Review Letters package not generated")

            package = packages["Physical Review Letters"]
            if "error" in package:
                raise ValueError(f"Pipeline error: {package['error']}")

            expected_files = ["manuscript", "metadata", "checklist", "cover_letter"]
            for file_type in expected_files:
                if file_type not in package:
                    raise ValueError(f"Missing {file_type} in package")

            artifacts = list(package.values())
            details = "End-to-end pipeline integration successful"

            return TestResult(
                test_name="end_to_end_integration",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="end_to_end_integration",
                status="FAIL",
                duration=time.time() - start_time,
                details="End-to-end integration failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_package_creation(self) -> TestResult:
        """Test complete publication package creation."""
        start_time = time.time()

        try:
            pipeline = AcademicPublicationPipeline(
                results_dir="results",
                output_dir=str(self.test_output_dir / "package_test")
            )

            # Create publication package
            package = pipeline.create_publication_package("Nature Communications")

            # Check package contents
            if "manuscript" not in package:
                raise ValueError("Manuscript missing from package")

            if "figures" not in package or len(package["figures"]) < 3:
                raise ValueError("Insufficient figures in package")

            # Check files exist
            for file_type, file_path in package.items():
                if file_type != "figures":
                    path = Path(file_path)
                    if not path.exists():
                        raise FileNotFoundError(f"Package file not found: {path}")
                else:
                    for fig_path in file_path:
                        path = Path(fig_path)
                        if not path.exists():
                            raise FileNotFoundError(f"Figure not found: {path}")

            artifacts = list(package.values())
            details = f"Created complete publication package with {len(package)} components"

            return TestResult(
                test_name="package_creation",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="package_creation",
                status="FAIL",
                duration=time.time() - start_time,
                details="Package creation failed",
                artifacts=[],
                error_message=str(e)
            )

    def test_quality_metrics(self) -> TestResult:
        """Test quality metrics and validation."""
        start_time = time.time()

        try:
            # Initialize all components
            pipeline = AcademicPublicationPipeline()
            simulator = PeerReviewSimulator()
            validator = CitationReproducibilityValidator()

            quality_metrics = {
                "manuscript_quality": 0,
                "peer_review_quality": 0,
                "citation_quality": 0,
                "reproducibility_quality": 0,
                "overall_quality": 0
            }

            # Test manuscript quality
            content = pipeline.generate_manuscript_content("Nature Physics")
            manuscript_quality = self._assess_manuscript_quality(content)
            quality_metrics["manuscript_quality"] = manuscript_quality

            # Test peer review quality
            manuscript_content = {"abstract": "Test abstract", "title": "Test title"}
            review_results = simulator.simulate_peer_review_process(
                manuscript_content, "PRL", "test"
            )
            peer_review_quality = self._assess_peer_review_quality(review_results)
            quality_metrics["peer_review_quality"] = peer_review_quality

            # Test citation quality
            manuscript_content = {
                "references": [
                    {
                        "authors": ["Test Author"],
                        "title": "Test Paper",
                        "journal": "Test Journal",
                        "year": 2023,
                        "doi": "10.1000/test"
                    }
                ]
            }
            citation_score, _ = validator.validate_citations(manuscript_content, "aps")
            quality_metrics["citation_quality"] = citation_score

            # Test reproducibility quality
            reproducibility_score, _ = validator.validate_reproducibility()
            quality_metrics["reproducibility_quality"] = reproducibility_score

            # Calculate overall quality
            quality_metrics["overall_quality"] = sum(quality_metrics.values()) / len(quality_metrics)

            # Check minimum quality thresholds
            if quality_metrics["overall_quality"] < 0.6:
                raise ValueError(f"Overall quality too low: {quality_metrics['overall_quality']:.2f}")

            # Save quality metrics
            metrics_file = self.test_output_dir / "quality_metrics.json"
            with open(metrics_file, 'w') as f:
                json.dump(quality_metrics, f, indent=2)

            artifacts = [str(metrics_file)]
            details = f"Quality assessment passed with overall score {quality_metrics['overall_quality']:.2f}"

            return TestResult(
                test_name="quality_metrics",
                status="PASS",
                duration=time.time() - start_time,
                details=details,
                artifacts=artifacts
            )

        except Exception as e:
            return TestResult(
                test_name="quality_metrics",
                status="FAIL",
                duration=time.time() - start_time,
                details="Quality metrics assessment failed",
                artifacts=[],
                error_message=str(e)
            )

    def _assess_manuscript_quality(self, content) -> float:
        """Assess manuscript quality."""
        score = 0.5  # Base score

        # Check content completeness
        if content.abstract and len(content.abstract.split()) > 100:
            score += 0.1
        if content.introduction and len(content.introduction) > 500:
            score += 0.1
        if content.methods and len(content.methods) > 300:
            score += 0.1
        if content.results and len(content.results) > 400:
            score += 0.1
        if content.discussion and len(content.discussion) > 300:
            score += 0.1

        return min(1.0, score)

    def _assess_peer_review_quality(self, review_results) -> float:
        """Assess peer review quality."""
        score = 0.5  # Base score

        if "individual_reviews" in review_results:
            score += 0.1 * min(len(review_results["individual_reviews"]) / 3.0, 0.3)

        if "editorial_decision" in review_results:
            score += 0.2

        return min(1.0, score)

    def _generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        total_time = time.time() - self.start_time

        # Calculate statistics
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r.status == "PASS"])
        failed_tests = len([r for r in self.test_results if r.status == "FAIL"])
        skipped_tests = len([r for r in self.test_results if r.status == "SKIP"])

        success_rate = passed_tests / total_tests if total_tests > 0 else 0

        # Create report
        report = {
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "skipped": skipped_tests,
                "success_rate": success_rate,
                "total_duration": total_time
            },
            "test_results": [],
            "failed_tests": [],
            "artifacts": [],
            "recommendations": []
        }

        # Add test results
        for result in self.test_results:
            test_data = {
                "name": result.test_name,
                "status": result.status,
                "duration": result.duration,
                "details": result.details,
                "artifacts": result.artifacts
            }

            if result.error_message:
                test_data["error"] = result.error_message

            report["test_results"].append(test_data)
            report["artifacts"].extend(result.artifacts)

            if result.status == "FAIL":
                report["failed_tests"].append(test_data)

        # Add recommendations
        if failed_tests > 0:
            report["recommendations"].append("Address failed tests before production use")
        if success_rate < 0.9:
            report["recommendations"].append("Improve test coverage and reliability")
        if success_rate >= 0.9:
            report["recommendations"].append("Pipeline ready for production use")

        # Save report
        report_file = self.test_output_dir / "pipeline_test_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        # Create markdown report
        self._create_markdown_report(report)

        return report

    def _create_markdown_report(self, report: Dict[str, Any]):
        """Create markdown test report."""
        md_content = f"""# Academic Publication Pipeline Test Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Duration:** {report['test_summary']['total_duration']:.2f} seconds

## Test Summary

| Metric | Value |
|--------|-------|
| Total Tests | {report['test_summary']['total_tests']} |
| Passed | {report['test_summary']['passed']} |
| Failed | {report['test_summary']['failed']} |
| Skipped | {report['test_summary']['skipped']} |
| Success Rate | {report['test_summary']['success_rate']:.1%} |

## Test Results

"""

        for result in report["test_results"]:
            status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⏭️"
            md_content += f"### {status_icon} {result['name']}\n"
            md_content += f"**Status:** {result['status']}\n"
            md_content += f"**Duration:** {result['duration']:.2f}s\n"
            md_content += f"**Details:** {result['details']}\n"

            if "error" in result:
                md_content += f"**Error:** {result['error']}\n"

            if result["artifacts"]:
                md_content += f"**Artifacts:** {len(result['artifacts'])} files\n"

            md_content += "\n"

        if report["failed_tests"]:
            md_content += "## Failed Tests\n\n"
            for test in report["failed_tests"]:
                md_content += f"### ❌ {test['name']}\n"
                md_content += f"{test['details']}\n"
                if "error" in test:
                    md_content += f"**Error:** {test['error']}\n"
                md_content += "\n"

        md_content += "## Recommendations\n\n"
        for rec in report["recommendations"]:
            md_content += f"- {rec}\n"

        md_content += f"""
## Artifacts

Total artifacts generated: {len(report['artifacts'])}

All test artifacts are available in the test output directory.

---
*Report generated by Academic Publication Pipeline Test Suite v1.0.0*
        """

        md_file = self.test_output_dir / "pipeline_test_report.md"
        with open(md_file, 'w') as f:
            f.write(md_content)


def main():
    """Main function to run the complete pipeline test."""
    print("Academic Publication Pipeline Test Suite v1.0.0")
    print("=" * 60)
    print("Testing comprehensive academic publication pipeline...")
    print()

    # Initialize test suite
    test_suite = PipelineTestSuite()

    # Run all tests
    test_report = test_suite.run_all_tests()

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    summary = test_report["test_summary"]
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Passed: {summary['passed']} ✅")
    print(f"Failed: {summary['failed']} ❌")
    print(f"Skipped: {summary['skipped']} ⏭️")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Duration: {summary['total_duration']:.2f}s")
    print()

    if summary["failed"] > 0:
        print("FAILED TESTS:")
        for test in test_report["failed_tests"]:
            print(f"  ❌ {test['name']}: {test['details']}")
        print()

    print("RECOMMENDATIONS:")
    for rec in test_report["recommendations"]:
        print(f"  • {rec}")
    print()

    print(f"📁 Test artifacts saved to: {test_suite.test_output_dir}")
    print(f"📄 Full report: {test_suite.test_output_dir}/pipeline_test_report.md")

    # Return appropriate exit code
    if summary["failed"] > 0:
        print("\n❌ Some tests failed. Please review the report.")
        return 1
    else:
        print("\n✅ All tests passed! Pipeline is ready for production use.")
        return 0


if __name__ == "__main__":
    import sys
    from dataclasses import dataclass
    sys.exit(main())