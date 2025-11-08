#!/usr/bin/env python3
"""
Citation and Reproducibility Validation System for Academic Publications
=======================================================================

This module provides comprehensive validation systems for ensuring proper citation
practices and complete reproducibility of academic research outputs.

Key Features:
- Automated citation validation and completeness checking
- Reference formatting verification for multiple journal styles
- Reproducibility pipeline validation with dependency tracking
- Computational environment specification and verification
- Data and code availability validation
- Version control and provenance tracking
- FAIR principles compliance checking

Author: Academic Standards Task Force
Version: 1.0.0 (Reproducibility-Focused)
"""

import os
import json
import yaml
import hashlib
import subprocess
import importlib.util
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import re
import requests
import zipfile
import tempfile
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CitationEntry:
    """Container for citation information."""
    citation_key: str
    authors: List[str]
    title: str
    journal: str
    volume: str
    pages: str
    year: int
    doi: Optional[str]
    arxiv_id: Optional[str]
    citation_count: Optional[int]
    relevance_score: float
    validation_status: str  # Valid, Missing_Info, Format_Error, Not_Found

@dataclass
class ReproducibilityCheck:
    """Container for reproducibility validation results."""
    check_name: str
    status: str  # Pass, Fail, Warning
    description: str
    details: str
    critical: bool
    automated_fix_available: bool
    estimated_effort: str  # Low, Medium, High

@dataclass
class ComputationalEnvironment:
    """Container for computational environment specification."""
    python_version: str
    package_versions: Dict[str, str]
    system_info: Dict[str, str]
    hardware_info: Dict[str, str]
    environment_hash: str
    creation_date: datetime

@dataclass
class ReproducibilityReport:
    """Container for complete reproducibility validation report."""
    manuscript_id: str
    validation_date: datetime
    overall_score: float
    citation_validity_score: float
    reproducibility_score: float
    citation_entries: List[CitationEntry]
    reproducibility_checks: List[ReproducibilityCheck]
    computational_environment: ComputationalEnvironment
    recommendations: List[str]
    critical_issues: List[str]

class CitationReproducibilityValidator:
    """
    Comprehensive validation system for citations and reproducibility.
    """

    def __init__(self, project_root: str = ".", output_dir: str = "validation_reports"):
        """Initialize the validator."""
        self.project_root = Path(project_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Journal citation styles
        self.citation_styles = {
            "aps": self._load_aps_style(),
            "nature": self._load_nature_style(),
            "elsevier": self._load_elsevier_style(),
            "springer": self._load_springer_style()
        }

        # Critical reference list for analog Hawking radiation research
        self.critical_references = self._load_critical_references()

        # Reproducibility check definitions
        self.reproducibility_checks = self._define_reproducibility_checks()

        logger.info(f"Citation and reproducibility validator initialized")

    def _load_aps_style(self) -> Dict[str, Any]:
        """Load APS citation style specifications."""
        return {
            "author_format": "First Initial. Last Name",
            "title_case": "sentence_case",
            "journal_abbrev": True,
            "volume_bold": True,
            "page_range_format": "start–end",
            "doi_required": False,
            "year_parentheses": True,
            "example": "A. Author, B. Coauthor, Title of paper, Phys. Rev. Lett. **123**, 012345 (2023)."
        }

    def _load_nature_style(self) -> Dict[str, Any]:
        """Load Nature citation style specifications."""
        return {
            "author_format": "First Initial Last Name",
            "title_case": "sentence_case",
            "journal_abbrev": False,
            "volume_bold": False,
            "page_range_format": "start–end",
            "doi_required": True,
            "year_parentheses": True,
            "example": "A. A. Author, B. B. Coauthor. Title of paper. Nature 123, 456–789 (2023)."
        }

    def _load_elsevier_style(self) -> Dict[str, Any]:
        """Load Elsevier citation style specifications."""
        return {
            "author_format": "Last Name, A. A.",
            "title_case": "title_case",
            "journal_abbrev": True,
            "volume_bold": False,
            "page_range_format": "start–end",
            "doi_required": True,
            "year_parentheses": False,
            "example": "Author, A. A., Coauthor, B. B., Title of Paper, Journal Name 123 (2023) 456–789."
        }

    def _load_springer_style(self) -> Dict[str, Any]:
        """Load Springer citation style specifications."""
        return {
            "author_format": "Last Name, F.",
            "title_case": "sentence_case",
            "journal_abbrev": True,
            "volume_bold": True,
            "page_range_format": "start–end",
            "doi_required": True,
            "year_parentheses": True,
            "example": "Lastname, F., Firstname, I.: Title of paper, Journal Name 123, 456–789 (2023)."
        }

    def _load_critical_references(self) -> List[Dict[str, Any]]:
        """Load list of critical references for the field."""

        return [
            {
                "title": "Experimental black-hole evaporation?",
                "authors": ["Unruh, W.G."],
                "journal": "Phys. Rev. Lett.",
                "year": 1981,
                "volume": "46",
                "pages": "1351",
                "doi": "10.1103/PhysRevLett.46.1351",
                "critical": True,
                "category": "foundational"
            },
            {
                "title": "Accelerating plasma mirrors to investigate the black hole information loss paradox",
                "authors": ["Chen, P.", "Mourou, G."],
                "journal": "Phys. Rev. Lett.",
                "year": 2017,
                "volume": "118",
                "pages": "045001",
                "doi": "10.1103/PhysRevLett.118.045001",
                "critical": True,
                "category": "anabhel_concept"
            },
            {
                "title": "AnaBHEL (Analog Black Hole Evaporation via Lasers) Experiment: Concept, Design, and Status",
                "authors": ["Chen, P.", "Mourou, G.", "Besancon, M.", "Fukuda, Y.", "Glicenstein, J.-F.", "Kawata, S.", "Kondo, K.", "Li, X.", "Matsukawa, M.", "Mima, K.", "Miura, E.", "Sakawa, Y.", "Sato, F.", "Shiraga, H.", "Sugiyama, T.", "Tampo, M.", "Yabuuchi, T.", "Yogo, A.", "Zhong, J."],
                "journal": "Photonics",
                "year": 2022,
                "volume": "9",
                "pages": "1003",
                "doi": "10.3390/photonics9121003",
                "critical": True,
                "category": "anabhel_collaboration"
            },
            {
                "title": "Observation of quantum Hawking radiation and its entanglement in an analogue black hole",
                "authors": ["Steinhauer, J."],
                "journal": "Nature",
                "year": 2016,
                "volume": "534",
                "pages": "204–207",
                "doi": "10.1038/nature18238",
                "critical": True,
                "category": "experimental_validation"
            },
            {
                "title": "Black-hole lasers in Bose-Einstein condensates",
                "authors": ["Finazzi, S.", "Carusotto, I."],
                "journal": "Phys. Rev. A",
                "year": 2010,
                "volume": "81",
                "pages": "033603",
                "doi": "10.1103/PhysRevA.81.033603",
                "critical": False,
                "category": "theoretical_background"
            }
        ]

    def _define_reproducibility_checks(self) -> List[Dict[str, Any]]:
        """Define comprehensive reproducibility checks."""

        return [
            {
                "name": "code_availability",
                "description": "Check if source code is publicly available",
                "critical": True,
                "validation_func": "check_code_availability"
            },
            {
                "name": "data_availability",
                "description": "Check if research data is accessible",
                "critical": True,
                "validation_func": "check_data_availability"
            },
            {
                "name": "dependency_specification",
                "description": "Verify complete dependency specification",
                "critical": True,
                "validation_func": "check_dependency_specification"
            },
            {
                "name": "computational_environment",
                "description": "Validate computational environment documentation",
                "critical": True,
                "validation_func": "check_computational_environment"
            },
            {
                "name": "version_control",
                "description": "Check proper version control practices",
                "critical": False,
                "validation_func": "check_version_control"
            },
            {
                "name": "test_coverage",
                "description": "Verify adequate test coverage",
                "critical": False,
                "validation_func": "check_test_coverage"
            },
            {
                "name": "documentation_completeness",
                "description": "Assess documentation completeness",
                "critical": False,
                "validation_func": "check_documentation_completeness"
            },
            {
                "name": "reproducibility_instructions",
                "description": "Check for clear reproduction instructions",
                "critical": True,
                "validation_func": "check_reproducibility_instructions"
            }
        ]

    def validate_citations(self, manuscript_content: Dict, journal_style: str = "aps") -> Tuple[float, List[CitationEntry]]:
        """Validate manuscript citations against journal standards."""

        logger.info(f"Validating citations for {journal_style} style")

        # Extract citations from manuscript
        extracted_citations = self._extract_citations(manuscript_content)

        # Validate each citation
        validated_citations = []
        for citation in extracted_citations:
            validated = self._validate_single_citation(citation, journal_style)
            validated_citations.append(validated)

        # Check critical references
        critical_check = self._check_critical_references(validated_citations)

        # Calculate overall citation validity score
        validity_score = self._calculate_citation_validity_score(validated_citations, critical_check)

        logger.info(f"Citation validation completed with score: {validity_score:.2f}")

        return validity_score, validated_citations

    def _extract_citations(self, manuscript_content: Dict) -> List[Dict[str, Any]]:
        """Extract citations from manuscript content."""

        citations = []

        # Look for references section
        if "references" in manuscript_content:
            for ref in manuscript_content["references"]:
                citations.append(ref)

        # Extract from text (simple regex-based approach)
        text_content = ""
        for section in ["introduction", "methods", "results", "discussion"]:
            if section in manuscript_content:
                text_content += manuscript_content[section] + " "

        # Find citation patterns like [1], [2-4], etc.
        citation_patterns = re.findall(r'\[(\d+(?:-\d+)*)\]', text_content)
        unique_numbers = set()
        for pattern in citation_patterns:
            if '-' in pattern:
                start, end = map(int, pattern.split('-'))
                unique_numbers.update(range(start, end + 1))
            else:
                unique_numbers.add(int(pattern))

        logger.info(f"Extracted {len(unique_numbers)} citation references from text")

        return citations

    def _validate_single_citation(self, citation: Dict, style: str) -> CitationEntry:
        """Validate a single citation entry."""

        style_specs = self.citation_styles.get(style, self.citation_styles["aps"])

        # Extract basic information
        citation_key = citation.get("key", f"ref_{hash(str(citation)) % 10000}")
        authors = citation.get("authors", [])
        title = citation.get("title", "")
        journal = citation.get("journal", "")
        volume = citation.get("volume", "")
        pages = citation.get("pages", "")
        year = citation.get("year", 0)
        doi = citation.get("doi")
        arxiv_id = citation.get("arxiv_id")

        # Validate required fields
        validation_status = "Valid"
        missing_fields = []

        if not authors:
            missing_fields.append("authors")
            validation_status = "Missing_Info"
        if not title:
            missing_fields.append("title")
            validation_status = "Missing_Info"
        if not journal:
            missing_fields.append("journal")
            validation_status = "Missing_Info"
        if year == 0:
            missing_fields.append("year")
            validation_status = "Missing_Info"

        # Check DOI format if present
        if doi and not self._validate_doi_format(doi):
            validation_status = "Format_Error"

        # Try to fetch citation count (if DOI available)
        citation_count = None
        if doi:
            citation_count = self._fetch_citation_count(doi)

        # Calculate relevance score (simplified)
        relevance_score = self._calculate_relevance_score(citation)

        return CitationEntry(
            citation_key=citation_key,
            authors=authors,
            title=title,
            journal=journal,
            volume=volume,
            pages=pages,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            citation_count=citation_count,
            relevance_score=relevance_score,
            validation_status=validation_status
        )

    def _validate_doi_format(self, doi: str) -> bool:
        """Validate DOI format."""
        doi_pattern = r'^10\.\d+/.+$'
        return re.match(doi_pattern, doi) is not None

    def _fetch_citation_count(self, doi: str) -> Optional[int]:
        """Try to fetch citation count for a DOI."""
        try:
            # Use Crossref API (simplified - would need API key for production)
            url = f"https://api.crossref.org/works/{doi}"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                data = response.json()
                # Citation count might be in different places in the response
                # This is a simplified version
                return data.get("message", {}).get("is-referenced-by-count", 0)
        except Exception as e:
            logger.warning(f"Failed to fetch citation count for DOI {doi}: {e}")

        return None

    def _calculate_relevance_score(self, citation: Dict) -> float:
        """Calculate relevance score for a citation."""
        score = 0.5  # Base score

        # Boost for recent publications
        year = citation.get("year", 0)
        if year > 2020:
            score += 0.2
        elif year > 2010:
            score += 0.1

        # Boost for high-impact journals
        journal = citation.get("journal", "").lower()
        high_impact_journals = ["nature", "science", "physical review letters", "nature physics"]
        if any(hj in journal for hj in high_impact_journals):
            score += 0.2

        # Boost for relevant keywords
        title = citation.get("title", "").lower()
        relevant_keywords = ["hawking", "analog", "black hole", "plasma", "laser", "radiation"]
        keyword_count = sum(1 for kw in relevant_keywords if kw in title)
        score += keyword_count * 0.05

        return min(1.0, score)

    def _check_critical_references(self, citations: List[CitationEntry]) -> Dict[str, bool]:
        """Check if critical references are cited."""
        critical_found = {}

        for critical_ref in self.critical_references:
            if critical_ref["critical"]:
                found = False
                for citation in citations:
                    if self._is_matching_reference(citation, critical_ref):
                        found = True
                        break
                critical_found[critical_ref["title"]] = found

        return critical_found

    def _is_matching_reference(self, citation: CitationEntry, critical_ref: Dict) -> bool:
        """Check if a citation matches a critical reference."""
        # Simple matching based on authors and year
        citation_authors = [author.split()[-1].lower() for author in citation.authors[:3]]  # Last 3 authors
        critical_authors = [author.split()[-1].lower() for author in critical_ref["authors"][:3]]

        authors_match = len(set(citation_authors) & set(critical_authors)) >= 2
        year_match = citation.year == critical_ref["year"]

        return authors_match and year_match

    def _calculate_citation_validity_score(self, citations: List[CitationEntry], critical_check: Dict[str, bool]) -> float:
        """Calculate overall citation validity score."""

        if not citations:
            return 0.0

        # Score based on validation status
        valid_count = sum(1 for c in citations if c.validation_status == "Valid")
        base_score = valid_count / len(citations)

        # Penalty for missing critical references
        missing_critical = sum(1 for found in critical_check.values() if not found)
        critical_penalty = missing_critical * 0.1

        # Bonus for good DOI coverage
        doi_coverage = sum(1 for c in citations if c.doi) / len(citations)
        doi_bonus = doi_coverage * 0.1

        final_score = max(0, base_score - critical_penalty + doi_bonus)
        return min(1.0, final_score)

    def validate_reproducibility(self, project_path: str = None) -> Tuple[float, List[ReproducibilityCheck]]:
        """Validate reproducibility of the research project."""

        if project_path is None:
            project_path = self.project_root

        logger.info(f"Validating reproducibility for {project_path}")

        project_path = Path(project_path)
        reproducibility_checks = []

        # Run all reproducibility checks
        for check_def in self.reproducibility_checks:
            check_func = getattr(self, check_def["validation_func"])
            try:
                check_result = check_func(project_path)
                reproducibility_checks.append(check_result)
            except Exception as e:
                logger.error(f"Error in reproducibility check {check_def['name']}: {e}")
                reproducibility_checks.append(ReproducibilityCheck(
                    check_name=check_def["name"],
                    status="Fail",
                    description=f"Error during check: {str(e)}",
                    details="Check failed due to unexpected error",
                    critical=check_def["critical"],
                    automated_fix_available=False,
                    estimated_effort="High"
                ))

        # Calculate overall reproducibility score
        overall_score = self._calculate_reproducibility_score(reproducibility_checks)

        logger.info(f"Reproducibility validation completed with score: {overall_score:.2f}")

        return overall_score, reproducibility_checks

    def check_code_availability(self, project_path: Path) -> ReproducibilityCheck:
        """Check if source code is publicly available."""

        # Check for common code repository indicators
        repo_indicators = [
            ".git",
            "README.md",
            "requirements.txt",
            "setup.py",
            "pyproject.toml"
        ]

        found_indicators = [indicator for indicator in repo_indicators
                          if (project_path / indicator).exists()]

        if len(found_indicators) >= 3:
            status = "Pass"
            details = f"Found {len(found_indicators)} repository indicators: {', '.join(found_indicators)}"
        else:
            status = "Fail"
            details = f"Insufficient repository indicators. Found: {', '.join(found_indicators)}"

        return ReproducibilityCheck(
            check_name="code_availability",
            status=status,
            details=details,
            description="Source code repository structure",
            critical=True,
            automated_fix_available=False,
            estimated_effort="Medium" if status == "Fail" else "Low"
        )

    def check_data_availability(self, project_path: Path) -> ReproducibilityCheck:
        """Check if research data is accessible."""

        # Look for data directories and files
        data_indicators = [
            "data/",
            "results/",
            "datasets/",
            "examples/data/"
        ]

        found_data = []
        for indicator in data_indicators:
            if (project_path / indicator).exists():
                found_data.append(indicator)

        # Check for data availability statements
        readme_path = project_path / "README.md"
        data_statement_found = False
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read().lower()
                if any(term in content for term in ["data available", "dataset", "results"]):
                    data_statement_found = True

        if found_data or data_statement_found:
            status = "Pass"
            details = f"Data indicators found: {found_data}, Data statement: {data_statement_found}"
        else:
            status = "Fail"
            details = "No data directories or availability statements found"

        return ReproducibilityCheck(
            check_name="data_availability",
            status=status,
            details=details,
            description="Research data accessibility",
            critical=True,
            automated_fix_available=False,
            estimated_effort="High"
        )

    def check_dependency_specification(self, project_path: Path) -> ReproducibilityCheck:
        """Verify complete dependency specification."""

        dependency_files = [
            "requirements.txt",
            "environment.yml",
            "pyproject.toml",
            "setup.py",
            "Pipfile"
        ]

        found_files = [f for f in dependency_files if (project_path / f).exists()]

        if found_files:
            # Check content quality
            status = "Pass"
            details = f"Found dependency files: {', '.join(found_files)}"

            # Analyze one of the files
            if (project_path / "requirements.txt").exists():
                with open(project_path / "requirements.txt", 'r') as f:
                    lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    if len(lines) < 5:
                        status = "Warning"
                        details += f" (Only {len(lines)} dependencies specified)"
        else:
            status = "Fail"
            details = "No dependency specification files found"

        return ReproducibilityCheck(
            check_name="dependency_specification",
            status=status,
            details=details,
            description="Dependency specification completeness",
            critical=True,
            automated_fix_available=True,
            estimated_effort="Low"
        )

    def check_computational_environment(self, project_path: Path) -> ReproducibilityCheck:
        """Validate computational environment documentation."""

        # Look for environment documentation
        env_files = [
            "environment.yml",
            "Dockerfile",
            "docker-compose.yml",
            ".dockerignore",
            "environment.yaml"
        ]

        found_env_files = [f for f in env_files if (project_path / f).exists()]

        # Check for Python version specification
        python_spec = False
        pyproject_path = project_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import tomllib
                with open(pyproject_path, 'rb') as f:
                    data = tomllib.load(f)
                    if "project" in data and "requires-python" in data["project"]:
                        python_spec = True
            except ImportError:
                # Fallback to manual parsing
                pass

        if found_env_files or python_spec:
            status = "Pass"
            details = f"Environment files: {found_env_files}, Python spec: {python_spec}"
        else:
            status = "Fail"
            details = "No computational environment documentation found"

        return ReproducibilityCheck(
            check_name="computational_environment",
            status=status,
            details=details,
            description="Computational environment documentation",
            critical=True,
            automated_fix_available=True,
            estimated_effort="Medium"
        )

    def check_version_control(self, project_path: Path) -> ReproducibilityCheck:
        """Check proper version control practices."""

        git_dir = project_path / ".git"
        if git_dir.exists():
            # Check for .gitignore
            gitignore_path = project_path / ".gitignore"
            has_gitignore = gitignore_path.exists()

            # Check for recent commits (simplified)
            try:
                result = subprocess.run(
                    ["git", "log", "--oneline", "-5"],
                    cwd=project_path,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                if result.returncode == 0:
                    recent_commits = len(result.stdout.strip().split('\n'))
                    status = "Pass"
                    details = f"Git repository with {recent_commits} recent commits, .gitignore: {has_gitignore}"
                else:
                    status = "Warning"
                    details = "Git repository exists but couldn't access log"
            except (subprocess.TimeoutExpired, FileNotFoundError):
                status = "Warning"
                details = "Git repository exists but git command not available"
        else:
            status = "Fail"
            details = "No version control repository found"

        return ReproducibilityCheck(
            check_name="version_control",
            status=status,
            details=details,
            description="Version control practices",
            critical=False,
            automated_fix_available=True,
            estimated_effort="Low"
        )

    def check_test_coverage(self, project_path: Path) -> ReproducibilityCheck:
        """Verify adequate test coverage."""

        # Look for test directories and files
        test_indicators = [
            "tests/",
            "test/",
            "pytest.ini",
            "tox.ini"
        ]

        test_dirs = [indicator for indicator in test_indicators if (project_path / indicator).exists()]

        # Count test files
        test_files = []
        for test_dir in test_indicators[:2]:  # Check actual directories
            test_path = project_path / test_dir
            if test_path.exists() and test_path.is_dir():
                test_files.extend(list(test_path.glob("**/*test*.py")))
                test_files.extend(list(test_path.glob("**/test_*.py")))

        if test_files:
            coverage = len(test_files)
            if coverage >= 10:
                status = "Pass"
            elif coverage >= 5:
                status = "Warning"
            else:
                status = "Fail"
            details = f"Found {coverage} test files in {len(test_dirs)} test directories"
        else:
            status = "Fail"
            details = "No test files found"

        return ReproducibilityCheck(
            check_name="test_coverage",
            status=status,
            details=details,
            description="Test coverage assessment",
            critical=False,
            automated_fix_available=False,
            estimated_effort="High"
        )

    def check_documentation_completeness(self, project_path: Path) -> ReproducibilityCheck:
        """Assess documentation completeness."""

        doc_files = [
            "README.md",
            "README.rst",
            "docs/",
            "doc/",
            "CHANGELOG.md",
            "CONTRIBUTING.md"
        ]

        found_docs = [doc for doc in doc_files if (project_path / doc).exists()]

        # Check README content quality
        readme_quality = "Low"
        readme_path = project_path / "README.md"
        if readme_path.exists():
            with open(readme_path, 'r') as f:
                content = f.read()
                word_count = len(content.split())
                if word_count > 500:
                    readme_quality = "High"
                elif word_count > 200:
                    readme_quality = "Medium"

        if len(found_docs) >= 2 and readme_quality in ["High", "Medium"]:
            status = "Pass"
        elif len(found_docs) >= 1:
            status = "Warning"
        else:
            status = "Fail"

        details = f"Documentation files: {found_docs}, README quality: {readme_quality}"

        return ReproducibilityCheck(
            check_name="documentation_completeness",
            status=status,
            details=details,
            description="Documentation completeness",
            critical=False,
            automated_fix_available=False,
            estimated_effort="Medium"
        )

    def check_reproducibility_instructions(self, project_path: Path) -> ReproducibilityCheck:
        """Check for clear reproduction instructions."""

        # Look for reproduction instructions
        instruction_files = [
            "README.md",
            "REPRODUCTION.md",
            "USAGE.md",
            "INSTALL.md"
        ]

        found_instructions = False
        instruction_quality = "None"

        for file_name in instruction_files:
            file_path = project_path / file_name
            if file_path.exists():
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    instruction_keywords = [
                        "how to", "installation", "setup", "reproduce", "run",
                        "usage", "getting started", "quick start"
                    ]
                    keyword_count = sum(1 for kw in instruction_keywords if kw in content)

                    if keyword_count >= 3:
                        found_instructions = True
                        instruction_quality = "Detailed"
                        break
                    elif keyword_count >= 1:
                        found_instructions = True
                        instruction_quality = "Basic"

        if found_instructions and instruction_quality == "Detailed":
            status = "Pass"
        elif found_instructions:
            status = "Warning"
        else:
            status = "Fail"

        details = f"Reproduction instructions: {instruction_quality}"

        return ReproducibilityCheck(
            check_name="reproducibility_instructions",
            status=status,
            details=details,
            description="Reproduction instructions availability",
            critical=True,
            automated_fix_available=False,
            estimated_effort="High"
        )

    def _calculate_reproducibility_score(self, checks: List[ReproducibilityCheck]) -> float:
        """Calculate overall reproducibility score."""

        if not checks:
            return 0.0

        # Weight critical checks more heavily
        total_weight = 0
        weighted_score = 0

        for check in checks:
            weight = 2.0 if check.critical else 1.0
            total_weight += weight

            if check.status == "Pass":
                score = 1.0
            elif check.status == "Warning":
                score = 0.7
            else:  # Fail
                score = 0.3

            weighted_score += score * weight

        return weighted_score / total_weight if total_weight > 0 else 0.0

    def generate_computational_environment_spec(self, project_path: Path = None) -> ComputationalEnvironment:
        """Generate comprehensive computational environment specification."""

        if project_path is None:
            project_path = self.project_root

        logger.info("Generating computational environment specification")

        # Python version
        python_version = f"{importlib.sys.version_info.major}.{importlib.sys.version_info.minor}.{importlib.sys.version_info.micro}"

        # Package versions
        package_versions = {}
        try:
            import pkg_resources
            for package in pkg_resources.working_set:
                package_versions[package.key] = package.version
        except ImportError:
            # Fallback method
            import subprocess
            try:
                result = subprocess.run(
                    ["pip", "list", "--format=freeze"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if '==' in line:
                            package, version = line.split('==')
                            package_versions[package.lower()] = version
            except (subprocess.TimeoutExpired, FileNotFoundError):
                logger.warning("Could not retrieve package versions")

        # System information
        import platform
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor()
        }

        # Hardware information
        hardware_info = {}
        try:
            import psutil
            hardware_info = {
                "cpu_count": psutil.cpu_count(),
                "memory_total": psutil.virtual_memory().total,
                "memory_available": psutil.virtual_memory().available
            }
        except ImportError:
            logger.warning("psutil not available for hardware information")

        # Generate environment hash
        env_data = f"{python_version}_{str(sorted(package_versions.items()))}_{platform.platform()}"
        environment_hash = hashlib.sha256(env_data.encode()).hexdigest()[:16]

        return ComputationalEnvironment(
            python_version=python_version,
            package_versions=package_versions,
            system_info=system_info,
            hardware_info=hardware_info,
            environment_hash=environment_hash,
            creation_date=datetime.now(timezone.utc)
        )

    def create_reproducibility_package(self, project_path: Path = None, output_path: str = None) -> str:
        """Create a complete reproducibility package."""

        if project_path is None:
            project_path = self.project_root

        if output_path is None:
            output_path = self.output_dir / f"reproducibility_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"

        logger.info(f"Creating reproducibility package: {output_path}")

        with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add source code
            for py_file in project_path.rglob("*.py"):
                if ".git" not in str(py_file) and "__pycache__" not in str(py_file):
                    zipf.write(py_file, py_file.relative_to(project_path))

            # Add configuration files
            config_files = ["requirements.txt", "pyproject.toml", "setup.py", "environment.yml"]
            for config_file in config_files:
                config_path = project_path / config_file
                if config_path.exists():
                    zipf.write(config_path, config_path.relative_to(project_path))

            # Add documentation
            doc_files = ["README.md", "CHANGELOG.md", "CONTRIBUTING.md"]
            for doc_file in doc_files:
                doc_path = project_path / doc_file
                if doc_path.exists():
                    zipf.write(doc_path, doc_path.relative_to(project_path))

            # Add test files
            tests_dir = project_path / "tests"
            if tests_dir.exists():
                for test_file in tests_dir.rglob("*"):
                    if not test_file.is_dir():
                        zipf.write(test_file, test_file.relative_to(project_path))

            # Add sample data (if reasonable size)
            data_dir = project_path / "data" / "sample"
            if data_dir.exists():
                for data_file in data_dir.rglob("*"):
                    if data_file.stat().st_size < 1024 * 1024:  # Less than 1MB
                        zipf.write(data_file, data_file.relative_to(project_path))

            # Add reproducibility specification
            env_spec = self.generate_computational_environment_spec(project_path)
            env_spec_file = project_path / "computational_environment.json"
            with open(env_spec_file, 'w') as f:
                json.dump(asdict(env_spec), f, indent=2, default=str)
            zipf.write(env_spec_file, env_spec_file.relative_to(project_path))

            # Add reproduction script
            reproduction_script = self._generate_reproduction_script()
            script_file = project_path / "reproduce.py"
            with open(script_file, 'w') as f:
                f.write(reproduction_script)
            zipf.write(script_file, script_file.relative_to(project_path))

        logger.info(f"Reproducibility package created: {output_path}")
        return str(output_path)

    def _generate_reproduction_script(self) -> str:
        """Generate a reproduction script."""

        return '''#!/usr/bin/env python3
"""
Reproduction Script for Analog Hawking Radiation Analysis
========================================================

This script reproduces the main results from the paper.
Run with: python reproduce.py
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up."""
    print("Checking environment...")

    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")

    # Check key packages
    required_packages = ["numpy", "matplotlib", "scipy"]
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} available")
        except ImportError:
            print(f"✗ {package} missing - install with: pip install {package}")
            return False

    return True

def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")

    if Path("requirements.txt").exists():
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    else:
        print("No requirements.txt found")

def run_analysis():
    """Run the main analysis."""
    print("Running analysis...")

    # Example analysis commands
    try:
        # Run basic pipeline
        subprocess.run([sys.executable, "scripts/run_full_pipeline.py", "--demo"], check=True)
        print("✓ Basic analysis completed")

        # Run validation
        subprocess.run([sys.executable, "-m", "pytest", "tests/"], check=True)
        print("✓ Tests passed")

    except subprocess.CalledProcessError as e:
        print(f"✗ Analysis failed: {e}")
        return False

    return True

def main():
    """Main reproduction function."""
    print("Analog Hawking Radiation Analysis - Reproduction Script")
    print("=" * 60)

    # Check environment
    if not check_environment():
        print("Environment check failed. Installing dependencies...")
        install_dependencies()

        # Re-check after installation
        if not check_environment():
            print("Failed to set up environment properly.")
            sys.exit(1)

    # Run analysis
    if run_analysis():
        print("\\n✓ Reproduction successful!")
        print("Check the 'results/' directory for output files.")
    else:
        print("\\n✗ Reproduction failed.")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    def generate_validation_report(self, manuscript_id: str, manuscript_content: Dict,
                                journal_style: str = "aps", project_path: str = None) -> ReproducibilityReport:
        """Generate comprehensive validation report."""

        logger.info(f"Generating validation report for manuscript {manuscript_id}")

        # Validate citations
        citation_score, citation_entries = self.validate_citations(manuscript_content, journal_style)

        # Validate reproducibility
        reproducibility_score, reproducibility_checks = self.validate_reproducibility(project_path)

        # Generate computational environment specification
        computational_env = self.generate_computational_environment_spec(project_path)

        # Calculate overall score
        overall_score = (citation_score * 0.4 + reproducibility_score * 0.6)

        # Generate recommendations
        recommendations = self._generate_recommendations(citation_entries, reproducibility_checks)

        # Identify critical issues
        critical_issues = []
        for check in reproducibility_checks:
            if check.critical and check.status == "Fail":
                critical_issues.append(f"{check.check_name}: {check.details}")

        for citation in citation_entries:
            if citation.validation_status in ["Missing_Info", "Format_Error"]:
                critical_issues.append(f"Citation {citation.citation_key}: {citation.validation_status}")

        report = ReproducibilityReport(
            manuscript_id=manuscript_id,
            validation_date=datetime.now(timezone.utc),
            overall_score=overall_score,
            citation_validity_score=citation_score,
            reproducibility_score=reproducibility_score,
            citation_entries=citation_entries,
            reproducibility_checks=reproducibility_checks,
            computational_environment=computational_env,
            recommendations=recommendations,
            critical_issues=critical_issues
        )

        # Save report
        report_file = self.output_dir / f"validation_report_{manuscript_id}.json"
        with open(report_file, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)

        logger.info(f"Validation report saved to {report_file}")

        return report

    def _generate_recommendations(self, citations: List[CitationEntry],
                                repro_checks: List[ReproducibilityCheck]) -> List[str]:
        """Generate improvement recommendations."""

        recommendations = []

        # Citation recommendations
        invalid_citations = [c for c in citations if c.validation_status != "Valid"]
        if invalid_citations:
            recommendations.append(f"Fix {len(invalid_citations)} invalid citations (missing DOIs, formatting issues, etc.)")

        citations_without_doi = [c for c in citations if not c.doi]
        if len(citations_without_doi) > len(citations) * 0.3:
            recommendations.append("Add DOIs to more citations to improve discoverability")

        # Reproducibility recommendations
        failed_checks = [c for c in repro_checks if c.status == "Fail"]
        for check in failed_checks:
            if check.automated_fix_available:
                recommendations.append(f"Fix {check.check_name}: {check.details}")
            else:
                recommendations.append(f"Address {check.check_name}: Manual intervention required")

        warning_checks = [c for c in repro_checks if c.status == "Warning"]
        if warning_checks:
            recommendations.append(f"Consider improving {len(warning_checks)} areas with warnings")

        return recommendations


def main():
    """Main function to demonstrate citation and reproducibility validation."""
    validator = CitationReproducibilityValidator()

    print("Citation and Reproducibility Validation System v1.0.0")
    print("=" * 60)
    print("Validating academic standards and reproducibility...")
    print()

    # Example manuscript content
    manuscript_content = {
        "title": "Comprehensive Computational Framework for Analog Hawking Radiation",
        "abstract": "We present a comprehensive framework...",
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
                "title": "Accelerating plasma mirrors to investigate the black hole information loss paradox",
                "journal": "Phys. Rev. Lett.",
                "volume": "118",
                "pages": "045001",
                "year": 2017,
                "doi": "10.1103/PhysRevLett.118.045001"
            }
        ]
    }

    # Generate validation report
    manuscript_id = f"manuscript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    report = validator.generate_validation_report(
        manuscript_id=manuscript_id,
        manuscript_content=manuscript_content,
        journal_style="aps",
        project_path="."
    )

    # Create reproducibility package
    package_path = validator.create_reproducibility_package()

    print(f"\n✅ Validation completed!")
    print(f"📊 Overall Score: {report.overall_score:.2f}")
    print(f"📄 Citation Score: {report.citation_validity_score:.2f}")
    print(f"🔄 Reproducibility Score: {report.reproducibility_score:.2f}")
    print(f"📁 Report saved in: {validator.output_dir}")
    print(f"📦 Reproducibility package: {package_path}")
    print()

    if report.critical_issues:
        print("⚠️  Critical Issues:")
        for issue in report.critical_issues:
            print(f"   • {issue}")
        print()

    if report.recommendations:
        print("💡 Recommendations:")
        for rec in report.recommendations:
            print(f"   • {rec}")

    return report


if __name__ == "__main__":
    main()