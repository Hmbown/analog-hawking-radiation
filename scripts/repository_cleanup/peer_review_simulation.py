#!/usr/bin/env python3
"""
Peer Review Simulation Framework for Academic Publications
==========================================================

This module provides a comprehensive peer review simulation system that generates
realistic reviewer comments, assessments, and recommendations for academic
manuscripts in the field of analog Hawking radiation research.

Key Features:
- Realistic reviewer persona generation with domain expertise
- Comprehensive manuscript evaluation with multiple criteria
- Context-aware comment generation based on content analysis
- Journal-specific review standards and expectations
- Detailed feedback with actionable revision suggestions
- Statistical analysis of review quality and consistency

Author: Peer Review Simulation Task Force
Version: 1.0.0 (Academic Review)
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import logging
import re
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ReviewerPersona:
    """Container for reviewer persona characteristics."""
    reviewer_id: str
    name: str
    institution: str
    expertise_areas: List[str]
    seniority_level: str  # Junior, Mid-career, Senior
    review_style: str  # Constructive, Critical, Balanced
    domain_focus: str  # Theoretical, Experimental, Computational
    typical_response_time: int  # days
    acceptance_rate: float  # 0-1
    strictness_level: float  # 0-1

@dataclass
class ReviewCriteria:
    """Container for review evaluation criteria."""
    scientific_rigor: float
    novelty_significance: float
    methodological_quality: float
    clarity_presentation: float
    reproducibility: float
    literature_context: float
    appropriateness_for_journal: float

@dataclass
class ReviewComment:
    """Container for individual review comments."""
    category: str  # Major, Minor, Technical, Suggestion
    severity: str  # Critical, Important, Minor, Suggestion
    text: str
    location_reference: Optional[str]  # e.g., "Section 2.3", "Figure 2"
    actionable: bool
    estimated_effort: str  # High, Medium, Low

@dataclass
class DetailedReview:
    """Container for complete peer review."""
    reviewer_id: str
    manuscript_id: str
    journal: str
    overall_recommendation: str  # Accept, Minor Revisions, Major Revisions, Reject
    confidence_level: str  # High, Medium, Low
    review_criteria: ReviewCriteria
    summary_comments: str
    major_comments: List[ReviewComment]
    minor_comments: List[ReviewComment]
    technical_comments: List[ReviewComment]
    suggestions: List[ReviewComment]
    strengths: List[str]
    weaknesses: List[str]
    review_date: datetime
    estimated_revision_time: Optional[int]  # days

class PeerReviewSimulator:
    """
    Comprehensive peer review simulation system for academic manuscripts.
    """

    def __init__(self, output_dir: str = "peer_review_simulations"):
        """Initialize the peer review simulator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Review criteria weights for different journals
        self.journal_weights = {
            "Nature Physics": {
                "novelty_significance": 0.25,
                "scientific_rigor": 0.20,
                "appropriateness_for_journal": 0.20,
                "clarity_presentation": 0.15,
                "methodological_quality": 0.10,
                "reproducibility": 0.05,
                "literature_context": 0.05
            },
            "Physical Review Letters": {
                "scientific_rigor": 0.25,
                "novelty_significance": 0.20,
                "clarity_presentation": 0.15,
                "methodological_quality": 0.15,
                "appropriateness_for_journal": 0.10,
                "reproducibility": 0.10,
                "literature_context": 0.05
            },
            "Physical Review E": {
                "methodological_quality": 0.25,
                "scientific_rigor": 0.20,
                "reproducibility": 0.20,
                "literature_context": 0.15,
                "clarity_presentation": 0.10,
                "novelty_significance": 0.05,
                "appropriateness_for_journal": 0.05
            }
        }

        # Initialize reviewer personas
        self.reviewer_personas = self._create_reviewer_personas()

        # Comment templates for different scenarios
        self.comment_templates = self._load_comment_templates()

        logger.info(f"Peer review simulator initialized with {len(self.reviewer_personas)} reviewer personas")

    def _create_reviewer_personas(self) -> List[ReviewerPersona]:
        """Create diverse reviewer personas for realistic simulation."""

        personas = [
            # Theoretical physics experts
            ReviewerPersona(
                reviewer_id="T001",
                name="Dr. Elena Rodriguez",
                institution="Institute for Advanced Study",
                expertise_areas=["quantum_field_theory", "curved_spacetime", "hawking_radiation"],
                seniority_level="Senior",
                review_style="Critical",
                domain_focus="Theoretical",
                typical_response_time=21,
                acceptance_rate=0.15,
                strictness_level=0.85
            ),
            ReviewerPersona(
                reviewer_id="T002",
                name="Prof. James Chen",
                institution="Stanford University",
                expertise_areas=["general_relativity", "analog_gravity", "black_hole_physics"],
                seniority_level="Senior",
                review_style="Balanced",
                domain_focus="Theoretical",
                typical_response_time=18,
                acceptance_rate=0.25,
                strictness_level=0.70
            ),

            # Experimental physics experts
            ReviewerPersona(
                reviewer_id="E001",
                name="Dr. Maria Gonzalez",
                institution="MIT",
                expertise_areas=["laser_plasma_physics", "high_intensity_lasers", "experimental_techniques"],
                seniority_level="Mid-career",
                review_style="Constructive",
                domain_focus="Experimental",
                typical_response_time=14,
                acceptance_rate=0.35,
                strictness_level=0.60
            ),
            ReviewerPersona(
                reviewer_id="E002",
                name="Prof. Thomas Weber",
                institution="Max Planck Institute",
                expertise_areas=["plasma_diagnostics", "radio_detection", "instrumentation"],
                seniority_level="Senior",
                review_style="Critical",
                domain_focus="Experimental",
                typical_response_time=20,
                acceptance_rate=0.20,
                strictness_level=0.80
            ),

            # Computational physics experts
            ReviewerPersona(
                reviewer_id="C001",
                name="Dr. Sarah Johnson",
                institution="University of Cambridge",
                expertise_areas=["computational_physics", "numerical_methods", "uncertainty_quantification"],
                seniority_level="Mid-career",
                review_style="Constructive",
                domain_focus="Computational",
                typical_response_time=16,
                acceptance_rate=0.40,
                strictness_level=0.55
            ),
            ReviewerPersona(
                reviewer_id="C002",
                name="Prof. Akira Tanaka",
                institution="University of Tokyo",
                expertise_areas=["simulation_methods", "plasma_modeling", "code_validation"],
                seniority_level="Senior",
                review_style="Balanced",
                domain_focus="Computational",
                typical_response_time=19,
                acceptance_rate=0.30,
                strictness_level=0.65
            )
        ]

        return personas

    def _load_comment_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load comment templates for different review scenarios."""

        return {
            "scientific_rigor": {
                "strengths": [
                    "The theoretical framework is mathematically rigorous and well-founded.",
                    "The mathematical derivations are clear and properly justified.",
                    "The physical assumptions are clearly stated and validated.",
                    "The analysis demonstrates deep understanding of the underlying physics.",
                    "The theoretical approach is sound and follows established methodologies."
                ],
                "weaknesses": [
                    "Some mathematical steps need clearer justification.",
                    "The theoretical framework could benefit from more rigorous derivation.",
                    "Certain physical assumptions require more detailed discussion.",
                    "The connection to fundamental principles could be strengthened.",
                    "Some approximations used need better justification and error analysis."
                ],
                "suggestions": [
                    "Provide more detailed mathematical derivations in the supplementary material.",
                    "Clarify the physical meaning of key approximations used.",
                    "Include discussion of alternative theoretical approaches.",
                    "Add validation against known analytical solutions where possible.",
                    "Expand discussion of regime of validity for theoretical framework."
                ]
            },
            "methodological_quality": {
                "strengths": [
                    "The computational methods are state-of-the-art and well-implemented.",
                    "The methodology is comprehensive and addresses key challenges.",
                    "The experimental design is thorough and well-justified.",
                    "Quality control procedures are robust and well-documented.",
                    "The approach demonstrates innovation in methodology."
                ],
                "weaknesses": [
                    "The methodology could benefit from more detailed description.",
                    "Some methodological choices need better justification.",
                    "The approach might miss important aspects of the problem.",
                    "Validation of methods against known cases is insufficient.",
                    "The computational approach could be more efficient."
                ],
                "suggestions": [
                    "Provide more detailed description of computational methods.",
                    "Include validation studies using benchmark problems.",
                    "Discuss potential limitations of the chosen methodology.",
                    "Consider alternative approaches for comparison.",
                    "Add sensitivity analysis for key methodological parameters."
                ]
            },
            "novelty_significance": {
                "strengths": [
                    "This work addresses a significant gap in the current literature.",
                    "The approach is novel and represents a clear advance in the field.",
                    "The findings have important implications for future research.",
                    "This work opens up new avenues for experimental investigation.",
                    "The contribution represents a significant step forward in the field."
                ],
                "weaknesses": [
                    "The novelty of the work could be more clearly emphasized.",
                    "The significance of the findings needs better contextualization.",
                    "The advance over existing work could be more substantial.",
                    "The broader impact of the work is not sufficiently explored.",
                    "The contribution might be incremental rather than transformative."
                ],
                "suggestions": [
                    "Better highlight the novel aspects of this work in the introduction.",
                    "Expand discussion of the broader implications of the findings.",
                    "Provide clearer comparison with existing approaches.",
                    "Emphasize how this work enables new types of studies.",
                    "Discuss potential applications and future directions enabled by this work."
                ]
            },
            "clarity_presentation": {
                "strengths": [
                    "The manuscript is well-written and easy to follow.",
                    "The figures are clear and effectively communicate key results.",
                    "The logical flow of the argument is excellent.",
                    "The presentation is professional and polished.",
                    "Key concepts are explained clearly and accessibly."
                ],
                "weaknesses": [
                    "Some sections could benefit from clearer organization.",
                    "The figures could be improved for better clarity.",
                    "The writing style is sometimes dense and hard to follow.",
                    "Key points could be emphasized more effectively.",
                    "Some technical terms need better definition."
                ],
                "suggestions": [
                    "Restructure the introduction for better flow.",
                    "Improve figure captions to be more self-explanatory.",
                    "Add definitions for technical terms on first use.",
                    "Use more subheadings to improve readability.",
                    "Consider adding a conceptual diagram for clarity."
                ]
            },
            "reproducibility": {
                "strengths": [
                    "The code and data are made available for reproducibility.",
                    "The methods are described in sufficient detail for replication.",
                    "Computational parameters are clearly specified.",
                    "The workflow is well-documented and systematic.",
                    "Uncertainty quantification is comprehensive."
                ],
                "weaknesses": [
                    "Insufficient information is provided for full reproducibility.",
                    "Key parameters and settings are not fully specified.",
                    "The code availability needs better documentation.",
                    "Data processing steps are not fully described.",
                    "Version information for software is missing."
                ],
                "suggestions": [
                    "Provide complete parameter files in supplementary material.",
                    "Include detailed software version information.",
                    "Add step-by-step reproduction instructions.",
                    "Make raw data available alongside processed results.",
                    "Consider creating a reproducibility package with all necessary files."
                ]
            }
        }

    def select_reviewers(self, manuscript_content: Dict, journal: str, num_reviewers: int = 3) -> List[ReviewerPersona]:
        """Select appropriate reviewers for a manuscript."""

        # Analyze manuscript content to determine expertise needed
        content_analysis = self._analyze_manuscript_content(manuscript_content)

        # Score reviewers based on expertise match
        reviewer_scores = []
        for reviewer in self.reviewer_personas:
            score = self._calculate_reviewer_match(reviewer, content_analysis, journal)
            reviewer_scores.append((reviewer, score))

        # Sort by score and select top reviewers
        reviewer_scores.sort(key=lambda x: x[1], reverse=True)
        selected_reviewers = [reviewer for reviewer, score in reviewer_scores[:num_reviewers]]

        logger.info(f"Selected {len(selected_reviewers)} reviewers for manuscript")
        return selected_reviewers

    def _analyze_manuscript_content(self, content: Dict) -> Dict[str, float]:
        """Analyze manuscript content to determine expertise areas needed."""

        analysis = {
            "theoretical_physics": 0.0,
            "experimental_physics": 0.0,
            "computational_physics": 0.0,
            "laser_plasma": 0.0,
            "quantum_field_theory": 0.0,
            "uncertainty_quantification": 0.0
        }

        # Simple keyword-based analysis (in real system, would use NLP)
        text_content = ""
        if "abstract" in content:
            text_content += content["abstract"] + " "
        if "introduction" in content:
            text_content += content["introduction"] + " "
        if "methods" in content:
            text_content += content["methods"] + " "

        text_content = text_content.lower()

        # Keywords for different areas
        keywords = {
            "theoretical_physics": ["theory", "theoretical", "derivation", "equation", "mathematical"],
            "experimental_physics": ["experiment", "measurement", "detection", "diagnostic", "facility"],
            "computational_physics": ["simulation", "computational", "numerical", "code", "algorithm"],
            "laser_plasma": ["laser", "plasma", "intensity", "beam", "target"],
            "quantum_field_theory": ["quantum", "field", "hawking", "radiation", "curved"],
            "uncertainty_quantification": ["uncertainty", "error", "confidence", "statistical", "validation"]
        }

        for area, words in keywords.items():
            count = sum(text_content.count(word) for word in words)
            analysis[area] = min(count / 10.0, 1.0)  # Normalize to [0, 1]

        return analysis

    def _calculate_reviewer_match(self, reviewer: ReviewerPersona, content_analysis: Dict, journal: str) -> float:
        """Calculate match score between reviewer and manuscript."""

        score = 0.0

        # Expertise matching
        if reviewer.domain_focus == "Theoretical":
            score += content_analysis["theoretical_physics"] * 0.4
            score += content_analysis["quantum_field_theory"] * 0.3
        elif reviewer.domain_focus == "Experimental":
            score += content_analysis["experimental_physics"] * 0.4
            score += content_analysis["laser_plasma"] * 0.3
        elif reviewer.domain_focus == "Computational":
            score += content_analysis["computational_physics"] * 0.4
            score += content_analysis["uncertainty_quantification"] * 0.3

        # Adjust for seniority and review style
        if reviewer.seniority_level == "Senior":
            score += 0.1
        elif reviewer.seniority_level == "Mid-career":
            score += 0.05

        # Add some randomness for realism
        score += random.uniform(-0.05, 0.05)

        return max(0, min(1, score))

    def generate_review(self, reviewer: ReviewerPersona, manuscript_content: Dict,
                       journal: str, manuscript_id: str) -> DetailedReview:
        """Generate a detailed peer review from a reviewer persona."""

        logger.info(f"Generating review from {reviewer.name} ({reviewer.reviewer_id})")

        # Assess manuscript quality based on reviewer characteristics
        quality_assessment = self._assess_manuscript_quality(reviewer, manuscript_content, journal)

        # Generate review criteria scores
        review_criteria = self._generate_review_criteria(reviewer, quality_assessment, journal)

        # Determine overall recommendation
        recommendation = self._determine_recommendation(review_criteria, reviewer.strictness_level)

        # Generate comments
        comments = self._generate_comments(reviewer, quality_assessment, manuscript_content)

        # Calculate estimated revision time
        revision_time = self._estimate_revision_time(comments, recommendation)

        review = DetailedReview(
            reviewer_id=reviewer.reviewer_id,
            manuscript_id=manuscript_id,
            journal=journal,
            overall_recommendation=recommendation,
            confidence_level=random.choice(["High", "Medium", "Low"]),
            review_criteria=review_criteria,
            summary_comments=comments["summary"],
            major_comments=comments["major"],
            minor_comments=comments["minor"],
            technical_comments=comments["technical"],
            suggestions=comments["suggestions"],
            strengths=comments["strengths"],
            weaknesses=comments["weaknesses"],
            review_date=datetime.now(timezone.utc),
            estimated_revision_time=revision_time
        )

        return review

    def _assess_manuscript_quality(self, reviewer: ReviewerPersona, content: Dict, journal: str) -> Dict[str, float]:
        """Assess manuscript quality from reviewer's perspective."""

        # Base quality assessment (would be more sophisticated in real system)
        base_quality = {
            "scientific_rigor": 0.75,
            "novelty_significance": 0.80,
            "methodological_quality": 0.70,
            "clarity_presentation": 0.75,
            "reproducibility": 0.65,
            "literature_context": 0.70,
            "appropriateness_for_journal": 0.75
        }

        # Adjust based on reviewer characteristics
        adjustment = (0.5 - reviewer.strictness_level) * 0.2  # Stricter reviewers give lower scores

        # Adjust based on domain expertise match
        content_analysis = self._analyze_manuscript_content(content)
        if reviewer.domain_focus == "Theoretical" and content_analysis["theoretical_physics"] > 0.5:
            adjustment += 0.1
        elif reviewer.domain_focus == "Experimental" and content_analysis["experimental_physics"] > 0.5:
            adjustment += 0.1
        elif reviewer.domain_focus == "Computational" and content_analysis["computational_physics"] > 0.5:
            adjustment += 0.1

        # Add some randomness for realism
        random_factor = random.uniform(-0.1, 0.1)

        # Apply adjustments
        for key in base_quality:
            base_quality[key] = max(0, min(1, base_quality[key] + adjustment + random_factor))

        return base_quality

    def _generate_review_criteria(self, reviewer: ReviewerPersona, quality_assessment: Dict, journal: str) -> ReviewCriteria:
        """Generate detailed review criteria scores."""

        weights = self.journal_weights.get(journal, self.journal_weights["Physical Review Letters"])

        # Apply reviewer's focus areas
        if reviewer.domain_focus == "Theoretical":
            quality_assessment["scientific_rigor"] += 0.1
            quality_assessment["literature_context"] += 0.05
        elif reviewer.domain_focus == "Experimental":
            quality_assessment["methodological_quality"] += 0.1
            quality_assessment["reproducibility"] += 0.05
        elif reviewer.domain_focus == "Computational":
            quality_assessment["methodological_quality"] += 0.05
            quality_assessment["reproducibility"] += 0.1

        # Normalize scores
        for key in quality_assessment:
            quality_assessment[key] = max(0, min(1, quality_assessment[key]))

        return ReviewCriteria(
            scientific_rigor=quality_assessment["scientific_rigor"],
            novelty_significance=quality_assessment["novelty_significance"],
            methodological_quality=quality_assessment["methodological_quality"],
            clarity_presentation=quality_assessment["clarity_presentation"],
            reproducibility=quality_assessment["reproducibility"],
            literature_context=quality_assessment["literature_context"],
            appropriateness_for_journal=quality_assessment["appropriateness_for_journal"]
        )

    def _determine_recommendation(self, criteria: ReviewCriteria, strictness: float) -> str:
        """Determine overall recommendation based on criteria scores."""

        # Calculate weighted average score
        scores = [
            criteria.scientific_rigor,
            criteria.novelty_significance,
            criteria.methodological_quality,
            criteria.clarity_presentation,
            criteria.reproducibility,
            criteria.literature_context,
            criteria.appropriateness_for_journal
        ]

        average_score = np.mean(scores)

        # Apply strictness adjustment
        adjusted_score = average_score - (strictness - 0.5) * 0.3

        # Determine recommendation
        if adjusted_score >= 0.85:
            return "Accept"
        elif adjusted_score >= 0.70:
            return "Minor Revisions"
        elif adjusted_score >= 0.50:
            return "Major Revisions"
        else:
            return "Reject"

    def _generate_comments(self, reviewer: ReviewerPersona, quality_assessment: Dict,
                         content: Dict) -> Dict[str, Any]:
        """Generate detailed review comments."""

        comments = {
            "summary": "",
            "major": [],
            "minor": [],
            "technical": [],
            "suggestions": [],
            "strengths": [],
            "weaknesses": []
        }

        # Generate summary comment
        comments["summary"] = self._generate_summary_comment(reviewer, quality_assessment)

        # Generate category-specific comments
        for category, assessment in quality_assessment.items():
            if category in self.comment_templates:
                category_comments = self.comment_templates[category]

                # Select strengths (high scores)
                if assessment > 0.7:
                    strength = random.choice(category_comments["strengths"])
                    comments["strengths"].append(strength)

                # Select weaknesses (low scores)
                if assessment < 0.6:
                    weakness = random.choice(category_comments["weaknesses"])
                    comments["weaknesses"].append(weakness)

                    # Create specific comment
                    comment = self._create_specific_comment(category, weakness, reviewer.review_style)
                    if assessment < 0.4:
                        comments["major"].append(comment)
                    else:
                        comments["minor"].append(comment)

                # Select suggestions (medium scores)
                if 0.5 < assessment < 0.8:
                    suggestion = random.choice(category_comments["suggestions"])
                    comment = self._create_specific_comment(category, suggestion, reviewer.review_style)
                    comments["suggestions"].append(comment)

        # Add technical comments based on domain
        technical_comments = self._generate_technical_comments(reviewer, content)
        comments["technical"] = technical_comments

        return comments

    def _generate_summary_comment(self, reviewer: ReviewerPersona, quality_assessment: Dict) -> str:
        """Generate overall summary comment."""

        overall_score = np.mean(list(quality_assessment.values()))

        if overall_score > 0.8:
            if reviewer.review_style == "Constructive":
                return "This is an excellent manuscript that makes a significant contribution to the field. The work is well-executed, thoroughly analyzed, and clearly presented. I recommend acceptance with only minor revisions."
            elif reviewer.review_style == "Critical":
                return "While this manuscript presents important findings, there are several aspects that require clarification and strengthening before publication. The overall approach is sound, but the implementation and presentation need improvement."
            else:
                return "This manuscript addresses an important topic and presents valuable results. The work is generally solid, though some aspects could be improved. I recommend minor revisions to strengthen the contribution."

        elif overall_score > 0.6:
            if reviewer.review_style == "Constructive":
                return "This manuscript has potential but requires substantial revision before it can be considered for publication. The core ideas are interesting, but the execution and presentation need significant improvement."
            elif reviewer.review_style == "Critical":
                return "This manuscript has serious flaws that must be addressed before it can be considered for publication. The theoretical framework, methodology, and presentation all require substantial revision."
            else:
                return "This manuscript presents potentially interesting work but requires major revisions. The authors should address the concerns raised in my detailed comments before resubmission."

        else:
            return "Unfortunately, this manuscript has fundamental flaws that make it unsuitable for publication in its current form. The work would require extensive revision and additional experiments/simulations before it could be reconsidered."

    def _create_specific_comment(self, category: str, template: str, review_style: str) -> ReviewComment:
        """Create a specific review comment."""

        # Customize comment based on category and review style
        location_references = {
            "scientific_rigor": ["Section 2", "Introduction", "Methods"],
            "methodological_quality": ["Section 3", "Methods", "Appendix"],
            "novelty_significance": ["Introduction", "Discussion", "Conclusions"],
            "clarity_presentation": ["Throughout manuscript", "Figures", "Tables"],
            "reproducibility": ["Methods", "Supplementary Material", "Code availability"]
        }

        location = random.choice(location_references.get(category, ["Methods"]))

        # Adjust severity and effort based on review style
        if review_style == "Critical":
            severity = "Important"
            effort = "Medium"
        elif review_style == "Constructive":
            severity = "Minor"
            effort = "Low"
        else:
            severity = random.choice(["Important", "Minor"])
            effort = random.choice(["Low", "Medium"])

        # Determine if major or minor issue
        if category in ["scientific_rigor", "methodological_quality"]:
            comment_category = "Major"
        else:
            comment_category = "Minor"

        return ReviewComment(
            category=comment_category,
            severity=severity,
            text=template,
            location_reference=location,
            actionable=True,
            estimated_effort=effort
        )

    def _generate_technical_comments(self, reviewer: ReviewerPersona, content: Dict) -> List[ReviewComment]:
        """Generate domain-specific technical comments."""

        technical_comments = []

        if reviewer.domain_focus == "Theoretical":
            technical_templates = [
                "The derivation of equation X could be more clearly explained.",
                "The physical interpretation of parameter Y needs clarification.",
                "Consider discussing the regime of validity for the theoretical framework.",
                "The connection to established theoretical results could be strengthened."
            ]
        elif reviewer.domain_focus == "Experimental":
            technical_templates = [
                "The experimental parameters should be justified in more detail.",
                "Consider discussing potential systematic uncertainties in the measurements.",
                "The diagnostic setup could benefit from more detailed description.",
                "Alternative experimental configurations should be considered."
            ]
        else:  # Computational
            technical_templates = [
                "The numerical convergence should be demonstrated more thoroughly.",
                "Consider providing benchmark calculations against known solutions.",
                "The computational parameters need better justification.",
                "The efficiency of the numerical method could be improved."
            ]

        # Select 1-3 technical comments
        num_comments = random.randint(1, 3)
        selected_templates = random.sample(technical_templates, min(num_comments, len(technical_templates)))

        for template in selected_templates:
            technical_comments.append(ReviewComment(
                category="Technical",
                severity="Minor",
                text=template,
                location_reference="Methods" if reviewer.domain_focus == "Computational" else random.choice(["Introduction", "Discussion"]),
                actionable=True,
                estimated_effort="Low"
            ))

        return technical_comments

    def _estimate_revision_time(self, comments: Dict[str, List[ReviewComment]], recommendation: str) -> int:
        """Estimate time required for revisions based on comments."""

        base_times = {
            "Accept": 7,
            "Minor Revisions": 21,
            "Major Revisions": 56,
            "Reject": 84
        }

        base_time = base_times.get(recommendation, 30)

        # Adjust based on number and type of comments
        major_count = len(comments.get("major", []))
        minor_count = len(comments.get("minor", []))
        technical_count = len(comments.get("technical", []))

        # Add time for each comment type
        additional_time = major_count * 7 + minor_count * 3 + technical_count * 2

        return base_time + additional_time

    def simulate_peer_review_process(self, manuscript_content: Dict, journal: str,
                                   manuscript_id: str) -> Dict[str, Any]:
        """Simulate complete peer review process for a manuscript."""

        logger.info(f"Starting peer review simulation for {journal}")

        # Select reviewers
        selected_reviewers = self.select_reviewers(manuscript_content, journal, num_reviewers=3)

        # Generate reviews from each reviewer
        reviews = []
        for reviewer in selected_reviewers:
            review = self.generate_review(reviewer, manuscript_content, journal, manuscript_id)
            reviews.append(review)

        # Compile editorial decision
        editorial_decision = self._make_editorial_decision(reviews, journal)

        # Generate review summary statistics
        review_statistics = self._calculate_review_statistics(reviews)

        # Save results
        review_results = {
            "manuscript_id": manuscript_id,
            "journal": journal,
            "review_date": datetime.now(timezone.utc).isoformat(),
            "selected_reviewers": [
                {
                    "reviewer_id": review.reviewer_id,
                    "name": next(r.name for r in selected_reviewers if r.reviewer_id == review.reviewer_id),
                    "institution": next(r.institution for r in selected_reviewers if r.reviewer_id == review.reviewer_id),
                    "expertise_areas": next(r.expertise_areas for r in selected_reviewers if r.reviewer_id == review.reviewer_id)
                }
                for review in reviews
            ],
            "individual_reviews": [asdict(review) for review in reviews],
            "editorial_decision": editorial_decision,
            "review_statistics": review_statistics
        }

        # Save to file
        output_file = self.output_dir / f"peer_review_{manuscript_id}_{journal.replace(' ', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(review_results, f, indent=2, default=str)

        logger.info(f"Peer review simulation saved to {output_file}")

        return review_results

    def _make_editorial_decision(self, reviews: List[DetailedReview], journal: str) -> Dict[str, Any]:
        """Make editorial decision based on reviewer recommendations."""

        recommendations = [review.overall_recommendation for review in reviews]

        # Count recommendations
        rec_counts = defaultdict(int)
        for rec in recommendations:
            rec_counts[rec] += 1

        # Determine decision (simplified editorial logic)
        if rec_counts["Accept"] >= 2:
            decision = "Accept"
            reasoning = "Strong positive reviews recommend acceptance"
        elif rec_counts["Minor Revisions"] >= 2:
            decision = "Minor Revisions"
            reasoning = "Reviewers recommend minor revisions"
        elif rec_counts["Major Revisions"] >= 2:
            decision = "Major Revisions"
            reasoning = "Significant revisions required before reconsideration"
        elif rec_counts["Reject"] >= 2:
            decision = "Reject"
            reasoning = "Fundamental issues identified by multiple reviewers"
        else:
            # Mixed reviews - editorial decision based on quality scores
            avg_score = np.mean([np.mean([
                review.review_criteria.scientific_rigor,
                review.review_criteria.novelty_significance,
                review.review_criteria.methodological_quality
            ]) for review in reviews])

            if avg_score > 0.7:
                decision = "Major Revisions"
                reasoning = "Mixed reviews but potential for publication after revision"
            else:
                decision = "Reject"
                reasoning = "Insufficient quality despite mixed reviews"

        return {
            "decision": decision,
            "reasoning": reasoning,
            "recommendation_counts": dict(rec_counts),
            "confidence": "High" if rec_counts.get(decision, 0) >= 2 else "Medium"
        }

    def _calculate_review_statistics(self, reviews: List[DetailedReview]) -> Dict[str, Any]:
        """Calculate statistics for the peer review process."""

        # Average scores by criterion
        criteria_averages = {}
        criteria_names = ["scientific_rigor", "novelty_significance", "methodological_quality",
                         "clarity_presentation", "reproducibility", "literature_context",
                         "appropriateness_for_journal"]

        for criterion in criteria_names:
            scores = [getattr(review.review_criteria, criterion) for review in reviews]
            criteria_averages[criterion] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }

        # Comment statistics
        total_comments = sum(len(review.major_comments) + len(review.minor_comments) +
                           len(review.technical_comments) + len(review.suggestions)
                           for review in reviews)

        avg_revision_time = np.mean([review.estimated_revision_time for review in reviews])

        return {
            "num_reviewers": len(reviews),
            "criteria_scores": criteria_averages,
            "total_comments": total_comments,
            "average_revision_time_days": avg_revision_time,
            "recommendation_distribution": {
                rec: len([r for r in reviews if r.overall_recommendation == rec])
                for rec in ["Accept", "Minor Revisions", "Major Revisions", "Reject"]
            }
        }

    def generate_review_report(self, review_results: Dict[str, Any]) -> str:
        """Generate a comprehensive peer review report."""

        report = f"""# Peer Review Simulation Report

**Manuscript ID:** {review_results['manuscript_id']}
**Journal:** {review_results['journal']}
**Review Date:** {review_results['review_date']}
**Number of Reviewers:** {review_results['review_statistics']['num_reviewers']}

## Editorial Decision

**Decision:** {review_results['editorial_decision']['decision']}
**Reasoning:** {review_results['editorial_decision']['reasoning']}
**Confidence:** {review_results['editorial_decision']['confidence']}

### Recommendation Distribution
"""

        for rec, count in review_results['editorial_decision']['recommendation_counts'].items():
            report += f"- {rec}: {count}\n"

        report += f"""
## Review Statistics

### Average Scores by Criterion
"""

        for criterion, stats in review_results['review_statistics']['criteria_scores'].items():
            report += f"- **{criterion.replace('_', ' ').title()}:** {stats['mean']:.2f} ± {stats['std']:.2f}\n"

        report += f"""
### Comments and Revision Time
- **Total Comments:** {review_results['review_statistics']['total_comments']}
- **Average Revision Time:** {review_results['review_statistics']['average_revision_time_days']:.1f} days

## Individual Reviewer Comments

"""

        for i, review in enumerate(review_results['individual_reviews'], 1):
            reviewer_info = next(r for r in review_results['selected_reviewers'] if r['reviewer_id'] == review['reviewer_id'])

            report += f"""### Reviewer {i}: {reviewer_info['name']}
**Institution:** {reviewer_info['institution']}
**Expertise:** {', '.join(reviewer_info['expertise_areas'])}
**Recommendation:** {review['overall_recommendation']}
**Confidence Level:** {review['confidence_level']}

#### Summary
{review['summary_comments']}

#### Major Comments
"""

            for comment in review['major_comments']:
                report += f"- {comment['text']}\n"

            report += "\n#### Minor Comments\n"
            for comment in review['minor_comments']:
                report += f"- {comment['text']}\n"

            report += "\n#### Technical Comments\n"
            for comment in review['technical_comments']:
                report += f"- {comment['text']}\n"

            report += "\n#### Strengths\n"
            for strength in review['strengths']:
                report += f"- {strength}\n"

            report += "\n#### Weaknesses\n"
            for weakness in review['weaknesses']:
                report += f"- {weakness}\n"

            report += "\n" + "="*50 + "\n\n"

        report += """
## Recommendations for Authors

Based on the reviewer comments, the authors should focus on:

1. **Addressing Major Concerns:** Prioritize issues raised by multiple reviewers
2. **Methodological Improvements:** Enhance description and validation of methods
3. **Clarity Enhancement:** Improve presentation and accessibility
4. **Contextualization:** Better position work within existing literature
5. **Reproducibility:** Ensure all computational details are provided

## Next Steps

1. **Revise Manuscript:** Address all reviewer comments systematically
2. **Response Letter:** Prepare detailed response to reviewer comments
3. **Resubmission:** Submit revised manuscript with appropriate documentation
4. **Timeline:** Allow adequate time for thorough revisions

---
*This peer review simulation was generated using the Academic Publication Pipeline*
        """

        # Save report
        report_file = self.output_dir / f"review_report_{review_results['manuscript_id']}_{review_results['journal'].replace(' ', '_')}.md"
        with open(report_file, 'w') as f:
            f.write(report)

        return str(report_file)


def main():
    """Main function to demonstrate peer review simulation."""
    simulator = PeerReviewSimulator()

    print("Peer Review Simulation Framework v1.0.0")
    print("=" * 50)
    print("Simulating realistic peer review process...")
    print()

    # Example manuscript content (would come from actual manuscript)
    manuscript_content = {
        "title": "Comprehensive Computational Framework for Analog Hawking Radiation",
        "abstract": "We present a comprehensive computational framework...",
        "introduction": "Analog gravity experiments offer unique opportunities...",
        "methods": "Our framework implements a multi-stage pipeline...",
        "results": "Systematic analysis reveals key findings...",
        "discussion": "Our results establish computational feasibility...",
        "conclusions": "We have developed a comprehensive framework..."
    }

    # Simulate peer review for Nature Physics
    manuscript_id = f"manuscript_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    journal = "Nature Physics"

    print(f"Simulating peer review for {journal}...")
    review_results = simulator.simulate_peer_review_process(manuscript_content, journal, manuscript_id)

    # Generate comprehensive report
    report_file = simulator.generate_review_report(review_results)

    print(f"\n✅ Peer review simulation completed!")
    print(f"📊 Editorial Decision: {review_results['editorial_decision']['decision']}")
    print(f"📁 Results saved in: {simulator.output_dir}")
    print(f"📄 Report available at: {report_file}")
    print()

    # Display summary statistics
    stats = review_results['review_statistics']
    print("Review Summary:")
    print(f"  Average Scientific Rigor: {stats['criteria_scores']['scientific_rigor']['mean']:.2f}")
    print(f"  Average Novelty: {stats['criteria_scores']['novelty_significance']['mean']:.2f}")
    print(f"  Average Methodological Quality: {stats['criteria_scores']['methodological_quality']['mean']:.2f}")
    print(f"  Total Comments: {stats['total_comments']}")
    print(f"  Avg Revision Time: {stats['average_revision_time_days']:.1f} days")

    return review_results


if __name__ == "__main__":
    main()