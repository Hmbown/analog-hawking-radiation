#!/usr/bin/env python3
"""
ELI Facility Compatibility Assessment Report Generator

This script generates comprehensive compatibility assessment reports for analog
Hawking radiation experiments at ELI facilities. It produces detailed reports
including facility comparisons, experimental recommendations, and feasibility
assessments.

Author: Claude Analysis Assistant
Date: November 2025
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import yaml

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.facilities.eli_capabilities import ELICapabilities, ELIFacility
from analog_hawking.facilities.eli_physics_validator import ELIPhysicsValidator
sys.path.insert(0, str(Path(__file__).parent))
from comprehensive_eli_facility_validator import ELIAnalogHawkingValidator


class ELICompatibilityReportGenerator:
    """Generate comprehensive ELI compatibility assessment reports"""

    def __init__(self):
        self.eli_validator = ELIAnalogHawkingValidator()
        self.physics_validator = ELIPhysicsValidator()
        self.eli_caps = ELICapabilities()

    def generate_comprehensive_report(
        self,
        base_intensity_W_m2: float,
        wavelength_nm: float = 800,
        pulse_duration_fs: float = 150,
        plasma_density_m3: float = 1e25,
        output_dir: str = "reports",
    ) -> Dict[str, Any]:
        """
        Generate comprehensive ELI compatibility report

        Args:
            base_intensity_W_m2: Base laser intensity in W/m²
            wavelength_nm: Laser wavelength in nm
            pulse_duration_fs: Pulse duration in fs
            plasma_density_m3: Plasma density in m⁻³
            output_dir: Output directory for reports

        Returns:
            Comprehensive report dictionary
        """

        print("📊 GENERATING COMPREHENSIVE ELI COMPATIBILITY REPORT")
        print("=" * 70)
        print(f"Base Intensity: {base_intensity_W_m2:.2e} W/m²")
        print(f"Wavelength: {wavelength_nm:.0f} nm")
        print(f"Pulse Duration: {pulse_duration_fs:.0f} fs")
        print(f"Plasma Density: {plasma_density_m3:.2e} m⁻³")
        print()

        # Initialize report structure
        report = {
            "metadata": {
                "generation_date": datetime.now().isoformat(),
                "analog_hawking_version": "0.3.0",
                "report_type": "comprehensive_eli_compatibility",
                "base_parameters": {
                    "intensity_W_m2": base_intensity_W_m2,
                    "wavelength_nm": wavelength_nm,
                    "pulse_duration_fs": pulse_duration_fs,
                    "plasma_density_m3": plasma_density_m3,
                },
            },
            "facility_assessments": {},
            "comparative_analysis": {},
            "experimental_recommendations": {},
            "risk_assessment": {},
            "feasibility_matrix": {},
            "next_steps": {},
        }

        # Generate facility-specific assessments
        for facility in ELIFacility:
            print(f"🏢 ASSESSING {facility.value.upper()}")
            facility_assessment = self._generate_facility_assessment(
                facility, base_intensity_W_m2, wavelength_nm, pulse_duration_fs, plasma_density_m3
            )
            report["facility_assessments"][facility.value] = facility_assessment

        # Generate comparative analysis
        print("\n📈 COMPARATIVE ANALYSIS")
        comparative_analysis = self._generate_comparative_analysis(report["facility_assessments"])
        report["comparative_analysis"] = comparative_analysis

        # Generate experimental recommendations
        print("\n💡 EXPERIMENTAL RECOMMENDATIONS")
        experimental_recommendations = self._generate_experimental_recommendations(
            report["facility_assessments"], report["comparative_analysis"]
        )
        report["experimental_recommendations"] = experimental_recommendations

        # Generate risk assessment
        print("\n⚠️  RISK ASSESSMENT")
        risk_assessment = self._generate_risk_assessment(report["facility_assessments"])
        report["risk_assessment"] = risk_assessment

        # Generate feasibility matrix
        print("\n✅ FEASIBILITY MATRIX")
        feasibility_matrix = self._generate_feasibility_matrix(report["facility_assessments"])
        report["feasibility_matrix"] = feasibility_matrix

        # Generate next steps
        print("\n🚀 NEXT STEPS")
        next_steps = self._generate_next_steps(report)
        report["next_steps"] = next_steps

        return report

    def _generate_facility_assessment(
        self,
        facility: ELIFacility,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        plasma_density_m3: float,
    ) -> Dict[str, Any]:
        """Generate detailed assessment for a specific facility"""

        print(f"   📍 Analyzing {facility.value}...")

        # Get facility-specific configuration
        config_file = Path(__file__).parent.parent / "configs" / f"{facility.value.lower().replace('-', '_')}_config.yaml"
        facility_config = {}
        if config_file.exists():
            with open(config_file, 'r') as f:
                facility_config = yaml.safe_load(f)

        # Run comprehensive validation
        validation_result = self.eli_validator.validate_full_configuration(
            intensity_W_m2,
            wavelength_nm,
            pulse_duration_fs,
            plasma_density_m3,
            facility_config.get("experimental_parameters", {}).get("target_type", "solid"),
            facility.value.lower(),
            include_hawking_analysis=True,
        )

        # Run physics validation
        gradient_scale_m = 1e-6  # Assume 1 μm gradient scale
        flow_velocity_ms = 2e6   # Assume 2×10⁶ m/s flow velocity

        physics_result = self.physics_validator.validate_comprehensive_configuration(
            intensity_W_m2,
            wavelength_nm,
            pulse_duration_fs,
            plasma_density_m3,
            gradient_scale_m,
            flow_velocity_ms,
            facility,
        )

        # Calculate facility-specific metrics
        facility_metrics = self._calculate_facility_metrics(
            facility, validation_result, physics_result, facility_config
        )

        # Generate facility-specific recommendations
        facility_recommendations = self._generate_facility_recommendations(
            facility, validation_result, physics_result, facility_metrics
        )

        assessment = {
            "facility_name": facility.value,
            "validation_result": validation_result,
            "physics_validation": physics_result,
            "facility_metrics": facility_metrics,
            "facility_config": facility_config,
            "recommendations": facility_recommendations,
            "summary": {
                "overall_feasibility": validation_result["overall_assessment"]["overall_feasibility"],
                "confidence_level": validation_result["overall_assessment"]["confidence_level"],
                "key_strengths": validation_result["overall_assessment"]["key_strengths"],
                "key_challenges": validation_result["overall_assessment"]["key_challenges"],
            },
        }

        # Print summary
        print(f"      Feasibility: {assessment['summary']['overall_feasibility']}")
        print(f"      Confidence: {assessment['summary']['confidence_level']:.2f}")
        print(f"      Strengths: {len(assessment['summary']['key_strengths'])}")
        print(f"      Challenges: {len(assessment['summary']['key_challenges'])}")

        return assessment

    def _calculate_facility_metrics(
        self,
        facility: ELIFacility,
        validation_result: Dict[str, Any],
        physics_result: Dict[str, Any],
        facility_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Calculate facility-specific performance metrics"""

        # Basic facility metrics
        facility_metrics = {
            "facility_advantages": [],
            "facility_limitations": [],
            "performance_indicators": {},
            "experimental_efficiency": {},
        }

        # Get facility constraints
        facility_constraints = self.eli_caps.facility_constraints[facility]

        # Calculate performance indicators
        facility_metrics["performance_indicators"] = {
            "intensity_utilization": (
                validation_result["input_parameters"]["intensity_W_m2"] /
                facility_constraints["max_intensity_W_cm2"] / 1e4
            ),
            "detection_feasibility_score": (
                validation_result.get("hawking_analysis", {}).get("detection_feasible", False)
            ),
            "physics_compliance_rate": physics_result["confidence_score"],
            "shot_efficiency": self._calculate_shot_efficiency(facility, facility_config),
            "data_quality_potential": self._estimate_data_quality(facility, validation_result),
        }

        # Identify advantages and limitations
        if facility == ELIFacility.ELI_BEAMLINES:
            facility_metrics["facility_advantages"] = [
                "Highest available intensity (10 PW)",
                "Good focal spot quality",
                "Established plasma physics program",
            ]
            facility_metrics["facility_limitations"] = [
                "Low repetition rate (1 shot/minute)",
                "Long experimental campaigns",
                "Limited shot statistics",
            ]

        elif facility == ELIFacility.ELI_NP:
            facility_metrics["facility_advantages"] = [
                "Dual-beam capability",
                "Excellent temporal contrast",
                "Magnetic field expertise",
                "Nuclear physics diagnostics",
            ]
            facility_metrics["facility_limitations"] = [
                "Very low repetition rate (1 shot/5 min)",
                "Complex safety procedures",
                "Limited shot availability",
            ]

        elif facility == ELIFacility.ELI_ALPS:
            facility_metrics["facility_advantages"] = [
                "High repetition rate",
                "Excellent for statistics",
                "Fast diagnostic capabilities",
                "Automated experimental control",
            ]
            facility_metrics["facility_limitations"] = [
                "Lower maximum intensity",
                "Shorter pulse duration challenges",
                "Plasma mirror timing precision",
            ]

        # Calculate experimental efficiency
        facility_metrics["experimental_efficiency"] = {
            "setup_time_days": facility_config.get("timeline", {}).get("setup_days", 7),
            "shots_per_day": facility_config.get("experimental_parameters", {}).get("repetition_rate_Hz", 1) * 86400,
            "total_campaign_days": facility_config.get("timeline", {}).get("total_campaign_days", 10),
            "data_collection_efficiency": self._calculate_data_efficiency(facility, validation_result),
        }

        return facility_metrics

    def _calculate_shot_efficiency(self, facility: ELIFacility, facility_config: Dict[str, Any]) -> float:
        """Calculate shot efficiency metric"""

        timeline = facility_config.get("timeline", {})
        collection_shots = timeline.get("data_collection_shots", 20)
        campaign_days = timeline.get("total_campaign_days", 10)

        if campaign_days > 0:
            return collection_shots / campaign_days
        return 0.0

    def _estimate_data_quality(self, facility: ELIFacility, validation_result: Dict[str, Any]) -> float:
        """Estimate potential data quality"""

        factors = []

        # Detection feasibility
        if validation_result.get("hawking_analysis", {}).get("detection_feasible", False):
            factors.append(0.8)
        else:
            factors.append(0.3)

        # Facility stability (repetition rate)
        if facility == ELIFacility.ELI_ALPS:
            factors.append(0.9)  # High rep rate = good statistics
        elif facility == ELIFacility.ELI_BEAMLINES:
            factors.append(0.6)  # Medium rep rate
        else:  # ELI-NP
            factors.append(0.4)  # Low rep rate = few shots

        # Diagnostic capabilities
        if facility == ELIFacility.ELI_NP:
            factors.append(0.8)  # Excellent diagnostics
        else:
            factors.append(0.6)

        return np.mean(factors)

    def _calculate_data_efficiency(self, facility: ELIFacility, validation_result: Dict[str, Any]) -> float:
        """Calculate data collection efficiency"""

        # Based on repetition rate and detection feasibility
        if facility == ELIFacility.ELI_ALPS:
            base_efficiency = 0.9  # High rep rate
        elif facility == ELIFacility.ELI_BEAMLINES:
            base_efficiency = 0.6  # Medium rep rate
        else:  # ELI-NP
            base_efficiency = 0.3  # Low rep rate

        # Adjust for detection feasibility
        if validation_result.get("hawking_analysis", {}).get("detection_feasible", False):
            return base_efficiency
        else:
            return base_efficiency * 0.5

    def _generate_facility_recommendations(
        self,
        facility: ELIFacility,
        validation_result: Dict[str, Any],
        physics_result: Dict[str, Any],
        facility_metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate facility-specific recommendations"""

        recommendations = []

        # Base recommendations from validation
        recommendations.extend(validation_result.get("recommendations", []))

        # Physics-based recommendations
        if physics_result["critical_issues"]:
            recommendations.append(f"CRITICAL: Address physics threshold violations for {facility.value}")

        if physics_result["warnings"]:
            recommendations.append(f"Consider physics optimization for {facility.value}")

        # Facility-specific recommendations
        if facility == ELIFacility.ELI_BEAMLINES:
            recommendations.extend([
                "Focus on high-intensity single-shot experiments",
                "Implement advanced plasma mirror timing control",
                "Plan for long data collection periods",
            ])

        elif facility == ELIFacility.ELI_NP:
            recommendations.extend([
                "Leverage dual-beam capability for enhanced diagnostics",
                "Utilize magnetic field expertise for plasma control",
                "Plan extended campaign due to low repetition rate",
                "Implement nuclear safety protocols early",
            ])

        elif facility == ELIFacility.ELI_ALPS:
            recommendations.extend([
                "Optimize for high-statistics data collection",
                "Implement fast target translation system",
                "Focus on parameter space mapping",
                "Utilize automated experimental control",
            ])

        # Feasibility-based recommendations
        overall_feasibility = validation_result["overall_assessment"]["overall_feasibility"]
        if overall_feasibility == "HIGH":
            recommendations.append("✅ Proceed with detailed experimental proposal")
        elif overall_feasibility == "MEDIUM":
            recommendations.append("⚠️  Address key challenges before proceeding")
        else:
            recommendations.append("❌ Significant modifications required")

        return recommendations

    def _generate_comparative_analysis(self, facility_assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparative analysis between facilities"""

        comparative = {
            "facility_ranking": [],
            "capability_comparison": {},
            "parameter_optimization": {},
            "best_use_cases": {},
        }

        # Rank facilities by overall feasibility
        facility_scores = []
        for facility_name, assessment in facility_assessments.items():
            score = assessment["summary"]["confidence_level"]
            facility_scores.append((facility_name, score))

        facility_scores.sort(key=lambda x: x[1], reverse=True)
        comparative["facility_ranking"] = facility_scores

        # Compare capabilities
        capabilities = {
            "max_intensity_W_m2": {},
            "repetition_rate_Hz": {},
            "data_quality_potential": {},
            "shot_efficiency": {},
        }

        for facility_name, assessment in facility_assessments.items():
            metrics = assessment["facility_metrics"]
            capabilities["max_intensity_W_m2"][facility_name] = metrics["performance_indicators"]["intensity_utilization"]
            capabilities["repetition_rate_Hz"][facility_name] = metrics["experimental_efficiency"]["shots_per_day"]
            capabilities["data_quality_potential"][facility_name] = metrics["performance_indicators"]["data_quality_potential"]
            capabilities["shot_efficiency"][facility_name] = metrics["experimental_efficiency"]["data_collection_efficiency"]

        comparative["capability_comparison"] = capabilities

        # Parameter optimization suggestions
        comparative["parameter_optimization"] = {
            "intensity_optimization": "ELI-Beamlines and ELI-NP offer highest intensity capabilities",
            "statistics_optimization": "ELI-ALPS provides best statistical data collection",
            "diagnostics_optimization": "ELI-NP offers most comprehensive diagnostic suite",
            "efficiency_optimization": "ELI-ALPS provides most efficient data collection",
        }

        # Best use cases
        comparative["best_use_cases"] = {
            "ELI-Beamlines": "High-intensity single-shot experiments and proof-of-concept demonstrations",
            "ELI-NP": "Advanced experiments with dual-beam configuration and enhanced diagnostics",
            "ELI-ALPS": "High-statistics parameter space mapping and optimization studies",
        }

        return comparative

    def _generate_experimental_recommendations(
        self, facility_assessments: Dict[str, Any], comparative_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate overall experimental recommendations"""

        recommendations = {
            "primary_facility": None,
            "experimental_phases": [],
            "parameter_optimizations": [],
            "risk_mitigation_strategies": [],
            "success_metrics": [],
        }

        # Determine primary facility recommendation
        top_facility = comparative_analysis["facility_ranking"][0][0]
        recommendations["primary_facility"] = {
            "name": top_facility,
            "confidence": comparative_analysis["facility_ranking"][0][1],
            "rationale": f"Highest overall feasibility score and best match for experimental requirements",
        }

        # Define experimental phases
        recommendations["experimental_phases"] = [
            {
                "phase": "Phase 1 - Proof of Concept",
                "facility": "ELI-ALPS",
                "rationale": "High repetition rate for rapid parameter optimization",
                "duration_weeks": 2,
                "goals": ["Plasma mirror optimization", "Basic horizon detection", "Diagnostic validation"],
            },
            {
                "phase": "Phase 2 - High-Performance Measurements",
                "facility": "ELI-Beamlines",
                "rationale": "Maximum intensity for optimal signal strength",
                "duration_weeks": 4,
                "goals": ["High-κ measurements", "Detailed spectrum analysis", "Reproducibility verification"],
            },
            {
                "phase": "Phase 3 - Advanced Characterization",
                "facility": "ELI-NP",
                "rationale": "Enhanced diagnostics and dual-beam capabilities",
                "duration_weeks": 3,
                "goals": ["Correlation studies", "Magnetic field effects", "Comprehensive validation"],
            },
        ]

        # Parameter optimizations
        recommendations["parameter_optimizations"] = [
            "Focus on intensity range 1e22-1e24 W/m² for optimal κ",
            "Target plasma density 1e25-1e26 m⁻³ for horizon formation",
            "Optimize density gradient scale 0.3-1.0 μm for strong gradients",
            "Maintain flow velocity 0.2-0.4c for good horizon dynamics",
        ]

        # Risk mitigation
        recommendations["risk_mitigation_strategies"] = [
            "Start with conservative parameters and gradually increase",
            "Implement redundant diagnostic systems",
            "Develop detailed plasma mirror timing protocols",
            "Create comprehensive data analysis pipelines",
            "Establish clear success criteria for each phase",
        ]

        # Success metrics
        recommendations["success_metrics"] = [
            "κ measurement with < 20% uncertainty",
            "Hawking-like spectrum detection with 5σ confidence",
            "Reproducible results across multiple shots",
            "Theoretical-experimental agreement within 50%",
            "Publication-ready data quality",
        ]

        return recommendations

    def _generate_risk_assessment(self, facility_assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive risk assessment"""

        risk_assessment = {
            "technical_risks": {},
            "facility_risks": {},
            "physics_risks": {},
            "mitigation_strategies": {},
            "overall_risk_level": "MEDIUM",
        }

        # Technical risks
        technical_risks = []
        for facility_name, assessment in facility_assessments.items():
            facility_risks = []
            if assessment["validation_result"]["physics_validation"]["critical_issues"]:
                facility_risks.append("Physics threshold violations")
            if not assessment["validation_result"]["plasma_mirror_analysis"]["mirror_formation_feasible"]:
                facility_risks.append("Plasma mirror formation challenges")
            if not assessment.get("hawking_analysis", {}).get("detection_feasible", False):
                facility_risks.append("Detection feasibility concerns")
            technical_risks.append((facility_name, facility_risks))

        risk_assessment["technical_risks"] = dict(technical_risks)

        # Facility-specific risks
        facility_risks = {
            "ELI-Beamlines": ["Low repetition rate limits statistics", "Long experimental campaigns"],
            "ELI-NP": ["Very low repetition rate", "Complex safety procedures", "Limited beam time"],
            "ELI-ALPS": ["Lower maximum intensity", "Plasma mirror timing precision requirements"],
        }
        risk_assessment["facility_risks"] = facility_risks

        # Physics risks
        physics_risks = [
            "Breakdown of fluid model assumptions at high intensity",
            "Unforeseen relativistic effects",
            "Plasma mirror instabilities",
            "Detection system limitations",
            "Background noise contamination",
        ]
        risk_assessment["physics_risks"] = physics_risks

        # Mitigation strategies
        mitigation_strategies = [
            "Start with conservative parameter ranges",
            "Implement comprehensive diagnostic suite",
            "Develop real-time data analysis capabilities",
            "Create backup experimental configurations",
            "Establish collaboration with facility experts",
            "Perform detailed simulations before experiments",
        ]
        risk_assessment["mitigation_strategies"] = mitigation_strategies

        return risk_assessment

    def _generate_feasibility_matrix(self, facility_assessments: Dict[str, Any]) -> Dict[str, Any]:
        """Generate feasibility matrix for different experimental goals"""

        feasibility_matrix = {
            "goals": {
                "proof_of_concept": {"requirement": "Basic horizon detection", "difficulty": "LOW"},
                "parameter_optimization": {"requirement": "Systematic parameter studies", "difficulty": "MEDIUM"},
                "high_precision_measurement": {"requirement": "Detailed spectrum analysis", "difficulty": "HIGH"},
                "reproducibility_study": {"requirement": "Multiple consistent results", "difficulty": "HIGH"},
            },
            "facility_scores": {},
        }

        # Score each facility for each goal
        for facility_name, assessment in facility_assessments.items():
            scores = {}
            confidence = assessment["summary"]["confidence_level"]

            # Adjust scores based on facility characteristics
            if facility_name == "ELI-ALPS":
                scores["proof_of_concept"] = confidence * 1.2  # Good for rapid testing
                scores["parameter_optimization"] = confidence * 1.5  # Excellent for optimization
                scores["high_precision_measurement"] = confidence * 0.8  # Limited by intensity
                scores["reproducibility_study"] = confidence * 1.3  # Good for statistics
            elif facility_name == "ELI-Beamlines":
                scores["proof_of_concept"] = confidence * 1.0
                scores["parameter_optimization"] = confidence * 0.7  # Limited by rep rate
                scores["high_precision_measurement"] = confidence * 1.2  # Good intensity
                scores["reproducibility_study"] = confidence * 0.6  # Limited shots
            else:  # ELI-NP
                scores["proof_of_concept"] = confidence * 0.9
                scores["parameter_optimization"] = confidence * 0.5  # Very low rep rate
                scores["high_precision_measurement"] = confidence * 1.4  # Excellent diagnostics
                scores["reproducibility_study"] = confidence * 0.4  # Very limited shots

            feasibility_matrix["facility_scores"][facility_name] = scores

        return feasibility_matrix

    def _generate_next_steps(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate recommended next steps"""

        next_steps = []

        # Immediate actions (next 1-2 months)
        next_steps.append({
            "timeline": "Immediate (1-2 months)",
            "actions": [
                "Contact recommended facilities for beam time inquiries",
                "Develop detailed experimental proposal",
                "Secure funding for campaign",
                "Form collaboration with facility experts",
            ],
            "priority": "HIGH",
        })

        # Short-term actions (2-6 months)
        next_steps.append({
            "timeline": "Short-term (2-6 months)",
            "actions": [
                "Complete detailed simulations with PIC codes",
                "Finalize diagnostic specifications",
                "Develop data analysis pipeline",
                "Prepare safety documentation",
            ],
            "priority": "HIGH",
        })

        # Medium-term actions (6-12 months)
        next_steps.append({
            "timeline": "Medium-term (6-12 months)",
            "actions": [
                "Schedule beam time at primary facility",
                "Fabricate and test target systems",
                "Install and validate diagnostic equipment",
                "Perform initial commissioning experiments",
            ],
            "priority": "MEDIUM",
        })

        # Long-term actions (12+ months)
        next_steps.append({
            "timeline": "Long-term (12+ months)",
            "actions": [
                "Execute full experimental campaign",
                "Analyze and publish results",
                "Plan follow-up experiments",
                "Explore facility upgrades or alternatives",
            ],
            "priority": "MEDIUM",
        })

        return next_steps

    def save_report(
        self, report: Dict[str, Any], output_dir: str = "reports", format: str = "json"
    ) -> str:
        """Save comprehensive report to file"""

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format.lower() == "json":
            filename = f"eli_compatibility_report_{timestamp}.json"
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)

        elif format.lower() == "yaml":
            filename = f"eli_compatibility_report_{timestamp}.yaml"
            filepath = output_path / filename
            with open(filepath, 'w') as f:
                yaml.dump(report, f, default_flow_style=False)

        elif format.lower() == "summary":
            filename = f"eli_compatibility_summary_{timestamp}.md"
            filepath = output_path / filename
            summary = self._generate_summary_markdown(report)
            with open(filepath, 'w') as f:
                f.write(summary)

        print(f"\n📄 Report saved to: {filepath}")
        return str(filepath)

    def _generate_summary_markdown(self, report: Dict[str, Any]) -> str:
        """Generate markdown summary of the report"""

        metadata = report["metadata"]
        comparative = report["comparative_analysis"]
        recommendations = report["experimental_recommendations"]

        summary = f"""# ELI Facility Compatibility Assessment Summary

**Generated:** {metadata['generation_date']}
**Base Parameters:**
- Intensity: {metadata['base_parameters']['intensity_W_m2']:.2e} W/m²
- Wavelength: {metadata['base_parameters']['wavelength_nm']} nm
- Pulse Duration: {metadata['base_parameters']['pulse_duration_fs']} fs
- Plasma Density: {metadata['base_parameters']['plasma_density_m3']:.2e} m⁻³

## Facility Ranking

"""
        for i, (facility, score) in enumerate(comparative["facility_ranking"], 1):
            summary += f"{i}. **{facility}** - Feasibility Score: {score:.2f}\n"

        summary += f"""

## Primary Recommendation

**Recommended Facility:** {recommendations['primary_facility']['name']}
**Confidence Level:** {recommendations['primary_facility']['confidence']:.2f}
**Rationale:** {recommendations['primary_facility']['rationale']}

## Experimental Phases

"""
        for phase in recommendations["experimental_phases"]:
            summary += f"""### {phase['phase']}
- **Facility:** {phase['facility']}
- **Duration:** {phase['duration_weeks']} weeks
- **Rationale:** {phase['rationale']}
- **Goals:** {', '.join(phase['goals'])}

"""

        summary += f"""## Key Success Metrics
"""
        for metric in recommendations["success_metrics"]:
            summary += f"- {metric}\n"

        summary += f"""

## Next Steps

"""
        for step in recommendations["next_steps"]:
            summary += f"### {step['timeline']} (Priority: {step['priority']})\n"
            for action in step['actions']:
                summary += f"- {action}\n\n"

        return summary


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Generate comprehensive ELI facility compatibility assessment reports"
    )
    parser.add_argument("--intensity", type=float, default=1e22, help="Base laser intensity (W/m²)")
    parser.add_argument("--wavelength", type=float, default=800, help="Laser wavelength (nm)")
    parser.add_argument("--pulse-duration", type=float, default=150, help="Pulse duration (fs)")
    parser.add_argument("--plasma-density", type=float, default=1e25, help="Plasma density (m⁻³)")
    parser.add_argument("--output-dir", type=str, default="reports", help="Output directory")
    parser.add_argument("--format", choices=["json", "yaml", "summary"], default="json", help="Output format")
    parser.add_argument("--all-formats", action="store_true", help="Generate all output formats")

    args = parser.parse_args()

    # Initialize report generator
    generator = ELICompatibilityReportGenerator()

    # Generate comprehensive report
    report = generator.generate_comprehensive_report(
        args.intensity,
        args.wavelength,
        args.pulse_duration,
        args.plasma_density,
        args.output_dir,
    )

    # Save report(s)
    if args.all_formats:
        formats = ["json", "yaml", "summary"]
    else:
        formats = [args.format]

    saved_files = []
    for fmt in formats:
        filepath = generator.save_report(report, args.output_dir, fmt)
        saved_files.append(filepath)

    print(f"\n✅ Report generation complete!")
    print(f"📁 Files saved: {len(saved_files)}")
    for file in saved_files:
        print(f"   - {file}")

    return 0


if __name__ == "__main__":
    exit(main())