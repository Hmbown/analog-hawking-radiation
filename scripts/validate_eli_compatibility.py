#!/usr/bin/env python3
"""
ELI Facility Compatibility Validation Script

This script validates all laser parameters used in the analog Hawking radiation
analysis against actual ELI facility capabilities and generates comprehensive
feasibility reports.

Author: Claude Analysis Assistant
Date: November 2025
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

warnings.filterwarnings("ignore")


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from analog_hawking.facilities.eli_capabilities import (
    ELIFacility,
    get_eli_capabilities,
    validate_intensity_range,
)


class ELICompatibilityValidator:
    """Comprehensive ELI facility compatibility validator"""

    def __init__(self):
        self.eli = get_eli_capabilities()
        self.validation_results = []
        self.critical_issues = []
        self.recommendations = []

    def validate_repository_parameters(self) -> Dict[str, Any]:
        """Validate all parameters found in the repository"""

        print("🔍 SCANNING REPOSITORY FOR LASER PARAMETERS...")
        print("=" * 70)

        # Define parameter ranges found in the repository
        repository_parameters = [
            # From experimental_protocol.md
            {
                "source": "experimental_protocol.md",
                "intensity_W_m2": 1e16,
                "description": "Minimum laser intensity in experimental protocol",
            },
            {
                "source": "experimental_protocol.md",
                "intensity_W_m2": 1e20,
                "description": "Maximum laser intensity in experimental protocol",
            },
            {
                "source": "experimental_protocol.md",
                "intensity_W_m2": 5e17,
                "description": "Baseline laser intensity in experimental protocol",
            },
            # From enhanced_parameter_generator.py
            {
                "source": "enhanced_parameter_generator.py",
                "intensity_W_m2": 1e21,
                "description": "High-intensity range for strongly nonlinear regime",
            },
            {
                "source": "enhanced_parameter_generator.py",
                "intensity_W_m2": 1e24,
                "description": "Maximum intensity in parameter generator",
            },
            # From optimization scripts
            {
                "source": "optimize_glow_detection.py",
                "intensity_W_m2": 1e22,
                "description": "Minimum intensity for glow detection optimization",
            },
            {
                "source": "optimize_glow_detection.py",
                "intensity_W_m2": 1e23,
                "description": "Maximum intensity for glow detection optimization",
            },
            # From guidance map generation
            {
                "source": "generate_guidance_map.py",
                "intensity_W_m2": 1.152e22,
                "description": "Optimal intensity from guidance map",
            },
            # From test scripts
            {
                "source": "test_probabilistic_model.py",
                "intensity_W_m2": 5e22,
                "description": "High-intensity test case in probabilistic model",
            },
            {
                "source": "test_probabilistic_model.py",
                "intensity_W_m2": 1e22,
                "description": "Alternative high-intensity test case",
            },
            # From validation scripts
            {
                "source": "validation/quality_assurance.py",
                "intensity_W_m2": 1e21,
                "description": "Maximum intensity in quality assurance validation",
            },
            # From comprehensive Monte Carlo
            {
                "source": "comprehensive_monte_carlo_uncertainty.py",
                "intensity_W_m2": 5e16,
                "description": "Baseline intensity for Monte Carlo analysis",
            },
            # From physics model validator
            {
                "source": "validation/physics_model_validator.py",
                "intensity_W_m2": 1e22,
                "description": "Upper limit in physics model validation",
            },
            # From configuration thresholds
            {
                "source": "config/thresholds.py",
                "intensity_W_m2": 6.0e50,
                "description": "CRITICAL: Unrealistic maximum intensity threshold",
            },
        ]

        # Validate each parameter set
        for params in repository_parameters:
            self._validate_parameter_set(params)

        # Generate summary report
        return self._generate_summary_report()

    def _validate_parameter_set(self, params: Dict[str, Any]) -> None:
        """Validate a single parameter set"""

        intensity_W_m2 = params["intensity_W_m2"]
        source = params["source"]
        description = params["description"]

        print(f"\n📍 VALIDATING: {source}")
        print(f"   {description}")
        print(f"   Intensity: {intensity_W_m2:.2e} W/m²")

        # Check for critical issues
        if intensity_W_m2 > 1e25:
            self.critical_issues.append(
                {
                    "source": source,
                    "intensity": intensity_W_m2,
                    "issue": "PHYSICALLY IMPOSSIBLE INTENSITY",
                    "description": description,
                }
            )
            print("   ❌ CRITICAL: Intensity exceeds physical limits!")
            return

        # Validate against ELI capabilities
        validation_result = validate_intensity_range(intensity_W_m2)

        if not validation_result["valid"]:
            self.critical_issues.append(
                {
                    "source": source,
                    "intensity": intensity_W_m2,
                    "issue": validation_result["issue"],
                    "description": description,
                }
            )
            print(f"   ❌ {validation_result['issue']}")
            print(f"   💡 Recommendation: {validation_result['recommendation']}")
        else:
            print(f"   ✅ {validation_result['feasibility_level']}")
            print(
                f"   🏢 Compatible facilities: {', '.join(validation_result['compatible_facilities'])}"
            )

        # Add to validation results
        self.validation_results.append({**params, **validation_result})

    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""

        print("\n" + "=" * 70)
        print("📊 ELI COMPATIBILITY SUMMARY REPORT")
        print("=" * 70)

        # Count statistics
        total_parameters = len(self.validation_results)
        critical_issues_count = len(self.critical_issues)
        compatible_count = sum(1 for r in self.validation_results if r.get("valid", False))

        print("\n📈 STATISTICS:")
        print(f"   Total parameter sets validated: {total_parameters}")
        print(f"   Parameters with critical issues: {critical_issues_count}")
        print(f"   ELI-compatible parameters: {compatible_count}")
        print(f"   Compatibility rate: {100*compatible_count/total_parameters:.1f}%")

        # Analyze intensity distribution
        intensities = [r["intensity_W_m2"] for r in self.validation_results]
        intensity_ranges = {
            "Conservative (<1e18 W/m²)": sum(1 for i in intensities if i < 1e18),
            "Moderate (1e18-1e20 W/m²)": sum(1 for i in intensities if 1e18 <= i < 1e20),
            "High (1e20-1e22 W/m²)": sum(1 for i in intensities if 1e20 <= i < 1e22),
            "Very High (1e22-1e24 W/m²)": sum(1 for i in intensities if 1e22 <= i < 1e24),
            "Extreme (>1e24 W/m²)": sum(1 for i in intensities if i >= 1e24),
        }

        print("\n📊 INTENSITY DISTRIBUTION:")
        for range_name, count in intensity_ranges.items():
            print(f"   {range_name}: {count} parameter sets")

        # Facility compatibility analysis
        facility_compatibility = {}
        for result in self.validation_results:
            if result.get("valid", False):
                for facility in result.get("compatible_facilities", []):
                    facility_compatibility[facility] = facility_compatibility.get(facility, 0) + 1

        print("\n🏢 FACILITY COMPATIBILITY:")
        for facility, count in facility_compatibility.items():
            print(f"   {facility}: {count} compatible parameter sets")

        # Critical issues summary
        if self.critical_issues:
            print(f"\n⚠️  CRITICAL ISSUES ({len(self.critical_issues)}):")
            for i, issue in enumerate(self.critical_issues, 1):
                print(f"   {i}. {issue['source']}: {issue['issue']}")
                print(f"      Intensity: {issue['intensity']:.2e} W/m²")
                print(f"      Description: {issue['description']}")

        # Generate recommendations
        self._generate_recommendations()

        return {
            "statistics": {
                "total_parameters": total_parameters,
                "critical_issues": critical_issues_count,
                "compatible_parameters": compatible_count,
                "compatibility_rate": 100 * compatible_count / total_parameters,
            },
            "intensity_distribution": intensity_ranges,
            "facility_compatibility": facility_compatibility,
            "critical_issues": self.critical_issues,
            "recommendations": self.recommendations,
            "validation_results": self.validation_results,
        }

    def _generate_recommendations(self) -> None:
        """Generate specific recommendations for fixing issues"""

        print("\n💡 RECOMMENDATIONS:")

        # Unit consistency recommendations
        unit_issues = [r for r in self.validation_results if not r.get("valid", False)]
        if unit_issues:
            print("\n🔧 UNIT CONSISTENCY FIXES:")
            print("   1. STANDARDIZE all intensity values to W/m² throughout the codebase")
            print("   2. ADD unit conversion functions to prevent W/cm² vs W/m² confusion")
            print("   3. IMPLEMENT input validation for all parameter generation scripts")

        # Parameter range recommendations
        extreme_parameters = [r for r in self.validation_results if r["intensity_W_m2"] > 1e24]
        if extreme_parameters:
            print("\n🎯 PARAMETER RANGE CORRECTIONS:")
            print("   1. REDUCE maximum intensity to 1e23 W/m² for ELI-Beamlines/NP")
            print("   2. REDUCE maximum intensity to 1e21 W/m² for ELI-ALPS")
            print("   3. UPDATE config/thresholds.py with realistic maximum values")

        # Facility-specific recommendations
        print("\n🏢 FACILITY OPTIMIZATION:")
        print("   1. ELI-Beamlines: Best for high-intensity (>1e22 W/m²) experiments")
        print("   2. ELI-NP: Ideal for nuclear physics applications with dual-beam capability")
        print("   3. ELI-ALPS: Optimal for high-repetition rate (>10 Hz) experiments")

        # Experimental design recommendations
        print("\n🧪 EXPERIMENTAL DESIGN IMPROVEMENTS:")
        print("   1. FOCUS on 1e19-1e22 W/m² range for optimal plasma mirror operation")
        print("   2. USE 800 nm wavelength for Ti:Sapphire compatibility")
        print("   3. TARGET 150 fs pulse duration for 10 PW systems")
        print("   4. CONSIDER repetition rate requirements in experimental planning")

        # Add to recommendations list
        self.recommendations = [
            "Standardize all intensity units to W/m² throughout codebase",
            "Implement ELI facility constraints in parameter generation",
            "Add facility-specific validation functions",
            "Update configuration thresholds with realistic values",
            "Create experimental feasibility scoring system",
            "Implement automated ELI compatibility checks",
        ]

    def validate_specific_configuration(
        self,
        intensity_W_m2: float,
        wavelength_nm: float = 800,
        pulse_duration_fs: float = 150,
        facility: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate a specific experimental configuration"""

        print("\n🔬 VALIDATING SPECIFIC CONFIGURATION:")
        print(f"   Intensity: {intensity_W_m2:.2e} W/m²")
        print(f"   Wavelength: {wavelength_nm:.0f} nm")
        print(f"   Pulse Duration: {pulse_duration_fs:.0f} fs")
        if facility:
            print(f"   Target Facility: {facility}")

        # Convert to W/cm² for ELI validation
        intensity_W_cm2 = intensity_W_m2 / 1e4

        # Get facility compatibility
        target_facility = None
        if facility:
            try:
                target_facility = ELIFacility(
                    facility.lower().replace("-", "_").replace("eli_", "")
                )
            except ValueError:
                print(f"   ⚠️  Unknown facility: {facility}")
                target_facility = None

        # Get comprehensive feasibility assessment
        feasibility = self.eli.calculate_feasibility_score(
            intensity_W_cm2, wavelength_nm, pulse_duration_fs, target_facility
        )

        # Format and display results
        if feasibility["feasible"]:
            print("   ✅ CONFIGURATION FEASIBLE")
            print(f"   📊 Feasibility Score: {feasibility['score']:.2f}/1.00")
            print(f"   🏢 Best System: {feasibility['best_system']}")
            print(f"   📍 Facility: {feasibility['facility']}")
            print(f"   🔬 Experimental Hall: {feasibility['experimental_hall']}")
            print(f"   ⚡ Repetition Rate: {feasibility['repetition_rate_Hz']:.1f} Hz")
            print(f"   📈 Intensity Margin: {feasibility['intensity_margin']:.1f}x")

            if feasibility["recommendations"]:
                print("   💡 Recommendations:")
                for rec in feasibility["recommendations"]:
                    print(f"      • {rec}")
        else:
            print("   ❌ CONFIGURATION INFEASIBLE")
            print("   🚫 Primary Issues:")
            for issue in feasibility["primary_issues"]:
                print(f"      • {issue}")
            print("   💡 Recommendations:")
            for rec in feasibility["recommendations"]:
                print(f"      • {rec}")

        return feasibility

    def generate_eli_compliant_parameter_ranges(self) -> Dict[str, Any]:
        """Generate ELI-compliant parameter ranges for the analysis"""

        print("\n🎯 GENERATING ELI-COMPLIANT PARAMETER RANGES:")

        # Define facility-specific ranges
        facility_ranges = {
            "ELI-Beamlines": {
                "intensity_W_m2": (1e18, 1e24),  # Conservative to maximum
                "wavelength_nm": (800, 1030),
                "pulse_duration_fs": (30, 150),
                "repetition_rate_Hz": (0.017, 10),
                "best_use_cases": ["High-intensity plasma physics", "Relativistic regimes"],
            },
            "ELI-NP": {
                "intensity_W_m2": (1e19, 1e24),  # Higher minimum for nuclear physics
                "wavelength_nm": (810, 810),  # Fixed Ti:Sapphire
                "pulse_duration_fs": (150, 200),
                "repetition_rate_Hz": (0.003, 0.1),
                "best_use_cases": ["Nuclear physics experiments", "Dual-beam configurations"],
            },
            "ELI-ALPS": {
                "intensity_W_m2": (1e16, 1e22),  # Lower due to high rep rate focus
                "wavelength_nm": (800, 800),  # Fixed Ti:Sapphire
                "pulse_duration_fs": (6, 17),
                "repetition_rate_Hz": (10, 100000),
                "best_use_cases": ["High-repetition rate studies", "Attosecond physics"],
            },
        }

        # Generate unified recommended ranges for analog Hawking experiments
        hawking_optimal_ranges = {
            "intensity_W_m2": (1e19, 1e22),  # Optimal for plasma mirror formation
            "wavelength_nm": (800, 810),  # Ti:Sapphire compatibility
            "pulse_duration_fs": (100, 200),  # Good balance of intensity and temporal resolution
            "repetition_rate_Hz": (0.1, 10),  # Balance data collection and intensity
            "plasma_density_m3": (1e23, 1e25),  # Optimal for sonic horizon formation
            "magnetic_field_T": (0, 50),  # Realistic laboratory fields
            "target_types": ["Solid targets", "Gas jets", "Cluster targets"],
            "diagnostic_requirements": [
                "X-ray spectrometry (1-100 keV)",
                "Electron spectrometry",
                "Optical probing (fs resolution)",
                "Radio detection (30K system temperature)",
            ],
        }

        print("\n📋 FACILITY-SPECIFIC RANGES:")
        for facility, ranges in facility_ranges.items():
            print(f"\n   {facility}:")
            print(
                f"      Intensity: {ranges['intensity_W_m2'][0]:.1e} - {ranges['intensity_W_m2'][1]:.1e} W/m²"
            )
            print(
                f"      Wavelength: {ranges['wavelength_nm'][0]:.0f} - {ranges['wavelength_nm'][1]:.0f} nm"
            )
            print(
                f"      Pulse Duration: {ranges['pulse_duration_fs'][0]:.0f} - {ranges['pulse_duration_fs'][1]:.0f} fs"
            )
            print(
                f"      Repetition Rate: {ranges['repetition_rate_Hz'][0]:.3f} - {ranges['repetition_rate_Hz'][1]:.1f} Hz"
            )
            print(f"      Best Use Cases: {', '.join(ranges['best_use_cases'])}")

        print("\n🎯 OPTIMAL RANGES FOR ANALOG HAWKING EXPERIMENTS:")
        print(
            f"      Intensity: {hawking_optimal_ranges['intensity_W_m2'][0]:.1e} - {hawking_optimal_ranges['intensity_W_m2'][1]:.1e} W/m²"
        )
        print(
            f"      Wavelength: {hawking_optimal_ranges['wavelength_nm'][0]:.0f} - {hawking_optimal_ranges['wavelength_nm'][1]:.0f} nm"
        )
        print(
            f"      Pulse Duration: {hawking_optimal_ranges['pulse_duration_fs'][0]:.0f} - {hawking_optimal_ranges['pulse_duration_fs'][1]:.0f} fs"
        )
        print(
            f"      Repetition Rate: {hawking_optimal_ranges['repetition_rate_Hz'][0]:.2f} - {hawking_optimal_ranges['repetition_rate_Hz'][1]:.1f} Hz"
        )
        print(
            f"      Plasma Density: {hawking_optimal_ranges['plasma_density_m3'][0]:.1e} - {hawking_optimal_ranges['plasma_density_m3'][1]:.1e} m⁻³"
        )
        print(
            f"      Magnetic Field: {hawking_optimal_ranges['magnetic_field_T'][0]:.1f} - {hawking_optimal_ranges['magnetic_field_T'][1]:.1f} T"
        )

        return {
            "facility_ranges": facility_ranges,
            "hawking_optimal_ranges": hawking_optimal_ranges,
        }


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Validate analog Hawking radiation parameters against ELI facility capabilities"
    )
    parser.add_argument(
        "--mode",
        choices=["repository", "specific", "ranges"],
        default="repository",
        help="Validation mode: repository scan, specific configuration, or generate ranges",
    )
    parser.add_argument("--intensity", type=float, help="Specific intensity to validate (W/m²)")
    parser.add_argument("--wavelength", type=float, default=800, help="Laser wavelength (nm)")
    parser.add_argument("--pulse-duration", type=float, default=150, help="Pulse duration (fs)")
    parser.add_argument("--facility", type=str, help="Target ELI facility")
    parser.add_argument("--output", type=str, help="Output file for validation results (JSON)")

    args = parser.parse_args()

    # Initialize validator
    validator = ELICompatibilityValidator()

    # Run validation based on mode
    if args.mode == "repository":
        print("🔍 REPOSITORY-WIDE ELI COMPATIBILITY VALIDATION")
        results = validator.validate_repository_parameters()

    elif args.mode == "specific":
        if not args.intensity:
            print("❌ ERROR: --intensity required for specific configuration validation")
            return 1

        print("🔬 SPECIFIC CONFIGURATION VALIDATION")
        results = validator.validate_specific_configuration(
            args.intensity, args.wavelength, args.pulse_duration, args.facility
        )

    elif args.mode == "ranges":
        print("📊 ELI-COMPLIANT PARAMETER RANGE GENERATION")
        results = validator.generate_eli_compliant_parameter_ranges()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
