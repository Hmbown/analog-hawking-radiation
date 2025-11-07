#!/usr/bin/env python3
"""
ELI Integration Validation Script

This script integrates the ELI facility compatibility validation with the existing
analog Hawking radiation validation framework. It provides unified validation
that combines physics constraints, experimental feasibility, and facility capabilities.

Author: Claude Analysis Assistant
Date: November 2025
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.facilities.eli_capabilities import ELICapabilities, ELIFacility
from analog_hawking.facilities.eli_physics_validator import ELIPhysicsValidator
from scripts.comprehensive_eli_facility_validator import ELIAnalogHawkingValidator
from scripts.validate_eli_compatibility import ELICompatibilityValidator


class ELIIntegrationValidator:
    """
    Integrates ELI validation with existing analog Hawking radiation framework
    """

    def __init__(self):
        self.eli_validator = ELIAnalogHawkingValidator()
        self.physics_validator = ELIPhysicsValidator()
        self.compatibility_validator = ELICompatibilityValidator()
        self.eli_caps = ELICapabilities()

    def run_comprehensive_integration_validation(
        self,
        intensity_W_m2: float,
        wavelength_nm: float = 800,
        pulse_duration_fs: float = 150,
        plasma_density_m3: float = 1e25,
        facility_preference: Optional[str] = None,
        integration_mode: str = "full",
    ) -> Dict[str, Any]:
        """
        Run comprehensive integration validation combining all ELI validation modules

        Args:
            intensity_W_m2: Laser intensity in W/m²
            wavelength_nm: Laser wavelength in nm
            pulse_duration_fs: Pulse duration in fs
            plasma_density_m3: Plasma density in m⁻³
            facility_preference: Preferred ELI facility
            integration_mode: Validation mode ("full", "physics", "facility", "compatibility")

        Returns:
            Comprehensive integration validation results
        """

        print("🔗 ELI INTEGRATION VALIDATION")
        print("=" * 60)
        print(f"Integration Mode: {integration_mode}")
        print(f"Base Parameters:")
        print(f"  Intensity: {intensity_W_m2:.2e} W/m²")
        print(f"  Wavelength: {wavelength_nm:.0f} nm")
        print(f"  Pulse Duration: {pulse_duration_fs:.0f} fs")
        print(f"  Plasma Density: {plasma_density_m3:.2e} m⁻³")
        if facility_preference:
            print(f"  Preferred Facility: {facility_preference}")
        print()

        # Initialize results structure
        integration_results = {
            "metadata": {
                "validation_mode": integration_mode,
                "input_parameters": {
                    "intensity_W_m2": intensity_W_m2,
                    "wavelength_nm": wavelength_nm,
                    "pulse_duration_fs": pulse_duration_fs,
                    "plasma_density_m3": plasma_density_m3,
                    "facility_preference": facility_preference,
                },
            },
            "validation_modules": {},
            "integration_analysis": {},
            "unified_assessment": {},
            "framework_compatibility": {},
            "recommendations": [],
        }

        # Run validation modules based on integration mode
        if integration_mode in ["full", "facility"]:
            print("1️⃣ FACILITY COMPATIBILITY VALIDATION")
            facility_results = self._run_facility_validation(
                intensity_W_m2, wavelength_nm, pulse_duration_fs, facility_preference
            )
            integration_results["validation_modules"]["facility"] = facility_results

        if integration_mode in ["full", "physics"]:
            print("\n2️⃣ PHYSICS THRESHOLD VALIDATION")
            physics_results = self._run_physics_validation(
                intensity_W_m2, wavelength_nm, pulse_duration_fs, plasma_density_m3, facility_preference
            )
            integration_results["validation_modules"]["physics"] = physics_results

        if integration_mode in ["full", "compatibility"]:
            print("\n3️⃣ COMPREHENSIVE COMPATIBILITY VALIDATION")
            compatibility_results = self._run_compatibility_validation(
                intensity_W_m2, wavelength_nm, pulse_duration_fs, plasma_density_m3, facility_preference
            )
            integration_results["validation_modules"]["compatibility"] = compatibility_results

        # Perform integration analysis
        print("\n4️⃣ INTEGRATION ANALYSIS")
        integration_analysis = self._perform_integration_analysis(integration_results["validation_modules"])
        integration_results["integration_analysis"] = integration_analysis

        # Check framework compatibility
        print("\n5️⃣ FRAMEWORK COMPATIBILITY")
        framework_results = self._check_framework_compatibility(integration_results)
        integration_results["framework_compatibility"] = framework_results

        # Generate unified assessment
        print("\n6️⃣ UNIFIED ASSESSMENT")
        unified_assessment = self._generate_unified_assessment(integration_results)
        integration_results["unified_assessment"] = unified_assessment

        # Generate integrated recommendations
        print("\n7️⃣ INTEGRATED RECOMMENDATIONS")
        recommendations = self._generate_integrated_recommendations(integration_results)
        integration_results["recommendations"] = recommendations

        return integration_results

    def _run_facility_validation(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        facility_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run facility compatibility validation"""

        # Use comprehensive ELI validator
        facility_result = self.eli_validator.validate_specific_configuration(
            intensity_W_m2, wavelength_nm, pulse_duration_fs, facility_preference
        )

        # Generate facility-specific configurations
        facility_configs = self.eli_validator.generate_eli_compliant_parameter_ranges()

        return {
            "configuration_validation": facility_result,
            "facility_configurations": facility_configs,
            "compatible_systems": self._get_compatible_systems_summary(
                intensity_W_m2, wavelength_nm, pulse_duration_fs
            ),
        }

    def _run_physics_validation(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        plasma_density_m3: float,
        facility_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run physics threshold validation"""

        # Determine target facility
        target_facility = None
        if facility_preference:
            facility_map = {
                "eli-beamlines": ELIFacility.ELI_BEAMLINES,
                "eli-np": ELIFacility.ELI_NP,
                "eli-alps": ELIFacility.ELI_ALPS,
                "beamlines": ELIFacility.ELI_BEAMLINES,
                "np": ELIFacility.ELI_NP,
                "alps": ELIFacility.ELI_ALPS,
            }
            target_facility = facility_map.get(facility_preference.lower())

        if not target_facility:
            # Use best available facility
            feasibility = self.eli_caps.calculate_feasibility_score(
                intensity_W_m2 / 1e4, wavelength_nm, pulse_duration_fs
            )
            if feasibility["feasible"]:
                facility_name = feasibility["facility"]
                target_facility = ELIFacility(facility_name.lower().replace("-", "_"))

        if target_facility:
            physics_result = self.physics_validator.validate_comprehensive_configuration(
                intensity_W_m2,
                wavelength_nm,
                pulse_duration_fs,
                plasma_density_m3,
                1e-6,  # Assume 1 μm gradient scale
                2e6,   # Assume 2×10⁶ m/s flow velocity
                target_facility,
            )
        else:
            physics_result = {"error": "No suitable facility found for physics validation"}

        return {
            "facility_physics_validation": physics_result,
            "threshold_analysis": self._analyze_threshold_compliance(intensity_W_m2, plasma_density_m3),
            "breakdown_risks": self._assess_breakdown_risks(intensity_W_m2, wavelength_nm, plasma_density_m3),
        }

    def _run_compatibility_validation(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        plasma_density_m3: float,
        facility_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run comprehensive compatibility validation"""

        # Use full configuration validation
        compatibility_result = self.eli_validator.validate_full_configuration(
            intensity_W_m2,
            wavelength_nm,
            pulse_duration_fs,
            plasma_density_m3,
            "solid",  # Default target type
            facility_preference,
            include_hawking_analysis=True,
        )

        return {
            "full_configuration_validation": compatibility_result,
            "experimental_feasibility": self._assess_experimental_feasibility(compatibility_result),
            "detection_prospects": self._evaluate_detection_prospects(compatibility_result),
        }

    def _get_compatible_systems_summary(
        self, intensity_W_m2: float, wavelength_nm: float, pulse_duration_fs: float
    ) -> Dict[str, Any]:
        """Get summary of compatible ELI systems"""

        compatible = self.eli_caps.get_compatible_systems(
            intensity_W_m2 / 1e4, wavelength_nm, pulse_duration_fs
        )

        summary = {
            "total_compatible": len(compatible),
            "systems_by_facility": {},
            "best_system": None,
            "facility_coverage": [],
        }

        for system in compatible:
            facility = system.facility.value
            if facility not in summary["systems_by_facility"]:
                summary["systems_by_facility"][facility] = []
            summary["systems_by_facility"][facility].append({
                "name": system.name,
                "peak_power_TW": system.peak_power_TW,
                "rep_rate_Hz": system.repetition_rate_Hz,
                "operational_status": system.operational_status,
            })

        if compatible:
            summary["best_system"] = compatible[0].name

        # Determine facility coverage
        facilities_with_systems = list(summary["systems_by_facility"].keys())
        summary["facility_coverage"] = facilities_with_systems

        return summary

    def _analyze_threshold_compliance(self, intensity_W_m2: float, plasma_density_m3: float) -> Dict[str, Any]:
        """Analyze compliance with physics thresholds"""

        # Load thresholds
        thresholds_path = Path(__file__).parent.parent / "configs" / "thresholds.yaml"
        try:
            import yaml
            with open(thresholds_path, 'r') as f:
                thresholds = yaml.safe_load(f)
        except:
            thresholds = {
                "v_max_fraction_c": 0.5,
                "dv_dx_max_s": 4.0e12,
                "intensity_max_W_m2": 1.0e24,
            }

        compliance = {
            "intensity_compliance": {
                "value": intensity_W_m2,
                "limit": thresholds["intensity_max_W_m2"],
                "compliant": intensity_W_m2 <= thresholds["intensity_max_W_m2"],
                "margin": thresholds["intensity_max_W_m2"] / intensity_W_m2 if intensity_W_m2 > 0 else float('inf'),
            },
            "overall_compliance": True,
            "critical_violations": [],
        }

        if not compliance["intensity_compliance"]["compliant"]:
            compliance["overall_compliance"] = False
            compliance["critical_violations"].append(
                f"Intensity exceeds threshold by factor {intensity_W_m2/thresholds['intensity_max_W_m2']:.1f}"
            )

        return compliance

    def _assess_breakdown_risks(
        self, intensity_W_m2: float, wavelength_nm: float, plasma_density_m3: float
    ) -> Dict[str, Any]:
        """Assess physics breakdown risks"""

        # Calculate key parameters
        a0 = 0.85 * np.sqrt(intensity_W_m2 / 1e22) * (wavelength_nm / 800)  # Normalized to 1e22 W/m², 800nm

        risks = {
            "relativistic_effects": {
                "a0_parameter": a0,
                "risk_level": "low" if a0 < 1 else "medium" if a0 < 10 else "high",
                "description": f"a₀ = {a0:.2f}",
            },
            "radiation_pressure": {
                "risk_level": "low" if a0 < 10 else "medium" if a0 < 50 else "high",
                "description": "Radiation pressure dominance",
            },
            "quantum_effects": {
                "chi_parameter": intensity_W_m2 / 1e29,  # Normalized quantum parameter
                "risk_level": "low" if intensity_W_m2 < 1e27 else "medium" if intensity_W_m2 < 1e29 else "high",
                "description": "Quantum electrodynamics effects",
            },
            "overall_risk": "low",
        }

        # Determine overall risk
        risk_levels = [risks[key]["risk_level"] for key in risks if key != "overall_risk"]
        if "high" in risk_levels:
            risks["overall_risk"] = "high"
        elif "medium" in risk_levels:
            risks["overall_risk"] = "medium"
        else:
            risks["overall_risk"] = "low"

        return risks

    def _assess_experimental_feasibility(self, compatibility_result: Dict[str, Any]) -> Dict[str, Any]:
        """Assess experimental feasibility"""

        overall = compatibility_result.get("overall_assessment", {})

        feasibility = {
            "overall_feasibility": overall.get("overall_feasibility", "UNKNOWN"),
            "confidence_level": overall.get("confidence_level", 0.0),
            "key_factors": {
                "facility_compatibility": len(overall.get("key_strengths", [])) > 0,
                "physics_validity": len(overall.get("key_challenges", [])) == 0,
                "detection_prospects": compatibility_result.get("hawking_analysis", {}).get("detection_feasible", False),
            },
            "success_probability": overall.get("confidence_level", 0.0),
        }

        return feasibility

    def _evaluate_detection_prospects(self, compatibility_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate Hawking radiation detection prospects"""

        hawking_analysis = compatibility_result.get("hawking_analysis", {})

        prospects = {
            "detection_feasible": hawking_analysis.get("detection_feasible", False),
            "estimated_kappa": hawking_analysis.get("surface_gravity_estimate_Hz", None),
            "estimated_temperature": hawking_analysis.get("hawking_temperature_K", None),
            "detection_time": hawking_analysis.get("estimated_detection_time_s", None),
            "signal_quality": "unknown",
        }

        if prospects["estimated_kappa"] and prospects["estimated_temperature"]:
            # Simple signal quality assessment
            if prospects["estimated_kappa"] > 1e12 and prospects["estimated_temperature"] > 1e-8:
                prospects["signal_quality"] = "good"
            elif prospects["estimated_kappa"] > 1e11 and prospects["estimated_temperature"] > 1e-9:
                prospects["signal_quality"] = "moderate"
            else:
                prospects["signal_quality"] = "challenging"

        return prospects

    def _perform_integration_analysis(self, validation_modules: Dict[str, Any]) -> Dict[str, Any]:
        """Perform integration analysis across validation modules"""

        analysis = {
            "module_consistency": {},
            "conflicting_results": [],
            "complementary_insights": [],
            "overall_integration_score": 0.0,
        }

        # Check consistency between modules
        module_results = {}
        for module_name, module_data in validation_modules.items():
            if module_name == "facility":
                feasibility = module_data.get("configuration_validation", {}).get("feasible", False)
                module_results[module_name] = {"feasible": feasibility}
            elif module_name == "physics":
                physics_passed = module_data.get("facility_physics_validation", {}).get("overall_passed", False)
                module_results[module_name] = {"physics_passed": physics_passed}
            elif module_name == "compatibility":
                overall_feasible = module_data.get("full_configuration_validation", {}).get(
                    "overall_assessment", {}
                ).get("overall_feasibility") == "HIGH"
                module_results[module_name] = {"overall_feasible": overall_feasible}

        analysis["module_consistency"] = module_results

        # Identify conflicting results
        feasibilities = [result.get("feasible", result.get("physics_passed", result.get("overall_feasible", False)))
                        for result in module_results.values()]

        if not all(feasibilities) and any(feasibilities):
            analysis["conflicting_results"].append("Mixed feasibility results across validation modules")

        # Calculate integration score
        if feasibilities:
            analysis["overall_integration_score"] = sum(feasibilities) / len(feasibilities)

        return analysis

    def _check_framework_compatibility(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check compatibility with existing analog Hawking radiation framework"""

        compatibility = {
            "physics_engine_compatibility": True,
            "validation_framework_compatibility": True,
            "threshold_system_compatibility": True,
            "diagnostic_compatibility": True,
            "overall_framework_compatibility": True,
            "integration_notes": [],
        }

        # Check if results are compatible with framework expectations
        validation_modules = integration_results.get("validation_modules", {})

        # Physics engine compatibility
        if "physics" in validation_modules:
            physics_result = validation_modules["physics"].get("facility_physics_validation", {})
            if not physics_result.get("overall_passed", False):
                compatibility["physics_engine_compatibility"] = False
                compatibility["integration_notes"].append("Physics validation reveals framework incompatibilities")

        # Threshold system compatibility
        if "physics" in validation_modules:
            threshold_analysis = validation_modules["physics"].get("threshold_analysis", {})
            if not threshold_analysis.get("overall_compliance", False):
                compatibility["threshold_system_compatibility"] = False
                compatibility["integration_notes"].append("Threshold violations detected")

        # Overall framework compatibility
        if compatibility["integration_notes"]:
            compatibility["overall_framework_compatibility"] = False

        return compatibility

    def _generate_unified_assessment(self, integration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate unified assessment from all validation results"""

        integration_analysis = integration_results.get("integration_analysis", {})
        framework_compatibility = integration_results.get("framework_compatibility", {})
        validation_modules = integration_results.get("validation_modules", {})

        assessment = {
            "overall_status": "READY",
            "confidence_level": 0.0,
            "readiness_indicators": {
                "facility_ready": False,
                "physics_ready": False,
                "framework_ready": False,
            },
            "key_readiness_factors": [],
            "remaining_obstacles": [],
        }

        # Determine facility readiness
        if "facility" in validation_modules:
            facility_feasible = validation_modules["facility"].get("configuration_validation", {}).get("feasible", False)
            assessment["readiness_indicators"]["facility_ready"] = facility_feasible

        # Determine physics readiness
        if "physics" in validation_modules:
            physics_passed = validation_modules["physics"].get("facility_physics_validation", {}).get("overall_passed", False)
            assessment["readiness_indicators"]["physics_ready"] = physics_passed

        # Determine framework readiness
        assessment["readiness_indicators"]["framework_ready"] = framework_compatibility.get("overall_framework_compatibility", False)

        # Calculate overall confidence
        readiness_values = list(assessment["readiness_indicators"].values())
        if readiness_values:
            assessment["confidence_level"] = sum(readiness_values) / len(readiness_values)

        # Determine overall status
        if assessment["confidence_level"] > 0.8:
            assessment["overall_status"] = "READY"
        elif assessment["confidence_level"] > 0.5:
            assessment["overall_status"] = "CONDITIONAL"
        else:
            assessment["overall_status"] = "NOT_READY"

        # Identify readiness factors and obstacles
        for indicator, ready in assessment["readiness_indicators"].items():
            if ready:
                assessment["key_readiness_factors"].append(f"{indicator.replace('_', ' ').title()} validated")
            else:
                assessment["remaining_obstacles"].append(f"{indicator.replace('_', ' ').title()} requires attention")

        return assessment

    def _generate_integrated_recommendations(self, integration_results: Dict[str, Any]) -> List[str]:
        """Generate integrated recommendations from all validation results"""

        recommendations = []
        unified_assessment = integration_results.get("unified_assessment", {})
        validation_modules = integration_results.get("validation_modules", {})

        # Status-based recommendations
        status = unified_assessment.get("overall_status", "UNKNOWN")

        if status == "READY":
            recommendations.append("✅ Configuration validated and ready for experimental proposal")
            recommendations.append("✅ Proceed with detailed planning and beam time application")
        elif status == "CONDITIONAL":
            recommendations.append("⚠️  Configuration partially ready - address remaining obstacles")
            recommendations.append("⚠️  Consider parameter optimization before final proposal")
        else:
            recommendations.append("❌ Configuration not ready - significant modifications required")
            recommendations.append("❌  Reconsider experimental approach or parameter ranges")

        # Module-specific recommendations
        for module_name, module_data in validation_modules.items():
            if module_name == "facility":
                feasibility = module_data.get("configuration_validation", {}).get("feasible", False)
                if not feasibility:
                    recommendations.append(f"🏢 Facility: {module_data.get('configuration_validation', {}).get('primary_issues', ['Unknown issues'])[0]}")

            elif module_name == "physics":
                physics_passed = module_data.get("facility_physics_validation", {}).get("overall_passed", False)
                if not physics_passed:
                    recommendations.append("⚛️  Physics: Address threshold violations and breakdown risks")

            elif module_name == "compatibility":
                feasibility = module_data.get("full_configuration_validation", {}).get("overall_assessment", {}).get("overall_feasibility")
                if feasibility != "HIGH":
                    recommendations.append("🔬 Compatibility: Optimize experimental parameters for better feasibility")

        # Framework integration recommendations
        framework_compatibility = integration_results.get("framework_compatibility", {})
        if not framework_compatibility.get("overall_framework_compatibility", True):
            recommendations.extend(framework_compatibility.get("integration_notes", []))

        # Add confidence-based guidance
        confidence = unified_assessment.get("confidence_level", 0.0)
        if confidence > 0.8:
            recommendations.append("📊 High confidence: Proceed with detailed experimental planning")
        elif confidence > 0.5:
            recommendations.append("📊 Medium confidence: Address specific concerns before proceeding")
        else:
            recommendations.append("📊 Low confidence: Major revisions needed before experimental consideration")

        return recommendations


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Integrate ELI validation with analog Hawking radiation framework"
    )
    parser.add_argument("--intensity", type=float, default=1e22, help="Laser intensity (W/m²)")
    parser.add_argument("--wavelength", type=float, default=800, help="Laser wavelength (nm)")
    parser.add_argument("--pulse-duration", type=float, default=150, help="Pulse duration (fs)")
    parser.add_argument("--plasma-density", type=float, default=1e25, help="Plasma density (m⁻³)")
    parser.add_argument("--facility", type=str, help="Preferred ELI facility")
    parser.add_argument("--mode", choices=["full", "physics", "facility", "compatibility"],
                       default="full", help="Integration validation mode")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")

    args = parser.parse_args()

    # Initialize integration validator
    validator = ELIIntegrationValidator()

    # Run integration validation
    results = validator.run_comprehensive_integration_validation(
        args.intensity,
        args.wavelength,
        args.pulse_duration,
        args.plasma_density,
        args.facility,
        args.mode,
    )

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())