#!/usr/bin/env python3
"""
Comprehensive ELI Facility Parameter Validation for Analog Hawking Radiation Analysis

This script provides detailed validation of laser-plasma experiment configurations
against the real capabilities of ELI facilities as of 2024-2025. It integrates
with the existing physics validation framework and provides facility-specific
recommendations for analog Hawking radiation experiments.

Author: Claude Analysis Assistant
Date: November 2025
Version: 1.0.0
"""

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

warnings.filterwarnings("ignore")

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.facilities.eli_capabilities import (
    ELICapabilities,
    ELIFacility,
    validate_intensity_range,
)
from analog_hawking.physics_engine import plasma_mirror


class ELIAnalogHawkingValidator:
    """
    Comprehensive validator for analog Hawking radiation experiments at ELI facilities
    """

    def __init__(self):
        self.eli_caps = ELICapabilities()
        self.thresholds = self._load_thresholds()

        # Physics constraints specific to analog Hawking experiments
        self.hawking_constraints = {
            "min_kappa_Hz": 1e9,  # Minimum surface gravity for detectable signal
            "max_kappa_Hz": 1e14,  # Maximum before breakdown (from thresholds.yaml)
            "optimal_kappa_Hz": (1e11, 1e13),  # Optimal range for detection
            "min_plasma_density_m3": 1e23,  # Critical density for 800nm
            "max_plasma_density_m3": 1e27,  # Before relativistic effects dominate
            "optimal_density_range_m3": (1e24, 1e26),  # For good horizon formation
            "min_velocity_c": 0.1,  # Minimum flow velocity fraction of c
            "max_velocity_c": 0.5,  # From thresholds.yaml
            "optimal_velocity_range_c": (0.2, 0.4),  # For good horizon dynamics
        }

    def _load_thresholds(self) -> Dict[str, Any]:
        """Load physics thresholds from configuration"""
        try:
            thresholds_path = Path(__file__).parent.parent / "configs" / "thresholds.yaml"
            with open(thresholds_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load thresholds.yaml: {e}")
            return {
                "v_max_fraction_c": 0.5,
                "dv_dx_max_s": 4.0e12,
                "intensity_max_W_m2": 1.0e24,
                "system_temperature_K": 50,
                "bandwidth_Hz": 1.0e9,
            }

    def validate_full_configuration(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        plasma_density_m3: float,
        target_type: str = "solid",
        facility_preference: Optional[str] = None,
        include_hawking_analysis: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of a complete experimental configuration

        Args:
            intensity_W_m2: Laser intensity in W/m²
            wavelength_nm: Laser wavelength in nm
            pulse_duration_fs: Pulse duration in femtoseconds
            plasma_density_m3: Plasma density in m⁻³
            target_type: Type of target (solid, gas_jet, cluster)
            facility_preference: Preferred ELI facility
            include_hawking_analysis: Whether to include Hawking radiation analysis

        Returns:
            Comprehensive validation report
        """

        print("🔬 COMPREHENSIVE ELI FACILITY VALIDATION")
        print("=" * 60)
        print(f"Intensity: {intensity_W_m2:.2e} W/m²")
        print(f"Wavelength: {wavelength_nm:.0f} nm")
        print(f"Pulse Duration: {pulse_duration_fs:.0f} fs")
        print(f"Plasma Density: {plasma_density_m3:.2e} m⁻³")
        print(f"Target Type: {target_type}")
        if facility_preference:
            print(f"Preferred Facility: {facility_preference}")
        print()

        # Initialize validation results
        validation_results = {
            "input_parameters": {
                "intensity_W_m2": intensity_W_m2,
                "wavelength_nm": wavelength_nm,
                "pulse_duration_fs": pulse_duration_fs,
                "plasma_density_m3": plasma_density_m3,
                "target_type": target_type,
                "facility_preference": facility_preference,
            },
            "facility_validation": {},
            "physics_validation": {},
            "hawking_analysis": {},
            "plasma_mirror_analysis": {},
            "overall_assessment": {},
            "recommendations": [],
        }

        # 1. Facility Compatibility Validation
        print("1️⃣ FACILITY COMPATIBILITY VALIDATION")
        facility_validation = self._validate_facility_compatibility(
            intensity_W_m2, wavelength_nm, pulse_duration_fs, facility_preference
        )
        validation_results["facility_validation"] = facility_validation

        # 2. Physics Threshold Validation
        print("\n2️⃣ PHYSICS THRESHOLD VALIDATION")
        physics_validation = self._validate_physics_thresholds(
            intensity_W_m2, plasma_density_m3, wavelength_nm, pulse_duration_fs
        )
        validation_results["physics_validation"] = physics_validation

        # 3. Plasma Mirror Analysis
        print("\n3️⃣ PLASMA MIRROR ANALYSIS")
        plasma_analysis = self._analyze_plasma_mirror_requirements(
            intensity_W_m2, wavelength_nm, pulse_duration_fs, target_type
        )
        validation_results["plasma_mirror_analysis"] = plasma_analysis

        # 4. Hawking Radiation Feasibility (if requested)
        if include_hawking_analysis:
            print("\n4️⃣ HAWKING RADIATION FEASIBILITY")
            hawking_analysis = self._analyze_hawking_feasibility(
                intensity_W_m2, plasma_density_m3, wavelength_nm
            )
            validation_results["hawking_analysis"] = hawking_analysis

        # 5. Overall Assessment and Recommendations
        print("\n5️⃣ OVERALL ASSESSMENT")
        overall_assessment = self._generate_overall_assessment(validation_results)
        validation_results["overall_assessment"] = overall_assessment

        # 6. Generate Recommendations
        print("\n6️⃣ RECOMMENDATIONS")
        recommendations = self._generate_comprehensive_recommendations(validation_results)
        validation_results["recommendations"] = recommendations

        return validation_results

    def _validate_facility_compatibility(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        facility_preference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Validate against ELI facility capabilities"""

        # Convert to W/cm² for ELI validation
        intensity_W_cm2 = intensity_W_m2 / 1e4

        # Determine target facility
        target_facility = None
        if facility_preference:
            try:
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
                    print(f"   ⚠️  Unknown facility preference: {facility_preference}")
                    print(f"   🔄 Validating against all ELI facilities")
            except Exception as e:
                print(f"   ⚠️  Error parsing facility preference: {e}")

        # Get feasibility assessment
        feasibility = self.eli_caps.calculate_feasibility_score(
            intensity_W_cm2, wavelength_nm, pulse_duration_fs, target_facility
        )

        # Format results
        if feasibility["feasible"]:
            print(f"   ✅ FEASIBLE: Score {feasibility['score']:.2f}/1.00")
            print(f"   🏢 Best System: {feasibility['best_system']} ({feasibility['facility']})")
            print(f"   📍 Experimental Hall: {feasibility['experimental_hall']}")
            print(f"   ⚡ Repetition Rate: {feasibility['repetition_rate_Hz']:.1f} Hz")
            print(f"   📈 Intensity Margin: {feasibility['intensity_margin']:.1f}x")

            if feasibility["all_compatible_systems"]:
                print(f"   🔧 Alternative Systems: {', '.join(feasibility['all_compatible_systems'][1:])}")
        else:
            print(f"   ❌ INFEASIBLE:")
            for issue in feasibility["primary_issues"]:
                print(f"      • {issue}")

        return feasibility

    def _validate_physics_thresholds(
        self,
        intensity_W_m2: float,
        plasma_density_m3: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
    ) -> Dict[str, Any]:
        """Validate against physics breakdown thresholds"""

        validation_results = {
            "threshold_checks": {},
            "critical_issues": [],
            "warnings": [],
            "passed_all_checks": True,
        }

        # Calculate derived parameters
        a0 = self._calculate_relativistic_parameter(intensity_W_m2, wavelength_nm)
        critical_power_GW = self._calculate_critical_power(plasma_density_m3, wavelength_nm)

        print(f"   📊 Derived Parameters:")
        print(f"      Relativistic parameter a₀: {a0:.2f}")
        print(f"      Critical power P_c: {critical_power_GW:.1f} GW")

        # Check each threshold
        checks = []

        # 1. Intensity threshold
        max_intensity = float(self.thresholds["intensity_max_W_m2"])
        intensity_check = {
            "parameter": "Intensity",
            "value": intensity_W_m2,
            "limit": max_intensity,
            "unit": "W/m²",
            "passed": intensity_W_m2 <= max_intensity,
        }
        checks.append(intensity_check)

        if intensity_check["passed"]:
            print(f"   ✅ Intensity: {intensity_W_m2:.2e} W/m² ≤ {max_intensity:.2e} W/m²")
        else:
            print(f"   ❌ Intensity: {intensity_W_m2:.2e} W/m² > {max_intensity:.2e} W/m²")
            validation_results["critical_issues"].append(
                f"Intensity exceeds maximum threshold by factor {intensity_W_m2/max_intensity:.1f}"
            )
            validation_results["passed_all_checks"] = False

        # 2. Relativistic parameter check
        if a0 > 10:
            print(f"   ⚠️  Strongly relativistic regime (a₀ = {a0:.1f})")
            validation_results["warnings"].append(
                "Strong relativistic effects may invalidate fluid assumptions"
            )
        elif a0 > 1:
            print(f"   ✅ Moderate relativistic regime (a₀ = {a0:.1f})")
        else:
            print(f"   ✅ Weakly relativistic regime (a₀ = {a0:.1f})")

        # 3. Critical power check
        # Estimate laser power from intensity and focal spot
        focal_spot_m = 3e-6  # Assume 3 μm focal spot
        laser_power_GW = intensity_W_m2 * np.pi * (focal_spot_m/2)**2 / 1e9

        if laser_power_GW > critical_power_GW:
            print(f"   ⚠️  Power exceeds critical: {laser_power_GW:.1f} GW > {critical_power_GW:.1f} GW")
            print(f"      Expected self-focusing and channel formation")
            validation_results["warnings"].append("Laser power above critical for self-focusing")
        else:
            print(f"   ✅ Power below critical: {laser_power_GW:.1f} GW < {critical_power_GW:.1f} GW")

        # 4. Plasma density optimization
        critical_density = self._calculate_critical_density(wavelength_nm)
        density_ratio = plasma_density_m3 / critical_density

        if 0.1 <= density_ratio <= 10:
            print(f"   ✅ Plasma density: {density_ratio:.2f} × n_c (optimal range)")
        else:
            print(f"   ⚠️  Plasma density: {density_ratio:.2f} × n_c (outside optimal range)")
            validation_results["warnings"].append(
                f"Density ratio {density_ratio:.2f} may affect plasma mirror formation"
            )

        validation_results["threshold_checks"] = {
            "a0": a0,
            "critical_power_GW": critical_power_GW,
            "laser_power_GW": laser_power_GW,
            "density_ratio": density_ratio,
            "critical_density_m3": critical_density,
        }

        return validation_results

    def _analyze_plasma_mirror_requirements(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        target_type: str,
    ) -> Dict[str, Any]:
        """Analyze plasma mirror formation and requirements"""

        analysis = {
            "mirror_formation_feasible": False,
            "formation_time_fs": None,
            "reflectivity_estimate": None,
            "expansion_velocity_ms": None,
            "optimal_timing_fs": None,
            "recommendations": [],
        }

        # Calculate plasma mirror parameters
        intensity_W_cm2 = intensity_W_m2 / 1e4

        # Ionization threshold considerations
        if intensity_W_cm2 > 1e14:  # Typical ionization threshold
            print(f"   ✅ Intensity sufficient for plasma mirror formation")
            analysis["mirror_formation_feasible"] = True

            # Estimate formation time (simplified model)
            formation_time_fs = 10 * (1e14 / intensity_W_cm2)**0.5
            analysis["formation_time_fs"] = formation_time_fs
            print(f"   ⏱️  Estimated formation time: {formation_time_fs:.1f} fs")

            # Estimate expansion velocity
            expansion_velocity_ms = 1e6 * (intensity_W_cm2 / 1e16)**0.5
            analysis["expansion_velocity_ms"] = expansion_velocity_ms
            print(f"   🚀 Expansion velocity: {expansion_velocity_ms:.2e} m/s")

            # Estimate reflectivity (simplified model)
            # For high-intensity lasers, reflectivity can approach 70-80%
            reflectivity = 0.7 * (1 - np.exp(-intensity_W_cm2 / 1e16))
            reflectivity = min(reflectivity, 0.85)  # Cap at 85%
            analysis["reflectivity_estimate"] = reflectivity
            print(f"   🔮 Estimated reflectivity: {reflectivity*100:.1f}%")

            # Optimal timing for main pulse interaction
            # Need to balance mirror formation with expansion
            optimal_delay = formation_time_fs + pulse_duration_fs * 0.1
            analysis["optimal_timing_fs"] = optimal_delay
            print(f"   ⏰ Optimal main pulse delay: {optimal_delay:.1f} fs")

            # Target-specific considerations
            if target_type == "solid":
                print(f"   🎯 Solid target: Good for plasma mirror formation")
                analysis["recommendations"].append("Use polished optical quality surface")
                analysis["recommendations"].append("Consider pre-pulse cleaning with plasma mirror")
            elif target_type == "gas_jet":
                print(f"   🎯 Gas jet: Challenging for plasma mirror formation")
                analysis["recommendations"].append("Use high-density gas jet or cluster target")
                analysis["recommendations"].append("Consider double-pulse scheme for mirror formation")
            elif target_type == "cluster":
                print(f"   🎯 Cluster target: Good for high-density plasma formation")
                analysis["recommendations"].append("Optimize cluster size for target density")

        else:
            print(f"   ❌ Intensity insufficient for reliable plasma mirror formation")
            analysis["recommendations"].append(
                f"Increase intensity to ≥1e18 W/m² for plasma mirror formation"
            )

        return analysis

    def _analyze_hawking_feasibility(
        self,
        intensity_W_m2: float,
        plasma_density_m3: float,
        wavelength_nm: float,
    ) -> Dict[str, Any]:
        """Analyze feasibility of analog Hawking radiation detection"""

        analysis = {
            "surface_gravity_estimate_Hz": None,
            "hawking_temperature_K": None,
            "detection_feasible": False,
            "estimated_detection_time_s": None,
            "signal_to_noise_ratio": None,
            "limitations": [],
            "optimization_suggestions": [],
        }

        # Simplified estimates for demonstration
        # In a real implementation, these would come from the full physics engine

        # Estimate surface gravity κ based on intensity and plasma parameters
        # This is a very simplified model - real calculation requires full fluid dynamics
        intensity_W_cm2 = intensity_W_m2 / 1e4

        # Rough scaling: κ increases with intensity and optimal density
        kappa_Hz = 1e11 * (intensity_W_cm2 / 1e18)**0.5 * (plasma_density_m3 / 1e25)**0.3
        analysis["surface_gravity_estimate_Hz"] = kappa_Hz

        # Calculate Hawking temperature
        h_bar = 1.054571817e-34  # J·s
        k_B = 1.380649e-23  # J/K
        T_H_K = h_bar * kappa_Hz / (2 * np.pi * k_B)
        analysis["hawking_temperature_K"] = T_H_K

        print(f"   🌡️  Estimated Hawking temperature: {T_H_K*1e6:.2f} μK")
        print(f"   🌀 Surface gravity κ: {kappa_Hz:.2e} Hz")

        # Check if within optimal range
        min_kappa, max_kappa = self.hawking_constraints["optimal_kappa_Hz"]
        if min_kappa <= kappa_Hz <= max_kappa:
            print(f"   ✅ κ within optimal range for detection")
            analysis["detection_feasible"] = True
        elif kappa_Hz < min_kappa:
            print(f"   ⚠️  κ below optimal range - weak signal expected")
            analysis["limitations"].append("Low surface gravity results in weak Hawking signal")
        else:
            print(f"   ⚠️  κ above optimal range - breakdown effects likely")
            analysis["limitations"].append("High surface gravity may trigger physics breakdown")

        # Estimate detection requirements
        if analysis["detection_feasible"]:
            # Simplified detection time estimate
            # Real calculation would include graybody factors, detector efficiency, etc.
            system_temp_K = float(self.thresholds["system_temperature_K"])
            bandwidth_Hz = float(self.thresholds["bandwidth_Hz"])

            # Very rough SNR estimate
            snr = T_H_K / system_temp_K * np.sqrt(bandwidth_Hz)  # Simplified
            analysis["signal_to_noise_ratio"] = snr

            # Detection time for 5σ confidence
            if snr > 0:
                detection_time_s = 25 / (snr**2)  # t = (5/σ)²
                analysis["estimated_detection_time_s"] = detection_time_s
                print(f"   📡 Estimated 5σ detection time: {detection_time_s:.2e} s")

                if detection_time_s < 1:
                    print(f"   ✅ Excellent detection prospects")
                elif detection_time_s < 3600:
                    print(f"   ✅ Good detection prospects")
                else:
                    print(f"   ⚠️  Challenging detection - long integration required")
                    analysis["limitations"].append("Long integration time may be impractical")

        # Generate optimization suggestions
        if kappa_Hz < min_kappa:
            analysis["optimization_suggestions"].append(
                "Increase laser intensity to boost surface gravity"
            )
            analysis["optimization_suggestions"].append(
                "Optimize plasma density profile for stronger gradients"
            )

        if T_H_K < 1e-9:  # Less than 1 nK
            analysis["optimization_suggestions"].append(
                "Consider alternative detection schemes (e.g., frequency mixing)"
            )

        return analysis

    def _generate_overall_assessment(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall feasibility assessment"""

        assessment = {
            "overall_feasibility": "UNKNOWN",
            "confidence_level": 0.0,
            "key_strengths": [],
            "key_challenges": [],
            "facility_recommendation": None,
            "readiness_level": "TRL 1-3",  # Technology Readiness Level
        }

        # Analyze facility validation
        facility_valid = validation_results["facility_validation"].get("feasible", False)
        facility_score = validation_results["facility_validation"].get("score", 0.0)

        # Analyze physics validation
        physics_passed = validation_results["physics_validation"].get("passed_all_checks", False)
        physics_warnings = len(validation_results["physics_validation"].get("warnings", []))

        # Analyze Hawking analysis
        hawking_feasible = validation_results.get("hawking_analysis", {}).get("detection_feasible", False)

        # Calculate overall confidence
        confidence_factors = []
        if facility_valid:
            confidence_factors.append(facility_score)
        if physics_passed:
            confidence_factors.append(0.8 - 0.1 * physics_warnings)  # Reduce confidence for warnings
        if hawking_feasible:
            confidence_factors.append(0.7)

        if confidence_factors:
            assessment["confidence_level"] = np.mean(confidence_factors)

        # Determine overall feasibility
        if facility_valid and physics_passed and hawking_feasible:
            if assessment["confidence_level"] > 0.7:
                assessment["overall_feasibility"] = "HIGH"
                assessment["readiness_level"] = "TRL 4-6"
            else:
                assessment["overall_feasibility"] = "MEDIUM"
                assessment["readiness_level"] = "TRL 3-5"
        elif facility_valid and physics_passed:
            assessment["overall_feasibility"] = "MEDIUM"
            assessment["readiness_level"] = "TRL 3-4"
        else:
            assessment["overall_feasibility"] = "LOW"
            assessment["readiness_level"] = "TRL 1-3"

        # Identify strengths and challenges
        if facility_valid:
            assessment["key_strengths"].append("Compatible with ELI facilities")
            assessment["facility_recommendation"] = validation_results["facility_validation"]["facility"]

        if physics_passed:
            assessment["key_strengths"].append("Within physics breakdown thresholds")

        if not facility_valid:
            assessment["key_challenges"].append("Facility compatibility issues")

        if not physics_passed:
            assessment["key_challenges"].append("Physics threshold violations")

        if not hawking_feasible:
            assessment["key_challenges"].append("Hawking signal detection challenging")

        return assessment

    def _generate_comprehensive_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Generate comprehensive recommendations"""

        recommendations = []

        # Facility recommendations
        facility_result = validation_results["facility_validation"]
        if facility_result.get("feasible"):
            facility_rec = facility_result.get("recommendations", [])
            recommendations.extend([f"Facility: {rec}" for rec in facility_rec])
        else:
            recommendations.append("Reduce laser intensity to meet facility constraints")
            recommendations.append("Consider alternative ELI facility with higher capability")

        # Physics recommendations
        physics_result = validation_results["physics_validation"]
        if physics_result.get("warnings"):
            for warning in physics_result["warnings"]:
                recommendations.append(f"Physics: {warning}")

        # Plasma mirror recommendations
        plasma_result = validation_results["plasma_mirror_analysis"]
        if plasma_result.get("recommendations"):
            recommendations.extend([f"Plasma Mirror: {rec}" for rec in plasma_result["recommendations"]])

        # Hawking analysis recommendations
        hawking_result = validation_results.get("hawking_analysis", {})
        if hawking_result.get("optimization_suggestions"):
            recommendations.extend([f"Hawking Detection: {rec}" for rec in hawking_result["optimization_suggestions"]])

        # General recommendations
        overall = validation_results["overall_assessment"]
        if overall["overall_feasibility"] == "HIGH":
            recommendations.append("✅ Proceed with detailed experimental proposal development")
            recommendations.append("✅ Contact facility for beam time application")
        elif overall["overall_feasibility"] == "MEDIUM":
            recommendations.append("⚠️  Address key challenges before proceeding")
            recommendations.append("⚠️  Consider proof-of-concept experiments at lower intensity")
        else:
            recommendations.append("❌ Significant modifications required for feasibility")
            recommendations.append("❌ Reconsider experimental approach or parameter ranges")

        return recommendations

    def _calculate_relativistic_parameter(self, intensity_W_m2: float, wavelength_nm: float) -> float:
        """Calculate normalized vector potential a₀"""
        # a₀ = 0.85 * sqrt(I[10¹⁸ W/cm²] * λ[μm]²)
        intensity_18 = intensity_W_m2 / 1e4 / 1e18  # Convert to 10¹⁸ W/cm²
        wavelength_um = wavelength_nm / 1000  # Convert to μm
        a0 = 0.85 * np.sqrt(intensity_18) * wavelength_um
        return a0

    def _calculate_critical_power(self, plasma_density_m3: float, wavelength_nm: float) -> float:
        """Calculate critical power for self-focusing in GW"""
        # P_c = 17.4 * (ω/ω_p)² GW
        critical_density = self._calculate_critical_density(wavelength_nm)
        omega_ratio_sq = plasma_density_m3 / critical_density
        P_c_GW = 17.4 / omega_ratio_sq
        return P_c_GW

    def _calculate_critical_density(self, wavelength_nm: float) -> float:
        """Calculate critical plasma density for given wavelength"""
        # n_c = ε₀ m_e ω² / e²
        epsilon_0 = 8.854187817e-12  # F/m
        m_e = 9.10938356e-31  # kg
        e = 1.602176634e-19  # C
        c = 2.99792458e8  # m/s

        omega = 2 * np.pi * c / (wavelength_nm * 1e-9)
        n_c = epsilon_0 * m_e * omega**2 / e**2
        return n_c

    def generate_facility_specific_configurations(
        self, base_intensity_W_m2: float, base_wavelength_nm: float = 800
    ) -> Dict[str, Any]:
        """Generate optimized configurations for each ELI facility"""

        print("\n🏢 GENERATING FACILITY-SPECIFIC CONFIGURATIONS")
        print("=" * 60)

        configurations = {}

        for facility in ELIFacility:
            print(f"\n📍 {facility.value}:")

            # Get facility constraints
            facility_constraints = self.eli_caps.facility_constraints[facility]
            max_intensity = facility_constraints["max_intensity_W_cm2"] * 1e4  # Convert to W/m²

            # Find compatible laser systems
            compatible_systems = self.eli_caps.get_compatible_systems(
                base_intensity_W_m2 / 1e4, base_wavelength_nm, 150, facility
            )

            if compatible_systems:
                best_system = compatible_systems[0]

                # Generate optimized configuration
                config = {
                    "facility": facility.value,
                    "laser_system": best_system.name,
                    "experimental_hall": best_system.experimental_hall,
                    "optimized_parameters": {
                        "intensity_W_m2": min(base_intensity_W_m2, max_intensity * 0.8),  # 80% of max for safety
                        "wavelength_nm": best_system.wavelength_nm,
                        "pulse_duration_fs": best_system.pulse_duration_fs,
                        "repetition_rate_Hz": best_system.repetition_rate_Hz,
                        "estimated_shots_per_day": best_system.repetition_rate_Hz * 86400,
                    },
                    "plasma_parameters": {
                        "target_density_m3": 1e25,  # Typical for analog Hawking experiments
                        "target_type": "solid",  # Default recommendation
                        "magnetic_field_T": 10,  # Moderate field for guidance
                    },
                    "diagnostic_requirements": facility_constraints["diagnostic_capabilities"],
                    "experimental_constraints": facility_constraints["experimental_constraints"],
                    "feasibility_score": 0.0,  # Will be calculated
                }

                # Calculate feasibility score
                feasibility = self._validate_full_configuration(
                    config["optimized_parameters"]["intensity_W_m2"],
                    config["optimized_parameters"]["wavelength_nm"],
                    config["optimized_parameters"]["pulse_duration_fs"],
                    config["plasma_parameters"]["target_density_m3"],
                    config["plasma_parameters"]["target_type"],
                    facility.value.lower(),
                    include_hawking_analysis=False,  # Skip for speed
                )

                config["feasibility_score"] = feasibility["overall_assessment"]["confidence_level"]

                print(f"   ✅ {best_system.name} (Score: {config['feasibility_score']:.2f})")
                print(f"      Intensity: {config['optimized_parameters']['intensity_W_m2']:.2e} W/m²")
                print(f"      Repetition Rate: {config['optimized_parameters']['repetition_rate_Hz']:.1f} Hz")
                print(f"      Shots/Day: {config['optimized_parameters']['estimated_shots_per_day']:.0f}")

                configurations[facility.value] = config
            else:
                print(f"   ❌ No compatible systems found")
                configurations[facility.value] = {"compatible": False}

        return configurations


def main():
    """Main execution function"""

    parser = argparse.ArgumentParser(
        description="Comprehensive ELI facility validation for analog Hawking radiation experiments"
    )
    parser.add_argument(
        "--mode",
        choices=["validate", "configurations", "demo"],
        default="validate",
        help="Validation mode to run",
    )
    parser.add_argument("--intensity", type=float, default=1e22, help="Laser intensity (W/m²)")
    parser.add_argument("--wavelength", type=float, default=800, help="Laser wavelength (nm)")
    parser.add_argument("--pulse-duration", type=float, default=150, help="Pulse duration (fs)")
    parser.add_argument("--plasma-density", type=float, default=1e25, help="Plasma density (m⁻³)")
    parser.add_argument("--target-type", type=str, default="solid", help="Target type")
    parser.add_argument("--facility", type=str, help="Preferred ELI facility")
    parser.add_argument("--output", type=str, help="Output file for results (JSON)")
    parser.add_argument("--no-hawking", action="store_true", help="Skip Hawking analysis")

    args = parser.parse_args()

    # Initialize validator
    validator = ELIAnalogHawkingValidator()

    # Run validation based on mode
    if args.mode == "validate":
        print("🔬 COMPREHENSIVE ELI FACILITY VALIDATION")
        print("=" * 70)

        results = validator.validate_full_configuration(
            args.intensity,
            args.wavelength,
            args.pulse_duration,
            args.plasma_density,
            args.target_type,
            args.facility,
            include_hawking_analysis=not args.no_hawking,
        )

    elif args.mode == "configurations":
        print("🏢 FACILITY-SPECIFIC CONFIGURATION GENERATION")
        print("=" * 70)

        results = validator.generate_facility_specific_configurations(
            args.intensity, args.wavelength
        )

    elif args.mode == "demo":
        print("🎭 DEMONSTRATION WITH REPRESENTATIVE PARAMETERS")
        print("=" * 70)

        # Demo with typical parameters for analog Hawking experiments
        demo_params = [
            {"intensity": 1e20, "name": "Conservative Configuration"},
            {"intensity": 1e22, "name": "Standard Configuration"},
            {"intensity": 5e23, "name": "High-Performance Configuration"},
        ]

        results = {"demo_results": []}

        for params in demo_params:
            print(f"\n🎯 {params['name']}:")
            result = validator.validate_full_configuration(
                params["intensity"],
                800,  # wavelength
                150,  # pulse duration
                1e25,  # plasma density
                "solid",
                None,  # no facility preference
                include_hawking_analysis=True,
            )
            result["config_name"] = params["name"]
            results["demo_results"].append(result)

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())