"""
ELI Facility Capabilities and Parameter Validation Module

This module provides detailed specifications for all ELI facilities and implements
validation functions to ensure experimental parameters are realistic and achievable.

Author: Claude Analysis Assistant
Date: November 2025
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np


class ELIFacility(Enum):
    """ELI facility locations and their primary capabilities"""
    ELI_BEAMLINES = "ELI-Beamlines"  # Czech Republic
    ELI_NP = "ELI-NP"               # Romania (Nuclear Physics)
    ELI_ALPS = "ELI-ALPS"           # Hungary (Attosecond Light Pulse Source)


@dataclass
class LaserSystemSpecs:
    """Detailed specifications for a laser system"""
    name: str
    facility: ELIFacility
    peak_power_TW: float  # Peak power in terawatts
    pulse_energy_J: float  # Pulse energy in joules
    pulse_duration_fs: float  # Pulse duration in femtoseconds
    wavelength_nm: float  # Central wavelength in nanometers
    repetition_rate_Hz: float  # Repetition rate in Hz
    max_intensity_W_cm2: float  # Maximum focused intensity in W/cm²
    beam_quality_M2: float  # Beam quality factor
    contrast_ratio: float  # Temporal contrast ratio
    polarization: str  # Polarization type
    operational_status: str  # Current operational status
    experimental_hall: Optional[str] = None  # Experimental hall identifier


class ELICapabilities:
    """Comprehensive ELI facility capabilities database"""

    def __init__(self):
        self.laser_systems = self._initialize_laser_systems()
        self.facility_constraints = self._initialize_facility_constraints()
        self.operational_limits = self._initialize_operational_limits()

    def _initialize_laser_systems(self) -> Dict[str, LaserSystemSpecs]:
        """Initialize all ELI laser systems with verified specifications"""

        systems = {
            # ELI-Beam Systems (Czech Republic)
            "L4_ATON": LaserSystemSpecs(
                name="L4 ATON",
                facility=ELIFacility.ELI_BEAMLINES,
                peak_power_TW=10000,  # 10 PW
                pulse_energy_J=1500,  # 1.5 kJ
                pulse_duration_fs=150,
                wavelength_nm=810,  # Ti:Sapphire
                repetition_rate_Hz=0.017,  # 1 shot per minute
                max_intensity_W_cm2=1e24,  # ~10^24 W/cm²
                beam_quality_M2=1.2,
                contrast_ratio=1e-10,
                polarization="linear",
                operational_status="commissioning",
                experimental_hall="E4"
            ),

            "L2_HAPLS": LaserSystemSpecs(
                name="L2 HAPLS",
                facility=ELIFacility.ELI_BEAMLINES,
                peak_power_TW=1000,  # 1 PW
                pulse_energy_J=30,
                pulse_duration_fs=30,
                wavelength_nm=1030,  # Yb:YAG
                repetition_rate_Hz=10,
                max_intensity_W_cm2=1e22,  # ~10^22 W/cm²
                beam_quality_M2=1.5,
                contrast_ratio=1e-9,
                polarization="linear",
                operational_status="operational",
                experimental_hall="E2"
            ),

            # ELI-NP Systems (Romania)
            "HPLS_10PW_A": LaserSystemSpecs(
                name="HPLS 10PW - Arm A",
                facility=ELIFacility.ELI_NP,
                peak_power_TW=10000,  # 10 PW
                pulse_energy_J=1500,
                pulse_duration_fs=150,
                wavelength_nm=810,  # Ti:Sapphire
                repetition_rate_Hz=0.003,  # 1 shot per 5 minutes
                max_intensity_W_cm2=1e24,  # ~10^24 W/cm²
                beam_quality_M2=1.3,
                contrast_ratio=1e-11,
                polarization="linear/circular",
                operational_status="operational",
                experimental_hall="E1"
            ),

            "HPLS_10PW_B": LaserSystemSpecs(
                name="HPLS 10PW - Arm B",
                facility=ELIFacility.ELI_NP,
                peak_power_TW=10000,  # 10 PW
                pulse_energy_J=1500,
                pulse_duration_fs=150,
                wavelength_nm=810,  # Ti:Sapphire
                repetition_rate_Hz=0.003,  # 1 shot per 5 minutes
                max_intensity_W_cm2=1e24,  # ~10^24 W/cm²
                beam_quality_M2=1.3,
                contrast_ratio=1e-11,
                polarization="linear/circular",
                operational_status="operational",
                experimental_hall="E2"
            ),

            "HPLS_1PW": LaserSystemSpecs(
                name="HPLS 1PW",
                facility=ELIFacility.ELI_NP,
                peak_power_TW=1000,  # 1 PW
                pulse_energy_J=200,
                pulse_duration_fs=200,
                wavelength_nm=810,  # Ti:Sapphire
                repetition_rate_Hz=0.1,  # 1 shot per 10 seconds
                max_intensity_W_cm2=1e23,  # ~10^23 W/cm²
                beam_quality_M2=1.2,
                contrast_ratio=1e-10,
                polarization="linear/circular",
                operational_status="operational",
                experimental_hall="E3"
            ),

            # ELI-ALPS Systems (Hungary)
            "HR1": LaserSystemSpecs(
                name="HR1 (High Repetition Rate)",
                facility=ELIFacility.ELI_ALPS,
                peak_power_TW=0.3,  # 300 TW
                pulse_energy_J=1,
                pulse_duration_fs=6,
                wavelength_nm=800,  # Ti:Sapphire
                repetition_rate_Hz=100000,  # 100 kHz
                max_intensity_W_cm2=1e20,  # ~10^20 W/cm²
                beam_quality_M2=1.1,
                contrast_ratio=1e-8,
                polarization="linear",
                operational_status="operational",
                experimental_hall="HR1"
            ),

            "SYLOS": LaserSystemSpecs(
                name="SYLOS 2PW",
                facility=ELIFacility.ELI_ALPS,
                peak_power_TW=2000,  # 2 PW
                pulse_energy_J=34,
                pulse_duration_fs=17,
                wavelength_nm=800,  # Ti:Sapphire
                repetition_rate_Hz=10,  # 10 Hz
                max_intensity_W_cm2=1e22,  # ~10^22 W/cm²
                beam_quality_M2=1.2,
                contrast_ratio=1e-9,
                polarization="linear",
                operational_status="operational",
                experimental_hall="SYLOS"
            ),
        }

        return systems

    def _initialize_facility_constraints(self) -> Dict[ELIFacility, Dict[str, Any]]:
        """Initialize facility-specific constraints and limitations"""

        constraints = {
            ELIFacility.ELI_BEAMLINES: {
                "max_intensity_W_cm2": 1e24,
                "wavelength_range_nm": (800, 1030),
                "pulse_duration_range_fs": (30, 150),
                "repetition_rate_limits_Hz": (0.017, 10),
                "experimental_constraints": [
                    "Vacuum chamber size: 2m x 2m x 2m",
                    "Target chamber pressure: <1e-6 mbar",
                    "Radiation shielding requirements for >10^22 W/cm²",
                    "Plasma mirror compatibility verified",
                    "Electron beamline integration available"
                ],
                "diagnostic_capabilities": [
                    "X-ray spectrometry (1-100 keV)",
                    "Electron spectrometry (MeV range)",
                    "Optical probing (fs resolution)",
                    "Thomson scattering diagnostics",
                    "Interferometry and shadowgraphy"
                ]
            },

            ELIFacility.ELI_NP: {
                "max_intensity_W_cm2": 1e24,
                "wavelength_range_nm": (810, 810),  # Fixed Ti:Sapphire
                "pulse_duration_range_fs": (150, 200),
                "repetition_rate_limits_Hz": (0.003, 0.1),
                "experimental_constraints": [
                    "Dual-beam configuration available",
                    "Gamma beam system integration",
                    "Nuclear physics target chamber",
                    "Enhanced radiation shielding",
                    "Heavy-ion target capability"
                ],
                "diagnostic_capabilities": [
                    "Gamma ray detection (20 MeV)",
                    "Nuclear activation analysis",
                    "High-energy particle spectrometry",
                    "Positron production detection",
                    "Nuclear reaction monitoring"
                ]
            },

            ELIFacility.ELI_ALPS: {
                "max_intensity_W_cm2": 1e22,  # Lower due to higher rep rate focus
                "wavelength_range_nm": (800, 800),  # Fixed Ti:Sapphire
                "pulse_duration_range_fs": (6, 17),
                "repetition_rate_limits_Hz": (10, 100000),
                "experimental_constraints": [
                    "Attosecond pulse generation focus",
                    "High repetition rate experiments",
                    "Ultra-fast diagnostics emphasis",
                    "Surface physics and thin film targets",
                    "Gas jet and cluster targets preferred"
                ],
                "diagnostic_capabilities": [
                    "Attosecond streaking",
                    "Frequency-resolved optical gating (FROG)",
                    "Spatially-resolved spectroscopy",
                    "Photoelectron spectroscopy",
                    "Time-of-flight mass spectrometry"
                ]
            }
        }

        return constraints

    def _initialize_operational_limits(self) -> Dict[str, Any]:
        """Initialize universal operational limits and safety constraints"""

        return {
            "safety_limits": {
                "max_intensity_W_cm2": 1e24,  # Hard limit for all facilities
                "min_focal_spot_um": 1.0,     # Minimum achievable focal spot
                "max_pulse_energy_J": 2000,   # Maximum pulse energy
                "radiation_zones": {
                    "controlled": 10,  # meters
                    "supervised": 30,  # meters
                    "public": 100      # meters
                }
            },

            "technical_constraints": {
                "pointing_stability_mrad": 0.05,  # Maximum pointing jitter
                "timing_jitter_fs": 30,           # Maximum timing jitter
                "energy_stability_percent": 3,    # Maximum energy fluctuation
                "wavefront_quality_PV": 200e-9,   # Peak-to-valley wavefront error (meters)
                "contrast_enhancement": 1e-12     # Maximum achievable contrast
            },

            "experimental_requirements": {
                "vacuum_level_mbar": 1e-7,        # Required vacuum level
                "target_positioning_um": 1,       # Target positioning accuracy
                "plasma_mirror_prep_time_s": 300, # Plasma mirror preparation time
                "diagnostic_integration_time_s": 600,  # Diagnostic setup time
                "shot_cycle_time_s": 60           # Minimum time between shots
            }
        }

    def get_compatible_systems(self, intensity_W_cm2: float,
                             wavelength_nm: float,
                             pulse_duration_fs: float,
                             facility: Optional[ELIFacility] = None) -> List[LaserSystemSpecs]:
        """Get laser systems compatible with experimental requirements"""

        compatible = []

        for system in self.laser_systems.values():
            # Skip if facility specified and doesn't match
            if facility and system.facility != facility:
                continue

            # Check intensity compatibility
            if intensity_W_cm2 > system.max_intensity_W_cm2:
                continue

            # Check wavelength compatibility (±10 nm tolerance)
            if abs(wavelength_nm - system.wavelength_nm) > 10:
                continue

            # Check pulse duration compatibility (±20% tolerance)
            duration_ratio = pulse_duration_fs / system.pulse_duration_fs
            if not 0.8 <= duration_ratio <= 1.2:
                continue

            # Check operational status
            if system.operational_status not in ["operational", "commissioning"]:
                continue

            compatible.append(system)

        # Sort by suitability (prefer operational systems with higher margin)
        compatible.sort(key=lambda s: (
            0 if s.operational_status == "operational" else 1,
            -np.log10(s.max_intensity_W_cm2 / intensity_W_cm2) if intensity_W_cm2 > 0 else 0
        ))

        return compatible

    def calculate_feasibility_score(self, intensity_W_cm2: float,
                                  wavelength_nm: float,
                                  pulse_duration_fs: float,
                                  facility: Optional[ELIFacility] = None) -> Dict[str, Any]:
        """Calculate feasibility score for experimental parameters"""

        compatible_systems = self.get_compatible_systems(
            intensity_W_cm2, wavelength_nm, pulse_duration_fs, facility
        )

        if not compatible_systems:
            return {
                "feasible": False,
                "score": 0.0,
                "primary_issues": [
                    f"Intensity {intensity_W_cm2:.1e} W/cm² exceeds all facility capabilities",
                    f"Wavelength {wavelength_nm:.0f} nm incompatible with available systems",
                    f"Pulse duration {pulse_duration_fs:.0f} fs outside operational range"
                ],
                "recommendations": [
                    "Reduce intensity to ≤10^23 W/cm² for broader compatibility",
                    "Use 800 nm or 1030 nm wavelength for Ti:Sapphire/Yb:YAG systems",
                    "Adjust pulse duration to 30-200 fs range"
                ]
            }

        best_system = compatible_systems[0]

        # Calculate intensity margin (higher is better)
        intensity_margin = best_system.max_intensity_W_cm2 / intensity_W_cm2

        # Calculate wavelength match (perfect match = 1.0)
        wavelength_match = 1.0 - abs(wavelength_nm - best_system.wavelength_nm) / 100.0

        # Calculate duration match (perfect match = 1.0)
        duration_match = 1.0 - abs(pulse_duration_fs - best_system.pulse_duration_fs) / best_system.pulse_duration_fs

        # Calculate operational factor (operational = 1.0, commissioning = 0.7)
        operational_factor = 1.0 if best_system.operational_status == "operational" else 0.7

        # Calculate repetition rate factor (higher rep rate is better for data collection)
        rep_rate_factor = np.log10(best_system.repetition_rate_Hz + 1) / 6.0  # Normalize to 0-1

        # Overall feasibility score
        score = (
            0.3 * np.log10(min(intensity_margin, 100)) / 2.0 +  # Intensity margin
            0.2 * wavelength_match +                           # Wavelength compatibility
            0.2 * duration_match +                             # Duration compatibility
            0.2 * operational_factor +                         # Operational status
            0.1 * rep_rate_factor                              # Repetition rate
        )

        score = max(0.0, min(1.0, score))

        # Generate recommendations
        recommendations = []
        if intensity_margin < 2:
            recommendations.append(f"Consider reducing intensity to ≤{best_system.max_intensity_W_cm2/2:.1e} W/cm² for safety margin")

        if wavelength_match < 0.9:
            recommendations.append(f"Use {best_system.wavelength_nm:.0f} nm wavelength for optimal compatibility")

        if duration_match < 0.9:
            recommendations.append(f"Adjust pulse duration to {best_system.pulse_duration_fs:.0f} fs for optimal performance")

        if best_system.operational_status == "commissioning":
            recommendations.append("System is in commissioning - operational schedule may be limited")

        return {
            "feasible": True,
            "score": score,
            "best_system": best_system.name,
            "facility": best_system.facility.value,
            "intensity_margin": intensity_margin,
            "wavelength_match": wavelength_match,
            "duration_match": duration_match,
            "operational_status": best_system.operational_status,
            "repetition_rate_Hz": best_system.repetition_rate_Hz,
            "experimental_hall": best_system.experimental_hall,
            "all_compatible_systems": [s.name for s in compatible_systems],
            "recommendations": recommendations if recommendations else ["Parameters are well within system capabilities"]
        }


def validate_intensity_range(intensity_W_m2: float,
                          facility: Optional[ELIFacility] = None) -> Dict[str, Any]:
    """
    Validate laser intensity against ELI facility capabilities

    Args:
        intensity_W_m2: Laser intensity in W/m²
        facility: Specific ELI facility to check against

    Returns:
        Validation results with detailed feedback
    """

    # Convert to W/cm² for comparison
    intensity_W_cm2 = intensity_W_m2 / 1e4

    eli = ELICapabilities()

    # Check against absolute limits
    if intensity_W_cm2 > 1e24:
        return {
            "valid": False,
            "intensity_W_cm2": intensity_W_cm2,
            "issue": "CRITICAL: Intensity exceeds maximum ELI capability",
            "max_eligible": 1e24,
            "facility_status": "INCOMPATIBLE",
            "recommendation": "Reduce intensity by factor of ≥10 for ELI compatibility"
        }

    # Check against facility-specific limits
    if facility:
        facility_limit = eli.facility_constraints[facility]["max_intensity_W_cm2"]
        if intensity_W_cm2 > facility_limit:
            return {
                "valid": False,
                "intensity_W_cm2": intensity_W_cm2,
                "issue": f"Intensity exceeds {facility.value} maximum",
                "max_eligible": facility_limit,
                "facility_status": "INCOMPATIBLE",
                "recommendation": f"Reduce intensity to ≤{facility_limit:.1e} W/cm² for {facility.value}"
            }

    # Determine facility compatibility
    compatible_facilities = []
    for fac in ELIFacility:
        if intensity_W_cm2 <= eli.facility_constraints[fac]["max_intensity_W_cm2"]:
            compatible_facilities.append(fac.value)

    # Generate feasibility assessment
    if intensity_W_cm2 <= 1e22:
        feasibility = "HIGH - Compatible with all ELI facilities"
    elif intensity_W_cm2 <= 1e23:
        feasibility = "MEDIUM - Compatible with ELI-Beamlines and ELI-NP"
    elif intensity_W_cm2 <= 1e24:
        feasibility = "LOW - Only compatible with 10 PW systems"
    else:
        feasibility = "INCOMPATIBLE"

    return {
        "valid": True,
        "intensity_W_cm2": intensity_W_cm2,
        "feasibility_level": feasibility,
        "compatible_facilities": compatible_facilities,
        "facility_status": "COMPATIBLE",
        "recommendations": _generate_intensity_recommendations(intensity_W_cm2)
    }


def _generate_intensity_recommendations(intensity_W_cm2: float) -> List[str]:
    """Generate specific recommendations based on intensity level"""

    recommendations = []

    if intensity_W_cm2 < 1e18:
        recommendations.extend([
            "Intensity is conservative - well within all facility capabilities",
            "Consider higher intensity for stronger plasma effects",
            "May be suitable for proof-of-concept experiments"
        ])
    elif intensity_W_cm2 < 1e20:
        recommendations.extend([
            "Good intensity range for initial plasma mirror studies",
            "Compatible with high repetition rate systems (ELI-ALPS)",
            "Excellent for statistical data collection"
        ])
    elif intensity_W_cm2 < 1e22:
        recommendations.extend([
            "Strong relativistic regime achieved",
            "Consider plasma mirror pre-pulse management",
            "Compatible with most ELI facilities"
        ])
    elif intensity_W_cm2 < 1e23:
        recommendations.extend([
            "Approaching maximum operational intensity",
            "Require careful plasma mirror timing",
            "Limited to ELI-Beamlines and ELI-NP facilities"
        ])
    else:
        recommendations.extend([
            "Maximum intensity regime - requires special approval",
            "Extensive safety procedures required",
            "Limited shot availability expected"
        ])

    return recommendations


# Create global instance for easy access
_eli_capabilities = ELICapabilities()

def get_eli_capabilities() -> ELICapabilities:
    """Get global ELI capabilities instance"""
    return _eli_capabilities

def quick_facility_check(intensity_W_m2: float, wavelength_nm: float = 800) -> str:
    """
    Quick facility compatibility check

    Args:
        intensity_W_m2: Laser intensity in W/m²
        wavelength_nm: Laser wavelength in nm

    Returns:
        Compatibility status string
    """

    result = validate_intensity_range(intensity_W_m2)

    if not result["valid"]:
        return "❌ INCOMPATIBLE: " + result["issue"]

    feasibility = result["feasibility_level"]
    facilities = ", ".join(result["compatible_facilities"])

    return f"✅ {feasibility} (Facilities: {facilities})"