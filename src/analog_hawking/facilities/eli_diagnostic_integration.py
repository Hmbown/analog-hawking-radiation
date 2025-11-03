"""
ELI Facility Diagnostic Integration for Analog Hawking Radiation Detection

This module provides comprehensive assessment of ELI facility diagnostic capabilities
and integration requirements for analog Hawking radiation experiments.

Author: Claude Analysis Assistant
Date: November 2025
"""

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


class ELIFacilityType(Enum):
    """ELI facility types"""
    ELI_BEAMLINES = "ELI-Beamlines"
    ELI_NP = "ELI-Nuclear Physics"
    ELI_ALPS = "ELI-Attosecond Light Pulse Source"


class DiagnosticCategory(Enum):
    """Diagnostic system categories"""
    TEMPORAL = "Temporal Diagnostics"
    SPECTRAL = "Spectral Diagnostics"
    SPATIAL = "Spatial Diagnostics"
    INTERFEROMETRIC = "Interferometric Diagnostics"
    PARTICLE = "Particle Diagnostics"
    RADIATION = "Radiation Diagnostics"


class ReadinessLevel(Enum):
    """Technology readiness levels for diagnostics"""
    LAB_PROTOTYPE = "Laboratory Prototype"
    FIELD_TESTED = "Field Tested"
    OPERATIONAL = "Operational"
    ELI_INTEGRATED = "ELI Integrated"
    PROVEN_ELI = "Proven at ELI"


@dataclass
class DiagnosticSystem:
    """ELI diagnostic system specification"""

    name: str
    category: DiagnosticCategory
    facility: ELIFacilityType
    readiness_level: ReadinessLevel

    # Performance specifications
    temporal_resolution: float  # s
    spectral_resolution: float  # Relative (Δλ/λ)
    spatial_resolution: float  # m
    dynamic_range: float  # dB
    sensitivity: float  # Minimum detectable signal

    # Operational parameters
    wavelength_range: Tuple[float, float]  # m
    field_of_view: float  # m
    data_rate: float  # Hz
    setup_time: float  # hours

    # Integration requirements
    vacuum_compatible: bool
    radiation_hardened: bool
    timing_synchronization: bool
    space_requirements: float  # m²

    # Cost and availability
    cost_tier: int  # 1-5 (1=lowest, 5=highest)
    availability: str  # "Standard", "Limited", "Special Request"
    maintenance_level: str  # "Low", "Medium", "High"


@dataclass
class IntegrationAssessment:
    """Diagnostic integration assessment result"""

    diagnostic_name: str
    detection_method: str
    compatibility_score: float  # 0-1
    integration_complexity: str  # "Low", "Medium", "High"
    required_modifications: List[str]
    integration_timeline: str  # Time required for integration
    cost_estimate: str
    technical_risks: List[str]
    mitigation_strategies: List[str]


@dataclass
class FacilityCapability:
    """Overall facility capability assessment"""

    facility: ELIFacilityType
    overall_readiness: float  # 0-1
    available_diagnostics: List[DiagnosticSystem]
    integration_assessments: List[IntegrationAssessment]

    # Capability scores
    temporal_capability: float  # 0-1
    spectral_capability: float  # 0-1
    spatial_capability: float  # 0-1
    overall_detection_capability: float  # 0-1

    # Operational constraints
    max_intensity_W_cm2: float
    repetition_rate_Hz: float
    beam_time_availability: str
    experimental_constraints: List[str]


class ELIDiagnosticDatabase:
    """Comprehensive ELI diagnostic capabilities database"""

    def __init__(self):
        self.diagnostics = self._initialize_diagnostic_database()
        self.facilities = self._initialize_facility_capabilities()

    def _initialize_diagnostic_database(self) -> Dict[str, DiagnosticSystem]:
        """Initialize comprehensive diagnostic system database"""

        diagnostics = {
            # ELI-Beamlines Diagnostics
            "optical_probing_system": DiagnosticSystem(
                name="Optical Probing System",
                category=DiagnosticCategory.TEMPORAL,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=10e-15,  # 10 fs
                spectral_resolution=0.001,  # 0.1%
                spatial_resolution=1e-6,  # 1 μm
                dynamic_range=60,  # dB
                sensitivity=1e-18,  # W
                wavelength_range=(400e-9, 800e-9),  # 400-800 nm
                field_of_view=1e-3,  # 1 mm
                data_rate=10,  # Hz
                setup_time=8,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=2.0,  # m²
                cost_tier=2,
                availability="Standard",
                maintenance_level="Medium"
            ),

            "frequency_resolved_optical_gating": DiagnosticSystem(
                name="Frequency-Resolved Optical Gating (FROG)",
                category=DiagnosticCategory.TEMPORAL,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.OPERATIONAL,
                temporal_resolution=5e-15,  # 5 fs
                spectral_resolution=0.0005,  # 0.05%
                spatial_resolution=5e-6,  # 5 μm
                dynamic_range=50,  # dB
                sensitivity=1e-16,  # W
                wavelength_range=(600e-9, 1100e-9),  # 600-1100 nm
                field_of_view=0.5e-3,  # 0.5 mm
                data_rate=1,  # Hz
                setup_time=12,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=1.5,  # m²
                cost_tier=2,
                availability="Standard",
                maintenance_level="Medium"
            ),

            "spectral_interferometer": DiagnosticSystem(
                name="Spectral Interferometer",
                category=DiagnosticCategory.INTERFEROMETRIC,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=1e-15,  # 1 fs
                spectral_resolution=1e-6,  # 10⁻⁶ relative
                spatial_resolution=2e-6,  # 2 μm
                dynamic_range=80,  # dB
                sensitivity=1e-20,  # W
                wavelength_range=(400e-9, 1000e-9),  # 400-1000 nm
                field_of_view=2e-3,  # 2 mm
                data_rate=10,  # Hz
                setup_time=16,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=3.0,  # m²
                cost_tier=3,
                availability="Standard",
                maintenance_level="High"
            ),

            "thz_time_domain_spectrometer": DiagnosticSystem(
                name="THz Time-Domain Spectrometer",
                category=DiagnosticCategory.SPECTRAL,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.OPERATIONAL,
                temporal_resolution=100e-15,  # 100 fs
                spectral_resolution=0.01,  # 1%
                spatial_resolution=100e-6,  # 100 μm
                dynamic_range=70,  # dB
                sensitivity=1e-15,  # W
                wavelength_range=(50e-6, 3e-3),  # 50 μm - 3 mm
                field_of_view=5e-3,  # 5 mm
                data_rate=100,  # Hz
                setup_time=24,  # hours
                vacuum_compatible=True,
                radiation_hardened=True,
                timing_synchronization=True,
                space_requirements=5.0,  # m²
                cost_tier=4,
                availability="Limited",
                maintenance_level="High"
            ),

            # ELI-NP Diagnostics
            "electron_spectrometer": DiagnosticSystem(
                name="Electron Spectrometer",
                category=DiagnosticCategory.PARTICLE,
                facility=ELIFacilityType.ELI_NP,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=1e-12,  # 1 ps
                spectral_resolution=0.01,  # 1% energy resolution
                spatial_resolution=1e-3,  # 1 mm
                dynamic_range=70,  # dB
                sensitivity=1e6,  # electrons
                wavelength_range=(1e-15, 1e-12),  # Gamma rays
                field_of_view=10e-3,  # 10 mm
                data_rate=10,  # Hz
                setup_time=20,  # hours
                vacuum_compatible=True,
                radiation_hardened=True,
                timing_synchronization=True,
                space_requirements=4.0,  # m²
                cost_tier=3,
                availability="Standard",
                maintenance_level="High"
            ),

            "gamma_detector_array": DiagnosticSystem(
                name="Gamma Detector Array",
                category=DiagnosticCategory.RADIATION,
                facility=ELIFacilityType.ELI_NP,
                readiness_level=ReadinessLevel.PROVEN_ELI,
                temporal_resolution=1e-9,  # 1 ns
                spectral_resolution=0.05,  # 5% energy resolution
                spatial_resolution=5e-3,  # 5 mm
                dynamic_range=90,  # dB
                sensitivity=1e-12,  # Gy
                wavelength_range=(1e-14, 1e-11),  # High-energy gamma
                field_of_view=50e-3,  # 50 mm
                data_rate=1000,  # Hz
                setup_time=40,  # hours
                vacuum_compatible=False,
                radiation_hardened=True,
                timing_synchronization=True,
                space_requirements=10.0,  # m²
                cost_tier=5,
                availability="Special Request",
                maintenance_level="High"
            ),

            # ELI-ALPS Diagnostics
            "attosecond_streak_camera": DiagnosticSystem(
                name="Attosecond Streak Camera",
                category=DiagnosticCategory.TEMPORAL,
                facility=ELIFacilityType.ELI_ALPS,
                readiness_level=ReadinessLevel.OPERATIONAL,
                temporal_resolution=100e-18,  # 100 as
                spectral_resolution=0.001,  # 0.1%
                spatial_resolution=10e-6,  # 10 μm
                dynamic_range=40,  # dB
                sensitivity=1e-17,  # W
                wavelength_range=(10e-9, 100e-9),  # 10-100 nm (XUV)
                field_of_view=0.1e-3,  # 0.1 mm
                data_rate=1,  # Hz
                setup_time=48,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=6.0,  # m²
                cost_tier=5,
                availability="Limited",
                maintenance_level="High"
            ),

            "xuv_spectrometer": DiagnosticSystem(
                name="XUV Spectrometer",
                category=DiagnosticCategory.SPECTRAL,
                facility=ELIFacilityType.ELI_ALPS,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=1e-15,  # 1 fs
                spectral_resolution=0.0001,  # 0.01%
                spatial_resolution=50e-6,  # 50 μm
                dynamic_range=60,  # dB
                sensitivity=1e-16,  # W
                wavelength_range=(1e-9, 100e-9),  # 1-100 nm
                field_of_view=1e-3,  # 1 mm
                data_rate=10,  # Hz
                setup_time=24,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=3.0,  # m²
                cost_tier=3,
                availability="Standard",
                maintenance_level="Medium"
            ),

            # General Purpose Diagnostics
            "optical_interferometer": DiagnosticSystem(
                name="Optical Interferometer",
                category=DiagnosticCategory.INTERFEROMETRIC,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=1e-15,  # 1 fs
                spectral_resolution=1e-6,  # High
                spatial_resolution=0.5e-6,  # 0.5 μm
                dynamic_range=80,  # dB
                sensitivity=1e-19,  # W
                wavelength_range=(400e-9, 800e-9),  # 400-800 nm
                field_of_view=5e-3,  # 5 mm
                data_rate=100,  # Hz
                setup_time=12,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=2.5,  # m²
                cost_tier=2,
                availability="Standard",
                maintenance_level="Medium"
            ),

            "shadowgraphy_system": DiagnosticSystem(
                name="Shadowgraphy System",
                category=DiagnosticCategory.SPATIAL,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=100e-15,  # 100 fs
                spectral_resolution=0.01,  # 1%
                spatial_resolution=1e-6,  # 1 μm
                dynamic_range=50,  # dB
                sensitivity=1e-16,  # W
                wavelength_range=(400e-9, 800e-9),  # 400-800 nm
                field_of_view=10e-3,  # 10 mm
                data_rate=10,  # Hz
                setup_time=8,  # hours
                vacuum_compatible=True,
                radiation_hardened=False,
                timing_synchronization=True,
                space_requirements=2.0,  # m²
                cost_tier=1,
                availability="Standard",
                maintenance_level="Low"
            ),

            "plasma_spectrometer": DiagnosticSystem(
                name="Plasma Spectrometer",
                category=DiagnosticCategory.SPECTRAL,
                facility=ELIFacilityType.ELI_BEAMLINES,
                readiness_level=ReadinessLevel.ELI_INTEGRATED,
                temporal_resolution=1e-12,  # 1 ps
                spectral_resolution=0.001,  # 0.1%
                spatial_resolution=10e-6,  # 10 μm
                dynamic_range=70,  # dB
                sensitivity=1e-17,  # W
                wavelength_range=(200e-9, 2000e-9),  # 200-2000 nm
                field_of_view=2e-3,  # 2 mm
                data_rate=1000,  # Hz
                setup_time=6,  # hours
                vacuum_compatible=True,
                radiation_hardened=True,
                timing_synchronization=True,
                space_requirements=1.5,  # m²
                cost_tier=2,
                availability="Standard",
                maintenance_level="Medium"
            )
        }

        return diagnostics

    def _initialize_facility_capabilities(self) -> Dict[ELIFacilityType, FacilityCapability]:
        """Initialize facility-specific capabilities"""

        facilities = {}

        # ELI-Beamlines
        beamlines_diagnostics = [d for d in self.diagnostics.values() if d.facility == ELIFacilityType.ELI_BEAMLINES]

        facilities[ELIFacilityType.ELI_BEAMLINES] = FacilityCapability(
            facility=ELIFacilityType.ELI_BEAMLINES,
            overall_readiness=0.85,
            available_diagnostics=beamlines_diagnostics,
            integration_assessments=[],
            temporal_capability=0.9,  # Excellent temporal diagnostics
            spectral_capability=0.8,  # Good spectral coverage
            spatial_capability=0.85,  # Good spatial diagnostics
            overall_detection_capability=0.85,
            max_intensity_W_cm2=1e22,  # 10²² W/cm²
            repetition_rate_Hz=10,  # Up to 10 Hz
            beam_time_availability="Good (3-6 months lead time)",
            experimental_constraints=[
                "Vacuum chamber size limitations",
                "Radiation safety requirements",
                "Target handling system compatibility"
            ]
        )

        # ELI-NP
        np_diagnostics = [d for d in self.diagnostics.values() if d.facility == ELIFacilityType.ELI_NP]

        facilities[ELIFacilityType.ELI_NP] = FacilityCapability(
            facility=ELIFacilityType.ELI_NP,
            overall_readiness=0.8,
            available_diagnostics=np_diagnostics,
            integration_assessments=[],
            temporal_capability=0.7,  # Moderate temporal resolution
            spectral_capability=0.9,  # Excellent for high-energy
            spatial_capability=0.8,  # Good spatial coverage
            overall_detection_capability=0.8,
            max_intensity_W_cm2=1e23,  # 10²³ W/cm² (highest)
            repetition_rate_Hz=0.01,  # Low repetition rate
            beam_time_availability="Limited (6-12 months lead time)",
            experimental_constraints=[
                "High radiation environment",
                "Strict radiation shielding requirements",
                "Limited access during operation",
                "Complex safety protocols"
            ]
        )

        # ELI-ALPS
        alps_diagnostics = [d for d in self.diagnostics.values() if d.facility == ELIFacilityType.ELI_ALPS]

        facilities[ELIFacilityType.ELI_ALPS] = FacilityCapability(
            facility=ELIFacilityType.ELI_ALPS,
            overall_readiness=0.75,
            available_diagnostics=alps_diagnostics,
            integration_assessments=[],
            temporal_capability=0.95,  # Best temporal resolution (attosecond)
            spectral_capability=0.85,  # Good XUV coverage
            spatial_capability=0.7,  # Limited spatial diagnostics
            overall_detection_capability=0.75,
            max_intensity_W_cm2=1e21,  # 10²¹ W/cm²
            repetition_rate_Hz=1,  # 1 kHz planned, currently 1 Hz
            beam_time_availability="Limited (6-9 months lead time)",
            experimental_constraints=[
                "XUV optics requirements",
                "Ultra-high vacuum needed",
                "Sensitive alignment requirements",
                "Specialized target systems"
            ]
        )

        return facilities


class ELIDiagnosticIntegrator:
    """ELI diagnostic integration and assessment system"""

    def __init__(self):
        self.db = ELIDiagnosticDatabase()
        self.detection_method_mapping = self._initialize_detection_mapping()

    def _initialize_detection_mapping(self) -> Dict[str, List[str]]:
        """Map detection methods to suitable diagnostics"""

        return {
            "Radio Spectroscopy": ["thz_time_domain_spectrometer"],
            "Optical Spectroscopy": ["plasma_spectrometer", "xuv_spectrometer"],
            "Interferometry": ["optical_interferometer", "spectral_interferometer"],
            "Imaging": ["shadowgraphy_system", "optical_probing_system"],
            "Quantum Correlation": ["spectral_interferometer", "optical_interferometer"],
            "Plasma Diagnostics": ["plasma_spectrometer", "shadowgraphy_system"],
            "Temporal Analysis": ["frequency_resolved_optical_gating", "attosecond_streak_camera"],
            "Particle Detection": ["electron_spectrometer"],
            "Radiation Detection": ["gamma_detector_array"]
        }

    def assess_diagnostic_compatibility(self,
                                      detection_method: str,
                                      facility: ELIFacilityType,
                                      signal_parameters: Dict[str, Any]) -> List[IntegrationAssessment]:
        """Assess diagnostic compatibility for specific detection method"""

        assessments = []

        # Find suitable diagnostics for this detection method
        suitable_diagnostics = self.detection_method_mapping.get(detection_method, [])

        for diag_name in suitable_diagnostics:
            if diag_name not in self.db.diagnostics:
                continue

            diagnostic = self.db.diagnostics[diag_name]

            # Check if diagnostic is available at this facility
            if diagnostic.facility != facility:
                continue

            # Calculate compatibility score
            compatibility_score = self._calculate_compatibility_score(diagnostic, signal_parameters)

            # Determine integration complexity
            integration_complexity = self._assess_integration_complexity(diagnostic, signal_parameters)

            # Identify required modifications
            required_modifications = self._identify_required_modifications(diagnostic, signal_parameters)

            # Estimate integration timeline
            integration_timeline = self._estimate_integration_timeline(diagnostic, integration_complexity)

            # Estimate cost
            cost_estimate = self._estimate_integration_cost(diagnostic, required_modifications)

            # Identify technical risks
            technical_risks = self._identify_technical_risks(diagnostic, signal_parameters)

            # Generate mitigation strategies
            mitigation_strategies = self._generate_mitigation_strategies(technical_risks)

            assessment = IntegrationAssessment(
                diagnostic_name=diagnostic.name,
                detection_method=detection_method,
                compatibility_score=compatibility_score,
                integration_complexity=integration_complexity,
                required_modifications=required_modifications,
                integration_timeline=integration_timeline,
                cost_estimate=cost_estimate,
                technical_risks=technical_risks,
                mitigation_strategies=mitigation_strategies
            )

            assessments.append(assessment)

        # Sort by compatibility score
        assessments.sort(key=lambda x: x.compatibility_score, reverse=True)

        return assessments

    def _calculate_compatibility_score(self,
                                     diagnostic: DiagnosticSystem,
                                     signal_parameters: Dict[str, Any]) -> float:
        """Calculate compatibility score between diagnostic and signal"""

        score = 0.0
        weights = {
            "frequency": 0.3,
            "temporal": 0.25,
            "sensitivity": 0.2,
            "spatial": 0.15,
            "readiness": 0.1
        }

        # Frequency compatibility
        signal_freq = signal_parameters.get("peak_frequency", 1e14)  # Hz
        signal_wavelength = 3e8 / signal_freq

        if (diagnostic.wavelength_range[0] <= signal_wavelength <= diagnostic.wavelength_range[1]):
            freq_score = 1.0
        else:
            # Calculate how far outside the range
            if signal_wavelength < diagnostic.wavelength_range[0]:
                freq_score = max(0, 1 - np.log10(diagnostic.wavelength_range[0] / signal_wavelength))
            else:
                freq_score = max(0, 1 - np.log10(signal_wavelength / diagnostic.wavelength_range[1]))
        score += weights["frequency"] * freq_score

        # Temporal resolution compatibility
        signal_duration = signal_parameters.get("pulse_duration", 1e-12)  # s
        if diagnostic.temporal_resolution <= signal_duration:
            temp_score = 1.0
        else:
            temp_score = max(0, 1 - np.log10(diagnostic.temporal_resolution / signal_duration))
        score += weights["temporal"] * temp_score

        # Sensitivity compatibility
        signal_power = signal_parameters.get("signal_power", 1e-20)  # W
        if diagnostic.sensitivity <= signal_power:
            sens_score = 1.0
        else:
            sens_score = max(0, 1 - np.log10(diagnostic.sensitivity / signal_power))
        score += weights["sensitivity"] * sens_score

        # Spatial resolution compatibility
        source_size = signal_parameters.get("source_size", 1e-6)  # m
        if diagnostic.spatial_resolution <= source_size:
            spatial_score = 1.0
        else:
            spatial_score = max(0, 1 - np.log10(diagnostic.spatial_resolution / source_size))
        score += weights["spatial"] * spatial_score

        # Readiness level
        readiness_scores = {
            ReadinessLevel.LAB_PROTOTYPE: 0.5,
            ReadinessLevel.FIELD_TESTED: 0.7,
            ReadinessLevel.OPERATIONAL: 0.8,
            ReadinessLevel.ELI_INTEGRATED: 0.9,
            ReadinessLevel.PROVEN_ELI: 1.0
        }
        score += weights["readiness"] * readiness_scores[diagnostic.readiness_level]

        return min(1.0, score)

    def _assess_integration_complexity(self,
                                      diagnostic: DiagnosticSystem,
                                      signal_parameters: Dict[str, Any]) -> str:
        """Assess integration complexity"""

        complexity_score = 0

        # Vacuum compatibility
        if not diagnostic.vacuum_compatible:
            complexity_score += 2

        # Radiation requirements
        if signal_parameters.get("high_radiation", False) and not diagnostic.radiation_hardened:
            complexity_score += 2

        # Timing requirements
        if signal_parameters.get("precise_timing", False) and not diagnostic.timing_synchronization:
            complexity_score += 1

        # Space requirements
        if diagnostic.space_requirements > 5:  # Large system
            complexity_score += 1

        # Readiness level
        if diagnostic.readiness_level in [ReadinessLevel.LAB_PROTOTYPE, ReadinessLevel.FIELD_TESTED]:
            complexity_score += 2
        elif diagnostic.readiness_level == ReadinessLevel.OPERATIONAL:
            complexity_score += 1

        if complexity_score <= 1:
            return "Low"
        elif complexity_score <= 3:
            return "Medium"
        else:
            return "High"

    def _identify_required_modifications(self,
                                       diagnostic: DiagnosticSystem,
                                       signal_parameters: Dict[str, Any]) -> List[str]:
        """Identify required modifications for integration"""

        modifications = []

        # Frequency range modifications
        signal_freq = signal_parameters.get("peak_frequency", 1e14)
        signal_wavelength = 3e8 / signal_freq

        if not (diagnostic.wavelength_range[0] <= signal_wavelength <= diagnostic.wavelength_range[1]):
            if signal_wavelength < diagnostic.wavelength_range[0]:
                modifications.append(f"Extend wavelength coverage to {signal_wavelength*1e9:.1f} nm (shorter wavelength)")
            else:
                modifications.append(f"Extend wavelength coverage to {signal_wavelength*1e9:.1f} nm (longer wavelength)")

        # Sensitivity improvements
        signal_power = signal_parameters.get("signal_power", 1e-20)
        if diagnostic.sensitivity > signal_power:
            improvement_factor = diagnostic.sensitivity / signal_power
            modifications.append(f"Improve sensitivity by factor {improvement_factor:.1f}")

        # Temporal resolution
        signal_duration = signal_parameters.get("pulse_duration", 1e-12)
        if diagnostic.temporal_resolution > signal_duration:
            modifications.append(f"Improve temporal resolution to {signal_duration*1e15:.1f} fs")

        # Vacuum requirements
        if not diagnostic.vacuum_compatible and signal_parameters.get("vacuum_required", True):
            modifications.append("Make system vacuum compatible")

        # Radiation hardening
        if not diagnostic.radiation_hardened and signal_parameters.get("high_radiation", False):
            modifications.append("Add radiation hardening")

        # Timing synchronization
        if not diagnostic.timing_synchronization and signal_parameters.get("precise_timing", False):
            modifications.append("Add timing synchronization capabilities")

        return modifications

    def _estimate_integration_timeline(self,
                                      diagnostic: DiagnosticSystem,
                                      integration_complexity: str) -> str:
        """Estimate integration timeline"""

        base_time = {
            ReadinessLevel.LAB_PROTOTYPE: 12,  # months
            ReadinessLevel.FIELD_TESTED: 6,
            ReadinessLevel.OPERATIONAL: 3,
            ReadinessLevel.ELI_INTEGRATED: 1,
            ReadinessLevel.PROVEN_ELI: 0.5
        }

        complexity_multiplier = {
            "Low": 1.0,
            "Medium": 1.5,
            "High": 2.5
        }

        base_months = base_time[diagnostic.readiness_level]
        total_months = base_months * complexity_multiplier[integration_complexity]

        if total_months <= 1:
            return "< 1 month"
        elif total_months <= 3:
            return "1-3 months"
        elif total_months <= 6:
            return "3-6 months"
        elif total_months <= 12:
            return "6-12 months"
        else:
            return f"> {int(total_months)} months"

    def _estimate_integration_cost(self,
                                  diagnostic: DiagnosticSystem,
                                  required_modifications: List[str]) -> str:
        """Estimate integration cost"""

        base_cost = {
            1: "€10k-€50k",
            2: "€50k-€100k",
            3: "€100k-€500k",
            4: "€500k-€1M",
            5: "€1M+"
        }

        modification_cost = len(required_modifications) * 0.2  # 20% per modification

        base_tier = diagnostic.cost_tier
        if modification_cost > 0:
            adjusted_tier = min(5, base_tier + int(modification_cost))
        else:
            adjusted_tier = base_tier

        return base_cost[adjusted_tier]

    def _identify_technical_risks(self,
                                 diagnostic: DiagnosticSystem,
                                 signal_parameters: Dict[str, Any]) -> List[str]:
        """Identify technical risks for integration"""

        risks = []

        # Readiness risks
        if diagnostic.readiness_level in [ReadinessLevel.LAB_PROTOTYPE, ReadinessLevel.FIELD_TESTED]:
            risks.append("Technology not yet proven at ELI facilities")

        # Performance risks
        signal_freq = signal_parameters.get("peak_frequency", 1e14)
        signal_wavelength = 3e8 / signal_freq

        if not (diagnostic.wavelength_range[0] <= signal_wavelength <= diagnostic.wavelength_range[1]):
            risks.append("Frequency range mismatch may require significant modifications")

        # Sensitivity risks
        signal_power = signal_parameters.get("signal_power", 1e-20)
        if diagnostic.sensitivity > signal_power * 10:  # More than 10x less sensitive
            risks.append("Insufficient sensitivity may prevent detection")

        # Timing risks
        signal_duration = signal_parameters.get("pulse_duration", 1e-12)
        if diagnostic.temporal_resolution > signal_duration * 10:  # More than 10x slower
            risks.append("Insufficient temporal resolution may miss transient phenomena")

        # Integration risks
        if diagnostic.space_requirements > 8:  # Very large system
            risks.append("Large footprint may limit integration options")

        if diagnostic.maintenance_level == "High":
            risks.append("High maintenance requirements may impact experimental schedule")

        return risks

    def _generate_mitigation_strategies(self, technical_risks: List[str]) -> List[str]:
        """Generate mitigation strategies for technical risks"""

        strategies = []

        for risk in technical_risks:
            if "not yet proven" in risk:
                strategies.append("Conduct extensive testing before experimental campaign")
                strategies.append("Allocate extra time for troubleshooting and optimization")

            elif "Frequency range" in risk:
                strategies.append("Explore alternative diagnostic approaches")
                strategies.append("Consider frequency conversion techniques")

            elif "sensitivity" in risk:
                strategies.append("Implement signal averaging techniques")
                strategies.append("Explore pre-amplification methods")

            elif "temporal resolution" in risk:
                strategies.append("Optimize experimental timing to maximize diagnostic overlap")
                strategies.append("Consider gating or streaking techniques")

            elif "footprint" in risk:
                strategies.append("Early integration planning with facility staff")
                strategies.append("Explore compact diagnostic alternatives")

            elif "maintenance" in risk:
                strategies.append("Schedule regular maintenance windows")
                strategies.append("Train on-site personnel for basic troubleshooting")

        # Add general strategies
        strategies.extend([
            "Early engagement with ELI diagnostic experts",
            "Comprehensive simulation of diagnostic performance",
            "Contingency planning for diagnostic failures"
        ])

        return list(set(strategies))  # Remove duplicates

    def generate_integration_plan(self,
                                 detection_methods: List[str],
                                 preferred_facility: ELIFacilityType,
                                 signal_parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive integration plan"""

        plan = {
            "facility": preferred_facility.value,
            "detection_methods": detection_methods,
            "signal_parameters": signal_parameters,
            "integration_assessments": {},
            "facility_recommendations": {},
            "timeline_overview": {},
            "budget_estimates": {},
            "risk_assessment": {}
        }

        # Assess each detection method
        for method in detection_methods:
            assessments = self.assess_diagnostic_compatibility(method, preferred_facility, signal_parameters)
            plan["integration_assessments"][method] = [
                {
                    "diagnostic": a.diagnostic_name,
                    "compatibility_score": a.compatibility_score,
                    "integration_complexity": a.integration_complexity,
                    "required_modifications": a.required_modifications,
                    "timeline": a.integration_timeline,
                    "cost": a.cost_estimate,
                    "technical_risks": a.technical_risks,
                    "mitigation_strategies": a.mitigation_strategies
                } for a in assessments
            ]

        # Generate facility recommendations
        facility_capability = self.db.facilities[preferred_facility]
        plan["facility_recommendations"] = {
            "overall_readiness": facility_capability.overall_readiness,
            "max_intensity": facility_capability.max_intensity_W_cm2,
            "repetition_rate": facility_capability.repetition_rate_Hz,
            "beam_time_availability": facility_capability.beam_time_availability,
            "experimental_constraints": facility_capability.experimental_constraints,
            "available_diagnostics": [d.name for d in facility_capability.available_diagnostics]
        }

        # Generate timeline overview
        all_timelines = []
        all_costs = []

        for method_assessments in plan["integration_assessments"].values():
            if method_assessments:  # If assessments exist
                best_assessment = method_assessments[0]  # Best compatibility
                all_timelines.append(best_assessment["timeline"])
                all_costs.append(best_assessment["cost"])

        plan["timeline_overview"] = {
            "integration_timeline": max(all_timelines) if all_timelines else "Unknown",
            "estimated_preparation_time": "3-6 months",
            "commissioning_time": "1-2 months",
            "total_time_to_first_detection": "6-12 months"
        }

        # Budget estimates
        plan["budget_estimates"] = {
            "diagnostic_integration": max(all_costs) if all_costs else "Unknown",
            "facility_overheads": "€50k-€100k",
            "personnel_costs": "€100k-€200k",
            "contingency": "20% of total",
            "total_estimated_budget": "€300k-€800k"
        }

        # Risk assessment
        all_risks = []
        for method_assessments in plan["integration_assessments"].values():
            for assessment in method_assessments:
                all_risks.extend(assessment["technical_risks"])

        plan["risk_assessment"] = {
            "identified_risks": list(set(all_risks)),
            "risk_level": "Medium" if len(all_risks) <= 5 else "High",
            "mitigation_priority": "Focus on sensitivity and timing requirements"
        }

        return plan

    def export_integration_plan(self, plan: Dict[str, Any], output_path: str):
        """Export integration plan to file"""

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if output_file.suffix == '.json':
            with open(output_file, 'w') as f:
                json.dump(plan, f, indent=2, default=str)
        else:
            # Generate markdown report
            self._generate_markdown_report(plan, output_file)

    def _generate_markdown_report(self, plan: Dict[str, Any], output_path: str):
        """Generate markdown integration report"""

        lines = []
        lines.append("# ELI Facility Diagnostic Integration Plan")
        lines.append("=" * 50)
        lines.append(f"Target Facility: {plan['facility']}")
        lines.append(f"Detection Methods: {', '.join(plan['detection_methods'])}")
        lines.append(f"Generated: {np.datetime64('now')}")
        lines.append("")

        # Executive Summary
        lines.append("## Executive Summary")
        lines.append("")

        facility_rec = plan["facility_recommendations"]
        lines.append(f"**Facility Readiness:** {facility_rec['overall_readiness']:.1%}")
        lines.append(f"**Maximum Intensity:** {facility_rec['max_intensity']:.2e} W/cm²")
        lines.append(f"**Repetition Rate:** {facility_rec['repetition_rate']} Hz")
        lines.append(f"**Beam Time Availability:** {facility_rec['beam_time_availability']}")
        lines.append("")

        # Detection Method Assessments
        lines.append("## Detection Method Assessments")
        lines.append("")

        for method, assessments in plan["integration_assessments"].items():
            lines.append(f"### {method}")
            lines.append("")

            if assessments:
                best = assessments[0]  # Best compatibility
                lines.append(f"**Recommended Diagnostic:** {best['diagnostic']}")
                lines.append(f"**Compatibility Score:** {best['compatibility_score']:.1%}")
                lines.append(f"**Integration Complexity:** {best['integration_complexity']}")
                lines.append(f"**Timeline:** {best['timeline']}")
                lines.append(f"**Cost Estimate:** {best['cost']}")
                lines.append("")

                if best['required_modifications']:
                    lines.append("**Required Modifications:**")
                    for mod in best['required_modifications']:
                        lines.append(f"- {mod}")
                    lines.append("")

                if best['technical_risks']:
                    lines.append("**Technical Risks:**")
                    for risk in best['technical_risks']:
                        lines.append(f"- {risk}")
                    lines.append("")

                if best['mitigation_strategies']:
                    lines.append("**Mitigation Strategies:**")
                    for strategy in best['mitigation_strategies']:
                        lines.append(f"- {strategy}")
                    lines.append("")
            else:
                lines.append("❌ **No compatible diagnostics found for this method**")
                lines.append("")

        # Timeline and Budget
        lines.append("## Timeline and Budget")
        lines.append("")

        timeline = plan["timeline_overview"]
        lines.append("### Project Timeline")
        for key, value in timeline.items():
            lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
        lines.append("")

        budget = plan["budget_estimates"]
        lines.append("### Budget Estimates")
        for key, value in budget.items():
            lines.append(f"**{key.replace('_', ' ').title()}:** {value}")
        lines.append("")

        # Risk Assessment
        lines.append("## Risk Assessment")
        lines.append("")

        risk_assessment = plan["risk_assessment"]
        lines.append(f"**Overall Risk Level:** {risk_assessment['risk_level']}")
        lines.append(f"**Mitigation Priority:** {risk_assessment['mitigation_priority']}")
        lines.append("")

        if risk_assessment["identified_risks"]:
            lines.append("### Identified Risks")
            for risk in risk_assessment["identified_risks"]:
                lines.append(f"- {risk}")
            lines.append("")

        # Recommendations
        lines.append("## Recommendations")
        lines.append("")

        lines.append("### Immediate Actions (Next 3 months)")
        lines.append("1. Contact ELI facility diagnostic experts to confirm compatibility")
        lines.append("2. Begin procurement of diagnostic equipment with longest lead times")
        lines.append("3. Initiate safety and integration documentation")
        lines.append("4. Schedule facility integration planning meetings")
        lines.append("")

        lines.append("### Short-term Goals (3-6 months)")
        lines.append("1. Complete diagnostic equipment procurement")
        lines.append("2. Conduct off-site testing and characterization")
        lines.append("3. Develop detailed integration procedures")
        lines.append("4. Train experimental team on new diagnostics")
        lines.append("")

        lines.append("### Long-term Goals (6+ months)")
        lines.append("1. Complete facility integration and commissioning")
        lines.append("2. Perform initial characterization measurements")
        lines.append("3. Optimize diagnostic performance for specific experiments")
        lines.append("4. Execute first analog Hawking radiation detection campaign")
        lines.append("")

        lines.append("---")
        lines.append("*Integration plan generated by ELI Diagnostic Integration System*")

        report = "\n".join(lines)
        Path(output_path).write_text(report)


# Convenience functions
def assess_eli_compatibility(detection_method: str,
                           facility: str,
                           signal_parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience function for ELI compatibility assessment"""

    integrator = ELIDiagnosticIntegrator()
    facility_map = {
        "beamlines": ELIFacilityType.ELI_BEAMLINES,
        "np": ELIFacilityType.ELI_NP,
        "alps": ELIFacilityType.ELI_ALPS
    }

    eli_facility = facility_map.get(facility.lower(), ELIFacilityType.ELI_BEAMLINES)

    assessments = integrator.assess_diagnostic_compatibility(detection_method, eli_facility, signal_parameters)

    return {
        "facility": facility,
        "detection_method": detection_method,
        "assessments": [
            {
                "diagnostic": a.diagnostic_name,
                "compatibility_score": a.compatibility_score,
                "integration_complexity": a.integration_complexity,
                "timeline": a.integration_timeline,
                "cost": a.cost_estimate,
                "risks": a.technical_risks
            } for a in assessments
        ]
    }


def generate_eli_integration_plan(detection_methods: List[str],
                                 facility: str,
                                 signal_parameters: Dict[str, Any],
                                 output_path: str) -> str:
    """Generate comprehensive ELI integration plan"""

    integrator = ELIDiagnosticIntegrator()
    facility_map = {
        "beamlines": ELIFacilityType.ELI_BEAMLINES,
        "np": ELIFacilityType.ELI_NP,
        "alps": ELIFacilityType.ELI_ALPS
    }

    eli_facility = facility_map.get(facility.lower(), ELIFacilityType.ELI_BEAMLINES)

    plan = integrator.generate_integration_plan(detection_methods, eli_facility, signal_parameters)
    integrator.export_integration_plan(plan, output_path)

    return output_path