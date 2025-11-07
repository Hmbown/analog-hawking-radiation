"""
ELI-Specific Physics Threshold Validation Module

This module provides physics validation specifically tailored to ELI facility capabilities
and constraints for analog Hawking radiation experiments. It integrates with the existing
validation framework and provides detailed threshold analysis.

Author: Claude Analysis Assistant
Date: November 2025
Version: 1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .eli_capabilities import ELIFacility


# Define physical constants locally to avoid import issues
class PhysicalConstants:
    """Physical constants for calculations"""
    c = 2.99792458e8      # Speed of light (m/s)
    e = 1.602176634e-19    # Elementary charge (C)
    m_e = 9.10938356e-31   # Electron mass (kg)
    m_p = 1.67262192e-27   # Proton mass (kg)
    epsilon_0 = 8.854187817e-12  # Vacuum permittivity (F/m)
    hbar = 1.054571817e-34  # Reduced Planck constant (J·s)
    k_B = 1.380649e-23      # Boltzmann constant (J/K)
    eV_to_J = 1.602176634e-19  # eV to Joules conversion


@dataclass
class PhysicsThresholdResult:
    """Result of physics threshold validation"""
    parameter_name: str
    value: float
    threshold_value: float
    unit: str
    passed: bool
    margin: float
    severity: str  # "critical", "warning", "info"
    message: str
    recommendation: str


class ELIPhysicsValidator:
    """
    Physics validation for ELI-specific analog Hawking radiation experiments
    """

    def __init__(self):
        self.constants = PhysicalConstants()
        self.thresholds = self._initialize_eli_thresholds()

    def _initialize_eli_thresholds(self) -> Dict[str, Any]:
        """Initialize ELI-specific physics thresholds"""

        return {
            # Universal physics limits (from thresholds.yaml)
            "velocity_fraction_c_max": 0.5,
            "gradient_max_s": 4.0e12,
            "intensity_max_W_m2": 1.0e24,

            # ELI facility-specific limits
            "eli_beamlines": {
                "intensity_max_W_m2": 1e24,
                "gradient_max_s": 3.5e12,  # Slightly conservative
                "repetition_rate_min_s": 60,  # Minimum time between shots
                "pulse_energy_max_J": 1500,
            },
            "eli_np": {
                "intensity_max_W_m2": 1e24,
                "gradient_max_s": 4.0e12,
                "repetition_rate_min_s": 300,  # 5 minutes between shots
                "pulse_energy_max_J": 1500,
                "magnetic_field_max_T": 50,
            },
            "eli_alps": {
                "intensity_max_W_m2": 1e22,  # Lower due to high rep rate
                "gradient_max_s": 3.0e12,  # Conservative for stability
                "repetition_rate_min_s": 0.1,  # 10 Hz operation
                "pulse_energy_max_J": 34,
            },

            # Analog Hawking specific thresholds
            "hawking_physics": {
                "kappa_min_Hz": 1e9,      # Minimum for detectable signal
                "kappa_max_Hz": 1e14,     # Maximum before breakdown
                "kappa_optimal_min_Hz": 1e11,
                "kappa_optimal_max_Hz": 1e13,
                "temperature_min_K": 1e-10,  # Minimum for detection
                "density_gradient_min_m": 1e-7,  # Minimum for horizon formation
                "density_gradient_max_m": 1e-5,  # Maximum before breakdown
            },

            # Plasma formation thresholds
            "plasma_formation": {
                "ionization_threshold_W_cm2": 1e13,  # Minimum for ionization
                "plasma_mirror_formation_W_cm2": 1e14,  # Minimum for mirror
                "relativistic_threshold_a0": 1.0,  # a0 for relativistic effects
                "radiation_pressure_dominance_a0": 10.0,  # Radiation pressure > thermal
            },
        }

    def validate_comprehensive_configuration(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        pulse_duration_fs: float,
        plasma_density_m3: float,
        gradient_scale_m: float,
        flow_velocity_ms: float,
        facility: ELIFacility,
    ) -> Dict[str, Any]:
        """
        Comprehensive physics validation for ELI configuration

        Args:
            intensity_W_m2: Laser intensity in W/m²
            wavelength_nm: Laser wavelength in nm
            pulse_duration_fs: Pulse duration in femtoseconds
            plasma_density_m3: Plasma density in m⁻³
            gradient_scale_m: Density gradient scale in meters
            flow_velocity_ms: Flow velocity in m/s
            facility: ELI facility

        Returns:
            Comprehensive validation results
        """

        validation_results = {
            "facility": facility.value,
            "input_parameters": {
                "intensity_W_m2": intensity_W_m2,
                "wavelength_nm": wavelength_nm,
                "pulse_duration_fs": pulse_duration_fs,
                "plasma_density_m3": plasma_density_m3,
                "gradient_scale_m": gradient_scale_m,
                "flow_velocity_ms": flow_velocity_ms,
            },
            "threshold_validations": [],
            "derived_parameters": {},
            "critical_issues": [],
            "warnings": [],
            "recommendations": [],
            "overall_passed": True,
            "confidence_score": 0.0,
        }

        # Calculate derived parameters
        derived_params = self._calculate_derived_parameters(
            intensity_W_m2, wavelength_nm, plasma_density_m3, gradient_scale_m, flow_velocity_ms
        )
        validation_results["derived_parameters"] = derived_params

        # Validate all thresholds
        threshold_results = []

        # 1. Universal physics limits
        universal_results = self._validate_universal_limits(
            intensity_W_m2, flow_velocity_ms, gradient_scale_m
        )
        threshold_results.extend(universal_results)

        # 2. Facility-specific limits
        facility_results = self._validate_facility_limits(
            intensity_W_m2, pulse_duration_fs, facility
        )
        threshold_results.extend(facility_results)

        # 3. Plasma formation physics
        plasma_results = self._validate_plasma_formation(
            intensity_W_m2, wavelength_nm, plasma_density_m3
        )
        threshold_results.extend(plasma_results)

        # 4. Hawking physics validity
        hawking_results = self._validate_hawking_physics(
            derived_params["surface_gravity_Hz"],
            derived_params["hawking_temperature_K"],
            gradient_scale_m,
        )
        threshold_results.extend(hawking_results)

        # 5. Relativistic effects
        relativistic_results = self._validate_relativistic_effects(
            derived_params["relativistic_parameter_a0"],
            flow_velocity_ms,
            plasma_density_m3,
        )
        threshold_results.extend(relativistic_results)

        validation_results["threshold_validations"] = threshold_results

        # Analyze results
        critical_issues = [r for r in threshold_results if r.severity == "critical" and not r.passed]
        warnings = [r for r in threshold_results if r.severity == "warning" and not r.passed]

        validation_results["critical_issues"] = [r.message for r in critical_issues]
        validation_results["warnings"] = [r.message for r in warnings]

        if critical_issues:
            validation_results["overall_passed"] = False

        # Generate recommendations
        recommendations = self._generate_recommendations(threshold_results, facility)
        validation_results["recommendations"] = recommendations

        # Calculate confidence score
        passed_checks = sum(1 for r in threshold_results if r.passed)
        total_checks = len(threshold_results)
        validation_results["confidence_score"] = passed_checks / total_checks

        return validation_results

    def _calculate_derived_parameters(
        self,
        intensity_W_m2: float,
        wavelength_nm: float,
        plasma_density_m3: float,
        gradient_scale_m: float,
        flow_velocity_ms: float,
    ) -> Dict[str, float]:
        """Calculate derived physics parameters"""

        # Convert units
        intensity_W_cm2 = intensity_W_m2 / 1e4
        wavelength_m = wavelength_nm * 1e-9

        # Relativistic parameter a0
        a0 = 0.85 * np.sqrt(intensity_W_cm2 / 1e18) * (wavelength_nm / 1000)

        # Critical density
        omega = 2 * np.pi * self.constants.c / wavelength_m
        n_critical = self.constants.epsilon_0 * self.constants.m_e * omega**2 / self.constants.e**2

        # Plasma frequency
        omega_p = np.sqrt(plasma_density_m3 * self.constants.e**2 /
                         (self.constants.epsilon_0 * self.constants.m_e))

        # Sound speed (assume fully ionized hydrogen)
        gamma = 5/3  # Adiabatic index
        kB_T = 1e3 * self.constants.eV_to_J  # Assume 1 keV temperature
        c_s = np.sqrt(gamma * kB_T / self.constants.m_p)

        # Surface gravity κ (simplified model)
        # κ = |d(v - cs)/dx| at horizon
        dv_dx = flow_velocity_ms / gradient_scale_m
        dcs_dx = c_s / gradient_scale_m
        kappa = abs(dv_dx - dcs_dx)

        # Hawking temperature
        T_H = self.constants.hbar * kappa / (2 * np.pi * self.constants.k_B)

        # Laser power (assuming focal spot)
        focal_spot = 3e-6  # 3 μm
        P_laser = intensity_W_m2 * np.pi * (focal_spot/2)**2

        # Critical power for self-focusing
        P_critical = 17.4e9 * (omega / omega_p)**2  # GW

        return {
            "relativistic_parameter_a0": a0,
            "critical_density_m3": n_critical,
            "plasma_frequency_Hz": omega_p / (2*np.pi),
            "sound_speed_ms": c_s,
            "surface_gravity_Hz": kappa,
            "hawking_temperature_K": T_H,
            "laser_power_W": P_laser,
            "critical_power_W": P_critical,
            "density_ratio": plasma_density_m3 / n_critical,
            "mach_number": flow_velocity_ms / c_s,
        }

    def _validate_universal_limits(
        self, intensity_W_m2: float, flow_velocity_ms: float, gradient_scale_m: float
    ) -> List[PhysicsThresholdResult]:
        """Validate universal physics limits"""

        results = []

        # 1. Intensity limit
        max_intensity = self.thresholds["intensity_max_W_m2"]
        intensity_margin = max_intensity / intensity_W_m2
        results.append(PhysicsThresholdResult(
            parameter_name="Intensity",
            value=intensity_W_m2,
            threshold_value=max_intensity,
            unit="W/m²",
            passed=intensity_W_m2 <= max_intensity,
            margin=intensity_margin,
            severity="critical" if intensity_W_m2 > max_intensity else "info",
            message=f"Intensity {intensity_W_m2:.2e} W/m² {'exceeds' if intensity_W_m2 > max_intensity else 'within'} universal limit",
            recommendation=f"Reduce intensity by factor {intensity_W_m2/max_intensity:.2f}" if intensity_W_m2 > max_intensity else "Intensity is acceptable"
        ))

        # 2. Velocity limit
        max_velocity = self.thresholds["velocity_fraction_c_max"] * self.constants.c
        velocity_margin = max_velocity / flow_velocity_ms
        results.append(PhysicsThresholdResult(
            parameter_name="Flow Velocity",
            value=flow_velocity_ms,
            threshold_value=max_velocity,
            unit="m/s",
            passed=flow_velocity_ms <= max_velocity,
            margin=velocity_margin,
            severity="critical" if flow_velocity_ms > max_velocity else "info",
            message=f"Flow velocity {flow_velocity_ms:.2e} m/s ({flow_velocity_ms/self.constants.c:.2f}c) {'exceeds' if flow_velocity_ms > max_velocity else 'within'} limit",
            recommendation=f"Reduce velocity to ≤{max_velocity:.2e} m/s" if flow_velocity_ms > max_velocity else "Velocity is acceptable"
        ))

        # 3. Gradient limit
        if gradient_scale_m > 0:
            max_gradient = self.thresholds["gradient_max_s"]
            gradient_value = self.constants.c / gradient_scale_m  # Approximate gradient
            gradient_margin = max_gradient / gradient_value
            results.append(PhysicsThresholdResult(
                parameter_name="Density Gradient",
                value=gradient_value,
                threshold_value=max_gradient,
                unit="s⁻¹",
                passed=gradient_value <= max_gradient,
                margin=gradient_margin,
                severity="critical" if gradient_value > max_gradient else "info",
                message=f"Density gradient {gradient_value:.2e} s⁻¹ {'exceeds' if gradient_value > max_gradient else 'within'} limit",
                recommendation=f"Increase gradient scale to ≥{self.constants.c/max_gradient:.2e} m" if gradient_value > max_gradient else "Gradient is acceptable"
            ))

        return results

    def _validate_facility_limits(
        self, intensity_W_m2: float, pulse_duration_fs: float, facility: ELIFacility
    ) -> List[PhysicsThresholdResult]:
        """Validate facility-specific limits"""

        results = []
        facility_limits = self.thresholds[facility.value.lower().replace("-", "_")]

        # 1. Facility intensity limit
        max_intensity = facility_limits["intensity_max_W_m2"]
        intensity_margin = max_intensity / intensity_W_m2
        results.append(PhysicsThresholdResult(
            parameter_name=f"Facility Intensity ({facility.value})",
            value=intensity_W_m2,
            threshold_value=max_intensity,
            unit="W/m²",
            passed=intensity_W_m2 <= max_intensity,
            margin=intensity_margin,
            severity="critical" if intensity_W_m2 > max_intensity else "warning",
            message=f"Intensity {'exceeds' if intensity_W_m2 > max_intensity else 'within'} {facility.value} capability",
            recommendation=f"Consider different facility or reduce intensity" if intensity_W_m2 > max_intensity else f"Compatible with {facility.value}"
        ))

        # 2. Repetition rate constraints
        min_shot_time = facility_limits["repetition_rate_min_s"]
        if pulse_duration_fs * 1e-15 > min_shot_time / 100:  # Pulse duration should be much less than shot cycle
            results.append(PhysicsThresholdResult(
                parameter_name="Pulse Duration vs Cycle Time",
                value=pulse_duration_fs * 1e-15,
                threshold_value=min_shot_time / 100,
                unit="s",
                passed=True,
                margin=min_shot_time / (100 * pulse_duration_fs * 1e-15),
                severity="info",
                message=f"Pulse duration compatible with {facility.value} repetition rate",
                recommendation="Pulse duration is acceptable for facility cycle time"
            ))

        return results

    def _validate_plasma_formation(
        self, intensity_W_m2: float, wavelength_nm: float, plasma_density_m3: float
    ) -> List[PhysicsThresholdResult]:
        """Validate plasma formation physics"""

        results = []
        intensity_W_cm2 = intensity_W_m2 / 1e4
        plasma_thresholds = self.thresholds["plasma_formation"]

        # 1. Ionization threshold
        ionization_threshold = plasma_thresholds["ionization_threshold_W_cm2"]
        results.append(PhysicsThresholdResult(
            parameter_name="Ionization Threshold",
            value=intensity_W_cm2,
            threshold_value=ionization_threshold,
            unit="W/cm²",
            passed=intensity_W_cm2 >= ionization_threshold,
            margin=intensity_W_cm2 / ionization_threshold,
            severity="warning" if intensity_W_cm2 < ionization_threshold else "info",
            message=f"Intensity {'insufficient' if intensity_W_cm2 < ionization_threshold else 'sufficient'} for ionization",
            recommendation=f"Increase intensity to ≥{ionization_threshold:.1e} W/cm²" if intensity_W_cm2 < ionization_threshold else "Ionization achievable"
        ))

        # 2. Plasma mirror formation
        mirror_threshold = plasma_thresholds["plasma_mirror_formation_W_cm2"]
        results.append(PhysicsThresholdResult(
            parameter_name="Plasma Mirror Formation",
            value=intensity_W_cm2,
            threshold_value=mirror_threshold,
            unit="W/cm²",
            passed=intensity_W_cm2 >= mirror_threshold,
            margin=intensity_W_cm2 / mirror_threshold,
            severity="warning" if intensity_W_cm2 < mirror_threshold else "info",
            message=f"Intensity {'insufficient' if intensity_W_cm2 < mirror_threshold else 'sufficient'} for plasma mirror formation",
            recommendation=f"Increase intensity to ≥{mirror_threshold:.1e} W/cm²" if intensity_W_cm2 < mirror_threshold else "Plasma mirror formation possible"
        ))

        # 3. Relativistic parameter
        a0 = 0.85 * np.sqrt(intensity_W_cm2 / 1e18) * (wavelength_nm / 1000)
        relativistic_threshold = plasma_thresholds["relativistic_threshold_a0"]
        results.append(PhysicsThresholdResult(
            parameter_name="Relativistic Parameter (a₀)",
            value=a0,
            threshold_value=relativistic_threshold,
            unit="dimensionless",
            passed=a0 >= relativistic_threshold,
            margin=a0 / relativistic_threshold,
            severity="info",
            message=f"Relativistic parameter a₀ = {a0:.2f} ({'relativistic' if a0 >= 1 else 'non-relativistic'} regime)",
            recommendation="Consider relativistic effects in analysis" if a0 >= 1 else "Non-relativistic regime valid"
        ))

        return results

    def _validate_hawking_physics(
        self, kappa_Hz: float, temperature_K: float, gradient_scale_m: float
    ) -> List[PhysicsThresholdResult]:
        """Validate Hawking radiation physics feasibility"""

        results = []
        hawking_thresholds = self.thresholds["hawking_physics"]

        # 1. Surface gravity range
        min_kappa = hawking_thresholds["kappa_min_Hz"]
        max_kappa = hawking_thresholds["kappa_max_Hz"]

        if min_kappa <= kappa_Hz <= max_kappa:
            kappa_passed = True
            kappa_severity = "info"
        elif kappa_Hz < min_kappa:
            kappa_passed = False
            kappa_severity = "warning"
        else:
            kappa_passed = False
            kappa_severity = "critical"

        results.append(PhysicsThresholdResult(
            parameter_name="Surface Gravity (κ)",
            value=kappa_Hz,
            threshold_value=(min_kappa, max_kappa),
            unit="Hz",
            passed=kappa_passed,
            margin=min(kappa_Hz/min_kappa, max_kappa/kappa_Hz) if kappa_passed else 0,
            severity=kappa_severity,
            message=f"κ = {kappa_Hz:.2e} Hz ({'within' if kappa_passed else 'outside'} valid range [{min_kappa:.1e}, {max_kappa:.1e}] Hz)",
            recommendation="Increase intensity or optimize density gradient" if kappa_Hz < min_kappa else "Reduce parameters to avoid breakdown" if kappa_Hz > max_kappa else "κ is in valid range"
        ))

        # 2. Temperature detection
        min_temp = hawking_thresholds["temperature_min_K"]
        temp_passed = temperature_K >= min_temp
        results.append(PhysicsThresholdResult(
            parameter_name="Hawking Temperature",
            value=temperature_K,
            threshold_value=min_temp,
            unit="K",
            passed=temp_passed,
            margin=temperature_K / min_temp if temp_passed else 0,
            severity="warning" if not temp_passed else "info",
            message=f"T_H = {temperature_K:.2e} K ({'detectable' if temp_passed else 'challenging to detect'})",
            recommendation="Consider enhanced detection methods or longer integration times" if not temp_passed else "Temperature potentially detectable"
        ))

        # 3. Gradient scale for horizon formation
        min_gradient = hawking_thresholds["density_gradient_min_m"]
        max_gradient = hawking_thresholds["density_gradient_max_m"]

        if min_gradient <= gradient_scale_m <= max_gradient:
            gradient_passed = True
            gradient_severity = "info"
        elif gradient_scale_m < min_gradient:
            gradient_passed = False
            gradient_severity = "warning"
        else:
            gradient_passed = False
            gradient_severity = "critical"

        results.append(PhysicsThresholdResult(
            parameter_name="Density Gradient Scale",
            value=gradient_scale_m,
            threshold_value=(min_gradient, max_gradient),
            unit="m",
            passed=gradient_passed,
            margin=min(gradient_scale_m/min_gradient, max_gradient/gradient_scale_m) if gradient_passed else 0,
            severity=gradient_severity,
            message=f"Gradient scale {gradient_scale_m:.2e} m ({'within' if gradient_passed else 'outside'} optimal range)",
            recommendation="Use sharper density gradient" if gradient_scale_m > max_gradient else "Use gentler gradient for stability" if gradient_scale_m < min_gradient else "Gradient scale is optimal"
        ))

        return results

    def _validate_relativistic_effects(
        self, a0: float, flow_velocity_ms: float, plasma_density_m3: float
    ) -> List[PhysicsThresholdResult]:
        """Validate relativistic effects and their impact"""

        results = []

        # 1. Relativistic regime check
        if a0 < 0.1:
            regime = "classical"
            severity = "info"
        elif a0 < 1:
            regime = "weakly relativistic"
            severity = "info"
        elif a0 < 10:
            regime = "moderately relativistic"
            severity = "warning"
        else:
            regime = "strongly relativistic"
            severity = "critical"

        results.append(PhysicsThresholdResult(
            parameter_name="Relativistic Regime",
            value=a0,
            threshold_value=1.0,
            unit="dimensionless",
            passed=a0 < 10,  # Strongly relativistic may break assumptions
            margin=1/a0 if a0 > 0 else 0,
            severity=severity,
            message=f"a₀ = {a0:.2f} ({regime} regime)",
            recommendation="Consider relativistic corrections in analysis" if a0 >= 1 else "Classical treatment valid"
        ))

        # 2. Flow velocity relativistic effects
        v_c = flow_velocity_ms / self.constants.c
        if v_c < 0.1:
            v_regime = "non-relativistic"
            v_severity = "info"
        elif v_c < 0.3:
            v_regime = "weakly relativistic"
            v_severity = "info"
        else:
            v_regime = "relativistic"
            v_severity = "warning"

        results.append(PhysicsThresholdResult(
            parameter_name="Flow Velocity Regime",
            value=v_c,
            threshold_value=0.5,
            unit="fraction of c",
            passed=v_c < 0.5,
            margin=0.5/v_c if v_c > 0 else 0,
            severity=v_severity,
            message=f"v/c = {v_c:.3f} ({v_regime} flow)",
            recommendation="Include relativistic corrections" if v_c >= 0.3 else "Non-relativistic flow valid"
        ))

        # 3. Plasma density effects
        critical_density = 1.9e27 / (800/1000)**2  # Approximate for 800nm
        density_ratio = plasma_density_m3 / critical_density

        if density_ratio < 0.01:
            density_regime = "underdense"
            density_severity = "info"
        elif density_ratio < 1:
            density_regime = "near-critical"
            density_severity = "info"
        else:
            density_regime = "overdense"
            density_severity = "warning"

        results.append(PhysicsThresholdResult(
            parameter_name="Plasma Density Regime",
            value=density_ratio,
            threshold_value=1.0,
            unit="fraction of critical",
            passed=density_ratio <= 10,
            margin=1/density_ratio if density_ratio > 0 else 0,
            severity=density_severity,
            message=f"n/n_c = {density_ratio:.3f} ({density_regime} plasma)",
            recommendation="Consider absorption and reflection effects" if density_ratio >= 1 else "Laser propagation in plasma valid"
        ))

        return results

    def _generate_recommendations(
        self, threshold_results: List[PhysicsThresholdResult], facility: ELIFacility
    ) -> List[str]:
        """Generate facility-specific recommendations"""

        recommendations = []

        # Critical issues first
        critical_issues = [r for r in threshold_results if r.severity == "critical" and not r.passed]
        for issue in critical_issues:
            recommendations.append(f"CRITICAL: {issue.recommendation}")

        # Warnings next
        warnings = [r for r in threshold_results if r.severity == "warning" and not r.passed]
        for warning in warnings:
            recommendations.append(f"WARNING: {warning.recommendation}")

        # Facility-specific recommendations
        if facility == ELIFacility.ELI_BEAMLINES:
            recommendations.extend([
                "Consider L4 ATON system for maximum intensity",
                "Allow 1 minute between shots for thermal recovery",
                "Use E4 experimental hall for optimal conditions",
            ])
        elif facility == ELIFacility.ELI_NP:
            recommendations.extend([
                "Leverage dual-beam capability for enhanced diagnostics",
                "Consider magnetic field effects on plasma dynamics",
                "Allow 5 minutes between shots for nuclear safety",
            ])
        elif facility == ELIFacility.ELI_ALPS:
            recommendations.extend([
                "Optimize for high repetition rate statistics",
                "Use fast target translation system",
                "Focus on parameter space mapping with many shots",
            ])

        # General physics recommendations
        passed_count = sum(1 for r in threshold_results if r.passed)
        total_count = len(threshold_results)

        if passed_count / total_count > 0.8:
            recommendations.append("✅ Configuration is highly feasible")
        elif passed_count / total_count > 0.6:
            recommendations.append("⚠️  Configuration is moderately feasible with optimizations")
        else:
            recommendations.append("❌ Configuration requires significant modifications")

        return recommendations