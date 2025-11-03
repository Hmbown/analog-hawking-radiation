"""
Experimental Feasibility Assessment Methodology for Analog Hawking Radiation

This module provides comprehensive feasibility assessment tools for evaluating
experimental configurations at ELI facilities, including technical readiness,
risk assessment, and success probability estimation.

Author: Claude Analysis Assistant
Date: November 2025
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Import ELI capabilities
from .eli_capabilities import ELICapabilities, ELIFacility


class RiskLevel(Enum):
    """Risk levels for experimental feasibility"""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class TechnicalReadiness(Enum):
    """Technical readiness levels"""
    TRL1 = "Basic principles observed"
    TRL2 = "Technology concept formulated"
    TRL3 = "Analytical & experimental proof of concept"
    TRL4 = "Component validation in laboratory"
    TRL5 = "Component validation in relevant environment"
    TRL6 = "System prototype demonstration in relevant environment"
    TRL7 = "System prototype demonstration in operational environment"
    TRL8 = "System complete and qualified"
    TRL9 = "Actual system proven in operational environment"


@dataclass
class FeasibilityCriteria:
    """Criteria for evaluating experimental feasibility"""

    # Physics requirements
    min_horizon_potential: float = 0.3      # Minimum horizon formation potential
    min_surface_gravity: float = 1e12       # Minimum surface gravity (s^-1)
    max_hawking_temp_ratio: float = 0.5    # Max T_H / T_plasma ratio for detection

    # Technical requirements
    min_plasma_mirror_contrast: float = 1e-10  # Minimum laser contrast
    max_pointing_jitter_mrad: float = 0.05     # Maximum pointing stability
    max_timing_jitter_fs: float = 30            # Maximum timing jitter
    min_vacuum_level_mbar: float = 1e-7         # Required vacuum level

    # Operational requirements
    min_shots_per_configuration: int = 100     # Minimum shots for statistics
    max_setup_time_hours: int = 24             # Maximum setup time
    max_downtime_percent: float = 20.0         # Maximum allowed downtime

    # Safety requirements
    max_radiation_dose_mSv: float = 0.5        # Maximum radiation dose per experiment
    min_shielding_thickness_cm: float = 50     # Minimum radiation shielding


@dataclass
class RiskAssessment:
    """Risk assessment for experimental configuration"""

    technical_risks: List[Tuple[str, RiskLevel, float]] = field(default_factory=list)
    operational_risks: List[Tuple[str, RiskLevel, float]] = field(default_factory=list)
    safety_risks: List[Tuple[str, RiskLevel, float]] = field(default_factory=list)
    physics_risks: List[Tuple[str, RiskLevel, float]] = field(default_factory=list)

    overall_risk_score: float = 0.0
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class FeasibilityAssessment:
    """Comprehensive feasibility assessment result"""

    configuration_id: str
    facility: str
    overall_feasibility_score: float

    # Component scores
    technical_feasibility: float
    physics_feasibility: float
    operational_feasibility: float
    safety_feasibility: float

    # Success metrics
    detection_probability: float
    confidence_interval: Tuple[float, float]
    required_shots: int
    estimated_experiment_time_hours: float

    # Risk assessment
    risk_assessment: RiskAssessment

    # Recommendations
    recommendations: List[str]
    alternative_configurations: List[Dict[str, Any]]

    # Technical readiness
    overall_trl: TechnicalReadiness
    critical_path_items: List[str]

    def is_feasible(self, threshold: float = 0.6) -> bool:
        """Check if configuration meets feasibility threshold"""
        return self.overall_feasibility_score >= threshold


class ExperimentalFeasibilityAssessor:
    """Comprehensive experimental feasibility assessment system"""

    def __init__(self):
        self.eli = ELICapabilities()
        self.criteria = FeasibilityCriteria()
        self.risk_weights = {
            "technical": 0.3,
            "physics": 0.3,
            "operational": 0.2,
            "safety": 0.2
        }

    def assess_configuration(self,
                           parameters: Dict[str, Any],
                           facility: Optional[ELIFacility] = None) -> FeasibilityAssessment:
        """
        Perform comprehensive feasibility assessment of experimental configuration

        Args:
            parameters: Dictionary containing experimental parameters
            facility: Target ELI facility (if specified)

        Returns:
            Comprehensive feasibility assessment
        """

        config_id = self._generate_config_id(parameters)

        print(f"🔍 ASSESSING FEASIBILITY: {config_id}")
        print(f"   Target facility: {facility.value if facility else 'Optimal facility'}")

        # Get ELI compatibility information
        intensity_W_cm2 = parameters["laser_intensity_W_m2"] / 1e4
        wavelength_nm = parameters.get("wavelength_nm", 800)
        pulse_duration_fs = parameters.get("pulse_duration_fs", 150)

        eli_feasibility = self.eli.calculate_feasibility_score(
            intensity_W_cm2, wavelength_nm, pulse_duration_fs, facility
        )

        if not eli_feasibility["feasible"]:
            return self._create_infeasible_assessment(config_id, parameters, eli_feasibility)

        # Assess individual feasibility components
        technical_score = self._assess_technical_feasibility(parameters, eli_feasibility)
        physics_score = self._assess_physics_feasibility(parameters)
        operational_score = self._assess_operational_feasibility(parameters, eli_feasibility)
        safety_score = self._assess_safety_feasibility(parameters, eli_feasibility)

        # Calculate overall feasibility score
        overall_score = (
            self.risk_weights["technical"] * technical_score +
            self.risk_weights["physics"] * physics_score +
            self.risk_weights["operational"] * operational_score +
            self.risk_weights["safety"] * safety_score
        )

        # Calculate detection probability
        detection_prob, confidence_interval = self._calculate_detection_probability(
            parameters, overall_score
        )

        # Assess risks
        risk_assessment = self._assess_risks(parameters, eli_feasibility)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            parameters, technical_score, physics_score, operational_score, safety_score
        )

        # Generate alternative configurations
        alternatives = self._generate_alternatives(parameters, eli_feasibility)

        # Determine technical readiness level
        overall_trl = self._determine_trl(overall_score, risk_assessment)

        # Identify critical path items
        critical_items = self._identify_critical_path(risk_assessment)

        return FeasibilityAssessment(
            configuration_id=config_id,
            facility=eli_feasibility["facility"],
            overall_feasibility_score=overall_score,

            technical_feasibility=technical_score,
            physics_feasibility=physics_score,
            operational_feasibility=operational_score,
            safety_feasibility=safety_score,

            detection_probability=detection_prob,
            confidence_interval=confidence_interval,
            required_shots=self._estimate_required_shots(parameters, detection_prob),
            estimated_experiment_time_hours=self._estimate_experiment_time(parameters),

            risk_assessment=risk_assessment,

            recommendations=recommendations,
            alternative_configurations=alternatives,

            overall_trl=overall_trl,
            critical_path_items=critical_items
        )

    def _generate_config_id(self, parameters: Dict[str, Any]) -> str:
        """Generate unique configuration identifier"""
        intensity_exp = int(np.log10(parameters["laser_intensity_W_m2"]))
        density_exp = int(np.log10(parameters["plasma_density_m3"]))
        return f"I{intensity_exp}_N{density_exp}_A{parameters.get('wavelength_nm', 800)}"

    def _assess_technical_feasibility(self,
                                    parameters: Dict[str, Any],
                                    eli_feasibility: Dict[str, Any]) -> float:
        """Assess technical feasibility of configuration"""

        score = 0.0

        # ELI facility compatibility (40% of technical score)
        score += 0.4 * eli_feasibility["score"]

        # Laser parameter quality (30% of technical score)
        intensity_W_cm2 = parameters["laser_intensity_W_m2"] / 1e4

        # Check intensity range
        if 1e19 <= intensity_W_cm2 <= 1e22:
            intensity_score = 1.0
        elif 1e18 <= intensity_W_cm2 < 1e19 or 1e22 < intensity_W_cm2 <= 1e23:
            intensity_score = 0.8
        elif 1e17 <= intensity_W_cm2 < 1e18 or 1e23 < intensity_W_cm2 <= 1e24:
            intensity_score = 0.6
        else:
            intensity_score = 0.3

        score += 0.3 * intensity_score

        # Plasma formation feasibility (20% of technical score)
        wavelength_nm = parameters.get("wavelength_nm", 800)
        plasma_density = parameters["plasma_density_m3"]

        # Critical density calculation
        omega = 2 * np.pi * 3e8 / (wavelength_nm * 1e-9)
        n_critical = 8.85e-12 * 9.11e-31 * omega**2 / (1.6e-19)**2

        if n_critical <= plasma_density <= 100 * n_critical:
            plasma_score = 1.0
        elif 0.1 * n_critical <= plasma_density < n_critical or 100 * n_critical < plasma_density <= 1000 * n_critical:
            plasma_score = 0.8
        else:
            plasma_score = 0.5

        score += 0.2 * plasma_score

        # Diagnostic availability (10% of technical score)
        # Assume good diagnostic coverage for established facilities
        diagnostic_score = 0.9 if eli_feasibility["facility"] in ["ELI-Beamlines", "ELI-NP"] else 0.8
        score += 0.1 * diagnostic_score

        return min(1.0, score)

    def _assess_physics_feasibility(self, parameters: Dict[str, Any]) -> float:
        """Assess physics feasibility for analog Hawking radiation"""

        score = 0.0

        # Horizon formation potential (40% of physics score)
        intensity_W_m2 = parameters["laser_intensity_W_m2"]
        wavelength_m = parameters.get("wavelength_nm", 800) * 1e-9
        plasma_density = parameters["plasma_density_m3"]
        temperature_K = parameters.get("temperature_K", 1e4)

        # Calculate normalized vector potential
        omega = 2 * np.pi * 3e8 / wavelength_m
        E_field = np.sqrt(2 * intensity_W_m2 / (3e8 * 8.85e-12))
        a0 = 1.6e-19 * E_field / (9.11e-31 * 3e8 * omega)

        # Estimate characteristic velocity
        v_char = 3e8 * a0 / np.sqrt(1 + a0**2)

        # Estimate sound speed
        c_s = np.sqrt(1.38e-23 * temperature_K / 1.67e-27)

        # Horizon formation potential
        if c_s > 0:
            horizon_potential = min(1.0, v_char / c_s)
        else:
            horizon_potential = 0.0

        if horizon_potential >= self.criteria.min_horizon_potential:
            horizon_score = 1.0
        elif horizon_potential >= 0.2:
            horizon_score = 0.7
        elif horizon_potential >= 0.1:
            horizon_score = 0.4
        else:
            horizon_score = 0.1

        score += 0.4 * horizon_score

        # Surface gravity calculation (30% of physics score)
        # Estimate surface gravity based on velocity gradient
        gradient_scale = wavelength_m  # Characteristic gradient scale
        surface_gravity = v_char / gradient_scale

        if surface_gravity >= self.criteria.min_surface_gravity:
            gravity_score = 1.0
        elif surface_gravity >= 1e11:
            gravity_score = 0.8
        elif surface_gravity >= 1e10:
            gravity_score = 0.6
        else:
            gravity_score = 0.3

        score += 0.3 * gravity_score

        # Temperature ratio for detection (20% of physics score)
        # Hawking temperature estimate
        h_bar = 1.05e-34
        k_B = 1.38e-23
        T_H = h_bar * surface_gravity / (2 * np.pi * k_B)

        T_ratio = T_H / temperature_K
        if T_ratio <= self.criteria.max_hawking_temp_ratio:
            temp_score = 1.0
        elif T_ratio <= 2.0:
            temp_score = 0.7
        elif T_ratio <= 5.0:
            temp_score = 0.4
        else:
            temp_score = 0.1

        score += 0.2 * temp_score

        # Signal-to-noise estimate (10% of physics score)
        # Rough SNR estimate based on temperature ratio and plasma conditions
        snr_estimate = T_ratio * np.sqrt(plasma_density / 1e24)
        if snr_estimate >= 0.1:
            snr_score = 1.0
        elif snr_estimate >= 0.05:
            snr_score = 0.7
        elif snr_estimate >= 0.01:
            snr_score = 0.4
        else:
            snr_score = 0.1

        score += 0.1 * snr_score

        return min(1.0, score)

    def _assess_operational_feasibility(self,
                                      parameters: Dict[str, Any],
                                      eli_feasibility: Dict[str, Any]) -> float:
        """Assess operational feasibility"""

        score = 0.0

        # Repetition rate considerations (40% of operational score)
        rep_rate = eli_feasibility.get("repetition_rate_Hz", 1.0)
        if rep_rate >= 1.0:
            rep_score = 1.0  # Good for statistics
        elif rep_rate >= 0.1:
            rep_score = 0.8  # Reasonable
        elif rep_rate >= 0.01:
            rep_score = 0.6  # Low but possible
        else:
            rep_score = 0.3  # Very challenging

        score += 0.4 * rep_score

        # Shot availability (30% of operational score)
        # Estimate total experiment time
        required_shots = self._estimate_required_shots(parameters, 0.5)  # Conservative estimate
        total_time_hours = required_shots / rep_rate / 3600  # Convert to hours

        if total_time_hours <= 8:  # Within single day
            time_score = 1.0
        elif total_time_hours <= 24:  # Within one day
            time_score = 0.8
        elif total_time_hours <= 168:  # Within one week
            time_score = 0.6
        else:
            time_score = 0.3

        score += 0.3 * time_score

        # Setup complexity (20% of operational score)
        wavelength_nm = parameters.get("wavelength_nm", 800)
        magnetic_field = parameters.get("magnetic_field_T", 0)

        # Simpler setup = higher score
        if abs(wavelength_nm - 800) <= 10 and magnetic_field <= 10:
            setup_score = 1.0  # Standard configuration
        elif abs(wavelength_nm - 800) <= 50 and magnetic_field <= 20:
            setup_score = 0.8  # Moderate complexity
        else:
            setup_score = 0.6  # High complexity

        score += 0.2 * setup_score

        # Facility experience (10% of operational score)
        facility = eli_feasibility["facility"]
        if facility in ["ELI-Beamlines", "ELI-NP"]:
            experience_score = 0.9  # More experience with high-intensity experiments
        else:
            experience_score = 0.7  # Less relevant experience

        score += 0.1 * experience_score

        return min(1.0, score)

    def _assess_safety_feasibility(self,
                                 parameters: Dict[str, Any],
                                 eli_feasibility: Dict[str, Any]) -> float:
        """Assess safety feasibility"""

        score = 0.0

        # Radiation safety (40% of safety score)
        intensity_W_cm2 = parameters["laser_intensity_W_m2"] / 1e4

        if intensity_W_cm2 <= 1e21:
            radiation_score = 1.0  # Standard shielding sufficient
        elif intensity_W_cm2 <= 1e22:
            radiation_score = 0.8  # Enhanced shielding required
        elif intensity_W_cm2 <= 1e23:
            radiation_score = 0.6  # Special shielding and procedures
        else:
            radiation_score = 0.3  # Extensive safety measures required

        score += 0.4 * radiation_score

        # Target handling safety (30% of safety score)
        pulse_duration_fs = parameters.get("pulse_duration_fs", 150)
        if pulse_duration_fs >= 50:  # Longer pulses, less target destruction
            target_score = 1.0
        elif pulse_duration_fs >= 30:
            target_score = 0.8
        else:
            target_score = 0.6  # Very short pulses, target fragments

        score += 0.3 * target_score

        # Operational safety (20% of safety score)
        rep_rate = eli_feasibility.get("repetition_rate_Hz", 1.0)
        if rep_rate <= 1.0:
            ops_score = 1.0  # Low repetition, easier safety management
        elif rep_rate <= 10:
            ops_score = 0.8  # Moderate repetition
        else:
            ops_score = 0.6  # High repetition, challenging safety

        score += 0.2 * ops_score

        # Facility safety infrastructure (10% of safety score)
        facility = eli_feasibility["facility"]
        if facility in ["ELI-NP", "ELI-Beamlines"]:
            infra_score = 1.0  # Established high-intensity safety infrastructure
        else:
            infra_score = 0.8  # Good but less high-intensity experience

        score += 0.1 * infra_score

        return min(1.0, score)

    def _calculate_detection_probability(self,
                                       parameters: Dict[str, Any],
                                       feasibility_score: float) -> Tuple[float, Tuple[float, float]]:
        """Calculate detection probability and confidence interval"""

        # Base probability from overall feasibility
        base_prob = feasibility_score * 0.7  # Conservative estimate

        # Adjust for physics parameters
        intensity_W_m2 = parameters["laser_intensity_W_m2"]
        plasma_density = parameters["plasma_density_m3"]

        # Higher intensity and density improve detection probability
        intensity_factor = min(1.5, np.log10(intensity_W_m2 / 1e19) * 0.2 + 1.0)
        density_factor = min(1.3, np.log10(plasma_density / 1e23) * 0.1 + 1.0)

        detection_prob = base_prob * intensity_factor * density_factor
        detection_prob = min(0.95, detection_prob)  # Cap at 95%

        # Calculate confidence interval (±15% relative uncertainty)
        uncertainty = 0.15 * detection_prob
        ci_lower = max(0.0, detection_prob - uncertainty)
        ci_upper = min(1.0, detection_prob + uncertainty)

        return detection_prob, (ci_lower, ci_upper)

    def _estimate_required_shots(self,
                               parameters: Dict[str, Any],
                               detection_probability: float) -> int:
        """Estimate number of shots required for detection"""

        # Base shot count for 5σ detection
        base_shots = 1000

        # Adjust for detection probability
        if detection_probability >= 0.5:
            prob_factor = 1.0
        elif detection_probability >= 0.3:
            prob_factor = 2.0
        elif detection_probability >= 0.1:
            prob_factor = 5.0
        else:
            prob_factor = 10.0

        # Adjust for repetition rate
        wavelength_nm = parameters.get("wavelength_nm", 800)
        if abs(wavelength_nm - 800) <= 10:
            rep_factor = 1.0  # Standard configuration
        else:
            rep_factor = 1.5  # Non-standard needs more shots

        required_shots = int(base_shots * prob_factor * rep_factor)

        return min(100000, required_shots)  # Cap at 100k shots

    def _estimate_experiment_time(self, parameters: Dict[str, Any]) -> float:
        """Estimate total experiment time in hours"""

        required_shots = self._estimate_required_shots(parameters, 0.5)

        # Assume average repetition rate of 0.1 Hz for planning
        avg_rep_rate = 0.1

        # Beam time calculation
        shot_time_hours = required_shots / avg_rep_rate / 3600

        # Add setup, alignment, and diagnostic time
        setup_hours = 24  # Initial setup
        alignment_hours = 8  # Alignment and optimization
        diagnostic_hours = 4  # Diagnostic setup

        total_hours = shot_time_hours + setup_hours + alignment_hours + diagnostic_hours

        return total_hours

    def _assess_risks(self,
                     parameters: Dict[str, Any],
                     eli_feasibility: Dict[str, Any]) -> RiskAssessment:
        """Comprehensive risk assessment"""

        assessment = RiskAssessment()

        # Technical risks
        intensity_W_cm2 = parameters["laser_intensity_W_m2"] / 1e4
        if intensity_W_cm2 > 1e22:
            assessment.technical_risks.append(
                ("High intensity may damage optics", RiskLevel.HIGH, 0.7)
            )

        if eli_feasibility["intensity_margin"] < 2:
            assessment.technical_risks.append(
                ("Low intensity margin reduces operational flexibility", RiskLevel.MEDIUM, 0.5)
            )

        # Physics risks
        a0 = self._calculate_a0(parameters)
        if a0 < 1:
            assessment.physics_risks.append(
                ("Sub-relativistic regime may not form proper horizon", RiskLevel.HIGH, 0.8)
            )

        # Operational risks
        rep_rate = eli_feasibility.get("repetition_rate_Hz", 1.0)
        if rep_rate < 0.01:
            assessment.operational_risks.append(
                ("Very low repetition rate extends experiment time", RiskLevel.MEDIUM, 0.6)
            )

        # Safety risks
        if intensity_W_cm2 > 1e23:
            assessment.safety_risks.append(
                ("Extreme intensity requires extensive safety measures", RiskLevel.HIGH, 0.8)
            )

        # Calculate overall risk score
        all_risks = (assessment.technical_risks + assessment.physics_risks +
                    assessment.operational_risks + assessment.safety_risks)

        if all_risks:
            assessment.overall_risk_score = np.mean([risk[2] for risk in all_risks])
        else:
            assessment.overall_risk_score = 0.1  # Low risk baseline

        # Generate mitigation strategies
        assessment.mitigation_strategies = self._generate_mitigation_strategies(
            assessment.technical_risks + assessment.physics_risks +
            assessment.operational_risks + assessment.safety_risks
        )

        return assessment

    def _calculate_a0(self, parameters: Dict[str, Any]) -> float:
        """Calculate normalized vector potential"""
        intensity_W_m2 = parameters["laser_intensity_W_m2"]
        wavelength_m = parameters.get("wavelength_nm", 800) * 1e-9

        omega = 2 * np.pi * 3e8 / wavelength_m
        E_field = np.sqrt(2 * intensity_W_m2 / (3e8 * 8.85e-12))
        a0 = 1.6e-19 * E_field / (9.11e-31 * 3e8 * omega)

        return a0

    def _generate_mitigation_strategies(self, risks: List[Tuple[str, RiskLevel, float]]) -> List[str]:
        """Generate risk mitigation strategies"""

        strategies = []

        for risk_desc, risk_level, risk_score in risks:
            if "intensity" in risk_desc.lower():
                strategies.append("Implement intensity ramp-up procedures and real-time monitoring")
                strategies.append("Prepare spare optics and establish damage thresholds")

            if "horizon" in risk_desc.lower() or "relativistic" in risk_desc.lower():
                strategies.append("Optimize plasma density profile for enhanced horizon formation")
                strategies.append("Consider pre-pulse for plasma conditioning")

            if "repetition" in risk_desc.lower():
                strategies.append("Schedule additional beam time and optimize data acquisition")
                strategies.append("Implement automated target alignment for faster turnaround")

            if "safety" in risk_desc.lower():
                strategies.append("Implement comprehensive radiation monitoring and shielding verification")
                strategies.append("Establish emergency shutdown procedures and safety protocols")

        # Add general strategies
        strategies.extend([
            "Conduct comprehensive system checkout before high-intensity operations",
            "Implement real-time plasma diagnostics for optimization",
            "Establish clear go/no-go decision points based on diagnostic feedback"
        ])

        return list(set(strategies))  # Remove duplicates

    def _generate_recommendations(self,
                                parameters: Dict[str, Any],
                                technical_score: float,
                                physics_score: float,
                                operational_score: float,
                                safety_score: float) -> List[str]:
        """Generate specific recommendations for configuration improvement"""

        recommendations = []

        # Technical recommendations
        if technical_score < 0.7:
            intensity_W_m2 = parameters["laser_intensity_W_m2"]
            if intensity_W_m2 < 1e19:
                recommendations.append("Consider increasing laser intensity to improve plasma formation")
            elif intensity_W_m2 > 1e23:
                recommendations.append("Consider reducing intensity to improve operational stability")

        # Physics recommendations
        if physics_score < 0.7:
            a0 = self._calculate_a0(parameters)
            if a0 < 1:
                recommendations.append("Increase intensity or adjust wavelength to achieve a0 > 1 for relativistic effects")

            # Check plasma density
            wavelength_nm = parameters.get("wavelength_nm", 800)
            omega = 2 * np.pi * 3e8 / (wavelength_nm * 1e-9)
            n_critical = 8.85e-12 * 9.11e-31 * omega**2 / (1.6e-19)**2
            plasma_density = parameters["plasma_density_m3"]

            if plasma_density < n_critical:
                recommendations.append("Increase plasma density to at least critical density")

        # Operational recommendations
        if operational_score < 0.7:
            recommendations.append("Consider facility with higher repetition rate for better statistics")
            recommendations.append("Optimize experimental setup for faster target replacement")

        # Safety recommendations
        if safety_score < 0.7:
            recommendations.append("Implement additional radiation shielding and monitoring")
            recommendations.append("Review and enhance safety protocols for high-intensity operations")

        return recommendations

    def _generate_alternatives(self,
                             parameters: Dict[str, Any],
                             eli_feasibility: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative configurations"""

        alternatives = []

        # Alternative 1: Higher intensity, same density
        alt1 = parameters.copy()
        alt1["laser_intensity_W_m2"] *= 2.0
        alt1["configuration_id"] = "higher_intensity"
        alternatives.append(alt1)

        # Alternative 2: Optimize density
        wavelength_nm = parameters.get("wavelength_nm", 800)
        omega = 2 * np.pi * 3e8 / (wavelength_nm * 1e-9)
        n_critical = 8.85e-12 * 9.11e-31 * omega**2 / (1.6e-19)**2

        alt2 = parameters.copy()
        alt2["plasma_density_m3"] = n_critical * 10  # 10x critical density
        alt2["configuration_id"] = "optimized_density"
        alternatives.append(alt2)

        # Alternative 3: Different facility
        if eli_feasibility["facility"] != "ELI-Beamlines":
            alt3 = parameters.copy()
            alt3["configuration_id"] = "eli_beamlines_alternative"
            alternatives.append(alt3)

        return alternatives[:3]  # Return top 3 alternatives

    def _determine_trl(self,
                      feasibility_score: float,
                      risk_assessment: RiskAssessment) -> TechnicalReadiness:
        """Determine technical readiness level"""

        if feasibility_score >= 0.8 and risk_assessment.overall_risk_score < 0.3:
            return TechnicalReadiness.TRL7
        elif feasibility_score >= 0.7 and risk_assessment.overall_risk_score < 0.4:
            return TechnicalReadiness.TRL6
        elif feasibility_score >= 0.6 and risk_assessment.overall_risk_score < 0.5:
            return TechnicalReadiness.TRL5
        elif feasibility_score >= 0.5 and risk_assessment.overall_risk_score < 0.6:
            return TechnicalReadiness.TRL4
        else:
            return TechnicalReadiness.TRL3

    def _identify_critical_path(self, risk_assessment: RiskAssessment) -> List[str]:
        """Identify critical path items for experimental success"""

        critical_items = []

        # High-risk items become critical path
        all_risks = (risk_assessment.technical_risks + risk_assessment.physics_risks +
                    risk_assessment.operational_risks + risk_assessment.safety_risks)

        for risk_desc, risk_level, risk_score in all_risks:
            if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
                critical_items.append(risk_desc)

        # Add standard critical items
        critical_items.extend([
            "Laser system performance verification",
            "Plasma target preparation and characterization",
            "Diagnostic system calibration and validation",
            "Data acquisition and analysis pipeline",
            "Safety system verification and approval"
        ])

        return critical_items

    def _create_infeasible_assessment(self,
                                   config_id: str,
                                   parameters: Dict[str, Any],
                                   eli_feasibility: Dict[str, Any]) -> FeasibilityAssessment:
        """Create assessment for infeasible configurations"""

        return FeasibilityAssessment(
            configuration_id=config_id,
            facility="None",
            overall_feasibility_score=0.0,

            technical_feasibility=0.0,
            physics_feasibility=0.0,
            operational_feasibility=0.0,
            safety_feasibility=0.0,

            detection_probability=0.0,
            confidence_interval=(0.0, 0.0),
            required_shots=0,
            estimated_experiment_time_hours=0.0,

            risk_assessment=RiskAssessment(
                overall_risk_score=1.0,
                technical_risks=[("Configuration exceeds ELI capabilities", RiskLevel.CRITICAL, 1.0)]
            ),

            recommendations=[
                "Reduce laser intensity to within ELI facility limits",
                "Consider alternative ELI facility with higher capabilities",
                "Review and adjust experimental parameters"
            ],
            alternative_configurations=[],

            overall_trl=TechnicalReadiness.TRL1,
            critical_path_items=["Resolve ELI compatibility issues"]
        )


# Create global assessor instance
_feasibility_assessor = ExperimentalFeasibilityAssessor()

def assess_experimental_feasibility(parameters: Dict[str, Any],
                                  facility: Optional[str] = None) -> FeasibilityAssessment:
    """
    Assess experimental feasibility of configuration

    Args:
        parameters: Experimental parameters
        facility: Target ELI facility

    Returns:
        Comprehensive feasibility assessment
    """

    target_facility = None
    if facility:
        facility_map = {
            "beamlines": ELIFacility.ELI_BEAMLINES,
            "np": ELIFacility.ELI_NP,
            "alps": ELIFacility.ELI_ALPS
        }
        target_facility = facility_map.get(facility.lower())

    return _feasibility_assessor.assess_configuration(parameters, target_facility)