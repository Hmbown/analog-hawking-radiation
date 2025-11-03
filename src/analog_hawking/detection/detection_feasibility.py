"""
Comprehensive Detection Feasibility Analysis for Analog Hawking Radiation

This module provides realistic detection feasibility assessment including:
- Comprehensive noise modeling (detector, background, plasma, laser system)
- Signal-to-noise ratio analysis with integration time calculations
- Detection strategy assessment for different diagnostic approaches
- ELI facility diagnostic capability evaluation
- Detection feasibility scoring and recommendations

Author: Claude Analysis Assistant
Date: November 2025
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Physical constants
from scipy.constants import e, h, hbar, k
from scipy.stats import norm


class DetectionMethod(Enum):
    """Available detection methods for analog Hawking radiation"""
    RADIO_SPECTROSCOPY = "Radio Spectroscopy"
    OPTICAL_SPECTROSCOPY = "Optical Spectroscopy"
    INTERFEROMETRY = "Interferometry"
    IMAGING = "Imaging"
    CORRELATION = "Quantum Correlation"
    PLASMA_DIAGNOSTICS = "Plasma Diagnostics"


class DiagnosticType(Enum):
    """Diagnostic system types"""
    HETERODYNE = "Heterodyne Receiver"
    BOLometer = "Bolometer"
    CCD = "CCD Camera"
    STREAK_CAMERA = "Streak Camera"
    SPECTROMETER = "Spectrometer"
    INTERFEROMETER = "Interferometer"
    PHOTOMULTIPLIER = "Photomultiplier Tube"
    ANTENNA_ARRAY = "Antenna Array"


class FeasibilityLevel(Enum):
    """Detection feasibility levels"""
    IMPOSSIBLE = "Impossible (SNR < 1)"
    HIGHLY_CHALLENGING = "Highly Challenging (SNR 1-3)"
    CHALLENGING = "Challenging (SNR 3-5)"
    FEASIBLE = "Feasible (SNR 5-10)"
    STRAIGHTFORWARD = "Straightforward (SNR > 10)"


@dataclass
class DetectorCharacteristics:
    """Detector noise and performance characteristics"""

    # Basic parameters
    name: str
    detection_method: DetectionMethod
    frequency_range: Tuple[float, float]  # Hz
    bandwidth: float  # Hz

    # Noise parameters
    noise_temperature: float  # K (system noise temperature)
    efficiency: float  # Quantum/detection efficiency
    noise_figure: float  # dB
    readout_noise: float  # W/√Hz
    dark_current: float  # A

    # Performance parameters
    response_time: float  # s
    dynamic_range: float  # dB
    saturation_power: float  # W

    # Operating conditions
    operating_temperature: float  # K
    cooling_required: bool
    cost_tier: int  # 1-5 (1=lowest, 5=highest)


@dataclass
class NoiseSource:
    """Individual noise source characterization"""

    name: str
    noise_type: str  # thermal, shot, readout, background, plasma, laser
    spectral_density: Callable[[float], float]  # Function of frequency
    frequency_dependence: str  # f^0, f^1, f^2, etc.
    temperature: Optional[float] = None  # K (for thermal noise)
    magnitude: Optional[float] = None  # General magnitude parameter


@dataclass
class SignalCharacteristics:
    """Analog Hawking radiation signal characteristics"""

    # Physical parameters
    hawking_temperature: float  # K
    surface_gravity: float  # s^-1
    peak_frequency: float  # Hz
    bandwidth: float  # Hz

    # Signal strength
    total_power: float  # W
    power_density: float  # W/Hz
    signal_temperature: float  # K (equivalent antenna temperature)

    # Temporal characteristics
    pulse_duration: float  # s
    rise_time: float  # s
    repetition_rate: float  # Hz

    # Spatial characteristics
    emitting_area: float  # m^2
    angular_distribution: str  # isotropic, directional, etc.
    polarization: Optional[str] = None


@dataclass
class DetectionAssessment:
    """Comprehensive detection feasibility assessment"""

    configuration_id: str
    detection_method: DetectionMethod
    detector_type: DiagnosticType

    # Signal and noise analysis
    snr_peak: float
    snr_integrated: float
    snr_optimal: float
    optimal_bandwidth: float
    optimal_integration_time: float

    # Detection feasibility
    feasibility_level: FeasibilityLevel
    detection_probability: float
    confidence_interval: Tuple[float, float]

    # Requirements
    minimum_integration_time: float
    required_shots: int
    total_experiment_time: float

    # Noise breakdown
    noise_breakdown: Dict[str, float]
    dominant_noise_source: str

    # Recommendations
    recommendations: List[str]
    alternative_methods: List[DetectionMethod]

    # Technical requirements
    technical_requirements: Dict[str, Any]
    cost_estimate: str
    timeline_estimate: str


class NoiseModelingFramework:
    """Comprehensive noise modeling for detection systems"""

    def __init__(self):
        self.noise_sources = []
        self.detectors = self._initialize_detector_database()
        self.background_sources = self._initialize_background_models()

    def _initialize_detector_database(self) -> Dict[DiagnosticType, DetectorCharacteristics]:
        """Initialize database of detector characteristics"""

        detectors = {
            DiagnosticType.HETERODYNE: DetectorCharacteristics(
                name="Millimeter-wave Heterodyne Receiver",
                detection_method=DetectionMethod.RADIO_SPECTROSCOPY,
                frequency_range=(1e11, 1e13),  # 0.1-10 THz
                bandwidth=1e9,  # 1 GHz
                noise_temperature=300,  # Room temperature
                efficiency=0.7,
                noise_figure=3.0,  # dB
                readout_noise=1e-19,  # W/√Hz
                dark_current=1e-12,  # A
                response_time=1e-9,  # 1 ns
                dynamic_range=60,  # dB
                saturation_power=1e-3,  # 1 mW
                operating_temperature=300,
                cooling_required=False,
                cost_tier=3
            ),

            DiagnosticType.BOLometer: DetectorCharacteristics(
                name="Cryogenic Bolometer",
                detection_method=DetectionMethod.RADIO_SPECTROSCOPY,
                frequency_range=(1e11, 5e13),  # 0.1-50 THz
                bandwidth=5e12,  # 50 THz
                noise_temperature=0.1,  # 100 mK
                efficiency=0.9,
                noise_figure=0.5,  # dB
                readout_noise=1e-20,  # W/√Hz
                dark_current=1e-15,  # A
                response_time=1e-3,  # 1 ms
                dynamic_range=80,  # dB
                saturation_power=1e-6,  # 1 μW
                operating_temperature=0.1,
                cooling_required=True,
                cost_tier=5
            ),

            DiagnosticType.CCD: DetectorCharacteristics(
                name="Scientific CCD",
                detection_method=DetectionMethod.IMAGING,
                frequency_range=(4e14, 8e14),  # 400-800 nm
                bandwidth=4e14,  # 400 nm
                noise_temperature=200,  # Cooled CCD
                efficiency=0.8,
                noise_figure=2.0,  # dB
                readout_noise=5e-19,  # W/√Hz
                dark_current=1e-4,  # A/cm²
                response_time=1e-6,  # 1 μs
                dynamic_range=70,  # dB
                saturation_power=1e-2,  # 10 mW
                operating_temperature=200,
                cooling_required=True,
                cost_tier=2
            ),

            DiagnosticType.STREAK_CAMERA: DetectorCharacteristics(
                name="Ultrafast Streak Camera",
                detection_method=DetectionMethod.IMAGING,
                frequency_range=(2e14, 1e15),  # 200-1000 nm
                bandwidth=8e14,  # 800 nm
                noise_temperature=300,
                efficiency=0.1,
                noise_figure=5.0,  # dB
                readout_noise=1e-17,  # W/√Hz
                dark_current=1e-8,  # A
                response_time=1e-12,  # 1 ps
                dynamic_range=50,  # dB
                saturation_power=1e-1,  # 100 mW
                operating_temperature=300,
                cooling_required=False,
                cost_tier=4
            ),

            DiagnosticType.SPECTROMETER: DetectorCharacteristics(
                name="Optical Spectrometer",
                detection_method=DetectionMethod.OPTICAL_SPECTROSCOPY,
                frequency_range=(1e14, 1e15),  # 100-1000 nm
                bandwidth=1e14,  # 100 nm
                noise_temperature=300,
                efficiency=0.6,
                noise_figure=4.0,  # dB
                readout_noise=1e-18,  # W/√Hz
                dark_current=1e-10,  # A
                response_time=1e-9,  # 1 ns
                dynamic_range=60,  # dB
                saturation_power=1e-3,  # 1 mW
                operating_temperature=300,
                cooling_required=False,
                cost_tier=2
            ),

            DiagnosticType.INTERFEROMETER: DetectorCharacteristics(
                name="Optical Interferometer",
                detection_method=DetectionMethod.INTERFEROMETRY,
                frequency_range=(4e14, 8e14),  # 400-800 nm
                bandwidth=4e14,  # 400 nm
                noise_temperature=300,
                efficiency=0.85,
                noise_figure=1.0,  # dB
                readout_noise=1e-19,  # W/√Hz
                dark_current=1e-12,  # A
                response_time=1e-15,  # 1 fs
                dynamic_range=90,  # dB
                saturation_power=1e-2,  # 10 mW
                operating_temperature=300,
                cooling_required=False,
                cost_tier=3
            ),

            DiagnosticType.PHOTOMULTIPLIER: DetectorCharacteristics(
                name="Photomultiplier Tube",
                detection_method=DetectionMethod.OPTICAL_SPECTROSCOPY,
                frequency_range=(2e14, 8e14),  # 250-800 nm
                bandwidth=6e14,  # 600 nm
                noise_temperature=300,
                efficiency=0.25,
                noise_figure=3.5,  # dB
                readout_noise=1e-20,  # W/√Hz
                dark_current=1e-9,  # A
                response_time=1e-9,  # 1 ns
                dynamic_range=80,  # dB
                saturation_power=1e-4,  # 100 μW
                operating_temperature=300,
                cooling_required=False,
                cost_tier=2
            ),

            DiagnosticType.ANTENNA_ARRAY: DetectorCharacteristics(
                name="Radio Antenna Array",
                detection_method=DetectionMethod.RADIO_SPECTROSCOPY,
                frequency_range=(1e6, 1e11),  # 1 MHz - 100 GHz
                bandwidth=1e9,  # 1 GHz
                noise_temperature=50,  # Cooled LNA
                efficiency=0.8,
                noise_figure=1.0,  # dB
                readout_noise=1e-21,  # W/√Hz
                dark_current=1e-15,  # A
                response_time=1e-9,  # 1 ns
                dynamic_range=70,  # dB
                saturation_power=1e-6,  # 1 μW
                operating_temperature=50,
                cooling_required=True,
                cost_tier=4
            )
        }

        return detectors

    def _initialize_background_models(self) -> Dict[str, Callable[[float], float]]:
        """Initialize background noise source models"""

        def thermal_noise(f: float, T: float = 300) -> float:
            """Thermal (Johnson-Nyquist) noise spectral density"""
            if T <= 0:
                return 0.0
            # kT W/Hz (single-sided)
            return k * T

        def shot_noise(f: float, photocurrent: float = 1e-6) -> float:
            """Shot noise spectral density from photocurrent"""
            if photocurrent <= 0:
                return 0.0
            # 2eI W/Hz
            return 2 * e * photocurrent

        def readout_noise(f: float, S_readout: float = 1e-19) -> float:
            """Frequency-independent readout noise"""
            return S_readout**2

        def cosmic_background(f: float) -> float:
            """Cosmic microwave background radiation"""
            T_cmb = 2.725  # K
            return k * T_cmb

        def atmospheric_emission(f: float, T_atm: float = 288) -> float:
            """Atmospheric thermal emission"""
            # Simplified model
            tau_atm = 0.1 * (f / 1e12)**2  # Atmospheric opacity
            return k * T_atm * tau_atm

        def plasma_bremsstrahlung(f: float, T_plasma: float = 1e6, n_e: float = 1e24) -> float:
            """Plasma bremsstrahlung emission"""
            # Simplified Gaunt factor approximation
            g_ff = 1.0  # Gaunt factor
            # Emissivity coefficient
            j_nu = 6.8e-38 * n_e**2 * np.sqrt(T_plasma) * g_ff / f
            return j_nu

        def laser_system_noise(f: float) -> float:
            """Laser system noise and jitter"""
            # 1/f noise component
            noise_floor = 1e-24  # W/Hz
            flicker_noise = noise_floor * (1e6 / f) if f > 1e6 else noise_floor
            return flicker_noise

        def electromagnetic_interference(f: float) -> float:
            """Environmental electromagnetic interference"""
            # Higher at low frequencies
            if f < 1e6:
                return 1e-20  # W/Hz
            elif f < 1e9:
                return 1e-22  # W/Hz
            else:
                return 1e-24  # W/Hz

        return {
            "thermal": lambda f: thermal_noise(f, 300),
            "cosmic": cosmic_background,
            "atmospheric": lambda f: atmospheric_emission(f, 288),
            "plasma": lambda f: plasma_bremsstrahlung(f, 1e6, 1e24),
            "laser": laser_system_noise,
            "emi": electromagnetic_interference,
            "shot": lambda f: shot_noise(f, 1e-6),
            "readout": lambda f: readout_noise(f, 1e-19)
        }

    def calculate_total_noise_spectral_density(self,
                                             frequencies: np.ndarray,
                                             detector: DetectorCharacteristics,
                                             include_background: bool = True,
                                             plasma_params: Optional[Dict[str, float]] = None) -> np.ndarray:
        """Calculate total noise spectral density across frequency range"""

        total_noise = np.zeros_like(frequencies)

        # Detector noise (thermal + readout)
        detector_thermal = k * detector.noise_temperature
        detector_readout = detector.readout_noise**2

        total_noise += detector_thermal + detector_readout

        # Dark current shot noise
        if detector.dark_current > 0:
            shot_noise = 2 * e * detector.dark_current
            total_noise += shot_noise

        # Background noise sources
        if include_background:
            for source_name, source_func in self.background_sources.items():
                if source_name == "plasma" and plasma_params:
                    # Use provided plasma parameters
                    T_plasma = plasma_params.get("temperature", 1e6)
                    n_e = plasma_params.get("density", 1e24)

                    def plasma_noise(f):
                        g_ff = 1.0
                        j_nu = 6.8e-38 * n_e**2 * np.sqrt(T_plasma) * g_ff / f
                        return j_nu

                    background_noise = np.array([plasma_noise(f) for f in frequencies])
                else:
                    background_noise = np.array([source_func(f) for f in frequencies])

                total_noise += background_noise

        # Apply efficiency factor (reduces signal, not noise, but affects SNR)
        # and noise figure
        total_noise *= (1 / detector.efficiency) * 10**(detector.noise_figure / 10)

        return total_noise

    def calculate_snr(self,
                     signal: SignalCharacteristics,
                     detector: DetectorCharacteristics,
                     integration_time: float,
                     bandwidth: Optional[float] = None,
                     plasma_params: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Calculate signal-to-noise ratio for detection"""

        if bandwidth is None:
            bandwidth = detector.bandwidth

        # Define frequency range around signal peak
        f_center = signal.peak_frequency
        f_min = max(detector.frequency_range[0], f_center - bandwidth/2)
        f_max = min(detector.frequency_range[1], f_center + bandwidth/2)

        frequencies = np.logspace(np.log10(f_min), np.log10(f_max), 1000)

        # Signal spectral density (simplified thermal spectrum)
        def signal_spectrum(f):
            # Planck distribution approximation for low temperature
            if signal.hawking_temperature <= 0:
                return 0.0
            x = h * f / (k * signal.hawking_temperature)
            if x > 100:  # Avoid overflow
                return 0.0
            try:
                return signal.power_density * (x**3) / (np.exp(x) - 1) if x > 0.01 else signal.power_density * x**2
            except (FloatingPointError, OverflowError, ZeroDivisionError):
                return 0.0

        signal_psd = np.array([signal_spectrum(f) for f in frequencies])

        # Calculate noise spectral density
        noise_psd = self.calculate_total_noise_spectral_density(
            frequencies, detector, include_background=True, plasma_params=plasma_params
        )

        # Integrate signal and noise over bandwidth
        signal_power = np.trapz(signal_psd, frequencies)
        noise_power = np.trapz(noise_psd, frequencies)

        # Scale by integration time and detector characteristics
        signal_power *= detector.efficiency * integration_time
        noise_power *= integration_time

        # Calculate SNR
        if noise_power > 0:
            snr = signal_power / np.sqrt(noise_power)
        else:
            snr = 0.0

        # Optimal SNR (radiometer equation)
        if detector.noise_temperature > 0 and bandwidth > 0:
            snr_radiometer = signal.signal_temperature / detector.noise_temperature * np.sqrt(bandwidth * integration_time)
        else:
            snr_radiometer = 0.0

        return {
            "snr": snr,
            "snr_radiometer": snr_radiometer,
            "signal_power": signal_power,
            "noise_power": noise_power,
            "signal_power_density": signal.power_density,
            "noise_power_density": noise_power / bandwidth if bandwidth > 0 else 0,
            "bandwidth_used": bandwidth,
            "integration_time": integration_time
        }

    def find_optimal_parameters(self,
                               signal: SignalCharacteristics,
                               detector: DetectorCharacteristics,
                               plasma_params: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """Find optimal bandwidth and integration time for detection"""

        # Test different bandwidths
        bandwidths = np.logspace(6, 13, 50)  # 1 MHz to 10 THz
        integration_times = np.logspace(-9, 0, 50)  # 1 ns to 1 s

        best_snr = 0
        best_params = {
            "bandwidth": detector.bandwidth,
            "integration_time": 1e-6,
            "snr": 0
        }

        # Optimize for 5σ detection (minimum integration time)
        target_snr = 5.0

        for bandwidth in bandwidths:
            # Find minimum integration time for target SNR
            for int_time in integration_times:
                snr_result = self.calculate_snr(
                    signal, detector, int_time, bandwidth, plasma_params
                )

                current_snr = snr_result["snr"]

                if current_snr > best_snr:
                    best_snr = current_snr
                    best_params = {
                        "bandwidth": bandwidth,
                        "integration_time": int_time,
                        "snr": current_snr,
                        "signal_power": snr_result["signal_power"],
                        "noise_power": snr_result["noise_power"]
                    }

                # Stop if we've achieved target SNR and this is faster than current best
                if current_snr >= target_snr and int_time < best_params["integration_time"]:
                    best_params.update({
                        "bandwidth": bandwidth,
                        "integration_time": int_time,
                        "snr": current_snr,
                        "signal_power": snr_result["signal_power"],
                        "noise_power": snr_result["noise_power"]
                    })

        return best_params


class DetectionFeasibilityAnalyzer:
    """Comprehensive detection feasibility analysis system"""

    def __init__(self):
        self.noise_model = NoiseModelingFramework()
        self.detection_methods = self._initialize_detection_methods()

    def _initialize_detection_methods(self) -> Dict[DetectionMethod, List[DiagnosticType]]:
        """Initialize detection method to diagnostic type mapping"""

        return {
            DetectionMethod.RADIO_SPECTROSCOPY: [
                DiagnosticType.HETERODYNE,
                DiagnosticType.BOLometer,
                DiagnosticType.ANTENNA_ARRAY
            ],
            DetectionMethod.OPTICAL_SPECTROSCOPY: [
                DiagnosticType.SPECTROMETER,
                DiagnosticType.PHOTOMULTIPLIER
            ],
            DetectionMethod.INTERFEROMETRY: [
                DiagnosticType.INTERFEROMETER
            ],
            DetectionMethod.IMAGING: [
                DiagnosticType.CCD,
                DiagnosticType.STREAK_CAMERA
            ],
            DetectionMethod.CORRELATION: [
                DiagnosticType.INTERFEROMETER,
                DiagnosticType.HETERODYNE
            ],
            DetectionMethod.PLASMA_DIAGNOSTICS: [
                DiagnosticType.SPECTROMETER,
                DiagnosticType.INTERFEROMETER
            ]
        }

    def assess_detection_feasibility(self,
                                   signal: SignalCharacteristics,
                                   plasma_params: Dict[str, float],
                                   target_snr: float = 5.0) -> List[DetectionAssessment]:
        """Comprehensive detection feasibility assessment"""

        assessments = []

        # Test each detection method
        for method, diagnostic_types in self.detection_methods.items():
            for diag_type in diagnostic_types:
                detector = self.noise_model.detectors[diag_type]

                # Check frequency compatibility
                if not (detector.frequency_range[0] <= signal.peak_frequency <= detector.frequency_range[1]):
                    continue  # Skip incompatible detectors

                # Find optimal detection parameters
                optimal_params = self.noise_model.find_optimal_parameters(
                    signal, detector, plasma_params
                )

                # Calculate detailed SNR analysis
                snr_analysis = self.noise_model.calculate_snr(
                    signal, detector,
                    optimal_params["integration_time"],
                    optimal_params["bandwidth"],
                    plasma_params
                )

                # Determine feasibility level
                feasibility_level = self._determine_feasibility_level(snr_analysis["snr"])

                # Calculate detection probability
                detection_prob, confidence_interval = self._calculate_detection_probability(
                    snr_analysis["snr"], optimal_params["integration_time"]
                )

                # Generate recommendations
                recommendations = self._generate_recommendations(
                    method, diag_type, snr_analysis, feasibility_level
                )

                # Calculate requirements
                requirements = self._calculate_requirements(
                    signal, detector, optimal_params, target_snr
                )

                # Noise breakdown
                noise_breakdown = self._analyze_noise_breakdown(
                    signal, detector, plasma_params
                )

                assessment = DetectionAssessment(
                    configuration_id=f"{method.value}_{diag_type.value}",
                    detection_method=method,
                    detector_type=diag_type,

                    snr_peak=snr_analysis["snr"],
                    snr_integrated=snr_analysis["snr_radiometer"],
                    snr_optimal=optimal_params["snr"],
                    optimal_bandwidth=optimal_params["bandwidth"],
                    optimal_integration_time=optimal_params["integration_time"],

                    feasibility_level=feasibility_level,
                    detection_probability=detection_prob,
                    confidence_interval=confidence_interval,

                    minimum_integration_time=requirements["min_integration_time"],
                    required_shots=requirements["required_shots"],
                    total_experiment_time=requirements["total_time"],

                    noise_breakdown=noise_breakdown,
                    dominant_noise_source=requirements["dominant_noise"],

                    recommendations=recommendations,
                    alternative_methods=self._find_alternative_methods(method),

                    technical_requirements=requirements["technical"],
                    cost_estimate=requirements["cost"],
                    timeline_estimate=requirements["timeline"]
                )

                assessments.append(assessment)

        # Sort by feasibility score
        assessments.sort(key=lambda x: x.snr_optimal, reverse=True)

        return assessments

    def _determine_feasibility_level(self, snr: float) -> FeasibilityLevel:
        """Determine detection feasibility level from SNR"""

        if snr < 1:
            return FeasibilityLevel.IMPOSSIBLE
        elif snr < 3:
            return FeasibilityLevel.HIGHLY_CHALLENGING
        elif snr < 5:
            return FeasibilityLevel.CHALLENGING
        elif snr < 10:
            return FeasibilityLevel.FEASIBLE
        else:
            return FeasibilityLevel.STRAIGHTFORWARD

    def _calculate_detection_probability(self,
                                       snr: float,
                                       integration_time: float) -> Tuple[float, Tuple[float, float]]:
        """Calculate detection probability and confidence interval"""

        # Detection probability based on SNR and integration time
        # Using gaussian detection theory
        if snr <= 0:
            return 0.0, (0.0, 0.0)

        # Probability of detecting with 5σ confidence
        detection_prob = norm.cdf(snr - 5)  # Probability that measurement exceeds 5σ threshold

        # Uncertainty in probability (±20% relative)
        uncertainty = 0.2 * detection_prob
        ci_lower = max(0.0, detection_prob - uncertainty)
        ci_upper = min(1.0, detection_prob + uncertainty)

        return detection_prob, (ci_lower, ci_upper)

    def _generate_recommendations(self,
                                method: DetectionMethod,
                                detector_type: DiagnosticType,
                                snr_analysis: Dict[str, float],
                                feasibility_level: FeasibilityLevel) -> List[str]:
        """Generate specific recommendations for detection improvement"""

        recommendations = []

        if feasibility_level in [FeasibilityLevel.IMPOSSIBLE, FeasibilityLevel.HIGHLY_CHALLENGING]:
            recommendations.append("Consider increasing integration time for better statistics")
            recommendations.append("Explore detectors with lower noise temperature")
            recommendations.append("Optimize plasma parameters for stronger signal generation")

        if feasibility_level == FeasibilityLevel.CHALLENGING:
            recommendations.append("Moderate integration time improvements may achieve detection")
            recommendations.append("Consider signal averaging over multiple shots")

        if method == DetectionMethod.RADIO_SPECTROSCOPY:
            if detector_type == DiagnosticType.HETERODYNE:
                recommendations.append("Consider cryogenic cooling to reduce system noise")
                recommendations.append("Implement frequency stabilization for better sensitivity")
            elif detector_type == DiagnosticType.BOLometer:
                recommendations.append("Ensure proper thermal shielding and vibration isolation")

        elif method == DetectionMethod.OPTICAL_SPECTROSCOPY:
            recommendations.append("Implement high-resolution spectrometer for narrow line features")
            recommendations.append("Consider background subtraction techniques")

        elif method == DetectionMethod.INTERFEROMETRY:
            recommendations.append("Maintain phase stability better than λ/100")
            recommendations.append("Implement active vibration control")

        return recommendations

    def _calculate_requirements(self,
                              signal: SignalCharacteristics,
                              detector: DetectorCharacteristics,
                              optimal_params: Dict[str, Any],
                              target_snr: float) -> Dict[str, Any]:
        """Calculate technical and operational requirements"""

        # Integration time for target SNR
        if optimal_params["snr"] > 0:
            integration_factor = (target_snr / optimal_params["snr"])**2
            min_integration_time = optimal_params["integration_time"] * integration_factor
        else:
            min_integration_time = float('inf')

        # Required shots
        shot_duration = signal.pulse_duration + detector.response_time
        required_shots = int(np.ceil(min_integration_time / shot_duration))

        # Total experiment time (including setup)
        total_time = required_shots * (shot_duration + 0.1) + 24  # 24h setup time

        # Cost estimate
        cost_map = {1: "$10k-$50k", 2: "$50k-$100k", 3: "$100k-$500k",
                   4: "$500k-$1M", 5: ">$1M"}

        # Timeline
        timeline_map = {
            1: "1-3 months",
            2: "3-6 months",
            3: "6-12 months",
            4: "1-2 years",
            5: "2+ years"
        }

        # Technical requirements
        technical = {
            "detector_cooling": detector.cooling_required,
            "vacuum_level": "1e-7 mbar or better" if detector.cost_tier >= 4 else "1e-5 mbar",
            "magnetic_shielding": detector.cost_tier >= 3,
            "vibration_isolation": detector.detection_method
            in {DetectionMethod.INTERFEROMETRY, DetectionMethod.IMAGING},
            "data_acquisition_rate": 1/optimal_params["integration_time"],
            "storage_requirements": f"{required_shots * 100} MB"  # Rough estimate
        }

        return {
            "min_integration_time": min_integration_time,
            "required_shots": min(required_shots, 100000),  # Cap at 100k shots
            "total_time": total_time,
            "dominant_noise": "thermal" if detector.noise_temperature > 100 else "readout",
            "technical": technical,
            "cost": cost_map[detector.cost_tier],
            "timeline": timeline_map[detector.cost_tier]
        }

    def _analyze_noise_breakdown(self,
                               signal: SignalCharacteristics,
                               detector: DetectorCharacteristics,
                               plasma_params: Dict[str, float]) -> Dict[str, float]:
        """Analyze contribution of different noise sources"""

        f_center = signal.peak_frequency

        # Calculate individual noise contributions
        noise_sources = {}

        # Detector thermal noise
        noise_sources["detector_thermal"] = k * detector.noise_temperature

        # Detector readout noise
        noise_sources["readout"] = detector.readout_noise**2

        # Dark current shot noise
        if detector.dark_current > 0:
            noise_sources["shot"] = 2 * e * detector.dark_current

        # Background sources
        for source_name, source_func in self.noise_model.background_sources.items():
            if source_name == "plasma":
                T_plasma = plasma_params.get("temperature", 1e6)
                n_e = plasma_params.get("density", 1e24)
                g_ff = 1.0
                j_nu = 6.8e-38 * n_e**2 * np.sqrt(T_plasma) * g_ff / f_center
                noise_sources[source_name] = j_nu
            else:
                noise_sources[source_name] = source_func(f_center)

        # Apply efficiency and noise figure
        total_noise = sum(noise_sources.values())
        adjusted_total = total_noise * (1 / detector.efficiency) * 10**(detector.noise_figure / 10)

        # Normalize to percentages
        if adjusted_total > 0:
            for key in noise_sources:
                noise_sources[key] = 100 * noise_sources[key] / adjusted_total

        return noise_sources

    def _find_alternative_methods(self, primary_method: DetectionMethod) -> List[DetectionMethod]:
        """Find alternative detection methods"""

        alternatives = []

        if primary_method == DetectionMethod.RADIO_SPECTROSCOPY:
            alternatives.extend([DetectionMethod.OPTICAL_SPECTROSCOPY, DetectionMethod.INTERFEROMETRY])
        elif primary_method == DetectionMethod.OPTICAL_SPECTROSCOPY:
            alternatives.extend([DetectionMethod.RADIO_SPECTROSCOPY, DetectionMethod.IMAGING])
        elif primary_method == DetectionMethod.INTERFEROMETRY:
            alternatives.extend([DetectionMethod.OPTICAL_SPECTROSCOPY, DetectionMethod.CORRELATION])
        elif primary_method == DetectionMethod.IMAGING:
            alternatives.extend([DetectionMethod.OPTICAL_SPECTROSCOPY, DetectionMethod.PLASMA_DIAGNOSTICS])

        return alternatives[:3]  # Return top 3 alternatives

    def generate_feasibility_report(self,
                                  assessments: List[DetectionAssessment],
                                  output_path: Optional[str] = None) -> str:
        """Generate comprehensive feasibility report"""

        report_lines = []
        report_lines.append("# Comprehensive Detection Feasibility Analysis Report")
        report_lines.append("=" * 60)
        report_lines.append(f"Analysis Date: {np.datetime64('now')}")
        report_lines.append(f"Number of Detection Methods Assessed: {len(assessments)}")
        report_lines.append("")

        # Summary table
        report_lines.append("## Detection Feasibility Summary")
        report_lines.append("")
        report_lines.append("| Method | Detector | SNR | Feasibility | Detection Probability | Integration Time |")
        report_lines.append("|--------|----------|-----|-------------|----------------------|------------------|")

        for assessment in assessments[:10]:  # Top 10 methods
            report_lines.append(f"| {assessment.detection_method.value} | "
                              f"{assessment.detector_type.value} | "
                              f"{assessment.snr_optimal:.2f} | "
                              f"{assessment.feasibility_level.value} | "
                              f"{assessment.detection_probability:.1%} | "
                              f"{assessment.optimal_integration_time:.2e} s |")

        report_lines.append("")

        # Detailed analysis for top methods
        report_lines.append("## Detailed Analysis of Top Detection Methods")
        report_lines.append("")

        for i, assessment in enumerate(assessments[:5]):  # Top 5 detailed
            report_lines.append(f"### {i+1}. {assessment.detection_method.value} with {assessment.detector_type.value}")
            report_lines.append("")
            report_lines.append(f"**Feasibility Level:** {assessment.feasibility_level.value}")
            report_lines.append(f"**Optimal SNR:** {assessment.snr_optimal:.2f}")
            report_lines.append(f"**Detection Probability:** {assessment.detection_probability:.1%} "
                              f"(95% CI: {assessment.confidence_interval[0]:.1%} - {assessment.confidence_interval[1]:.1%})")
            report_lines.append("")

            report_lines.append("**Optimal Parameters:**")
            report_lines.append(f"- Bandwidth: {assessment.optimal_bandwidth:.2e} Hz")
            report_lines.append(f"- Integration Time: {assessment.optimal_integration_time:.2e} s")
            report_lines.append(f"- Required Shots: {assessment.required_shots}")
            report_lines.append(f"- Total Experiment Time: {assessment.total_experiment_time:.1f} hours")
            report_lines.append("")

            report_lines.append("**Noise Breakdown:**")
            for source, contribution in assessment.noise_breakdown.items():
                if contribution > 1:  # Only show significant contributors
                    report_lines.append(f"- {source}: {contribution:.1f}%")
            report_lines.append(f"- Dominant Noise Source: {assessment.dominant_noise_source}")
            report_lines.append("")

            report_lines.append("**Recommendations:**")
            for rec in assessment.recommendations:
                report_lines.append(f"- {rec}")
            report_lines.append("")

            report_lines.append("**Technical Requirements:**")
            for req, value in assessment.technical_requirements.items():
                report_lines.append(f"- {req}: {value}")
            report_lines.append("")

            report_lines.append(f"**Cost Estimate:** {assessment.cost_estimate}")
            report_lines.append(f"**Timeline:** {assessment.timeline_estimate}")
            report_lines.append("")

        # Near-term achievable goals
        report_lines.append("## Near-Term Achievable Detection Goals")
        report_lines.append("")

        feasible_methods = [a for a in assessments if a.feasibility_level in
                          [FeasibilityLevel.FEASIBLE, FeasibilityLevel.STRAIGHTFORWARD]]

        if feasible_methods:
            report_lines.append("### Immediately Feasible Detection Methods:")
            for method in feasible_methods:
                report_lines.append(f"- **{method.detection_method.value}**: "
                                  f"SNR {method.snr_optimal:.1f}, "
                                  f"requires {method.required_shots} shots, "
                                  f"estimated {method.cost_estimate}")
        else:
            report_lines.append("### Most Promising Detection Methods (requiring optimization):")
            for method in assessments[:3]:
                report_lines.append(f"- **{method.detection_method.value}**: "
                                  f"SNR {method.snr_optimal:.1f} (target: 5+), "
                                  f"requires signal enhancement by factor {5/max(0.1, method.snr_optimal):.1f}")

        report_lines.append("")

        # ELI facility recommendations
        report_lines.append("## ELI Facility Diagnostic Integration")
        report_lines.append("")
        report_lines.append("### Recommended ELI Facility Diagnostics:")
        report_lines.append("- **High-bandwidth heterodyne receivers** for microwave/THz detection")
        report_lines.append("- **Optical spectrometers** with cryogenic CCDs for visible/IR detection")
        report_lines.append("- **Ultrafast streak cameras** for temporal resolution")
        report_lines.append("- **Interferometric diagnostics** for phase-sensitive detection")
        report_lines.append("")

        report_lines.append("### Facility Integration Requirements:")
        report_lines.append("- **Timing synchronization** better than 100 fs")
        report_lines.append("- **Vacuum compatibility** with existing target chambers")
        report_lines.append("- **Radiation shielding** for high-intensity operations")
        report_lines.append("- **Data acquisition** integration with ELI control systems")
        report_lines.append("")

        # Conclusion and recommendations
        report_lines.append("## Conclusions and Strategic Recommendations")
        report_lines.append("")

        best_method = assessments[0] if assessments else None

        if best_method and best_method.snr_optimal >= 5:
            report_lines.append("✅ **Detection is feasible** with current technology and ELI facilities.")
            report_lines.append(f"**Recommended primary method:** {best_method.detection_method.value}")
            report_lines.append(f"**Expected detection timeline:** {best_method.timeline_estimate}")
        elif best_method and best_method.snr_optimal >= 1:
            report_lines.append("⚠️ **Detection is challenging but achievable** with optimization.")
            report_lines.append("**Required improvements:**")
            report_lines.append("- Enhanced detector sensitivity (cryogenic cooling)")
            report_lines.append("- Signal averaging over multiple shots")
            report_lines.append("- Optimized plasma parameters for stronger signal")
        else:
            report_lines.append("❌ **Detection is not currently feasible** with predicted signal levels.")
            report_lines.append("**Required breakthroughs:**")
            report_lines.append("- Significant signal enhancement (10-100×)")
            report_lines.append("- Revolutionary detector technology")
            report_lines.append("- Alternative detection paradigms")

        report_lines.append("")
        report_lines.append("---")
        report_lines.append("*Report generated by Analog Hawking Radiation Detection Feasibility Analyzer*")

        report = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            Path(output_path).write_text(report)

        return report


# Convenience functions for external use
def create_signal_characteristics(hawking_temperature: float,
                                 surface_gravity: float,
                                 peak_frequency: float,
                                 emitting_area: float = 1e-12) -> SignalCharacteristics:
    """Create signal characteristics from basic parameters"""

    # Calculate signal power using Stefan-Boltzmann-like law for Hawking radiation
    # P = σ A T^4 (adjusted for Hawking radiation)
    sigma_eff = hbar / (240 * np.pi**2)  # Effective radiation constant for Hawking radiation
    total_power = sigma_eff * emitting_area * hawking_temperature**4

    # Power density
    bandwidth = peak_frequency / 10  # Assume 10% fractional bandwidth
    power_density = total_power / bandwidth

    # Signal temperature (radiometer equation)
    signal_temperature = total_power / (k * bandwidth)

    return SignalCharacteristics(
        hawking_temperature=hawking_temperature,
        surface_gravity=surface_gravity,
        peak_frequency=peak_frequency,
        bandwidth=bandwidth,
        total_power=total_power,
        power_density=power_density,
        signal_temperature=signal_temperature,
        pulse_duration=1e-12,  # 1 ps typical
        rise_time=1e-13,  # 0.1 ps
        repetition_rate=1.0,  # 1 Hz
        emitting_area=emitting_area,
        angular_distribution="isotropic"
    )


def assess_detection_feasibility_simple(hawking_temperature: float,
                                      surface_gravity: float,
                                      peak_frequency: float,
                                      plasma_temperature: float = 1e6,
                                      plasma_density: float = 1e24,
                                      emitting_area: float = 1e-12) -> List[DetectionAssessment]:
    """Simple interface for detection feasibility assessment"""

    analyzer = DetectionFeasibilityAnalyzer()

    signal = create_signal_characteristics(
        hawking_temperature, surface_gravity, peak_frequency, emitting_area
    )

    plasma_params = {
        "temperature": plasma_temperature,
        "density": plasma_density
    }

    return analyzer.assess_detection_feasibility(signal, plasma_params)
