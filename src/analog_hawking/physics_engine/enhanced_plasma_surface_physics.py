"""
Enhanced Plasma-Surface Interaction Physics for Analog Hawking Radiation Analysis

This module implements comprehensive plasma-surface interaction models for high-intensity
laser-plasma interactions at ELI facility conditions. It addresses the critical missing
plasma-surface physics identified in the scientific review:

1. Detailed plasma mirror formation dynamics
2. Surface roughness and pre-plasma effects
3. Absorption mechanisms (Brunel heating, J×B heating, vacuum heating)
4. Reflection dynamics with proper boundary conditions
5. Pre-plasma expansion and scale length evolution

Author: Enhanced Physics Implementation
Date: November 2025
References:
- Gibbon, Short Pulse Laser Interactions with Matter
- R. Fedosejevs et al., Plasma Mirror Physics
- M. Tabak et al., Ignition and High Gain with Ultra Powerful Lasers
- ELI Beamlines Plasma Mirror Documentation

NOTE: Experimental scaffolding; coefficients and scaling laws require expert
validation before quantitative use.
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, hbar, k, m_p, pi
from scipy.special import erf, jv, yv
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable

class PlasmaMirrorFormation:
    """
    Comprehensive plasma mirror formation model including ionization dynamics
    and surface evolution
    """

    def __init__(self, target_material: str, initial_density: float = 1e28):
        """
        Initialize plasma mirror formation model

        Args:
            target_material: Target material name
            initial_density: Initial solid density in m^-3
        """
        self.target_material = target_material
        self.n_solid = initial_density
        self.critical_density_map = self._get_critical_density_map()

        # Material properties
        self.material_properties = self._get_material_properties()

    def _get_critical_density_map(self) -> Dict[float, float]:
        """Get critical density as function of wavelength"""
        wavelengths = np.logspace(-7, -5, 100)  # 10 nm to 10 μm
        critical_densities = epsilon_0 * m_e * (2 * pi * c / wavelengths)**2 / e**2
        return {float(w): float(n_c) for w, n_c in zip(wavelengths, critical_densities)}

    def _get_material_properties(self) -> Dict:
        """Get material-specific properties"""
        properties = {
            'Al': {
                'Z': 13,
                'A': 27,
                'work_function': 4.08 * e,  # Joules
                'ionization_potentials': [5.99, 18.8, 28.4]  # First few in eV
            },
            'Si': {
                'Z': 14,
                'A': 28,
                'work_function': 4.85 * e,
                'ionization_potentials': [8.15, 16.3, 33.5]
            },
            'Au': {
                'Z': 79,
                'A': 197,
                'work_function': 5.31 * e,
                'ionization_potentials': [9.22, 20.5]
            }
        }
        return properties.get(self.target_material, properties['Al'])

    def ionization_threshold_intensity(self, wavelength: float) -> float:
        """
        Calculate ionization threshold intensity for plasma mirror formation

        Args:
            wavelength: Laser wavelength in meters

        Returns:
            Ionization threshold intensity in W/m^2
        """
        # Critical density at given wavelength
        omega_l = 2 * pi * c / wavelength
        n_critical = epsilon_0 * m_e * omega_l**2 / e**2

        # Barrier suppression ionization field
        # E_BSI = 16 * Ip^2 / (9 * e^3 * a0^2) where a0 is Bohr radius
        a0 = 5.29e-11  # Bohr radius in meters
        Ip = self.material_properties['ionization_potentials'][0] * e  # First ionization potential

        E_BSI = 16 * Ip**2 / (9 * e**3 * a0**2)
        I_BSI = 0.5 * epsilon_0 * c * E_BSI**2

        return I_BSI

    def pre_plasma_scale_length(self, intensity: float, pulse_duration: float,
                              wavelength: float) -> float:
        """
        Calculate pre-plasma scale length from laser prepulse

        Args:
            intensity: Laser intensity in W/m^2
            pulse_duration: Pulse duration in seconds
            wavelength: Laser wavelength in meters

        Returns:
            Pre-plasma scale length in meters
        """
        # Simplified model: L ~ c_s * t_expansion
        # where c_s is sound speed and t_expansion is characteristic time

        # Characteristic sound speed (assuming 1 keV temperature)
        T_e = 1e3 * e  # 1 keV in Joules
        m_i = self.material_properties['A'] * m_p
        c_s = np.sqrt(self.material_properties['Z'] * k * T_e / m_i)

        # Expansion time (prepulse duration estimate)
        t_expansion = 1e-12  # 1 ps typical prepulse timescale

        L_pre = c_s * t_expansion

        # Intensity-dependent correction
        I_normalized = intensity / 1e18  # Normalize to 10^18 W/m^2
        L_pre *= (1 + np.log10(I_normalized)) if I_normalized > 1 else 1

        return min(L_pre, wavelength)  # Don't exceed wavelength

    def plasma_mirror_reflectivity(self, intensity: float, wavelength: float,
                                 scale_length: float, incident_angle: float = 0) -> float:
        """
        Calculate plasma mirror reflectivity

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            scale_length: Plasma scale length in meters
            incident_angle: Incident angle in radians

        Returns:
            Plasma mirror reflectivity (0-1)
        """
        # Critical density
        omega_l = 2 * pi * c / wavelength
        n_critical = epsilon_0 * m_e * omega_l**2 / e**2

        # Normalize scale length to wavelength
        k_L = 2 * pi / wavelength
        normalized_scale_length = k_L * scale_length

        # Reflectivity model based on scale length
        if normalized_scale_length < 0.1:
            # Sharp interface - high reflectivity
            R = 0.95
        elif normalized_scale_length < 1:
            # Moderate scale length - reduced reflectivity
            R = 0.95 * np.exp(-2 * normalized_scale_length)
        else:
            # Long scale length - low reflectivity
            R = 0.1

        # Intensity-dependent correction
        a0 = self.calculate_a0(intensity, wavelength)
        if a0 > 1:
            # Relativistic effects reduce reflectivity
            R *= (1 - 0.1 * np.log10(a0))

        # Angular dependence
        cos_theta = np.cos(incident_angle)
        R *= cos_theta**0.5

        return np.clip(R, 0, 1)

    def calculate_a0(self, intensity: float, wavelength: float) -> float:
        """Calculate normalized vector potential a0"""
        E_0 = np.sqrt(2 * intensity / (c * epsilon_0))
        omega_l = 2 * pi * c / wavelength
        return e * E_0 / (m_e * omega_l * c)


class SurfaceRoughnessEffects:
    """
    Surface roughness effects on plasma mirror performance
    """

    def __init__(self, roughness_rms: float = 1e-9, correlation_length: float = 10e-9):
        """
        Initialize surface roughness model

        Args:
            roughness_rms: RMS roughness in meters
            correlation_length: Surface correlation length in meters
        """
        self.sigma = roughness_rms
        self.l_c = correlation_length

    def roughness_reflection_loss(self, wavelength: float, incident_angle: float = 0) -> float:
        """
        Calculate reflection loss due to surface roughness

        Args:
            wavelength: Laser wavelength in meters
            incident_angle: Incident angle in radians

        Returns:
            Reflection loss factor (0-1)
        """
        # Rayleigh criterion for roughness
        k = 2 * pi / wavelength
        cos_theta = np.cos(incident_angle)

        # Roughness parameter
        roughness_param = (4 * pi * self.sigma * cos_theta / wavelength)**2

        # Reflection loss factor
        loss_factor = np.exp(-roughness_param)

        return loss_factor

    def scattering_distribution(self, wavelength: float) -> Tuple[Callable, float]:
        """
        Get angular scattering distribution and total scattered power

        Args:
            wavelength: Laser wavelength in meters

        Returns:
            Tuple of (scattering_function, total_scattered_power)
        """
        k = 2 * pi / wavelength

        # Simplified Gaussian scattering model
        def scattering_function(theta):
            # theta: scattering angle in radians
            return np.exp(-(k * self.sigma * np.sin(theta))**2) / (pi * self.l_c**2)

        # Total scattered power (simplified)
        P_scattered = (k * self.sigma)**2

        return scattering_function, P_scattered

    def enhanced_absorption(self, intensity: float, wavelength: float) -> float:
        """
        Calculate enhanced absorption due to surface roughness

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters

        Returns:
            Enhanced absorption fraction (0-1)
        """
        # Roughness-induced absorption
        k = 2 * pi / wavelength
        roughness_factor = (k * self.sigma)**2

        # Intensity-dependent enhancement
        a0 = np.sqrt(2 * intensity / (epsilon_0 * c)) * e / (m_e * (2 * pi * c / wavelength) * c)

        if a0 > 1:
            # Relativistic regime - enhanced absorption
            absorption = roughness_factor * (1 + 0.1 * np.log10(a0))
        else:
            # Non-relativistic regime
            absorption = roughness_factor * 0.1

        return np.clip(absorption, 0, 1)


class AbsorptionMechanisms:
    """
    Various absorption mechanisms in laser-plasma interactions
    """

    def __init__(self):
        """Initialize absorption mechanisms model"""
        pass

    def brunel_heating(self, intensity: float, wavelength: float,
                      scale_length: float) -> float:
        """
        Calculate Brunel (vacuum) heating absorption

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            scale_length: Plasma scale length in meters

        Returns:
            Brunel absorption fraction (0-1)
        """
        # Brunel heating is dominant for steep density gradients
        k_L = 2 * pi / wavelength
        normalized_scale_length = k_L * scale_length

        # Absorption coefficient
        if normalized_scale_length < 0.1:
            # Sharp gradient - strong Brunel heating
            a0 = self.calculate_a0(intensity, wavelength)
            eta_brunel = 0.1 * a0 / (1 + a0)
        else:
            # Gentle gradient - reduced Brunel heating
            eta_brunel = 0.01 * np.exp(-normalized_scale_length)

        return np.clip(eta_brunel, 0, 1)

    def jxb_heating(self, intensity: float, wavelength: float,
                   electron_temperature: float) -> float:
        """
        Calculate J×B (ponderomotive) heating absorption

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            electron_temperature: Electron temperature in Joules

        Returns:
            J×B absorption fraction (0-1)
        """
        # J×B heating scales with intensity and inversely with temperature
        a0 = self.calculate_a0(intensity, wavelength)

        # Characteristic absorption
        eta_jxb = 0.05 * a0**2 / (1 + a0**2)

        # Temperature correction
        T_e_keV = electron_temperature / (1e3 * e)
        eta_jxb *= np.exp(-T_e_keV / 10)  # Reduced at high temperature

        return np.clip(eta_jxb, 0, 1)

    def resonance_absorption(self, intensity: float, wavelength: float,
                           scale_length: float, incident_angle: float) -> float:
        """
        Calculate resonance absorption at critical density

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            scale_length: Plasma scale length in meters
            incident_angle: Incident angle in radians

        Returns:
            Resonance absorption fraction (0-1)
        """
        # Resonance absorption requires p-polarized light and oblique incidence
        k_L = 2 * pi / wavelength
        normalized_scale_length = k_L * scale_length

        # Optimal angle for resonance absorption
        sin_theta_opt = np.sqrt(normalized_scale_length / (1 + normalized_scale_length))

        # Absorption coefficient
        theta_abs = abs(incident_angle - np.arcsin(sin_theta_opt))
        eta_resonance = 0.3 * np.exp(-theta_abs**2 / (0.1)**2)

        # Scale length dependence
        eta_resonance *= np.exp(-normalized_scale_length)

        return np.clip(eta_resonance, 0, 1)

    def vacuum_heating(self, intensity: float, wavelength: float) -> float:
        """
        Calculate vacuum heating absorption

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters

        Returns:
            Vacuum heating absorption fraction (0-1)
        """
        # Vacuum heating for overdense plasma
        a0 = self.calculate_a0(intensity, wavelength)

        # Absorption increases with a0
        eta_vacuum = 0.2 * a0 / (1 + a0**2)

        return np.clip(eta_vacuum, 0, 1)

    def calculate_a0(self, intensity: float, wavelength: float) -> float:
        """Calculate normalized vector potential a0"""
        E_0 = np.sqrt(2 * intensity / (c * epsilon_0))
        omega_l = 2 * pi * c / wavelength
        return e * E_0 / (m_e * omega_l * c)

    def total_absorption(self, intensity: float, wavelength: float,
                        scale_length: float, incident_angle: float,
                        electron_temperature: float, polarization: str = 'p') -> float:
        """
        Calculate total absorption from all mechanisms

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            scale_length: Plasma scale length in meters
            incident_angle: Incident angle in radians
            electron_temperature: Electron temperature in Joules
            polarization: Laser polarization ('p' or 's')

        Returns:
            Total absorption fraction (0-1)
        """
        # Individual absorption mechanisms
        eta_brunel = self.brunel_heating(intensity, wavelength, scale_length)
        eta_jxb = self.jxb_heating(intensity, wavelength, electron_temperature)
        eta_vacuum = self.vacuum_heating(intensity, wavelength)

        # Resonance absorption only for p-polarization
        eta_resonance = 0
        if polarization.lower() == 'p':
            eta_resonance = self.resonance_absorption(intensity, wavelength,
                                                     scale_length, incident_angle)

        # Total absorption (approximate - mechanisms aren't strictly additive)
        eta_total = eta_brunel + eta_jxb + eta_vacuum + eta_resonance
        eta_total = min(eta_total, 0.8)  # Cap at 80% absorption

        return eta_total


class PlasmaDynamicsAtSurface:
    """
    Plasma dynamics specifically at the surface-plasma interface
    """

    def __init__(self, target_material: str):
        """
        Initialize plasma surface dynamics model

        Args:
            target_material: Target material name
        """
        self.target_material = target_material
        self.plasma_mirror = PlasmaMirrorFormation(target_material)
        self.roughness = SurfaceRoughnessEffects()
        self.absorption = AbsorptionMechanisms()

    def surface_expansion_velocity(self, intensity: float, scale_length: float) -> float:
        """
        Calculate surface expansion velocity

        Args:
            intensity: Laser intensity in W/m^2
            scale_length: Plasma scale length in meters

        Returns:
            Expansion velocity in m/s
        """
        # Radiation pressure driven expansion
        P_rad = 2 * intensity / c  # Perfect reflector assumption
        mass_density = 2700 if self.target_material == 'Al' else 2330  # kg/m^3

        # Expansion velocity from pressure balance
        v_exp = np.sqrt(2 * P_rad / mass_density)

        # Scale length correction
        k_L = 2 * pi / 800e-9  # Assuming 800 nm wavelength
        normalized_scale_length = k_L * scale_length
        v_exp *= np.exp(-normalized_scale_length)

        return v_exp

    def electron_temperature_at_surface(self, intensity: float, wavelength: float) -> float:
        """
        Estimate electron temperature at the plasma surface

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters

        Returns:
            Electron temperature in Joules
        """
        # Scaling law for hot electron temperature
        # T_h ≈ m_e*c²*(√(1 + a0²/2) - 1)
        a0 = self.plasma_mirror.calculate_a0(intensity, wavelength)

        T_hot = m_e * c**2 * (np.sqrt(1 + a0**2/2) - 1)

        # Limit to reasonable range
        T_hot = min(T_hot, 10e6 * e)  # Cap at 10 MeV

        return T_hot

    def plasma_frequency_at_surface(self, density: float) -> float:
        """
        Calculate plasma frequency at the surface

        Args:
            density: Electron density in m^-3

        Returns:
            Plasma frequency in rad/s
        """
        return np.sqrt(e**2 * density / (epsilon_0 * m_e))

    def skin_depth_at_surface(self, density: float, wavelength: float) -> float:
        """
        Calculate skin depth at the plasma surface

        Args:
            density: Electron density in m^-3
            wavelength: Laser wavelength in meters

        Returns:
            Skin depth in meters
        """
        omega_l = 2 * pi * c / wavelength
        omega_pe = self.plasma_frequency_at_surface(density)

        # Skin depth: δ = c/√(ω_l² - ω_pe²)
        if omega_pe < omega_l:
            delta = c / np.sqrt(omega_l**2 - omega_pe**2)
        else:
            # Overdense plasma
            delta = c / omega_pe

        return delta

    def full_surface_interaction(self, intensity: float, wavelength: float,
                               pulse_duration: float, incident_angle: float = 0,
                               polarization: str = 'p') -> Dict[str, float]:
        """
        Complete surface interaction model

        Args:
            intensity: Laser intensity in W/m^2
            wavelength: Laser wavelength in meters
            pulse_duration: Pulse duration in seconds
            incident_angle: Incident angle in radians
            polarization: Laser polarization

        Returns:
            Dictionary with surface interaction parameters
        """
        # Calculate pre-plasma scale length
        scale_length = self.plasma_mirror.pre_plasma_scale_length(
            intensity, pulse_duration, wavelength)

        # Calculate electron temperature
        T_e = self.electron_temperature_at_surface(intensity, wavelength)

        # Calculate absorption
        eta_abs = self.absorption.total_absorption(
            intensity, wavelength, scale_length, incident_angle, T_e, polarization)

        # Calculate reflectivity
        eta_refl = self.plasma_mirror.plasma_mirror_reflectivity(
            intensity, wavelength, scale_length, incident_angle)

        # Apply roughness effects
        eta_refl *= self.roughness.roughness_reflection_loss(wavelength, incident_angle)
        eta_abs += self.roughness.enhanced_absorption(intensity, wavelength)

        # Ensure energy conservation
        eta_abs = min(eta_abs, 1 - eta_refl)

        # Surface expansion
        v_exp = self.surface_expansion_velocity(intensity, scale_length)

        # Other parameters
        a0 = self.plasma_mirror.calculate_a0(intensity, wavelength)

        return {
            'scale_length': scale_length,
            'electron_temperature': T_e,
            'absorption_fraction': eta_abs,
            'reflectivity': eta_refl,
            'expansion_velocity': v_exp,
            'a0_parameter': a0,
            'skin_depth': self.skin_depth_at_surface(1e28, wavelength),  # Solid density
            'plasma_frequency': self.plasma_frequency_at_surface(1e28)  # Solid density
        }


def test_plasma_surface_physics():
    """
    Test the plasma surface physics implementation
    """
    print("Testing Enhanced Plasma-Surface Interaction Physics")
    print("=" * 60)

    # Create plasma surface dynamics model for Aluminum
    surface_dynamics = PlasmaDynamicsAtSurface('Al')

    # ELI-like parameters
    intensity = 1e20  # 10^20 W/m^2
    wavelength = 800e-9  # 800 nm
    pulse_duration = 30e-15  # 30 fs
    incident_angle = np.radians(45)  # 45 degrees

    print(f"Testing with {surface_dynamics.target_material} target")
    print(f"Intensity: {intensity:.2e} W/m^2")
    print(f"Wavelength: {wavelength*1e9:.1f} nm")
    print(f"Pulse duration: {pulse_duration*1e15:.1f} fs")
    print(f"Incident angle: {np.degrees(incident_angle):.1f}°")

    # Test full surface interaction
    results = surface_dynamics.full_surface_interaction(
        intensity, wavelength, pulse_duration, incident_angle, 'p')

    print("\nSurface Interaction Results:")
    print(f"  Pre-plasma scale length: {results['scale_length']*1e9:.2f} nm")
    print(f"  Electron temperature: {results['electron_temperature']/e/1e3:.1f} keV")
    print(f"  Absorption fraction: {results['absorption_fraction']:.3f}")
    print(f"  Reflectivity: {results['reflectivity']:.3f}")
    print(f"  Expansion velocity: {results['expansion_velocity']/1e6:.2f} Mm/s")
    print(f"  a0 parameter: {results['a0_parameter']:.2f}")
    print(f"  Skin depth: {results['skin_depth']*1e9:.2f} nm")

    # Test different absorption mechanisms
    absorption = AbsorptionMechanisms()
    scale_length = results['scale_length']
    T_e = results['electron_temperature']

    print("\nIndividual Absorption Mechanisms:")
    eta_brunel = absorption.brunel_heating(intensity, wavelength, scale_length)
    eta_jxb = absorption.jxb_heating(intensity, wavelength, T_e)
    eta_resonance = absorption.resonance_absorption(intensity, wavelength,
                                                    scale_length, incident_angle)
    eta_vacuum = absorption.vacuum_heating(intensity, wavelength)

    print(f"  Brunel heating: {eta_brunel:.3f}")
    print(f"  J×B heating: {eta_jxb:.3f}")
    print(f"  Resonance absorption: {eta_resonance:.3f}")
    print(f"  Vacuum heating: {eta_vacuum:.3f}")

    # Test surface roughness effects
    roughness = SurfaceRoughnessEffects(roughness_rms=5e-9, correlation_length=20e-9)
    loss_factor = roughness.roughness_reflection_loss(wavelength, incident_angle)
    enhanced_abs = roughness.enhanced_absorption(intensity, wavelength)

    print("\nSurface Roughness Effects:")
    print(f"  Roughness-induced reflection loss: {loss_factor:.3f}")
    print(f"  Enhanced absorption: {enhanced_abs:.3f}")

    # Test intensity scaling
    print("\nIntensity Scaling:")
    intensities = np.logspace(18, 22, 5)
    for I in intensities:
        res = surface_dynamics.full_surface_interaction(I, wavelength, pulse_duration, 0, 'p')
        print(f"  I = {I:.1e} W/m^2: R = {res['reflectivity']:.3f}, A = {res['absorption_fraction']:.3f}")

    print("\nPlasma Surface Physics Test Complete!")


if __name__ == "__main__":
    test_plasma_surface_physics()
