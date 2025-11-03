"""
Enhanced Relativistic Physics for Analog Hawking Radiation Analysis

This module implements candidate relativistic corrections for high-intensity
laser-plasma interactions at ELI facility conditions. It addresses the critical
missing physics identified in the scientific review:

1. Relativistic γ-factor corrections for plasma dynamics
2. Relativistic modifications to dispersion relations
3. Relativistic wave propagation effects
4. Relativistic effects on horizon formation and surface gravity

NOTE: The routines below are experimental scaffolding rather than validated
physics. They should be cross-checked against trusted derivations before use
in research results.

Author: Enhanced Physics Implementation
Date: November 2025
References:
- Landau & Lifshitz, Classical Theory of Fields
- Ginzburg, Propagation of Electromagnetic Waves in Plasma
- ELI-NP User Manual for High-Intensity Laser Physics
"""

import warnings
from typing import Dict, Optional, Union

import numpy as np
from scipy.constants import c, e, epsilon_0, hbar, k, m_e, m_p


class RelativisticPlasmaPhysics:
    """
    Comprehensive relativistic plasma physics model for high-intensity laser interactions.
    Implements proper relativistic corrections for ELI facility conditions.
    """

    def __init__(
        self,
        electron_density: float = 1e18,
        laser_wavelength: float = 800e-9,
        laser_intensity: float = 1e19,
        ion_mass: float = m_p,
        gamma_ad: float = 5.0 / 3.0,
        include_quantum_corrections: bool = False,
    ):
        """
        Initialize relativistic plasma physics model with ELI-relevant parameters.

        Args:
            electron_density: Electron density in m^-3
            laser_wavelength: Laser wavelength in meters
            laser_intensity: Laser intensity in W/m^2
            ion_mass: Ion mass in kg
            gamma_ad: Adiabatic index
            include_quantum_corrections: Include quantum electrodynamics corrections
        """
        self.n_e = electron_density
        self.lambda_l = laser_wavelength
        self.I_0 = laser_intensity
        self.m_i = ion_mass
        self.gamma_ad = gamma_ad
        self.include_QED = include_quantum_corrections

        # Basic plasma parameters
        self.omega_l = 2 * np.pi * c / self.lambda_l
        self.omega_pe = np.sqrt(e**2 * self.n_e / (epsilon_0 * m_e))
        self.n_critical = epsilon_0 * m_e * self.omega_l**2 / e**2

        # Relativistic laser parameters
        E_0 = np.sqrt(2 * self.I_0 / (c * epsilon_0))
        self.a0 = e * E_0 / (m_e * self.omega_l * c)
        self.gamma_osc = np.sqrt(1 + self.a0**2 / 2)  # Oscillatory gamma factor

        # Relativistic corrections
        self.mass_ratio = self.m_i / m_e

        # Check regime
        if self.a0 > 1:
            print(f"Relativistic regime: a0 = {self.a0:.2f}")
        if self.I_0 > 1e22:
            print(f"Ultra-relativistic regime: I = {self.I_0:.2e} W/m^2")
            if self.include_QED:
                print("Including QED corrections")

    def relativistic_gamma_factor(
        self,
        momentum: Optional[np.ndarray] = None,
        velocity: Optional[np.ndarray] = None,
        use_oscillatory: bool = False,
    ) -> np.ndarray:
        """
        Calculate relativistic γ-factor with proper handling of different regimes.

        Args:
            momentum: Particle momentum in kg⋅m/s
            velocity: Particle velocity in m/s
            use_oscillatory: Use oscillatory gamma factor for laser field

        Returns:
            Relativistic γ-factor (dimensionless)
        """
        if use_oscillatory:
            return self.gamma_osc

        if momentum is not None:
            # γ = √(1 + (p/mc)²)
            p_mc = momentum / (m_e * c)
            return np.sqrt(1 + p_mc**2)

        elif velocity is not None:
            # γ = 1/√(1 - v²/c²)
            v_over_c = velocity / c
            v_over_c = np.clip(v_over_c, -0.999, 0.999)  # Prevent singularities
            return 1.0 / np.sqrt(1 - v_over_c**2)

        else:
            return 1.0

    def relativistic_plasma_frequency(self, gamma_factor: np.ndarray) -> np.ndarray:
        """
        Calculate relativistically corrected plasma frequency.

        The plasma frequency is reduced by γ^(-1/2) due to relativistic mass increase.

        Args:
            gamma_factor: Relativistic γ-factor

        Returns:
            Relativistic plasma frequency in rad/s
        """
        return self.omega_pe / np.sqrt(gamma_factor)

    def relativistic_dielectric_function(
        self, omega: np.ndarray, k_vector: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistic dielectric function for electromagnetic waves.

        Args:
            omega: Angular frequency in rad/s
            k_vector: Wave vector in m^-1
            gamma_factor: Relativistic γ-factor

        Returns:
            Dielectric function ε(ω,k,γ)
        """
        omega_pe_rel = self.relativistic_plasma_frequency(gamma_factor)
        k_mag = np.linalg.norm(k_vector, axis=-1) if k_vector.ndim > 1 else np.abs(k_vector)

        # Relativistic correction to dispersion relation
        # ε = 1 - ω_pe²/[γ(ω² - k²c²)]
        denom = omega**2 - k_mag**2 * c**2
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
        epsilon = 1 - (self.omega_pe**2 / gamma_factor) / denom

        return epsilon

    def relativistic_dispersion_relation(
        self,
        omega: np.ndarray,
        gamma_factor: np.ndarray,
        wave_mode: str = "electromagnetic",
        temperature: Optional[np.ndarray] = None,
        magnetic_field: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Calculate relativistic dispersion relation for different wave modes.

        Args:
            omega: Angular frequency in rad/s
            gamma_factor: Relativistic γ-factor
            wave_mode: Type of wave ('electromagnetic', 'electrostatic', 'whistler')
            temperature: Temperature in Kelvin (required for electrostatic mode)
            magnetic_field: Magnetic field in Tesla (required for whistler mode)

        Returns:
            Wave vector magnitude in m^-1
        """
        omega = np.asarray(omega, dtype=float)
        gamma_factor = np.asarray(gamma_factor, dtype=float)
        omega_pe_sq_over_gamma = self.omega_pe**2 / gamma_factor

        if wave_mode == "electromagnetic":
            # EM wave: ω² = ω_pe²/γ + k²c²
            k_squared = np.maximum(omega**2 - omega_pe_sq_over_gamma, 0.0) / c**2

        elif wave_mode == "electrostatic":
            # Electrostatic (Langmuir) wave: ω² = ω_pe²/γ + 3k²v_th²
            if temperature is None:
                raise ValueError("temperature must be provided for electrostatic dispersion.")
            temperature = np.asarray(temperature, dtype=float)
            v_th = np.sqrt(self.gamma_ad * k * temperature / (self.m_i * gamma_factor))
            thermal_term = 3.0 * np.maximum(v_th**2, 1e-30)
            k_squared = np.maximum(omega**2 - omega_pe_sq_over_gamma, 0.0) / thermal_term

        elif wave_mode == "whistler":
            if magnetic_field is None:
                raise ValueError("magnetic_field must be provided for whistler dispersion.")
            magnetic_field = np.asarray(magnetic_field, dtype=float)
            omega_ce = e * magnetic_field / (m_e * gamma_factor)
            denom = np.maximum(omega_ce - omega, 1e-30)
            k_squared = np.maximum(omega * omega_pe_sq_over_gamma, 0.0) / (c**2 * denom)

        else:
            raise ValueError(f"Unknown wave mode: {wave_mode}")

        k_mag = np.sqrt(np.maximum(k_squared, 0.0))

        return k_mag

    def relativistic_sound_speed(
        self, temperature: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistically corrected sound speed.

        Args:
            temperature: Temperature in Kelvin
            gamma_factor: Relativistic γ-factor

        Returns:
            Relativistic sound speed in m/s
        """
        # Relativistic correction to sound speed
        # c_s,rel = c_s,classical / √γ
        c_s_classical = np.sqrt(self.gamma_ad * k * temperature / self.m_i)
        c_s_rel = c_s_classical / np.sqrt(gamma_factor)

        # Ensure sound speed doesn't exceed c/√3 (relativistic limit)
        c_s_max = c / np.sqrt(3)
        c_s_rel = np.minimum(c_s_rel, c_s_max)

        return c_s_rel

    def relativistic_critical_density(self, gamma_factor: np.ndarray) -> np.ndarray:
        """
        Calculate relativistically corrected critical density.

        The critical density increases by γ due to relativistic effects.

        Args:
            gamma_factor: Relativistic γ-factor

        Returns:
            Relativistic critical density in m^-3
        """
        return self.n_critical * gamma_factor

    def relativistic_ponderomotive_potential(
        self, E_field: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistically corrected ponderomotive potential.

        Args:
            E_field: Electric field amplitude in V/m
            gamma_factor: Relativistic γ-factor

        Returns:
            Ponderomotive potential in Joules
        """
        # U_p = (γ - 1)m_e*c² - classical limit
        # For laser field: U_p ≈ m_e*c²(√(1 + a²/2) - 1)
        a_local = e * E_field / (m_e * self.omega_l * c)
        gamma_local = np.sqrt(1 + a_local**2 / 2)
        U_p = m_e * c**2 * (gamma_local - 1)

        return U_p

    def relativistic_radiation_pressure(
        self, intensity: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistic radiation pressure.

        Args:
            intensity: Laser intensity in W/m²
            gamma_factor: Relativistic γ-factor

        Returns:
            Radiation pressure in Pa
        """
        # P_rad = 2I/c for perfect reflector (classical)
        # Relativistic correction includes Doppler shift and momentum change
        doppler_factor = gamma_factor * (1 + 0)  # Simplified for normal incidence
        P_rad = 2 * intensity / (c * doppler_factor)

        return P_rad

    def relativistic_hole_boring_velocity(
        self, intensity: np.ndarray, density: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate hole-boring velocity with relativistic corrections.

        Args:
            intensity: Laser intensity in W/m²
            density: Plasma density in m^-3
            gamma_factor: Relativistic γ-factor

        Returns:
            Hole-boring velocity in m/s
        """
        # v_hb = √(2I/ρc) for classical case
        # Relativistic corrections account for momentum change
        mass_density = density * self.m_i
        v_hb_squared = 2 * intensity / (mass_density * c * gamma_factor)
        v_hb = np.sqrt(np.maximum(v_hb_squared, 0))

        # Ensure v_hb < c
        v_hb = np.minimum(v_hb, 0.9 * c)

        return v_hb

    def relativistic_surface_gravity(
        self,
        velocity_gradient: np.ndarray,
        sound_speed_gradient: np.ndarray,
        gamma_factor: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate relativistic surface gravity for analog horizon.

        Args:
            velocity_gradient: Gradient of flow velocity ∂v/∂x in s^-1
            sound_speed_gradient: Gradient of sound speed ∂c_s/∂x in s^-1
            gamma_factor: Relativistic γ-factor

        Returns:
            Relativistic surface gravity in s^-1
        """
        # Classical: κ = |∂(c_s - |v|)/∂x|
        # Relativistic correction includes time dilation
        kappa_classical = np.abs(sound_speed_gradient - velocity_gradient)
        gamma_factor = np.asarray(gamma_factor, dtype=float)
        if np.any(np.abs(gamma_factor - 1.0) > 1e-12):
            warnings.warn(
                "Relativistic surface gravity currently returns the classical value. "
                "Provide a frame-consistent derivation before applying gamma-scaling.",
                RuntimeWarning,
                stacklevel=2,
            )

        return kappa_classical

    def relativistic_hawking_temperature(
        self, kappa: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistically corrected Hawking temperature.

        Args:
            kappa: Surface gravity in s^-1
            gamma_factor: Relativistic γ-factor

        Returns:
            Hawking temperature in Kelvin
        """
        # T_H = ħκ/(2πk_B) with relativistic corrections
        # Time dilation reduces observed temperature
        T_classical = hbar * kappa / (2 * np.pi * k)
        gamma_factor = np.asarray(gamma_factor, dtype=float)
        if np.any(np.abs(gamma_factor - 1.0) > 1e-12):
            warnings.warn(
                "Relativistic Hawking temperature returns the classical value; "
                "time-dilation corrections require a dedicated derivation.",
                RuntimeWarning,
                stacklevel=2,
            )

        return T_classical

    def check_relativistic_regime(self) -> Dict[str, Union[bool, float, str]]:
        """
        Check which relativistic effects are important for current parameters.

        Returns:
            Dictionary with regime information
        """
        regimes = {
            "relativistic_laser": self.a0 > 1,
            "relativistic_electrons": self.gamma_osc > 1.1,
            "radiation_pressure_dominant": self.I_0 > 1e18,
            "pair_production_possible": self.I_0 > 1e24,
            "quantum_corrections_important": self.a0 > 100,
        }

        # Add numerical values
        regimes.update(
            {
                "a0_parameter": self.a0,
                "gamma_oscillatory": self.gamma_osc,
                "intensity_W_m2": self.I_0,
                "normalized_intensity": self.I_0 / 1e18,
            }
        )

        # Add regime classification
        if self.a0 < 0.1:
            regimes["regime"] = "Non-relativistic"
        elif self.a0 < 1:
            regimes["regime"] = "Weakly relativistic"
        elif self.a0 < 10:
            regimes["regime"] = "Relativistic"
        elif self.a0 < 100:
            regimes["regime"] = "Highly relativistic"
        else:
            regimes["regime"] = "Ultra-relativistic"

        return regimes

    def relativistic_wave_breaking_field(
        self, density: np.ndarray, gamma_factor: np.ndarray
    ) -> np.ndarray:
        """
        Calculate relativistic wave breaking field.

        Args:
            density: Plasma density in m^-3
            gamma_factor: Relativistic γ-factor

        Returns:
            Wave breaking field in V/m
        """
        # E_WB = m_e*c*ω_pe/e * √γ
        omega_pe_local = np.sqrt(e**2 * density / (epsilon_0 * m_e))
        E_WB = m_e * c * omega_pe_local / e * np.sqrt(gamma_factor)

        return E_WB


def test_relativistic_physics():
    """
    Test the relativistic physics implementation with ELI-relevant parameters.
    """
    print("Testing Relativistic Physics Implementation")
    print("=" * 50)

    # ELI-like parameters
    params = {
        "electron_density": 1e21,  # Solid density
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e20,  # 10^20 W/m^2 (relativistic)
        "include_quantum_corrections": False,
    }

    # Create relativistic plasma model
    plasma = RelativisticPlasmaPhysics(**params)

    # Check regime
    regime = plasma.check_relativistic_regime()
    print(f"Regime: {regime['regime']}")
    print(f"a0 parameter: {regime['a0_parameter']:.2f}")
    print(f"γ_osc: {regime['gamma_oscillatory']:.2f}")

    # Test relativistic corrections
    gamma_test = np.array([1.0, 1.5, 2.0, 5.0, 10.0])

    print("\nRelativistic Plasma Frequency:")
    omega_pe_rel = plasma.relativistic_plasma_frequency(gamma_test)
    for i, gamma in enumerate(gamma_test):
        print(f"  γ = {gamma:.1f}: ω_pe,rel = {omega_pe_rel[i]:.2e} rad/s")

    print("\nRelativistic Sound Speed (T = 1e6 K):")
    T = 1e6  # 1 MK
    c_s_rel = plasma.relativistic_sound_speed(T, gamma_test)
    for i, gamma in enumerate(gamma_test):
        print(f"  γ = {gamma:.1f}: c_s,rel = {c_s_rel[i]:.2e} m/s ({c_s_rel[i]/c:.4f}c)")

    print("\nRelativistic Hawking Temperature:")
    kappa = 1e13  # Surface gravity in s^-1
    T_hawking = plasma.relativistic_hawking_temperature(kappa, gamma_test)
    for i, gamma in enumerate(gamma_test):
        print(f"  γ = {gamma:.1f}: T_H = {T_hawking[i]:.2e} K")

    print("\nWave Breaking Field:")
    n_test = np.array([1e18, 1e19, 1e20, 1e21])  # Different densities
    E_WB = plasma.relativistic_wave_breaking_field(n_test, gamma_test[0])
    for i, n in enumerate(n_test):
        print(f"  n = {n:.1e} m^-3: E_WB = {E_WB[i]:.2e} V/m")

    print("\nRelativistic Physics Test Complete!")


if __name__ == "__main__":
    test_relativistic_physics()
