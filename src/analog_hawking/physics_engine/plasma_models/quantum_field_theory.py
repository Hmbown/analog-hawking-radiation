"""
Quantum Field Theory in Curved Spacetime for Analog Hawking Radiation

This module implements the fundamental quantum field theory calculations
needed to properly model Hawking radiation in analog systems.
"""

import numpy as np
from scipy.constants import c, h, hbar, k, G
from scipy.special import gamma as gamma_func
from scipy.integrate import quad
import warnings

class QuantumFieldTheory:
    """
    Class to model quantum field theory in curved spacetime for analog Hawking radiation
    """
    
    def __init__(self, surface_gravity=1e12, temperature=None, emitting_area_m2=None, solid_angle_sr=None, coupling_efficiency=1.0):
        """
        Initialize QFT model with surface gravity or temperature
        
        Args:
            surface_gravity (float): Surface gravity in s^-1
            temperature (float): Hawking temperature in K (optional, calculated from surface_gravity if not provided)
        """
        self.kappa = surface_gravity  # Surface gravity in s^-1
        if temperature is None:
            self.T_H = hbar * surface_gravity / (2 * np.pi * k)  # Hawking temperature in K
        else:
            self.T_H = temperature  # Hawking temperature in K
        self.emitting_area_m2 = emitting_area_m2
        self.solid_angle_sr = solid_angle_sr
        self.coupling_efficiency = coupling_efficiency
    
    def hawking_spectrum(self, omega, transmission=None):
        """
        Calculate the Hawking radiation power spectrum
        
        Args:
            omega (array or float): Angular frequency in rad/s
            
        Returns:
            Power spectrum in W·sr⁻¹·m⁻²·Hz⁻¹
        """
        # The correct Hawking spectrum follows the Planck distribution with proper units
        # P(ω) = (ħω³) / (2π²c²) * 1 / (exp(ħω/kT) - 1)  (power per unit frequency)
        # We include the graybody factor which accounts for transmission through potential barrier
        
        frequencies = omega / (2 * np.pi)
        B_nu = self.thermal_spectral_density(frequencies)
        if (self.emitting_area_m2 is not None) and (self.solid_angle_sr is not None):
            base_psd = B_nu * float(self.emitting_area_m2) * float(self.solid_angle_sr) * float(self.coupling_efficiency)
        else:
            base_psd = B_nu

        if transmission is not None:
            gray = transmission
        else:
            gray = self.graybody_factor(omega)

        return base_psd * gray
    
    def graybody_factor(self, omega):
        """
        Calculate graybody factor for transmission through potential barrier
        
        Args:
            omega (array or float): Angular frequency in rad/s
            
        Returns:
            Transmission coefficient
        """
        # Simplified graybody factor for analog systems
        # In real black holes, this comes from solving wave equation in curved spacetime
        # For analog systems, this depends on the specific velocity profile
        
        omega0 = float(self.kappa) if self.kappa != 0 else 1.0
        return (omega**2) / (omega**2 + omega0**2)
    
    def thermal_spectral_density(self, frequency):
        """
        Calculate thermal spectral density following Planck's law
        
        Args:
            frequency (array or float): Frequency in Hz
            
        Returns:
            Spectral density in W·sr⁻¹·m⁻²·Hz⁻¹
        """
        # Planck's law: B_ν(T) = (2hν³/c²) / (exp(hν/kT) - 1)
        nu = np.asarray(frequency, dtype=float)
        x = (h * nu) / (k * self.T_H)
        pre = (2.0 * h * nu**3) / (c**2)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            # Use expm1 for small x; for large x use e^{-x} asymptotic
            spectral_density = np.where(
                x > 50.0,
                pre * np.exp(-x),
                pre / np.expm1(x)
            )
        return spectral_density
    
    def hawking_temperature_from_kappa(self, surface_gravity):
        """
        Calculate Hawking temperature from surface gravity
        
        Args:
            surface_gravity (float): Surface gravity in s^-1
            
        Returns:
            Hawking temperature in K
        """
        return hbar * surface_gravity / (2 * np.pi * k)
    
    def kappa_from_hawking_temperature(self, temperature):
        """
        Calculate surface gravity from Hawking temperature
        
        Args:
            temperature (float): Hawking temperature in K
            
        Returns:
            Surface gravity in s^-1
        """
        return 2 * np.pi * k * temperature / hbar
    
    def energy_distribution(self, omega):
        """
        Calculate energy distribution per logarithmic frequency interval
        
        Args:
            omega (array or float): Angular frequency in rad/s
            
        Returns:
            Energy distribution
        """
        # dE/dln(ω) = ω * dN/dω where dN/dω is the occupation number
        x = (hbar * omega) / (k * self.T_H)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            occupation = np.where(
                x > 50.0,
                np.exp(-x),
                1.0 / np.expm1(x)
            )
        
        # Energy per mode is ħ·ω, so total energy distribution:
        return (hbar * omega) * occupation * self.density_of_states(omega)
    
    def density_of_states(self, omega):
        """
        Calculate density of states for massless scalar field
        
        Args:
            omega (array or float): Angular frequency in rad/s
            
        Returns:
            Density of states
        """
        # For 3D massless scalar field: g(ω) = ω² / (π² * c³)
        return omega**2 / (np.pi**2 * c**3)
    
    def occupation_number(self, omega):
        """
        Calculate occupation number of thermal radiation
        
        Args:
            omega (array or float): Angular frequency in rad/s
            
        Returns:
            Occupation number (average number of photons per mode)
        """
        x = (hbar * omega) / (k * self.T_H)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            return np.where(
                x > 50.0,
                np.exp(-x),
                1.0 / np.expm1(x)
            )

class BogoliubovTransformations:
    """
    Class to handle Bogoliubov transformations for particle creation in curved spacetime
    """
    
    def __init__(self, initial_modes, final_modes, alpha_beta_coeffs=None):
        """
        Initialize Bogoliubov transformation
        
        Args:
            initial_modes: Basis modes for initial vacuum
            final_modes: Basis modes for final vacuum
            alpha_beta_coeffs: Precomputed alpha and beta Bogoliubov coefficients
        """
        self.initial_modes = initial_modes
        self.final_modes = final_modes
        self.alpha_beta_coeffs = alpha_beta_coeffs
        self.dimension = len(initial_modes) if isinstance(initial_modes, (list, np.ndarray)) else 1
    
    def compute_bogoliubov_coefficients(self, potential_barrier_func, energy_range):
        """
        Compute Bogoliubov coefficients for a given potential barrier
        
        Args:
            potential_barrier_func: Function describing the effective potential
            energy_range: Energy range to compute coefficients for
            
        Returns:
            Dictionary with alpha and beta coefficients
        """
        # This is a simplified calculation - in reality, this requires solving the
        # wave equation in the curved spacetime background
        # For an analog system, this corresponds to solving the wave equation
        # in the fluid flow with the effective metric
        
        if self.alpha_beta_coeffs is not None:
            return self.alpha_beta_coeffs
        
        # For a simple potential barrier, the Bogoliubov coefficients are related to
        # transmission and reflection coefficients
        # |beta|^2 / |alpha|^2 = exp(-4 * integral of Im(sqrt(V-E)) dx)
        
        energies = np.linspace(energy_range[0], energy_range[1], 100)
        alpha_coeffs = np.ones_like(energies, dtype=complex)
        beta_coeffs = np.zeros_like(energies, dtype=complex)
        
        # Approximate calculation for Hawking-like radiation
        # The imaginary part comes from the horizon crossing
        for i, E in enumerate(energies):
            # For Hawking radiation, |β/α|² ≈ exp(-2πE/kT)
            ratio_squared = np.exp(-2 * np.pi * E / (k * 1.2e9))  # Using example T_H
            alpha_sq = 1 / (1 - ratio_squared)  # Normalization |α|² - |β|² = 1
            beta_sq = ratio_squared / (1 - ratio_squared)
            
            alpha_coeffs[i] = np.sqrt(alpha_sq)
            beta_coeffs[i] = np.sqrt(beta_sq)
        
        return {
            'alpha': alpha_coeffs,
            'beta': beta_coeffs,
            'occupation': np.abs(beta_coeffs)**2  # Average number of created particles
        }
    
    def particle_creation_rate(self, omega, surface_gravity):
        """
        Calculate particle creation rate per mode
        
        Args:
            omega (array or float): Angular frequency in rad/s
            surface_gravity (float): Surface gravity in s^-1
            
        Returns:
            Particle creation rate
        """
        # The rate of particle creation is given by |β_ω|²
        # For Hawking radiation: |β_ω|² = 1 / (exp(ħω/kT) - 1)
        
        h_omega = hbar * omega
        # Using T = ħκ/(2πk), exponent argument x = ħω/(kT) = 2π ω / κ
        x = (2.0 * np.pi * omega) / float(surface_gravity)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            return np.where(
                x > 50.0,
                np.exp(-x),
                1.0 / np.expm1(x)
            )
    
    def total_particle_flux(self, omega_range, surface_gravity):
        """
        Calculate total particle flux across all frequencies
        
        Args:
            omega_range: Tuple of (min_omega, max_omega) in rad/s
            surface_gravity: Surface gravity in s^-1
            
        Returns:
            Total particle flux per unit area
        """
        def integrand(omega):
            return self.particle_creation_rate(omega, surface_gravity) * omega**2 / (np.pi**2 * c**3)
        
        result, _ = quad(integrand, omega_range[0], omega_range[1])
        return result

class HawkingRadiationModel:
    """
    Complete model for Hawking radiation with proper QFT calculations
    """
    
    def __init__(self, surface_gravity=1e12):
        """
        Initialize the complete Hawking radiation model
        
        Args:
            surface_gravity (float): Surface gravity in s^-1
        """
        self.qft = QuantumFieldTheory(surface_gravity)
        self.bogoliubov = BogoliubovTransformations([], [])
        self.surface_gravity = surface_gravity
        self.temperature = self.qft.hawking_temperature_from_kappa(surface_gravity)
    
    def calculate_spectrum(self, frequency_range, num_points=1000):
        """
        Calculate the complete Hawking radiation spectrum over a frequency range
        
        Args:
            frequency_range: Tuple of (min_freq, max_freq) in Hz
            num_points: Number of frequency points to calculate
            
        Returns:
            Tuple of (frequencies, spectrum)
        """
        frequencies = np.logspace(np.log10(frequency_range[0]), np.log10(frequency_range[1]), num_points)
        omega = 2 * np.pi * frequencies
        
        # Calculate the thermal spectrum with graybody corrections
        spectrum = self.qft.hawking_spectrum(omega)
        
        # Also calculate energy flux
        energy_spectrum = hbar * omega * spectrum
        
        return {
            'frequencies': frequencies,
            'angular_frequencies': omega,
            'power_spectrum': spectrum,
            'energy_spectrum': energy_spectrum,
            'temperature': self.temperature,
            'surface_gravity': self.surface_gravity
        }
    
    def detectable_signal(self, detector_bandwidth, efficiency=0.5):
        """
        Calculate the detectable signal in a realistic detector
        
        Args:
            detector_bandwidth: Detector bandwidth in Hz
            efficiency: Detector efficiency (0-1)
            
        Returns:
            Detectable signal characteristics
        """
        # Calculate signal in detector band
        freq_min = 1e15  # Start from optical range
        freq_max = 1e18  # Up to soft X-rays
        freq_range = (freq_min, freq_max)
        
        # Calculate total power in detector band
        spectrum_data = self.calculate_spectrum(freq_range)
        
        # Integrate power over detector band with efficiency
        total_power = np.trapz(spectrum_data['power_spectrum'] * efficiency, 
                                   x=spectrum_data['frequencies'])
        
        # Calculate signal-to-noise ratio (simplified)
        # For now, we assume background is thermal noise + detector noise
        noise_power = 2 * k * 300 * detector_bandwidth  # Thermal noise at room temp
        snr = total_power / noise_power if noise_power > 0 else np.inf
        
        return {
            'total_power': total_power,
            'noise_power': noise_power,
            'snr': snr,
            'optimal_frequency': spectrum_data['frequencies'][np.argmax(spectrum_data['power_spectrum'])],
            'peak_power': np.max(spectrum_data['power_spectrum'])
        }
    
    def compare_with_classical_limits(self):
        """
        Compare quantum results with classical limits
        
        Returns:
            Dictionary with comparison results
        """
        # In the low frequency limit, quantum results approach classical
        # Calculate classical Rayleigh-Jeans limit for comparison
        omega_test = 2 * np.pi * 1e12  # 1 THz
        
        quantum_result = self.qft.occupation_number(omega_test)
        classical_result = k * self.temperature / (hbar * omega_test)  # Rayleigh-Jeans limit
        
        return {
            'quantum_occupation': quantum_result,
            'classical_occupation': classical_result,
            'ratio': quantum_result / classical_result,
            'temperature': self.temperature
        }

def simulate_quantum_field_in_analog_spacetime(surface_gravity, flow_profile):
    """
    Simulate quantum field behavior in an analog spacetime created by fluid flow
    
    Args:
        surface_gravity (float): Surface gravity in s^-1
        flow_profile (function): Function describing fluid velocity profile v(x)
        
    Returns:
        Dictionary with simulation results
    """
    # Initialize the model
    hawking_model = HawkingRadiationModel(surface_gravity)
    
    # Calculate the full spectrum
    spectrum = hawking_model.calculate_spectrum((1e12, 1e18))  # 1 THz to 1 PHz
    
    # Calculate detectable signal
    detectable = hawking_model.detectable_signal(detector_bandwidth=1e12, efficiency=0.3)
    
    # Compare with classical limits
    classical_comparison = hawking_model.compare_with_classical_limits()
    
    # Calculate total emission rate
    total_rate = hawking_model.bogoliubov.total_particle_flux(
        (2 * np.pi * 1e12, 2 * np.pi * 1e18), surface_gravity
    )
    
    return {
        'spectrum': spectrum,
        'detectable_signal': detectable,
        'classical_comparison': classical_comparison,
        'total_emission_rate': total_rate,
        'temperature': hawking_model.temperature,
        'surface_gravity': surface_gravity
    }

if __name__ == "__main__":
    # Example usage
    print("Quantum Field Theory in Curved Spacetime - Hawking Radiation Simulation")
    print("=" * 70)
    
    # Typical surface gravity for analog systems
    surface_gravity = 2e12  # s^-1 (typical for intense laser-plasma systems)
    
    # Simulate quantum field
    result = simulate_quantum_field_in_analog_spacetime(surface_gravity, lambda x: 0)
    
    print(f"Hawking Temperature: {result['temperature']:.2e} K")
    print(f"Surface Gravity: {result['surface_gravity']:.2e} s^-1")
    print(f"Total Emission Rate: {result['total_emission_rate']:.2e} m^-2 s^-1")
    print(f"Detectable Power: {result['detectable_signal']['total_power']:.2e} W")
    print(f"SNR: {result['detectable_signal']['snr']:.2e}")
    print(f"Optimal Detection Frequency: {result['detectable_signal']['optimal_frequency']:.2e} Hz")
