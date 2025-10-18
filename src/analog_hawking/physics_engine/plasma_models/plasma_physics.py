"""
Proper plasma physics model for analog Hawking radiation simulations

This module implements the fundamental physics of laser-plasma interactions
relevant to creating analog black holes and detecting Hawking radiation.
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, hbar, k, m_p, mu_0
from scipy.special import gamma
import warnings

class PlasmaPhysicsModel:
    """
    Class to model realistic plasma physics for analog Hawking radiation experiments
    """
    
    def __init__(self, plasma_density=1e18, laser_wavelength=800e-9, laser_intensity=1e17, ion_mass=m_p, gamma_e=5.0/3.0):
        """
        Initialize plasma physics model with realistic parameters
        self.m_i = ion_mass  # ion mass (kg), default proton
        self.gamma_e = gamma_e  # adiabatic index
        
        Args:
            plasma_density (float): Electron density in m^-3
            laser_wavelength (float): Laser wavelength in meters
            laser_intensity (float): Laser intensity in W/m^2
        """
        self.n_e = plasma_density  # electron density (m^-3)
        self.lambda_l = laser_wavelength  # laser wavelength (m)
        self.I_0 = laser_intensity  # laser intensity (W/m^2)
        self.m_i = ion_mass  # ion mass (kg)
        self.gamma_e = gamma_e  # adiabatic index
        
        # Derived parameters
        self.omega_pe = np.sqrt(e**2 * self.n_e / (epsilon_0 * m_e))  # Plasma frequency
        self.omega_l = 2 * np.pi * c / self.lambda_l  # Laser frequency
        # Correct a0 calculation: a0 = e * E0 / (m_e * omega_l * c)
        E0 = np.sqrt(2 * self.I_0 / (c * epsilon_0))
        self.a0 = e * E0 / (m_e * self.omega_l * c)  # Correct normalized vector potential
        
        # Critical density for given laser wavelength
        self.n_critical = epsilon_0 * m_e * self.omega_l**2 / e**2
        
        # Check if we're in relativistic regime
        if self.a0 > 1:
            warnings.warn(f"Relativistic parameter a0 = {self.a0:.2f} > 1, relativistic effects important")
    
    def cold_plasma_dielectric(self, omega, k_parallel):
        """
        Calculate dielectric function for cold plasma
        
        Args:
            omega: Angular frequency
            k_parallel: Wavevector parallel to magnetic field
            
        Returns:
            Dielectric function value
        """
        return 1 - (self.omega_pe**2) / (omega**2 - k_parallel**2 * c**2)
    
    def plasma_frequency(self, density=None):
        """
        Calculate plasma frequency for given density
        
        Args:
            density: Electron density in m^-3 (if None, uses self.n_e)
            
        Returns:
            Plasma frequency in rad/s
        """
        if density is None:
            density = self.n_e
        return np.sqrt(e**2 * density / (epsilon_0 * m_e))
    
    def relativistic_plasma_frequency(self, gamma_factor):
        """
        Calculate relativistic plasma frequency
        
        Args:
            gamma_factor: Relativistic gamma factor
            
        Returns:
            Relativistic plasma frequency in rad/s
        """
        return self.omega_pe / np.sqrt(gamma_factor)
    
    def wakefield_phase_velocity(self):
        """
        Calculate phase velocity of plasma wakefields
        
        Returns:
            Phase velocity in m/s
        """
        return c * np.sqrt(1 - (self.omega_pe / self.omega_l)**2)
    
    def debye_length(self, T_e=10000):
        """
        Calculate Debye length for given electron temperature
        
        Args:
            T_e: Electron temperature in Kelvin
            
        Returns:
            Debye length in meters
        """
        return np.sqrt(epsilon_0 * k * T_e / (e**2 * self.n_e))

    def mass_density(self, density: np.ndarray | float | None = None) -> np.ndarray:
        rho = self.n_e if density is None else density
        rho = np.asarray(rho, dtype=float)
        return rho * self.m_i

    def sound_speed(self, T_e):
        """
        Adiabatic sound speed c_s = sqrt(gamma k T_e / m_i)
        Args:
            T_e: electron temperature (K), scalar or array
        Returns: c_s array-like
        """
        T_e = np.asarray(T_e)
        return np.sqrt(np.maximum(self.gamma_e * k * T_e / self.m_i, 0.0))

    def fast_magnetosonic_speed(self, T_e, density=None, B=None):
        c_s = self.sound_speed(T_e)
        if density is None:
            density = self.n_e
        rho = self.mass_density(density)
        if B is None:
            return c_s
        B = np.asarray(B)
        v_A = np.where(rho > 0, B / np.sqrt(mu_0 * rho), 0.0)
        return np.sqrt(c_s**2 + v_A**2)
    
    def skin_depth(self):
        """
        Calculate electromagnetic skin depth in plasma
        
        Returns:
            Skin depth in meters
        """
        return c / self.omega_pe
    
    def relativistic_gamma(self):
        """
        Calculate relativistic gamma factor for laser-plasma interaction
        
        Returns:
            Relativistic gamma
        """
        return np.sqrt(1 + self.a0**2)
    
    def ponderomotive_force(self, E_laser):
        """
        Calculate ponderomotive force in laser field
        
        Args:
            E_laser: Laser electric field in V/m
            
        Returns:
            Ponderomotive force in N
        """
        # Ponderomotive potential: U_p = e^2*E^2/(4*m_e*omega_l^2*gamma)
        gamma_factor = self.relativistic_gamma()
        U_p = e**2 * E_laser**2 / (4 * m_e * self.omega_l**2 * gamma_factor)
        return -e * np.gradient(U_p)  # Force = -dU_p/dx
    
    def simulate_plasma_response(self, t, x, E_laser_func):
        """
        Simulate plasma response to laser field
        
        Args:
            t: Time array
            x: Position array
            E_laser_func: Function that returns laser electric field E(x,t)
            
        Returns:
            Dictionary containing plasma response parameters
        """
        # This is a simplified simulation - in reality, you'd need PIC codes
        n_e = np.full_like(x, self.n_e)  # Initialize density
        v_e = np.zeros_like(x)  # Initialize velocity
        
        # Calculate laser field at each point
        E_laser = np.array([E_laser_func(x_i, t[0]) for x_i in x])  # Simplified for first time step
        
        # Calculate plasma response
        # Drift velocity in laser field
        v_drift = e * E_laser / (m_e * self.omega_l)  # Simple harmonic oscillator approximation
        
        # Update density with continuity equation (simplified)
        # In reality, this requires solving full fluid equations
        density_perturbation = np.gradient(v_drift)
        
        return {
            'density': n_e + density_perturbation,
            'velocity': v_drift,
            'electric_field': E_laser,
            'current_density': -e * n_e * v_drift
        }

class AnalogHorizonPhysics:
    """
    Class to model the physics of analog event horizons in plasma
    """
    
    def __init__(self, plasma_model):
        """
        Initialize analog horizon physics with plasma model
        
        Args:
            plasma_model: Instance of PlasmaPhysicsModel
        """
        self.plasma = plasma_model
    
    def effective_metric(self, flow_velocity, sound_speed):
        """
        Calculate effective spacetime metric for analog gravity
        
        Args:
            flow_velocity: Fluid flow velocity
            sound_speed: Speed of sound in medium
            
        Returns:
            Dictionary containing metric components
        """
        # For acoustic metric: ds^2 = g_μν dx^μ dx^ν
        # In 1+1D with (-,+) signature: ds^2 = -(c_s^2 - v^2) dt^2 + 2 v dt dx - dx^2
        
        c_s = sound_speed
        v = flow_velocity
        
        # Metric components
        g_tt = -(c_s**2 - v**2)
        g_tx = g_xt = v
        g_xx = -1.0
        
        return {
            'g_tt': g_tt,
            'g_tx': g_tx,
            'g_xt': g_xt,
            'g_xx': g_xx,
            'det_g': -(c_s**2 - v**2)  # Determinant (up to overall conformal factor)
        }
    
    def horizon_condition(self, flow_velocity, sound_speed):
        """
        Determine where analog horizon forms
        
        Args:
            flow_velocity: Fluid flow velocity
            sound_speed: Speed of sound in medium
            
        Returns:
            Boolean array indicating horizon locations
        """
        return np.abs(flow_velocity) == sound_speed
    
    def surface_gravity(self, gradient_flow_velocity):
        """
        Legacy surface gravity surrogate used in earlier versions.

        Note: The main pipeline now uses an acoustic approximation κ ≈ |∂x(c_s − |v|)|
        at the horizon. This method retains the historical 0.5·|∇v| form for
        backward compatibility in auxiliary scripts.

        Args:
            gradient_flow_velocity: Gradient of flow velocity at horizon

        Returns:
            Surface gravity surrogate (s^-1)
        """
        return np.abs(gradient_flow_velocity) / 2.0
    
    def hawking_temperature(self, surface_gravity):
        """
        Calculate theoretical Hawking temperature
        
        Args:
            surface_gravity: Surface gravity in s^-1
            
        Returns:
            Hawking temperature in Kelvin
        """
        return hbar * surface_gravity / (2 * np.pi * k)

class QEDPhysics:
    """
    Quantum electrodynamics effects relevant to high-intensity laser-plasma interactions
    """
    
    def __init__(self):
        # Fundamental constants for QED calculations
        self.alpha = 7.2973525693e-3  # Fine structure constant
        self.m_e_c2 = m_e * c**2  # Electron rest energy
        self.E_S = m_e_c2**2 / (e * hbar)  # Schwinger field
        self.lambda_C = hbar / (m_e * c)  # Compton wavelength
    
    def pair_production_rate(self, E_field):
        """
        Calculate electron-positron pair production rate (Schwinger mechanism)
        
        Args:
            E_field: Electric field strength in V/m
            
        Returns:
            Pair production rate per unit volume per unit time
        """
        if E_field < 0.1 * self.E_S:
            # Below Schwinger limit, exponential suppression
            rate = (E_field**2 / (8 * np.pi**2 * self.lambda_C**3)) * np.exp(-np.pi * self.E_S / E_field)
        else:
            # Above Schwinger limit, full rate
            rate = (E_field**2 / (8 * np.pi**2 * self.lambda_C**3))
        
        return rate
    
    def critical_field(self):
        """
        Return the critical Schwinger field strength
        
        Returns:
            Critical field in V/m
        """
        return self.E_S
    
    def radiation_reaction_force(self, E_field, B_field, velocity):
        """
        Calculate radiation reaction force on electron
        
        Args:
            E_field: Electric field vector
            B_field: Magnetic field vector
            velocity: Electron velocity vector
            
        Returns:
            Radiation reaction force vector
        """
        # Classical radiation reaction (Lorentz-Abraham-Dirac) force
        gamma = 1 / np.sqrt(1 - np.sum(velocity**2) / c**2)
        E_cross_B = np.cross(E_field, B_field)
        force = (2 * self.alpha * self.m_e_c2) / (3 * self.lambda_C * c) * gamma**3 * E_cross_B / c
        
        return force

def simulate_analog_black_hole(laser_params, plasma_params):
    """
    Main function to simulate analog black hole formation in plasma
    
    Args:
        laser_params: Dictionary with laser parameters (intensity, wavelength, pulse duration)
        plasma_params: Dictionary with plasma parameters (density, temperature, dimensions)
        
    Returns:
        Dictionary with simulation results
    """
    # Create plasma model
    plasma = PlasmaPhysicsModel(
        plasma_density=plasma_params['density'],
        laser_wavelength=laser_params['wavelength'],
        laser_intensity=laser_params['intensity']
    )
    
    # Create analog horizon physics model
    horizon = AnalogHorizonPhysics(plasma)
    
    # Create QED physics model
    qed = QEDPhysics()
    
    # Calculate basic parameters
    gamma_rel = plasma.relativistic_gamma()
    omega_pe = plasma.plasma_frequency()
    v_phase = plasma.wakefield_phase_velocity()
    
    # Simulate plasma wakefield
    z_grid = np.linspace(0, 50e-6, 1000)  # 50 micron simulation domain
    density_profile = plasma.n_e * (1 + 0.1 * np.sin(2 * np.pi * z_grid / (2 * np.pi * c / plasma.omega_pe)))
    
    # Calculate effective velocity in wakefield
    v_fluid = 0.1 * c * np.sin(2 * np.pi * z_grid / (2 * np.pi * c / plasma.omega_pe))
    c_sound = 0.1 * c  # Effective sound speed in wakefield (simplified)
    
    # Find horizon locations
    horizon_locations = horizon.horizon_condition(v_fluid, c_sound)
    
    if np.any(horizon_locations):
        # Calculate surface gravity where horizon exists
        dv_dx = np.gradient(v_fluid, z_grid)
        surface_gravity = horizon.surface_gravity(dv_dx[horizon_locations])
        
        if len(surface_gravity) > 0:
            hawking_temp = horizon.hawking_temperature(surface_gravity[0])
        else:
            hawking_temp = 0
    else:
        hawking_temp = 0
        surface_gravity = 0
    
    # Calculate QED effects
    E_max = np.sqrt(2 * laser_params['intensity'] / (c * epsilon_0))
    pair_rate = qed.pair_production_rate(E_max)
    
    return {
        'plasma_frequency': omega_pe,
        'relativistic_gamma': gamma_rel,
        'phase_velocity': v_phase,
        'density_profile': density_profile,
        'horizon_exists': np.any(horizon_locations),
        'horizon_positions': z_grid[horizon_locations] if np.any(horizon_locations) else [],
        'surface_gravity': surface_gravity if isinstance(surface_gravity, (float, int)) else float(surface_gravity) if len(surface_gravity) > 0 else 0,
        'hawking_temperature': hawking_temp,
        'pair_production_rate': pair_rate,
        'critical_field': qed.critical_field()
    }


if __name__ == "__main__":
    # Example usage
    laser_params = {
        'intensity': 1e18,  # W/m^2
        'wavelength': 800e-9,  # m
        'pulse_duration': 30e-15  # s
    }
    
    plasma_params = {
        'density': 1e18,  # m^-3
        'temperature': 10000  # K
    }
    
    result = simulate_analog_black_hole(laser_params, plasma_params)
    print("Analog Black Hole Simulation Results:")
    print(f"Plasma frequency: {result['plasma_frequency']:.2e} rad/s")
    print(f"Relativistic gamma: {result['relativistic_gamma']:.2f}")
    print(f"Horizon exists: {result['horizon_exists']}")
    print(f"Hawking temperature: {result['hawking_temperature']:.2e} K")
    print(f"Pair production rate: {result['pair_production_rate']:.2e} m^-3 s^-1")
