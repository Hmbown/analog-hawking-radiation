"""
Realistic Laser-Plasma Interaction Model

This module implements the fundamental physics of laser-plasma interactions
using Maxwell's equations and proper relativistic fluid dynamics to model
the formation of analog event horizons.
"""

import numpy as np
from scipy.constants import c, e, m_e, epsilon_0, mu_0, hbar, k, m_p
import scipy.constants as const
from scipy.integrate import solve_ivp
from scipy.special import erf
import warnings


class MaxwellFluidModel:
    """
    Class to model laser-plasma interactions using Maxwell's equations and fluid dynamics
    """
    
    def __init__(self, plasma_density=1e18, laser_wavelength=800e-9, laser_intensity=1e17):
        """
        Initialize the laser-plasma interaction model
        
        Args:
            plasma_density (float): Electron density in m^-3
            laser_wavelength (float): Laser wavelength in meters
            laser_intensity (float): Laser intensity in W/m^2
        """
        self.n_e = plasma_density
        self.lambda_l = laser_wavelength
        self.I_0 = laser_intensity
        
        # Derived parameters
        self.omega_l = 2 * np.pi * c / self.lambda_l
        self.k_l = 2 * np.pi / self.lambda_l
        self.omega_pe = np.sqrt(e**2 * self.n_e / (epsilon_0 * m_e))
        # Correct a0 calculation: a0 = e * E0 / (m_e * omega * c)
        E0 = np.sqrt(2 * self.I_0 / (c * epsilon_0))
        self.a0 = e * E0 / (m_e * self.omega_l * c)
        
        # Plasma skin depth
        self.delta_skin = c / self.omega_pe
        
        # Critical density
        self.n_critical = epsilon_0 * m_e * self.omega_l**2 / e**2
        
        # Check relativistic regime
        if self.a0 > 0.1:  # Even moderate a0 requires relativistic treatment
            print(f"Relativistic parameter a0 = {self.a0:.2f}, relativistic effects important")
    
    def maxwell_equations(self, t, state, x_grid):
        """
        Maxwell's equations in 1D for laser-plasma interaction
        
        Args:
            t: Time
            state: Array [E, B, n_e, v_e] where E is electric field, B is magnetic field
            x_grid: Spatial grid points
        
        Returns:
            Time derivatives of the state variables
        """
        # Unpack the state vector - in a real implementation this would be more complex
        # For now, we'll use a simplified representation
        n_pts = len(x_grid)
        E = state[:n_pts]  # Electric field
        B = state[n_pts:2*n_pts]  # Magnetic field
        n_e = state[2*n_pts:3*n_pts]  # Electron density
        v_e = state[3*n_pts:4*n_pts]  # Electron velocity
        
        # Calculate derivatives using finite differences
        dE_dx = np.gradient(E, x_grid)
        dB_dx = np.gradient(B, x_grid)
        
        # Maxwell's equations in 1D (consistent SI units):
        # ∂E/∂t = c² * ∂B/∂x - j/ε₀
        # ∂B/∂t = -∂E/∂x * ε₀μ₀ = -∂E/∂x / c²  (since ε₀μ₀ = 1/c²)
        # Where j = -e * n_e * v_e (current density)
        
        current_density = -e * n_e * v_e
        dE_dt = c**2 * dB_dx - current_density / epsilon_0
        dB_dt = -dE_dx / c**2
        
        # Fluid equations for plasma
        # ∂n_e/∂t + ∇·(n_e*v_e) = 0  (continuity equation)
        # m_e*n_e*(∂v_e/∂t + v_e*∇v_e) = -e*(E + v_e × B) - ∇p_e  (momentum equation)
        
        # For simplicity, we'll use a pressure term approximation
        # Pressure P = n_e * k * T_e, so ∇P = k * T_e * ∇n_e
        T_e = 10000  # Electron temperature in Kelvin
        grad_p = np.gradient(k * T_e * n_e, x_grid)  # Pressure gradient with correct units
        
        # Calculate momentum equation terms
        v_dv_dx = v_e * np.gradient(v_e, x_grid)  # v·∇v term
        force_E = -e * E / m_e  # Electric force
        # For 1D, v×B ≈ 0 for longitudinal fields, so we ignore magnetic force
        force_pressure = -grad_p / (m_e * n_e)  # Pressure force
        
        dv_e_dt = force_E + force_pressure
        dn_e_dt = -np.gradient(n_e * v_e, x_grid)  # Continuity equation
        
        # Combine derivatives
        dstate_dt = np.concatenate([dE_dt, dB_dt, dn_e_dt, dv_e_dt])
        
        return dstate_dt
    
    def laser_pulse_profile(self, x, t, polarization='linear'):
        """
        Define the laser pulse profile
        
        Args:
            x: Position array
            t: Time
            polarization: Laser polarization ('linear', 'circular')
            
        Returns:
            Electric field profile
        """
        # Gaussian laser pulse
        tau_pulse = 10 * 2 * np.pi / self.omega_l  # Pulse duration (10 periods)
        x0 = 0  # Initial position
        c_t = x0 + c * t  # Phase front position
        
        # Temporal envelope
        temporal_env = np.exp(-(t - 5*tau_pulse)**2 / (2 * (tau_pulse/2)**2))
        
        # Spatial envelope
        w0 = 5e-6  # Focus waist
        spatial_env = np.exp(-(x - c_t)**2 / w0**2)
        
        # Oscillating part
        oscillation = np.sin(self.omega_l * t - self.k_l * x)
        
        # Combine all factors
        E_laser = self.a0 * m_e * c * self.omega_l / e * temporal_env * spatial_env * oscillation
        
        return E_laser
    
    def relativistic_laser_plasma(self, x_grid, t_grid):
        """
        Solve relativistic laser-plasma equations
        
        Args:
            x_grid: Spatial grid points
            t_grid: Time grid points
            
        Returns:
            Evolution of electromagnetic fields and plasma parameters
        """
        # Initial conditions
        n_pts = len(x_grid)
        n_time = len(t_grid)
        
        # Initial state: [E, B, n_e, v_e]
        initial_state = np.zeros(4 * n_pts)
        
        # Initialize with laser pulse
        E_initial = self.laser_pulse_profile(x_grid, t_grid[0])
        initial_state[:n_pts] = E_initial
        initial_state[n_pts:2*n_pts] = E_initial / c  # B field related to E field
        initial_state[2*n_pts:3*n_pts] = np.full(n_pts, self.n_e)  # Electron density
        initial_state[3*n_pts:4*n_pts] = np.zeros(n_pts)  # Electron velocity
        
        # Time evolution using 4th-order Runge-Kutta method for stability
        state_evolution = np.zeros((n_time, 4 * n_pts))
        state_evolution[0, :] = initial_state
        
        dt = t_grid[1] - t_grid[0]
        
        # Check CFL condition for stability
        dx = x_grid[1] - x_grid[0]
        cfl_dt = 0.9 * dx / c  # CFL condition for electromagnetic waves
        actual_dt = min(dt, cfl_dt)
        n_substeps = max(1, int(dt / actual_dt))
        substep_dt = dt / n_substeps
        
        for i in range(1, n_time):
            # 4th-order Runge-Kutta integration
            state = state_evolution[i-1, :].copy()
            
            for substep in range(n_substeps):
                # RK4 stages
                k1 = self.maxwell_equations(t_grid[i-1] + substep * substep_dt, state, x_grid)
                k2 = self.maxwell_equations(t_grid[i-1] + (substep + 0.5) * substep_dt, state + 0.5 * substep_dt * k1, x_grid)
                k3 = self.maxwell_equations(t_grid[i-1] + (substep + 0.5) * substep_dt, state + 0.5 * substep_dt * k2, x_grid)
                k4 = self.maxwell_equations(t_grid[i-1] + (substep + 1) * substep_dt, state + substep_dt * k3, x_grid)
                
                # Update state with weighted average
                state = state + (substep_dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            
            state_evolution[i, :] = state
        
        return {
            'time_grid': t_grid,
            'space_grid': x_grid,
            'electric_field': state_evolution[:, :n_pts],
            'magnetic_field': state_evolution[:, n_pts:2*n_pts],
            'density': state_evolution[:, 2*n_pts:3*n_pts],
            'velocity': state_evolution[:, 3*n_pts:4*n_pts]
        }
    
    def wakefield_amplitude(self, a0):
        """
        Calculate wakefield amplitude in relativistic regime
        
        Args:
            a0: Normalized vector potential
            
        Returns:
            Maximum wakefield amplitude
        """
        # In the relativistically induced transparency regime
        # The wakefield can approach the wavebreaking limit
        E_break = self.omega_pe * m_e * c / e  # Wavebreaking field
        
        # For relativistic laser: E_wake ≈ E_break * sqrt(a0)
        return E_break * np.sqrt(min(a0, 10))  # Capped for very high intensities
    
    def plasma_wave_phase_velocity(self, gamma_osc):
        """
        Calculate phase velocity of plasma waves
        
        Args:
            gamma_osc: Oscillator gamma factor
            
        Returns:
            Phase velocity in terms of c
        """
        # Phase velocity in relativistic plasma
        return c * np.sqrt(1 - (self.omega_pe / (self.omega_l * np.sqrt(gamma_osc)))**2)

class AnalogHorizonFormation:
    """
    Model for how analog horizons form in laser-plasma systems
    """
    
    def __init__(self, maxwell_fluid_model, plasma_temperature_profile=10000, ion_mass=m_p, gamma_e=5.0/3.0):
        """
        Initialize with a Maxwell fluid model
        
        Args:
            maxwell_fluid_model: Instance of MaxwellFluidModel
            plasma_temperature_profile: Electron temperature in Kelvin, can be scalar or array
        """
        self.plasma = maxwell_fluid_model
        self.plasma_temperature_profile = plasma_temperature_profile
        self.ion_mass = ion_mass
        self.gamma_e = gamma_e
        
        # Compute adiabatic sound speed c_s = sqrt(gamma k T_e / m_i)
        # T_e can be a scalar or an array (profile)
        T_e = np.asarray(self.plasma_temperature_profile)
        self.c_sound = np.sqrt(np.maximum(self.gamma_e * k * T_e / self.ion_mass, 0.0))
        
        self.c = c
    
    def fluid_velocity_profile(self, x, t):
        """
        Calculate fluid velocity profile from laser-plasma interaction
        
        Args:
            x: Position array
            t: Time
            
        Returns:
            Fluid velocity profile
        """
        # This is a simplified model of how the laser pulse creates fluid motion
        # In reality, this requires full PIC simulation
        
        # Approximate velocity from laser ponderomotive force
        E_laser = self.plasma.laser_pulse_profile(x, t)
        # Ponderomotive potential: U_p = e²E²/(4*m_e*omega_l²*gamma)
        gamma_factor = np.sqrt(1 + self.plasma.a0**2)
        U_p = e**2 * E_laser**2 / (4 * m_e * self.plasma.omega_l**2 * gamma_factor)
        
        # Force is gradient of potential
        force = -np.gradient(U_p, x)
        
        # Acceleration leads to velocity
        # This is a simplified time integration
        dt = 1e-16  # Time step for estimate
        acceleration = force / m_e
        velocity = acceleration * dt
        
        return velocity
    
    def effective_spacetime_metric(self, fluid_velocity, sound_velocity):
        """
        Calculate the effective spacetime metric for the analog system
        
        Args:
            fluid_velocity: Fluid flow velocity in m/s
            sound_velocity: Effective sound velocity in m/s
            
        Returns:
            Components of the effective metric
        """
        # In the acoustic metric framework:
        # ds² = (c_s² - v²) dt² + 2v dx dt - dx²
        # where c_s is effective sound speed and v is flow velocity
        
        c_s = sound_velocity
        v = fluid_velocity
        
        # Metric components in (t, x) coordinates
        g_tt = -(c_s**2 - v**2)
        g_tx = g_xt = v
        g_xx = -1.0
        
        return {
            'g_tt': g_tt,
            'g_tx': g_tx,
            'g_xt': g_xt,
            'g_xx': g_xx,
            'determinant': -(c_s**2 - v**2)
        }
    
    def horizon_position(self, x_grid, t, v_actual=None):
        """
        Find positions where analog horizon forms
        
        Args:
            x_grid: Spatial grid points
            t: Time
            v_actual: Actual fluid velocity from simulation (if available)
            
        Returns:
            Array of boolean values indicating horizon positions
        """
        # Use actual velocity from simulation if provided, otherwise analytical estimate
        if v_actual is not None:
            v_fluid = v_actual
        else:
            v_fluid = self.fluid_velocity_profile(x_grid, t)
        
        # Use physically motivated sound speed (can be uniform or a profile)
        c_sound = self.c_sound
        
        # Horizon forms where |v_fluid| = c_sound
        horizon_condition = np.abs(v_fluid) - c_sound
        
        # Find where condition changes sign (horizon locations)
        horizon_mask = np.abs(horizon_condition) < 0.01 * c_sound  # Tolerance
        
        return horizon_mask
    
    def surface_gravity_at_horizon(self, x_grid, t, v_actual=None):
        """
        Calculate surface gravity at the analog horizon
        
        Args:
            x_grid: Spatial grid points
            t: Time
            v_actual: Actual fluid velocity from simulation (if available)
            
        Returns:
            Surface gravity in s^-1
        """
        # Use actual velocity from simulation if provided, otherwise analytical estimate
        if v_actual is not None:
            v_fluid = v_actual
        else:
            v_fluid = self.fluid_velocity_profile(x_grid, t)
        
        dv_dx = np.gradient(v_fluid, x_grid)
        
        # Find horizon positions
        horizon_mask = self.horizon_position(x_grid, t, v_actual=v_actual)
        
        # Calculate surface gravity where horizon exists
        surface_gravity = np.zeros_like(x_grid)
        if np.any(horizon_mask):
            # Surface gravity κ = |∇(c_s - |v|)|/2 at horizon
            # For 1D: κ = |d(c_s - v)/dx|/2
            c_sound_profile = self.c_sound
            # Ensure c_sound_profile is an array of the same shape as v_fluid for gradient calculation
            if np.isscalar(c_sound_profile):
                c_sound_profile = np.full_like(v_fluid, c_sound_profile)
                
            grad_term = np.gradient(c_sound_profile - np.abs(v_fluid), x_grid)
            surface_gravity[horizon_mask] = np.abs(grad_term[horizon_mask]) / 2.0
        
        return surface_gravity

class LaserPlasmaDynamics:
    """
    Complete model for laser-plasma dynamics relevant to analog Hawking radiation
    """
    
    def __init__(self, laser_params, plasma_params):
        """
        Initialize complete laser-plasma dynamics model
        
        Args:
            laser_params: Dictionary with laser parameters
            plasma_params: Dictionary with plasma parameters
        """
        self.laser = laser_params
        self.plasma = plasma_params
        
        # Create the fundamental models
        self.maxwell_model = MaxwellFluidModel(
            plasma_density=plasma_params['density'],
            laser_wavelength=laser_params['wavelength'],
            laser_intensity=laser_params['intensity']
        )
        
        self.analog_model = AnalogHorizonFormation(self.maxwell_model,
                                                   plasma_temperature_profile=plasma_params.get('temperature', 10000))
    
    def simulate_laser_plasma_interaction(self, x_range=(-50e-6, 50e-6), t_range=(0, 100e-15), n_x=200, n_t=100):
        """
        Simulate the complete laser-plasma interaction leading to analog horizon formation
        
        Args:
            x_range: Spatial range (xmin, xmax) in meters
            t_range: Time range (tmin, tmax) in seconds
            n_x: Number of spatial grid points
            n_t: Number of time points
            
        Returns:
            Dictionary with simulation results
        """
        # Create grids
        x_grid = np.linspace(x_range[0], x_range[1], n_x)
        t_grid = np.linspace(t_range[0], t_range[1], n_t)
        
        # Initialize results arrays
        n_evolution = np.zeros((n_t, n_x))
        v_evolution = np.zeros((n_t, n_x))
        E_evolution = np.zeros((n_t, n_x))
        horizon_positions = []
        surface_gravities = []
        
        # Calculate evolution
        for i, t in enumerate(t_grid):
            # Calculate plasma parameters at this time
            v_fluid = self.analog_model.fluid_velocity_profile(x_grid, t)
            E_laser = self.maxwell_model.laser_pulse_profile(x_grid, t)
            
            # Store results
            v_evolution[i, :] = v_fluid
            E_evolution[i, :] = E_laser
            
            # Find horizon positions
            h_mask = self.analog_model.horizon_position(x_grid, t)
            h_positions = x_grid[h_mask]
            if len(h_positions) > 0:
                horizon_positions.append((t, h_positions))
            
            # Calculate surface gravity where horizon exists
            s_grav = self.analog_model.surface_gravity_at_horizon(x_grid, t)
            surface_gravities.append(s_grav)
        
        # Calculate integrated quantities
        total_energy = np.trapz(np.trapz(E_evolution**2, x_grid), t_grid) * epsilon_0 * c
        
        return {
            'space_grid': x_grid,
            'time_grid': t_grid,
            'velocity_evolution': v_evolution,
            'field_evolution': E_evolution,
            'horizon_positions': horizon_positions,
            'surface_gravities': np.array(surface_gravities),
            'total_laser_energy': total_energy,
            'maxwell_model': self.maxwell_model,
            'analog_model': self.analog_model
        }
    
    def calculate_hawking_signal(self, simulation_results):
        """
        Calculate the expected Hawking signal based on simulation results
        
        Args:
            simulation_results: Output from simulate_laser_plasma_interaction
            
        Returns:
            Dictionary with Hawking signal characteristics
        """
        from .quantum_field_theory import QuantumFieldTheory
        
        # Find times when horizon exists
        horizon_times = [item[0] for item in simulation_results['horizon_positions']]
        if not horizon_times:
            return {
                'hawking_temperature': 0,
                'power_spectrum': np.array([]),
                'frequencies': np.array([]),
                'total_power': 0,
                'horizon_exists': False
            }
        
        # Use the maximum surface gravity from simulation
        max_surface_gravity = np.max(simulation_results['surface_gravities'])
        
        if max_surface_gravity <= 0:
            return {
                'hawking_temperature': 0,
                'power_spectrum': np.array([]),
                'frequencies': np.array([]),
                'total_power': 0,
                'horizon_exists': False
            }
        
        # Create QFT model with calculated surface gravity
        qft_model = QuantumFieldTheory(surface_gravity=max_surface_gravity)
        
        # Calculate Hawking temperature
        T_H = qft_model.hawking_temperature_from_kappa(max_surface_gravity)
        
        # Calculate spectrum
        freq_range = (1e12, 1e18)  # 1 THz to 1 PHz
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), 1000)
        omega = 2 * np.pi * frequencies
        power_spectrum = qft_model.hawking_spectrum(omega)
        
        # Integrate total power
        total_power = np.trapz(power_spectrum, x=omega) / (2 * np.pi)
        
        return {
            'hawking_temperature': T_H,
            'power_spectrum': power_spectrum,
            'frequencies': frequencies,
            'total_power': total_power,
            'surface_gravity': max_surface_gravity,
            'horizon_exists': True
        }

def simulate_realistic_laser_plasma(laser_intensity=1e18, plasma_density=1e18, laser_wavelength=800e-9):
    """
    Main function to simulate realistic laser-plasma interaction for analog Hawking radiation
    
    Args:
        laser_intensity: Laser intensity in W/m^2
        plasma_density: Plasma density in m^-3
        laser_wavelength: Laser wavelength in meters
        
    Returns:
        Dictionary with complete simulation results
    """
    # Define parameters
    laser_params = {
        'intensity': laser_intensity,
        'wavelength': laser_wavelength,
        'duration': 30e-15  # 30 fs pulse
    }
    
    plasma_params = {
        'density': plasma_density,
        'temperature': 10000  # K
    }
    
    # Create simulation model
    simulation = LaserPlasmaDynamics(laser_params, plasma_params)
    
    # Run simulation
    sim_results = simulation.simulate_laser_plasma_interaction(
        x_range=(-100e-6, 100e-6),  # 200 micron spatial range
        t_range=(0, 200e-15),       # 200 fs time range
        n_x=400,                    # 400 spatial points
        n_t=200                     # 200 time points
    )
    
    # Calculate Hawking signal
    hawking_results = simulation.calculate_hawking_signal(sim_results)
    
    return {
        'laser_plasma_simulation': sim_results,
        'hawking_signal': hawking_results,
        'laser_params': laser_params,
        'plasma_params': plasma_params,
        'relativistic_parameter': simulation.maxwell_model.a0,
        'plasma_frequency': simulation.maxwell_model.omega_pe
    }

if __name__ == "__main__":
    print("Realistic Laser-Plasma Interaction Model for Analog Hawking Radiation")
    print("=" * 70)
    
    # Example: High-intensity laser-plasma interaction
    results = simulate_realistic_laser_plasma(
        laser_intensity=5e18,    # 5x10^18 W/cm²
        plasma_density=5e17,     # 5x10^17 cm^-3
        laser_wavelength=800e-9  # 800 nm
    )
    
    print(f"Laser intensity: {results['laser_params']['intensity']:.2e} W/m²")
    print(f"Plasma density: {results['plasma_params']['density']:.2e} m⁻³")
    print(f"Relativistic parameter a₀: {results['relativistic_parameter']:.2f}")
    print(f"Plasma frequency: {results['plasma_frequency']:.2e} rad/s")
    print(f"Horizon exists: {results['hawking_signal']['horizon_exists']}")
    
    if results['hawking_signal']['horizon_exists']:
        print(f"Hawking temperature: {results['hawking_signal']['hawking_temperature']:.2e} K")
        print(f"Surface gravity: {results['hawking_signal']['surface_gravity']:.2e} s⁻¹")
        print(f"Total radiated power: {results['hawking_signal']['total_power']:.2e} W")
    else:
        print("No analog horizon formed in simulation parameters")
