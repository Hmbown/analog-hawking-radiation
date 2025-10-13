#!/usr/bin/env python3
"""
Analysis of the impact of a position-dependent sound speed profile `c_s(x)` on
analog horizon formation.

This script compares the horizon formation condition for a constant `c_s` versus
a realistic `c_s(x)` profile motivated by laser heating of the plasma.

Outputs:
  - figures/cs_profile_impact.png
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.plasma_models.laser_plasma_interaction import MaxwellFluidModel, AnalogHorizonFormation
from scipy.constants import c

def ensure_dirs():
    """Create output directories if they don't exist."""
    os.makedirs('figures', exist_ok=True)

def analyze_cs_profile_impact():
    """
    Generates and plots a comparison of horizon formation for constant vs.
    profiled sound speed.
    """
    ensure_dirs()

    # --- 1. Setup Physics Models ---
    # Shared parameters for the simulation
    plasma_density = 1e24 # m^-3
    laser_wavelength = 800e-9 # m
    laser_intensity = 5e22 # W/m^2
    
    maxwell_model = MaxwellFluidModel(
        plasma_density=plasma_density,
        laser_wavelength=laser_wavelength,
        laser_intensity=laser_intensity
    )

    # --- 2. Define Spatial Grid and Velocity Profile ---
    x_grid = np.linspace(-50e-6, 50e-6, 500)
    t0 = 50e-15 # s, a representative time slice

    # Use a simplified velocity profile for clarity
    # A Gaussian fluid velocity pulse driven by the laser's ponderomotive force
    v_fluid = 0.05 * c * np.exp(-((x_grid - 5e-6) / (15e-6))**2)

    # --- 3. Case A: Constant Sound Speed ---
    const_T = 2e6 # K, uniform background temperature
    analog_model_const = AnalogHorizonFormation(
        maxwell_model,
        plasma_temperature_profile=const_T
    )
    c_s_const = analog_model_const.c_sound

    # --- 4. Case B: Position-Dependent Sound Speed Profile ---
    # Assume a Gaussian temperature profile from laser heating
    T_peak = 10e6 # K
    T_background = 2e6 # K
    T_profile = T_background + (T_peak - T_background) * np.exp(-(x_grid / (20e-6))**2)
    
    analog_model_profile = AnalogHorizonFormation(
        maxwell_model,
        plasma_temperature_profile=T_profile
    )
    c_s_profile = analog_model_profile.c_sound
    
    # --- 5. Generate Plot ---
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot velocities
    ax1.plot(x_grid * 1e6, v_fluid / c, 'k-', label=r'$|v_{fluid}|/c$ (Fluid Velocity)')
    ax1.plot(x_grid * 1e6, np.full_like(x_grid, c_s_const) / c, 'b--', label=r'$c_s/c$ (Constant T)')
    ax1.plot(x_grid * 1e6, c_s_profile / c, 'r-', label=r'$c_s(x)/c$ (Profiled T)')

    # Find and mark horizon points
    horizon_const_indices = np.where(np.abs(v_fluid - c_s_const) < (0.001 * c))[0]
    horizon_profile_indices = np.where(np.abs(v_fluid - c_s_profile) < (0.001 * c))[0]

    if horizon_const_indices.any():
        ax1.plot(x_grid[horizon_const_indices] * 1e6, v_fluid[horizon_const_indices] / c, 'bo', markersize=10, label='Horizon (Constant T)')
    if horizon_profile_indices.any():
        ax1.plot(x_grid[horizon_profile_indices] * 1e6, v_fluid[horizon_profile_indices] / c, 'ro', markersize=10, label='Horizon (Profiled T)')

    ax1.set_xlabel('Position (μm)')
    ax1.set_ylabel('Velocity / c')
    ax1.set_title('Impact of Sound Speed Profile on Analog Horizon Formation')
    ax1.legend(loc='upper left')
    ax1.grid(True, linestyle=':')

    # --- 6. Add Temperature Profile on a Second Y-axis ---
    ax2 = ax1.twinx()
    ax2.plot(x_grid * 1e6, T_profile / 1e6, 'g--', alpha=0.5, label='Temperature Profile (MK)')
    ax2.set_ylabel('Temperature (MK)', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.savefig('figures/cs_profile_impact.png', dpi=200)
    print('Saved figure to figures/cs_profile_impact.png')

if __name__ == '__main__':
    analyze_cs_profile_impact()
