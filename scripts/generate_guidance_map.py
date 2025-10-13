"""
Generates a 2D visualization of the merit score landscape and highlights
the optimal parameters found by the Bayesian optimization.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physics_engine.optimization.merit_function import GlowMeritFunction

def plot_guidance_map():
    """
    Generates and saves a plot of the parameter landscape and the optimal point.
    """
    print("Generating new experimental guidance map...")
    os.makedirs('figures', exist_ok=True)

    # --- 1. Define the Parameter Space to VIsualize ---
    # We will create a 2D slice of the landscape, holding one parameter constant.
    # Let's vary plasma density and laser intensity, keeping T_peak at its optimal value.
    
    plasma_density_range = np.linspace(1e23, 5e24, 20)
    laser_intensity_range = np.linspace(1e22, 1e23, 20)
    
    # The optimal parameters found in the previous step
    optimal_params = {
        'plasma_density': 4.758e+24,
        'laser_intensity': 1.152e+22,
        'T_peak': 1.115e+07
    }

    # --- 2. Calculate the Merit Score across the Grid ---
    merit_grid = np.zeros((len(laser_intensity_range), len(plasma_density_range)))

    for i, intensity in enumerate(laser_intensity_range):
        for j, density in enumerate(plasma_density_range):
            print(f"  Calculating merit for point ({i+1}, {j+1}) of ({len(laser_intensity_range)}, {len(plasma_density_range)})...")
            
            base_params = {
                'plasma_density': density,
                'laser_intensity': intensity,
                'T_peak': optimal_params['T_peak'], # Keep T_peak constant at its optimal value
                'T_background': 2e6
            }
            param_uncertainties = {key: 0.1 * val for key, val in base_params.items()}
            snr_config = {'system_temperature': 50, 'bandwidth': 10e6, 'integration_time': 3600}

            merit_func = GlowMeritFunction(
                base_params=base_params,
                param_uncertainties=param_uncertainties,
                snr_config=snr_config,
                n_samples=100 # Lower sample count for faster map generation
            )

            x_grid = np.linspace(-50e-6, 50e-6, 300)
            t0 = 50e-15
            
            merit_grid[i, j] = merit_func.calculate_merit(x_grid, t0)

    # --- 3. Create the Plot ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use a logarithmic color scale for better visualization
    im = ax.imshow(merit_grid, origin='lower', aspect='auto',
                   extent=[plasma_density_range[0], plasma_density_range[-1],
                           laser_intensity_range[0], laser_intensity_range[-1]],
                   norm=LogNorm(vmin=np.min(merit_grid[merit_grid > 0]), vmax=np.max(merit_grid)))

    # Plot the optimal point found by the optimizer
    ax.plot(optimal_params['plasma_density'], optimal_params['laser_intensity'], 'r*',
            markersize=15, label='Optimal Point')

    ax.set_xlabel('Plasma Density (m⁻³)')
    ax.set_ylabel('Laser Intensity (W/m²)')
    ax.set_title('Experimental Guidance Map: Merit Score Landscape')
    ax.legend()
    
    # Format ticks to scientific notation
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1e}'))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x:.1e}'))
    plt.xticks(rotation=45)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Merit Score (P_horizon * SNR)')

    fig.tight_layout()
    plt.savefig('figures/optimal_glow_parameters.png', dpi=200)
    print("\nSaved new guidance map to 'figures/optimal_glow_parameters.png'")

if __name__ == '__main__':
    plot_guidance_map()
