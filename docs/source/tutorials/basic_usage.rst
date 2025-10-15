Basic Usage Tutorial
====================

This tutorial provides a step-by-step introduction to using the Analog Hawking Radiation Simulation Framework.

Installation
------------

First, ensure the framework is installed:

.. code-block:: bash

   pip install analog-hawking-radiation

If you're developing or want the latest version, you can install from source:

.. code-block:: bash

   git clone https://github.com/Shannon-Labs/bayesian-analysis-hawking-radiation.git
   cd bayesian-analysis-hawking-radiation
   pip install -e .

Importing the Framework
-----------------------

To use the framework in your Python code, import the necessary modules:

.. code-block:: python

   import numpy as np
   from analog_hawking.physics_engine.plasma_models.plasma_physics import PlasmaPhysicsModel
   from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
   from analog_hawking.detection.radio_snr import sweep_time_for_5sigma

Creating a Plasma Model
-----------------------

Start by creating a plasma model with your desired parameters:

.. code-block:: python

   # Create a plasma model with typical parameters
   plasma = PlasmaPhysicsModel(
       plasma_density=1e18,      # Electron density in m^-3
       laser_wavelength=800e-9,  # Laser wavelength in meters
       laser_intensity=1e17      # Laser intensity in W/m^2
   )

   # Calculate plasma frequency
   omega_pe = plasma.plasma_frequency()
   print(f"Plasma frequency: {omega_pe:.2e} rad/s")

Setting Up a Simulation Grid
----------------------------

Create spatial and temporal grids for your simulation:

.. code-block:: python

   # Spatial grid
   x = np.linspace(-50e-6, 50e-6, 1000)  # 1000 points from -50 to 50 micrometers

   # Example velocity profile (this would typically come from a plasma simulation)
   v = 0.1 * 3e8 * np.tanh(x / 10e-6)  # Velocity profile with characteristic scale

   # Sound speed (this would depend on plasma temperature)
   T_e = 10000  # Electron temperature in Kelvin
   c_s = plasma.sound_speed(T_e)
   c_s_profile = np.full_like(x, c_s)  # Uniform sound speed for simplicity

Finding Horizons
----------------

Use the horizon detection algorithm to find analog event horizons:

.. code-block:: python

   # Find horizons with uncertainty quantification
   horizon_result = find_horizons_with_uncertainty(x, v, c_s_profile)

   print(f"Number of horizons found: {len(horizon_result.positions)}")
   if len(horizon_result.positions) > 0:
       print(f"Horizon positions: {horizon_result.positions}")
       print(f"Surface gravity: {horizon_result.kappa}")
       print(f"Uncertainty: {horizon_result.kappa_err}")

Calculating Detection Feasibility
---------------------------------

Estimate the time required to detect Hawking radiation:

.. code-block:: python

   # Example: Calculate integration time for 5σ detection
   if len(horizon_result.kappa) > 0:
       # Convert surface gravity to Hawking temperature
       from scipy.constants import hbar, k
       T_H = hbar * horizon_result.kappa[0] / (2 * np.pi * k)
       
       # System parameters for detection
       T_sys_vals = np.array([50])  # System temperature in Kelvin
       B_vals = np.array([10e6])    # Bandwidth in Hz
       
       # Calculate integration time
       t_5sigma = sweep_time_for_5sigma(T_sys_vals, B_vals, T_H)
       print(f"Time for 5σ detection: {t_5sigma[0, 0]:.2e} seconds")

Running a Complete Example
--------------------------

Here's a complete example that puts it all together:

.. code-block:: python

   import numpy as np
   from scipy.constants import hbar, k
   from analog_hawking.physics_engine.plasma_models.plasma_physics import PlasmaPhysicsModel
   from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
   from analog_hawking.detection.radio_snr import sweep_time_for_5sigma

   def basic_hawking_simulation():
       """Run a basic analog Hawking radiation simulation."""
       
       # Create plasma model
       plasma = PlasmaPhysicsModel(
           plasma_density=1e18,
           laser_wavelength=800e-9,
           laser_intensity=1e17
       )
       
       # Create spatial grid
       x = np.linspace(-50e-6, 50e-6, 1000)
       
       # Create velocity profile (simplified example)
       v = 0.1 * 3e8 * np.tanh(x / 10e-6)
       
       # Create sound speed profile
       T_e = 10000
       c_s = plasma.sound_speed(T_e)
       c_s_profile = np.full_like(x, c_s)
       
       # Find horizons
       horizon_result = find_horizons_with_uncertainty(x, v, c_s_profile)
       
       print("=== Horizon Detection Results ===")
       print(f"Number of horizons: {len(horizon_result.positions)}")
       
       if len(horizon_result.positions) > 0:
           print(f"Horizon positions: {horizon_result.positions}")
           print(f"Surface gravity κ: {horizon_result.kappa}")
           print(f"Uncertainty Δκ: {horizon_result.kappa_err}")
           
           # Calculate Hawking temperature
           T_H = hbar * horizon_result.kappa[0] / (2 * np.pi * k)
           print(f"Hawking temperature: {T_H:.2e} K")
           
           # Calculate detection time
           T_sys_vals = np.array([50])  # 50 K system temperature
           B_vals = np.array([10e6])    # 10 MHz bandwidth
           t_5sigma = sweep_time_for_5sigma(T_sys_vals, B_vals, T_H)
           
           print(f"Time for 5σ detection: {t_5sigma[0, 0]:.2e} seconds")
       else:
           print("No horizons detected with these parameters.")
       
       return horizon_result

   # Run the simulation
   if __name__ == "__main__":
       result = basic_hawking_simulation()

Next Steps
----------

This basic tutorial shows the fundamental workflow of the framework. For more advanced usage, explore:

1. **Multi-beam simulations** for enhanced gradient effects
2. **Bayesian optimization** for parameter space exploration
3. **Full plasma simulations** with fluid or PIC backends
4. **Advanced detection modeling** with graybody corrections

Refer to the other tutorials for detailed guidance on these topics.