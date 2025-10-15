"""
Convergence testing for the physics simulation

This module performs proper convergence testing to ensure
the numerical methods are working correctly.
"""

import numpy as np
from scipy.constants import c, hbar, k
import matplotlib.pyplot as plt

# Import our physics modules
from physics_engine.plasma_models.plasma_physics import PlasmaPhysicsModel
from physics_engine.plasma_models.laser_plasma_interaction import LaserPlasmaDynamics
from physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory

def test_spatial_convergence():
    """
    Test spatial convergence of the Maxwell-fluid solver
    """
    print("Testing Spatial Convergence...")
    
    # Test parameters
    laser_params = {
        'intensity': 1e17,
        'wavelength': 800e-9,
        'duration': 30e-15
    }
    
    plasma_params = {
        'density': 1e17,
        'temperature': 10000
    }
    
    # Different spatial resolutions
    nx_values = [50, 100, 200, 400]
    results = []
    
    for nx in nx_values:
        simulation = LaserPlasmaDynamics(laser_params, plasma_params)
        sim_result = simulation.simulate_laser_plasma_interaction(
            x_range=(-20e-6, 20e-6),
            t_range=(0, 50e-15),
            n_x=nx,
            n_t=50
        )
        
        # Extract a key quantity to test convergence (e.g., max field amplitude)
        max_field = np.max(np.abs(sim_result['field_evolution']))
        results.append(max_field)
    
    # Check if results are converging (difference between successive refinements decreases)
    diffs = np.diff(results)
    convergence_rates = []
    
    for i in range(1, len(diffs)):
        if abs(diffs[i-1]) > 1e-15:
            rate = np.log(abs(diffs[i]/diffs[i-1])) / np.log(2.0)
            convergence_rates.append(rate)
    
    print(f"  Grid points: {nx_values}")
    print(f"  Max field values: {[f'{val:.2e}' for val in results]}")
    print(f"  Differences: {[f'{diff:.2e}' for diff in diffs]}")
    
    if len(convergence_rates) > 0:
        avg_rate = np.mean(convergence_rates)
        print(f"  Average convergence rate: {avg_rate:.2f}")
        
        # For second-order methods, expect convergence rate ~2
        if avg_rate > 1.5:
            print("  ✅ Spatial convergence: PASSED")
            return True
        else:
            print("  ⚠️  Spatial convergence: WARNING - rate below expected")
            return False
    else:
        print("  ❌ Spatial convergence: FAILED - no convergence detected")
        return False

def test_temporal_convergence():
    """
    Test temporal convergence of the Maxwell-fluid solver
    """
    print("Testing Temporal Convergence...")
    
    # Test parameters
    laser_params = {
        'intensity': 1e17,
        'wavelength': 800e-9,
        'duration': 30e-15
    }
    
    plasma_params = {
        'density': 1e17,
        'temperature': 10000
    }
    
    # Different temporal resolutions
    nt_values = [25, 50, 100, 200]
    results = []
    
    for nt in nt_values:
        simulation = LaserPlasmaDynamics(laser_params, plasma_params)
        sim_result = simulation.simulate_laser_plasma_interaction(
            x_range=(-20e-6, 20e-6),
            t_range=(0, 50e-15),
            n_x=100,
            n_t=nt
        )
        
        # Extract a key quantity to test convergence
        max_field = np.max(np.abs(sim_result['field_evolution']))
        results.append(max_field)
    
    # Check if results are converging
    diffs = np.diff(results)
    convergence_rates = []
    
    for i in range(1, len(diffs)):
        if abs(diffs[i-1]) > 1e-15:
            rate = np.log(abs(diffs[i]/diffs[i-1])) / np.log(2.0)
            convergence_rates.append(rate)
    
    print(f"  Time steps: {nt_values}")
    print(f"  Max field values: {[f'{val:.2e}' for val in results]}")
    print(f"  Differences: {[f'{diff:.2e}' for diff in diffs]}")
    
    if len(convergence_rates) > 0:
        avg_rate = np.mean(convergence_rates)
        print(f"  Average convergence rate: {avg_rate:.2f}")
        
        # For second-order methods, expect convergence rate ~2
        if avg_rate > 1.5:
            print("  ✅ Temporal convergence: PASSED")
            return True
        else:
            print("  ⚠️  Temporal convergence: WARNING - rate below expected")
            return False
    else:
        print("  ❌ Temporal convergence: FAILED - no convergence detected")
        return False

def test_hawking_spectrum_convergence():
    """
    Test convergence of Hawking spectrum calculation
    """
    print("Testing Hawking Spectrum Convergence...")
    
    # Test with different frequency resolutions
    n_freq_values = [100, 500, 1000, 2000]
    results = []
    
    # Fixed surface gravity for test
    surface_gravity = 1e12  # s⁻¹
    T_H = hbar * surface_gravity / (2 * np.pi * k)
    
    for n_freq in n_freq_values:
        qft = QuantumFieldTheory(surface_gravity=surface_gravity)
        
        # Calculate spectrum over fixed range with different resolutions
        freq_range = (1e12, 1e16)  # 1 THz to 10 PHz
        frequencies = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freq)
        omega = 2 * np.pi * frequencies
        
        spectrum = qft.hawking_spectrum(omega)
        
        # Integrate total power
        total_power = np.trapz(spectrum, omega) / (2 * np.pi)
        results.append(total_power)
    
    # Check convergence
    diffs = np.diff(results)
    print(f"  Frequency points: {n_freq_values}")
    print(f"  Integrated power: {[f'{power:.2e}' for power in results]}")
    print(f"  Differences: {[f'{diff:.2e}' for diff in diffs]}")
    
    # For convergent integration, differences should decrease
    if len(diffs) > 1:
        # Check if last difference is smaller than first
        if abs(diffs[-1]) < abs(diffs[0]):
            print("  ✅ Hawking spectrum convergence: PASSED")
            return True
        else:
            print("  ⚠️  Hawking spectrum convergence: WARNING")
            return False
    else:
        print("  ❌ Hawking spectrum convergence: FAILED - insufficient data")
        return False

def test_parameter_sensitivity():
    """
    Test sensitivity of results to small parameter changes
    """
    print("Testing Parameter Sensitivity...")
    
    # Baseline parameters
    baseline_laser_params = {
        'intensity': 1e17,
        'wavelength': 800e-9,
        'duration': 30e-15
    }
    
    baseline_plasma_params = {
        'density': 1e17,
        'temperature': 10000
    }
    
    # Run baseline simulation
    baseline_sim = LaserPlasmaDynamics(baseline_laser_params, baseline_plasma_params)
    baseline_result = baseline_sim.simulate_laser_plasma_interaction(
        x_range=(-20e-6, 20e-6),
        t_range=(0, 50e-15),
        n_x=100,
        n_t=50
    )
    
    baseline_max_field = np.max(np.abs(baseline_result['field_evolution']))
    
    # Run simulation with slightly perturbed parameters
    perturbed_laser_params = {
        'intensity': 1.01e17,  # 1% increase
        'wavelength': 800e-9,
        'duration': 30e-15
    }
    
    perturbed_sim = LaserPlasmaDynamics(perturbed_laser_params, baseline_plasma_params)
    perturbed_result = perturbed_sim.simulate_laser_plasma_interaction(
        x_range=(-20e-6, 20e-6),
        t_range=(0, 50e-15),
        n_x=100,
        n_t=50
    )
    
    perturbed_max_field = np.max(np.abs(perturbed_result['field_evolution']))
    
    # Calculate relative change
    relative_change = abs(perturbed_max_field - baseline_max_field) / baseline_max_field
    relative_parameter_change = 0.01  # 1% parameter change
    
    print(f"  Baseline max field: {baseline_max_field:.2e}")
    print(f"  Perturbed max field: {perturbed_max_field:.2e}")
    print(f"  Relative field change: {relative_change:.2e}")
    print(f"  Relative parameter change: {relative_parameter_change:.2e}")
    
    # For well-behaved systems, field change should be proportional to parameter change
    if relative_change < 10 * relative_parameter_change:  # Allow for some nonlinearity
        print("  ✅ Parameter sensitivity: PASSED")
        return True
    else:
        print("  ⚠️  Parameter sensitivity: WARNING - excessive sensitivity")
        return False

def run_convergence_suite():
    """
    Run all convergence tests
    """
    print("RUNNING CONVERGENCE TESTING SUITE")
    print("=" * 40)
    
    results = []
    
    try:
        spatial_result = test_spatial_convergence()
        results.append(spatial_result)
        
        temporal_result = test_temporal_convergence()
        results.append(temporal_result)
        
        spectrum_result = test_hawking_spectrum_convergence()
        results.append(spectrum_result)
        
        sensitivity_result = test_parameter_sensitivity()
        results.append(sensitivity_result)
        
        print(f"\nCONVERGENCE TESTING SUMMARY:")
        print(f"  Spatial convergence: {'✅ PASSED' if spatial_result else '❌ FAILED'}")
        print(f"  Temporal convergence: {'✅ PASSED' if temporal_result else '❌ FAILED'}")
        print(f"  Spectrum convergence: {'✅ PASSED' if spectrum_result else '❌ FAILED'}")
        print(f"  Parameter sensitivity: {'PASSED' if sensitivity_result else 'FAILED'}")
        
        overall_success = all(results)
        print(f"\n{'ALL CONVERGENCE TESTS PASSED' if overall_success else 'SOME CONVERGENCE TESTS FAILED'}")
        
        return overall_success
        
    except Exception as e:
        print(f"❌ CONVERGENCE TESTING FAILED: {e}")
        return False

if __name__ == "__main__":
    success = run_convergence_suite()
    exit(0 if success else 1)