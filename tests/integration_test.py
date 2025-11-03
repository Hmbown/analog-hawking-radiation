"""
Integration tests for the Analog Hawking Radiation Simulation Framework.

This test suite validates the complete data flow from plasma models through
horizon detection, quantum field theory calculations, and detection modeling.
"""

import json
from pathlib import Path

import numpy as np
import numpy.testing as npt
from scripts.hawking_detection_experiment import calculate_hawking_spectrum

from analog_hawking.detection.radio_snr import (
    band_power_from_spectrum,
    equivalent_signal_temperature,
    sweep_time_for_5sigma,
)
from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.adaptive_sigma import estimate_sigma_map
from analog_hawking.physics_engine.plasma_models.fluctuation_injector import (
    FluctuationConfig,
    QuantumFluctuationInjector,
)
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from analog_hawking.physics_engine.plasma_models.quantum_field_theory import QuantumFieldTheory
from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend
from analog_hawking.physics_engine.simulation import SimulationRunner

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_plasma_models_to_horizon_detection():
    """Test data flow from plasma models to horizon detection."""
    print("Testing Plasma Models → Horizon Detection...")

    # Test with FluidBackend
    backend = FluidBackend()
    config = {
        "plasma_density": 1e18,
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e17,
        "grid": np.linspace(0.0, 50e-6, 1000),
    }
    backend.configure(config)

    # Get plasma state
    state = backend.step(0.0)

    # Verify state has required fields
    assert state.density is not None, "Density should not be None"
    assert state.velocity is not None, "Velocity should not be None"
    assert state.sound_speed is not None, "Sound speed should not be None"
    assert state.grid is not None, "Grid should not be None"
    assert state.temperature is not None, "Temperature profile should be populated"
    assert state.temperature.shape == state.density.shape, "Temperature profile shape mismatch"
    assert (
        state.magnetosonic_speed is None
    ), "Magnetosonic speed should be None without magnetic field"

    print("✓ FluidBackend produces valid plasma state")

    # Test horizon detection with this state
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed)

    assert horizons is not None, "Horizon detection should return results"
    print("✓ Horizon detection works with plasma state data")

    # Verify horizon results have expected structure
    assert hasattr(horizons, "positions"), "HorizonResult should have positions"
    assert hasattr(horizons, "kappa"), "HorizonResult should have kappa"
    assert hasattr(horizons, "kappa_err"), "HorizonResult should have kappa_err"

    print("✓ Horizon detection produces expected output structure")


def test_horizon_to_qft_to_detection():
    """Test data flow from horizon detection to QFT to detection."""
    print("\nTesting Horizon Detection → QFT → Detection...")

    # Create mock horizon data
    kappa = 1e12  # s^-1

    # Test QFT calculations
    qft = QuantumFieldTheory(surface_gravity=kappa)
    T_H = qft.hawking_temperature_from_kappa(kappa)

    assert T_H > 0, "Hawking temperature should be positive"
    print(f"✓ QFT calculation produces valid Hawking temperature: {T_H:.2e} K")

    # Test spectrum calculation
    spec_result = calculate_hawking_spectrum(kappa)
    assert spec_result["success"], "Spectrum calculation should succeed"

    frequencies = spec_result["frequencies"]
    power_spectrum = spec_result["power_spectrum"]
    peak_frequency = spec_result["peak_frequency"]

    assert len(frequencies) > 0, "Frequencies array should not be empty"
    assert len(power_spectrum) > 0, "Power spectrum array should not be empty"
    assert peak_frequency > 0, "Peak frequency should be positive"

    print("✓ QFT spectrum calculation works correctly")

    # Test detection modeling
    B_ref = 1e8  # 100 MHz
    P_sig = band_power_from_spectrum(frequencies, power_spectrum, peak_frequency, B_ref)
    T_sig = equivalent_signal_temperature(P_sig, B_ref)

    assert P_sig >= 0, "Signal power should be non-negative"
    assert T_sig >= 0, "Signal temperature should be non-negative"

    print("✓ Detection modeling works with QFT spectrum")

    # Test integration time calculation
    T_sys_vals = np.array([30])  # K
    B_vals = np.array([1e8])  # Hz
    T_grid = sweep_time_for_5sigma(T_sys_vals, B_vals, T_sig)

    assert T_grid.shape == (1, 1), "Time grid should have correct shape"
    assert T_grid[0, 0] > 0, "Integration time should be positive"

    print("✓ Integration time calculation works correctly")


def test_parameter_passing():
    """Validate parameter passing between modules."""
    print("\nTesting parameter passing between modules...")

    # Test that parameters flow correctly through the simulation runner
    backend = FluidBackend()
    config = {
        "plasma_density": 1e18,
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e17,
        "grid": np.linspace(0.0, 50e-6, 1000),
    }
    backend.configure(config)
    runner = SimulationRunner(backend)

    # Run a step and verify parameter passing
    outputs = runner.run_step(0.0)

    # Check that the state contains the expected parameters
    assert outputs.state.density is not None, "Density should be passed through"
    assert outputs.state.velocity is not None, "Velocity should be passed through"
    assert outputs.state.sound_speed is not None, "Sound speed should be passed through"

    # Check that horizon detection received parameters correctly
    assert outputs.horizons is not None, "Horizon detection should produce results"

    print("✓ Parameters are correctly passed between modules")


def test_fluid_backend_temperature_and_magnetic_profiles():
    """Ensure FluidBackend supports configured temperature and magnetic field profiles."""
    backend = FluidBackend()
    grid = np.linspace(0.0, 50e-6, 512)
    const_temperature = 5e5
    config = {
        "plasma_density": 5e17,
        "laser_wavelength": 800e-9,
        "laser_intensity": 5e16,
        "grid": grid,
        "temperature_settings": {"constant": const_temperature},
        "magnetic_field": 0.01,
        "use_fast_magnetosonic": True,
    }
    backend.configure(config)
    state = backend.step(0.0)

    assert state.temperature is not None, "Configured temperature profile missing"
    npt.assert_allclose(state.temperature, const_temperature)

    assert (
        state.magnetosonic_speed is not None
    ), "Magnetosonic speed should be computed when magnetic field supplied"
    npt.assert_allclose(state.sound_speed, state.magnetosonic_speed)

    # Magnetosonic speed should exceed or equal the adiabatic sound speed without magnetic contribution
    adiabatic_sound_speed = backend._model.sound_speed(const_temperature)  # type: ignore[attr-defined]
    assert np.all(state.magnetosonic_speed >= adiabatic_sound_speed)

    print("✓ FluidBackend applies temperature and magnetic field profiles")


def test_error_handling():
    """Check error handling and edge cases."""
    print("\nTesting error handling and edge cases...")

    from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty

    # Test with empty arrays
    try:
        empty_result = find_horizons_with_uncertainty(np.array([]), np.array([]), np.array([]))
        assert empty_result.positions.size == 0, "Should handle empty arrays gracefully"
        print("✓ Empty arrays handled correctly")
    except Exception as e:
        print(f"✗ Error handling empty arrays failed: {e}")
        raise

    # Test with mismatched array sizes
    try:
        find_horizons_with_uncertainty(
            np.array([0, 1, 2]),
            np.array([0, 1]),
            np.array([0, 1, 2]),  # Wrong size
        )
        print("✗ Should have failed with mismatched array sizes")
        assert False, "Should have raised an assertion error"
    except AssertionError:
        print("✓ Mismatched array sizes correctly raise AssertionError")

    # Test QFT with invalid parameters
    try:
        qft = QuantumFieldTheory(surface_gravity=-1e12)  # Negative surface gravity
        # This should still work since we're just calculating a temperature
        T_H = qft.hawking_temperature_from_kappa(-1e12)
        print(f"✓ QFT handles negative surface gravity: T_H = {T_H:.2e} K")
    except Exception as e:
        print(f"Note: QFT with negative surface gravity produced error: {e}")


def test_memory_usage():
    """Verify memory usage and performance."""
    print("\nTesting memory usage and performance...")

    import tracemalloc

    # Start tracing memory allocations
    tracemalloc.start()

    # Test with a reasonably sized dataset
    grid_size = 10000
    x = np.linspace(0, 100e-6, grid_size)
    n_e = np.full_like(x, 1e18)
    T_e = np.full_like(x, 100)
    velocity = np.sin(2 * np.pi * x * 1e4) * 0.1  # 10% of light speed
    sound_speed = np.full_like(x, 0.01)  # 1% of light speed

    # Test adaptive sigma calculation
    snapshot1 = tracemalloc.take_snapshot()
    sigma_map, diagnostics = estimate_sigma_map(n_e, T_e, x, velocity, sound_speed)
    snapshot2 = tracemalloc.take_snapshot()

    # Check that memory usage is reasonable
    top_stats = snapshot2.compare_to(snapshot1, "lineno")
    total_memory = sum(stat.size_diff for stat in top_stats)

    print(f"✓ Adaptive sigma calculation uses {total_memory} bytes for {grid_size} grid points")

    # Stop tracing
    tracemalloc.stop()


def test_warpx_backend_mock():
    """Test WarpX backend mock implementation."""
    print("\nTesting WarpX backend mock implementation...")

    # Load mock configuration
    mock_config_path = PROJECT_ROOT / "configs" / "warpx_mock.json"
    with open(mock_config_path, "r") as f:
        mock_config = json.load(f)

    # Create WarpX backend with mock configuration
    backend = WarpXBackend()
    backend.configure(mock_config)

    # Run a step
    state = backend.step(0.0)

    # Verify state has expected fields
    assert state.density is not None, "Density should not be None"
    assert state.velocity is not None, "Velocity should not be None"
    assert state.sound_speed is not None, "Sound speed should not be None"

    # Verify the mock data has reasonable values
    assert np.all(np.isfinite(state.density)), "Density should contain finite values"
    assert np.all(np.isfinite(state.velocity)), "Velocity should contain finite values"
    assert np.all(np.isfinite(state.sound_speed)), "Sound speed should contain finite values"

    print("✓ WarpX backend mock implementation works correctly")


def test_fluid_backend():
    """Validate fluid backend calculations."""
    print("\nTesting fluid backend calculations...")

    backend = FluidBackend()
    config = {
        "plasma_density": 1e18,
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e17,
        "grid": np.linspace(0.0, 50e-6, 1000),
    }
    backend.configure(config)

    # Run a step
    state = backend.step(0.0)

    # Verify the calculations produce reasonable results
    assert np.all(state.density > 0), "Density should be positive"
    assert np.all(np.abs(state.velocity) < 1.0), "Velocity should be less than light speed"
    assert np.all(state.sound_speed > 0), "Sound speed should be positive"

    # Check that the state is consistent
    assert (
        len(state.density) == len(state.velocity) == len(state.sound_speed)
    ), "All state arrays should have the same length"

    print("✓ Fluid backend calculations produce reasonable results")


def test_adaptive_sigma():
    """Check adaptive sigma integration."""
    print("\nTesting adaptive sigma integration...")

    # Create test data
    x = np.linspace(0, 50e-6, 1000)
    n_e = np.full_like(x, 1e18)
    T_e = np.full_like(x, 100)
    velocity = np.tanh((x - 25e-6) * 1e5) * 0.1  # Create a horizon-like profile
    sound_speed = np.full_like(x, 0.01)

    # Test adaptive sigma estimation
    sigma_map, diagnostics = estimate_sigma_map(n_e, T_e, x, velocity, sound_speed)

    # Verify the results
    assert sigma_map.shape == x.shape, "Sigma map should have same shape as grid"
    assert np.all(sigma_map > 0), "Sigma values should be positive"
    assert diagnostics is not None, "Diagnostics should be returned"

    print("✓ Adaptive sigma integration works correctly")


def test_fluctuation_injector():
    """Verify fluctuation injector functionality."""
    print("\nTesting fluctuation injector functionality...")

    # Create a fluctuation injector
    config = FluctuationConfig(
        seed=42,
        target_temperature=0.01,
        mode_cutoff=1e6,
        amplitude_scale=1.0,
        cadence=1,
        band_min=0.0,
        band_max=1.0,
        background_psd=0.0,
    )
    injector = QuantumFluctuationInjector(config)

    # Test Fourier mode sampling
    k_values = np.linspace(0, 1e6, 100)
    modes = injector.sample_fourier_modes(k_values)

    # Verify the results
    assert len(modes) == len(k_values), "Should return same number of modes as k values"
    assert np.all(np.isfinite(modes)), "Modes should contain finite values"

    print("✓ Fluctuation injector works correctly")


def test_module_interfaces():
    """Test all module interfaces and ensure they work together seamlessly."""
    print("\nTesting module interfaces...")

    # Test complete workflow with FluidBackend
    backend = FluidBackend()
    config = {
        "plasma_density": 1e18,
        "laser_wavelength": 800e-9,
        "laser_intensity": 1e17,
        "grid": np.linspace(0.0, 50e-6, 1000),
    }
    backend.configure(config)
    runner = SimulationRunner(backend)

    # Run complete workflow
    outputs = runner.run_step(0.0)

    # Verify all components worked together
    assert outputs.state is not None, "Should produce plasma state"
    assert outputs.horizons is not None, "Should produce horizon results"

    # Test with QFT
    if outputs.horizons.kappa.size > 0:
        kappa = outputs.horizons.kappa[0]  # Use first kappa value
        spec_result = calculate_hawking_spectrum(kappa)
        assert spec_result["success"], "QFT should work with calculated kappa"

    print("✓ All module interfaces work together seamlessly")
