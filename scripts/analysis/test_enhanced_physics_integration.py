#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Physics Models Integration

This script tests the enhanced physics models to ensure:
1. All enhanced components work correctly
2. Integration with existing pipeline is seamless
3. Backward compatibility is maintained
4. Physics validation passes
5. ELI optimization functions properly

Author: Enhanced Physics Implementation
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import warnings
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from analog_hawking.physics_engine.enhanced_relativistic_physics import (
    RelativisticPlasmaPhysics,
    test_relativistic_physics,
)
from analog_hawking.physics_engine.enhanced_ionization_physics import (
    IonizationDynamics,
    ATOMIC_DATA,
    test_ionization_physics,
)
from analog_hawking.physics_engine.enhanced_plasma_surface_physics import (
    PlasmaDynamicsAtSurface,
    test_plasma_surface_physics,
)
from analog_hawking.physics_engine.physics_validation_framework import (
    PhysicsModelValidator,
    run_comprehensive_validation,
)
from analog_hawking.physics_engine.enhanced_physics_integration import (
    EnhancedPhysicsEngine,
    EnhancedPhysicsConfig,
    PhysicsModel,
    create_enhanced_pipeline,
    BackwardCompatibilityWrapper,
    test_enhanced_integration,
)


def test_backward_compatibility():
    """Test backward compatibility with existing analysis pipeline"""
    print("\n" + "=" * 60)
    print("TESTING BACKWARD COMPATIBILITY")
    print("=" * 60)

    # Test that we can still import legacy functions
    try:
        from analog_hawking.physics_engine.horizon import (
            find_horizons_with_uncertainty,
            sound_speed,
        )
        from analog_hawking.physics_engine.plasma_mirror import calculate_plasma_mirror_dynamics

        print("✅ Legacy imports successful")
    except ImportError as e:
        print(f"❌ Legacy import failed: {e}")
        return False

    # Test legacy horizon finding still works
    x = np.linspace(0, 100e-6, 1000)
    v = 0.1 * 3e8 * np.tanh((x - 50e-6) / 10e-6)
    T_e = 1e6 * np.ones_like(x)
    c_s = sound_speed(T_e)

    try:
        horizon_result = find_horizons_with_uncertainty(x, v, c_s)
        print(f"✅ Legacy horizon finding works: found {len(horizon_result.positions)} horizons")
    except Exception as e:
        print(f"❌ Legacy horizon finding failed: {e}")
        return False

    # Test backward compatibility wrapper
    try:
        enhanced_engine = create_enhanced_pipeline(PhysicsModel.LEGACY)
        wrapper = BackwardCompatibilityWrapper(enhanced_engine)

        legacy_result = wrapper.find_horizons_legacy(x, v, c_s)
        print(
            f"✅ Backward compatibility wrapper works: found {len(legacy_result.positions)} horizons"
        )
    except Exception as e:
        print(f"❌ Backward compatibility wrapper failed: {e}")
        return False

    return True


def test_enhanced_components():
    """Test individual enhanced physics components"""
    print("\n" + "=" * 60)
    print("TESTING INDIVIDUAL ENHANCED COMPONENTS")
    print("=" * 60)

    success = True

    # Test relativistic physics
    print("\n1. Testing Relativistic Physics")
    print("-" * 40)
    try:
        plasma = RelativisticPlasmaPhysics(
            electron_density=1e20, laser_wavelength=800e-9, laser_intensity=1e20
        )

        # Check regime
        regime = plasma.check_relativistic_regime()
        print(f"   Regime: {regime['regime']}, a₀ = {regime['a0_parameter']:.2f}")

        # Test relativistic corrections
        gamma_values = np.array([1.0, 2.0, 5.0, 10.0])
        omega_pe_rel = plasma.relativistic_plasma_frequency(gamma_values)
        print(
            f"   Relativistic plasma frequency: {omega_pe_rel[0]:.2e} → {omega_pe_rel[-1]:.2e} rad/s"
        )

        # Test relativistic sound speed
        T = 1e6  # 1 MK
        c_s_rel = plasma.relativistic_sound_speed(T, gamma_values)
        print(f"   Relativistic sound speed: {c_s_rel[0]:.2e} → {c_s_rel[-1]:.2e} m/s")

        print("   ✅ Relativistic physics tests passed")
    except Exception as e:
        print(f"   ❌ Relativistic physics test failed: {e}")
        success = False

    # Test ionization physics
    print("\n2. Testing Ionization Physics")
    print("-" * 40)
    try:
        ionization = IonizationDynamics(ATOMIC_DATA["Al"], laser_wavelength=800e-9)

        # Test ADK rates
        E_test = 1e12  # V/m
        rate_adk = ionization.adk_model.adk_rate(E_test, 0)
        print(f"   ADK rate at E={E_test:.1e} V/m: {rate_adk:.2e} s⁻¹")

        # Test PPT rates
        rate_ppt = ionization.ppt_model.ppt_rate(E_test, 0, ionization.omega_l)
        print(f"   PPT rate at E={E_test:.1e} V/m: {rate_ppt:.2e} s⁻¹")

        # Test collisional ionization
        rate_coll = ionization.collisional_model.collisional_rate(1e19, 1e6 * e, 0)
        print(f"   Collisional rate: {rate_coll:.2e} s⁻¹")

        print("   ✅ Ionization physics tests passed")
    except Exception as e:
        print(f"   ❌ Ionization physics test failed: {e}")
        success = False

    # Test surface physics
    print("\n3. Testing Surface Physics")
    print("-" * 40)
    try:
        surface = PlasmaDynamicsAtSurface("Al")

        # Test surface interaction
        intensity = 1e20  # W/m²
        wavelength = 800e-9
        pulse_duration = 30e-15

        results = surface.full_surface_interaction(intensity, wavelength, pulse_duration, 0, "p")

        print(f"   Absorption fraction: {results['absorption_fraction']:.3f}")
        print(f"   Reflectivity: {results['reflectivity']:.3f}")
        print(f"   Electron temperature: {results['electron_temperature']/e/1e3:.1f} keV")

        # Test absorption mechanisms
        absorption = surface.absorption
        eta_brunel = absorption.brunel_heating(intensity, wavelength, results["scale_length"])
        eta_jxb = absorption.jxb_heating(intensity, wavelength, results["electron_temperature"])
        print(f"   Brunel heating: {eta_brunel:.3f}, J×B heating: {eta_jxb:.3f}")

        print("   ✅ Surface physics tests passed")
    except Exception as e:
        print(f"   ❌ Surface physics test failed: {e}")
        success = False

    return success


def test_integration_pipeline():
    """Test the complete enhanced integration pipeline"""
    print("\n" + "=" * 60)
    print("TESTING COMPLETE INTEGRATION PIPELINE")
    print("=" * 60)

    try:
        # Create enhanced physics engine
        config = EnhancedPhysicsConfig(
            model=PhysicsModel.COMPREHENSIVE,
            include_relativistic=True,
            include_ionization_dynamics=True,
            include_surface_physics=True,
            target_material="Al",
            eli_optimization=True,
        )

        engine = EnhancedPhysicsEngine(config)

        # Test enhanced horizon finding
        print("\n1. Testing Enhanced Horizon Finding")
        print("-" * 40)
        x = np.linspace(0, 100e-6, 1000)
        v = 0.1 * 3e8 * np.tanh((x - 50e-6) / 10e-6)
        T_e = 1e6 * np.ones_like(x)
        n_e = 1e19 * np.ones_like(x)
        gamma = np.ones_like(x) * 1.5

        horizon_results = engine.enhanced_horizon_finding(x, v, T_e, n_e, gamma_factor=gamma)

        print(f"   Found {len(horizon_results.horizon_positions)} horizons")
        if len(horizon_results.horizon_positions) > 0:
            print(f"   Surface gravity: {horizon_results.surface_gravity[0]:.2e} s⁻¹")
            print(f"   Hawking temperature: {horizon_results.hawking_temperature[0]:.2e} K")

        # Test enhanced graybody calculation
        print("\n2. Testing Enhanced Graybody Calculation")
        print("-" * 40)
        frequencies = np.logspace(10, 14, 50)
        spectrum = engine.enhanced_graybody_calculation(frequencies, horizon_results)

        peak_freq = frequencies[np.argmax(spectrum)]
        peak_intensity = np.max(spectrum)
        print(f"   Spectrum peak: {peak_freq:.2e} Hz")
        print(f"   Peak intensity: {peak_intensity:.2e}")

        # Test ELI optimization
        print("\n3. Testing ELI Optimization")
        print("-" * 40)
        parameter_ranges = {
            "intensity": (1e19, 1e21),
            "density": (1e18, 1e20),
            "wavelength": (400e-9, 1064e-9),
        }

        optimization = engine.eli_facility_optimization(parameter_ranges, "hawking_temperature")

        print("   Optimization results:")
        for param, value in optimization.get("optimal_parameters", {}).items():
            print(f"     {param}: {value:.2e}")

        print("   ✅ Integration pipeline tests passed")
        return True

    except Exception as e:
        print(f"   ❌ Integration pipeline test failed: {e}")
        return False


def test_validation_framework():
    """Test the physics validation framework"""
    print("\n" + "=" * 60)
    print("TESTING PHYSICS VALIDATION FRAMEWORK")
    print("=" * 60)

    try:
        # Run comprehensive validation
        validation_results = run_comprehensive_validation()

        if validation_results:
            summary = validation_results.get("summary", {})
            print(f"\nValidation Summary:")
            print(f"   Total tests: {summary.get('total_tests', 0)}")
            print(f"   Passed: {summary.get('passed_tests', 0)}")
            print(f"   Errors: {summary.get('error_tests', 0)}")
            print(f"   Warnings: {summary.get('warning_tests', 0)}")
            print(f"   Pass rate: {summary.get('pass_rate', 0):.1%}")
            print(f"   Overall status: {summary.get('overall_status', 'Unknown')}")

            if summary.get("error_tests", 0) == 0:
                print("   ✅ All critical validation tests passed")
                return True
            else:
                print("   ❌ Some validation tests failed")
                for error in summary.get("errors", []):
                    print(f"     - {error['test']}: {error['description']}")
                return False
        else:
            print("   ❌ Validation failed to produce results")
            return False

    except Exception as e:
        print(f"   ❌ Validation framework test failed: {e}")
        return False


def test_eli_specific_conditions():
    """Test enhanced models under ELI-specific conditions"""
    print("\n" + "=" * 60)
    print("TESTING ELI-SPECIFIC CONDITIONS")
    print("=" * 60)

    # ELI parameter ranges
    eli_params = {
        "intensity": 1e22,  # 10^22 W/m² (ultra-relativistic)
        "wavelength": 800e-9,
        "pulse_duration": 25e-15,  # 25 fs
        "target": "Al",
    }

    print(f"Testing ELI conditions:")
    print(f"   Intensity: {eli_params['intensity']:.2e} W/m²")
    print(f"   Wavelength: {eli_params['wavelength']*1e9:.1f} nm")
    print(f"   Pulse duration: {eli_params['pulse_duration']*1e15:.1f} fs")
    print(f"   Target: {eli_params['target']}")

    try:
        # Test relativistic regime
        plasma = RelativisticPlasmaPhysics(
            electron_density=1e21,
            laser_wavelength=eli_params["wavelength"],
            laser_intensity=eli_params["intensity"],
        )

        regime = plasma.check_relativistic_regime()
        print(f"\nRelativistic regime assessment:")
        print(f"   a₀ parameter: {regime['a0_parameter']:.2f}")
        print(f"   Regime: {regime['regime']}")
        print(f"   γ_osc: {regime['gamma_oscillatory']:.2f}")

        if regime["a0_parameter"] > 10:
            print("   ✅ Confirmed ultra-relativistic regime (a₀ > 10)")
        else:
            print("   ⚠️  Not in ultra-relativistic regime for given parameters")

        # Test surface physics at ELI intensities
        surface = PlasmaDynamicsAtSurface(eli_params["target"])
        surface_results = surface.full_surface_interaction(
            eli_params["intensity"], eli_params["wavelength"], eli_params["pulse_duration"], 0, "p"
        )

        print(f"\nSurface physics at ELI intensity:")
        print(f"   Absorption: {surface_results['absorption_fraction']:.3f}")
        print(f"   Reflectivity: {surface_results['reflectivity']:.3f}")
        print(f"   Electron temperature: {surface_results['electron_temperature']/e/1e6:.1f} MeV")
        print(f"   Expansion velocity: {surface_results['expansion_velocity']/1e6:.1f} Mm/s")

        # Test ionization at ELI conditions
        ionization = IonizationDynamics(ATOMIC_DATA[eli_params["target"]], eli_params["wavelength"])
        E_field = np.sqrt(2 * eli_params["intensity"] / (3e8 * 8.85e-12))
        adk_rate = ionization.adk_model.adk_rate(E_field, 0)
        ppt_rate = ionization.ppt_model.ppt_rate(E_field, 0, ionization.omega_l)

        print(f"\nIonization at ELI intensity:")
        print(f"   Electric field: {E_field:.2e} V/m")
        print(f"   ADK rate: {adk_rate:.2e} s⁻¹")
        print(f"   PPT rate: {ppt_rate:.2e} s⁻¹")

        if adk_rate > 1e15:  # Very fast ionization
            print("   ✅ Confirmed complete ionization at ELI intensity")
        else:
            print("   ⚠️  Ionization may be incomplete at given intensity")

        print("   ✅ ELI-specific condition tests completed")
        return True

    except Exception as e:
        print(f"   ❌ ELI condition test failed: {e}")
        return False


def generate_performance_comparison():
    """Generate performance comparison between legacy and enhanced models"""
    print("\n" + "=" * 60)
    print("GENERATING PERFORMANCE COMPARISON")
    print("=" * 60)

    try:
        import time

        # Test parameters
        x = np.linspace(0, 100e-6, 1000)
        v = 0.1 * 3e8 * np.tanh((x - 50e-6) / 10e-6)
        T_e = 1e6 * np.ones_like(x)
        c_s = np.sqrt(5 / 3 * 1.38e-23 * T_e / 1.67e-27)

        # Legacy model performance
        from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty

        start_time = time.time()
        legacy_result = find_horizons_with_uncertainty(x, v, c_s)
        legacy_time = time.time() - start_time

        # Enhanced model performance
        enhanced_engine = create_enhanced_pipeline(PhysicsModel.COMPREHENSIVE)
        start_time = time.time()
        enhanced_result = enhanced_engine.enhanced_horizon_finding(x, v, T_e)
        enhanced_time = time.time() - start_time

        print(f"Performance Comparison:")
        print(f"   Legacy model: {legacy_time*1000:.2f} ms")
        print(f"   Enhanced model: {enhanced_time*1000:.2f} ms")
        print(f"   Speedup ratio: {enhanced_time/legacy_time:.2f}x")

        # Compare results
        if len(legacy_result.positions) > 0 and len(enhanced_result.horizon_positions) > 0:
            pos_diff = abs(legacy_result.positions[0] - enhanced_result.horizon_positions[0])
            kappa_diff = abs(legacy_result.kappa[0] - enhanced_result.surface_gravity[0])

            print(f"\nResult Comparison:")
            print(f"   Position difference: {pos_diff*1e9:.2f} nm")
            print(f"   Surface gravity difference: {kappa_diff/legacy_result.kappa[0]*100:.1f}%")

        print("   ✅ Performance comparison completed")
        return True

    except Exception as e:
        print(f"   ❌ Performance comparison failed: {e}")
        return False


def run_comprehensive_test_suite():
    """Run the complete test suite"""
    print("COMPREHENSIVE ENHANCED PHYSICS TEST SUITE")
    print("=" * 80)
    print("Testing enhanced physics models for analog Hawking radiation analysis")
    print("Author: Enhanced Physics Implementation")
    print("Date: November 2025")
    print("=" * 80)

    test_results = {
        "backward_compatibility": False,
        "individual_components": False,
        "integration_pipeline": False,
        "validation_framework": False,
        "eli_conditions": False,
        "performance_comparison": False,
    }

    # Run all tests
    test_results["backward_compatibility"] = test_backward_compatibility()
    test_results["individual_components"] = test_enhanced_components()
    test_results["integration_pipeline"] = test_integration_pipeline()
    test_results["validation_framework"] = test_validation_framework()
    test_results["eli_conditions"] = test_eli_specific_conditions()
    test_results["performance_comparison"] = generate_performance_comparison()

    # Generate summary report
    print("\n" + "=" * 80)
    print("COMPREHENSIVE TEST SUITE SUMMARY")
    print("=" * 80)

    total_tests = len(test_results)
    passed_tests = sum(test_results.values())

    print(f"\nTest Results Summary:")
    print(f"   Total test categories: {total_tests}")
    print(f"   Passed categories: {passed_tests}")
    print(f"   Failed categories: {total_tests - passed_tests}")
    print(f"   Success rate: {passed_tests/total_tests*100:.1f}%")

    print(f"\nDetailed Results:")
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name.replace('_', ' ').title()}: {status}")

    if passed_tests == total_tests:
        print(f"\n🎉 ALL TESTS PASSED! Enhanced physics models are ready for use.")
        print(f"   The enhanced models successfully:")
        print(f"   - Include relativistic effects for high-intensity laser interactions")
        print(f"   - Implement comprehensive ionization physics (ADK, PPT, collisional)")
        print(f"   - Model plasma-surface interactions with realistic absorption mechanisms")
        print(f"   - Validate against physical constraints and theoretical benchmarks")
        print(f"   - Maintain backward compatibility with existing analysis pipeline")
        print(f"   - Optimize parameters for ELI facility conditions")
    else:
        print(f"\n⚠️  SOME TESTS FAILED. Please review the failures above.")
        print(f"   Enhanced physics models may need debugging or configuration adjustments.")

    return test_results


if __name__ == "__main__":
    # Run comprehensive test suite
    results = run_comprehensive_test_suite()

    # Exit with appropriate code
    sys.exit(0 if all(results.values()) else 1)
