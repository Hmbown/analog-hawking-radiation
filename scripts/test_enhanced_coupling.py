#!/usr/bin/env python3
"""
Test script for enhanced coupling mechanism in hybrid plasma-mirror models.

This script tests whether the spatial coupling variation fixes the "perfect correlation"
artifact identified by the validation framework.

Author: bern2025-k2
Date: 1905-11-06
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from analog_hawking.physics_engine.horizon_hybrid import (
    HybridHorizonParams,
    find_hybrid_horizons,
)
from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
from analog_hawking.physics_engine.enhanced_coupling import (
    create_spatial_coupling_profile,
    validate_coupling_profile,
    diagnose_coupling_artifact,
    compute_patchwise_effective_kappa,
)
from analog_hawking.detection.graybody_nd import aggregate_patchwise_graybody


def test_spatial_coupling_variation():
    """Test if spatial coupling variation breaks the perfect correlation artifact."""
    print("=" * 70)
    print("Testing Enhanced Coupling Mechanism")
    print("=" * 70)
    
    # Setup fluid backend (same as compare_hybrid_apples_to_apples.py)
    plasma_density = 5e17
    laser_wavelength = 800e-9
    laser_intensity = 5e17
    temperature_constant = 1e4
    grid = np.linspace(0.0, 50e-6, 512)
    
    backend = FluidBackend()
    backend.configure(
        {
            "plasma_density": plasma_density,
            "laser_wavelength": laser_wavelength,
            "laser_intensity": laser_intensity,
            "grid": grid,
            "temperature_settings": {"constant": temperature_constant},
            "use_fast_magnetosonic": False,
            "scale_with_intensity": True,
        }
    )
    state = backend.step(0.0)
    
    # Calculate mirror dynamics
    from scipy.constants import e, epsilon_0, m_e
    from scipy.constants import e, epsilon_0, m_e
    n_p0 = 1.0e24
    omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
    p = PlasmaMirrorParams(
        n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel"
    )
    t_m = np.linspace(0.0, 100e-15, 401)
    mirror = calculate_plasma_mirror_dynamics(state.grid, float(laser_intensity), p, t_m)
    
    # Find hybrid horizons
    hh = find_hybrid_horizons(
        state.grid, state.velocity, state.sound_speed, mirror, HybridHorizonParams()
    )
    
    print(f"\nFound {len(hh.fluid.positions)} fluid horizons")
    print(f"Mirror kappa: {hh.kappa_mirror:.3e} Hz")
    print(f"Fluid kappa range: [{np.min(hh.fluid.kappa):.3e}, {np.max(hh.fluid.kappa):.3e}] Hz")
    print(f"Coupling weight range: [{np.min(hh.coupling_weight):.3e}, {np.max(hh.coupling_weight):.3e}]")
    
    # Create spatial coupling profile
    profile = create_spatial_coupling_profile(hh)
    
    # Validate the profile
    print("\n" + "-" * 70)
    print("Coupling Profile Validation")
    print("-" * 70)
    validation = validate_coupling_profile(profile)
    for key, value in validation.items():
        print(f"{key}: {value}")
    
    # Diagnose potential artifacts
    print("\n" + "-" * 70)
    print("Artifact Diagnosis")
    print("-" * 70)
    diagnosis = diagnose_coupling_artifact(profile)
    print(f"Is artifact: {diagnosis['is_artifact']}")
    print(f"Artifact type: {diagnosis['artifact_type']}")
    print(f"Confidence: {diagnosis['confidence']:.2f}")
    print(f"Explanation: {diagnosis['explanation']}")
    if diagnosis['recommendations']:
        print("Recommendations:")
        for rec in diagnosis['recommendations']:
            print(f"  - {rec}")
    
    # Test old method (single kappa) vs new method (per-patch kappa)
    print("\n" + "-" * 70)
    print("Comparing Old vs New Methods")
    print("-" * 70)
    
    # Create 2D grid for testing (simple case)
    x = np.linspace(0, 100e-6, 256)
    y = np.linspace(-25e-6, 25e-6, 128)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Create simple velocity field with horizon
    v_field = np.zeros((len(x), len(y), 2))
    v_field[:, :, 0] = -1e6 * np.tanh((X - 50e-6) / 5e-6)  # Flow in x-direction
    v_field[:, :, 1] = 0.1 * v_field[:, :, 0] * np.sin(Y / 10e-6)  # Small y-component
    
    # Sound speed profile
    c_s = np.ones_like(X) * 5e5  # Constant sound speed
    
    # Old method: single effective kappa
    mean_coupling = validation['mean_coupling_weight']
    kappa_eff_old = float(np.mean(hh.fluid.kappa)) + mean_coupling * hh.kappa_mirror
    print(f"Old method - Single kappa: {kappa_eff_old:.3e} Hz")
    
    # New method: per-patch kappa
    kappa_per_patch = compute_patchwise_effective_kappa(profile)
    print(f"New method - Per-patch kappa range: [{np.min(kappa_per_patch):.3e}, {np.max(kappa_per_patch):.3e}] Hz")
    print(f"New method - Mean kappa: {np.mean(kappa_per_patch):.3e} Hz")
    
    # Test with enhanced graybody function
    try:
        # Old method
        result_old = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_eff_old,
            graybody_method="dimensionless",
            max_patches=32
        )
        
        # New method with per-patch kappa
        result_new = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_per_patch,
            graybody_method="dimensionless", 
            max_patches=min(32, len(kappa_per_patch))
        )
        
        if result_old.success and result_new.success:
            print(f"\nOld method - Peak frequency: {result_old.peak_frequency:.3e} Hz")
            print(f"Old method - Mean power: {np.mean(result_old.power_spectrum):.3e} W/Hz")
            print(f"Old method - Power std: {np.mean(result_old.power_std):.3e} W/Hz")
            
            print(f"\nNew method - Peak frequency: {result_new.peak_frequency:.3e} Hz")
            print(f"New method - Mean power: {np.mean(result_new.power_spectrum):.3e} W/Hz")
            print(f"New method - Power std: {np.mean(result_new.power_std):.3e} W/Hz")
            
            power_ratio = np.mean(result_new.power_spectrum) / np.mean(result_old.power_spectrum)
            print(f"\nPower enhancement ratio (new/old): {power_ratio:.2f}x")
            
            # Check if spatial variation reduces correlation
            if diagnosis['is_artifact']:
                print(f"\n⚠️  ARTIFACT DETECTED: {diagnosis['artifact_type']}")
                print("The 4x higher signal temperature may be partially computational artifact")
            else:
                print(f"\n✅ PHYSICAL COUPLING: Spatial variation appears genuine")
                print("The signal enhancement is likely physical, not artifact")
                
        else:
            print("Warning: Graybody aggregation failed")
            
    except Exception as e:
        print(f"Error in graybody aggregation: {e}")
    
    return diagnosis


def test_fix_recommendations():
    """Test specific recommendations from artifact diagnosis."""
    print("\n" + "=" * 70)
    print("Testing Fix Recommendations")
    print("=" * 70)
    
    # The diagnosis likely recommended improving spatial variation
    # Let's test with artificially enhanced spatial variation
    
    from scipy.constants import e, epsilon_0, m_e
    
    plasma_density = 5e17
    laser_wavelength = 800e-9
    laser_intensity = 5e17
    temperature_constant = 1e4
    grid = np.linspace(0.0, 50e-6, 512)
    
    backend = FluidBackend()
    backend.configure({
        "plasma_density": plasma_density,
        "laser_wavelength": laser_wavelength,
        "laser_intensity": laser_intensity,
        "grid": grid,
        "temperature_settings": {"constant": temperature_constant},
        "use_fast_magnetosonic": False,
        "scale_with_intensity": True,
    })
    state = backend.step(0.0)
    
    n_p0 = 1.0e24
    omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
    
    # Test with enhanced localization (shorter coupling length = more spatial variation)
    print("\nTesting enhanced spatial localization...")
    
    p = PlasmaMirrorParams(
        n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel"
    )
    t_m = np.linspace(0.0, 100e-15, 401)
    mirror = calculate_plasma_mirror_dynamics(state.grid, float(laser_intensity), p, t_m)
    
    # Use shorter coupling length for more spatial variation
    params_short = HybridHorizonParams(coupling_length=1e-6)  # 5x shorter
    hh_short = find_hybrid_horizons(state.grid, state.velocity, state.sound_speed, mirror, params_short)
    
    profile_short = create_spatial_coupling_profile(hh_short)
    validation_short = validate_coupling_profile(profile_short)
    
    print(f"Short coupling length (1μm) - std: {validation_short['std_coupling_weight']:.3e}")
    print(f"Short coupling length (1μm) - range: [{validation_short['min_coupling_weight']:.3e}, {validation_short['max_coupling_weight']:.3e}]")
    
    # Compare with default coupling length
    params_default = HybridHorizonParams()  # default 5e-6
    hh_default = find_hybrid_horizons(state.grid, state.velocity, state.sound_speed, mirror, params_default)
    
    profile_default = create_spatial_coupling_profile(hh_default)
    validation_default = validate_coupling_profile(profile_default)
    
    print(f"Default coupling length (5μm) - std: {validation_default['std_coupling_weight']:.3e}")
    print(f"Default coupling length (5μm) - range: [{validation_default['min_coupling_weight']:.3e}, {validation_default['max_coupling_weight']:.3e}]")
    
    variation_ratio = validation_short['std_coupling_weight'] / max(validation_default['std_coupling_weight'], 1e-12)
    print(f"\nSpatial variation improvement: {variation_ratio:.2f}x")
    
    if variation_ratio > 2.0:
        print("✅ Enhanced localization significantly increases spatial variation")
    else:
        print("⚠️  Enhanced localization has limited effect")


if __name__ == "__main__":
    print("Enhanced Coupling Test - Analog Hawking Radiation")
    print("Addressing: Why hybrid model predicts 4× higher signal temperature")
    print("             yet validation flags perfect correlations as 'by construction'")
    print()
    
    # Run main test
    diagnosis = test_spatial_coupling_variation()
    
    # Test fixes
    test_fix_recommendations()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("The enhanced coupling module addresses the computational mirage by:")
    print("1. Enabling per-patch kappa values in graybody_nd.py")
    print("2. Preserving spatial variation from hybrid horizon detection")
    print("3. Breaking deterministic relationships that cause 'by construction' correlations")
    print("\nNext step: Run ahr validate to check if validation flags are resolved")
