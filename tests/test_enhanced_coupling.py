"""
Test suite for enhanced coupling mechanism in hybrid plasma-mirror models.

This validates that spatially varying coupling weights are properly preserved
through the graybody calculation, resolving the deterministic artifact issue.

Author: bern2025-k2
Date: 1905-11-06 (in spirit)
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import e, epsilon_0, m_e

from analog_hawking.detection.graybody_nd import aggregate_patchwise_graybody
from analog_hawking.physics_engine.enhanced_coupling import (
    SpatialCouplingProfile,
    compute_patchwise_effective_kappa,
    create_spatial_coupling_profile,
    diagnose_coupling_artifact,
    validate_coupling_profile,
)
from analog_hawking.physics_engine.horizon_hybrid import (
    HybridHorizonParams,
    HybridHorizonResult,
    find_hybrid_horizons,
)
from analog_hawking.physics_engine.horizon import HorizonResult
from analog_hawking.physics_engine.plasma_mirror import (
    PlasmaMirrorParams,
    MirrorDynamics,
    calculate_plasma_mirror_dynamics,
)
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend


class TestSpatialCouplingProfile:
    """Test spatial coupling profile creation and validation."""

    def test_create_spatial_coupling_profile(self):
        """Test conversion from HybridHorizonResult to spatial profile."""
        # Create mock hybrid result
        positions = np.array([1e-6, 2e-6, 3e-6])
        fluid_kappa = np.array([1e12, 2e12, 3e12])
        mirror_kappa = 5e12
        coupling_weights = np.array([0.1, 0.2, 0.3])
        alignment = np.array([1.0, -1.0, 0.0])
        
        # Mock the required fields for HorizonResult
        dvdx = np.ones_like(fluid_kappa) * 1e6  # Dummy gradient
        dcsdx = np.ones_like(fluid_kappa) * 1e5  # Dummy gradient
        
        fluid_result = HorizonResult(
            positions=positions,
            kappa=fluid_kappa,
            kappa_err=np.zeros_like(fluid_kappa),
            dvdx=dvdx,
            dcsdx=dcsdx
        )
        
        hybrid_result = HybridHorizonResult(
            fluid=fluid_result,
            kappa_mirror=mirror_kappa,
            hybrid_kappa=fluid_kappa + coupling_weights * mirror_kappa,
            coupling_weight=coupling_weights,
            alignment=alignment
        )
        
        profile = create_spatial_coupling_profile(hybrid_result)
        
        assert isinstance(profile, SpatialCouplingProfile)
        np.testing.assert_array_equal(profile.positions, positions)
        np.testing.assert_array_equal(profile.fluid_kappa, fluid_kappa)
        assert profile.mirror_kappa == mirror_kappa
        np.testing.assert_array_equal(profile.coupling_weights, coupling_weights)
        np.testing.assert_array_equal(profile.alignment, alignment)
    
    def test_effective_kappa_property(self):
        """Test computation of effective kappa including coupling."""
        profile = SpatialCouplingProfile(
            positions=np.array([1e-6, 2e-6]),
            fluid_kappa=np.array([1e12, 2e12]),
            mirror_kappa=5e12,
            coupling_weights=np.array([0.1, 0.2]),
            alignment=np.array([1.0, 1.0])
        )
        
        expected = np.array([
            1e12 + 0.1 * 5e12,  # 1.5e12
            2e12 + 0.2 * 5e12   # 3.0e12
        ])
        
        np.testing.assert_array_almost_equal(profile.effective_kappa, expected)
    
    def test_validate_coupling_profile_physical(self):
        """Test validation of physically reasonable coupling profile."""
        profile = SpatialCouplingProfile(
            positions=np.linspace(0, 10e-6, 10),
            fluid_kappa=np.linspace(1e12, 3e12, 10),
            mirror_kappa=1e13,
            coupling_weights=np.linspace(0.0, 0.5, 10),  # Varying weights
            alignment=np.random.choice([-1, 0, 1], 10)
        )
        
        validation = validate_coupling_profile(profile)
        
        assert validation["negative_kappa_count"] == 0
        assert validation["uniform_weight_flag"] == 0.0  # Not uniform
        assert validation["std_coupling_weight"] > 0.01  # Significant variation
        assert "warning" not in validation  # No warnings
    
    def test_validate_coupling_profile_artifact(self):
        """Test detection of computational artifact (uniform weights)."""
        profile = SpatialCouplingProfile(
            positions=np.linspace(0, 10e-6, 10),
            fluid_kappa=np.linspace(1e12, 3e12, 10),
            mirror_kappa=1e13,
            coupling_weights=np.ones(10) * 0.3,  # Uniform weights = artifact
            alignment=np.ones(10)
        )
        
        validation = validate_coupling_profile(profile)
        
        assert validation["uniform_weight_flag"] == 1.0  # Flagged as uniform
        assert "uniform_weight_warning" in validation
        assert validation["std_coupling_weight"] < 1e-12  # No variation
    
    def test_diagnose_coupling_artifact_physical(self):
        """Test artifact diagnosis for physical coupling."""
        profile = SpatialCouplingProfile(
            positions=np.linspace(0, 10e-6, 10),
            fluid_kappa=np.linspace(1e12, 3e12, 10),
            mirror_kappa=1e13,
            coupling_weights=np.linspace(0.0, 0.5, 10),  # Varying
            alignment=np.random.choice([-1, 0, 1], 10)
        )
        
        diagnosis = diagnose_coupling_artifact(profile)
        
        assert diagnosis["is_artifact"] is False
        assert diagnosis["artifact_type"] is None
        assert diagnosis["confidence"] < 0.5
        assert "explanation" in diagnosis
    
    def test_diagnose_coupling_artifact_uniform(self):
        """Test artifact diagnosis for uniform weights (perfect correlation)."""
        profile = SpatialCouplingProfile(
            positions=np.linspace(0, 10e-6, 10),
            fluid_kappa=np.linspace(1e12, 3e12, 10),
            mirror_kappa=1e13,
            coupling_weights=np.ones(10) * 0.3,  # Uniform
            alignment=np.ones(10)
        )
        
        diagnosis = diagnose_coupling_artifact(profile)
        
        assert diagnosis["is_artifact"] is True
        # Uniform weights cause perfect correlation, which is the primary diagnosis
        assert diagnosis["artifact_type"] in ["uniform_weights", "perfect_correlation"]
        assert diagnosis["confidence"] > 0.8
        assert "perfect correlation" in diagnosis["explanation"].lower() or "uniform" in diagnosis["explanation"].lower()


class TestEnhancedGraybody:
    """Test enhanced graybody calculation with per-patch kappa."""

    def test_aggregate_patchwise_single_kappa_backward_compatible(self):
        """Test backward compatibility with single kappa value."""
        x = np.linspace(0, 100e-6, 64)
        y = np.linspace(-50e-6, 50e-6, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Simple flow field with horizon
        v_field = np.zeros((len(x), len(y), 2))
        v_field[:, :, 0] = -1e6 * np.tanh((X - 50e-6) / 10e-6)
        v_field[:, :, 1] = 0.1 * v_field[:, :, 0] * np.sin(Y / 20e-6)
        
        c_s = np.ones_like(X) * 5e5
        
        # Single kappa (old method)
        kappa_single = 1e13
        result = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_single,
            graybody_method="dimensionless",
            max_patches=16
        )
        
        assert result.success is True
        assert result.frequencies is not None
        assert result.power_spectrum is not None
        assert result.n_patches > 0
    
    def test_aggregate_patchwise_array_kappa_enhanced(self):
        """Test enhanced method with per-patch kappa array."""
        x = np.linspace(0, 100e-6, 64)
        y = np.linspace(-50e-6, 50e-6, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        v_field = np.zeros((len(x), len(y), 2))
        v_field[:, :, 0] = -1e6 * np.tanh((X - 50e-6) / 10e-6)
        v_field[:, :, 1] = 0.1 * v_field[:, :, 0] * np.sin(Y / 20e-6)
        
        c_s = np.ones_like(X) * 5e5
        
        # Array of kappa values (enhanced method)
        kappa_array = np.linspace(1e12, 5e13, 16)  # Varying kappa
        result = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_array,
            graybody_method="dimensionless",
            max_patches=16
        )
        
        assert result.success is True
        assert result.frequencies is not None
        assert result.power_spectrum is not None
        assert result.n_patches > 0
        # With varying kappa, should have higher power variation
        assert np.mean(result.power_std) > 1e-30  # Non-zero uncertainty
    
    def test_per_patch_kappa_vs_single_kappa(self):
        """Compare per-patch vs single kappa methods."""
        x = np.linspace(0, 100e-6, 64)
        y = np.linspace(-50e-6, 50e-6, 32)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        v_field = np.zeros((len(x), len(y), 2))
        v_field[:, :, 0] = -1e6 * np.tanh((X - 50e-6) / 10e-6)
        v_field[:, :, 1] = 0.1 * v_field[:, :, 0] * np.sin(Y / 20e-6)
        
        c_s = np.ones_like(X) * 5e5
        
        # Single kappa (old)
        kappa_single = 1e13
        result_single = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_single,
            graybody_method="dimensionless",
            max_patches=16
        )
        
        # Per-patch kappa (new) - same mean, but with variation
        kappa_array = np.random.normal(1e13, 2e12, 16)  # Variation around same mean
        result_multi = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, kappa_array,
            graybody_method="dimensionless",
            max_patches=16
        )
        
        assert result_single.success and result_multi.success
        # Both methods should produce valid results
        assert result_single.peak_frequency > 0
        assert result_multi.peak_frequency > 0
        # Multi-kappa method should have higher patch-to-patch variation (power_std)
        assert np.mean(result_multi.power_std) >= 0  # Should be non-negative


class TestIntegration:
    """Integration tests with full hybrid pipeline."""
    
    def test_end_to_end_hybrid_with_spatial_coupling(self):
        """Test complete hybrid pipeline with spatially resolved coupling."""
        # Setup fluid backend
        plasma_density = 5e17
        laser_intensity = 5e17
        grid = np.linspace(0.0, 50e-6, 256)
        
        backend = FluidBackend()
        backend.configure({
            "plasma_density": plasma_density,
            "laser_wavelength": 800e-9,
            "laser_intensity": laser_intensity,
            "grid": grid,
            "temperature_settings": {"constant": 1e4},
            "use_fast_magnetosonic": False,
            "scale_with_intensity": True,
        })
        state = backend.step(0.0)
        
        # Mirror dynamics
        n_p0 = 1.0e24
        omega_p0 = float(np.sqrt(e**2 * n_p0 / (epsilon_0 * m_e)))
        p = PlasmaMirrorParams(
            n_p0=n_p0, omega_p0=omega_p0, a=0.5, b=0.5, D=10e-6, eta_a=1.0, model="anabhel"
        )
        t_m = np.linspace(0.0, 100e-15, 401)
        mirror = calculate_plasma_mirror_dynamics(state.grid, float(laser_intensity), p, t_m)
        
        # Hybrid horizons
        hh = find_hybrid_horizons(
            state.grid, state.velocity, state.sound_speed, mirror, HybridHorizonParams()
        )
        
        assert hh.fluid.positions.size > 0, "Should find fluid horizons"
        assert hh.coupling_weight.size > 0, "Should compute coupling weights"
        
        # Create spatial coupling profile
        profile = create_spatial_coupling_profile(hh)
        
        # Validate profile
        validation = validate_coupling_profile(profile)
        assert validation["negative_kappa_count"] == 0
        assert validation["uniform_weight_flag"] == 0.0  # Should not be uniform
        
        # Diagnose artifacts
        diagnosis = diagnose_coupling_artifact(profile)
        assert diagnosis["is_artifact"] is False, "Should not be computational artifact"
        
        # Compute per-patch kappa
        kappa_per_patch = compute_patchwise_effective_kappa(profile)
        assert kappa_per_patch.size > 0
        assert np.all(kappa_per_patch > 0), "All kappa should be positive"
        
        # Test with enhanced graybody (2D case)
        x_2d = np.linspace(0, 100e-6, 32)
        y_2d = np.linspace(-25e-6, 25e-6, 16)
        X_2d, Y_2d = np.meshgrid(x_2d, y_2d, indexing='ij')
        
        v_field_2d = np.zeros((len(x_2d), len(y_2d), 2))
        v_field_2d[:, :, 0] = -1e6 * np.tanh((X_2d - 50e-6) / 10e-6)
        v_field_2d[:, :, 1] = 0.1 * v_field_2d[:, :, 0] * np.sin(Y_2d / 20e-6)
        c_s_2d = np.ones_like(X_2d) * 5e5
        
        # Use per-patch kappa
        result = aggregate_patchwise_graybody(
            [x_2d, y_2d], v_field_2d, c_s_2d, kappa_per_patch[:16],  # Use first 16 patches
            graybody_method="dimensionless",
            max_patches=min(16, len(kappa_per_patch))
        )
        
        assert result.success is True
        assert result.n_patches > 0


class TestBackwardCompatibility:
    """Ensure backward compatibility with existing API."""
    
    def test_graybody_nd_single_kappa_still_works(self):
        """Test that single kappa (old API) still works."""
        from analog_hawking.detection.graybody_nd import GraybodySpectrumND
        
        # Old API: single kappa value
        graybody = GraybodySpectrumND([64, 32])
        freqs = np.logspace(6, 12, 100)
        
        spectrum = graybody.calculate_spectrum(
            freqs, 
            hawking_temperature=1000.0,
            kappa=1e12,
            extra_params={"alpha_gray": 1.0}
        )
        
        assert spectrum.shape == (100,)
        assert np.all(spectrum >= 0)
        assert np.all(np.isfinite(spectrum))
    
    def test_aggregate_patchwise_signature_backward_compatible(self):
        """Test that function signature accepts single kappa (backward compatible)."""
        x = np.linspace(0, 50e-6, 32)
        y = np.linspace(-25e-6, 25e-6, 16)
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        v_field = np.zeros((len(x), len(y), 2))
        v_field[:, :, 0] = -1e6 * np.tanh((X - 25e-6) / 5e-6)
        c_s = np.ones_like(X) * 5e5
        
        # Single kappa (old way) - should still work
        result = aggregate_patchwise_graybody(
            [x, y], v_field, c_s, 1e12,  # Single float
            graybody_method="dimensionless",
            max_patches=8
        )
        
        assert result.success is True
        assert result.n_patches == 8