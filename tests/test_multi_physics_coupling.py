"""Tests for multi-physics coupling in advanced simulations (Phase 3)."""

from __future__ import annotations

import numpy as np
import pytest

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.nonlinear_plasma import NonlinearPlasmaSolver
from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend


@pytest.fixture
def mock_run_config():
    """Mock configuration for WarpX backend with Phase 3 features."""
    return {
        "mock": True,
        "mhd_enabled": True,
        "qft_3d": True,
        "nonlinear_config": {
            "nonlinear_strength": 0.1,
            "qft_modes": 5,
            "kappa_enhancement": 10.0,
            "t_h_target": 1e-3,
            "universality_r2": 0.98,
        },
        "default_density": 1e18,
        "default_temperature": 1e4,
        "species": [
            {"name": "electrons", "charge": -1.6e-19, "mass": 9.1e-31},
            {"name": "ions", "charge": 1.6e-19, "mass": 1.67e-27},
        ],
        "grid": np.linspace(0, 1e-4, 50),
        "field_getters": {
            "electric_field": {"type": "mock_data", "data": np.sin(np.linspace(0, 2*np.pi, 50)) * 1e5},
            "magnetic_field": {"type": "mock_data", "data": np.full(50, 0.1)},
        },
        "moment_getters": {
            "electrons": {
                "density": {"type": "mock_data", "data": np.full(50, 1e18)},
                "bulk_velocity": {"type": "mock_data", "data": np.linspace(-0.5, 0.5, 50)},
                "temperature": {"type": "mock_data", "data": np.full(50, 1e4)},
            },
            "ions": {
                "density": {"type": "mock_data", "data": np.full(50, 1e18)},
                "bulk_velocity": {"type": "mock_data", "data": np.linspace(-0.5, 0.5, 50)},
                "temperature": {"type": "mock_data", "data": np.full(50, 1e4)},
            }
        },
        "electron_species": "electrons",
        "ion_species": "ions",
        "sigma_smoothing": {"adaptive": True},
        "geometry": "cartesian_3d",
        "lower_bounds": [0.0, 0.0, 0.0],
        "upper_bounds": [1e-4, 1e-4, 1e-4],
        "cells_each_dim": [50, 25, 25],
        "lasers": [{"name": "laser1", "profile": "gaussian", "amplitude": 1e18}],
    }


def test_warpx_backend_mhd_coupling(mock_run_config):
    """Test MHD coupling initialization and field updates."""
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    assert backend._mhd_enabled is True
    assert "E" in backend._em_fields
    assert "B" in backend._em_fields
    assert len(backend._em_fields["E"]) == 50  # Grid size

    # Step simulation
    state = backend.step(1e-15)
    assert "density_mhd" in state.observables
    assert np.allclose(state.electric_field, mock_run_config["field_getters"]["electric_field"]["data"], atol=1e-5)


def test_nonlinear_solver_integration(mock_run_config):
    """Test nonlinear solver with QFT 3D metrics."""
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    state = backend.step(1e-15)

    solver = NonlinearPlasmaSolver(mock_run_config["nonlinear_config"])
    enhanced_obs = solver.solve(state.observables)

    assert "enhanced_kappa" in enhanced_obs
    assert "t_hawking" in enhanced_obs
    assert "universality_r2" in enhanced_obs
    assert enhanced_obs["universality_r2"] >= 0.95  # Close to target
    assert enhanced_obs["kappa_stability"] < 0.05  # <5% for test
    assert enhanced_obs["t_5sigma"] < 2.0  # Relaxed for mock


def test_horizon_kappa_enhancement(mock_run_config):
    """Test kappa enhancement and stability <3%."""
    mock_run_config["nonlinear_config"]["kappa_enhancement"] = 50.0  # Mid-range
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    state = backend.step(1e-15)

    solver = NonlinearPlasmaSolver(mock_run_config["nonlinear_config"])
    enhanced_obs = solver.solve(state.observables)

    # Find horizons
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, np.full_like(state.velocity, 0.5))
    base_kappa = np.mean(horizons.kappa) if horizons.kappa.size > 0 else 1.0
    enhanced_kappa = enhanced_obs["enhanced_kappa"]

    assert enhanced_kappa > base_kappa * 10  # At least 10x
    kappa_stability = enhanced_obs["kappa_stability"]
    assert kappa_stability < 0.03  # <3%


def test_5sigma_detection_time(mock_run_config):
    """Test t_5σ <1s in mock config."""
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    state = backend.step(1e-15)

    solver = NonlinearPlasmaSolver(mock_run_config["nonlinear_config"])
    enhanced_obs = solver.solve(state.observables)

    t_5sigma = enhanced_obs["t_5sigma"]
    assert t_5sigma < 1.0  # Target


def test_universality_r2(mock_run_config):
    """Test R² >0.98 for universality."""
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    state = backend.step(1e-15)

    solver = NonlinearPlasmaSolver(mock_run_config["nonlinear_config"])
    enhanced_obs = solver.solve(state.observables)

    r2 = enhanced_obs["universality_r2"]
    assert r2 > 0.98  # Target


@pytest.mark.parametrize("kappa_enh", [10.0, 50.0, 100.0])
def test_param_sweep_integration(kappa_enh, mock_run_config):
    """Test integration with sweep params (subset)."""
    mock_run_config["nonlinear_config"]["kappa_enhancement"] = kappa_enh
    backend = WarpXBackend()
    backend.configure(mock_run_config)
    state = backend.step(1e-15)

    solver = NonlinearPlasmaSolver(mock_run_config["nonlinear_config"])
    enhanced_obs = solver.solve(state.observables)

    assert enhanced_obs["enhanced_kappa"] > 10.0  # Minimum enhancement
    assert enhanced_obs["t_hawking"] > 1e-3  # >1 mK GHz


def test_resource_adaptation():
    """Test CPU fallback (mock mode)."""
    config = {"mock": True, "mhd_enabled": False, "qft_3d": False}  # No GPU req
    backend = WarpXBackend()
    backend.configure(config)
    state = backend.step(1e-15)
    assert state is not None  # No crash on CPU


if __name__ == "__main__":
    pytest.main([__file__])
