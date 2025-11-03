import numpy as np

from analog_hawking.physics_engine.horizon import find_horizons_with_uncertainty
from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend


def test_openpmd_getter_reads_series(tmp_path):
    # Use h5py for fallback testing (no openpmd-api dependency)
    try:
        import h5py  # type: ignore
    except Exception:
        import pytest
        pytest.skip("h5py not available for testing")
    data = np.linspace(0.0, 1.0, 32)
    h5path = str(tmp_path / "sample.h5")
    with h5py.File(h5path, 'w') as f:
        f.create_dataset('/data', data=data)

    grid = np.linspace(0.0, 1.0, data.size)
    backend = WarpXBackend()
    backend.configure({
        "mock": True,
        "grid": grid,
        "moment_getters": {
            "electrons": {
                "bulk_velocity": {"type": "mock_data", "data": np.zeros_like(grid)},
                "sound_speed": {"type": "mock_data", "data": np.full_like(grid, 0.5)},
                "density": {"type": "mock_data", "data": np.full_like(grid, 1.0)},
                "temperature": {"type": "mock_data", "data": np.full_like(grid, 1.0)},
            }
        },
        "field_getters": {
            "vel": {
                "type": "openpmd",
                "series_path": h5path,
                "dataset": "/data"
            }
        }
    })

    # Run one step to populate observables
    state = backend.step(0.0)
    out = backend.export_observables(["vel"])
    assert "vel" in out
    np.testing.assert_allclose(out["vel"], data, rtol=1e-5)

    # Check core PlasmaState fields
    assert state.density.size > 0
    assert state.velocity.size > 0
    assert state.sound_speed.size > 0
    assert np.allclose(state.sound_speed, 0.5)  # From mock data


def test_horizons_from_openpmd_data():
    # Test integration with horizon finding (analytic check)
    N = 101
    x = np.linspace(0.0, 2.0, N)
    v = 0.6 * x  # Adjusted for horizon around x=0.8333, kappa ≈ 0.6
    cs = np.full(N, 0.5)
    density = np.full(N, 1.0)
    T_e = np.full(N, 1.0)  # cs = sqrt(T_e / density) = 1, but we override with cs=0.5

    grid = x
    backend = WarpXBackend()
    backend.configure({
        "mock": True,
        "grid": grid,
        "moment_getters": {
            "electrons": {
                "bulk_velocity": {"type": "mock_data", "data": v},
                "sound_speed": {"type": "mock_data", "data": cs},
                "density": {"type": "mock_data", "data": density},
                "temperature": {"type": "mock_data", "data": T_e},
            }
        },
    })

    state = backend.step(0.0)
    horizons = find_horizons_with_uncertainty(state.grid, state.velocity, state.sound_speed, kappa_method="acoustic_exact")

    # Analytic: horizon where v = cs =0.5, 0.6x = 0.5 => x = 0.5 / 0.6 = 0.8333, kappa = |dv/dx| =0.6
    expected_pos = 0.8333
    expected_kappa = 0.6
    assert len(horizons.positions) > 0
    # Take the first (only) horizon
    pos = horizons.positions[0]
    kap = horizons.kappa[0]
    assert abs(pos - expected_pos) / expected_pos < 0.05  # <5% error
    assert abs(kap - expected_kappa) / expected_kappa < 0.05  # <5% error
