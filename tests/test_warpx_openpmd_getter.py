import os
import numpy as np

from analog_hawking.physics_engine.plasma_models.warpx_backend import WarpXBackend


def test_openpmd_getter_reads_hdf5(tmp_path):
    # Create a small HDF5 file with a dataset
    import h5py  # type: ignore
    data = np.linspace(0.0, 1.0, 32)
    h5path = os.path.join(tmp_path, "sample.h5")
    with h5py.File(h5path, 'w') as f:
        f.create_dataset('/data', data=data)

    grid = np.linspace(0.0, 1.0, data.size)
    backend = WarpXBackend()
    backend.configure({
        "mock": True,
        "grid": grid,
        "moment_getters": {
            "electrons": {
                "bulk_velocity": {"type": "pywarpx", "moment": "bulk_velocity"},
                "sound_speed": {"type": "pywarpx", "moment": "sound_speed"},
                "density": {"type": "pywarpx", "moment": "density"},
            }
        },
        "field_getters": {
            "vel": {"type": "openpmd", "path": h5path, "dataset": "/data"}
        }
    })

    # Run one step to populate observables
    _ = backend.step(0.0)
    out = backend.export_observables(["vel"])  # pulls from raw observables
    assert "vel" in out
    np.testing.assert_allclose(out["vel"], data)
