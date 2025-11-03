import os

import numpy as np
import pytest

try:
    import h5py  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    h5py = None


def test_openpmd_slice_to_profile_roundtrip(tmp_path):
    if h5py is None:
        pytest.skip("h5py is not installed; skipping openPMD roundtrip test.")
    # Create synthetic HDF5 with velocity and temperature datasets
    N = 64
    x = np.linspace(0.0, 1.0, N)
    v = 0.3 * (2 * x - 1.0)
    Te = 1.0e4 * np.ones_like(x)
    h5path = os.path.join(tmp_path, "sample.h5")
    with h5py.File(h5path, 'w') as f:
        f.create_dataset('/x', data=x)
        f.create_dataset('/vel', data=v)
        f.create_dataset('/Te', data=Te)

    out_npz = os.path.join(tmp_path, "out_profile.npz")
    # Invoke script via its CLI interface
    import subprocess
    import sys
    cmd = [sys.executable, 'scripts/openpmd_slice_to_profile.py', '--in', h5path, '--x-dataset', '/x', '--vel-dataset', '/vel', '--Te-dataset', '/Te', '--out', out_npz]
    subprocess.check_call(cmd)

    npz = np.load(out_npz)
    assert 'x' in npz and 'v' in npz and 'c_s' in npz
    assert npz['x'].shape == (N,)
    assert npz['v'].shape == (N,)
    assert npz['c_s'].shape == (N,)
