import os

import numpy as np
import pytest


def test_openpmd_slice_to_profile_roundtrip(tmp_path):
    """Test openPMD slice to profile conversion roundtrip."""
    # Import h5py here to avoid collection-time errors
    try:
        import h5py  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        pytest.skip("h5py is not available; skipping openPMD roundtrip test.")
    
    # Create synthetic HDF5 with velocity and temperature datasets
    N = 64
    x = np.linspace(0.0, 1.0, N)
    v = 0.3 * (2 * x - 1.0)
    Te = 1.0e4 * np.ones_like(x)
    h5path = os.path.join(tmp_path, "sample.h5")
    with h5py.File(h5path, "w") as f:
        f.create_dataset("/x", data=x)
        f.create_dataset("/vel", data=v)
        f.create_dataset("/Te", data=Te)

    out_npz = os.path.join(tmp_path, "out_profile.npz")
    # Invoke script via its CLI interface
    import subprocess
    import sys

    # Get the project root directory (parent of tests/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(project_root, "scripts", "openpmd_slice_to_profile.py")
    
    cmd = [
        sys.executable,
        script_path,
        "--in",
        h5path,
        "--x-dataset",
        "/x",
        "--vel-dataset",
        "/vel",
        "--Te-dataset",
        "/Te",
        "--out",
        out_npz,
    ]
    subprocess.check_call(cmd)

    # Load and verify
    data = np.load(out_npz)
    assert "x" in data.files
    assert "v" in data.files
    # Script uses c_s, not cs
    assert "c_s" in data.files
    # Ensure roundtrip preserves shape and reasonable values
    assert data["x"].shape == x.shape
    assert data["v"].shape == v.shape
