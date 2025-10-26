import json
import os
import numpy as np
import subprocess
import sys


def test_run_pic_pipeline_with_synthetic_profile(tmp_path):
    # Create a simple profile NPZ with known analytic horizon
    x = np.linspace(-1.0, 1.0, 2001)
    v = 1.0 * x  # dv/dx = 1.0
    c = np.full_like(x, 0.2)  # cs = 0.2
    npz_path = os.path.join(tmp_path, 'profile.npz')
    np.savez(npz_path, x=x, v=v, c_s=c)

    # Run the PIC pipeline
    cmd = [sys.executable, 'scripts/run_pic_pipeline.py', '--profile', npz_path, '--graybody', 'dimensionless', '--kappa-method', 'acoustic_exact']
    subprocess.check_call(cmd)

    out_path = os.path.join('results', 'pic_pipeline_summary.json')
    assert os.path.exists(out_path)
    data = json.loads(open(out_path).read())
    assert 'kappa' in data
    kappas = data['kappa']
    assert len(kappas) >= 2  # Two horizons at +/-0.2
    # Check kappa ≈1.0 within 5%
    for kap in kappas:
        assert abs(kap - 1.0) / 1.0 < 0.05
    # Check positions ≈ +/-0.2
    positions = data['horizon_positions']
    assert len(positions) >= 2
    pos_set = set([round(p, 1) for p in positions])
    assert -0.2 in pos_set or round(-0.2, 1) in pos_set
    assert 0.2 in pos_set or round(0.2, 1) in pos_set
