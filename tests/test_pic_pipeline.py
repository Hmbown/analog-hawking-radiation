import json
import os
import numpy as np
import subprocess
import sys


def test_run_pic_pipeline_with_synthetic_profile(tmp_path):
    # Create a simple profile NPZ
    x = np.linspace(-1.0, 1.0, 2001)
    v = 1.0 * x
    c = np.full_like(x, 0.2)
    npz_path = os.path.join(tmp_path, 'profile.npz')
    np.savez(npz_path, x=x, v=v, c_s=c)

    # Run the PIC pipeline
    cmd = [sys.executable, 'scripts/run_pic_pipeline.py', '--profile', npz_path, '--graybody', 'dimensionless']
    subprocess.check_call(cmd)

    out_path = os.path.join('results', 'pic_pipeline_summary.json')
    assert os.path.exists(out_path)
    data = json.loads(open(out_path).read())
    assert 'kappa' in data
