import json
import os
import subprocess
import sys


def test_run_full_pipeline_cli_flags(tmp_path):
    # Run with new CLI flags; expect a summary JSON output
    out_dir = tmp_path / "results"
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str((tmp_path.parent / "src")) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        sys.executable,
        "scripts/run_full_pipeline.py",
        "--demo",
        "--kappa-method",
        "acoustic",
        "--graybody",
        "dimensionless",
        "--alpha-gray",
        "1.0",
        "--Tsys",
        "25",
        "--window-cells",
        "16",
    ]
    subprocess.check_call(cmd, env=env)
    summary_path = os.path.join("results", "full_pipeline_summary.json")
    assert os.path.exists(summary_path)
    with open(summary_path) as f:
        data = json.load(f)
    assert "kappa" in data
    assert "t5sigma_s" in data
