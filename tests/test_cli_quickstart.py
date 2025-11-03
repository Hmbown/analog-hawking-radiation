import json
from pathlib import Path

import pytest

from analog_hawking.cli.main import main


def test_quickstart_generates_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # Disable plotting in CI
    monkeypatch.setenv("ANALOG_HAWKING_NO_PLOTS", "1")
    outdir = tmp_path / "quick"
    rc = main(["quickstart", "--out", str(outdir), "--nx", "200"])  # small grid
    assert rc == 0
    manifest_path = outdir / "manifest.json"
    assert manifest_path.exists(), "manifest.json not written"
    data = json.loads(manifest_path.read_text())
    assert "git_commit" in data
    assert "environment" in data
    assert data["outputs"]["n_horizons"] >= 0

