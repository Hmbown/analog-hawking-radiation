from __future__ import annotations

import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analog_hawking.physics_engine.horizon import sound_speed


def test_sound_speed_kelvin_vs_ev_equivalence(tmp_path: Path):
    # Temperature: 30 eV
    Te_eV = 30.0
    Te_K = Te_eV * 11604.51812

    cs_from_K = sound_speed(Te_K)
    # Should be same when converted from eV
    cs_from_eV = sound_speed(Te_K)
    assert np.isclose(cs_from_K, cs_from_eV, rtol=1e-12)

