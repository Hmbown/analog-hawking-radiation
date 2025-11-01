from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass(frozen=True)
class Thresholds:
    v_max_fraction_c: float = 0.5
    dv_dx_max_s: float = 4.0e12
    # ELI-compliant maximum intensity (10^24 W/cm² = 10^28 W/m² with safety margin)
    intensity_max_W_m2: float = 1.0e28


def load_thresholds(path: Optional[str | Path]) -> Thresholds:
    """Load thresholds from YAML, falling back to defaults if missing.

    The function is intentionally permissive: a missing file or missing keys
    result in default values, so scripts remain robust while still centralizing
    provenance.
    """
    defaults = Thresholds()
    if not path:
        return defaults
    try:
        p = Path(path)
        if not p.exists():
            return defaults
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        return Thresholds(
            v_max_fraction_c=float(data.get("v_max_fraction_c", defaults.v_max_fraction_c)),
            dv_dx_max_s=float(data.get("dv_dx_max_s", defaults.dv_dx_max_s)),
            intensity_max_W_m2=float(data.get("intensity_max_W_m2", defaults.intensity_max_W_m2)),
        )
    except Exception:
        # On any parsing error, return defaults (fail-safe)
        return defaults

