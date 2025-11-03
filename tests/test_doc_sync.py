from __future__ import annotations

import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _sweep_values() -> dict:
    sweep_path = ROOT / "results/gradient_limits_production/gradient_catastrophe_sweep.json"
    if not sweep_path.exists():
        pytest.skip(f"Gradient catastrophe sweep results missing: {sweep_path}")
    data = json.loads(_read(sweep_path))
    a = data["analysis"]
    return {
        "kmax": float(a["max_kappa"]),
        "exp_a0": float(a["scaling_relationships"]["kappa_vs_a0_exponent"]),
        "exp_a0_lo": float(a["scaling_relationships"]["kappa_vs_a0_exponent_ci95"][0]),
        "exp_a0_hi": float(a["scaling_relationships"]["kappa_vs_a0_exponent_ci95"][1]),
        "exp_ne": float(a["scaling_relationships"]["kappa_vs_ne_exponent"]),
        "exp_ne_lo": float(a["scaling_relationships"]["kappa_vs_ne_exponent_ci95"][0]),
        "exp_ne_hi": float(a["scaling_relationships"]["kappa_vs_ne_exponent_ci95"][1]),
    }


def _extract_highlights_numbers(text: str) -> dict:
    # κ_max in sci notion e.g., 5.94e12 or 5.94×10¹²
    kmax = None
    # Fallback to the explicit sci block we render
    m2 = re.search(r"κ_max\s*=\s*([0-9.+-eE]+)\s*Hz", text)
    if m2:
        try:
            kmax = float(m2.group(1))
        except Exception:
            pass

    # Exponents lines (rendered form)
    def _grab(tag: str) -> tuple[float, float, float]:
        rx = (
            rf"{tag}: exponent ≈\s*([+-]?[0-9.]+)\s*\(95% CI \[([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\]\)"
        )
        m = re.search(rx, text)
        if not m:
            return math.nan, math.nan, math.nan
        return float(m.group(1)), float(m.group(2)), float(m.group(3))

    a0 = _grab(r"κ vs a₀")
    ne = _grab(r"κ vs nₑ")
    return {
        "kmax": kmax,
        "exp_a0": a0[0],
        "exp_a0_lo": a0[1],
        "exp_a0_hi": a0[2],
        "exp_ne": ne[0],
        "exp_ne_lo": ne[1],
        "exp_ne_hi": ne[2],
    }


def test_docs_match_results_to_95ci_tolerances():
    vals = _sweep_values()
    hl = _read(ROOT / "RESEARCH_HIGHLIGHTS.md")
    nums = _extract_highlights_numbers(hl)

    # κ_max must be present and within ~0.5% relative tolerance
    assert nums["kmax"] is not None
    assert math.isclose(nums["kmax"], vals["kmax"], rel_tol=5e-3)

    # Exponents and CI bounds must match closely
    for key in ("exp_a0", "exp_a0_lo", "exp_a0_hi", "exp_ne", "exp_ne_lo", "exp_ne_hi"):
        assert not math.isnan(nums[key]), f"Missing {key} in RESEARCH_HIGHLIGHTS.md"
        assert math.isclose(nums[key], vals[key], rel_tol=5e-3), f"Mismatch for {key}"
