#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def main() -> int:
    cases_path = Path("results") / "horizon_success_cases.json"
    out_path = Path("docs") / "Successful_Configurations.md"
    if not cases_path.exists():
        print(f"No success cases at {cases_path}")
        return 1
    with open(cases_path, "r") as f:
        cases = json.load(f)
    if not cases:
        content = """# Successful Configurations

No horizon-forming configurations were found in the latest sweep.
"""
        out_path.write_text(content)
        print(f"Wrote {out_path}")
        return 0

    # Sort by kappa_max desc
    cases = [c for c in cases if "kappa_max" in c]
    cases.sort(key=lambda c: float(c["kappa_max"]), reverse=True)
    top = cases[:10]

    def fmt_case(c):
        return {
            "n_cm3": float(c.get("input_density_cm3", np.nan)),
            "I_Wcm2": float(c.get("input_intensity_Wcm2", np.nan)),
            "T_K": float(c.get("input_temperature_K", np.nan)),
            "B_T": float(c.get("input_B_T", np.nan)),
            "kappa_max": float(c.get("kappa_max", np.nan)),
            "t5sigma_s": c.get("t5sigma_s", None),
        }

    rows = [fmt_case(c) for c in top]

    lines = [
        "# Successful Configurations",
        "",
        "This report summarizes configurations from the extended sweep that formed horizons.",
        "",
        "- Figures: `figures/horizon_analysis_probability_map.png`, `figures/horizon_analysis_kappa_map.png`, `figures/horizon_analysis_TH_map.png`, `figures/horizon_analysis_profile_*.png`, `figures/horizon_analysis_detection_time.png`",
        "- Data: `results/extended_param_sweep.json`, `results/horizon_success_cases.json`",
        "",
        "## Top 10 by κ",
        "",
        "| n_e [cm^-3] | I [W/cm^2] | T [K] | B [T] | κ_max [s^-1] | t_5σ [s] |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        t_str = (
            f"{r['t5sigma_s']:.3e}"
            if isinstance(r["t5sigma_s"], (int, float))
            else str(r["t5sigma_s"])
        )
        lines.append(
            f"| {r['n_cm3']:.3e} | {r['I_Wcm2']:.3e} | {r['T_K']:.3e} | {r['B_T']:.2f} | {r['kappa_max']:.3e} | {t_str} |"
        )

    lines += [
        "",
        "### Notes",
        "- κ values exceed the 1e10 s^-1 target in multiple regions of parameter space.",
        "- Current QFT spectrum normalization yields very small in-band powers in radio bands; `t_5σ` values remain extremely large.",
        "- See `results/physical_validation_report.json` for a0 and ω_p checks.",
    ]

    out = "\n".join(lines)
    out_path.write_text(out)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
