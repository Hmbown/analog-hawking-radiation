#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional, Sequence

import numpy as np

import sys
from pathlib import Path
# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from run_full_pipeline import run_full_pipeline, FullPipelineSummary


@dataclass
class SweepResult:
    temperature_values: list[float]
    magnetic_field_values: list[Optional[float]]
    entries: list[FullPipelineSummary]


def run_sweep(
    temperatures: Sequence[float] = (1e5, 3e5, 1e6),
    magnetic_fields: Sequence[Optional[float]] = (None, 0.0, 0.005, 0.02),
) -> SweepResult:
    entries: list[FullPipelineSummary] = []
    for T in temperatures:
        for B in magnetic_fields:
            summary = run_full_pipeline(
                temperature_constant=float(T),
                magnetic_field=None if B is None else float(B),
            )
            entries.append(summary)
    return SweepResult(
        temperature_values=[float(x) for x in temperatures],
        magnetic_field_values=[None if x is None else float(x) for x in magnetic_fields],
        entries=entries,
    )


def main() -> int:
    result = run_sweep()
    os.makedirs("results", exist_ok=True)
    out_path = os.path.join("results", "param_sweep_summary.json")
    # Convert dataclasses to serializable dicts
    payload = {
        "temperature_values": result.temperature_values,
        "magnetic_field_values": result.magnetic_field_values,
        "entries": [asdict(e) for e in result.entries],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved parameter sweep summary to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
