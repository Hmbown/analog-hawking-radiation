#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

FIGS = [
    # Results figures referenced in paper/main.tex
    "formation_frontier.png",
    "graybody_impact.png",
    "geometry_vs_kappa.png",
    "horizon_probability_bands.png",
    # Existing paper figures
    "horizon_analysis_probability_map.png",
    "horizon_analysis_kappa_map.png",
    "horizon_analysis_TH_map.png",
    "horizon_analysis_detection_time.png",
    "horizon_analysis_detection_time_TH.png",
    "horizon_analysis_detection_time_radio.png",
]

ROOT = Path(__file__).resolve().parents[1]
FIG_SRC = ROOT / "figures"
PAPER = ROOT / "paper"
PAPER_FIG = PAPER / "figures"
BUILD = PAPER / "build_arxiv"


def main() -> int:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    # Copy figures
    copied = []
    for fname in FIGS:
        src = FIG_SRC / fname
        if src.exists():
            shutil.copy2(src, PAPER_FIG / fname)
            copied.append(fname)
        else:
            print(f"[warn] missing figure: {src}")

    # Copy TeX and bib to build dir
    for item in ["main.tex", "refs.bib"]:
        src = PAPER / item
        if not src.exists():
            print(f"[error] missing {src}")
            return 1
        shutil.copy2(src, BUILD / item)

    # Copy figures dir into build dir
    dest_fig_dir = BUILD / "figures"
    if dest_fig_dir.exists():
        shutil.rmtree(dest_fig_dir)
    shutil.copytree(PAPER_FIG, dest_fig_dir)

    # Create zip
    out_zip = PAPER / "arxiv_package"
    if (out_zip.with_suffix('.zip')).exists():
        (out_zip.with_suffix('.zip')).unlink()
    shutil.make_archive(str(out_zip), 'zip', root_dir=BUILD)
    print(f"[ok] Wrote {out_zip.with_suffix('.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
