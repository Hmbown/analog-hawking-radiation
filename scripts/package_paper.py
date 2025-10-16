#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
from pathlib import Path

FIGS = [
    # Minimal figure set used in the reduced paper
    "formation_frontier.png",                 # Figure 1
    "horizon_analysis_detection_time.png",    # Figure 2 (left)
    "horizon_analysis_detection_time_TH.png", # Figure 2 (right)
]

ROOT = Path(__file__).resolve().parents[1]
FIG_SRC = ROOT / "figures"
PAPER = ROOT / "paper"
PAPER_FIG = PAPER / "figures"
BUILD = PAPER / "build_arxiv"


def main() -> int:
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    BUILD.mkdir(parents=True, exist_ok=True)

    # Ensure required figures exist in paper/figures; copy from root figures/ if needed
    selected: list[str] = []
    for fname in FIGS:
        dst_pf = PAPER_FIG / fname
        if dst_pf.exists():
            selected.append(fname)
            continue
        src_root = FIG_SRC / fname
        if src_root.exists():
            shutil.copy2(src_root, dst_pf)
            selected.append(fname)
        else:
            print(f"[warn] missing figure in both locations: {fname}")

    # Copy TeX and bib to build dir
    for item in ["main.tex", "refs.bib"]:
        src = PAPER / item
        if not src.exists():
            print(f"[error] missing {src}")
            return 1
        shutil.copy2(src, BUILD / item)

    # Copy only selected figures into build dir
    dest_fig_dir = BUILD / "figures"
    if dest_fig_dir.exists():
        shutil.rmtree(dest_fig_dir)
    dest_fig_dir.mkdir(parents=True, exist_ok=True)
    for fname in selected:
        src_pf = PAPER_FIG / fname
        if src_pf.exists():
            shutil.copy2(src_pf, dest_fig_dir / fname)

    # Create zip
    out_zip = PAPER / "arxiv_package"
    if (out_zip.with_suffix('.zip')).exists():
        (out_zip.with_suffix('.zip')).unlink()
    shutil.make_archive(str(out_zip), 'zip', root_dir=BUILD)
    print(f"[ok] Wrote {out_zip.with_suffix('.zip')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
