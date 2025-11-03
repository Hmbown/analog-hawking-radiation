#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from string import Template

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analog_hawking.analysis.gradient_sweep import generate_catastrophe_plots


def _load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _render_card(sweep: dict, pic: dict, tmpl_path: Path, out_path: Path) -> None:
    analysis = sweep["analysis"]
    ctx = {
        "kappa_max_pretty": f"{analysis['max_kappa']:.2e}",
        "opt_a0": f"{analysis['max_kappa_config']['a0']:.2f}" if analysis.get("max_kappa_config") else "na",
        "opt_ne_pretty": f"{analysis['max_kappa_config']['n_e']:.2e}" if analysis.get("max_kappa_config") else "na",
        "opt_grad": f"{analysis['max_kappa_config']['gradient_factor']:.2f}" if analysis.get("max_kappa_config") else "na",
        "exp_a0": f"{analysis['scaling_relationships']['kappa_vs_a0_exponent']:.2f}",
        "exp_a0_lo": f"{analysis['scaling_relationships']['kappa_vs_a0_exponent_ci95'][0]:.2f}",
        "exp_a0_hi": f"{analysis['scaling_relationships']['kappa_vs_a0_exponent_ci95'][1]:.2f}",
        "exp_ne": f"{analysis['scaling_relationships']['kappa_vs_ne_exponent']:.2f}",
        "exp_ne_lo": f"{analysis['scaling_relationships']['kappa_vs_ne_exponent_ci95'][0]:.2f}",
        "exp_ne_hi": f"{analysis['scaling_relationships']['kappa_vs_ne_exponent_ci95'][1]:.2f}",
        "pic_horizons": ", ".join(f"{x:.2e}" for x in pic.get("horizon_positions", [])),
        "pic_kappas": ", ".join(f"{x:.2e}" for x in pic.get("kappa", [])),
    }
    tmpl = Template(tmpl_path.read_text(encoding="utf-8"))
    out_path.write_text(tmpl.safe_substitute(ctx), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Build a shareable results pack")
    p.add_argument("--sweep", required=True, help="gradient_catastrophe_sweep.json")
    p.add_argument("--pic", required=True, help="pic_pipeline_summary.json")
    p.add_argument("--out", required=True, help="output directory for the pack")
    args = p.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    sweep = _load(args.sweep)
    pic = _load(args.pic)

    # Copy source JSONs
    shutil.copy2(args.sweep, outdir / Path(args.sweep).name)
    shutil.copy2(args.pic, outdir / Path(args.pic).name)

    # Re-generate plots into the pack folder
    generate_catastrophe_plots(sweep, str(outdir))

    # Render 1-page card
    card_tmpl = Path("docs/templates/findings_card.tmpl.md")
    _render_card(sweep, pic, card_tmpl, outdir / "summary.md")

    print(f"Results pack created at {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

