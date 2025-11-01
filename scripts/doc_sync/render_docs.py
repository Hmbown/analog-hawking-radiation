#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path
from string import Template


def _sci(x: float) -> str:
    return f"{x:.2e}"


def _pretty_si(x: float, unit: str = "") -> str:
    # Basic SI pretty printing; fallback to sci if extreme
    import math
    if x == 0:
        return f"0{unit}"
    exp = int(math.floor(math.log10(abs(x)) / 3) * 3)
    exp = max(min(exp, 12), -12)
    prefixes = {
        -12: "p", -9: "n", -6: "µ", -3: "m", 0: "",
        3: "k", 6: "M", 9: "G", 12: "T",
    }
    scaled = x / (10 ** exp)
    if exp in prefixes and 0.1 <= abs(scaled) < 1000:
        return f"{scaled:.3g}{prefixes[exp]}{unit}"
    return f"{x:.3e}{unit}"


def _load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _dominant_mode(stats: dict) -> str:
    items = [(k, v.get("rate", 0.0)) for k, v in stats.items() if k != "total_breakdown_rate"]
    if not items:
        return "unknown"
    return max(items, key=lambda kv: kv[1])[0]


def _render_template(tmpl_path: Path, out_path: Path, context: dict) -> None:
    text = tmpl_path.read_text(encoding="utf-8")
    rendered = Template(text).safe_substitute(context)
    out_path.write_text(rendered, encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Render docs from results JSONs")
    p.add_argument("--sweep", required=True, help="Path to gradient_catastrophe_sweep.json")
    p.add_argument("--pic", required=True, help="Path to pic_pipeline_summary.json")
    p.add_argument("--highlights-tmpl", default="docs/templates/RESEARCH_HIGHLIGHTS.tmpl.md")
    p.add_argument("--card-tmpl", default="docs/templates/findings_card.tmpl.md")
    p.add_argument("--out-highlights", default="RESEARCH_HIGHLIGHTS.md")
    p.add_argument("--out-card", default="docs/GradientCatastropheFindings.md")
    args = p.parse_args()

    sweep = _load_json(args.sweep)
    pic = _load_json(args.pic)
    analysis = sweep["analysis"]

    # Context
    kmax = float(analysis["max_kappa"])
    cfg = analysis["max_kappa_config"]
    sr = analysis["scaling_relationships"]
    b = analysis["breakdown_statistics"]
    # Pre-format values that need fixed decimals
    def exp_fmt(v: float) -> str:
        return f"{v:.2f}"
    opt_a0_val = float(cfg.get("a0", float("nan")))
    opt_grad_val = float(cfg.get("gradient_factor", float("nan")))
    ctx = {
        "date": dt.date.today().strftime("%B %Y"),
        "n_samples": int(sweep.get("n_samples", 0)),
        "kappa_max_pretty": _pretty_si(kmax, ""),
        "kappa_max_sci": _sci(kmax),
        "opt_a0": f"{opt_a0_val:.2f}",
        "opt_ne_pretty": _sci(float(cfg.get("n_e", float("nan")))),
        "opt_grad": f"{opt_grad_val:.2f}",
        "opt_intensity_pretty": _sci(float(cfg.get("intensity", float("nan")))),
        "exp_a0": exp_fmt(float(sr.get("kappa_vs_a0_exponent", float("nan")))),
        "exp_ne": exp_fmt(float(sr.get("kappa_vs_ne_exponent", float("nan")))),
        "exp_a0_lo": exp_fmt(float(sr.get("kappa_vs_a0_exponent_ci95", [float("nan"), float("nan")])[0])),
        "exp_a0_hi": exp_fmt(float(sr.get("kappa_vs_a0_exponent_ci95", [float("nan"), float("nan")])[1])),
        "exp_ne_lo": exp_fmt(float(sr.get("kappa_vs_ne_exponent_ci95", [float("nan"), float("nan")])[0])),
        "exp_ne_hi": exp_fmt(float(sr.get("kappa_vs_ne_exponent_ci95", [float("nan"), float("nan")])[1])),
        "valid": int(analysis.get("valid_configurations", 0)),
        "valid_rate": float(analysis.get("valid_configurations", 0)) / max(int(sweep.get("n_samples", 1)), 1),
        "breakdown_rate": float(b.get("total_breakdown_rate", 0.0)),
        "dominant_mode": _dominant_mode(b),
        "pic_horizons": ", ".join(_sci(float(x)) for x in pic.get("horizon_positions", [])),
        "pic_kappas": ", ".join(_sci(float(x)) for x in pic.get("kappa", [])),
        "pic_kappa_errs": ", ".join(_sci(float(x)) for x in pic.get("kappa_err", [])),
    }

    _render_template(Path(args.highlights_tmpl), Path(args.out_highlights), ctx)
    _render_template(Path(args.card_tmpl), Path(args.out_card), ctx)
    print(f"Rendered {args.out_highlights} and {args.out_card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
