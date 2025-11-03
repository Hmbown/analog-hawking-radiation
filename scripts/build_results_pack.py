#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Summary:
    total_configs: int
    coupling_count: int
    D_count: int
    temp_enhancement_mean: float
    speed_improvement_mean: float
    superiority_pct: float
    pareto_count: int
    pareto_pct: float
    top_param: str
    top_param_score: float
    temp_slope_hybrid: float
    time_slope_hybrid: float


def compute_summary(df: pd.DataFrame) -> Summary:
    total = len(df)
    coupling_count = int(df['coupling_strength'].nunique())
    D_count = int(df['D'].nunique())

    # Improvements (dataset-level means)
    temp_enh = (df['T_sig_hybrid'].mean() / df['T_sig_fluid'].mean()) if df['T_sig_fluid'].mean() else float('nan')
    speed_imp = (df['t5_fluid'].mean() / df['t5_hybrid'].mean()) if df['t5_hybrid'].mean() else float('nan')

    # Superiority percentage (both higher T and shorter t)
    temp_better = int((df['T_sig_hybrid'] > df['T_sig_fluid']).sum())
    time_better = int((df['t5_hybrid'] < df['t5_fluid']).sum())
    superiority_pct = 100.0 * min(temp_better, time_better) / max(1, total)

    # Pareto frontier (maximize T, minimize t)
    vals = df[['T_sig_hybrid', 't5_hybrid']].to_numpy()
    norm = np.column_stack([-vals[:, 0], vals[:, 1]])
    pareto = np.ones(total, dtype=bool)
    for i in range(total):
        for j in range(total):
            if i != j and np.all(norm[j] <= norm[i]) and np.any(norm[j] < norm[i]):
                pareto[i] = False
                break
    pareto_count = int(pareto.sum())
    pareto_pct = 100.0 * pareto_count / max(1, total)

    # Correlation-based parameter influence
    parameters = ['coupling_strength', 'D', 'w_effective', 'kappa_mirror']
    objectives = ['T_sig_hybrid', 't5_hybrid', 'ratio_fluid_over_hybrid']
    corr = df[parameters + objectives].corr()
    avg_abs = corr.loc[parameters, objectives].abs().mean(axis=1)
    top_param = str(avg_abs.idxmax())
    top_param_score = float(avg_abs.max())

    # Simple hybrid scaling (log-log) vs coupling_strength
    log_c = np.log10(df['coupling_strength'])
    temp_slope_hybrid, *_ = stats.linregress(log_c, np.log10(df['T_sig_hybrid']))
    time_slope_hybrid, *_ = stats.linregress(log_c, np.log10(df['t5_hybrid']))

    return Summary(
        total_configs=total,
        coupling_count=coupling_count,
        D_count=D_count,
        temp_enhancement_mean=float(temp_enh),
        speed_improvement_mean=float(speed_imp),
        superiority_pct=float(superiority_pct),
        pareto_count=pareto_count,
        pareto_pct=float(pareto_pct),
        top_param=top_param,
        top_param_score=float(top_param_score),
        temp_slope_hybrid=float(temp_slope_hybrid),
        time_slope_hybrid=float(time_slope_hybrid),
    )


def write_summary_md(out_dir: Path, s: Summary) -> None:
    text = f"""
# Analog Hawking Radiation – Results Pack

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M %Z')}

Contents
- Figures: analysis PNGs in this directory
- Data: hybrid_sweep.csv (dataset used)
- Docs: Reproducibility, Dataset Notes, Glossary, FAQ
- Citation and License

Dataset Overview
- Total configurations: {s.total_configs}
- Coupling strengths: {s.coupling_count}
- Diffusion coefficients D: {s.D_count}

Key Metrics (dataset averages)
- Temperature enhancement (hybrid vs fluid): ~{s.temp_enhancement_mean:.1f}×
- Detection speed improvement (fluid t5 / hybrid t5): ~{s.speed_improvement_mean:.1f}×
- Hybrid superiority (T higher and t lower): {s.superiority_pct:.0f}%
- Pareto-optimal configurations: {s.pareto_count} ({s.pareto_pct:.1f}%)
- Most influential parameter (by avg |corr|): {s.top_param} (|r| ≈ {s.top_param_score:.3f})

Scaling Snapshot (log–log vs coupling_strength)
- T_sig_hybrid ∝ coupling_strength^{s.temp_slope_hybrid:.3f}
- t5_hybrid ∝ coupling_strength^{s.time_slope_hybrid:.3f}

Interpretation Guardrails
- Some correlations are near-perfect by construction (constants or derived quantities).
- Scaling with coupling_strength is flat/weak in this dataset; do not generalize.
- 4×/16× improvements follow radiometer scaling and are model-dependent.

Reproduce & Extend
- See Reproducibility.md for exact commands.
- Re-run `make comprehensive` to update figures for modified datasets.
"""
    (out_dir / "RESULTS_README.md").write_text(text.strip() + "\n", encoding="utf-8")


def _plot_ratio_heatmap(df: pd.DataFrame, out: Path) -> Path:
    pivot = df.pivot_table(values='ratio_fluid_over_hybrid', index='D', columns='coupling_strength')
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=150)
    im = ax.imshow(pivot.values, aspect='auto', origin='lower',
                   extent=(pivot.columns.min(), pivot.columns.max(), pivot.index.min()*1e6, pivot.index.max()*1e6),
                   cmap='viridis')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r't$_{5\sigma}$ fluid / t$_{5\sigma}$ hybrid (×)')
    ax.set_xlabel('Coupling strength')
    ax.set_ylabel('D [µm]')
    ax.set_title('Hybrid Detection Speed Improvement')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_detection_scatter(df: pd.DataFrame, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(4.5, 4.5), dpi=150)
    sc = ax.scatter(df['t5_fluid'], df['t5_hybrid'], c=df['D']*1e6, cmap='plasma', s=50, alpha=0.9)
    lims = [min(df['t5_fluid'].min(), df['t5_hybrid'].min()), max(df['t5_fluid'].max(), df['t5_hybrid'].max())]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.7)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('t$_{5\sigma}$ fluid [s]')
    ax.set_ylabel('t$_{5\sigma}$ hybrid [s]')
    ax.set_title('Detection Time Comparison')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('D [µm]')
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_enhancement_strip(df: pd.DataFrame, out: Path) -> Path:
    enh = df['T_sig_hybrid'] / df['T_sig_fluid']
    fig, ax = plt.subplots(figsize=(6.0, 2.8), dpi=150)
    y = np.zeros_like(enh.values)
    ax.scatter(enh, y, c=df['coupling_strength'], cmap='viridis', s=40, alpha=0.9, edgecolor='none')
    ax.axvline(enh.mean(), color='k', lw=1, linestyle='--', alpha=0.7, label=f"mean ~ {enh.mean():.1f}×")
    ax.set_yticks([])
    ax.set_xlabel('Temperature enhancement (T$_{sig}$ hybrid / fluid) [×]')
    ax.set_title('Signal Temperature Enhancement')
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def _plot_pareto(df: pd.DataFrame, out: Path) -> Path:
    vals = df[['T_sig_hybrid', 't5_hybrid']].to_numpy()
    norm = np.column_stack([-vals[:, 0], vals[:, 1]])
    pareto = np.ones(len(df), dtype=bool)
    for i in range(len(df)):
        for j in range(len(df)):
            if i != j and np.all(norm[j] <= norm[i]) and np.any(norm[j] < norm[i]):
                pareto[i] = False
                break
    fig, ax = plt.subplots(figsize=(5.2, 4.0), dpi=150)
    ax.scatter(df['T_sig_hybrid'], df['t5_hybrid'], c='lightgray', s=40, label='All')
    ax.scatter(df.loc[pareto, 'T_sig_hybrid'], df.loc[pareto, 't5_hybrid'], c='crimson', s=60, label='Pareto')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('T$_{sig}$ hybrid [K]')
    ax.set_ylabel('t$_{5\sigma}$ hybrid [s]')
    ax.set_title('Pareto Frontier (T vs Speed)')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_curated_figures(df: pd.DataFrame, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figs: List[Path] = []
    figs.append(_plot_ratio_heatmap(df, out_dir / 'ratio_heatmap.png'))
    figs.append(_plot_detection_scatter(df, out_dir / 'detection_time_comparison.png'))
    figs.append(_plot_enhancement_strip(df, out_dir / 'temperature_enhancement.png'))
    figs.append(_plot_pareto(df, out_dir / 'pareto_frontier.png'))
    return figs


def collect_files(out_dir: Path, curated_figs: List[Path]) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    collected: List[Path] = []
    root = Path.cwd()

    # Curated figures only
    for p in curated_figs:
        if p.exists():
            dest = out_dir / p.name
            if p.resolve() != dest.resolve():
                shutil.copy2(p, dest)
            collected.append(dest)

    # Data
    data_src = root / 'results' / 'hybrid_sweep.csv'
    if data_src.exists():
        dest = out_dir / data_src.name
        shutil.copy2(data_src, dest)
        collected.append(dest)

    # Docs
    for doc in [
        root / 'docs' / 'Reproducibility.md',
        root / 'docs' / 'DatasetNotes.md',
        root / 'docs' / 'Glossary.md',
        root / 'docs' / 'FAQ.md',
        root / 'docs' / 'Limitations.md',
        root / 'docs' / 'Validation.md',
    ]:
        if doc.exists():
            dest = out_dir / doc.name
            shutil.copy2(doc, dest)
            collected.append(dest)

    # Legal/citation
    for meta in [root / 'CITATION.cff', root / 'LICENSE']:
        if meta.exists():
            dest = out_dir / meta.name
            shutil.copy2(meta, dest)
            collected.append(dest)

    return collected


def main() -> int:
    root = Path.cwd()
    csv_path = root / "results" / "hybrid_sweep.csv"
    if not csv_path.exists():
        print("results/hybrid_sweep.csv not found. Run 'make comprehensive' first.")
        return 1

    df = pd.read_csv(csv_path)
    summary = compute_summary(df)

    pack_dir = root / "results" / "results_pack"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    pack_dir.mkdir(parents=True)

    # Curated figures
    curated = generate_curated_figures(df, pack_dir)

    # Write summary and collect curated files + data + docs
    write_summary_md(pack_dir, summary)
    collected = collect_files(pack_dir, curated)

    # Write a machine-readable JSON summary
    (pack_dir / "pack_summary.json").write_text(json.dumps(summary.__dict__, indent=2), encoding="utf-8")

    # Create ZIP
    zip_path = root / "results" / "results_pack.zip"
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for p in pack_dir.rglob('*'):
            zf.write(p, p.relative_to(pack_dir.parent))

    print(f"Results pack created: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
