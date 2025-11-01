Reproducibility Guide
=====================

This guide shows how to reproduce the core analysis and build a shareable results pack, fully offline.

Environment
-----------

- Python 3.9–3.11 (tested)
- Recommended: a virtual environment

Setup
-----

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the comprehensive bundle
----------------------------

```bash
make comprehensive
```

This executes, in order:

1) `python comprehensive_analysis.py` – descriptive stats, correlations, model comparison, scaling
2) `python optimization_analysis.py` – Pareto + weighted multi‑objective + visualizations

Create a shareable results pack
-------------------------------

```bash
make results-pack
```

This creates `results/results_pack.zip` containing figures, data, a summary, and key docs.

Inputs and outputs
------------------

- Input CSV: `results/hybrid_sweep.csv`
- Figures and reports: `results/analysis/*`

Tips
----

- If you modify the dataset, re‑run `make comprehensive` to refresh plots and HTML.
- For non‑interactive environments, copy the HTML to your workstation and open locally.
