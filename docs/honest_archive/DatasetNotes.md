Dataset Notes & Guardrails
==========================

This repository’s analysis scripts often use `results/hybrid_sweep.csv`.

Columns (semantics)
-------------------

- `coupling_strength` – coefficient controlling hybrid coupling weight
- `D` – mirror scale parameter (m)
- `eta_a` – fixed to 1.0 in this sweep
- `w_effective` – effective coupling weight (monotone with `coupling_strength` here)
- `kappa_mirror` – mirror “surface gravity” scale (s⁻¹)
- `T_sig_fluid` – fluid baseline signal temperature (constant in this dataset)
- `T_sig_hybrid` – hybrid signal temperature (slightly varies with D)
- `t5_fluid` – fluid 5σ detection time (constant)
- `t5_hybrid` – hybrid 5σ detection time
- `ratio_fluid_over_hybrid` – `t5_fluid / t5_hybrid`

Why some correlations look perfect
----------------------------------

- `T_sig_fluid` and `t5_fluid` are constant → correlations with them are undefined/NaN or misleading.
- `w_effective` is a direct monotone map of `coupling_strength` in this sweep → r ≈ 1.
- `ratio_fluid_over_hybrid` shares `t5_hybrid` in the denominator and a constant numerator → very strong relationships by construction.

How to read the plots
---------------------

- Correlation heatmaps exclude zero‑variance columns to avoid NaNs.
- Flat scaling with `coupling_strength` is dataset‑specific. Do not generalize beyond this sweep.
- 4× temperature and 16× speed improvements follow the radiometer equation; present as model‑dependent.

Reproducibility
---------------

- Run `make comprehensive` to regenerate all analyses and figures.
- Run `make results-pack` to create `results/results_pack.zip` for sharing.
