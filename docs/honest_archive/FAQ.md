FAQ
===

- What is analog Hawking radiation, in one sentence?
  - A thermal‑like signal predicted to occur at “sonic horizons” in certain flowing media, analogous to black hole horizons.

- Does this code prove Hawking radiation exists in the lab?
  - No. It provides physically motivated models, validation checks, and detection forecasts to plan and interpret experiments.

- Why do some correlations look perfect (r ≈ 1)?
  - The dataset includes constants (fluid baseline) and derived quantities (ratios), which mathematically force near‑perfect correlations. These are “by construction,” not new physics.

- What does “4× temperature and 16× faster detection” mean?
  - In this dataset, the hybrid model’s average signal temperature is ~4× the fluid baseline. By the radiometer equation, detection time scales like 1/T², so time shortens by ~16×. This is model‑ and dataset‑dependent.

- Why is scaling with coupling_strength flat?
  - In this particular grid, other parameters dominate the variation; fluid baselines are fixed. The result is not a general law.

- What is κ (surface gravity) physically?
  - It’s the near‑horizon gradient scale that sets the thermal temperature T_H = ħ κ / (2π k_B).

- How do I share results with collaborators?
  - Run `make comprehensive && make results-pack` and share `results/results_pack.zip` (contains figures, data, and a summary).

- What Python version and dependencies should I use?
  - Python 3.9–3.11. Install with `pip install -r requirements.txt`. Then run `make comprehensive`.
