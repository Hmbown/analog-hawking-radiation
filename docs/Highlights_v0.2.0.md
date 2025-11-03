Science Highlights — v0.2.0
===========================

This short note summarizes the physically meaningful improvements in v0.2.0 and how they strengthen the credibility and usefulness of results.

Exact Acoustic Surface Gravity (κ_exact)
----------------------------------------

- Definition: κ = |∂x(c_s² − v²)| / (2 c_H), evaluated at the horizon (|v| = c_s). Here c_H is the sound speed on the horizon, obtained by interpolating c_s at the root.
- Why it matters: This form arises directly from the acoustic metric for 1D flows and avoids ambiguities from absolute values on |v|. It improves interpretability and consistency in the near-horizon limit.
- Diagnostics: The pipeline now exports horizon-side quantities c_H and ∂x(c_s² − v²) alongside κ and its numerical uncertainty (from multi-stencil gradients). This makes checks reproducible and transparent.

Acoustic-WKB Graybody via Tortoise Coordinate
---------------------------------------------

- Construction: Define the tortoise coordinate x* by dx*/dx = 1 / |c − |v|| so that the near-horizon region is properly stretched. Build an effective, κ-scaled barrier V(x*) ∝ (α κ)² S(x*), where S encodes the profile shape through the local gap |c − |v||.
- Why it matters: This approach enforces unit consistency and reproduces the qualitative expectations for transmission across the barrier: suppression at low frequencies (ω ≪ κ) and transparency at high frequencies (ω ≫ κ).
- Robustness: The discrete WKB integral in x* is evaluated over a compact region, and small profile perturbations yield an uncertainty band for transmission. A conservative, dimensionless fallback remains available — and the two loosely agree near the turnover (ω ~ κ).

Uncertainty Propagation to Detection Metrics
-------------------------------------------

- The end-to-end pipeline propagates κ±δκ and transmission envelopes to band power and t5σ bounds. This provides traceable, conservative detectability estimates rather than single-point claims.

What to Look For in the Repo
----------------------------

- κ_exact computation and horizon diagnostics: `src/analog_hawking/physics_engine/horizon.py`
- Acoustic-WKB graybody: `src/analog_hawking/physics_engine/optimization/graybody_1d.py`
- Pipeline uncertainty propagation: `scripts/run_full_pipeline.py`
- Comparison figures (κ methods and graybody methods): `scripts/make_comparison_figures.py` and `docs/img/`

Limitations and Scope
---------------------

- The acoustic-WKB treatment is 1D and near-horizon in spirit; it does not include full multi-dimensional scattering or dissipation.
- κ uncertainties are numerical (from stencil variation) rather than physical error bars on model parameters.
- The optional hybrid (laser-plasma mirror) mode remains exploratory and is clearly labeled as such.

Bottom line: v0.2.0 prioritizes physically grounded definitions and uncertainty reporting, enabling cautious, reproducible claims while keeping conservative fallbacks in place.
