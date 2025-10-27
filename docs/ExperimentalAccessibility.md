# Experimental Accessibility Analysis

This note addresses the immediate follow-up action highlighted in `docs/TECHNICAL_REVIEW_2025.md` to quantify how the repository's theoretical optima compare with realistic experimental capabilities. Values are pulled from existing analysis notebooks and re-expressed in common experimental units so the gap is explicit.

## Parameter Comparison

| Metric | Theoretical optimum (GradientCatastropheAnalysis v0.3.0) | Current facility capability (Apollon / ELI / BELLA) | Practical sweet spot from repo guidance | Notes |
| --- | --- | --- | --- | --- |
| Peak laser intensity | 5.7e50 W/m^2 (5.7e46 W/cm^2) with a0 ~ 1.6 | ~1e27 W/m^2 (1e23 W/cm^2) demonstrated; next-gen aims for 1e28 W/m^2 | 1.1e22 W/m^2 (1.1e18 W/cm^2) from `docs/Results.md` Bayesian map | Optimum in the sweep is ~23 orders of magnitude above hardware. Even "sweet spot" guidance remains five orders above routine shots; needs staged roadmap. |
| Plasma electron density | 1.4e19 m^-3 (1.4e13 cm^-3) | Gas jets routinely deliver 1e25 m^-3; near-critical solid-density plasmas reach 1e28 m^-3 | 4.8e24 m^-3 (4.8e18 cm^-3) in high-merit region | The theoretical optimum is deep in the underdense regime. Experiments can dial down to ~1e22 m^-3, but matching 1e19 m^-3 requires specialized long gas cells. |
| Target detection time (t_5sigma) | 1.3e-13 s lower bound from kappa_max ~ 3.8e12 s^-1 | Radio radiometry at achievable intensities projects t_5sigma >> 1e6 s (see `docs/Successful_Configurations.md`) | Microsecond-scale only if signal normalization is boosted >=1e9 | Detection times are dominated by signal normalization. Improving coupling/mirror models or moving to higher-frequency diagnostics is essential. |
| Normalized laser amplitude (a0) | 1.6 at optimum configuration | 2-5 at 1e27 W/m^2 (for lambda = 0.8 um); >10 requires petawatt-class focusing | 1-3 per Bayesian guidance | Bridging intensity gap without overshooting a0 sweet spot requires shaping rather than brute-force power. |

## Key Takeaways

- The gradient-catastrophe optimum corroborates that simply increasing intensity is counter-productive: realistic facilities already overshoot the preferred a0 while still falling 20+ orders short of the nominal intensity requirement.
- Practical progress requires retuning the framework toward the 1e22-1e24 W/m^2 band where present lasers operate, then revisiting surface-gravity and detection estimates under those constraints.
- Detection feasibility is currently limited by the radiometer normalization (see outlandish `t_5sigma` values in `docs/Successful_Configurations.md`); upgrading spectra normalization and coupling assumptions should be prioritized alongside experimental outreach.

## Suggested Next Actions

1. Add parameterized configuration sets that clamp intensity to <=1e24 W/m^2 and density to 1e23-1e25 m^-3, then re-run the guidance map to provide realistic targets.
2. Introduce an "achievable vs. theoretical" flag in horizon and detection summaries so downstream plots clearly separate the two regimes.
3. Pair this accessibility table with draft emails/briefing material for prospective laser-facility partners (AnaBHEL, ELI, BELLA) as suggested in `docs/TECHNICAL_REVIEW_2025.md`.
