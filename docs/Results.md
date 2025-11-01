Results Summary
===============

Plain‑language summary
----------------------

The hybrid scenario tends to raise the effective “signal temperature,” which—by a standard radio formula—shortens the time needed to reach a confident detection. In this dataset that averages to ~4× hotter and ~16× faster than the fluid baseline. These numbers are specific to how this dataset was built and the model choices used.

Gradient Enhancements (Equal Total Power; Coarse-Grained)
---------------------------------------------------------

Under physically motivated coarse-graining (≈ skin depth), simulated multi-beam geometries yield modest enhancements relative to a single-beam baseline. Naive "multiply by N" factors are not supported by the physics.

See `figures/enhancement_bar.png` and `results/enhancement_stats.json` for detailed results.

Radio SNR
---------

From the quantum field theory spectrum at low Hawking temperatures (radio-band), radiometer sweeps provide time-to-5σ as a function of system temperature and bandwidth. This analysis demonstrates the practical detection requirements for observing analog Hawking radiation signatures.

See `figures/radio_snr_from_qft.png` for the detection time heatmap.

Guidance Maps and δ-Matched Geometries
--------------------------------------

The framework provides guidance maps for experimental optimization:

* `figures/match_delta_geometries.png`: Density-dependent small-angle matching (Λ≈δ) and corresponding enhancement trends
* `figures/bayesian_guidance_map.png`: Surrogate "where to look" map combining envelope-scale matching with radiometer feasibility (score ∝ 1/t_5σ)

Horizon Presence Sweep
----------------------

A comprehensive parameter sweep demonstrates the critical bottleneck in analog Hawking radiation experiments:

* `figures/horizon_analysis_probability_map.png`: Presence/absence probability across (density, temperature) under a fixed intensity and realistic c_s = √(γ k T_e / m_i)

This analysis emphasizes that horizon formation—not detection—remains the critical bottleneck for experimental success.

Impact of Sound Speed Profile on Horizon Formation
--------------------------------------------------

The model incorporates realistic sound speed profiles to improve horizon prediction accuracy:

* A non-uniform temperature profile, induced by laser heating, creates a position-dependent sound speed `c_s(x)`
* This significantly shifts the locations where the horizon condition `|v(x)| = c_s(x)` is met compared to a constant sound speed assumption
* The updated model captures this critical effect, providing more accurate predictions for where horizons are likely to form

See `figures/cs_profile_impact.png` for visualization of this effect.

Bayesian Optimization for Hawking Radiation Detection
-----------------------------------------------------

A Bayesian optimization framework was developed to efficiently search the multi-dimensional parameter space and identify the experimental conditions most likely to produce a detectable Hawking signal.

* **Merit Function**: The optimizer maximizes a unified merit function defined as `Merit = P_horizon * E[SNR]`, which balances the probability of forming a horizon with the expected signal-to-noise ratio
* **Optimization Results**: The search identified a high-merit region with the following parameters:
  - Plasma Density: 4.758e+24 m⁻³
  - Laser Intensity: 1.152e+22 W/m²
  - Peak Temperature: 1.115e+07 K
* **Experimental Guidance**: The results are summarized in a guidance map, `figures/optimal_glow_parameters.png`, which provides a target for experimental efforts

Key Numerical Results
---------------------

### Enhancement Statistics

Quantitative enhancement factors for various beam geometries (see `results/enhancement_stats.json`):

* Single beam baseline: 1.0×
* Small-angle crossings (10°): 1.18× enhancement
* Most symmetric geometries: ~0.54-0.57× reduction
* Standing wave configurations: ~1.0× (minimal enhancement)

### Horizon Summary

Horizon formation statistics including (see `results/horizon_summary.json`):

* Position uncertainty estimates from multi-stencil finite differences
* Surface gravity (κ) calculations with error bounds
* Gradient components (dv/dx, dc_s/dx) at horizon locations

These results demonstrate that while multi-beam configurations can provide modest enhancements, the fundamental challenge remains horizon formation rather than detection.
