Results Summary
---------------

Gradient Enhancements (equal total power; coarse-grained)
---------------------------------------------------------
Under physically motivated coarse-graining (≈ skin depth), simulated multi-beam geometries yield modest enhancements relative to a single-beam baseline. Naïve “multiply by N” factors are not supported.

See `figures/enhancement_bar.png` and `results/enhancement_stats.json`.

Radio SNR
---------
From the QFT spectrum at low T_H (radio-band), radiometer sweeps provide time-to-5σ as a function of system temperature and bandwidth. See `figures/radio_snr_from_qft.png`.

Guidance Maps and δ-Matched Geometries
--------------------------------------
- `figures/match_delta_geometries.png`: density-dependent small-angle matching (Λ≈δ) and corresponding enhancement trends.
- `figures/bayesian_guidance_map.png`: surrogate “where to look” map combining envelope-scale matching with radiometer feasibility (score ∝ 1/t_5σ).

Horizon Presence Sweep
--------------------------------------
- `figures/horizon_sweep_presence.png`: presence/absence of horizons across (density, temperature) under a fixed intensity and realistic c_s = √(γ k T_e / m_i). This emphasizes that formation—not detection—remains the critical bottleneck.

Impact of Sound Speed Profile on Horizon Formation
--------------------------------------------------
- A non-uniform temperature profile, induced by laser heating, creates a position-dependent sound speed `c_s(x)`.
- This significantly shifts the locations where the horizon condition `|v(x)| = c_s(x)` is met compared to a constant sound speed assumption.
- The updated model captures this critical effect, providing more accurate predictions for where horizons are likely to form. See `figures/cs_profile_impact.png`.

Bayesian Optimization for Hawking Radiation Detection
-----------------------------------------------------
A Bayesian optimization framework was developed to efficiently search the multi-dimensional parameter space and identify the experimental conditions most likely to produce a detectable Hawking signal.

- Merit Function: The optimizer maximizes a unified merit function defined as `Merit = P_horizon * E[SNR]`, which balances the probability of forming a horizon with the expected signal-to-noise ratio.
- The search identified a high-merit region with the following parameters:
  - Plasma Density: `4.758e+24 m⁻³`
  - Laser Intensity: `1.152e+22 W/m²`
  - Peak Temperature: `1.115e+07 K`
- Experimental Guidance: The results are summarized in a guidance map, `figures/optimal_glow_parameters.png`, which provides a target for experimental efforts.
