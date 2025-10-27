Results Summary
=================

This document consolidates the main findings from the v0.3 GPU-accelerated campaigns and the supporting workflow studies. All figures and JSON artefacts live under 
esults/ with the directories noted below.

## 1. Parameter-Space Mapping (GPU Campaign)

* Coverage: 1,800 configurations across a0 in [1, 40], ne in [1e18, 1e22] m^-3, gradient factors up to 500. (
esults/gpu_rtx3080/gradient_limits_gpu/)
* Validity: 49% of samples satisfied conservation, positivity, and numerical stability checks. Invalid cases were dominated by relativistic breakdown once a0 exceeded roughly 2; gradient blow-up and negative densities were rare.
* Scaling relationships: Least-squares fits within the surrogate family yielded kappa proportional to a0^-0.2 ne^-0.05. Intensity followed the expected quadratic dependence on a0. Treat these exponents as model-specific until PIC validation is complete.
* Breakdown metrics: Relativistic thresholds (v > 0.5c) set the sharpest boundary; wave-breaking and numerical instabilities were sub-dominant in this sweep.

## 2. Spectral Universality

* Experiment: scripts/experiment_universality_collapse.py with GPU acceleration (64 flow families drawn from analytic profiles and PIC-derived surrogates).
* Outcome: Kappa-normalised graybody spectra collapsed with RMS deviation about 7e-3 over the 10^7–10^9 Hz range, indicating geometry-agnostic behaviour inside the validated regime.
* Caveat: Universality has been established experimentally in BEC systems; this result provides computational confirmation for laser-plasma surrogates pending PIC verification.

## 3. Detection and Inference Benchmarks

* Experiment: scripts/experiment_kappa_inference_benchmark.py with 128 noise realisations (
esults/gpu_rtx3080/detection_gpu/).
* Outcome: The kappa-inference workflow recovered ground-truth values with median relative error below 8% when input configurations remained in the validated regime. Performance degraded rapidly once breakdown flags appeared, underscoring the importance of upstream validation.

## 4. Experimental Accessibility

* Comparison: docs/ExperimentalAccessibility.md summarises the gap between theoretical optima and existing facilities (Apollon, ELI, BELLA). Current intensities (<=1e23 W/m^2) correspond to kappa <= 1e8 Hz in the surrogate models; higher kappa values require future hardware or alternative analogue platforms.
* Guidance: Focus near-term experimental design on the validated low-kappa region, leveraging the universality and detection analyses to prioritise diagnostics.

## 5. Legacy Workflow Highlights

The original workflow studies remain relevant for cross-checking new simulations:

* Gradient enhancements (equal total power): multi-beam configurations offer modest gains (<=20%) compared to single-beam baselines (
esults/enhancement_stats.json).
* Radio SNR estimates: radiometer sweeps map t_5sigma across bandwidth and system temperature (igures/radio_snr_from_qft.png).
* Horizon presence sweeps: parameter maps emphasise that horizon formation, not detection, is the limiting step for many setups (igures/horizon_analysis_probability_map.png).
* Impact of cs(x): spatially varying sound-speed profiles shift horizon locations and should be included in experimental modelling (igures/cs_profile_impact.png).

## 6. Data Availability

* Gradient catastrophe sweep (GPU): 
esults/gpu_rtx3080/gradient_limits_gpu/gradient_catastrophe_sweep.json
* Universality campaign: 
esults/gpu_rtx3080/universality_gpu/universality_summary.json
* Detection benchmark: 
esults/gpu_rtx3080/detection_gpu/kappa_inference_summary.json
* Supplementary guidance maps and enhancement statistics: see 
esults/ with the references above.

These summaries will be extended as new validation data (e.g., WarpX PIC runs) and experimental feedback arrive. Update this document when new campaigns conclude so collaborators can see the evolving landscape at a glance.
