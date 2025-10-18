# Universality + κ-Inference Experiments

This document describes two repo-integrated experiments that probe the
universality of Hawking spectra after κ-normalization, and the identifiability
of κ from radio-band power spectra with realistic graybody transmission.

## Big Idea

Demonstrate that when frequency is normalized by the acoustic surface gravity
κ, Hawking spectra from disparate flows (analytic and PIC-derived) collapse
onto a narrow, universal band when graybody transmission is modeled with the
acoustic WKB/tortoise approach. Close the loop by inverting noisy spectra to
recover κ with calibrated uncertainty. Negative controls verify specificity.

## Core Hypotheses

- Universal shape: PSDs vs ω/κ collapse to a common curve across diverse flows.
- Identifiability: A simple Bayesian/MLE inversion recovers κ within 1σ of the
  acoustic κ at the horizon in >90% of cases.
- Robustness: Collapse and κ-inference persist under moderate profile jitter,
  magnetization, and discretization changes.
- Controls: No-horizon and white-hole profiles fail collapse and yield near-zero
  T_sig.

## Design Overview

Flow families:
- Analytic: linear, tanh, exponential, and piecewise ramp profiles (v(x), c_s(x)).
- PIC/OpenPMD: optionally convert 1D slices with `scripts/openpmd_slice_to_profile.py`.
  Then include them in the collapse via `--pic-profiles` (accepts paths or globs).

Physics settings:
- κ: computed via `acoustic_exact` in `find_horizons_with_uncertainty(...)`.
- Graybody: `acoustic_wkb` (tortoise), with α scanning for sensitivity; fallback
  `dimensionless` available for sanity checks.
- Radio band defaults: `B=100 MHz`, `T_sys=30 K`, with optional sweeps.

Normalization and targets:
- Frequency axis: ω/κ; power scaling left in physical units (area×solid-angle×coupling).
- Universality judged by RMS relative deviation from the family mean over
  ω/κ ∈ [0.2, 5].

Controls:
- No-horizon (|v| < c_s everywhere) and white-hole (wrong gradient sign) profiles
  do not collapse and yield near-zero T_sig.

κ-inference benchmark:
- Simulate PSDs from κ_true with chosen graybody; add multiplicative noise to
  mimic measurement uncertainty; invert for κ (and α) by grid-search MLE in
  log-PSD space; report parity, residuals, and coverage.

## Scripts

- `scripts/experiment_universality_collapse.py`
  - Generates families, computes κ and spectra (acoustic_WKB), produces ω/κ
    collapse plots and RMS deviation stats.
  - Outputs JSON summary at `results/experiments/universality/` with figures.

- `scripts/experiment_kappa_inference_benchmark.py`
  - Simulates PSDs, runs κ inversion (grid-search MLE), reports parity plot,
    median relative error, and 1σ coverage.

- `scripts/experiment_hybrid_onoff.py` (optional)
  - Scans coupling weight for a hybrid fluid + mirror model, reporting ΔT_sig
    and Δt_5σ.

## Running

```
# Universality collapse
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --n 32 --alpha 0.8 --seed 7 --include-controls

# Include PIC/OpenPMD profiles (NPZ with x, v, c_s) in the collapse
python scripts/openpmd_slice_to_profile.py --in data/slice.h5 \
  --x-dataset /x --vel-dataset /vel --Te-dataset /Te --out results/warpx_profile.npz
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --n 16 --alpha 0.8 \
  --pic-profiles results/*.npz

# κ-inference benchmark
python scripts/experiment_kappa_inference_benchmark.py \
  --n 64 --families linear tanh ramp --noise 0.08 --out results/experiments/kappa_inference
```

## Acceptance Criteria

- Universality: normalized PSDs across ≥4 families satisfy RMS deviation <10%
  over ω/κ ∈ [0.2, 5].
- Identifiability: median relative error <10%, and >90% coverage within 1σ.
- Robustness: collapse persists under ±5–10% profile/discretization jitter and
  with magnetization enabled (as available in inputs).
- Controls: no-horizon/white-hole fail collapse; T_sig consistent with zero.

## Reproducibility

All experiments write JSON summaries with seeds and parameters into
`results/experiments/...`. Consider bundling profiles and spectra into a Zenodo
deposit for third-party replication.
