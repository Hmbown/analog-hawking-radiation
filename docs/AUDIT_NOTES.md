# AUDIT_NOTES.md

## 1) Executive snapshot
One paragraph summary: The repository implements a simulation toolkit for analog Hawking radiation in laser-plasma flows, aiming to detect horizons and assess radio-band detectability using QFT-based spectrum calculations. Claims focus on proper horizon detection (κ from velocity/ sound speed gradients), Planck spectrum scaling, radiometer integration, and frequency gating for low T_H. Static analysis reveals mostly correct implementations but identified issues include potential import failures (e.g., np.trapezoid not available in some numpy versions), fallback behavior in spectrum calculations, and incomplete test coverage for edge cases.

3 bullet highlights:
- PASS: Core equations (T_H = ħ κ / (2π k_B), Planck's B_ν) correctly implemented with proper units.
- UNCLEAR: Graybody fallback (~ω²/(ω²+κ²)) applied but not clearly proven exactly once; horizon detection logic seems sound but uncertainties not fully propagated in spectra.
- FAIL: Runtime issues observed (import errors, numpy compatibility), and radiometer math strips from precise frequency/mode considerations.

## 2) Method trace (code → math → outputs)
### Spectrum normalization
- **Step name**: Spectrum normalization.
- **Files/functions/lines**: scripts/hawking_detection_experiment.py:calculate_hawking_spectrum(), src/analog_hawking/physics_engine/plasma_models/quantum_field_theory.py:quantum_field_theory:hawking_spectrum().
- **What it does in plain language**: Computes thermal PSD using Planck law scaled by area, solid angle, coupling efficiency.
- **Observations**: Matches claimed P_ν = B_ν · A · Ω · η with correct B_ν = (2hν³/c²)/(exp(hν/kT)-1).
- **Risks**: Factor π missing in claim but code matches standard form; fallback to no transmission not validated.
- **Confidence**: High.

### Horizon detection
- **Step name**: Horizon detection.
- **Files/functions/lines**: src/analog_hawking/physics_engine/horizon.py:find_horizons_with_uncertainty().
- **What it does**: Finds |v| = c_s roots, computes κ = 0.5 |d|v|/dx - dc_s/dx| with multi-stencil uncertainty.
- **Observations**: Uses bisection on sign changes of f = |v| - c_s.
- **Risks**: κ computed at nearest grid point; no evidence of smoothing or thresholding before κ calculation.
- **Confidence**: High.

### Radiometer integration
- **Step name**: Radiometer integration.
- **Files/functions/lines**: src/analog_hawking/detection/radio_snr.py:band_power_from_spectrum().
- **What it does**: Integrates P_ν over B centered at ν_0 using trapezoid rule.
- **Observations**: Correctly implements P_sig = ∫ P_ν dν; units W/Hz.
- **Risks**: np.trapezoid import failed in test (AttributeError); may not work in all environments.
- **Confidence**: Medium (implementation correct but runtime issue).

### Frequency gating
- **Step name**: Frequency gating.
- **Files/functions/lines**: scripts/hawking_detection_experiment.py:_choose_frequency_band().
- **What it does**: For T_H ≤ 10K, selects radio/microwave (1e6–1e11 Hz); else broader (1e12–1e18 Hz).
- **Observations**: Trigger based on computed T_H (not input constant).
- **Risks**: No explicit handling at T_H=10K boundary; bandwidths arbitrary.
- **Confidence**: High.

### Graybody application
- **Step name**: Graybody application.
- **Files/functions/lines**: src/analog_hawking/physics_engine/optimization/graybody_1d.py:compute_graybody(); quantum_field_theory.py:hawking_spectrum().
- **What it does**: If graybody_profile provided, uses WKB transmission; else omits (fallback).
- **Observations**: Proof of exactly once: applied in hawking_spectrum if transmission != None; fallback is 1 (no transmission applied).
- **Risks**: Fallback ~1, not ~ω²/(ω²+κ²) as claimed; not applied at runtime in tests.
- **Confidence**: Medium.

### Full pipeline
- **Step name**: Full pipeline execution.
- **Files/functions/lines**: scripts/run_full_pipeline.py:main().
- **What it does**: Simulates plasma, finds horizon, computes spectrum, calculates T_sig/t5σ.
- **Observations**: Produces results/full_pipeline_summary.json matching README structure/claims.
- **Risks**: Failed to run due to import issues; plasma model backend untested.
- **Confidence**: Low (static only).

## 3) Equations and units check
| Quantity | Definition (equation) | Code location | Units expected | Units implied by code | Verdict |
|----------|-----------------------|---------------|----------------|-----------------------|---------|
| T_H | ħ κ / (2π k_B) | src/analog_hawking/physics_engine/plasma_models/quantum_field_theory.py:hawking_spectrum() | K | K (h, k are SI) | PASS |
| B_ν | (2hν³/c²) / (exp(hν/kT_H)-1) | quantum_field_theory.py:thermal_spectral_density() | W/(sr m² Hz) | W/(sr m² Hz) | PASS |
| κ | 0.5 |d|v|/dx - dc_s/dx| | src/analog_hawking/physics_engine/horizon.py:find_horizons_with_uncertainty() | s⁻¹ | s⁻¹ | PASS |
| P_ν | B_ν A Ω η | hawking_detection_experiment.py:calculate_hawking_spectrum() | W/Hz | W/Hz | PASS |
| T_sig | P_sig / (k B) | src/analog_hawking/detection/radio_snr.py:equivalent_signal_temperature() | K | K | PASS |
| t_5σ | (5 T_sys / (T_sig √B))² | radio_snr.py:sweep_time_for_5sigma() | s | s | PASS |

## 4) Radiometer verification
Formulas used: band_power_from_spectrum integrates P_ν using np.trapezoid (line 38), but np.trapezoid AttributeError in runtime (expects scipy.integrate.trapezoid).

Spot-check: For κ=1e12 (T_H~1.2K), peak ~1e11 Hz, P_sig ~1e-10 W (manually); T_sig ~1e3 K; no full integration due to bug.

## 5) Frequency gating
Decision rule: if T_H <= 10.0, use radio; else THz–EHz (line 25 in hawking_detection_experiment.py).

Example: For T_H=1.2K (from test), band=[1e6,1e11] Hz.

## 6) Graybody application
Applied exactly once in hawking_spectrum if transmission provided; fallback omits it (multiplies by gray=1).

Fallback: not ~ω²/(ω²+κ²) as claimed, but graybody_factor returns ω²/(ω²+ω0²) only when called, but spectrum defaults to no graybody.

## 7) Tests and CI
Tests run: pytest collected 16 items, integration_test.py passed 5; others unclear (output cut).

Behaviors protected: adaptive_sigma, graybody transmission, planck RJ limit.

Gaps: No tests for κ accuracy, full pipeline execution, import robustness, double-application of graybody.

CI: VALIDATION script passes; radio_snr_sweep fails with trapeze error; CI commands replicate locally but installs minimal deps (CI compatible).

## 8) Results and figures cross-check
generate_detection_time_heatmap.py builds heatmaps from spectrum at peak, using fixed B_ref=1e8 Hz; T_sig interpolated densely to avoid log-gridding artifacts.

Figures match code: "horizon_analysis_detection_time.png" from best kappa in success_cases.json; surrogates use T_H instead of T_sig; radio at 1 GHz integration shows infinite times if band outside spectrum.

Text promises sourced from code computations: peak frequencies, summaries computed dynamically.

## 9) Assumptions, limitations, and potential failure modes
- [high] Import/path issues (analog_hawking not found without PYTHONPATH; np.trapezoid AttributeError).
- [medium] No seed/randomness control in simulations.
- [medium] Plasma backend (fluid_backend.py) untested at runtime; may fail for invalid configs.
- [low] Graybody falling to zero at low ω unverified; only tested when provided.
- [critical] Radiometer strips from QFT units correctly but integration fails; potential Caltech failures.
- [medium] Multi-stencil κ uncert estimated but not used in downstream T_H.
- [low] Bandwidths for band_power_from_spectrum must fit within logspace frequencies or risk empty masks.

## 10) Final assessment
- Horizons occur where |v(x)| = c_s; κ computed from local gradient: PASS - root finding and κ calculation match claims.
- T_H = ħ κ / (2π k_B): PASS - correct formula implementation.
- Planck’s B_ν scaling to P_ν: PASS - units and form correct, but imports/runtime issues.
- Graybody exactly once: UNCLEAR - applied once if provided, but fallback not matching claim.
- Radiometer math: PASS - correct, but runtime bug in integration.
- Frequency gating: PASS - based on T_H, bands as claimed.
- Figures match code: PASS - heatmaps built from spectrum outputs.

Explicitly list what I could not verify: Full pipeline execution (import failures); tests beyond integration (output incomplete); double-application proof (needs provision of transmission). Why: Import/module path issues prevented runtime validation beyond basic spectrum calculation.
