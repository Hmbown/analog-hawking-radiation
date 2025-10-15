Methods
=======

Horizon Finder
--------------

The horizon finder implements robust algorithms for detecting analog event horizons in plasma flows:

* Root finding: f(x) = |v(x)| − c_s solved via bracketing and bisection methods
* Surface gravity calculation: κ = 0.5 |d/dx(|v| − c_s)| evaluated at the root
* Uncertainty quantification: Multi-stencil (±1,2,3) central differences for error estimation

Sound Speed Profile
-------------------

The sound speed `c_s` can be initialized as either a constant value or as a one-dimensional position-dependent profile `c_s(x)`. This flexibility allows for more realistic modeling where laser-plasma interactions create non-uniform temperature profiles, directly affecting the horizon condition `|v(x)| = c_s(x)`.

Multi-Beam Superposition
------------------------

The multi-beam superposition module calculates time-averaged intensity gradients from multiple coherent laser beams:

* Coherent Gaussian beams with total peak power conservation (weights sum to 1)
* Time-averaged intensity computed on a 2D grid with Gaussian kernel coarse-graining (≈ envelope/skin-depth scale)
* Gradient enhancement calculated as max |∇I| within a small radius compared to single-beam baseline (equal total power)
* Optional surface gravity surrogate via ponderomotive potential: U_p ∝ E²/ω², a ∝ −∇U_p, v ≈ a τ, κ ∝ 0.5 |∂|v|/∂x|
* Support for various geometric configurations: rings (N beams), small-angle crossings (θ), non-equal weights, lab-fixed elliptical waists, two-color beat (Δλ/λ)

Radio SNR Modeling
------------------

The detection modeling component implements radio astronomy techniques for assessing experimental feasibility:

* Radiometer equation: SNR = (T_sig/T_sys) √(B t)
* Time for 5σ detection: t_5σ = (5 T_sys / (T_sig √B))²
* Signal temperature calculation from quantum field theory spectrum via band power P_sig and T_sig = P_sig/(k B)

Bayesian-Style Guidance (Surrogate)
----------------------------------

The optimization framework provides surrogate models for efficient parameter space exploration:

* Envelope matching: enhancement peaks near Λ≈δ with κ ≈ κ0×enhancement
* Radiometer feasibility assessment using T_sig and T_sys to compute bandwidth-dependent integration times t_5σ
* Merit score calculation proportional to 1/t_5σ for optimization

WarpX Diagnostics Enhancements
------------------------------

Enhanced diagnostics capabilities have been added to the WarpX backend:

* Adaptive sigma estimation via `physics_engine/plasma_models/adaptive_sigma.py`, selecting smoothing scales using plasma lengths and κ plateaus
* `WarpXBackend` supports adaptive smoothing, exports raw versus smoothed observables, and records sigma diagnostics

Graybody Solver Integration
---------------------------

Implementation of graybody corrections for more accurate Hawking radiation modeling:

* `physics_engine/optimization/graybody_1d.py` provides WKB transmission estimates with uncertainties
* `scripts/hawking_detection_experiment.py` incorporates graybody profiles when provided (via `results/warpx_last_profile.npz`)
* `scripts/radio_snr_from_qft.py` consumes graybody-adjusted spectra when available

Magnetized Horizon Scan
-----------------------

Extension to magnetized plasma systems:

* `scripts/scan_Bfield_horizons.py` sweeps magnetic field strengths using WarpX backend and records κ statistics

Fluctuation Seeding
------------------

Enhanced fluctuation injection capabilities:

* Extended `physics_engine/plasma_models/fluctuation_injector.py` to support cadence and band-limited sampling
* Configuration stored in `configs/fluctuation_seeding.yml`
* Validation via `scripts/validate_fluctuation_statistics.py`

Testing
-------

Comprehensive testing protocols ensure reliability:

* Added `tests/test_adaptive_sigma.py` and `tests/test_graybody.py` to cover new sigma selection and graybody solver routines
* Unit tests validate core physics formulas against analytical solutions
* Integration tests verify module coupling and data flow
* Convergence tests ensure numerical stability and accuracy