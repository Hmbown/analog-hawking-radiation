Methods
=======

Horizon Finder
--------------

The horizon finder implements robust algorithms for detecting analog event horizons in plasma flows:

* Root finding: f(x) = |v(x)| − c_s solved via bracketing and bisection methods
* Surface gravity calculation (default): κ ≈ |∂x(c_s − |v|)| evaluated at the horizon (v≈±c_s)
  - Backward-compatible option: κ_legacy = 0.5·|∂x(|v| − c_s)|
  - Exact acoustic option: κ_exact = |∂x(c_s² − v²)|/(2 c_H) with c_H interpolated at the root
* Diagnostics exported: c_H at the horizon and ∂x(c_s² − v²) for sidecar JSONs
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

* Adaptive sigma estimation via `analog_hawking.physics_engine.plasma_models.adaptive_sigma`, selecting smoothing scales using plasma lengths and κ plateaus
* `WarpXBackend` supports adaptive smoothing, exports raw versus smoothed observables, and records sigma diagnostics

Graybody Solver Integration
---------------------------

Implementation of graybody corrections for more accurate Hawking radiation modeling:

* `analog_hawking.physics_engine.optimization.graybody_1d` now defaults to a conservative, dimensionless transmission T(ω/κ) = (ω²)/(ω² + (ακ)²). A new physically consistent acoustic-WKB option (`method="acoustic_wkb"`) constructs the tortoise coordinate via dx* = dx / |c − |v|| and evaluates a Schrödinger-like transmission with an effective potential V(x*) ~ (α κ)² S(x*), where S encodes the near-horizon profile shape. The legacy WKB routine is retained as an experimental option.
* `scripts/hawking_detection_experiment.py` incorporates graybody profiles when provided (via `results/warpx_last_profile.npz`), passing κ for consistent scaling
* `scripts/radio_snr_from_qft.py` consumes graybody-adjusted spectra when available

Magnetized Horizon Scan
-----------------------

Extension to magnetized plasma systems:

* `scripts/scan_Bfield_horizons.py` sweeps magnetic field strengths using WarpX backend and records κ statistics

Fluctuation Seeding
------------------

Enhanced fluctuation injection capabilities:

* Extended `analog_hawking.physics_engine.plasma_models.fluctuation_injector` to support cadence and band-limited sampling
* Configuration stored in `configs/fluctuation_seeding.yml`
* Validation via `scripts/validate_fluctuation_statistics.py`

Testing
-------

Comprehensive testing protocols ensure reliability:

* Added `tests/test_adaptive_sigma.py` and `tests/test_graybody.py` to cover new sigma selection and graybody solver routines
* Unit tests validate core physics formulas against analytical solutions
* Integration tests verify module coupling and data flow
* Convergence tests ensure numerical stability and accuracy

Integration Testing and Validation
---------------------------------

* `tests/integration_test.py` steps the `FluidBackend` and `WarpXBackend` mocks through the full workflow, asserting that plasma states expose density, velocity, sound speed, and grid data required by downstream modules.
* The same suite pipes horizon locations from `find_horizons_with_uncertainty()` into quantum field theory utilities and radio detection models, checking spectra, signal power, and integration time calculations.
* Error-handling coverage includes empty inputs, mismatched array lengths, and sanity checks on adaptive smoothing diagnostics to ensure graceful failure modes.
* `analog_hawking.physics_engine.plasma_models.validation_protocols` implements the `PhysicsValidationFramework`, combining conservation laws, physical bounds, numerical stability, and theoretical consistency checks for post-processing simulation outputs.

Detection Pipeline Overview
---------------------------

* Plasma solvers such as `analog_hawking.physics_engine.plasma_models.fluid_backend` produce one-dimensional profiles of density, velocity, and sound speed on configurable grids.
* `analog_hawking.physics_engine.simulation` orchestrates backend stepping, invokes `find_horizons_with_uncertainty()` to locate analog horizons, and optionally records horizon metadata sidecars.
* `scripts/hawking_detection_experiment.py` and `analog_hawking.physics_engine.plasma_models.quantum_field_theory` translate surface gravity κ into Hawking spectra, optionally folding in graybody transmission maps.
* Detection utilities in `analog_hawking.detection.radio_snr` integrate spectra over radio bands, convert to equivalent signal temperatures, and estimate integration times for 5σ detection scenarios.

Physics Engine Architecture
---------------------------

* The `SimulationRunner` coordinates any `PlasmaBackend` implementation, persisting last-step observables and exporting user-requested diagnostics.
* `PlasmaBackend` adapters (e.g., `FluidBackend`, `WarpXBackend`) encapsulate configuration, stepping, and shutdown logic while emitting standardized `PlasmaState` data structures.
* Horizon diagnostics encapsulate kappa estimates, finite-difference gradients, and optional uncertainty metrics. Three κ definitions are available: `legacy` (0.5|∂x(|v|−c)|), `acoustic` (|∂x(c−|v|)|), and the exact acoustic form `acoustic_exact` (|∂x(c²−v²)|/(2 c_H)) evaluated at the horizon. Horizon sidecars now include `c_H` and `d(c²−v²)/dx` for diagnostics.
* Validation and optimization layers consume these outputs to inform experiment design, from adaptive smoothing (`analog_hawking.physics_engine.plasma_models.adaptive_sigma`) to radio SNR feasibility analyses (`scripts/generate_radio_snr_sweep.py`).

Fluid Backend Configuration
--------------------------

The `FluidBackend` accepts the following configuration keys via `backend.configure({...})`:

- `plasma_density` (float)
- `laser_wavelength` (float, meters)
- `laser_intensity` (float, W/m²)
- `grid` (array of positions)
- `temperature_settings`:
  - `{ "constant": <K> }` or `{ "profile": <array|callable> }` or `{ "file": <path> }`
- `magnetic_field`:
  - scalar Tesla, array over `grid`, or callable `B(x)`
- `use_fast_magnetosonic` (bool): when true, `sound_speed` is set to fast magnetosonic speed
- `velocity_profile` (array or callable): overrides model velocity if provided

Example:

```python
from analog_hawking.physics_engine.plasma_models.fluid_backend import FluidBackend
import numpy as np

backend = FluidBackend()
grid = np.linspace(0.0, 50e-6, 512)
backend.configure({
    "plasma_density": 5e17,
    "laser_wavelength": 800e-9,
    "laser_intensity": 5e16,
    "grid": grid,
    "temperature_settings": {"constant": 5e5},
    "magnetic_field": 0.01,  # Tesla
    "use_fast_magnetosonic": True,
})
state = backend.step(0.0)
```

PlasmaState Fields
------------------

`analog_hawking.physics_engine.plasma_models.backend.PlasmaState` now includes:

- `density`, `velocity`, `sound_speed`, `grid` (unchanged core fields)
- `temperature` (optional array): electron temperature used to compute sound speed
- `magnetosonic_speed` (optional array): fast magnetosonic speed when B-field is provided
