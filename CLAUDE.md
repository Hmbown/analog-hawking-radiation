# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Computational framework for modeling analog Hawking radiation in laser-plasma systems. The framework simulates flow profiles to identify sonic horizons where `|v(x)| = c_s(x)`, computes surface gravity κ and Hawking temperature, and generates physically normalized spectra. Focus is on horizon formation as the primary experimental bottleneck, not detection optimization.

**Core Physics**: Analog black holes in laser-plasma systems where high-intensity electromagnetic fields create effective spacetime curvature analogs through ponderomotive forces and plasma heating.

## Development Commands

### Installation & Setup
```bash
# Install package in editable mode with dependencies
pip install -e .

# Install with development tools
pip install -e ".[dev]"
```

### Running Tests
```bash
# Run all tests with pytest
pytest

# Run specific test file
pytest tests/test_horizon_kappa_analytic.py

# Run tests with coverage
pytest --cov=src/analog_hawking tests/

# Run integration tests (comprehensive, longer runtime)
pytest tests/integration_test.py -v
```

### Validation & Reproducibility
```bash
# Generate key figures (radio SNR sweep, QFT spectrum, phase jitter, shapes)
make figures

# Validate frequency gating logic
make validate

# Generate enhancement summary statistics
make enhancements

# Run everything
make all
```

### Key Analysis Scripts
```bash
# Full pipeline: plasma modeling → horizon detection → QFT → radio detection
python scripts/run_full_pipeline.py --demo

# Parameter space sweeps (temperature × magnetic field)
python scripts/run_param_sweep.py --progress --progress-every 20

# Formation frontier mapping (minimum laser intensity thresholds)
python scripts/compute_formation_frontier.py

# Multi-beam geometry optimization (power-conserving)
python scripts/geometry_optimize_kappa.py

# Statistical uncertainty analysis
python scripts/monte_carlo_horizon_uncertainty.py

# Trans-Planckian regime experiments (requires WarpX)
python scripts/run_trans_planckian_experiment.py  # Currently mock mode
```

## Code Architecture

### Module Organization

**Core Physics Engine** (`src/analog_hawking/physics_engine/`)
- `horizon.py`: Horizon detection via root finding (`|v| = c_s`), surface gravity computation κ = 0.5|d/dx(|v| - c_s)|, multi-stencil uncertainty quantification
- `multi_beam_superposition.py`: Power-conserving field superposition with envelope-scale coarse-graining (avoids unrealistic optical-fringe effects)
- `simulation.py`: Top-level simulation orchestration

**Plasma Models** (`src/analog_hawking/physics_engine/plasma_models/`)
- `backend.py`: Abstract base class for plasma simulation backends
- `fluid_backend.py`: Analytic fluid model implementation (primary mode)
- `warpx_backend.py`: WarpX PIC integration (currently mock mode, needs full implementation)
- `plasma_physics.py`: Core plasma parameter calculations (ω_p, a₀, ponderomotive potential)
- `quantum_field_theory.py`: Hawking spectrum calculation from first principles
- `laser_plasma_interaction.py`: Laser-plasma coupling physics
- `adaptive_sigma.py`: Adaptive smoothing scale selection using κ-plateau diagnostics
- `fluctuation_injector.py`: Vacuum fluctuation seeding for PIC runs
- `validation_protocols.py`: Physical consistency checks

**Optimization Framework** (`src/analog_hawking/physics_engine/optimization/`)
- `merit_function.py`: Bayesian merit = P_horizon × E[SNR(T_H(κ))]
- `graybody_1d.py`: WKB-based transmission probability calculations
- `probabilistic_horizon.py`: Horizon formation probability modeling
- `snr_model.py`: Signal-to-noise ratio evaluation

**Detection** (`src/analog_hawking/detection/`)
- `radio_snr.py`: Radiometric analysis using SNR = (T_sig/T_sys)√(B·t)

### Data Flow

```
Plasma Configuration → Backend (Fluid/WarpX) → Velocity & Sound Speed Profiles
    ↓
Horizon Detection (find_horizons_with_uncertainty) → positions, κ, κ_err
    ↓
Quantum Field Theory → Hawking Spectrum (with optional graybody correction)
    ↓
Detection Modeling → Integration time for 5σ detection
```

### Key Classes & Functions

**HorizonResult** (dataclass in `horizon.py`)
- `positions`: horizon x-positions (m)
- `kappa`: surface gravity estimates (s⁻¹)
- `kappa_err`: numerical uncertainty from multi-stencil finite differences
- `dvdx`, `dcsdx`: velocity and sound speed gradients at horizons

**find_horizons_with_uncertainty(x, v, c_s, sigma_cells=None)**
- Robust root finding with bracketing and bisection
- Multi-stencil gradient calculation (stencils 1, 2, 3)
- Returns median kappa with standard deviation as uncertainty

**calculate_hawking_spectrum()** (in `quantum_field_theory.py`)
- Critical normalization parameters:
  - `emitting_area_m2`: effective emission area (default: 1×10⁻⁶ m²)
  - `solid_angle_sr`: detector solid angle (default: 5×10⁻² sr)
  - `coupling_efficiency`: system coupling (default: 0.1)
- Planck's law implementation with `expm1` for numerical stability
- Frequency gating for radio-band (low T_H) calculations

## Important Implementation Details

### NumPy Compatibility
- Uses `np.trapezoid` throughout for numerical integration (NumPy 2.x compatible)
- Prior uses of `np.trapz` have been migrated
- Code is tested with NumPy ≥1.21 and compatible with 2.x

### Numerical Stability
- Exponential overflows in Planck factors handled via `expm1` and large-argument asymptotics
- CFL-controlled time stepping for stability in fluid backend
- Adaptive smoothing prevents spurious oscillations in gradients

### Power Conservation in Multi-Beam
The multi-beam superposition enforces **total peak power normalization** across all configurations. This is physically crucial:
- Single beam at intensity I₀ has peak E-field E₀
- N-beam configuration conserves total power: each beam gets I₀/N
- Prevents unphysical N× enhancement claims from naive field addition
- Envelope-scale coarse-graining (skin-depth scale) avoids optical-fringe artifacts

### Graybody Transmission
Two modes available:
1. **Profile-derived**: Uses actual (x, v, c_s) near-horizon data for WKB calculation
2. **Fallback**: Generic transmission model when profile data unavailable
- Always verify which mode was used (check output/logs)
- Profile-derived is more accurate but requires well-resolved near-horizon region

### Horizon Detection Edge Cases
- Empty results (no horizons) are valid physical outcomes
- Multiple horizons indicate multiple sonic points (common in complex profiles)
- Uncertainty estimates are **numerical/grid-based**, not physical parameter uncertainties
- Adaptive sigma selection helps find optimal smoothing scale via κ-plateau identification

## Physical Parameters & Units

All calculations use SI units internally:
- Lengths: meters
- Velocities: m/s
- Temperatures: Kelvin
- Surface gravity κ: s⁻¹
- Hawking temperature: T_H = ħκ/(2πk_B) in Kelvin
- Laser intensity: W/m²
- Plasma density: m⁻³
- Magnetic field: Tesla

**Typical Parameter Ranges**:
- Laser intensity: 10¹⁷–10¹⁹ W/m²
- Plasma density: 10²³–10²⁵ m⁻³
- Electron temperature: 10⁵–10⁷ K
- Surface gravity κ: 10¹¹–10¹³ s⁻¹
- Hawking temperature: 1–100 K (radio/microwave regime)

## Testing & Validation

### Test Coverage
**Unit Tests** (`tests/test_*.py`):
- `test_horizon_kappa_analytic.py`: Horizon detection against analytical solutions
- `test_graybody.py`: Graybody transmission calculations
- `test_planck_rj_limit.py`: Rayleigh-Jeans limit validation
- `test_radiometer_sanity.py`: Detection time calculations
- `test_adaptive_sigma.py`: Smoothing scale selection
- `test_frequency_gating_boundary.py`: Band selection logic
- `test_graybody_fallback_and_single_application.py`: Transmission mode switching

**Integration Tests** (`tests/integration_test.py`):
- Full pipeline validation
- Cross-module data flow
- Physical consistency checks

### Validation Protocols
Located in `TESTING_PLAN.md` with role-based verification:
- Physics Validator: theoretical formulas, physical consistency
- Computational Analyst: numerical methods, algorithm correctness
- Experimental Verifier: parameter ranges, feasibility
- Results Auditor: figure/data consistency, reproducibility

## Common Development Tasks

### Adding New Plasma Backend
1. Subclass `PlasmaBackend` in `plasma_models/backend.py`
2. Implement `run_simulation()` → return (x, v, c_s, n_e, T_e)
3. Add backend selection logic in calling scripts
4. Write integration test in `tests/`

### Modifying Detection Parameters
Edit normalization in `plasma_models/quantum_field_theory.py`:
```python
emitting_area_m2 = 1e-6    # Effective emission area
solid_angle_sr = 0.05      # Detector solid angle
coupling_efficiency = 0.1   # System coupling
```

### Adding New Beam Geometry
In `multi_beam_superposition.py`:
1. Add geometry function to `create_*_beam_configs()`
2. Ensure power conservation: sum of intensities = I_total
3. Apply envelope-scale coarse-graining
4. Test gradient enhancement under power constraints

### Running Trans-Planckian Experiments
Currently requires WarpX installation (not in requirements):
```bash
# Install WarpX + pywarpx on compute cluster
# Configure in scripts/run_trans_planckian_experiment.py
# Currently runs in mock mode without real PIC backend
```

## Results & Outputs

### JSON Data Files (`results/`)
- `full_pipeline_summary.json`: Complete pipeline run results
- `horizon_summary.json`: Horizon formation statistics
- `enhancement_stats.json`: Multi-beam geometry enhancement factors
- `formation_frontier.json`: Minimum intensity thresholds vs density/temperature
- `horizon_probability_bands.json`: Monte Carlo uncertainty quantification
- `bayesian_optimization_trace.json`: Parameter optimization history

### Figures (`figures/`)
Generated by `make figures`:
- `graybody_impact.png`: Profile-derived vs fallback transmission
- `radio_snr_from_qft.png`: Time-to-5σ detection heatmap
- `formation_frontier.png`: Horizon formation parameter space
- `geometry_vs_kappa.png`: Multi-beam enhancement analysis
- `horizon_probability_bands.png`: Statistical robustness
- `phase_jitter_stability.png`: Multi-beam phase sensitivity

## Known Limitations & Future Work

### Current Limitations (see `docs/Limitations.md`)
- **No full PIC/fluid validation**: Surrogate models not yet cross-checked with full simulations
- **Coarse-graining scale**: Envelope/skin-depth assumed; actual coupling may differ
- **κ surrogate mapping**: Simple ponderomotive scaling; absolute values trend-level only
- **Uniform c_s approximation**: Real position-dependent c_s(x) profiles can shift horizons
- **Magnetized plasma**: Fast magnetosonic speed approximations need validation

### Next Steps (see README.md "Launch Next Steps")
1. Close validation checklist (`docs/launch_readiness_plan.md`)
2. Bring WarpX online (install + configure on HPC)
3. Secure multi-GPU resources (≥8×H100/A100, 10TB storage)
4. Run pre-launch studies (geometry sweeps, magnetized scans, PIC/fluid cross-checks)

## Documentation

**Primary Docs** (`docs/`):
- `Overview.md`: High-level framework description
- `Methods.md`: Detailed methodology and implementation
- `Results.md`: Key findings and numerical results
- `Limitations.md`: Current constraints and uncertainties
- `Validation.md`: Testing protocols
- `AUDIT_NOTES.md`: Detailed code review notes
- `launch_readiness_plan.md`: Pre-launch validation roadmap
- `trans_planckian_next_steps.md`: Future experimental directions

## Git Workflow Notes

- Main branch: `main`
- Version: 0.1.0 (see `setup.py`, `__init__.py`)
- License: MIT
- Repository: https://github.com/hmbown/analog-hawking-radiation
- Currently in Alpha development status
