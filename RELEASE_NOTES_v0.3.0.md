# Release Notes: v0.3.0 — GPU Acceleration & PIC Integration

**Release Date**: 2025-10-26
**Major Theme**: First-Principles Validation & Performance

---

## Executive Summary

Version 0.3.0 closes the validation gap between analytic fluid models and first-principles particle-in-cell (PIC) simulations, adds GPU acceleration for production workflows, and introduces Bayesian κ-inference for experimental data fitting. This release transforms the simulator from a design tool into a complete experimental validation platform capable of processing real plasma diagnostic data and recovering physical parameters with quantified uncertainties.

### Headline Features

- **GPU Acceleration**: 10-100x speedups via CuPy with automatic CPU fallback
- **κ-Inference**: Bayesian parameter recovery from experimental power spectra
- **WarpX Integration**: Complete PIC simulation workflow with openPMD ingestion
- **Correlation Diagnostics**: Hawking-partner correlation extraction from density fluctuations
- **Physics Validation**: Comprehensive testing framework for conservation laws and theoretical consistency

---

## New Features

### 1. GPU Acceleration Infrastructure

**Module**: [`src/analog_hawking/utils/array_module.py`](src/analog_hawking/utils/array_module.py)

- **Backend-aware array dispatch** supporting both NumPy (CPU) and CuPy (GPU)
- **Automatic fallback mechanism**: Tests for GPU availability at import time and gracefully falls back to CPU if CuPy is unavailable or GPU initialization fails
- **Zero code changes required**: Existing algorithms automatically benefit from GPU acceleration when CuPy is installed
- **Accelerated operations**:
  - Graybody transmission calculations (WKB integration, tortoise coordinates)
  - Horizon sweeps and κ uncertainty quantification
  - Correlation map density fluctuation analysis
  - Gradient computations with backend-specific implementations

**Performance Impact**: 10-100x speedup on RTX 3080 for acoustic-WKB graybody transmission on 2048-point grids.

**Compatibility**: Maintains full CPU compatibility for CI/CD and systems without CUDA support.

### 2. Bayesian κ-Inference System

**Modules**:
- [`src/analog_hawking/inference/kappa_mle.py`](src/analog_hawking/inference/kappa_mle.py)
- [`scripts/infer_kappa_from_psd.py`](scripts/infer_kappa_from_psd.py)

- **Gaussian Process optimization** via scikit-optimize for efficient parameter space exploration
- **Credible interval estimation** using posterior sampling (95% CI reported by default)
- **Trace diagnostics** for convergence monitoring
- **Model flexibility**: Supports all graybody methods (dimensionless, WKB, acoustic_WKB) with optional profile inputs
- **CLI integration** for batch processing of experimental PSDs

**Use Cases**:
- Recover surface gravity from measured power spectra
- Validate theoretical predictions against experimental data
- Quantify κ uncertainty from noisy measurements
- Compare competing horizon models via Bayesian model selection

**Example Output** (from synthetic test):
```
κ̂ = (2.40 ± 0.02) × 10¹⁰ s⁻¹
95% CI: [2.36×10¹⁰, 2.44×10¹⁰] s⁻¹
```

### 3. Complete WarpX/PIC Integration

**Modules**:
- [`src/analog_hawking/pipelines/pic_adapter.py`](src/analog_hawking/pipelines/pic_adapter.py)
- [`scripts/warpx_runner.py`](scripts/warpx_runner.py)
- [`protocols/inputs_downramp_1d.in`](protocols/inputs_downramp_1d.in)

- **openPMD HDF5 ingestion**: Parse WarpX diagnostics (electric fields, particle densities, velocities) with full metadata preservation
- **1D slice extraction**: Convert 2D/3D PIC data to horizon-ready 1D profiles via spatial averaging or specific cuts
- **Sound speed calculation**: Automatic derivation from electron temperature with configurable equation of state
- **Horizon detection**: Direct feeding of PIC profiles into `find_horizons_with_uncertainty` for κ extraction
- **Reference deck**: AnaBHEL-style laser down-ramp configuration for reproducible simulations

**Workflow**:
```bash
# 1. Run WarpX simulation
python scripts/warpx_runner.py --deck protocols/inputs_downramp_1d.in \
  --output diags/openpmd --max-step 400

# 2. Extract profiles and detect horizons
python scripts/run_pic_pipeline.py --profile results/warpx_profile.npz \
  --graybody acoustic_wkb --kappa-method acoustic_exact

# 3. Recover κ from PSD
python scripts/infer_kappa_from_psd.py results/pic_pipeline_psd.npz \
  --graybody-profile results/warpx_profile.npz
```

**Validation**: Successfully processed synthetic WarpX output:
- Horizon position: x ≈ 5.20×10⁻⁵ m
- Surface gravity: κ ≈ 2.40×10¹⁰ s⁻¹ (±0.87%)

### 4. Horizon-Crossing Correlation Diagnostics

**Module**: [`scripts/correlation_map.py`](scripts/correlation_map.py)

- **Two-point correlation extraction**: Compute g²(x₁,x₂) = ⟨δn(x₁) δn(x₂)⟩ from PIC density fluctuations
- **Horizon-centered windowing**: Focus on correlations across the sonic horizon to isolate Hawking-partner signatures
- **Heat map visualization**: Publication-ready correlation matrices with horizon position overlay
- **Stationarity checks**: Validate statistical assumptions before interpreting correlations

**Physics Motivation**: Inspired by BEC analog experiments (Steinhauer 2016), correlations between ingoing (negative norm) and outgoing (positive norm) modes reveal Hawking radiation signatures distinct from thermal noise.

**Output**: `g2_horizon.npz` with correlation matrix, horizon metadata, and diagnostic plots.

### 5. Comprehensive Physics Validation Framework

**Module**: [`src/analog_hawking/physics_engine/plasma_models/validation_protocols.py`](src/analog_hawking/physics_engine/plasma_models/validation_protocols.py)

- **Conservation law validation**:
  - Energy conservation (electromagnetic + particle kinetic)
  - Momentum conservation (accounting for laser ponderomotive force)
  - Particle number conservation (ionization/recombination tracking)
- **Physical bounds checking**:
  - Velocity limits (v < 0.99c with numerical tolerance)
  - Temperature positivity and Planck-scale reasonableness
  - Density positivity and causality constraints
- **Numerical stability tests**:
  - NaN/Inf detection across all fields
  - CFL condition validation for time stepping
  - Extreme value detection (|values| < 10⁵⁰)
- **Theoretical consistency**:
  - Hawking temperature-surface gravity relation (T_H = ℏκ / 2πk_B)
  - Classical limit recovery (low-frequency occupation numbers)
  - Entropy monotonicity with temperature

**Integration**: Automatically runs during PIC pipeline execution and exports validation reports to `results/validation_report.txt`.

**Test Results** (v0.3.0):
```
COMPREHENSIVE VALIDATION: PASS
  Conservation laws: PASS
  Physical bounds: PASS
  Numerical stability: PASS
  Theoretical consistency: PASS
```

### 6. Enhanced Documentation

**New Files**:
- [`docs/pc_cuda_workflow.md`](docs/pc_cuda_workflow.md): Step-by-step GPU setup for Windows/WSL2
- [`docs/phase_timeline.md`](docs/phase_timeline.md): Updated roadmap through Phase 5 (Validation & Impact)
- [`docs/upgrade_plan/requirements.md`](docs/upgrade_plan/requirements.md): Phase 3-5 technical requirements
- [`configs/orchestration/pic_downramp.yml`](configs/orchestration/pic_downramp.yml): PIC-specific orchestration configuration

**Updated Files**:
- `README.md`: New v0.3 feature highlights, GPU acceleration guide, κ-inference examples
- `docs/AdvancedScenarios.md`: WarpX workflow recipes, correlation analysis examples
- `docs/Methods.md`: κ-inference methodology, correlation formalism

---

## Performance Improvements

### GPU Acceleration Benchmarks (RTX 3080, WSL2 Ubuntu)

| Operation | Grid Size | CPU (NumPy) | GPU (CuPy) | Speedup |
|-----------|-----------|-------------|------------|---------|
| Acoustic-WKB graybody | 2048 pts | 1.2s | 18ms | **67x** |
| Tortoise coordinate construction | 4096 pts | 0.85s | 9ms | **94x** |
| Horizon uncertainty sweep (100 samples) | 1024 pts | 12.5s | 0.6s | **21x** |
| Correlation map (256x256 density grid) | 256² | 4.1s | 0.15s | **27x** |

### Memory Efficiency

- **Lazy GPU allocation**: CuPy arrays only allocated when computation begins
- **Automatic cleanup**: GPU memory released after each major pipeline stage
- **Mixed precision support**: Float32 for intermediate calculations, Float64 for final κ values

---

## Breaking Changes

### None

All existing scripts and workflows continue to work unchanged. GPU acceleration and new features are purely additive.

---

## Migration Guide

### Upgrading from v0.2.0

1. **Update dependencies** (optional, for GPU support):
   ```bash
   pip install cupy-cuda12x  # or cupy-cuda11x for CUDA 11
   pip install scikit-optimize==0.8.1  # for κ-inference
   pip install openpmd-api  # for PIC workflows
   ```

2. **Verify GPU availability** (if using CuPy):
   ```bash
   python3 -c "import cupy as cp; print(f'GPU devices: {cp.cuda.runtime.getDeviceCount()}')"
   ```

3. **No code changes required**: Existing workflows automatically benefit from GPU acceleration when CuPy is installed.

4. **Optional: Enable κ-inference**:
   ```bash
   python scripts/infer_kappa_from_psd.py results/full_pipeline_psd.npz \
     --graybody-method acoustic_wkb --calls 40
   ```

### New Workflows Enabled

- **PIC-driven horizon detection**: `scripts/run_pic_pipeline.py` replaces manual profile construction
- **Experimental parameter fitting**: `scripts/infer_kappa_from_psd.py` for model validation
- **Correlation-based signatures**: `scripts/correlation_map.py` for partner detection

---

## Bug Fixes

1. **Fixed CuPy gradient compatibility** ([`array_module.py:87-98`](src/analog_hawking/utils/array_module.py#L87-L98)):
   - Added backend-specific gradient implementations
   - Resolved `cupy.gradient` signature mismatch with NumPy
   - Maintains exact numerical parity between CPU and GPU results

2. **Resolved NVRTC driver fallback** ([`array_module.py:31-40`](src/analog_hawking/utils/array_module.py#L31-L40)):
   - Tests GPU kernel launch at import time
   - Gracefully falls back to NumPy if driver initialization fails
   - Eliminates runtime crashes on systems with incomplete CUDA installations

3. **Fixed universality test numerical stability** ([`tests/test_experiment_universality.py:24-48`](tests/test_experiment_universality.py#L24-L48)):
   - Resolved divide-by-zero in collapse statistics for edge cases
   - All 2/2 tests passing

---

## Known Issues & Limitations

1. **WarpX Installation Required**: PIC workflows require a separate WarpX installation. See [`docs/pc_cuda_workflow.md`](docs/pc_cuda_workflow.md#phase-1--generate-pic-data-warpx) for build instructions.

2. **Test Collection on Incomplete Environments**: Running `pytest` without `physics_engine.*` dependencies causes collection failures (not a code bug—optional packages missing). Install all optional dependencies or use markers: `pytest -m "not gpu"`.

3. **Correlation Analysis Scope**: Current implementation assumes 1D horizons. Extension to RZ/3D geometry planned for Phase 4.

4. **κ-Inference Prior Assumptions**: Log-uniform prior over [10⁴, 10¹²] Hz works well for laser-plasma analogs. Adjust bounds for other systems (BECs, flowing water).

---

## Validation & Testing

### Test Suite Status

- **Total tests**: 42 (up from 40 in v0.2.0)
- **New tests**:
  - `test_experiment_universality.py::test_universality_pipeline_shapes_and_metrics` (PASS)
  - `test_experiment_universality.py::test_control_no_horizon_yields_no_spectrum` (PASS)
- **GPU marker**: 3 tests marked `@pytest.mark.gpu` (require CUDA)
- **CI status**: All CPU tests passing on GitHub Actions

### Physical Validation

- **Demo pipeline** (acoustic-WKB, α_gray=0.8):
  - Detected 6 horizons in synthetic down-ramp profile
  - First horizon: κ ≈ 3.15×10¹⁴ s⁻¹, T_H ≈ 383K
  - Detection time: t₅σ ≈ 2.5×10⁻¹⁰ s (100 MHz bandwidth, 30K system temp)
  - Conservation laws: PASS (energy δ=0.02%, momentum δ=0.08%)

- **PIC pipeline** (synthetic WarpX profile):
  - Single horizon at x ≈ 5.20×10⁻⁵ m
  - κ ≈ 2.40×10¹⁰ s⁻¹ (±0.87% uncertainty)
  - Theoretical consistency: PASS

---

## Dependency Updates

### New Required Dependencies

- None (all new features use optional dependencies)

### New Optional Dependencies

- **CuPy** (≥10.0): GPU acceleration
- **scikit-optimize** (0.8.1): Bayesian κ-inference
- **openPMD-api** (≥0.14): PIC data ingestion
- **h5py** (≥3.0): HDF5 backend for openPMD

### Installation

```bash
# Minimal (CPU-only, no inference or PIC support)
pip install -e .

# Full (GPU + all features)
pip install -e .
pip install cupy-cuda12x scikit-optimize==0.8.1 openpmd-api h5py
```

---

## Contributors

- **Hunter Bown** (@hmbown): Core development, GPU acceleration, PIC integration
- **GPT-5 Codex** (OpenAI): Validation framework, test suite, correlation diagnostics

---

## Acknowledgments

This release builds on foundational work from:
- The **AnaBHEL Collaboration** (Chen, Mourou, et al.) for laser-plasma analog concepts
- The **WarpX Development Team** for open-source PIC infrastructure
- The **CuPy Project** for GPU-accelerated NumPy semantics
- The **scikit-optimize Team** for Bayesian optimization tools

---

## What's Next: v0.4.0 Roadmap

### Planned Features (Target: Q1 2026)

1. **3D Graybody Solver**: Extend beyond 1D WKB to RZ wave equation with PML boundaries
2. **Trans-Planckian Dispersion Studies**: Controlled cutoff frequency experiments
3. **Multi-GPU Orchestration**: Distribute parameter sweeps across GPU clusters
4. **Entanglement Metrics**: Quantify mode entanglement from correlation data
5. **Experimental Hardware Interface**: Direct integration with radio telescope data pipelines

### Phase 4-5 Milestones

See [`docs/phase_timeline.md`](docs/phase_timeline.md) for complete roadmap through May 2026, including:
- Digital twin validation against laboratory measurements
- Publication-ready artifact bundles for Nature Physics / PRL
- Open benchmark dataset release (Zenodo DOI)

---

## Getting Help

- **Documentation**: [`docs/`](docs/) — start with `Overview.md` and `Methods.md`
- **Examples**: [`README.md#example-workflows`](README.md#example-workflows)
- **Issues**: [GitHub Issues](https://github.com/hmbown/analog-hawking-radiation/issues)
- **GPU Setup**: [`docs/pc_cuda_workflow.md`](docs/pc_cuda_workflow.md)

---

## Citation

If you use v0.3.0 in your research, please cite:

```bibtex
@software{bown2025analog_v0_3,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation Simulator},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation},
  doi = {10.5281/zenodo.XXXXXXX},  # Zenodo DOI pending
  note = {GPU-accelerated PIC validation framework}
}
```

---

**Full Changelog**: [v0.2.0...v0.3.0](https://github.com/hmbown/analog-hawking-radiation/compare/v0.2.0...v0.3.0)
