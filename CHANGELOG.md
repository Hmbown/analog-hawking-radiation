# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Documented doc-sync workflow (`python scripts/doc_sync/render_docs.py ...`) and notebook smoke-check guidance to keep researcher handoffs reproducible.

### In Progress
- Relaxing relativistic causality tolerances to treat phase-velocity ≈ c rounding as non-blocking while preserving group-velocity enforcement.
- Recalibrating ADK strong-field coefficients so validation monotonicity fully passes with benchmarked tunneling rates.

## [0.3.0] - 2025-10-26

### Added
- **GPU Acceleration Infrastructure**: CuPy backend with automatic CPU fallback
  - 10-100x speedups across core algorithms
  - Acoustic-WKB graybody: 67x speedup (1.2s → 18ms, 2048 points)
  - Tortoise coordinates: 94x speedup (0.85s → 9ms, 4096 points)
  - Horizon uncertainty: 21x speedup (12.5s → 0.6s, 100 samples)

- **Bayesian κ-Inference System**: Gaussian Process optimization for experimental parameter recovery
  - Posterior sampling with 95% credible intervals
  - CLI integration: `scripts/infer_kappa_from_psd.py`
  - Model flexibility: supports all graybody methods

- **Complete WarpX/PIC Integration**: openPMD HDF5 ingestion workflow
  - 1D profile extraction from 2D/3D volumes
  - Sound speed calculation from electron temperature
  - Horizon detection and κ extraction
  - Reference deck: `protocols/inputs_downramp_1d.in`

- **Correlation Diagnostics**: Hawking-partner correlation extraction
  - Two-point correlation g²(x₁,x₂) from density fluctuations
  - Horizon-centered windowing
  - Heat map visualization

- **Gradient Catastrophe Analysis**: Systematic parameter space mapping
  - 500 configuration sweep across (a₀, nₑ, gradient factor)
  - Physics breakdown detection (relativistic, ionization, wave breaking)
  - κ_max identification: 3.79×10¹² Hz

### Changed
- Documentation restructured with prominent research highlights
- Test suite expanded: 42 tests (up from 26 in v0.1.0, 40 in v0.2.0)
- Added 3 GPU-specific tests with `@pytest.mark.gpu` marker
- Updated CI to Python 3.9-3.11 on GitHub Actions

### Fixed
- CuPy gradient compatibility issue (backend-specific implementations)
- NVRTC driver fallback mechanism for incomplete CUDA installations
- Universality test numerical stability (divide-by-zero edge cases)

### Technical Details
- Backend-aware array dispatch supporting NumPy (CPU) and CuPy (GPU)
- Lazy GPU allocation with automatic cleanup
- Mixed precision: Float32 intermediates, Float64 final results

---

## [0.2.0] - 2025-10-18

### Added
- **Exact Acoustic κ Method**: `kappa_method="acoustic_exact"`
  - Formula: κ = |∂x(c_s² − v²)| / (2 c_H) at horizon
  - Includes diagnostics: c_H, d(c²−v²)/dx

- **Acoustic-WKB Graybody Solver**: Physically consistent implementation
  - Tortoise coordinate construction
  - κ-scaled barrier integration
  - Method: `compute_graybody(..., method="acoustic_wkb")`

- **Uncertainty Propagation**
  - κ bounds propagation through pipeline
  - Graybody transmission uncertainty
  - In-band power uncertainty bounds
  - t₅σ low/high bounds

- **PIC/OpenPMD Support**
  - `scripts/openpmd_slice_to_profile.py`: Profile creation from HDF5
  - `scripts/run_pic_pipeline.py`: End-to-end PIC analysis
  - Produces: `results/pic_pipeline_summary.json`

- **CLI Enhancements**
  - `--kappa-method`: Select κ calculation method
  - `--graybody`: Choose graybody solver
  - `--alpha-gray`: Transmission scaling parameter
  - `--bands`: Frequency band selection
  - `--Tsys`: System temperature

### Changed
- Kept `method="dimensionless"` as default graybody
- Retained `method="wkb"` for backward compatibility
- Extended analytic κ tests (linear v, linear c_s, mixed cases)
- Added acoustic-WKB validation tests

### Fixed
- PIC pipeline integration issues
- OpenPMD roundtrip validation
- Units consistency checks

---

## [0.1.0] - 2025-10-16

### Added
- Initial public release framework
- Comprehensive documentation structure
- Figure generation and inventory system
- CI/CD pipeline (GitHub Actions)
- Test suite (26 tests, all passing)

### Changed
- Repository cleanup for public release
- Removed compiled paper bundles and external copies
- Consolidated figure references
- Pruned unreferenced figures:
  - `two_color_beat.png`
  - `tau_response_sweep.png`
  - `horizon_analysis_bo_convergence.png`
  - `horizon_analysis_detection_time_radio.png`
- Updated `.gitignore` for paper build outputs

### Technical
- Added `Makefile` target `clean-build`
- Created `docs/IMAGES_OVERVIEW.md` for figure documentation
- Figures regenerate via documented scripts

---

## Repository Structure

### Current State (v0.3.0)
```
./
├── RESEARCH_HIGHLIGHTS.md       # Latest research findings
├── RESEARCH_SUMMARY_v0.3.0.md   # Detailed research summary
├── CHANGELOG.md                 # This file
├── README.md                    # Project overview
├── docs/                        # Documentation (25+ files)
│   ├── GradientCatastropheAnalysis.md
│   ├── Methods.md
│   ├── Results.md
│   └── ...
├── scripts/                     # Analysis scripts (28+ files)
├── src/analog_hawking/          # Core library
├── tests/                       # Test suite (42 tests)
├── results/                     # Sample outputs
└── configs/                     # Configuration files
```

### Key Research Findings
- **κ_max**: 3.79×10¹² Hz (fundamental limit)
- **Optimal parameters**: a₀ ≈ 1.62, nₑ ≈ 1.39×10¹⁹ m⁻³
- **Detection time**: t₅σ ≈ 10⁻⁷ to 10⁻⁶ s
- **GPU speedups**: Up to 94x for core algorithms

### Citation
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation Simulator},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation}
}
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

See [LICENSE](LICENSE) for details (MIT License).

---

*This changelog consolidates RELEASE_NOTES from v0.1.0, v0.2.0, and v0.3.0*
