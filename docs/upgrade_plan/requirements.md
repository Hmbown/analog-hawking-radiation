# Realistic Pipeline Upgrade Requirements

## Plasma Regime Targets
- Electron density range: 1e17–1e20 m^-3
- Temperature profiles: 10^3–10^6 K with spatial gradients up to 10^4 K/mm
- Velocity envelopes: subsonic-to-supersonic transitions within <10 μm, optional stochastic perturbations (≤5% rms)
- Magnetization: optional axial magnetic field 0–10 T with fast magnetosonic adjustments

## Backends and Interfaces
- Preserve `PlasmaBackend` API (`configure`, `step`, `export_observables`, `shutdown`)
- Extend configuration schema to accept:
  - Temperature profiles (`temperature_profile` callable or array)
  - Magnetic field specification (`B_field` scalar/array)
  - Imported datasets (`profile_path`) for experimental inputs
- Ensure `SimulationRunner.run_step()` continues returning `SimulationOutputs` with populated `horizons`

## Horizon and QFT Coupling
- `find_horizons_with_uncertainty()` must accept optional `sigma_cells` and magnetized sound speed inputs
- `QuantumFieldTheory` should ingest graybody transmission bundles automatically when provided
- `calculate_hawking_spectrum()` must annotate outputs with frequency band, graybody metadata, and numerical settings

## Validation Benchmarks
- Energy, particle, momentum conservation drift <2%
- Surface gravity stability: κ variance <5% across neighboring cells in steady segments
- Numerical stability: enforce CFL condition `dt < 0.9 * dx / c_fast`
- Theoretical consistency: Hawking temperature relation error <0.5%
- Cross-check against reference dataset (WarPX mock or analytic solution) stored under `results/validation_reference/`

## Detection Pipeline Enhancements
- Support receiver system temperature curves and sky background tables (CSV)
- Compute integration time grids with noise figures and bandwidth masks
- Export summary JSON including SNR, required time, and assumptions per run

## Performance Constraints
- Single run with 10k grid points completes <5 s on standard laptop (2.6 GHz quad-core)
- Memory footprint for adaptive sigma computation <500 MB

## Deliverables
- Hybrid backend implementation with modular physics components
- Validation report generator writing to `results/validation_reports/`
- Updated integration tests and documentation reflecting new capabilities

## Phased Implementation Plan

This phased plan refines the upgrade roadmap to address current limitations, progressively integrating full-fidelity simulations, 3D effects, fluctuations, experimental designs, and validation. Each phase includes rigorous testing and conservative claims, building on prior work to achieve simulated 5σ detections and publication-ready outputs.

### Phase 1: Full WarpX Integration
**Objectives:** Eliminate mock dependencies by integrating real WarpX PIC simulations via OpenPMD, enabling accurate plasma profile ingestion and horizon diagnostics. This addresses the experimental validation gap in WarpX integration.

**Key Tasks:**
- Replace mock data handling in [`src/analog_hawking/physics_engine/plasma_models/warpx_backend.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/src/analog_hawking/physics_engine/plasma_models/warpx_backend.py) with OpenPMD-standard file ingestion (using openPMD-api), extracting density, velocity, and EM fields for horizon finding.
- Implement data preprocessing for alignment with 1D surrogate assumptions, including spatial averaging and uncertainty propagation.
- Enhance error handling for incomplete or noisy PIC outputs, with fallback to fluid approximations.
- Update configuration schema to specify OpenPMD file paths and diagnostic slices.

**Validation and Testing:**
- Expand [`tests/test_warpx_openpmd_getter.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/tests/test_warpx_openpmd_getter.py) with real WarpX sample datasets (e.g., from public repositories or generated benchmarks).
- Verify conservation laws and κ estimates against analytic 1D cases, targeting <5% discrepancy.
- Performance benchmark: Ensure ingestion overhead <20% of total runtime.
- Conservative Claims: Limit to validated density regimes (1e17–1e19 cm^{-3}); document numerical biases from grid resolution.

**Timeline Estimate:** 4-6 weeks, including WarpX installation and sample data generation.

### Phase 2: 3D/Fluctuation Enhancements
**Objectives:** Extend to 3D geometries and seed vacuum fluctuations to capture multi-dimensional scattering and quantum noise, targeting enhanced κ for detectable T_H > 1 mK in GHz radio. This tackles 1D approximations, fluctuation seeding gaps, and nonlinear effects.

**Key Tasks:**
- Refactor [`src/analog_hawking/physics_engine/optimization/graybody_1d.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/src/analog_hawking/physics_engine/optimization/graybody_1d.py) to 3D by generalizing tortoise coordinates (r* = ∫ dr / |c - v|) and potential V(r) via finite-difference solvers for axisymmetric or Cartesian grids.
- Develop fluctuation seeding in [`src/analog_hawking/physics_engine/plasma_models/fluctuation_injector.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/src/analog_hawking/physics_engine/plasma_models/fluctuation_injector.py), injecting band-limited quantum noise (e.g., Ornstein-Uhlenbeck process) at sub-grid scales, coupled to QFT modes.
- Optimize plasma mirror setups (multi-beam interference) for κ boosts, using Bayesian loops to scan geometries for T_H targets.
- Integrate 3D outputs into detection pipeline, including angular resolution for radio emission.

**Validation and Testing:**
- Benchmark 3D graybody against 2D analytic solutions (e.g., spherical waves); test fluctuation statistics for Gaussianity and power spectrum match.
- Run convergence studies on grid resolution (target dx < λ/10 for GHz waves).
- Conservative Claims: Focus on near-horizon regions; quantify 3D corrections to 1D κ (<10% deviation expected); rigorous uncertainty from Monte Carlo fluctuation runs.
- Use new configs: [`configs/3d_simulation.yml`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/configs/3d_simulation.yml) for grid params, [`configs/trans_planckian_enhancements.yml`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/configs/trans_planckian_enhancements.yml) for noise injection.

**Timeline Estimate:** 6-8 weeks, dependent on Phase 1 stability.

### Phase 3: Experimental Design
**Objectives:** Bridge simulation to lab by designing executable protocols for facilities like ELI-NP or NIF, specifying hardware and diagnostics for analog horizon formation. This fills the experimental ties limitation.

**Key Tasks:**
- Expand [`protocols/experimental_protocol.md`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/protocols/experimental_protocol.md) with:
  - Laser specs: 800 nm Ti:Sapphire, 30 fs pulses, 1-10 J energy, focused to 10 μm spot for plasma channel creation.
  - Plasma targets: Gas jet (He, 10^17-10^19 cm^{-3}) or preformed underdense plasma; include B-field coils for magnetization (0-5 T).
  - Horizon formation: Counter-propagating beams for velocity gradients; monitor via pump-probe interferometry.
  - Diagnostics: GHz radio antennas (heterodyne receivers, T_sys < 50 K), streak cameras for temporal resolution, spectrometers for emission spectra.
  - Alignment/Safety: Beam pointing stability <1 μrad, radiation shielding protocols.
- Simulate protocol feasibility using Phase 2 models, forecasting SNR for 5σ detection in <1 hour integration.

**Validation and Testing:**
- Cross-validate specs against literature (e.g., ELI-NP beamline capabilities); iterate with facility experts if available.
- Conservative Claims: Assume ideal alignment; include sensitivity to misalignments (±10% κ error); prioritize non-magnetized baseline.
- Output: Step-by-step execution guide, risk assessment, and cost estimates.

**Timeline Estimate:** 4 weeks, post-simulation enhancements.

### Phase 4: Validation & Publication Prep
**Objectives:** Ensure scientific rigor by benchmarking against standards (AnaBHEL), simulating full noise, and structuring publication. This addresses all remaining gaps for conservative, validated claims toward 5σ detections.

**Key Tasks:**
- Develop benchmarking against AnaBHEL: Compare universality spectra and graybody factors in shared parameter space (e.g., sonic black holes).
- Implement noise models: Detector (T_sys, bandwidth), background (cosmic microwave), systematics (jitter, finite size effects).
- Draft paper outline:
  - Introduction: Analog Hawking in laser-plasma, motivations.
  - Methods: WarpX integration, 3D QFT, fluctuation seeding.
  - Results: Simulated spectra, κ optimizations, 5σ forecasts.
  - Discussion: Experimental roadmap, limitations (e.g., 3D assumptions).
  - Appendices: Configs, validation data, code availability.
- Generate publication assets: Figures (spectra, diagrams), tables (SNR sweeps), supplementary simulations.

**Validation and Testing:**
- Quantitative benchmarks: <5% deviation from AnaBHEL for thermal tails; full noise Monte Carlo for detection thresholds.
- Conservative Claims: Report upper/lower bounds on T_H; emphasize simulation uncertainties; no overclaims on lab detection without data.
- Final integration tests across pipeline; update docs with phase outcomes.

**Timeline Estimate:** 6 weeks, culminating in draft submission.

**Key Gaps Addressed:**
- **1D Approximations & No 3D Effects:** Phases 2-3 introduce 3D graybody and experimental 3D designs, with benchmarks ensuring minimal distortion.
- **No Full PIC (WarpX):** Phase 1 directly integrates real WarpX, validated against tests.
- **No Fluctuation Seeding:** Phase 2 implements and tests seeding, tied to trans-Planckian configs.
- **No Experimental Ties:** Phase 3 provides concrete protocols; Phase 4 validates feasibility.
- **Overall:** Phased testing prioritizes conservative claims (e.g., uncertainty bands, benchmarks); rigorous validation prevents overinterpretation.

This plan ensures incremental progress, with each phase gated by tests, toward publication-quality simulations of detectable analog Hawking radiation.
