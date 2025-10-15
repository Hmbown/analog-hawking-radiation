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
