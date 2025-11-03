# PIC Migration Roadmap

## 1. Module Impact Survey

- `physics_engine/plasma_models/plasma_physics.py`: replaces fluid solver core with a backend interface.
- `physics_engine/horizon.py`: must consume backend-agnostic flow and sound-speed data.
- `physics_engine/multi_beam_superposition.py`: feeds laser drivers; needs hooks for particle/field deposition.
- `physics_engine/optimization/` package: updates Bayesian loop to call backend-neutral simulation entrypoints.
- Validation scripts (`physics_validation.py`, `convergence_testing.py`, `benchmark_testing.py`): require new baselines and regression targets.
- Downstream analyses in `scripts/` (e.g., `analyze_cs_profile_impact.py`): expect identical data schema; ensure adapters translate PIC outputs.

## 2. PIC Library Evaluation

| Library | Language/API | License | GPU Support | Diagnostics | Notes |
|---------|--------------|---------|-------------|-------------|-------|
| WarpX | C++/Fortran core with Python + PICMI bindings | BSD-3 | CUDA, HIP, OpenMP | Built-in field/particle diagnostics, back-end for openPMD | Strong DOE support, scalable; Python interface simplifies integration.
| Smilei | C++ core with Python steering | CeCILL-B (BSD-compatible) | OpenMP, MPI; GPU via pragma loop | Extensive probing, synthetic diagnostics | Flexible scripting, but GPU acceleration still maturing.
| PIConGPU | CUDA/C++ | GPLv3 | CUDA-centric | Rich diagnostics but heavier build | Requires tighter GPU environment; GPL may complicate redistribution.

Recommendation: Integrate WarpX first due to permissive license, mature Python API, and robust GPU/CPU support. Smilei remains a candidate for cross-validation once infrastructure is stable.

## 3. PlasmaBackend Architecture Sketch

- Define abstract base `PlasmaBackend` with lifecycle methods:
  - `configure(run_config)` → prepare grid, species, diagnostics.
  - `step(dt)` → advance simulation, returning state summary.
  - `export_observables(requests)` → retrieve fields, particle moments, spectra.
  - `shutdown()` → finalize resources.
- Implement `FluidBackend` adapter around existing fluid solver to preserve current functionality.
- Implement `WarpXBackend` wrapping WarpX Python interface:
  - Manage PICMI input generation.
  - Register callbacks to feed laser envelopes and retrieve horizon diagnostics.
- Introduce `SimulationContext` orchestrator to mediate between laser drivers, optimizer, and backend.
- Update downstream modules to request data through backend interfaces rather than direct array access.

## 4. Quantum Fluctuation Injection Requirements

- **Noise Model**: Generate zero-mean broadband fluctuations with variance set by half-quantum per mode in chosen gauge; support particle seeding (macro-pairs) and field perturbations.
- **Implementation Hooks**:
  - Field mode injection via Fourier-space random phases before each deposition step.
  - Particle birth module creating correlated electron-ion pairs sampled from thermal vacuum spectrum.
- **Controls**: Deterministic RNG seeds for reproducible sweeps; amplitude scaling parameters exposed for optimizer tuning.
- **Validation Plan**:
  - Compare injected spectra against analytic Unruh/Hawking predictions in 1D benchmark setups.
  - Run quiet-start vs. noise-injected simulations to ensure measured radiation matches fluctuation statistics.
  - Unit tests for statistical moments and correlation lengths on extracted noise samples.

## 5. Diagnostic and Data Strategy

- **Trans-Planckian Metrics**:
  - Track instantaneous velocity gradients, local Debye length, and ratio of Hawking mode wavelength to inter-particle spacing.
  - Record emitted spectrum (power vs. frequency) per run with thermal-fit residuals.
- **Information-Paradox Proxies**:
  - Store time-resolved two-point correlations of field modes straddling the horizon.
  - Capture particle pair emission timestamps and momenta to estimate mutual information.
- **Data Handling**:
  - Standardize outputs via `openPMD` for field/particle data and attach JSON sidecars for diagnostics.
  - Implement checkpointing for long-duration runs and metadata catalogs enabling Bayesian optimizer queries.
- **Post-Processing Pipeline**:
  - Automate spectral fitting, entropy estimation, and correlation analysis within `scripts/`.
  - Maintain provenance logs linking run configuration, backend version, and diagnostic results.
