# Trans-Planckian Experiment Roadmap

## Current Status

- Backend refactor complete: `PlasmaBackend` interface, `FluidBackend`, `WarpXBackend` skeleton, and quantum fluctuation injector scaffolding are in place (`physics_engine/plasma_models/`).
- Horizon diagnostics now run via `SimulationRunner`, ensuring future PIC outputs feed into `find_horizons_with_uncertainty` without additional refactors.
- `scripts/optimize_for_kappa.py` retargets the Bayesian optimizer toward maximal surface gravity; presently driven by fluid surrogates while awaiting PIC-derived κ estimates.

## Outstanding Tasks

1. **PIC Execution Layer**
   - Install WarpX/pywarpx and MPI toolchain on a multi-GPU cluster.
   - Implement field/particle extraction in `WarpXBackend` using openPMD interfaces.
   - Integrate `QuantumFluctuationInjector` with WarpX callbacks for mode seeding.

2. **High-Fidelity Experiment Script**
   - Finalize `scripts/run_trans_planckian_experiment.py` to: load optimized parameters, configure WarpX domain (≥512×256×256), enable fluctuation seeding, and register high-cadence diagnostics.
   - Establish checkpointing and streaming of diagnostics to cloud storage to handle multi-terabyte outputs.

3. **Spectrum Analysis Pipeline**
   - Implement `analysis/analyze_trans_planckian_spectrum.py` to generate Fourier spectra, compare to Hawking thermal predictions, and produce `figures/trans_planckian_spectrum_comparison.png`.
   - Validate pipeline on downsampled/analytic datasets before processing full PIC outputs.

4. **Resource Provisioning**
   - Secure access to ≥8×H100/A100 GPUs (or equivalent) with ≥200 GB aggregate RAM.
   - Allocate ≥10 TB fast storage (local SSD + bucket) for diagnostics and intermediate products.
   - Budget for multi-day runs (estimate: USD 1k–2k per full Trans-Planckian campaign on cloud GPU hardware).

## Collaboration & Support

- We are seeking collaborators with access to large-scale PIC infrastructure (WarpX, Smilei, PIConGPU) and interest in analog gravity phenomenology.
- Immediate contributions needed:
  - Providing compute time or cluster queue access for the Trans-Planckian experiment.
  - Assisting with WarpX integration (field diagnostics, particle coarse-graining, openPMD workflows).
  - Co-developing the quantum fluctuation injector to ensure physically accurate seeding.
- Potential outcomes:
  - First comparison of PIC-based analog Hawking spectra with thermal predictions at sub-Debye wavelengths.
  - Publishable insight into Trans-Planckian signatures and groundwork for analog information-paradox studies.

If your team is interested in partnering, please contact the maintainers via the repository discussion board or email listed in `README.md`. We are ready to share detailed design documents, preliminary scripts, and validation results to accelerate collaboration.

