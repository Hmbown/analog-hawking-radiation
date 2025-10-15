<!-- 6a640fea-aeae-40af-845f-d4fdb52b94f2 0e6ad046-999a-4e2b-948c-0cb04f741daf -->
# Roadmap: WarpX Diagnostics & Post-Processing

## Phase 1 – WarpX Reduced Diagnostics

- Define config schema for `field_getters`/`moment_getters` pointing to WarpX Python API or openPMD readers.
- Implement data adapters in `warpx_backend.py` to load density, velocity, temperature arrays per species.
- Verify adaptive σ + horizon outputs on sample WarpX timestep; capture sidecar JSON and quick plots.

## Phase 2 – Graybody & Radiometer from PIC Profiles

- Extract v(x), c_s(x) near horizons from the new diagnostics and persist profile snapshots (`results/warpx_profiles.npz`).
- Feed those profiles through `compute_graybody` and update `scripts/radio_snr_from_qft.py` to consume live transmissions.
- Regenerate SNR figures and document graybody impact.

## Phase 3 – Fluctuation & Magnetized Validation

- Hook fluctuation injector into WarpX runs; add config plumbing plus PSD validation script updates.
- Execute magnetic-field sweep via `scan_Bfield_horizons.py` on WarpX data and produce κ(B) figures.
- Summarize findings, update docs (Methods/Results), and list remaining risks.

### To-dos

- [ ] Integrate WarpX reduced diagnostics into backend and validate horizon sidecars.
- [ ] Pipe WarpX horizon profiles into graybody solver and refresh SNR analysis.
- [ ] Validate fluctuation injector within WarpX runs and execute B-field sweep outputs.