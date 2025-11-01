v0.2.0 (2025-10-18)

Highlights
- Physics: Added exact acoustic κ option ("acoustic_exact") and diagnostics (c_H, d(c^2−v^2)/dx).
- Graybody: New physically consistent acoustic-WKB solver with tortoise coordinate; kept dimensionless default.
- Uncertainty: Pipeline now propagates κ bounds and graybody transmission uncertainty to in-band power and t5σ bounds.
- CLI: run_full_pipeline.py gains --kappa-method, --graybody, --alpha-gray, --bands, --Tsys flags.
- PIC path: Added scripts/openpmd_slice_to_profile.py and scripts/run_pic_pipeline.py for ingestion and end-to-end analysis.

Details
- Horizon
  - Implemented kappa_method="acoustic_exact": κ = |∂x(c_s² − v²)| / (2 c_H) at the horizon.
  - HorizonResult now includes c_H and d_c2_minus_v2_dx.
  - Sidecar includes c_H and d(c^2−v^2)/dx.
- Graybody
  - compute_graybody(..., method="acoustic_wkb") builds tortoise coordinate and integrates a κ-scaled barrier.
  - Kept method="dimensionless" as default; legacy method="wkb" retained.
- Pipeline
  - Transmission uncertainty envelopes are used to form T_sig low/high and combined with κ±δκ bounds for t5σ_low/high.
  - Added CLI flags for κ method, graybody solver, α scaling, band list, and system temperature.
- PIC/OpenPMD
  - scripts/openpmd_slice_to_profile.py creates results/warpx_profile.npz from HDF5 datasets.
  - scripts/run_pic_pipeline.py produces results/pic_pipeline_summary.json and PSD plot from a profile.
- Tests
  - Extended analytic κ tests for linear v, linear c_s, and mixed cases; added acoustic-WKB tests and openPMD roundtrip.
  - Added lightweight units checks and PIC pipeline smoke tests.
- CI & Docs
  - New GitHub Actions workflow runs the full test suite on Python 3.9–3.11.
  - Added comparison figures for graybody methods and κ definitions; linked in README.

Breaking changes
- None expected; defaults remain dimensionless graybody and kappa_method="acoustic".
