# PC CUDA Workflow Roadmap

This note distills the cross-chat guidance (ChatGPT + Gemini) into a concrete plan for running the analog Hawking pipeline on a CUDA-capable Windows PC (with optional WSL2). It focuses on GPU-enabled Particle-in-Cell (PIC) production, rapid ingestion, and the analysis extensions requested in Phases 4–6.

## What’s Already in Place

- **Core pipeline**: exact acoustic `κ` finder, acoustic-WKB graybody transmission, Hawking PSD + 5σ detection metrics, JSON/plot outputs.[^github]
- **PIC/openPMD ingestion**: utilities to ingest WarpX/openPMD profiles, run universality tests, and drive experiments through the orchestration engine (exploration → refinement → Bayesian optimization → validation).[^github]
- **Roadmap context**: calls for the WarpX execution layer, fluctuation injector, trans-Planckian workflow, and correlation diagnostics.

## Flagship Objective

Deliver a **GPU PIC-to-Hawking pipeline with κ-inference and horizon-crossing correlations**:

1. Run a minimal WarpX down-ramp simulation on the PC GPU and export openPMD data.
2. Ingest the series into this repo to produce graybody-filtered spectra, Bayesian `κ` posteriors, and Hawking–partner correlation maps.
3. Package outputs into an artifact bundle suitable for publication or collaboration.

## Phase 0 — PC CUDA Environment

- **Windows 11 + WSL2 (recommended)**: enable WSL2, install Ubuntu, and follow NVIDIA’s CUDA-on-WSL instructions. Do **not** install Linux GPU drivers inside WSL; the Windows host driver is sufficient.[^cuda_wsl]
- **Python environment**:
  ```bash
  python -m venv .venv
  source .venv/bin/activate
  pip install -e .
  pip install scikit-optimize==0.8.1 seaborn
  ```
- **GPU array support**: install CuPy matching the host CUDA version (`pip install cupy-cuda12x` or `conda install -c conda-forge cupy`).[^cupy]
- **openPMD I/O**: `pip install openpmd-api` (or Conda equivalent).[^openpmd]
- **Sanity check**: run `pytest -q` once optional plotting deps are present.

## Current PC Run Notes (2025-10-26)

- Had to rely on the system interpreter because `python3 -m venv` is blocked without `python3.12-venv`; installed the project with `pip3 install --user --break-system-packages -e .` and added `cupy-cuda12x` plus `openpmd-api`.
- GPU bindings validated locally (`python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"` → `1`), confirming the RTX 3080 is visible through WSL.
- Demo pipeline executed with `python3 scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb --alpha-gray 0.8`; first horizon κ ≈ `3.15×10^14 s⁻¹` and `t_5σ ≈ 2.5×10^-10 s` are recorded in `results/full_pipeline_summary.json`.
- PIC workflow exercised via a synthetic profile (`results/custom_pic_profile.npz`) passed to `python3 scripts/run_pic_pipeline.py --profile … --graybody acoustic_wkb --kappa-method acoustic_exact`, yielding a single horizon at `x ≈ 5.2×10^-5 m` with κ ≈ `2.40×10^10 s⁻¹` (`results/pic_pipeline_summary.json`).
- Repository tests currently halt during collection because optional `physics_engine` support packages (`physics_engine.*`) are absent on this machine; the failure surfaces before any project code assertions run.

## Phase 1 — Generate PIC Data (WarpX)

1. Build or fetch WarpX with `USE_GPU=TRUE` and `USE_OPENPMD=TRUE`; optionally enable MPI for multi-GPU runs.[^warpx]
2. Configure a 1D or RZ laser-down-ramp density profile to form a plasma mirror and trans-sonic region (AnaBHEL-style geometry).[^anabhel]
3. Emit field/particle diagnostics via openPMD (HDF5 or ADIOS2). If HDF5 locking stalls on network storage, set `HDF5_USE_FILE_LOCKING=FALSE`. [^openpmd_locking]
4. Use `scripts/warpx_runner.py` to regenerate the bundled deck (`protocols/inputs_downramp_1d.in`) and launch WarpX with consistent diagnostics:

   ```bash
   python scripts/warpx_runner.py --deck protocols/inputs_downramp_1d.in \
     --output diags/openpmd --max-step 400
   ```

   Append `--dry-run` to only emit the deck or `--force-write` to overwrite local edits.

## Phase 2 — Run the Existing Pipeline

- Convert the openPMD series using the repo’s PIC adapters; generate fluid velocity, sound-speed profiles, horizons, and exact `κ` estimates.
- Run the acoustic-WKB graybody solver, produce Hawking PSDs, and compute `t_{5σ}` detection times.
- Use `scripts.orchestration_engine` to explore and refine around horizon-forming regimes (start from `configs/orchestration/base.yml`).

## Phase 3 — Bayesian κ-Inference

- Implement `analog_hawking/inference/kappa_mle.py` backed by `skopt` Bayesian optimization.[^skopt]
- Expose `infer_kappa(psd, model, prior) → κ̂ ± σ_κ`, persist results next to existing diagnostics, and wire into `phase_3_optimization`.

## Phase 4 — Horizon-Crossing Correlations

- Develop `scripts/correlation_map.py --series <openpmd> --window <...>` that:
  - extracts fluctuations around the dynamic horizon,
  - computes two-point correlation `g^{(2)}(x_1,x_2) = ⟨δn(x_1) δn(x_2)⟩`,
  - saves `g2_horizon.npz` and a PNG/Matplotlib heat map highlighting Hawking–partner bands.[^nature]
- Condition the statistics on the horizon trajectory and document pitfalls (windowing, stationarity, background subtraction).

## Phase 5 — CuPy Acceleration

- Identify hot NumPy kernels (tortoise coordinate builders, WKB integrals, horizon sweeps) and refactor to accept a shared array module (`xp = cupy` when available, otherwise `numpy`).
- Maintain CPU reference paths for CI; add a marker so GPU tests run only when CuPy is present.

## Phase 6 — Reproducibility & Reporting

- Use the dashboard/report aggregator after each experiment ID; ensure optional plotting deps are installed.
- Export a minimal artifact bundle:
  1. WarpX input deck (PICMI or native),
  2. openPMD subset for the analyzed window,
  3. κ-diagnostics JSON,
  4. graybody PSDs,
  5. `g²` horizon map,
  6. universality collapse figure,
  7. κ-inference posterior/trace plots.

## Issue Backlog (PC-Friendly Cuts)

1. **WarpX GPU runner** — add `scripts/warpx_runner.py`, ship a reference deck in `protocols/`, and document CUDA runtime flags. **(M)**
2. **PIC→profile adapter** — expose `from_openpmd(series_uri, t)` returning `(v(x), c_s(x))`, uncertainties, and horizon metadata; document in `docs/AdvancedScenarios.md`. **(M)**
3. **κ-inference module** — new `analog_hawking/inference/kappa_mle.py` with `skopt` optimizer, invoked during `phase_3_optimization`. **(M)**
4. **Correlation diagnostic** — finalize `scripts/correlation_map.py`, saving `g2_horizon.npz` and export figure. **(H)**
5. **CuPy hooks** — enable array-module dispatch and mark GPU-only CI job. **(M)**
6. **Config example** — add `configs/orchestration/pic_downramp.yml` with ramp length, peak `n_e`, laser `a0`, detection bandwidth/system temperature. **(S)**
7. **Docs update** — extend `docs/Methods.md` with correlation formalism, practical pitfalls, and plasma observable mapping. **(S)**

## Suggested Command Walkthrough

1. **PIC run (inside WSL2 Ubuntu)**:
   ```bash
   warpx inputs_downramp_1d max_step=400 diag.openpmd=1
   ```
2. **Full analysis pipeline**:
   ```bash
   python scripts/run_full_pipeline.py \
     --from-openpmd ./diags/openpmd \
     --kappa-method acoustic_exact \
     --graybody acoustic_wkb \
     --alpha-gray 0.8
   ```
3. **Orchestration sweep**:
   ```bash
   python -m scripts.orchestration_engine \
     --config configs/orchestration/base.yml \
     --phases phase_1_initial_exploration phase_2_refinement
   ```
4. **κ-inference & correlations** (after implementing Phases 3–4):
   ```bash
   python scripts/infer_kappa_from_psd.py results/psd_*.npz
   python scripts/correlation_map.py --series ./diags/openpmd --t-index 350
   ```

## Stretch Goals Toward “Revolutionary”

- **Controlled dispersion / trans-Planckian study** — compare graybody spectra and correlations under explicit short-wavelength cutoffs.
- **Beyond 1D graybody** — add RZ wave-equation scatterer with PMLs to benchmark against 1D WKB results.
- **Hardware-aware detection** — incorporate instrument bandwidths/system temperatures, reporting both `t_{5σ}` and null-result power.
- **Open benchmark kit** — publish a compact openPMD dataset + scripts that reproduce universality collapse and a horizon correlation band.

## Risks & Mitigations

- **No `t_{5σ}` detection** — tighten search around higher gradients (larger `κ`), relax `alpha_gray`, widen detector bandwidth assumptions.
- **I/O contention** — disable HDF5 file locking when running on shared/networked storage.[^openpmd_locking]
- **Windows CUDA quirks** — prefer WSL2; ensure CUDA toolkit and CuPy wheels match.[^cuda_wsl]

## Why This Path Is High Leverage

The PC CUDA workflow executes the repository roadmap end-to-end: WarpX-based horizon formation, GPU-accelerated analysis, Bayesian κ recovery, and BEC-style correlation diagnostics. Achieving this closes the largest validation gap (analytic profiles → first-principles PIC) and yields a publishable figure set plus a reproducible artifact bundle—positioning the project for collaboration and experimental follow-up.

---

[^github]: Repository documentation: <https://github.com/Hmbown/analog-hawking-radiation>
[^warpx]: WarpX build documentation: <https://warpx.readthedocs.io/en/20.10/building/building.html>
[^cuda_wsl]: NVIDIA CUDA on WSL guide: <https://docs.nvidia.com/cuda/wsl-user-guide/>
[^cupy]: CuPy installation guide: <https://docs.cupy.dev/en/stable/install.html>
[^openpmd]: openPMD-api installation: <https://openpmd-api.readthedocs.io/en/latest/install/install.html>
[^openpmd_locking]: openPMD-api HDF5 backend notes: <https://openpmd-api.readthedocs.io/en/0.14.5/backends/hdf5.html>
[^anabhel]: AnaBHEL experiment overview: <https://www.mdpi.com/2304-6732/9/12/1003>
[^nature]: Observation of quantum Hawking radiation (BEC correlation reference): <https://www.nature.com/articles/nphys3863>
[^skopt]: scikit-optimize Bayesian optimization examples: <https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html>
