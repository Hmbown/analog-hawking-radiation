# External Execution Guide

Some upgrade-plan tasks cannot be completed inside the repository alone. This guide distils what is needed from external infrastructure, collaborators, or experiments so new contributors can pick them up efficiently.

## Real WarpX Validation Campaign

**Goal:** Replace mock adapters with comparisons against fully fledged WarpX particle-in-cell simulations.

1. **Data sourcing**
   - Contact: WarpX collaboration (LBNL) or AnaBHEL team (Chen/Mourou).
   - Required assets: openPMD HDF5 diagnostics containing density, velocity, field components, and metadata for laser/plasma parameters.
   - Storage: upload to agreed object store (see `implementation_roadmap.md` WP-0.3). Avoid committing raw data.

2. **Compute resources**
   - Recommended cluster: Perlmutter (NERSC), Frontier, or institutional GPU nodes with ≥40 GB VRAM.
   - Rough budget: 50 GPU-hours for baseline down-ramp campaign, plus 10 GPU-hours for parameter sweeps.
   - Queue considerations: request debugging queue access to iterate quickly before long runs.

3. **Execution steps**
   - Generate simulation inputs (WarpX `inputs` file) aligned with fluid-model parameters.
   - Run WarpX, produce slices via `diag.fields.species/lev=0`. Capture run logs for provenance.
   - Use new ingestion CLI (planned WP-2.2) to translate data into pipeline profiles.
   - Compare κ, horizon positions, and breakdown flags; record deviations in `results/validation_reports/`.

4. **Acceptance criteria**
   - κ disagreement ≤10% over regions where both models remain physical.
   - Conservation diagnostics (energy, particle number) within 2% drift.
   - Report summarising regimes of agreement and breakdown.

## Analytical Benchmarking

**Goal:** Anchor the pipeline with exact solutions to guard against regression.

1. **Cases to implement**
   - Linear velocity gradient with constant sound speed (closed-form κ).
   - Step-index profile with known transmission/reflection coefficients.
   - Square-well graybody potential (reference formulas in Macher & Parentani 2009).

2. **Resources**
   - Symbolic derivations can be performed with SymPy to generate reference data.
   - Store golden outputs in `tests/data/analytic_baselines/` as JSON/NPZ.

3. **Acceptance criteria**
   - Unit tests verifying κ and transmission within 1% of analytic values.
   - Continuous integration job exercising the cases across CPU and GPU backends.

## Experimental Outreach

**Goal:** Move from theoretical maxima to achievable experiments.

1. **Briefing packet**
   - Compose 2-page summary using `docs/ExperimentalAccessibility.md` table and gradient findings.
   - Include caveats about intensity gaps and model dependence.

2. **Targets**
   - AnaBHEL (Apollon): focus on down-ramp plasma mirror concepts.
   - ELI Beamlines: emphasise mid-term κ achievable with existing lasers.
   - BELLA Center: explore co-development with WarpX team.

3. **Interaction plan**
   - Month 1: send introductory email with GitHub highlights and Docker image link.
   - Month 2: schedule technical deep dive; capture feedback in meeting notes.
   - Month 3+: iterate on shared datasets or co-design experiments.

## Publication Preparation

1. **Computer Physics Communications**
   - Assemble validation figures (analytical baselines + WarpX study).
   - Highlight GPU acceleration and reproducibility (Docker image, CI).
   - Draft in Overleaf; circulate for review before submission.

2. **Journal of Open Source Software**
   - Prepare repository badge (DOI, CI, coverage).
   - Create demonstration notebook (Google Colab friendly) showing basic pipeline run.
   - Follow JOSS checklist; ensure reviewer instructions are clear.

## Long-Horizon Research Tasks

- **Trans-Planckian extensions:** build on `docs/trans_planckian_next_steps.md`, likely requires collaboration with analog gravity theorists (Liberati, Parentani).
- **3D graybody solver:** design finite-difference or spectral solvers; schedule separate RFC once 1D validation complete.
- **Community integration:** liaise with WarpX maintainers for possible upstream inclusion; consider creating an issue template for external contributors.

Use this guide alongside `implementation_roadmap.md` to coordinate efforts beyond the scope of routine PRs.

