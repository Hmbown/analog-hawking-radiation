# Upgrade Program Roadmap (2025-2026)

This roadmap expands the follow-up tracker into concrete work packages with suggested leads, prerequisites, tooling, and acceptance criteria. It is intended to keep the upgrade campaign moving even when different contributors own different slices.

## Phase 0 — Mobilisation (Week 0-1)

| Work package | Key tasks | Owner(s) | Dependencies | Deliverables |
| --- | --- | --- | --- | --- |
| WP-0.1 Repository hygiene | Confirm `main` is synced, cut feature branches for each workstream, ensure pre-commit hooks in place | Maintainer | None | Branch matrix, CONTRIBUTING update |
| WP-0.2 Environment parity | Finalise Docker image, publish to registry, capture SHA-pinned dependencies | DevOps | Dockerfile present | `ghcr.io/hmbown/ahr:0.3.0` image, `ENVIRONMENT.md` |
| WP-0.3 Data governance | Define storage for WarpX datasets (institutional object storage, Zenodo staging) | Maintainer + collaborator | WarpX datasets | Storage policy doc |

## Phase 1 — Validation Foundations (Week 2-6)

| Work package | Key tasks | Resources | Notes |
| --- | --- | --- | --- |
| WP-1.1 Analytical baselines | Implement three closed-form test cases (linear gradient, step horizon, square-well graybody). Add pytest fixtures and regression data. | 1 dev-week | Extends `tests/test_horizon_kappa_analytic.py` |
| WP-1.2 Conservation diagnostics | Add Poynting flux, particle number checks, dispersion benchmarks into `validation_protocols.py`. | 1 dev-week | Needs unit tests with tolerances |
| WP-1.3 Uncertainty propagation | Create Monte Carlo module to sample input distributions and propagate into κ error bands. | 1 dev-week | Align with `inference` package |
| WP-1.4 Documentation pass | Update `docs/GradientCatastropheAnalysis.md`, `Methods.md`, README to emphasise model dependence and experimental infeasibility at optima. | 3 dev-days | Include new FAQ |

## Phase 2 — WarpX Integration (Week 4-12)

| Work package | Key tasks | Resources | Dependencies | Tooling |
| --- | --- | --- | --- | --- |
| WP-2.1 Dataset acquisition | Coordinate with WarpX/AnaBHEL teams, gather openPMD HDF5 slices, document metadata | PI + collaborator | WP-0.3 | NDAs if required |
| WP-2.2 Pipeline hardening | Replace mock adapters with dataset ingestion, add CLI for dataset catalog, automate smoothing parameter selection | 2 dev-weeks | WP-1.1, WP-2.1 | Snakemake optional |
| WP-2.3 Validation study | Run fluid vs WarpX comparison suites, capture metrics, publish report in `results/validation_reports/warpx_vs_fluid.md` | 2 dev-weeks + GPU hours | HPC queue | Use cluster (NERSC/Perlmutter or local GPU) |

## Phase 3 — Reproducible Publications (Week 10-20)

| Work package | Description | Outputs |
| --- | --- | --- |
| WP-3.1 CPC manuscript | Focus on architecture, GPU acceleration, validation. Include figures from WP-2.3 and WP-1.1 | Draft, Overleaf repo |
| WP-3.2 JOSS submission | Prepare short paper, create tutorial notebooks (Colab ready), ensure tests pass in Docker | JOSS PDF, review checklist |
| WP-3.3 Data/Artifact release | Upload datasets, regression results, Docker image to Zenodo with DOI | DOI badges in README |

## Phase 4 — Community & Experiments (Month 6 onwards)

| Work package | Key goals | Stakeholders | Notes |
| --- | --- | --- | --- |
| WP-4.1 AnaBHEL collaboration | Provide code briefings, support experiment design, share validation findings | Chen/Mourou team | Set up regular sync calls |
| WP-4.2 Workshop engagement | Submit talk to analog gravity workshop, create tutorial session | Community | Reuse notebooks from WP-3.2 |
| WP-4.3 Experimental pilot | Draft beam time proposal, specify diagnostics, align KPIs with code predictions | Laser facility partners | Use accessibility table as appendix |
| WP-4.4 Advanced physics roadmap | Continue trans-Planckian studies, 3D graybody solver, community modules | Theory collaborators | Track in separate milestones |

## Tracking and Governance

- Update `docs/upgrade_plan/review_followup.md` weekly with status changes.
- Use GitHub Projects board (`Projects/Upgrade 2025`) with columns: Backlog → In Progress → Needs Review → Done.
- Require RFCs for any physics model changes (keep in `docs/rfcs/`).
- Store large datasets in external object storage; commit only manifests.

## Resource Estimates

- Development: 1.5 FTEs over 6 months (core maintainer + contributor).
- HPC: Roughly 50 GPU-hours for initial WarpX campaign; budget for more with reservation system.
- Collaboration: Expect bi-weekly coordination with experimental/theory partners.

This roadmap should evolve as tasks complete or new constraints arise; treat it as a living document.

