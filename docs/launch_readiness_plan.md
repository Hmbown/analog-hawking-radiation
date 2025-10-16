# Launch Readiness Plan

This document tracks the concrete actions required to move from the current
main branch to a launch-ready state. Tasks come directly from the validation
checklist, audit findings, and roadmap documents in this repository.

## Snapshot (updated 2025-03-21)

| Area | Task | Owner | Due | Status | Notes |
|------|------|-------|-----|--------|-------|
| Validation | Close `TESTING_PLAN.md` Phase 1–5 acceptance tests (document evidence and sign-off) | Physics Validator (A. Patel) | 2025-03-28 | In progress | Checklist copy stored in `results/testing_signoff/` once complete |
| Validation | Publish integration validation memo | Results Auditor (L. Chen) | 2025-03-31 | Not started | Derive from `integration_testing_report.md` with current data |
| Code Quality | Ensure `make validate` works in clean env (`python3`, PYTHONPATH handling) | Computational Analyst (M. Rivera) | 2025-03-21 | Complete | `Makefile` updated to use `python3`; frequency gating script injects `src/` |
| Code Quality | Replace remaining `np.trapezoid` usage with `np.trapz` | Computational Analyst (M. Rivera) | 2025-03-21 | Complete | `scripts/generate_detection_time_heatmap.py` updated |
| Pipeline | Smoke-test `scripts/run_full_pipeline.py --demo` and archive outputs | Integration Tester (AI harness) | 2025-03-21 | Complete | Artifacts in `results/full_pipeline_summary.json`; log attached |
| Docs | Update audit notes to reflect resolved integrator bug | Results Auditor (L. Chen) | 2025-03-21 | Complete | `docs/AUDIT_NOTES.md` revised |
| PIC Readiness | Implement WarpX backend data-export stubs & config harness | Plasma Backend Lead (J. Kim) | 2025-04-04 | In progress | Blocked on cluster access; see TODO in `plasma_models/warpx_backend.py` |
| PIC Readiness | Author `scripts/run_trans_planckian_experiment.py` skeleton with CLI args | Plasma Backend Lead (J. Kim) | 2025-03-29 | In progress | CLI scaffold needed before backend wiring |
| PIC Readiness | Draft spectrum analysis pipeline (`analysis/analyze_trans_planckian_spectrum.py`) | Data Analyst (S. Nguyen) | 2025-04-08 | Not started | Should ingest openPMD FFT dumps |
| Resources | Secure 8×H100/A100 multi-day allocation + 10 TB storage | Program Manager (H. Bown) | 2025-04-12 | In progress | Pending cloud vendor quote |
| Experiments | Run envelope-matched small-angle crossing study (`scripts/run_param_sweep.py` variant) | Experimental Verifier (R. Singh) | 2025-04-01 | Not started | Use coarse-grain gradient metrics for comparison |
| Experiments | Execute magnetized horizon scan & update results | Experimental Verifier (R. Singh) | 2025-04-05 | Not started | Combine with B-field sweeps in `scan_Bfield_horizons.py` |
| Experiments | Run reduced PIC/fluid cross-check, document delta vs surrogate κ | Physics Validator (A. Patel) | 2025-04-10 | Not started | Requires at least 1D PIC or fluid reference |

## Immediate Next Steps

1. **Testing evidence collation** – Export artefacts (plots, JSON outputs, logs) for each checked item in `TESTING_PLAN.md` and attach signatures from the responsible role. Store under `results/testing_signoff/`.
2. **WarpX execution layer** – Finalise backend stubs so the experiment script can issue a dry-run on development hardware; confirm diagnostic schema and storage paths.
3. **Resource procurement** – Lock-in GPU quota and shared storage so large-scale PIC runs are unblocked before mid-April.
4. **Experimental reruns** – Prioritise envelope-matched geometry sweep and magnetised horizon scan so documentation reflects post-fix numbers.

## Communication Cadence

- **Twice weekly stand-up** (Tue/Fri) focused on validation and PIC readiness blockers.
- **Audit review** (weekly, Mondays) to update risk log and close findings.
- **Launch readiness checkpoint** (2025-04-12) deciding on go/no-go based on validation evidence and PIC progress.

For updates, append meeting notes to `docs/launch_readiness_plan.md` and link supporting artefacts from the `results/` directory.
