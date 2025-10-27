# Technical Review Follow-up Tracker

Status tracker for the recommendations captured in `docs/TECHNICAL_REVIEW_2025.md`. This document will evolve as each action item moves from planning to completion.

## Immediate Actions (0-2 Months)

| Task | Source reference | Status | Notes / Next steps |
| --- | --- | --- | --- |
| Add analytical validation cases (3 exact solutions) | `docs/TECHNICAL_REVIEW_2025.md:604` | Pending | Existing horizon unit tests cover linear gradients; extend to graybody and full pipeline verifications with closed-form answers. |
| Extend validation protocol with Poynting theorem checks | `docs/TECHNICAL_REVIEW_2025.md:230-260` | Pending | Incorporate energy-flux diagnostics so conservation failures are quantified, not just detected. |
| Implement explicit particle number conservation test | `docs/TECHNICAL_REVIEW_2025.md:262-280` | Pending | Add diagnostics that track particle count in fluid/PIC adapters and fail runs when drift exceeds tolerance. |
| Propagate model uncertainties into kappa error bands | `docs/TECHNICAL_REVIEW_2025.md:145-180` | Pending | Move beyond numerical stencil variance by threading upstream profile uncertainty into `find_horizons_with_uncertainty` outputs. |
| Document kappa literature alignment (Unruh 1981, Barcelo 2005) | `docs/TECHNICAL_REVIEW_2025.md:145-160` | Pending | Add inline references/comments so readers understand which definitions match the literature. |
| Run WarpX PIC simulations and compare to fluid models | `docs/TECHNICAL_REVIEW_2025.md:610` | Blocked (needs data) | Requires access to real WarpX diagnostics; identify sharable datasets or generate with facility partners. |
| Harden graybody solver docs around 1D limits and dissipation | `docs/TECHNICAL_REVIEW_2025.md:170-210` | Pending | Clarify assumptions in `docs/Methods.md` and code docstrings; outline path to 2D/3D or dissipative extensions. |
| Add explicit model-dependence caveats for kappa scaling law | `docs/TECHNICAL_REVIEW_2025.md:270-320` | Pending | Update `docs/GradientCatastropheAnalysis.md` to emphasize that inverse scaling comes from the chosen profile family and needs PIC confirmation. |
| Benchmark dispersion relations against analytical limits | `docs/TECHNICAL_REVIEW_2025.md:282-300` | Pending | Compare numerical dispersion/wave speeds to known linear solutions to validate surrogate models. |
| Quantify universality collapse significance | `docs/TECHNICAL_REVIEW_2025.md:320-360` | Pending | Report RMS deviation/confidence bands so universality claims have statistical backing. |
| Experimental accessibility analysis table | `docs/TECHNICAL_REVIEW_2025.md:617` | Completed | See `docs/ExperimentalAccessibility.md` for intensity/density/detection comparisons and follow-up actions. |
| Publish Docker container for reproducibility | `docs/TECHNICAL_REVIEW_2025.md:624` | Completed | New `Dockerfile` plus quickstart instructions in `README.md`. Consider pushing to a registry once GPU story is defined. |

## Short-Term Goals (3-6 Months)

| Task | Status | Owner / Dependencies |
| --- | --- | --- |
| Draft Computer Physics Communications manuscript | Not started | Dependent on PIC validation and analytical benchmarks. |
| Prepare parallel JOSS submission | Not started | Needs tutorial notebooks and reproducibility checklist. |
| Outreach to AnaBHEL collaboration | Not started | Package `docs/ExperimentalAccessibility.md` and gradient findings into briefing deck. |
| Community engagement (workshops, preprint) | Not started | Coordinate after manuscript outline stabilizes. |

## Long-Term Vision (6-18 Months)

| Task | Status | Notes |
| --- | --- | --- |
| Experimental validation campaign with laser facility | Not started | Track candidate facilities (AnaBHEL, ELI, BELLA); requires resource commitments. |
| Trans-Planckian physics extensions | Not started | Align with `docs/trans_planckian_next_steps.md`. |
| 3D solver and community integration | Not started | Fold into upgrade plan phases once WarpX backend is mature. |

## Recently Addressed Items

- Added `docs/ExperimentalAccessibility.md` to quantify experimental vs. theoretical parameter gaps.
- Introduced a Docker-based workflow that encapsulates runtime dependencies and enables containerized testing.
- Linked the new documentation from the README quick links so collaborators can discover the accessibility study.

## Open Questions

1. Can we source public WarpX datasets to bootstrap the validation campaign, or do we need to generate them in-house?
2. Should analytical validation target graybody transmission, full spectra, or both?
3. What signal normalization upgrades are necessary to bring `t_5sigma` estimates into physically meaningful ranges for accessible intensities?
