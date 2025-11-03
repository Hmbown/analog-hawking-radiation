Release Notes — v0.1.0
======================

Date: 2025-10-16

Highlights
----------
- Cleaned repository for public release; removed compiled paper bundle and external copies.
- Consolidated and clarified figure references; added image inventory.
- Preserved a lean, logical figure set for reviewers and collaborators.
- All tests pass (26/26) and core analysis scripts remain reproducible.

Changes
-------
- Ignore paper build outputs going forward (`paper/build_arxiv/`, `paper/arxiv_package.zip`).
- Remove tracked build artifacts and duplicate figure copies from `paper/build_arxiv/`.
- Remove external paper markdown copies at repository root.
- Prune a small number of unreferenced figures:
  - `figures/two_color_beat.png`
  - `figures/tau_response_sweep.png`
  - `figures/horizon_analysis_bo_convergence.png`
  - `figures/horizon_analysis_detection_time_radio.png`
- Update docs to reference only tracked/generatable figures and DOI-based references.
- Add `Makefile` target `clean-build` to purge build artifacts and logs.
- Add `docs/IMAGES_OVERVIEW.md` documenting figure generation and usage.

Integrity
---------
- Test suite: 26 tests passed locally (`pytest -q`).
- Figures regenerate via documented scripts (`scripts/README.md`, `Makefile`).
- Paper sources remain under `paper/` with selected figure copies in `paper/figures/`.

Next Steps
----------
- Make repository public on GitHub (if not already):
  - `gh repo edit --visibility public` (requires GitHub CLI auth)
- Publish GitHub Release "v0.1.0" and paste these notes.
- Optionally add issue templates and badges.
