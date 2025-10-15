# Pre-Submission Audit Summary

**Date:** October 15, 2025

**Manuscript:** `paper/main.tex`

**Deadline:** Tomorrow, 1pm

## Compile Status
- **Status:** Failed
- **Issue:** `pdflatex` command not found on the system.
- **Recommendation:** Install a LaTeX distribution (e.g., MacTeX for macOS) to enable compilation. Alternatively, use an online LaTeX editor like Overleaf for final compilation and PDF generation.

## Inconsistencies Found and Fixes
- **Figures:** All figures referenced in `paper/main.tex` exist in `paper/figures/` with adequate resolution (based on file size). No issues detected.
- **Physics Consistency:**
  - Hawking temperature formula `T_H = ħ κ / (2π k_B)` is consistently used in the manuscript and code.
  - Spectral density in QFT module (`quantum_field_theory.py`) is per-Hz (W/Hz), based on Planck's law scaled by area, solid angle, and coupling efficiency. Graybody factor applied exactly once.
  - Radiometer math in `hawking_detection_experiment.py` and related scripts matches the specified `T_sig = P_sig/(kB)` and `t_5σ = (5 T_sys / (T_sig sqrt(B)))^2`.
- **Normalization Constants:** Confirmed in `generate_detection_time_heatmap.py` as `emitting_area_m2=1e-6`, `solid_angle_sr=5e-2`, `coupling_efficiency=0.1`. These are clearly documented in the script.
- **Captions and Text Matching Figures:** Captions in `paper/main.tex` accurately describe the figures (PSD-based, T_H-surrogate, radio-only at 1 GHz). κ and T_H ranges in text align with figure content.
- **Citations:** All cited references in `paper/main.tex` are present in `paper/refs.bib` with correct DOIs. No dangling or duplicate entries found. `natbib` numeric style is correctly implemented.
- **Language and Formatting:** Notation (κ, T_H) is consistent. No typos or grammar issues detected. `hyperref` and `natbib` are loaded in a compatible order (`hyperref` before `natbib` with `numbers,sort&compress` option).

## Final Recommended arXiv Categories
- **Primary:** `physics.plasm-ph` (Plasma Physics)
- **Cross-List:** `gr-qc` (General Relativity and Quantum Cosmology)

## Checklist for Upload
- [ ] Install LaTeX distribution or use an online editor to compile `paper/main.tex` and generate PDF.
- [ ] Verify final PDF output for formatting issues (e.g., figure placement, citation numbering).
- [ ] Include all ancillary files under `paper/code/` and `paper/results/` in the arXiv submission.
- [ ] Confirm all figures in `paper/figures/` are embedded correctly in the PDF.
- [ ] Double-check arXiv submission categories (`physics.plasm-ph` primary, `gr-qc` cross-list).
- [ ] Submit before deadline (tomorrow, 1pm).

**Note:** No diffs are provided for `paper/main.tex` or `paper/refs.bib` as no changes were necessary based on the audit.
