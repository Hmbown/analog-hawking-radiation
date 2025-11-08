# AnaBHEL-Oriented Comparison and Usage Notes

> **Status ℹ️** Collaboration guidance mixing validated results with experimental scaffolding
> **Navigate:** [Docs Index](index.md) · [Limitations](Limitations.md)

This repository provides exploratory modeling tools for analog Hawking systems
in laser–plasma settings. For AnaBHEL researchers, this note explains how to
use the nD horizon and detection workflow and how to interpret κ values and
detection estimates in light of the literature.

Scope and assumptions
- 1D and nD flows with synthetic (tanh‑like) profiles; PIC/OpenPMD ingestion is
  supported for nD via dataset paths or NPZ conversion.
- Surface gravity evaluates the acoustic‑exact form: κ = |n·∇(c_s² − |v|²)|/(2 c_H).
- Graybody is estimated patch‑wise by sampling 1D lines and aggregating spectra;
  this is an approximation and not a full nD scattering calculation.
- Reported uncertainties mostly reflect numerical effects; physical systematics
  are not included unless you inject them explicitly.

Quick start (2D/3D)
1) Convert OpenPMD/HDF5 → grid NPZ
```bash
python scripts/openpmd_to_grid_nd.py --in sample.h5 \
  --x /mesh/x --y /mesh/y --z /mesh/z \
  --vx /fields/vx --vy /fields/vy --vz /fields/vz \
  --cs /fields/c_s --out results/grid_nd_profile.npz
```

2) Run nD horizon + detection
```bash
python scripts/run_pic_pipeline.py --nd-npz results/grid_nd_profile.npz \
  --nd-scan-axis 0 --graybody acoustic_wkb --alpha-gray 1.0 --B 1e8 --Tsys 30
```

The output includes horizon κ summary and an aggregated power spectrum with
in‑band power, equivalent signal temperature, and 5σ detection time.

Direct HDF5 mode (no NPZ)
```bash
python scripts/run_pic_pipeline.py --nd-h5-in sample.h5 \
  --x-ds /mesh/x --y-ds /mesh/y --z-ds /mesh/z \
  --vx-ds /fields/vx --vy-ds /fields/vy --vz-ds /fields/vz \
  --cs-ds /fields/c_s --nd-scan-axis 0 --nd-patches 64
```

Interpretation and literature
- κ scaling depends on profile shape and thresholds; this repo reports a
  threshold‑limited upper bound for the provided synthetic sweeps, not a
  universal limit. Use your own thresholds and profiles to re‑evaluate.
- The intensity cap used in shared sweeps ($I < 1\times10^{24}\,\text{W/m}^2$) is a
  theoretical 1D breakdown bound to standardize comparisons; it exceeds current
  ELI-scale laser capabilities and should not be interpreted as an achievable
  experimental setting without additional justification.
- Chen & Mourou (2017, 2022) motivate accelerating plasma mirrors and
  information tests rather than specifying a κ_max; our estimates are not
  directly comparable without a detailed coupling model.
- BEC experiments (e.g., Steinhauer) probe a very different regime with κ
  orders of magnitude smaller. Cross‑platform comparison requires care.

Recommended next steps for validation
- Use real PIC outputs for v(x) and c_s(x) where available; avoid synthetic
  tanh profiles for quantitative claims.
- Vary thresholds (e.g., velocity fraction of c, gradient caps) to assess how
  κ bounds change; document the sensitivity.
- Replace patch‑wise approximation with a dedicated nD scattering model if
  needed (e.g., Helmholtz solver in tortoise coordinates).

FAQ
- Why does my κ exceed the “bound” in the README? The bound is tied to a
  specific sweep and thresholds. If you use different thresholds/profiles,
  κ can be higher or lower; this is expected.
- How should I set graybody parameters? Use `acoustic_wkb` for near‑horizon
  acoustic consistency and sweep `alpha_gray` to reflect coupling uncertainty.

---

Back: [Limitations](Limitations.md) · Next: [References »](REFERENCES.md)
