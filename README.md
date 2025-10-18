# Analog Hawking Radiation Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml/badge.svg)](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-38%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases)
[![Release Notes](https://img.shields.io/badge/release%20notes-v0.2.0-informational.svg)](RELEASE_NOTES_v0.2.0.md)

![Workflow Pipeline](docs/img/workflow_diagram.png)

A physics-first, end-to-end modeling environment for analog Hawking radiation in laser–plasma systems. The toolkit combines horizon diagnostics, graybody transmission modeling, radio-band detectability estimates, and an exploratory hybrid coupling to accelerating plasma mirrors.

## Universality Experiment (new)

- Collapse Hawking spectra by normalizing frequency to ω/κ and applying an acoustic-WKB graybody. Across ≥4 analytic families (and optional PIC slices), the normalized PSDs collapse onto a narrow band.
- κ-inference closes the loop: invert noisy PSDs via grid-search MLE to recover κ with calibrated uncertainty and coverage.

Quick run:

```bash
python scripts/experiment_universality_collapse.py \
  --out results/experiments/universality --n 32 --alpha 0.8 --seed 7 --include-controls
```

See docs/Experiments.md for details and acceptance criteria.

## Key Innovations (v0.2.0)

- **Exact acoustic surface gravity**: new `kappa_method="acoustic_exact"` evaluates $\kappa = |\partial_x(c_s^2 - v^2)| / (2 c_H)$ at the horizon, exports the on-horizon sound speed and gradient diagnostics, and keeps the legacy definitions for comparison.
- **Acoustic-WKB graybody solver**: constructs the tortoise coordinate $x^*$, forms a barrier potential scaled by $\alpha\kappa$, and returns transmission curves with uncertainty envelopes. Conservative dimensionless and legacy WKB options remain available.
- **Uncertainty-aware detection metrics**: propagates surface-gravity error bars and graybody envelopes through band power, signal temperature, and 5σ integration times, yielding `t5sigma_low/high` bounds in pipeline summaries.
- **PIC/OpenPMD ingestion path**: converts HDF5 slices into 1D profiles (`scripts/openpmd_slice_to_profile.py`) and runs the same physics+radio pipeline (`scripts/run_pic_pipeline.py`) for particle-in-cell outputs.
- **Hybrid plasma mirror exploration** *(optional)*: couples fluid horizons to AnaBHEL-inspired mirror dynamics (`--hybrid --hybrid-model {anabhel,unruh}`) for speculative studies of mirror-assisted surface gravity enhancement.

## End-to-End Workflow

1. **Profile generation** – analytical fluid backend, PIC-derived profiles, or supplied numpy arrays.
2. **Horizon finding** – locate $|v| = c_s$ crossings, evaluate $\kappa$ with numerical uncertainty, and log diagnostics.
3. **Graybody transmission** – choose `dimensionless`, `wkb`, or `acoustic_wkb` transmission curves centred on the detected horizon.
4. **Spectrum + detection** – compute Hawking spectra, integrate over radio bands, and estimate 5σ detection times under user-specified system temperature and bandwidth.
5. **Reporting** – persist JSON summaries, PSD plots, and optional hybrid comparisons in `results/` and `figures/`.

## Quick Start

### Install

```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
```

### Run the fluid demo

```bash
python scripts/run_full_pipeline.py --demo \
  --kappa-method acoustic_exact \
  --graybody acoustic_wkb \
  --alpha-gray 0.8
cat results/full_pipeline_summary.json
```

### PIC/OpenPMD pipeline

```bash
python scripts/openpmd_slice_to_profile.py --in data/slice.h5 \
  --x-dataset /x --vel-dataset /vel --Te-dataset /Te \
  --out results/warpx_profile.npz
python scripts/run_pic_pipeline.py --profile results/warpx_profile.npz \
  --kappa-method acoustic_exact --graybody acoustic_wkb
```

### Exploratory hybrid coupling

```bash
python scripts/run_full_pipeline.py --demo --hybrid \
  --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0
```

### Compare configurations

- `python scripts/demo_hybrid_comparison.py`
- `python scripts/sweep_hybrid_params.py`
- `python scripts/generate_detection_time_heatmap.py`

## Interpreting Results

Pipeline summaries (e.g. `results/full_pipeline_summary.json`) report:

- `kappa` / `kappa_err`: surface gravity and numerical uncertainty (s⁻¹)
- `T_H_K`: Hawking temperature implied by κ
- `T_sig_K`: radio-band signal temperature after graybody transmission
- `t5sigma_s`, `t5sigma_s_low`, `t5sigma_s_high`: baseline and conservative 5σ integration times for the specified system temperature and bandwidth
- `hybrid_*`: effective κ and detectability metrics when hybrid mode is enabled

Example figures appear in `docs/img/`, including graybody comparisons and κ definition overlays.

## Documentation Map

- `docs/Overview.md` – conceptual overview and motivation
- `docs/Methods.md` – algorithms for horizon finding, graybody solvers, and detection modeling
- `docs/Highlights_v0.2.0.md` – physics summary of the v0.2.0 release
- `docs/Results.md` – representative outputs and interpretation guidance
- `docs/Limitations.md` – scope, assumptions, and open questions
- `RELEASE_NOTES_v0.2.0.md` – changelog with implementation details

## Physics Background

Analog black holes form where the flow speed $|v|$ exceeds the local sound speed $c_s$, creating a sonic horizon. The associated surface gravity governs the Hawking temperature via $T_H = \hbar\kappa/(2\pi k_B)$. This framework implements multiple κ definitions, graybody transmission models, and radio detection estimates to explore whether realistic laser–plasma profiles can produce measurable thermal signatures.

The optional hybrid branch couples these fluid horizons to accelerating plasma mirrors inspired by the AnaBHEL program (Chen & Mourou 2015; Chen et al. 2022). It is intended as a computational thought experiment rather than a validated prediction.

## Reproducibility & Validation

- **Tests**: `pytest -q` (38 unit + integration tests across horizon finding, graybody solvers, PIC round-trips, and hybrid logic)
- **Continuous Integration**: GitHub Actions test matrix on Python 3.9–3.11
- **Sample outputs**: `results/samples/` and executed notebooks (`notebooks/Quickstart.ipynb`)
- **Configuration**: reusable parameter sets in `configs/`

## Repository Layout

```
src/analog_hawking/        # Physics engines, detection models, hybrid coupling
scripts/                   # CLI pipelines, sweeps, figure generation
analysis/                  # Research notebooks and analytic studies
results/                   # JSON summaries, spectra, sample outputs
figures/                   # Generated plots (PSD, comparisons, etc.)
docs/                      # Methods, highlights, limitations, references
paper/                     # ArXiv-ready TeX manuscript and figures
tests/                     # Unit and integration suite
```

## Limitations & Scope

- Hybrid mirror coupling is exploratory and should be treated as speculative.
- κ uncertainties capture numerical stencil variation, not experimental error bars.
- Graybody models are 1D and near-horizon—multi-dimensional effects and dissipation are outside the current scope.
- No experimental validation is claimed; outputs provide trends and order-of-magnitude estimates only.

## Development

- Run tests: `pytest -q`
- Keep patches focused; adhere to CONTRIBUTING.md for style and review expectations
- Use `make lint` / `make docs` (where available) before submitting PRs

## Citation

If you use this work, please cite both the framework and the foundational AnaBHEL research:

**This Framework**
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.2.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation},
  note = {Speculative extension of AnaBHEL concepts}
}
```

**Foundational AnaBHEL Work**
```bibtex
@article{chen2022anabhel,
  title={AnaBHEL (Analog Black Hole Evaporation via Lasers) Experiment: Concept, Design, and Status},
  author={Chen, Pisin and Mourou, Gerard and Besancon, Marc and Fukuda, Yasuhiko and Glicenstein, Jean-Fran\c{c}ois and others},
  journal={Photonics},
  volume={9},
  number={12},
  pages={1003},
  year={2022},
  publisher={MDPI}
}

@article{chen2015plasma,
  title={Accelerating plasma mirrors to investigate the black hole information loss paradox},
  author={Chen, Pisin and Mourou, Gerard},
  journal={Physical Review Letters},
  volume={118},
  number={4},
  pages={045001},
  year={2015},
  publisher={APS}
}
```

## References & Further Reading

- Unruh (1981) – acoustic analog of black hole radiation
- Hawking (1974, 1975) – black hole radiation theory
- Steinhauer (2016) – experimental analog Hawking radiation in Bose–Einstein condensates
- Faccio & Wright (2013) – laser-driven analog gravity systems
- Mourou et al. (2006) – ultrafast laser innovations enabling high-field experiments
- Complete bibliography: `docs/REFERENCES.md`

---

**Framework Version**: 0.2.0 · **License**: MIT · **Tests**: 38/38 passing (local)
