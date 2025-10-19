# Analog Hawking Radiation Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml/badge.svg)](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml)
[![Tests](https://img.shields.io/badge/tests-40%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases)
[![Release Notes](https://img.shields.io/badge/release%20notes-v0.2.0-informational.svg)](RELEASE_NOTES_v0.2.0.md)

Analog Hawking Radiation Simulator helps laboratory plasma teams explore how to reproduce Hawking-like signals—analog Hawking radiation, meaning a lab-based imitation of black hole Hawking glow—using realistic equipment. It is built for experimental physicists and simulation specialists who need clear guidance from design choices to data interpretation. The tool delivers a single place to plan scenarios, quantify detection odds, and compare outcomes before committing to expensive runs.

- Model laser–plasma setups to check whether a horizon (a sonic boundary that mimics a black hole edge) can form under given gradients.
- Track how potential signals propagate so you can estimate detection timelines before running hardware.
- Combine diagnostics, data conditioning, and forecast charts in one workflow to decide which probe settings to prioritize.
- Experiment with optional plasma mirror couplings—accelerating reflective fronts that may amplify surface gravity (the effective gravitational pull at the sonic horizon)—without editing core code.

## Table of Contents

- [Universality Experiment (New in v0.2.0)](#universality-experiment-new-in-v020)
- [Getting Started](#getting-started)
- [Key Features (v0.2.0)](#key-features-v020)
- [End-to-End Workflow](#end-to-end-workflow)
- [Interpreting Results](#interpreting-results)
- [Documentation & Resources](#documentation--resources)
- [Physics Background](#physics-background)
- [Reproducibility & Validation](#reproducibility--validation)
- [Repository Layout](#repository-layout)

![Workflow Pipeline](docs/img/workflow_diagram.png)

## Universality Experiment (New in v0.2.0)

**Test the universal nature of Hawking radiation**: When frequency is normalized by surface gravity (ω/κ), Hawking spectra from completely different plasma configurations should collapse onto a single universal curve.

- **Spectrum Collapse**: Normalize frequency by κ and apply acoustic-WKB graybody transmission. Across ≥4 analytic flow families (plus optional PIC simulation data), the power spectral densities collapse onto a narrow universal band.
- **Parameter Recovery**: Close the loop by inverting noisy spectra to recover the original κ value using maximum likelihood estimation with calibrated uncertainty bounds.

For step-by-step instructions on running the universality collapse experiments—including analytic-only and PIC-augmented setups—refer to the [Advanced Scenarios guide](docs/AdvancedScenarios.md) alongside `docs/Experiments.md` for methodology and interpretation.

## Key Features (v0.2.0)

- **Universal Spectrum Collapse**: Test whether Hawking spectra from different plasma configurations collapse onto a universal curve when frequency is normalized by surface gravity (ω/κ). Includes support for both analytic profiles and PIC simulation data.

- **Exact Acoustic Surface Gravity**: New `kappa_method="acoustic_exact"` evaluates $\kappa = |\partial_x(c_s^2 - v^2)| / (2 c_H)$ precisely at the horizon, with full diagnostic export and legacy compatibility.

- **Advanced Graybody Modeling**: Acoustic-WKB solver constructs tortoise coordinates, computes barrier potentials scaled by $\alpha\kappa$, and returns transmission curves with uncertainty envelopes.

- **PIC/OpenPMD Integration**: Convert HDF5 slices from particle-in-cell simulations into 1D profiles and run the complete physics pipeline, including universality testing.

- **Uncertainty-Aware Detection**: Propagates surface-gravity uncertainties and graybody envelopes through detection metrics, yielding realistic `t5sigma` bounds.

- **Hybrid Plasma Mirror Coupling** *(exploratory)*: Optional coupling to AnaBHEL-inspired mirror dynamics for speculative studies of enhanced surface gravity.

## Research Highlights

- **Gradient-managed enhancements**: Multi-beam geometries deliver only modest power boosts once coarse-grained at the skin-depth scale, reinforcing the need for realistic gradient budgeting instead of multiplicative scaling assumptions.
- **Horizon statistics under diagnostics**: Horizon sweeps quantify position uncertainty, surface gravity, and gradient components, confirming that horizon formation remains the primary experimental bottleneck.
- **Guidance and optimization tooling**: Bayesian merit maps and δ-matching guidance visualize where gradient control and radio detectability align, while sound-speed-aware models capture horizon shifts from realistic heating profiles.

See `docs/Results.md` for full figures and quantitative tables.

## Research Roadmap

- Deploy the WarpX execution layer, fluctuation injector, and high-fidelity trans-Planckian experiment workflow ([`docs/trans_planckian_next_steps.md`](docs/trans_planckian_next_steps.md)).
- Close validation gaps by benchmarking against PIC/fluid simulations, refining coarse-graining, and expanding experimental implementation ([`docs/Limitations.md`](docs/Limitations.md)).

## End-to-End Workflow

1. **Profile generation** – analytical fluid backend, PIC-derived profiles, or supplied numpy arrays.
2. **Horizon finding** – locate $|v| = c_s$ crossings, evaluate $\kappa$ with numerical uncertainty, and log diagnostics.
3. **Graybody transmission** – choose `dimensionless`, `wkb`, or `acoustic_wkb` transmission curves centred on the detected horizon.
4. **Spectrum + detection** – compute Hawking spectra, integrate over radio bands, and estimate 5σ detection times under user-specified system temperature and bandwidth.
5. **Reporting** – persist JSON summaries, PSD plots, and optional hybrid comparisons in `results/` and `figures/`.

## Getting Started

### Installation

```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
```

### Basic Example

Spin up the fluid demo with acoustic surface-gravity diagnostics and inspect the resulting summary JSON.

```bash
python scripts/run_full_pipeline.py --demo \
  --kappa-method acoustic_exact \
  --graybody acoustic_wkb \
  --alpha-gray 0.8
cat results/full_pipeline_summary.json
```

### Advanced Scenarios

Extended workflows—including PIC/OpenPMD ingestion, universality collapse campaigns, hybrid coupling comparisons, and parameter sweeps—are documented in the [Advanced Scenarios guide](docs/AdvancedScenarios.md). Each recipe links back to supporting discussions in `docs/Experiments.md` and `docs/Methods.md` for deeper context.

## Interpreting Results

Pipeline summaries (e.g. `results/full_pipeline_summary.json`) report:

- `kappa` / `kappa_err`: surface gravity and numerical uncertainty (s⁻¹)
- `T_H_K`: Hawking temperature implied by κ
- `T_sig_K`: radio-band signal temperature after graybody transmission
- `t5sigma_s`, `t5sigma_s_low`, `t5sigma_s_high`: baseline and conservative 5σ integration times for the specified system temperature and bandwidth
- `hybrid_*`: effective κ and detectability metrics when hybrid mode is enabled

Example figures appear in `docs/img/`, including graybody comparisons and κ definition overlays.

## Documentation & Resources

**Essential Reading**:
- `docs/Overview.md` – conceptual overview and physics motivation
- `docs/Methods.md` – algorithms for horizon finding, graybody solvers, and detection modeling
- `docs/Experiments.md` – **universality experiments and PIC integration guide**
- `docs/AdvancedScenarios.md` – command recipes for PIC, universality, and hybrid workflows
- `docs/Highlights_v0.2.0.md` – physics summary of the current release
- `docs/Results.md` – representative outputs and interpretation guidance
- `docs/Limitations.md` – scope, assumptions, and open questions

**Release Information**:
- `RELEASE_NOTES_v0.2.0.md` – detailed changelog with implementation details

## Physics Background

Analog black holes form where the flow speed $|v|$ exceeds the local sound speed $c_s$, creating a sonic horizon. The associated surface gravity governs the Hawking temperature via $T_H = \hbar\kappa/(2\pi k_B)$. This framework implements multiple κ definitions, graybody transmission models, and radio detection estimates to explore whether realistic laser–plasma profiles can produce measurable thermal signatures.

The optional hybrid branch couples these fluid horizons to accelerating plasma mirrors inspired by the AnaBHEL program (Chen & Mourou 2015; Chen et al. 2022). It is intended as a computational thought experiment rather than a validated prediction.

## Reproducibility & Validation

- **Tests**: `pytest -q` (40 unit + integration tests across horizon finding, graybody solvers, PIC round-trips, and hybrid logic)
- **Continuous Integration**: GitHub Actions test matrix on Python 3.9–3.11
- **Sample outputs**: `results/samples/` and executed notebooks (`notebooks/Quickstart.ipynb`)
- **Configuration**: reusable parameter sets in `configs/`

## Repository Layout

```
.github/                  # Continuous integration workflows and project automation
configs/                  # YAML/JSON parameter presets for pipelines and experiments
docs/                     # User guides, methodology notes, references, and figures
examples/                 # Ready-to-run scripts showcasing core modeling workflows
results/                  # Checked-in sample outputs and experiment artifacts
scripts/                  # CLI entry points for experiments, sweeps, and conversions
src/analog_hawking/       # Core library: physics engines, diagnostics, hybrid coupling
tests/                    # Unit and integration suites for the library and scripts
```

## Limitations & Scope

- Hybrid mirror coupling is exploratory and should be treated as speculative.
- κ uncertainties capture numerical stencil variation, not experimental error bars.
- Graybody models are 1D and near-horizon—multi-dimensional effects and dissipation are outside the current scope.
- No experimental validation is claimed; outputs provide trends and order-of-magnitude estimates only.

## Development

- Run tests: `pytest -q`
- Keep patches focused; adhere to CONTRIBUTING.md for style and review expectations
- Linting: there is no dedicated `make lint` target—follow the PEP 8 guidance in CONTRIBUTING and optionally run your preferred static analysis locally
- Documentation: update the Markdown files under `docs/` directly (no generated docs build or `make docs` step)

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

**Framework Version**: 0.2.0 · **License**: MIT · **Tests**: 40/40 passing (local)
