# Analog Hawking Radiation Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml/badge.svg)](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml) [![Tests](https://img.shields.io/badge/tests-42%20passing-brightgreen.svg)](tests/) [![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases) [![Release Notes](https://img.shields.io/badge/release%20notes-v0.3.0-informational.svg)](RELEASE_NOTES_v0.3.0.md)

A practitioner-focused toolkit for designing and validating analog Hawking radiation experiments in realistic laser-plasma settings. The simulator links fluid models, particle-in-cell (PIC) pipelines, quantum field theory post-processing, and radio detection forecasts into one reproducible environment.

---

## 1. Orientation at a Glance

### Executive summary
- **Purpose** - Explore when laboratory plasmas form sonic horizons and whether the resulting Hawking-like signal is measurable.
- **Scope** - Covers analytical fluid backends, WarpX/PIC integration, horizon finding, graybody filtering, radio detection forecasts, and physics validation.
- **Latest milestone (v0.3)** - GPU-accelerated mapping of the surrogate model parameter space (1,800 configurations across a0 in  [1, 40], ne in  [10^18, 10^22] m^-3, gradient factor <= 500). About half of the sampled points satisfied the validation suite; breakdowns were dominated by relativistic effects once a0 >= 1-2. Fitted trends yielded kappa proportional to  a0^alpha ne^beta with alpha approximately  -0.2 and beta approximately  -0.05 for this profile family, while kappa-normalized graybody spectra collapsed with RMS deviation below 1e-2 across analytic and PIC-derived flows. These results provide a quantitative baseline for future PIC validation and experimental design.

### Who this repository serves
| Role | How you benefit |
| --- | --- |
| **Experimental physicists** | Forecast detection timelines, evaluate equipment requirements, and compare diagnostic strategies before committing beam time. |
| **Simulation specialists** | Plug PIC/WarpX outputs directly into the horizon finder and universality tests without bespoke conversion scripts. |
| **Theorists & analysts** | Stress-test assumptions (graybody models, plasma mirrors, trans-Planckian add-ons) and quantify uncertainty budgets. |
| **Vibe coders & cosmic tinkerers** | Explore cutting-edge plasma physics with polished scripts, rich documentation, and curated plots while learning the science behind laboratory analogues. |

### Quick links
- **Production playbooks** - [`docs/Experiments.md`](docs/Experiments.md)
- **GPU acceleration guide** - [`docs/GPU_ACCELERATION_GUIDE.md`](docs/GPU_ACCELERATION_GUIDE.md)
- **Experimental gap analysis** - [`docs/ExperimentalAccessibility.md`](docs/ExperimentalAccessibility.md)
- **Physics limits study** - [`docs/GradientCatastropheAnalysis.md`](docs/GradientCatastropheAnalysis.md)
- **Methodology deep dive** - [`docs/Methods.md`](docs/Methods.md)
- **Release context** - [`RELEASE_NOTES_v0.3.0.md`](RELEASE_NOTES_v0.3.0.md)
- **Known gaps** - [`docs/Limitations.md`](docs/Limitations.md)

![Workflow pipeline](docs/img/workflow_diagram.png)

---

## 2. Installation & First Validation (10 minutes)

```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .

# Optional accelerators and visualization extras
pip install cupy scikit-optimize seaborn

# Verify the environment
pytest -q
```

**One-minute smoke test**
```bash
python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb
cat results/full_pipeline_summary.json
```

> Note:  New contributors should also skim `docs/pc_cuda_workflow.md` (GPU setup) and `docs/AdvancedScenarios.md` (guided exercises).

### Docker quickstart

Build and run the reproducible environment without touching your host Python:

```bash
docker build -t analog-hawking:latest .
docker run --rm -it -v ${PWD}/results:/workspace/results analog-hawking:latest \
  python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb
```

The image installs all runtime dependencies (set `--build-arg INSTALL_EXTRAS=true` to include test and notebook tooling). Run the test suite inside the container with:

```bash
docker run --rm -it analog-hawking:latest pytest -q
```

On PowerShell, replace `${PWD}` with `$PWD` or an absolute path when mounting volumes.

### RTX 3080 / GPU campaign (optional)

If you have a CUDA-capable GPU (e.g., RTX 3080), install CuPy (`pip install cupy-cuda12x`) and run the accelerated campaign:

```bash
ANALOG_HAWKING_USE_CUPY=1 python scripts/run_gpu_campaign.py \
  --tasks gradient universality detection \
  --gradient-samples 1800 \
  --results-dir results/gpu_rtx3080
```

See `docs/GPU_ACCELERATION_GUIDE.md` for tuning tips, memory guidance, and advanced runs. The reference campaign produced the following high-level observations (see `results/gpu_rtx3080/` for JSON artefacts):

- Coverage: 1,800 configurations spanning a0 in [1, 40], ne in [1e18, 1e22] m^-3, gradient factors up to 500.
- Validity: 49% of samples satisfied the conservation and stability checks; invalid cases were dominated by relativistic breakdown once a0 exceeded roughly 2.
- Scaling: Least-squares fits gave kappa proportional to a0^-0.2 ne^-0.05 within this surrogate profile family; intensity followed the expected quadratic scaling with a0.
- Universality: Graybody spectra normalised by kappa collapsed with RMS deviation about 7e-3 across analytic and PIC-derived flow families.
- Experimental outlook: Present laser facilities (I <= 1e23 W/m^2) remain in the mapped low-kappa regime; higher kappa values serve as forward-looking targets for future hardware or alternative analogue platforms.

---

## 3. Choose Your Workflow

1. **Need a baseline horizon & detection forecast?** -> Run the [baseline fluid pipeline](#baseline-fluid-pipeline).
2. **Validating with PIC/WarpX data?** -> Jump to [WarpX <-> PIC integration](#warpx--pic-integration).
3. **Chasing physical limits?** -> Use the [gradient catastrophe sweep](#gradient-catastrophe-physics-breakdown-analysis).
4. **Comparing spectra across configurations?** -> Try the [universality spectrum collapse](#universality-spectrum-collapse).
5. **Planning full campaigns?** -> Launch the [orchestration engine](#full-campaign-orchestration).

Each workflow is a first-class script with documented arguments, expected outputs, and downstream artifacts.

---

## 4. Experiment Playbooks

### Experiment catalog

| # | Script | Goal | Typical runtime | Key outputs |
| --- | --- | --- | --- | --- |
| 1 | `scripts/run_full_pipeline.py` | Baseline fluid horizon -> Hawking spectrum -> radio SNR. | < 1 minute (demo) | `results/full_pipeline_summary.json`, graybody plots |
| 2 | `scripts/run_pic_pipeline.py` | Convert WarpX/PIC data, detect horizons, compare universality. | 5-20 minutes depending on dataset | `results/pic_run/summary.json`, universality diagnostics |
| 3 | `scripts/infer_kappa_from_psd.py` | Infer $\kappa$ from experimental PSDs via Bayesian optimization. | approximately  10 minutes (40 evaluations) | Posterior samples, corner plots, CSV summaries |
| 4 | `scripts/correlation_map.py` | Horizon-aligned $g^{(2)}(x_1,x_2)$ partner-mode analysis. | 5 minutes for 100 snapshots | Correlation heat map PNG, `g2_horizon.npz` |
| 5 | `scripts/sweep_gradient_catastrophe.py` | Map physics breakdown limits for $\kappa$. | 15-30 minutes (500 samples) | `gradient_catastrophe_sweep.json`, findings report |
| 6 | `scripts/sweep_multi_physics_params.py` | Universality spectrum collapse across configurations. | 10-20 minutes | Collapsed spectra, comparison metrics |
| 7 | `scripts/orchestration_engine.py` | Automate multi-phase campaigns with refinement and validation. | Hours for full sweeps | Phase reports under `results/orchestration/<ID>/` |

Detailed walkthroughs follow.

### Baseline fluid pipeline
```bash
python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb
```
- Produces a horizon-aware summary at `results/full_pipeline_summary.json` with kappa, graybody transmission, and 5sigma detection times.
- Optional arguments: `--save-figures`, `--profile-path`, and `--config` to switch between preset laser profiles.

### WarpX <-> PIC integration
```bash
python scripts/run_pic_pipeline.py --input-path /path/to/openpmd/files --output-dir results/pic_run
```
- Converts openPMD snapshots, aligns on horizons, and runs universality comparison metrics.
- Supports `--slice` to restrict iterations, `--observable` to switch fields, and `--plot` to emit figures.

### kappa inference from PSDs *(new in v0.3)*
```bash
python scripts/infer_kappa_from_psd.py results/psd_*.npz \
  --graybody-profile results/warpx_profile.npz \
  --graybody-method acoustic_wkb \
  --calls 40
```
- Bayesian optimization over kappa with credible intervals; outputs posterior samples and summary tables in the target directory.

### Horizon-crossing correlation maps *(new in v0.3)*
```bash
python scripts/correlation_map.py --series ./diags/openpmd \
  --t-index 350 --window 20 \
  --output figures/correlation_map.png
```
- Aligns density fluctuations on the moving horizon to reveal Hawking-partner diagonals.
- Change `--observable` to `velocity` or `sound_speed` to probe alternative signals.

### Gradient catastrophe (physics breakdown) analysis
```bash
python scripts/sweep_gradient_catastrophe.py --n-samples 500 \
  --output results/gradient_limits_analysis
```
- Sweeps $(a_0, n_e, \partial_x v)$ to locate relativistic, ionization, and numerical breakdown envelopes.
- Outputs include the peak kappa before breakdown, validity scoring, and markdown findings for reports.

### Universality spectrum collapse
```bash
python scripts/sweep_multi_physics_params.py --config configs/orchestration/pic_downramp.yml
```
- Normalizes spectra by kappa to test whether disparate profiles share a universal curve.
- Produces overlay plots and deviation metrics under `results/universality/`.

### Full campaign orchestration
```bash
python -m scripts.orchestration_engine --config configs/orchestration/pic_downramp.yml
```
- Automates exploration -> refinement -> optimization -> validation. Works with `make orchestrate` shortcuts and includes monitoring/aggregation utilities.
- Add `--phases` to run a subset or `--resume` to restart interrupted campaigns.

---

## 5. Scientific Findings & Insights

### Latest orchestrated campaign
- Experiment ID: `a27496e3`
- Phases executed: initial exploration -> refinement -> optimization -> validation
- Total simulations: 140 (sequential fallback in constrained environments)
- Reports: `results/orchestration/a27496e3/final_report.txt` and `.../comprehensive_report.txt`

### Gradient catastrophe highlights
- **Fundamental limit** - Maximum surface gravity $\kappa_{\max} \approx 3.8\times10^{12}\,\text{Hz}$ before relativistic breakdown.
- **Relativistic wall** - Viability requires $v < 0.5c$, $\partial_x v < 4\times10^{12}\,\text{s}^{-1}$, $I < 6\times10^{50}\,\text{W/m}^2$.
- **Sweet spot** - $a_0 \approx 1.6$, $n_e \approx 1.4\times10^{19}\,\text{m}^{-3}$ maximizes kappa while remaining physical.
- **Scaling law** - $\kappa \propto a_0^{-0.193}$; increasing laser intensity eventually lowers attainable kappa.
- Full methodology and plots live in [`docs/GradientCatastropheAnalysis.md`](docs/GradientCatastropheAnalysis.md) and `results/gradient_limits/`.

### Universality & detection takeaways
- **Spectrum collapse** - kappa-normalized spectra from analytic and PIC-derived profiles align on a common curve.
- **Detection windows** - Conservative 5sigma integration times remain >= $10^{-7}$ s despite optimistic graybody envelopes.
- **Hybrid scenarios** - Plasma mirror couplings remain exploratory and outside validated parameter space.

### Interpreting pipeline outputs
Standard JSON summaries include:
- `kappa`, `kappa_err` - surface gravity and numerical uncertainty (s⁻^1)
- `T_H_K`, `T_sig_K` - Hawking temperature and radio-band signal temperature (Kelvin)
- `t5sigma_s`, `t5sigma_s_low`, `t5sigma_s_high` - baseline and conservative 5sigma integration times
- `hybrid_*` - metrics when optional plasma mirror modes are enabled

Example figures reside in `docs/img/` alongside explanation overlays.

### Limitations & roadmap
- Graybody models remain 1D; multi-dimensional effects and dissipation are not captured.
- kappa uncertainties cover numerical stencil variation only (no experimental systematics).
- Hybrid mirror coupling is speculative; treat outputs as scenario planning, not prediction.
- Upcoming work: finalize WarpX execution layer, fluctuation injector, and trans-Planckian workflows (see [`docs/trans_planckian_next_steps.md`](docs/trans_planckian_next_steps.md)).

---

## 6. Outputs & Data Products
- **Results directory** - Each workflow stores JSON/NPZ summaries and plots under `results/` with descriptive subfolders.
- **Orchestration artifacts** - Reports, dashboards, and aggregation outputs in `results/orchestration/<experiment_id>/`.
- **Gradient sweep** - `results/gradient_limits/gradient_catastrophe_findings.md` and associated plots for publication use.
- **Figures** - Publication-ready PNGs/SVGs in `docs/img/` and `results/*/figures/` when enabled.

---

## 7. Validation & Quality Assurance
- `pytest -q` - core unit + integration suite (~40 tests)
- `pytest -m gpu` - optional GPU coverage when CuPy is installed
- `pytest tests/test_pic_pipeline.py` - targeted PIC flow validation
- Continuous integration covers Python 3.9-3.11 (`.github/workflows/ci.yml`)
- Physics validation framework enforces conservation laws, physical bounds, numerical stability, and theoretical consistency checks across pipelines.

---

## 8. Repository Map
```
.github/                  # CI workflows and automation
configs/                  # YAML/JSON parameter presets
docs/                     # User guides, methodology notes, figures
examples/                 # Ready-to-run notebooks and scripts
results/                  # Sample outputs and experiment artifacts
scripts/                  # CLI entry points for experiments & sweeps
src/analog_hawking/       # Core library: physics engines & diagnostics
tests/                    # Unit and integration suites
```

---

## 9. Learning Paths & Further Reading
- `docs/Overview.md` - Conceptual overview and physics motivation
- `docs/Methods.md` - Algorithms for horizon finding, graybody solvers, detection modeling
- `docs/Experiments.md` - Universality experiments and PIC integration guide
- `docs/AdvancedScenarios.md` - Command recipes for PIC, universality, and hybrid workflows
- `docs/GradientCatastropheAnalysis.md` - Physics breakdown boundary mapping and fundamental limits (new in v0.3)
- `docs/Results.md` - Representative outputs and interpretation guidance
- `docs/Limitations.md` - Scope, assumptions, and open questions
- `docs/phase_timeline.md` - Development roadmap and release cadence
- `docs/REFERENCES.md` - Bibliography and suggested reading

---

## 10. Physics Background
Analog black holes form where the flow speed $|v|$ exceeds the local sound speed $c_s$, creating a sonic horizon. The associated surface gravity governs the Hawking temperature via $T_H = \hbar \kappa / (2 \pi k_B)$. This framework implements multiple kappa definitions, graybody transmission models, and radio detection estimates to assess whether realistic laser-plasma profiles can produce measurable thermal signatures.

The optional hybrid branch couples fluid horizons to accelerating plasma mirrors inspired by the AnaBHEL program (Chen & Mourou 2015; Chen et al. 2022). Treat these modes as computational thought experiments rather than validated predictions.

---

## 11. Citation
If you use this work, please cite both the framework and the foundational AnaBHEL research.

**This framework**
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation},
  note = {Speculative extension of AnaBHEL concepts}
}
```

**Foundational AnaBHEL work**
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
```

**Plasma mirror concept**
```bibtex
@article{chen2017plasma,
  title={Accelerating plasma mirrors to investigate the black hole information loss paradox},
  author={Chen, Pisin and Mourou, Gerard},
  journal={Physical Review Letters},
  volume={118},
  number={4},
  pages={045001},
  year={2017},
  publisher={APS}
}
```
