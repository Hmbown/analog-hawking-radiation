# Analog Hawking Radiation Simulator

[![Python Version](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml/badge.svg)](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/) [![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases) [![Cite](https://img.shields.io/badge/Cite-CITATION.cff-orange.svg)](CITATION.cff)

[![κ_max](https://img.shields.io/badge/κ_max-5.94×10¹²%20Hz-red.svg)](RESEARCH_HIGHLIGHTS.md) [![GPU Speedup](https://img.shields.io/badge/GPU-10--100×-green.svg)](docs/pc_cuda_workflow.md) [![Validation](https://img.shields.io/badge/validations-42%20tests-brightgreen.svg)](tests/)

A practitioner-focused toolkit for designing and validating analog Hawking radiation experiments in realistic laser–plasma settings. The simulator links fluid models, particle-in-cell (PIC) pipelines, quantum field theory post-processing, and radio detection forecasts into one reproducible environment.

---

## Quick Start — 15 Seconds

**What this does**: Predicts when laser-created plasma flows form "sonic horizons" and estimates the strength of Hawking-like radiation signals using standard radio detection physics.

**Two commands to everything**:
```bash
make comprehensive    # Generate all analyses and figures
make results-pack     # Create complete results package
```

Or use the new CLI:
```bash
ahr quickstart        # Synthetic example → horizons + manifest
ahr validate          # Run physics validation suite
ahr regress           # Golden baseline regression check
```

GPU acceleration (Linux/CUDA):
```bash
pip install -e .[gpu]   # installs CuPy CUDA 12 wheels
ahr gpu-info            # verify backend
```

**Key findings in this dataset**:
- Hybrid model: ~4× higher signal temperature, ~16× faster detection than fluid baseline
- Model-dependent results (follow radiometer scaling), not universal constants
- Small dataset (20 configurations) — interpret with caution

**Critical guardrails**:
- Some perfect correlations (r ≈ 1) are "by construction" from constants/derived fields
- Flat coupling_strength scaling is dataset-specific
- Results are model-dependent, not fundamental physics

---

## Results Package

**Generate complete results package**:
```bash
make comprehensive && make results-pack
```

**Package contents**:
- **Figures**: 4 curated plots (speedup heatmap, detection scatter, enhancement chart, Pareto frontier)
- **Data**: hybrid_sweep.csv (20 configurations, 5 coupling strengths)
- **Summary**: RESULTS_README.md with 1-page overview
- **Documentation**: Reproducibility, Dataset Notes, Limitations
- **Citation**: CITATION.cff + BibTeX ready format

---

## For Scientists — At a Glance

**Methods**: Horizon finding → κ→T_H mapping (ħκ/2πk_B) → graybody transmission → band-limited PSD integration → radiometer detection time → parameter sweeps → Pareto ranking

**Validation**: Analytical checks (κ→T_H, plasma frequency), numerical convergence, frequency-gating sanity checks

**Reproducibility**: Python 3.9–3.11; `make comprehensive` → all figures; `make results-pack` → complete results package

**Quick setup**:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-verified.txt  # Pinned deps
pytest -q                                  # Verify environment (42 tests pass)
```

---

## 🎯 Latest Research (v0.3.0 - October 2025)

### Parametric Upper Bound (threshold‑limited)

**Surface Gravity Upper Bound (this run)**: κ_max ≈ **5.94×10¹² Hz** (acoustic‑exact κ; thresholds enforced)

This bound comes from a sweep over synthetic 1D profiles with explicit breakdown thresholds. It is not a fundamental constant and shifts with thresholds and model assumptions.

#### Key Findings
- **Optimal Configuration**: a₀ ≈ 1.6, nₑ ≈ 1.39×10¹⁹ m⁻³
- **Detection Times**: 10⁻⁷ to 10⁻⁶ seconds (realistic with fast diagnostics)
- **Scaling (this run)**: κ vs a₀ exponent ≈ +0.66 (95% CI [0.44, 0.89]); κ vs nₑ exponent ≈ −0.02 (95% CI [−0.14, 0.10])
- **Breakdown thresholds enforced**: v < 0.5c, |dv/dx| < 4×10¹² s⁻¹, intensity < 1×10²⁴ W/m²

#### Impact
Systematic mapping of breakdown constraints in our 1D models. This study:
- Sets realistic expectations for laboratory analog experiments
- Guides laser-plasma parameter optimization
- Establishes theoretical foundation for detection prospects

📄 **[Full Research Details](RESEARCH_HIGHLIGHTS.md)** | 📊 **[Gradient Catastrophe Analysis](docs/GradientCatastropheAnalysis.md)**

---

## 1. Orientation at a Glance

### At a Glance (Pipeline)
- 1) Build a plasma profile → find sonic horizons (|v| ≈ c_s)
- 2) Compute surface gravity κ → map to thermal scale T_H = ħκ/(2πk_B)
- 3) Apply graybody transmission → band‑limit spectrum
- 4) Convert band power to signal temperature → estimate 5σ detection time

![Workflow pipeline](docs/img/workflow_diagram.png)

### Executive summary
- **Purpose** – Explore when laboratory plasmas form sonic horizons and whether the resulting Hawking-like signal is measurable.
- **Scope** – Covers analytical fluid backends, WarpX/PIC integration, horizon finding, graybody filtering, radio detection forecasts, and physics validation.
- **Latest milestone (v0.3)** – Gradient catastrophe campaign mapping a threshold‑limited upper bound $\kappa_{\max} \approx 5.94\times10^{12}\,\text{Hz}$ for this production run.

### Who this repository serves
| Role | How you benefit |
| --- | --- |
| **Experimental physicists** | Forecast detection timelines, evaluate equipment requirements, and compare diagnostic strategies before committing beam time. |
| **Simulation specialists** | Plug PIC/WarpX outputs directly into the horizon finder and universality tests without bespoke conversion scripts. |
| **Theorists & analysts** | Stress-test assumptions (graybody models, plasma mirrors, trans-Planckian add-ons) and quantify uncertainty budgets. |
| **Vibe coders & cosmic tinkerers** | Explore cutting-edge plasma physics with polished scripts, rich documentation, and curated plots—learn the science while hacking on the universe’s weirdest lab analogies. |

### Quick links
- **Production playbooks** – [`docs/Experiments.md`](docs/Experiments.md)
- **Physics limits study** – [`docs/GradientCatastropheAnalysis.md`](docs/GradientCatastropheAnalysis.md)
- **Methodology deep dive** – [`docs/Methods.md`](docs/Methods.md)
- **Glossary** – [`docs/Glossary.md`](docs/Glossary.md)
- **FAQ** – [`docs/FAQ.md`](docs/FAQ.md)
- **Reproducibility** – [`docs/Reproducibility.md`](docs/Reproducibility.md)
- **Dataset notes** – [`docs/DatasetNotes.md`](docs/DatasetNotes.md)
- **nD horizons** – [`docs/horizon_nd.md`](docs/horizon_nd.md)
- **AnaBHEL comparison** – [`docs/AnaBHEL_Comparison.md`](docs/AnaBHEL_Comparison.md)
- **Release context** – [`RELEASE_NOTES_v0.3.0.md`](RELEASE_NOTES_v0.3.0.md)
- **Known gaps** – [`docs/Limitations.md`](docs/Limitations.md)

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
# Conservative demo that respects production bounds (caps reported metrics if exceeded)
python scripts/run_full_pipeline.py --demo --safe-demo --respect-thresholds --kappa-method acoustic_exact --graybody acoustic_wkb
cat results/full_pipeline_summary.json
```

> 💡 New contributors should also skim `docs/pc_cuda_workflow.md` (GPU setup) and `docs/AdvancedScenarios.md` (guided exercises).

### Performance & CI/CD Optimization

**For fast execution and CI/CD environments**, graphics generation can be disabled to significantly reduce runtime:

```bash
# Method 1: Command line flag (recommended)
python scripts/comprehensive_monte_carlo_uncertainty.py --no-plots
python scripts/enhanced_analysis_pipeline.py --no-plots

# Method 2: Environment variable (useful for CI/CD)
export ANALOG_HAWKING_NO_PLOTS=1
python scripts/run_full_pipeline.py --demo

# Method 3: Combined approach
ANALOG_HAWKING_NO_PLOTS=1 python scripts/enhanced_analysis_pipeline.py --no-correlation --no-significance
```

**Performance benefits**:
- **CI/CD speed**: 10-100x faster test execution by skipping matplotlib operations
- **Memory usage**: Reduced RAM consumption in headless environments
- **Batch processing**: Enables large-scale analysis without graphics overhead
- **Reproducible results**: Computational outputs remain identical with or without graphics

**When to use graphics control**:
- **CI/CD pipelines**: Always disable graphics for faster builds
- **HPC clusters**: Use `ANALOG_HAWKING_NO_PLOTS=1` for batch jobs
- **Development**: Enable graphics for interactive exploration
- **Production**: Disable graphics for automated analysis workflows

**Supported scripts with graphics control**:
- `comprehensive_monte_carlo_uncertainty.py` - Monte Carlo uncertainty analysis
- `enhanced_analysis_pipeline.py` - Enhanced analysis with uncertainty quantification
- Additional scripts are being updated progressively

---

## 3. Choose Your Workflow

1. **Need a baseline horizon & detection forecast?** → Run the [baseline fluid pipeline](#baseline-fluid-pipeline).
2. **Validating with PIC/WarpX data?** → Jump to [WarpX ↔ PIC integration](#warpx--pic-integration).
3. **Chasing physical limits?** → Use the [gradient catastrophe sweep](#gradient-catastrophe-physics-breakdown-analysis).
4. **Comparing spectra across configurations?** → Try the [universality spectrum collapse](#universality-spectrum-collapse).
5. **Planning full campaigns?** → Launch the [orchestration engine](#full-campaign-orchestration).

Each workflow is a first-class script with documented arguments, expected outputs, and downstream artifacts.

---

## 4. Experiment Playbooks

### Experiment catalog

| # | Script | Goal | Typical runtime | Key outputs |
| --- | --- | --- | --- | --- |
| 1 | `scripts/run_full_pipeline.py` | Baseline fluid horizon → Hawking spectrum → radio SNR. | < 1 minute (demo) | `results/full_pipeline_summary.json`, graybody plots |
| 2 | `scripts/run_pic_pipeline.py` | Convert WarpX/PIC data, detect horizons, compare universality. | 5–20 minutes depending on dataset | `results/pic_run/summary.json`, universality diagnostics |
| 3 | `scripts/infer_kappa_from_psd.py` | Infer $\kappa$ from experimental PSDs via Bayesian optimization. | ≈ 10 minutes (40 evaluations) | Posterior samples, corner plots, CSV summaries |
| 4 | `scripts/correlation_map.py` | Horizon-aligned $g^{(2)}(x_1,x_2)$ partner-mode analysis. | 5 minutes for 100 snapshots | Correlation heat map PNG, `g2_horizon.npz` |
| 5 | `scripts/sweep_gradient_catastrophe.py` | Map physics breakdown limits for $\kappa$. | 15–30 minutes (500 samples) | `gradient_catastrophe_sweep.json`, findings report |
| 6 | `scripts/sweep_multi_physics_params.py` | Universality spectrum collapse across configurations. | 10–20 minutes | Collapsed spectra, comparison metrics |
| 7 | `scripts/orchestration_engine.py` | Automate multi-phase campaigns with refinement and validation. | Hours for full sweeps | Phase reports under `results/orchestration/<ID>/` |

Detailed walkthroughs follow.

### Baseline fluid pipeline
```bash
python scripts/run_full_pipeline.py --demo --kappa-method acoustic_exact --graybody acoustic_wkb
python scripts/run_full_pipeline.py --demo --safe-demo --respect-thresholds --kappa-method acoustic_exact --graybody acoustic_wkb
```
- Produces a horizon-aware summary at `results/full_pipeline_summary.json` with κ, graybody transmission, and 5σ detection times.
- When a demo produces κ above the production sweep’s parametric upper bound, the summary includes `sanity_violation: true` and a `sanity_note`. Use `--safe-demo` and/or `--respect-thresholds` to keep reported metrics conservative.
- Optional arguments: `--save-figures`, `--profile-path`, and `--config` to switch between preset laser profiles.

### WarpX ↔ PIC integration
```bash
python scripts/run_pic_pipeline.py --input-path /path/to/openpmd/files --output-dir results/pic_run
```
- Converts openPMD snapshots, aligns on horizons, and runs universality comparison metrics.
- Supports `--slice` to restrict iterations, `--observable` to switch fields, and `--plot` to emit figures.

### κ inference from PSDs *(new in v0.3)*
```bash
python scripts/infer_kappa_from_psd.py results/psd_*.npz \
  --graybody-profile results/warpx_profile.npz \
  --graybody-method acoustic_wkb \
  --calls 40
```
- Bayesian optimization over κ with credible intervals; outputs posterior samples and summary tables in the target directory.

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
- Outputs include the peak κ before breakdown, validity scoring, and markdown findings for reports.

### Universality spectrum collapse
```bash
python scripts/sweep_multi_physics_params.py --config configs/orchestration/pic_downramp.yml
```
- Normalizes spectra by κ to test whether disparate profiles share a universal curve.
- Produces overlay plots and deviation metrics under `results/universality/`.

### Full campaign orchestration
```bash
python -m scripts.orchestration_engine --config configs/orchestration/pic_downramp.yml
```
- Automates exploration → refinement → optimization → validation. Works with `make orchestrate` shortcuts and includes monitoring/aggregation utilities.
- Add `--phases` to run a subset or `--resume` to restart interrupted campaigns.

---

## 5. Scientific Findings & Insights

### Latest orchestrated campaign
- Experiment ID: `a27496e3`
- Phases executed: initial exploration → refinement → optimization → validation
- Total simulations: 140 (sequential fallback in constrained environments)
- Reports: `results/orchestration/a27496e3/final_report.txt` and `.../comprehensive_report.txt`

### Gradient catastrophe highlights
- **Upper bound (this run)** – Threshold‑limited $\kappa_{\max} \approx 5.94\times10^{12}\,\text{Hz}$ (acoustic‑exact; see `results/gradient_limits_production/`).
- **Breakdown thresholds enforced** – $v < 0.5c$, $|\partial_x v| < 4\times10^{12}\,\text{s}^{-1}$, $I < 1\times10^{24}\,\text{W/m}^2$.
- **Scaling (this run)** – $\kappa$ vs $a_0$ exponent ≈ +0.66 (95% CI [0.44, 0.89]); $\kappa$ vs $n_e$ exponent ≈ −0.02 (95% CI [−0.14, 0.10]).
- **Interpretation** – These values are specific to our 1D synthetic profiles and threshold choices; they are not fundamental constants.
- Full methodology and plots: [`docs/GradientCatastropheAnalysis.md`](docs/GradientCatastropheAnalysis.md) and `results/gradient_limits_production/`.

### Universality & detection takeaways
- **Spectrum collapse** – κ-normalized spectra from analytic and PIC-derived profiles align on a common curve.
- **Detection windows** – Conservative 5σ integration times remain ≥ $10^{-7}$ s despite optimistic graybody envelopes.
- **Hybrid scenarios** – Plasma mirror couplings remain exploratory and outside validated parameter space.

### Interpreting pipeline outputs
Standard JSON summaries include:
- `kappa`, `kappa_err` – surface gravity and numerical uncertainty (s⁻¹)
- `T_H_K`, `T_sig_K` – Hawking temperature and radio-band signal temperature (Kelvin)
- `t5sigma_s`, `t5sigma_s_low`, `t5sigma_s_high` – baseline and conservative 5σ integration times
- `hybrid_*` – metrics when optional plasma mirror modes are enabled

---

## How to Cite

If you use this repository, please cite it. See `CITATION.cff` for machine‑readable metadata.

BibTeX example
```
@software{bown2025_analog_hawking,
  author  = {Bown, Hunter},
  title   = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  year    = {2025},
  version = {0.3.0},
  url     = {https://github.com/hmbown/analog-hawking-radiation}
}
```

---

## Usage

**Generate complete results package**:
```bash
make comprehensive && make results-pack
```
Creates `results/results_pack.zip` containing figures, data, and documentation.

**Additional resources**:
- Plasma mirror comparison: `docs/AnaBHEL_Comparison.md`
- Example figures: `docs/img/` with detailed explanations

### Important Limitations
- **Model scope**: Graybody models are 1D only; multi-dimensional effects not captured
- **Uncertainties**: κ values reflect numerical variation only, not experimental systematics
- **Hybrid coupling**: Speculative plasma mirror scenarios — treat as scenario planning, not prediction
- **Dataset size**: Small dataset (20 configurations) limits statistical confidence

### Scientific Context
- κ values are from synthetic 1D profiles, not validated laser-plasma predictions
- Detection time estimates are illustrative; recompute with experiment-specific parameters
- Results are model-dependent and dataset-specific, not universal laws

**Upcoming work**: WarpX execution layer, fluctuation injector, trans-Planckian workflows (see [`docs/trans_planckian_next_steps.md`](docs/trans_planckian_next_steps.md))

---

## 6. Outputs & Data Products
- **Results directory** – Each workflow stores JSON/NPZ summaries and plots under `results/` with descriptive subfolders.
- **Orchestration artifacts** – Reports, dashboards, and aggregation outputs in `results/orchestration/<experiment_id>/`.
- **Gradient sweep** – `results/gradient_limits/gradient_catastrophe_findings.md` and associated plots for publication use.
- **Figures** – Publication-ready PNGs/SVGs in `docs/img/` and `results/*/figures/` when enabled.

---

## 7. Validation & Quality Assurance
- `pytest -q` – core unit + integration suite (~40 tests)
- `pytest -m gpu` – optional GPU coverage when CuPy is installed
- `pytest tests/test_pic_pipeline.py` – targeted PIC flow validation
- Continuous integration covers Python 3.9–3.11 (`.github/workflows/ci.yml`)
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
- `docs/Overview.md` – Conceptual overview and physics motivation
- `docs/Methods.md` – Algorithms for horizon finding, graybody solvers, detection modeling
- `docs/Experiments.md` – Universality experiments and PIC integration guide
- `docs/AdvancedScenarios.md` – Command recipes for PIC, universality, and hybrid workflows
- `docs/GradientCatastropheAnalysis.md` – Physics breakdown boundary mapping and threshold‑limited upper bounds (new in v0.3)
- `docs/Results.md` – Representative outputs and interpretation guidance
- `docs/Limitations.md` – Scope, assumptions, and open questions
- `docs/phase_timeline.md` – Development roadmap and release cadence
- `docs/REFERENCES.md` – Bibliography and suggested reading

---

## 10. Physics Background
Analog black holes form where the flow speed $|v|$ exceeds the local sound speed $c_s$, creating a sonic horizon. The associated surface gravity governs the Hawking temperature via $T_H = \hbar \kappa / (2 \pi k_B)$. This framework implements multiple κ definitions, graybody transmission models, and radio detection estimates to assess whether realistic laser–plasma profiles can produce measurable thermal signatures.

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
### nD horizon demo (2D/3D)
```bash
# 2D
python scripts/run_horizon_nd_demo.py --dim 2 --nx 160 --ny 40 --sigma 4e-7 --v0 2.0e6 --cs0 1.0e6 --x0 5e-6

# 3D
python scripts/run_horizon_nd_demo.py --dim 3 --nx 64 --ny 24 --nz 16 --sigma 6e-7 --v0 1.8e6 --cs0 1.0e6 --x0 5e-6
```
- Saves horizon points and κ statistics to `results/horizon_nd_demo/summary.json` and plots `horizon_2d.png` for 2D runs.

### OpenPMD → nD grid converter (experimental)
```bash
python scripts/openpmd_to_grid_nd.py --in sample.h5 \
  --x /mesh/x --y /mesh/y --z /mesh/z \
  --vx /fields/vx --vy /fields/vy --vz /fields/vz \
  --cs /fields/c_s --out results/grid_nd_profile.npz
```
- You can then build `v_field` by stacking vx, vy, (vz) and call the nD horizon API.
  Or run directly:
```bash
python scripts/run_nd_from_npz.py results/grid_nd_profile.npz --scan-axis 0
```
