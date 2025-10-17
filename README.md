# Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases)
[![DOI](https://img.shields.io/badge/DOI-pending-lightgrey.svg)](https://zenodo.org)

![Workflow Pipeline](docs/img/workflow_diagram.png)

---

## TL;DR

- **What**: A reproducible modeling toolkit for analog Hawking radiation in laser–plasma flows.
- **Focus**: Horizon identification, spectra, and radio-band detectability. Includes an optional hybrid fluid–mirror coupling model to explore configuration space.
- **Run a demo**:

```bash
pip install -e .
python scripts/run_full_pipeline.py --demo --hybrid --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0
cat results/full_pipeline_summary.json | head -n 20
```

---

## Repository Map

- `src/analog_hawking/` — core library (physics, detection)
- `scripts/` — runnable analyses and figure generation
- `tests/` — unit and integration tests
- `docs/` — detailed narrative docs: see `docs/Overview.md`, `docs/Methods.md`, `docs/Results.md`, `docs/Validation.md`, `docs/Limitations.md`
- `results/` — generated outputs (gitignored; samples in `results/samples/`)
- `figures/` — generated figures (gitignored)

For details, see `docs/Overview.md`.

---

## Quick Start

```bash
# Install and run hybrid demo (recommended)
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .

# Run with optional hybrid coupling model
python scripts/run_full_pipeline.py --demo --hybrid --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0

# Generate README image (workflow diagram)
make readme-images

# Inspect example outputs
cat results/full_pipeline_summary.json | head -n 20
```

See `results/samples/` for small representative outputs.

---

## Installation and Usage

### System Requirements

- Python ≥ 3.8
- NumPy ≥ 1.21 (compatible with NumPy 2.x)
- SciPy
- Matplotlib

**Computational Resources**:
- Typical laptop (4–8 cores): small sweeps in minutes
- Full parameter sweeps: hours on standard workstation

### Installation

```bash
# Clone repository
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation

# Install with dependencies
pip install -e .
```

### Basic Usage

**Hybrid Pipeline Execution (Recommended)**:
```bash
# Run with optional hybrid coupling model
python scripts/run_full_pipeline.py --demo --hybrid --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0
```

Output: `results/full_pipeline_summary.json` containing complete metrics including:
- `hybrid_used`: true
- `hybrid_kappa_eff`: surface gravity values
- `hybrid_T_sig_K`: antenna temperature

**Fluid-Only Pipeline (Baseline)**:
```bash
python scripts/run_full_pipeline.py --demo
```

Output: Standard pipeline results for comparison with hybrid approach.

---

## Detailed Results: Hybrid Method

### Hybrid Method: Primary Innovation

**The hybrid fluid + plasma-mirror coupling represents the main scientific contribution of this work.** It demonstrates that strategically coupling accelerating plasma mirrors to fluid horizons enables comparative analysis under controlled assumptions.

#### Coupling Physics

Local surface gravity near fluid horizons:

```
κ_eff(x_h) = κ_fluid(x_h) + w(x_h) · κ_mirror

w(x_h) = coupling_strength · exp(-|x_h - x_M|/L) · alignment_gate
```

Combined with effective temperature:
```
T_eff = T_f + w · T_m + cross · sqrt(T_f · T_m)
```

#### Rigorous Comparison Protocol

**Apples-to-apples validation**:
- Identical `graybody_profile` transmission from near-horizon WKB calculations
- Same normalization: emitting area (1×10⁻⁶ m²), solid angle (0.05 sr), coupling efficiency (0.1)
- Same frequency integration band centered at fluid spectrum peak
- Conservative mirror→κ mapping (AnaBHEL model: `κ_mirror = 2π·η_a/D`)

#### Parameter Optimization

**Sensitivity sweeps** explore key parameter ranges:
- **Coupling strength**: Controls overall magnitude
- **Mirror parameters**: Diameter, efficiency, and acceleration mapping
- **Alignment settings**: Proximity and directional coupling controls

---

## Recommended Experimental Strategies

Based on computational analysis and the demonstrated effectiveness of the hybrid method, the following approaches show highest probability for successful horizon formation and detection:

### Priority 1: Hybrid Fluid + Plasma-Mirror Implementation

1. **Mirror-Enhanced Configuration**: Implement accelerating plasma mirrors proximal to fluid horizon formation regions for comparative evaluation under controlled assumptions.

2. **Alignment Optimization**: Carefully align mirror acceleration with fluid velocity gradients. Misalignment reduces coupling effectiveness.

3. **Conservative Parameter Selection**: Use validated mirror→κ mappings (AnaBHEL model: `κ_mirror = 2π·η_a/D`) rather than optimistic scaling.

---

## Summary of Key Achievements

### 🔬 Primary Innovation: Hybrid Coupling Method
- **Comparative analysis** under controlled assumptions
- **Conservative, physics-based** coupling model explored
- **Apples-to-apples validation** with identical normalization and transmission

### 🧮 Comprehensive Computational Framework
- **Robust horizon detection** with uncertainty quantification
- **First-principles QFT** spectrum calculations
- **Power-conserving** multi-beam envelope modeling
- **26/26 tests passing** with full validation suite

---

## Citation

If you use this computational framework in your research, please cite:

```bibtex
Bown, Hunter. (2025). Analog Hawking Radiation: Gradient-Limited Horizon Formation
and Radio-Band Detection Modeling (Version 0.1.0) [Computer software].
https://github.com/hmbown/analog-hawking-radiation
```

Complete BibTeX citation information is available in `CITATION.cff`.

---

## Next Steps for Development

1. **Complete Validation Testing**: Continue systematic validation outlined in `TESTING_PLAN.md`. Document results and evidence for each validation category.

2. **WarpX Integration**: Install/configure WarpX + pywarpx on target compute environment. Extend `scripts/run_trans_planckian_experiment.py` beyond mock mode for full PIC validation runs.

3. **Secure Computational Resources**: Obtain multi-GPU allocation (≥8×H100/A100) and 10 TB fast storage capacity for Trans-Planckian regime validation campaigns.

4. **Extended Parameter Studies**: Execute comprehensive parameter sweeps including envelope-matched geometry variations, magnetized horizon scans, and PIC/fluid cross-validation studies.

---

## References

See `docs/REFERENCES.md` for primary literature in analog gravity, Hawking radiation, plasma physics, radiometry, and AnaBHEL context used to guide this framework.

**Framework Version**: 0.1.0 | **Last Updated**: October 2025 | **Repository**: https://github.com/hmbown/analog-hawking-radiation
