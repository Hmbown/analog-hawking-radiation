# Analog Hawking Radiation Simulator

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-26%20passing-brightgreen.svg)](tests/)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases)

![Workflow Pipeline](docs/img/workflow_diagram.png)

A computational framework for modeling analog Hawking radiation in laser-plasma systems. Simulates sonic horizons in flowing plasmas and calculates quantum field theory spectra with novel hybrid fluid-plasma mirror coupling.

## Quick Start

```bash
# Install
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .

# Run standard demo
python scripts/run_full_pipeline.py --demo

# Run with hybrid plasma mirror coupling
python scripts/run_full_pipeline.py --demo --hybrid --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0

# Check results
cat results/full_pipeline_summary.json
```

## Physics Background

### Analog Black Holes
In flowing fluids, sound waves can become trapped when the local flow velocity **v** exceeds the sound speed **c_s**. This creates a **sonic horizon** - an acoustic analog to a black hole's event horizon where information cannot escape upstream.

The key physics parameter is the **surface gravity**:
```
κ = (1/2) * d(v² - c_s²)/dx |_{horizon}
```

Just as black holes emit Hawking radiation due to quantum vacuum fluctuations near the event horizon, these analog systems can produce thermal radiation with temperature:
```
T_H = ħκ/(2πk_B)
```

### AnaBHEL Model
**AnaBHEL** (Analog Black Hole Evaporation in Lasers) provides a phenomenological mapping between accelerating plasma mirrors and effective surface gravity. For a plasma mirror with diameter **D** and acceleration efficiency **η_a**:

```
κ_mirror = 2π * η_a / D
```

This framework allows systematic exploration of how plasma mirror dynamics might enhance analog Hawking signatures.

## What This Framework Does

### Core Capabilities
1. **Horizon Detection**: Identifies sonic horizon formation regions in plasma flow profiles
2. **Quantum Spectra**: Calculates Hawking radiation using near-horizon WKB graybody factors
3. **Hybrid Coupling**: Novel plasma mirror enhancement of fluid horizons 
4. **Radio Detection**: Estimates detectability with realistic antenna parameters

### Key Innovation: Hybrid Fluid-Mirror Coupling
The framework's primary contribution is systematic exploration of **hybrid coupling** between fluid sonic horizons and accelerating plasma mirrors. The effective surface gravity becomes:

```
κ_eff(x) = κ_fluid(x) + w(x) * κ_mirror

w(x) = coupling_strength * exp(-|x - x_mirror|/L_coupling) * alignment_factor
```

This allows comparative analysis of enhanced vs. baseline configurations under controlled assumptions.

## System Requirements

- Python ≥ 3.8 with NumPy ≥ 1.21, SciPy, Matplotlib  
- **Runtime**: Minutes on laptop for demos, hours for full parameter sweeps
- **Validation**: 26/26 unit and integration tests passing

## Repository Structure

```
src/analog_hawking/     # Core physics library
├── physics/           # Horizon detection, QFT calculations
├── detection/         # Radio detection modeling  
└── hybrid/            # Plasma mirror coupling

scripts/               # Analysis and figure generation
tests/                # Comprehensive test suite  
docs/                 # Technical documentation
results/samples/      # Representative outputs
```

## Usage Examples

### Basic Analysis
```bash
# Standard fluid-only analysis
python scripts/run_full_pipeline.py --demo

# Hybrid mirror-enhanced analysis  
python scripts/run_full_pipeline.py --demo --hybrid --hybrid-model anabhel --mirror-D 1e-5 --mirror-eta 1.0
```

### Comparative Studies
```bash
# Direct hybrid vs fluid comparison
python scripts/demo_hybrid_comparison.py

# Parameter sensitivity sweeps
python scripts/sweep_hybrid_params.py

# Detection time analysis
python scripts/generate_detection_time_heatmap.py
```

### Output Interpretation
Key results in `results/full_pipeline_summary.json`:
- **`kappa`**: Surface gravity values (s⁻¹)
- **`T_hawking_K`**: Hawking temperature (K) 
- **`T_sig_K`**: Antenna signal temperature (K)
- **`t5sigma_s`**: 5σ detection time (s) for T_sys=30K, B=100MHz
- **`hybrid_used`**: Boolean flag for hybrid mode

## Scientific Methodology

### Modeling Approach
This framework implements a **conservative, physics-based** approach to analog Hawking radiation:

1. **Horizon Detection**: Systematic identification of sonic horizon regions where ∇(v² - c_s²) changes sign
2. **Quantum Calculation**: Near-horizon WKB approximation for graybody transmission factors
3. **Hybrid Enhancement**: Phenomenological plasma mirror coupling via AnaBHEL mapping
4. **Detection Modeling**: Radiometer-style SNR with configurable system parameters

### Validation Protocol
- **Identical Normalization**: All comparisons use same emitting area (1×10⁻⁶ m²), solid angle (0.05 sr), coupling efficiency (0.1)
- **Conservative Parameters**: AnaBHEL model κ_mirror = 2πη_a/D rather than optimistic scaling
- **Comprehensive Testing**: 26 unit and integration tests covering all physics modules

### Key Assumptions
- **Spatial Scale**: Envelope/skin-depth modeling (no full PIC validation in this repository)  
- **Transmission**: WKB graybody factors near horizons, conservative fallbacks elsewhere
- **Detection**: Radiometer-style SNR with user-configurable T_sys and bandwidth
- **Hybrid Mapping**: Phenomenological mirror→κ relation for comparative analysis

## Limitations and Scope

This is a **computational modeling framework** designed for comparative studies under controlled assumptions:

- **No experimental validation**: PIC/fluid cross-validation is pending
- **Phenomenological mapping**: Absolute calibration of hybrid coupling uncertain  
- **Order-of-magnitude guidance**: Results are not definitive performance predictions
- **Hardware considerations**: Real observatory geometry and noise may shift detectability

**Intended use**: Structured comparative analysis within stated modeling assumptions, not extrapolation beyond scope.

## Documentation

- **`docs/Overview.md`**: Physics background and methodology
- **`docs/Methods.md`**: Detailed computational approaches  
- **`docs/Results.md`**: Example outputs and interpretation
- **`docs/Limitations.md`**: Comprehensive scope discussion
- **`TESTING_PLAN.md`**: Validation methodology and test coverage

## Citation

```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.1.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation}
}
```

## Development Roadmap

1. **Enhanced Validation**: Complete systematic testing outlined in `TESTING_PLAN.md`
2. **PIC Integration**: Full WarpX integration beyond current mock mode  
3. **Extended Studies**: Comprehensive parameter sweeps and cross-validation
4. **Hardware Modeling**: Realistic observatory geometry and noise pipelines

## References

Key literature foundations:
- **Analog Gravity**: Unruh (1981), Jacobson & Volovik (1998), Barceló et al. (2005)
- **Plasma Horizons**: Shukla et al. (2011), Eliasson (2015)  
- **AnaBHEL Framework**: Belgiorno et al. (2010), Faccio & Wright (2013)
- **Detection Theory**: Thompson et al. (2017) - Radio Interferometry

Complete bibliography available in `docs/REFERENCES.md`.

---

**Framework Version**: 0.1.0 | **License**: MIT | **Tests**: 26/26 passing | **Updated**: October 2025
