# Analog Hawking Radiation Simulator 🌌

[![Python Version](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![CI](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml/badge.svg)](https://github.com/hmbown/analog-hawking-radiation/actions/workflows/ci.yml) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/) [![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases) [![Cite](https://img.shields.io/badge/Cite-CITATION.cff-orange.svg)](CITATION.cff)

[![κ_max](https://img.shields.io/badge/κ_max-5.94×10¹²%20Hz-red.svg)](RESEARCH_HIGHLIGHTS.md) [![GPU Speedup](https://img.shields.io/badge/GPU-10--100×-green.svg)](docs/pc_cuda_workflow.md) [![Validation](https://img.shields.io/badge/validations-42%20tests-brightgreen.svg)](tests/)

> **Laboratory Black Hole Detection, Quantified**  
> A practitioner-focused toolkit for designing and validating analog Hawking radiation experiments in realistic laser–plasma settings. We transform plasma flow analysis into actionable experimental guidance through rigorous physics validation and uncertainty quantification.

**In 30 seconds**: We predict when laser-created plasma flows form "sonic horizons" and estimate if the resulting Hawking-like radiation is measurable with radio detectors.

---

## ⚡ Try It Now (15 seconds)

```bash
# Clone and run
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
ahr quickstart
```

**What just happened?**
1. 🌊 Created a synthetic plasma flow profile
2. 🎯 Detected sonic horizon(s) where flow speed equals sound speed  
3. ⚡ Computed surface gravity κ (governs Hawking temperature)
4. 📊 Saved results and visualization

**See the results**: `open results/quickstart/quickstart_profile.png`

**Next steps**:
```bash
ahr pipeline --demo       # Full detection pipeline
ahr tutorial 1            # Learn the physics
ahr docs                  # Open documentation
```

---

## 🎯 Choose Your Path

<table>
<tr>
<td width="25%">

### 👨‍🔬 Experimentalist
**Design experiments & forecast detection**

**Start**: `ahr experiment --eli`

**Learn**: [Experimental Planning](./docs/ELI_Experimental_Planning_Guide.md)

</td>
<td width="25%">

### 🧑‍🔬 Theorist
**Test models & validate physics**

**Start**: `ahr validate --dashboard`

**Learn**: [Methods & Algorithms](./docs/Methods.md)

</td>
<td width="25%">

### 🎓 Student
**Learn analog gravity concepts**

**Start**: `ahr tutorial 1`

**Learn**: [Scientific Narrative](./docs/scientific_narrative.md)

</td>
<td width="25%">

### 💻 Developer  
**Contribute code & features**

**Start**: `ahr dev --setup`

**Learn**: [Contributing Guide](./CONTRIBUTING.md)

</td>
</tr>
</table>

**Not sure?** → [Read the scientific narrative](./docs/scientific_narrative.md) or [explore the glossary](./docs/Glossary.md)

---

## 🏗️ Architecture in 30 Seconds

```
Laser Pulse → Plasma Flow → Sonic Horizon → Hawking Radiation → Radio Detection
     ↓              ↓              ↓               ↓               ↓
  Profile      Hydrodynamic    Critical       Quantum        Antenna
  Generation   Simulation      Surface        Field          + Signal
                              Gravity        Theory         Processing
```

The simulator links fluid models, particle-in-cell (PIC) pipelines, quantum field theory post-processing, and radio detection forecasts into one reproducible environment where every assumption is documented and every uncertainty propagated.

---

## ✅ Validated vs Experimental

| Component | Status | Use Case |
|-----------|--------|----------|
| **Horizon Finding** | ✅ Validated | Production analysis |
| **Graybody Models** | ✅ Validated | Detection forecasts |
| **Parameter Sweeps** | ✅ Validated | Systematic studies |
| **Plasma Mirror Coupling** | ⚠️ Experimental | Scenario planning |
| **Enhanced Relativity** | ⚠️ Experimental | Research exploration |
| **nD Horizons** | 🔬 Prototype | Method development |

**Always check**: [Current Limitations](./docs/Limitations.md) before publication

---

## 📊 Latest Research (v0.3.0 - October 2025)

- **Threshold-limited sweep yields κ_max ≈ 5.94×10¹² Hz** with acoustic-exact κ and enforced breakdown thresholds
- **Scaling**: κ ∝ a₀^0.66 (95% CI [0.44, 0.89]); κ ∝ nₑ^-0.02 (95% CI [-0.14, 0.10])
- **Velocity < 0.5c**, |dv/dx| < 4×10¹² s⁻¹, intensity < 1×10²⁴ W/m² (1D theoretical cap, exceeds current ELI facilities)

📄 **[Research Highlights](RESEARCH_HIGHLIGHTS.md)** | 📊 **[Gradient Catastrophe Analysis](docs/GradientCatastropheAnalysis.md)** | 🧭 **[Full Documentation](docs/index.md)**

---

## 🎮 Core Commands

```bash
# Discovery & Learning
ahr quickstart          # 15-second demo
ahr tutorial --list     # Interactive tutorials
ahr docs                # Open documentation

# Validation & Testing  
ahr validate            # Physics validation
ahr validate --dashboard # Visual validation status
ahr bench               # Performance benchmarks

# Analysis & Experiments
ahr pipeline --demo     # Full detection pipeline
ahr sweep --gradient    # Parameter space exploration
ahr experiment --eli    # Facility-specific planning

# Development
ahr dev --setup         # Development environment
ahr info                # System information
```

**See all commands**: `ahr --help`

---

## 📦 Results Package

Generate complete results package for sharing or publication:

```bash
make comprehensive && make results-pack
```

**Package includes**:
- 📊 **Figures**: 4 curated plots (speedup, detection, enhancement, Pareto)
- 📄 **Data**: hybrid_sweep.csv (20 configurations, 5 coupling strengths)  
- 📝 **Summary**: RESULTS_README.md with 1-page overview
- 🔬 **Documentation**: Reproducibility notes, dataset notes, limitations
- 📚 **Citation**: CITATION.cff + BibTeX format

---

## 🧪 Quick Validation

```bash
# One-minute smoke test
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-verified.txt
pytest -q                                  # Verify environment (42 tests pass)

# Run demo
ahr quickstart
```

> Default pytest discovery is scoped to `tests/` to keep core runs self-contained. Install optional extras and invoke `pytest scripts/` if you need to exercise the demo pipelines.

---

## 🔬 Scientific Context

Analog black holes form where the flow speed |v| exceeds the local sound speed c_s, creating a sonic horizon. The associated surface gravity governs the Hawking temperature via T_H = ħκ/(2πk_B). This framework implements multiple κ definitions, graybody transmission models, and radio detection estimates to assess whether realistic laser–plasma profiles can produce measurable thermal signatures.

The optional hybrid branch couples fluid horizons to accelerating plasma mirrors inspired by the AnaBHEL program (Chen & Mourou 2017; Chen et al. 2022). Treat these modes as computational thought experiments rather than validated predictions.

---

## 📚 Documentation Hub

**Getting Started**:
- [Quick Links](./docs/QUICKLINKS.md) - Navigation hub
- [Playbooks](./docs/playbooks.md) - Common workflows
- [Glossary](./docs/Glossary.md) - Terms explained
- [FAQ](./docs/FAQ.md) - Common questions

**Deep Dives**:
- [Methods & Algorithms](./docs/Methods.md) - Technical details
- [Gradient Catastrophe Analysis](./docs/GradientCatastropheAnalysis.md) - Physics limits
- [Validation Framework](./docs/Validation.md) - How we test
- [Limitations & Assumptions](./docs/Limitations.md) - Scope & caveats

---

## 🤝 Contributing

We welcome contributions from the community!

**Ways to contribute**:
- Report bugs via [GitHub Issues](https://github.com/hmbown/analog-hawking-radiation/issues)
- Request features via [GitHub Discussions](https://github.com/hmbown/analog-hawking-radiation/discussions)
- Improve documentation (see [docs/](./docs/))
- Add tests for edge cases
- Submit pull requests for new features

**Getting started**:
1. Read [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines
2. Run `ahr dev --setup` for development environment
3. Browse issues labeled `good-first-issue`

---

## 📖 Citation

If you use this work, please cite both the framework and foundational research:

**This framework**:
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation}
}
```

**Foundational AnaBHEL work**:
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

See [CITATION.cff](./CITATION.cff) for machine-readable metadata.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

This work builds on the pioneering AnaBHEL program and the broader analog gravity community. Special thanks to all contributors and early users who provided feedback and bug reports.

---

<div align="center">

**[Quick Links](./docs/QUICKLINKS.md)** | **[Full Documentation](./docs/index.md)** | **[Tutorials](./docs/FAQ.md#tutorials)**

*Laboratory Black Hole Detection, Quantified*

</div>
