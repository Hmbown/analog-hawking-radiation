# 📚 Analog Hawking Radiation - Documentation Hub

**Quick Navigation**: [Overview](index.md) | [Quick Links](./QUICKLINKS.md) | [Getting Started](./getting-started/)

Welcome to the documentation hub for the Analog Hawking Radiation toolkit. This page helps you find what you need quickly.

---

## ✅ Status at a Glance

| Component | Status | What It Does | Where to Start |
|-----------|--------|--------------|----------------|
| **Horizon Finding** | ✅ Validated | Detect sonic horizons in plasma flows | [`ahr quickstart`](./playbooks.md) |
| **Graybody Models** | ✅ Validated | Compute Hawking radiation spectra | [`docs/Methods.md`](./Methods.md) |
| **Detection Modeling** | ✅ Validated | Forecast radio detection times | [`docs/GradientCatastropheAnalysis.md`](./GradientCatastropheAnalysis.md) |
| **Parameter Sweeps** | ✅ Validated | Map physics breakdown boundaries | [`ahr sweep --gradient`](./playbooks.md) |
| **CLI & Pipelines** | ✅ Validated | Reproducible analysis workflows | [`README.md`](../index.md) |
| **Plasma Mirror Coupling** | ⚠️ Experimental | Hybrid fluid-mirror models | [`docs/Limitations.md`](./Limitations.md) |
| **Enhanced Relativity** | ⚠️ Experimental | Advanced physics modules | [`docs/Enhanced_Physics_Models_Documentation.md`](./Enhanced_Physics_Models_Documentation.md) |
| **nD Horizons** | 🔬 Prototype | Multi-dimensional analysis | [`docs/horizon_nd.md`](./horizon_nd.md) |

**Legend**: ✅ Validated (production-ready) | ⚠️ Experimental (use with caution) | 🔬 Prototype (research code)

---

## 🎯 Choose Your Path

### 👨‍🔬 [Experimentalist Guide](./getting-started/experimentalist.md)
Design experiments, forecast detection, plan beam time

**Start**: `ahr experiment --eli`

### 🧑‍🔬 [Theorist Guide](./getting-started/theorist.md)
Test models, validate assumptions, explore physics

**Start**: `ahr validate --dashboard`

### 🎓 [Student Guide](./getting-started/student.md)
Learn analog gravity, reproduce results, build intuition

**Start**: `ahr tutorial 1`

### 💻 [Developer Guide](./getting-started/developer.md)
Contribute code, add features, fix bugs

**Start**: `ahr dev --setup`

---

## 🚀 Quick Actions

### Try It Now
```bash
ahr quickstart          # 15-second demo
ahr info               # System information
ahr validate           # Physics validation
ahr tutorial --list    # Interactive tutorials
```

### Common Workflows
```bash
ahr pipeline --demo              # Full detection pipeline
ahr sweep --gradient            # Parameter exploration
ahr experiment --eli            # Facility planning
make comprehensive              # Complete analysis suite
```

### Documentation Navigation
```bash
ahr docs               # Open documentation
ahr docs --path       # Show documentation paths
```

---

## 📖 Documentation by Topic

### Getting Started
- [Quick Links](./QUICKLINKS.md) - Navigation hub
- [Playbooks](./playbooks.md) - Common workflows
- [Glossary](./Glossary.md) - Key terms explained
- [FAQ](./FAQ.md) - Frequently asked questions

### Physics & Methods
- [Scientific Overview](./Overview.md) - Conceptual introduction
- [Methods & Algorithms](./Methods.md) - Technical details
- [Validation Framework](./Validation.md) - How we test
- [Gradient Catastrophe Analysis](./GradientCatastropheAnalysis.md) - Physics limits

### Experiments & Facilities
- [ELI Experimental Planning](./ELI_Experimental_Planning_Guide.md) - Facility-specific guide
- [AnaBHEL Comparison](./AnaBHEL_Comparison.md) - Relation to AnaBHEL project
- [Facility Integration](./facilities/) - Connect to real experiments

### Development & Contribution
- [Contributing Guide](../CONTRIBUTING.md) - How to contribute
- [Developer Guide](./getting-started/developer.md) - Development setup
- [Architecture](./project_identity.md) - System design
- [Code of Conduct](../CODE_OF_CONDUCT.md) - Community standards

### Advanced Topics
- [Enhanced Physics Models](./Enhanced_Physics_Models_Documentation.md) - Advanced features
- [Limitations & Assumptions](./Limitations.md) - Scope and caveats
- [GPU Acceleration](./GPU.md) - Performance optimization
- [nD Horizons](./horizon_nd.md) - Multi-dimensional analysis

---

## 🔬 Scientific Context

### Key Findings (v0.3.0)
- **κ_max ≈ 5.94×10¹² Hz** - Threshold-limited upper bound
- **Scaling**: κ ∝ a₀^0.66, κ ∝ nₑ^-0.02
- **Detection times**: 10⁻⁷ - 10⁻³ s for realistic parameters

### How to Cite
```bibtex
@software{bown2025analog,
  author = {Bown, Hunter},
  title = {Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling},
  version = {0.3.0},
  year = {2025},
  url = {https://github.com/hmbown/analog-hawking-radiation}
}
```

See [CITATION.cff](../CITATION.cff) for full metadata.

---

## 🤝 Community & Support

### Getting Help
- **GitHub Issues**: [Report bugs](https://github.com/hmbown/analog-hawking-radiation/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/hmbown/analog-hawking-radiation/discussions)
- **Email**: [hunter@shannonlabs.dev](mailto:hunter@shannonlabs.dev)

### Contributing
- [Contributing Guide](../CONTRIBUTING.md)
- [Development Setup](./getting-started/developer.md)
- [Good First Issues](https://github.com/hmbown/analog-hawking-radiation/labels/good-first-issue)

---

## 📊 Repository Statistics

- **42** physics validation tests
- **90+** analysis scripts
- **40+** documentation pages
- **500+** test cases
- **10-100×** GPU speedup

---

<div align="center">

**[Back to README](../index.md)** | **[Quick Links](./QUICKLINKS.md)** | **[Getting Started](./getting-started/)**

*Laboratory Black Hole Detection, Quantified*

</div>
