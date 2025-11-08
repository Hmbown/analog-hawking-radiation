# Analog Hawking Radiation - Quick Links

**Navigation Hub** | [Back to README](../README.md) | [Full Documentation Index](./index.md)

This page provides quick access to the most important documentation and resources.

---

## 🚀 Getting Started (5 minutes)

### Quick Start
- **Try it now**: `ahr quickstart` → [What just happened?](#what-just-happened)
- **System check**: `ahr info` - Verify your setup
- **Validation**: `ahr validate` - Run physics validation suite
- **Tutorials**: `ahr tutorial --list` - Interactive learning

### Installation & Setup
- [Installation Guide](../README.md#2-installation--first-validation-10-minutes)
- [System Requirements](./GPU.md)
- [Development Setup](./CONTRIBUTING.md)

---

## 🎯 Choose Your Path

### 👨‍🔬 Experimental Physicist
**Goal**: Design experiments, forecast detection, plan beam time

**Start here**:
- [Experimental Planning Guide](./ELI_Experimental_Planning_Guide.md)
- `ahr experiment --eli` - ELI facility validation
- [Detection Feasibility Analysis](./docs/GradientCatastropheAnalysis.md)

**Key workflows**:
1. `ahr pipeline --demo` - See detection pipeline
2. `ahr experiment --feasibility` - Check your parameters
3. [Facility-specific guides](./facilities/)

### 🧑‍🔬 Theorist / Analyst
**Goal**: Test models, validate assumptions, explore physics

**Start here**:
- [Methods & Algorithms](./Methods.md)
- [Physics Validation](./Validation.md)
- `ahr validate --dashboard` - See validation status

**Key workflows**:
1. `ahr quickstart` - Understand horizon finding
2. `ahr sweep --gradient` - Explore parameter space
3. [Advanced Physics Models](./Enhanced_Physics_Models_Documentation.md)

### 🎓 Student / Researcher
**Goal**: Learn analog gravity, reproduce results, build intuition

**Start here**:
- [Scientific Narrative](./scientific_narrative.md)
- [Glossary](./Glossary.md)
- `ahr tutorial 1` - Interactive learning

**Key workflows**:
1. `ahr tutorial --list` - Choose your topic
2. [Example Notebooks](../notebooks/)
3. [Conceptual Overview](./Overview.md)

### 💻 Developer
**Goal**: Contribute code, add features, fix bugs

**Start here**:
- [Contributing Guide](../CONTRIBUTING.md)
- `ahr dev --setup` - Development environment
- [Architecture Guide](./project_identity.md)

**Key workflows**:
1. `ahr dev --setup` - Get started
2. `ahr validate` - Run tests
3. [Code Organization](./project_identity.md)

### 🌌 Vibe Coder / Explorer
**Goal**: Play with cool physics, generate visualizations, explore

**Start here**:
- `ahr quickstart` - Immediate results
- [Output Gallery](./IMAGES.md)
- [Example Scripts](../examples/)

**Key workflows**:
1. `ahr quickstart` - See horizons form
2. `ahr pipeline --demo` - Generate plots
3. Experiment with parameters

---

## 📚 Documentation by Topic

### Core Concepts
- [Scientific Overview](./Overview.md) - What is analog Hawking radiation?
- [Methods & Algorithms](./Methods.md) - How does it work?
- [Glossary](./Glossary.md) - Key terms explained
- [FAQ](./FAQ.md) - Common questions

### Physics & Validation
- [Physics Validation](./Validation.md) - How we validate models
- [Gradient Catastrophe Analysis](./GradientCatastropheAnalysis.md) - Physics limits
- [Limitations & Assumptions](./Limitations.md) - What we don't model
- [Enhanced Physics Models](./Enhanced_Physics_Models_Documentation.md) - Advanced features

### Experiments & Facilities
- [ELI Experimental Planning](./ELI_Experimental_Planning_Guide.md) - ELI-specific guide
- [AnaBHEL Comparison](./AnaBHEL_Comparison.md) - Relation to AnaBHEL project
- [Facility Integration](./facilities/) - Connect to real experiments

### Technical Reference
- [CLI Reference](./CLI.md) - Command-line interface
- [Configuration](./configs/) - Parameter files and settings
- [API Documentation](./reference/) - Code reference
- [Architecture](./project_identity.md) - System design

---

## 🛠️ Tools & Commands

### Core CLI Commands
```bash
ahr quickstart          # Quick demo (15 seconds)
ahr validate            # Physics validation
ahr pipeline --demo     # Full pipeline demo
ahr sweep --gradient    # Parameter sweeps
ahr info                # System information
ahr docs                # Open documentation
ahr tutorial --list     # List tutorials
```

### Development Commands
```bash
ahr dev --setup         # Development setup
make lint               # Code formatting
make test               # Run tests
make results-pack       # Package results
```

### Make Targets
```bash
make quickstart         # Quickstart demo
make validate           # Validation suite
make comprehensive      # Full analysis
make help               # List all targets
```

---

## 📊 Validated vs Experimental

### ✅ Validated Features (Production Ready)
- **Horizon finding** - Sonic horizon detection
- **Graybody models** - Acoustic WKB & exact methods
- **Detection modeling** - Radio frequency predictions
- **Parameter sweeps** - Gradient catastrophe analysis
- **CLI & pipelines** - Core workflows

### ⚠️ Experimental Features (Use with Caution)
- **Plasma mirror coupling** - Hybrid models
- **Enhanced relativity** - Advanced physics modules
- **nD horizons** - Multi-dimensional analysis
- **Trans-Planckian** - Beyond standard models

**Always check**: [Current Limitations](./Limitations.md)

---

## 📁 Repository Structure

```
analog-hawking-radiation/
├── ahr                    # CLI command (install with pip install -e .)
├── src/analog_hawking/    # Core package
├── scripts/               # Analysis scripts
├── docs/                  # Documentation
├── examples/              # Example code
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── results/               # Output directory
└── tests/                 # Test suite
```

---

## 🔬 Scientific Context

### Key Publications
- [Chen & Mourou (2017)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.118.045001) - Plasma mirror concept
- [Chen et al. (2022)](https://www.mdpi.com/2304-6732/9/12/1003) - AnaBHEL experiment
- [Unruh (1981)](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.46.1351) - Original analog gravity proposal

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

See [CITATION.cff](../CITATION.cff) for full details.

---

## 🤝 Community & Support

### Getting Help
- **GitHub Issues**: [Report bugs](https://github.com/hmbown/analog-hawking-radiation/issues)
- **GitHub Discussions**: [Ask questions](https://github.com/hmbown/analog-hawking-radiation/discussions)
- **Documentation**: [Full docs](./index.md)
- **Email**: [hunter@shannonlabs.dev](mailto:hunter@shannonlabs.dev)

### Contributing
- [Contributing Guide](../CONTRIBUTING.md)
- [Code of Conduct](../CODE_OF_CONDUCT.md)
- [Development Setup](./CONTRIBUTING.md#development-setup)

---

## 🎯 Next Steps

**Just getting started?**
1. Run `ahr quickstart`
2. Read [What just happened?](#what-just-happened)
3. Try `ahr tutorial 1`
4. Choose your path above

**Ready for more?**
- [Advanced Scenarios](./AdvancedScenarios.md)
- [Experimental Workflows](./playbooks.md)
- [Parameter Sweeps](./GradientCatastropheAnalysis.md)

**Need something specific?**
- [Search documentation](./) - Use GitHub search in docs/
- [Ask a question](https://github.com/hmbown/analog-hawking-radiation/discussions) - GitHub Discussions
- [Report an issue](https://github.com/hmbown/analog-hawking-radiation/issues) - GitHub Issues

---

*Last updated: 2025-11-08* | [Edit this page](./QUICKLINKS.md)
