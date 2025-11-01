# Documentation Guide

Plain‑language guide (start here):

- If you’re new to analog Hawking radiation, read Overview first, then browse the FAQ and Glossary.
- If you’re in a hurry, run `make comprehensive && make results-pack` and share `results/results_pack.zip`.
- If you’re a scientist reviewing methods, jump to Methods, Validation, and Limitations, then see Reproducibility.

This directory contains comprehensive documentation for the Analog Hawking Radiation Simulator. Use this guide to navigate to the information you need.

---

## Quick Start

**New to the project?** Start here:
1. **[Overview.md](Overview.md)** - Conceptual introduction and physics motivation
2. **[Methods.md](Methods.md)** - Core algorithms and computational methods
3. **Installation**: See `README.md` in repository root
4. **[Glossary.md](Glossary.md)** - Key terms in plain words
5. **[FAQ.md](FAQ.md)** - Answers for non‑experts and experts
6. **[Reproducibility.md](Reproducibility.md)** - How to re‑run analyses and dashboards

**Want to run analysis?** Go to:
1. **[Experiments.md](Experiments.md)** - Step-by-step workflow guides
2. **Repository root `README.md`** - Quick start commands

---

## Research & Results

### Latest Research (v0.3.0)
- **[GradientCatastropheAnalysis.md](GradientCatastropheAnalysis.md)** - Complete analysis of κ_max ≈ 5.94×10¹² Hz
- **[Results.md](Results.md)** - Representative outputs and interpretation
- **[Validation.md](Validation.md)** - Physics validation framework and results
- **[DatasetNotes.md](DatasetNotes.md)** - How to read the dataset and avoid common pitfalls

### Historical Documentation
- **[Highlights_v0.2.0.md](Highlights_v0.2.0.md)** - v0.2.0 feature highlights
- **[Final_Answer.md](Final_Answer.md)** - Historical milestone documentation
- **See [CHANGELOG.md](../CHANGELOG.md)** - Complete version history

---

## Technical Documentation

### Core Methodologies
- **[Methods.md](Methods.md)** - Horizon finding, graybody solvers, detection modeling
- **[AdvancedScenarios.md](AdvancedScenarios.md)** - Complex workflows and PIC integration
- **[Integration_testing_report.md](integration_testing_report.md)** - Integration test results

### Experimental Design
- **[Experiments.md](Experiments.md)** - Experiment playbooks and workflows
- **[Successful_Configurations.md](Successful_Configurations.md)** - Proven parameter sets
- **[References.md](References.md)** - Bibliography and recommended reading

### Limitations & Validation
- **[Limitations.md](Limitations.md)** - Scope, assumptions, and open questions
- **[Validation.md](Validation.md)** - Physics validation protocols
- **[AUDIT_NOTES.md](AUDIT_NOTES.md)** - Code audit and review notes

---

## Development & Advanced Usage

### Development Workflow
- **[phase_timeline.md](phase_timeline.md)** - Development roadmap (Phases 3-5)
- **[pic_migration_plan.md](pic_migration_plan.md)** - PIC integration planning
- **[transformation_summary.md](transformation_summary.md)** - Repository transformation summary

### GPU & High-Performance Computing
- **[pc_cuda_workflow.md](pc_cuda_workflow.md)** - Complete GPU setup guide (Windows/WSL2)
- **[advanced_simulations.md](advanced_simulations.md)** - Advanced simulation techniques

### Future Work
- **[trans_planckian_next_steps.md](trans_planckian_next_steps.md)** - Trans-Planckian dispersion studies
- **[upgrade_plan/](upgrade_plan/)** - Technical requirements for Phases 3-5

---

## System Overview

### Documentation Structure

```
docs/
├── 📖 Getting Started
│   ├── Overview.md                 # Conceptual introduction
│   ├── Methods.md                  # Core algorithms
│   ├── Glossary.md                 # Key terms
│   ├── FAQ.md                      # Common questions
│   ├── Reproducibility.md          # End‑to‑end instructions
│   └── References.md               # Bibliography
│
├── 🔬 Research & Results
│   ├── GradientCatastropheAnalysis.md  # κ_max discovery
│   ├── Results.md                      # Output interpretation
│   ├── Validation.md                   # Validation framework
│   └── Highlights_v0.2.0.md            # Previous release
│
├── 🧪 Experiments & Workflows
│   ├── Experiments.md              # Step-by-step guides
│   ├── AdvancedScenarios.md        # Complex workflows
│   ├── Successful_Configurations.md # Proven parameters
│   └── integration_testing_report.md  # Test results
│
├── ⚙️  Advanced & Development
│   ├── advanced_simulations.md     # Advanced techniques
│   ├── pc_cuda_workflow.md         # GPU setup guide
│   ├── phase_timeline.md           # Development roadmap
│   ├── pic_migration_plan.md       # PIC integration
│   └── trans_planckian_next_steps.md # Future work
│
├── 🛠️  System & Maintenance
│   ├── Limitations.md              # Scope & assumptions
│   ├── Validation.md               # Physics checks
│   ├── DatasetNotes.md             # Dataset structure & caveats
│   ├── AUDIT_NOTES.md              # Code audit
│   ├── transformation_summary.md   # Repository changes
│   └── IMAGES.md                   # Figure documentation
│
└── 📦 Upgrades
    └── upgrade_plan/               # Technical requirements
        ├── requirements.md         # Phase 3-5 requirements
        └── workflow_diagram.md     # Workflow visualization
```

---

## Key Research Findings

### v0.3.0 Breakthrough
- **Surface gravity maximum**: κ_max ≈ 5.94×10¹² Hz
- **Optimal configuration**: a₀ ≈ 1.6, nₑ ≈ 1.39×10¹⁹ m⁻³
- **Detection time**: t₅σ ≈ 10⁻⁷ to 10⁻⁶ s
- **Scaling (this run)**: κ vs a₀ exponent ≈ +0.66 (95% CI [0.44, 0.89]); κ vs nₑ exponent ≈ −0.02 (95% CI [−0.14, 0.10])

### Technical Achievements
- **GPU acceleration**: 10-100x speedups
- **PIC integration**: Complete WarpX/openPMD workflow
- **Bayesian inference**: Parameter recovery from experimental data
- **Validation framework**: 42 tests, all passing

---

## Common Tasks

### Generate Figures
```bash
make figures          # All figures
make readme-images    # README-only images
```

### Run Analysis
```bash
# Gradient catastrophe sweep
python scripts/sweep_gradient_catastrophe.py --n-samples 500

# Full pipeline
python scripts/run_full_pipeline.py --demo

# PIC data
python scripts/run_pic_pipeline.py --input-path /path/to/openpmd
```

### Validation
```bash
pytest -q            # All tests
pytest -m gpu        # GPU tests (requires CUDA)
```

---

## File Descriptions

| File | Purpose | Audience |
|------|---------|----------|
| Overview.md | Physics background, concepts | New users, students |
| Methods.md | Algorithms, equations, theory | Researchers, developers |
| GradientCatastropheAnalysis.md | Latest research (κ_max) | All users |
| Experiments.md | How-to guides | Practitioners |
| AdvancedScenarios.md | Complex workflows | Advanced users |
| pc_cuda_workflow.md | GPU setup | Developers |
| phase_timeline.md | Roadmap | Contributors |
| Limitations.md | Constraints, assumptions | Researchers |
| Validation.md | Verification protocols | Reviewers |

---

## Support & Contact

**Questions?** Open an issue on GitHub or contact hunter@shannonlabs.dev

**Contributing?** See `CONTRIBUTING.md` in repository root

**Paper/Collaboration?** See `results/RESEARCH_SUMMARY_v0.3.0.md`

---

## Navigation Tips

1. **Start with Overview.md** if new to the project
2. **Use GradientCatastropheAnalysis.md** for latest research
3. **Check Methods.md** for detailed algorithms
4. **See Experiments.md** for practical workflows
5. **Reference Limitations.md** when interpreting results

**Pro tip**: Most documentation includes links to relevant scripts in `scripts/` directory.

---

*For complete version history, see [CHANGELOG.md](../CHANGELOG.md)*
