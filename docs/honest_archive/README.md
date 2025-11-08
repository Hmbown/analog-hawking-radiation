# Documentation Guide

**Quick start**:
- New to analog Hawking radiation? → [Overview.md](Overview.md) → [FAQ.md](FAQ.md) → [Glossary.md](Glossary.md)
- Need results now? → `make comprehensive && make results-pack` → creates `results/results_pack.zip`
- Reviewing methods? → [Methods.md](Methods.md) → [Validation.md](Validation.md) → [Limitations.md](Limitations.md) → [Reproducibility.md](Reproducibility.md)

**Critical guardrails** - **Read these first when interpreting results**:
- **[DatasetNotes.md](DatasetNotes.md)** - How to read the dataset and avoid common pitfalls
- **[Limitations.md](Limitations.md)** - Model scope, uncertainties, and what not to generalize
- **[Validation.md](Validation.md)** - What has been validated and what hasn't

This directory contains comprehensive documentation for the Analog Hawking Radiation Simulator. Use this guide to navigate to the information you need.

---

## Quick Start

**New to analog Hawking radiation?**
1. **[Overview.md](Overview.md)** - Physics motivation and concepts
2. **[Glossary.md](Glossary.md)** - Key terms in plain language
3. **[FAQ.md](FAQ.md)** - Common questions for all levels

**Ready to run analyses?**
1. **Repository root `README.md`** - Installation and quick commands
2. **[Experiments.md](Experiments.md)** - Step-by-step workflow guides
3. **[Reproducibility.md](Reproducibility.md)** - Exact commands to reproduce results

**Reviewing scientific validity?**
1. **[Methods.md](Methods.md)** - Core algorithms and theory
2. **[Validation.md](Validation.md)** - Physics validation framework
3. **[Limitations.md](Limitations.md)** - Scope, assumptions, and caveats
4. **[DatasetNotes.md](DatasetNotes.md)** - Dataset interpretation guide

---

## Research & Results

### Latest Research (v0.3.0) - **Start here for current findings**
- **[GradientCatastropheAnalysis.md](GradientCatastropheAnalysis.md)** - Complete analysis of κ_max ≈ 5.94×10¹² Hz
- **[Results.md](Results.md)** - Representative outputs and interpretation
- **[reports/](reports/)** - Detailed analysis reports, validation studies, and methodological summaries

### **Critical Context for Interpreting Results**
- **[DatasetNotes.md](DatasetNotes.md)** - ⚠️ **Must read**: How to avoid misinterpreting correlations and scaling
- **[Validation.md](Validation.md)** - What has been validated vs. what remains theoretical
- **[Limitations.md](Limitations.md)** - Model constraints and uncertainty boundaries

### Historical Documentation
- **[Highlights_v0.2.0.md](Highlights_v0.2.0.md)** - v0.2.0 feature highlights
- **See [CHANGELOG.md](https://github.com/Hmbown/analog-hawking-radiation/blob/main/CHANGELOG.md)** - Complete version history

---

## Technical Documentation

### Core Methodologies
- **[Methods.md](Methods.md)** - Horizon finding, graybody solvers, detection modeling
- **[AdvancedScenarios.md](AdvancedScenarios.md)** - Complex workflows and PIC integration
- **[Integration_testing_report.md](integration_testing_report.md)** - Integration test results

### Experimental Design
- **[Experiments.md](Experiments.md)** - Experiment playbooks and workflows
- **[Successful_Configurations.md](Successful_Configurations.md)** - Proven parameter sets
- **[REFERENCES.md](REFERENCES.md)** - Bibliography and recommended reading

### Limitations & Validation
- **[Limitations.md](Limitations.md)** - Scope, assumptions, and open questions
- **[Validation.md](Validation.md)** - Physics validation protocols

---

## Development & Advanced Usage

### Development Workflow
- **[phase_timeline.md](phase_timeline.md)** - Development roadmap (Phases 3-5)
- **[pic_migration_plan.md](pic_migration_plan.md)** - PIC integration planning

### GPU & High-Performance Computing
- **[pc_cuda_workflow.md](pc_cuda_workflow.md)** - Complete GPU setup guide (Windows/WSL2)
- **[advanced_simulations.md](advanced_simulations.md)** - Advanced simulation techniques

### Future Work
- **[trans_planckian_next_steps.md](trans_planckian_next_steps.md)** - Trans-Planckian dispersion studies
- **[upgrade_plan/requirements.md](upgrade_plan/requirements.md)** - Technical requirements for Phases 3-5

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
│   └── REFERENCES.md               # Bibliography
│
├── 🔬 Research & Results
│   ├── GradientCatastropheAnalysis.md  # κ_max discovery
│   ├── Results.md                      # Output interpretation
│   ├── Validation.md                   # Validation framework
│   ├── Highlights_v0.2.0.md            # Previous release
│   └── reports/                        # Detailed analysis reports
│       ├── README.md                   # Reports index
│       ├── ENHANCED_PHYSICS_IMPLEMENTATION_SUMMARY.md
│       ├── ELI_FACILITY_VALIDATION_REPORT.md
│       ├── STATISTICAL_VALIDATION_REPORT.md
│       ├── COMPREHENSIVE_UNCERTAINTY_BUDGET.md
│       ├── DETECTION_FEASIBILITY_ASSESSMENT.md
│       └── ... (see reports/README.md for full list)
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
│   └── IMAGES.md                   # Figure documentation
│
└── 📦 Upgrades & Archive
    ├── upgrade_plan/               # Technical requirements
    │   ├── requirements.md         # Phase 3-5 requirements
    │   └── workflow_diagram.md     # Workflow visualization
    └── archive/                    # Superseded documentation
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
| Methods.md | Algorithms, equations, theory | Scientific users, developers |
| GradientCatastropheAnalysis.md | Latest research (κ_max) | All users |
| reports/ | Detailed analysis reports | Researchers, reviewers |
| Experiments.md | How-to guides | Practitioners |
| AdvancedScenarios.md | Complex workflows | Advanced users |
| pc_cuda_workflow.md | GPU setup | Developers |
| phase_timeline.md | Roadmap | Contributors |
| Limitations.md | Constraints, assumptions | Scientific users |
| Validation.md | Verification protocols | Scientific reviewers |

---

## Contributing

See `CONTRIBUTING.md` in repository root for contribution guidelines.

**Research context**: See `results/RESEARCH_SUMMARY_v0.3.0.md` for detailed research background.

---

## Navigation Tips

1. **Start with Overview.md** if new to the project
2. **Use GradientCatastropheAnalysis.md** for latest research
3. **Check Methods.md** for detailed algorithms
4. **See Experiments.md** for practical workflows
5. **Reference Limitations.md** when interpreting results

**Pro tip**: Most documentation includes links to relevant scripts in `scripts/` directory.

---

*For complete version history, see [CHANGELOG.md](https://github.com/Hmbown/analog-hawking-radiation/blob/main/CHANGELOG.md)*
