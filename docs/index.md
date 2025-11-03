# AnaBHEL Collaboration Guide

Welcome to the working documentation hub for the analog Hawking radiation toolkit.
Start here to understand which pieces are validated, what remains experimental,
and where to find deeper dives.

## Status at a Glance

| Component | Status | Where to Start | Notes |
|-----------|--------|----------------|-------|
| Horizon finding (`physics_engine/horizon*.py`) | ✅ Validated | `README.md` quickstart; [`docs/playbooks.md`](playbooks.md#validated-workflows) | CI-covered; SI units enforced |
| Graybody and detection (`detection/graybody_nd.py`) | ✅ Validated | [`docs/GradientCatastropheAnalysis.md`](GradientCatastropheAnalysis.md) | Benchmarked against analytic limits |
| Enhanced relativistic modules (`physics_engine/enhanced_*`) | ⚠️ Experimental | [`docs/Limitations.md`](Limitations.md#experimental-enhanced-modules) | Emits runtime warnings; collaboration scaffolding |
| Plasma mirror mapping | ⚠️ Experimental | [`src/analog_hawking/physics_engine/plasma_mirror.py`](../src/analog_hawking/physics_engine/plasma_mirror.py) | Chen & Mourou (2017) mapping; hybrid coupling |

## Navigation

- **Validated Physics** – [`README.md`](../README.md#scope-of-validated-physics)
- **Analysis Findings** – [`docs/GradientCatastropheAnalysis.md`](GradientCatastropheAnalysis.md)
- **Limitations & Assumptions** – [`docs/Limitations.md`](Limitations.md)
- **AnaBHEL Context** – [`docs/AnaBHEL_Comparison.md`](AnaBHEL_Comparison.md)
- **Common Workflows** – [`docs/playbooks.md`](playbooks.md)

## Quick Actions

- Run smoke tests or the full validation sweep → see [`docs/playbooks.md`](playbooks.md#validated-workflows)
- Share summary with collaborators → reference [`docs/AnaBHEL_Comparison.md`](AnaBHEL_Comparison.md#interpretation-and-literature)
- Deep-dive the latest gradient campaign → [`RESEARCH_HIGHLIGHTS.md`](../RESEARCH_HIGHLIGHTS.md)

---

Next: [Common Playbooks »](playbooks.md)
