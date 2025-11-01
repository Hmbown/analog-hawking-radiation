# Research Highlights

## Parametric Upper Bound on Surface Gravity (v0.3.0)

**Date**: October 2025
**Version**: 0.3.0
**Analysis Module**: `scripts/sweep_gradient_catastrophe.py`

---

## Key Result: κ_max ≈ 5.94T Hz

### Executive Summary

Our systematic analysis of 500 synthetic laser–plasma configurations identified an approximate upper bound on achievable surface gravity in our 1D models, given specific breakdown thresholds:

**κ_max = 5.9405655557e+12 Hz** (acoustic‑exact κ; thresholds enforced)

---

## Configuration at Upper Bound (this run)

The configuration achieving maximum κ (this production run):

| Parameter | Value |
|-----------|-------|
| **Laser amplitude (a₀)** | 6.95 |
| **Plasma density (nₑ)** | 1.00e+20 m⁻³ |
| **Gradient factor** | 2.15 |
| **Required intensity** | 1.03e+24 W/m² |

---

## 🔬 Methodology

Parameter sweep over a₀ ∈ [1, 100], nₑ ∈ [1e18, 1e22] m⁻³, gradient factor ∈ [1, 1000]. Physics breakdown enforced via thresholds (see configs/thresholds.yaml). Surface gravity computed with `kappa_method="acoustic_exact"` at horizon crossings.

---

## 🧪 Key Findings

### Scaling Relationships
1. κ vs a₀: exponent ≈ 0.664 (95% CI [0.441, 0.888])
2. κ vs nₑ: exponent ≈ -0.0200 (95% CI [-0.1367, 0.0967])

### Breakdown Statistics
- Valid physics: 68/500 (${valid_rate:.1%})
- Total breakdown rate: ${breakdown_rate:.1%}
- Dominant mode: gradient_catastrophe

---

## PIC Tie‑In (synthetic reproduction)

- Horizon positions: 3.12e-05, 5.20e-05
- κ (s⁻¹): 2.39e+10, 2.40e+10
- κ_err: 3.43e-05, 6.96e-06

---

## Getting Started

```bash
python scripts/sweep_gradient_catastrophe.py --n-samples 500 \\
  --output results/gradient_limits_production \\
  --thresholds configs/thresholds.yaml
```

---

## Collaboration

This is a preliminary computational study. We welcome feedback from the community; see `outreach/`.
