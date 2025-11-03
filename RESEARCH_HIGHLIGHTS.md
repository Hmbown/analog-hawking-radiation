# Research Highlights

## Parametric Upper Bound on Surface Gravity (v0.3.0)

**Date**: November 2025
**Version**: 0.3.0
**Analysis Module**: `scripts/sweep_gradient_catastrophe.py`

---

## Key Result: κ_max ≈ 5.94T Hz

### Executive Summary

Our systematic analysis of 500 synthetic configurations identified an approximate upper bound on achievable surface gravity in 1D models given specific breakdown thresholds:

**κ_max = 5.94e+12 Hz** (acoustic‑exact κ; thresholds enforced)

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
1. κ vs a₀: exponent ≈ 0.66 (95% CI [0.44, 0.89])
2. κ vs nₑ: exponent ≈ -0.02 (95% CI [-0.14, 0.10])

### Breakdown Statistics
- Valid physics: 68/500 (${valid_rate:.1%})
- Total breakdown rate: ${breakdown_rate:.1%}
- Dominant mode: gradient_catastrophe

---

## ⚠️ Validation Notes (November 2025)

- **Relativistic causality guardrail**: Phase-velocity checks clip at numerical precision (~1e-8 fractional excess). Tolerance relaxation is in progress; group-velocity enforcement remains strict.
- **ADK strong-field monotonicity**: Placeholder tunneling coefficients violate the monotonic increase test above 1e¹² V/m. Recalibration against benchmark atomic data is scheduled before external release.

---

## PIC Tie‑In (synthetic reproduction)

- Horizon positions: -2.00e-01, 2.00e-01
- κ (s⁻¹): 1.00e+00, 1.01e+00
- κ_err: 1.11e-16, 0.00e+00

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
