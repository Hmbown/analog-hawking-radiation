# Research Highlights

## Latest Discovery: Fundamental Limit of Analog Hawking Radiation (v0.3.0)

**Date**: ${date}
**Version**: 0.3.0
**Analysis Module**: `scripts/sweep_gradient_catastrophe.py`

---

## 🏆 Key Discovery: κ_max ≈ ${kappa_max_pretty} Hz

### Executive Summary

Our systematic analysis of ${n_samples} laser-plasma configurations identified a fundamental physical limit on achievable surface gravity in analog Hawking systems:

**κ_max = ${kappa_max_sci} Hz** (acoustic‑exact κ; thresholds enforced)

---

## 📊 Optimal Configuration

The configuration achieving maximum κ (this production run):

| Parameter | Value |
|-----------|-------|
| **Laser amplitude (a₀)** | ${opt_a0} |
| **Plasma density (nₑ)** | ${opt_ne_pretty} m⁻³ |
| **Gradient factor** | ${opt_grad} |
| **Required intensity** | ${opt_intensity_pretty} W/m² |

---

## 🔬 Methodology

Parameter sweep over a₀ ∈ [1, 100], nₑ ∈ [1e18, 1e22] m⁻³, gradient factor ∈ [1, 1000]. Physics breakdown enforced via thresholds (see configs/thresholds.yaml). Surface gravity computed with `kappa_method="acoustic_exact"` at horizon crossings.

---

## 🧪 Key Findings

### Scaling Relationships
1. κ vs a₀: exponent ≈ ${exp_a0} (95% CI [${exp_a0_lo}, ${exp_a0_hi}])
2. κ vs nₑ: exponent ≈ ${exp_ne} (95% CI [${exp_ne_lo}, ${exp_ne_hi}])

### Breakdown Statistics
- Valid physics: ${valid}/${n_samples} (${valid_rate:.1%})
- Total breakdown rate: ${breakdown_rate:.1%}
- Dominant mode: ${dominant_mode}

---

## 📡 PIC Tie‑In (synthetic reproduction)

- Horizon positions: ${pic_horizons}
- κ (s⁻¹): ${pic_kappas}
- κ_err: ${pic_kappa_errs}

---

## 🏃 Getting Started

```bash
python scripts/sweep_gradient_catastrophe.py --n-samples 500 \\
  --output results/gradient_limits_production \\
  --thresholds configs/thresholds.yaml
```

---

## 📞 Outreach & Collaboration

This research is ready for peer review and collaboration. See outreach/.
