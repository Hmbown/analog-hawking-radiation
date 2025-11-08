# Threshold Sensitivity of κ_max

This note explains how to quantify the dependence of the reported κ upper bound
on modeling thresholds.

Usage
```bash
python scripts/sweep_kappa_thresholds.py \
  --n-samples 60 \
  --v-fracs 0.4,0.5,0.6 \
  --dv-max 2e12,4e12,8e12 \
  --intensity-max 1e24 \
  --out results/threshold_sensitivity.json
```

Interpretation
- κ_max is not fundamental in this setting; it depends on the breakdown
  thresholds. This sweep shows how different v/c caps and gradient caps shift
  κ_max.
- Use the results to report a band of plausible κ values rather than a single
  number; include threshold settings alongside κ.

Recommendations
- For outreach, present κ_max with explicit thresholds and provide a link or
  appendix to the threshold sweep results.
- For rigorous work, connect thresholds to independent physics arguments or
  external constraints and document their provenance.
