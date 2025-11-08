# Regression & Validation

To lock down scientific outputs and catch regressions, use the golden regression check:

```bash
ahr regress
```

This compares the quickstart horizon positions and surface gravity against a golden baseline
(`goldens/quickstart_horizons.json`) within specified tolerances.

- Positions: absolute tolerance (`positions_abs`)
- Kappa: relative tolerance (`kappa_rel`)

For broader physics validation with analytic checks and convergence tests, run:

```bash
ahr validate --report results/validation_summary.json
```

The report JSON includes a pass/fail summary and categorized details.
