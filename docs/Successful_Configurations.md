# Successful Configurations

This report summarizes configurations from the extended sweep that formed horizons.

- Figures: `figures/horizon_analysis_probability_map.png`, `figures/horizon_analysis_kappa_map.png`, `figures/horizon_analysis_TH_map.png`, `figures/horizon_analysis_profile_*.png`, `figures/horizon_analysis_detection_time.png`
- Data: `results/extended_param_sweep.json`, `results/horizon_success_cases.json`

## Top 10 by κ

| n_e [cm^-3] | I [W/cm^2] | T [K] | B [T] | κ_max [s^-1] | t_5σ [s] |
|---:|---:|---:|---:|---:|---:|
| 1.000e+17 | 1.000e+19 | 1.000e+04 | 0.00 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+04 | 0.05 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+04 | 0.10 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+05 | 0.00 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+05 | 0.05 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+05 | 0.10 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+06 | 0.00 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+06 | 0.05 | 8.195e+14 | 9.932e+105 |
| 1.000e+17 | 1.000e+19 | 1.000e+06 | 0.10 | 8.195e+14 | 9.932e+105 |
| 3.162e+17 | 1.000e+19 | 1.000e+04 | 0.00 | 8.195e+14 | 9.932e+105 |

### Notes
- κ values exceed the 1e10 s^-1 target in multiple regions of parameter space.
- Current QFT spectrum normalization yields very small in-band powers in radio bands; `t_5σ` values remain extremely large.
- See `results/physical_validation_report.json` for a0 and ω_p checks.