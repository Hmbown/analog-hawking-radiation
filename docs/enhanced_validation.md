# Enhanced Validation Framework

This framework performs professional-grade validation of the analog Hawking radiation pipeline by combining parameter-space exploration with physics-backed horizon detection and uncertainty quantification.

Key features:
- Latin Hypercube sampling across 5D space (`a0`, `n_e`, gradient factor, `T_e`, `B`) with grid `resolution`.
- Physics-based κ estimation using `analog_hawking.physics_engine.horizon.find_horizons_with_uncertainty`.
- Uncertainty budget partitioned into statistical, numerical, physics-model, and experimental components.
- Detection feasibility via radiometer equation using `system_temperature_K` and `bandwidth_Hz`.

Run locally:
- Ensure the `src/` package is importable (editable install or rely on the script’s automatic `sys.path` fallback).
- Optional knobs in `configs/thresholds.yaml`:
  - `system_temperature_K` (default 50 K)
  - `bandwidth_Hz` (default 1e9 Hz)

CLI:
```bash
# Default: 100 configs
python enhanced_validation_framework.py

# Smaller smoke run
python enhanced_validation_framework.py --n-configs 25 --seed 42

# Custom config/results directory
python enhanced_validation_framework.py --config configs/thresholds.yaml --results-dir results/enhanced_validation
```

Outputs:
- `results/enhanced_validation/enhanced_sweep_*.h5` — per-configuration DataFrame
- `results/enhanced_validation/enhanced_analysis_*.json` — summary, uncertainty budget, validation tests
- `results/enhanced_validation/validation_report_*.md` — human-readable report
  - If HDF5 (PyTables) is unavailable, the framework falls back to `enhanced_sweep_*.csv`.
  - Quick-look figures `kappa_hist_*.png` and `detection_time_hist_*.png` are generated unless `ANALOG_HAWKING_NO_PLOTS=1`.

Notes:
- If the physics modules are unavailable at runtime, the framework falls back to statistically reasonable placeholders and clearly logs the fallback.
- κ is computed via the “acoustic_exact” definition by default; see `src/analog_hawking/physics_engine/horizon.py` for alternatives.
