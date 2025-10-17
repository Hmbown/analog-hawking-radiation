# Analysis Scripts

This directory contains scripts for analyzing analog Hawking radiation in laser-plasma systems.

## Core Analysis Scripts

These scripts implement the main analysis workflows documented in the repository README.

### Primary Pipeline
- **run_full_pipeline.py** - Complete simulation workflow from plasma modeling through detection analysis
  - Usage: `python run_full_pipeline.py --demo`
  - Outputs: `results/full_pipeline_summary.json`
  - Generates horizon detection results, Hawking spectrum, and radio detection metrics

### Parameter Space Exploration
- **run_param_sweep.py** - Systematic parameter space sweeps over temperature and magnetic field
  - Usage: `python run_param_sweep.py [--progress] [--progress-every N]`
  - Outputs: `results/extended_param_sweep.json`
  - Grid sweep across temperature [10⁵, 3×10⁵, 10⁶] K and B-field [None, 0.0, 0.005, 0.02] T

- **compute_formation_frontier.py** - Maps minimum laser intensity thresholds vs density/temperature
  - Usage: `python compute_formation_frontier.py`
  - Outputs: `results/formation_frontier.json`, `figures/formation_frontier.png`
  - Identifies attainable horizon formation parameter regimes

### Optimization and Uncertainty
- **geometry_optimize_kappa.py** - Multi-beam geometry optimization under power-conserving constraints
  - Evaluates rings, crossings, and standing wave configurations
  - Outputs: `results/geometry_vs_kappa.json`, `figures/geometry_vs_kappa.png`

- **monte_carlo_horizon_uncertainty.py** - Statistical robustness analysis via Monte Carlo
  - Quantifies horizon formation probability and κ uncertainty bounds
  - Outputs: `results/horizon_probability_bands.json`, `figures/horizon_probability_bands.png`

## Supporting Scripts

### Figure Generation (used by Makefile)
- **generate_radio_snr_sweep.py** - Radio SNR parameter sweeps
- **radio_snr_from_qft.py** - Time-to-5σ detection heatmaps from QFT spectrum
- **sweep_phase_jitter.py** - Multi-beam phase stability analysis
- **sweep_shapes.py** - Beam profile shape comparisons

### Validation and Analysis
- **script_validate_frequency_gating.py** - Validates frequency band selection logic (used by `make validate`)
- **run_fluid_backend_validation.py** - Fluid backend physics validation
- **validate_physical_configs.py** - Physical parameter range validation
- **validate_fluctuation_statistics.py** - Fluctuation injector statistics validation

### Specialized Studies
- **analyze_cs_profile_impact.py** - Sound speed profile impact on horizon formation
- **scan_Bfield_horizons.py** - Magnetized plasma horizon scans
- **generate_detection_time_heatmap.py** - Detection time parameter heatmaps
- **generate_paramspace_maps.py** - General parameter space mapping
- **generate_guidance_map.py** - Bayesian optimization guidance maps
- **run_bayesian_optimization.py** - Bayesian merit function optimization
- **optimize_for_kappa.py** - Surface gravity maximization
- **optimize_glow_detection.py** - Detection optimization studies

### WarpX Integration (currently mock mode)
- **run_trans_planckian_experiment.py** - Trans-Planckian regime PIC experiments (requires WarpX)
- **run_warpx_horizon_diagnostics.py** - WarpX reduced diagnostics for horizon analysis

### Analysis Tools
- **analysis/analyze_trans_planckian_spectrum.py** - Trans-Planckian spectrum analysis
- **plot_horizon_profiles.py** - Horizon profile visualization
- **hawking_detection_experiment.py** - Detection feasibility experiments
- **test_probabilistic_model.py** - Probabilistic horizon model testing

### Archival Scripts
- **build_success_docs.py** - Documentation generation utility
- **package_paper.py** - Paper/preprint packaging tool

## Exploratory/Development Scripts

Experimental scripts for algorithm development and exploratory analysis are located in:
- **archive_exploratory/** - Contains convergence testing, benchmark studies, and experimental features

These scripts were used during development and validation but are not part of the core analysis workflow.

## Typical Workflow

1. **Quick Start**: `python run_full_pipeline.py --demo`
2. **Parameter Survey**: `python run_param_sweep.py --progress`
3. **Formation Analysis**: `python compute_formation_frontier.py`
4. **Uncertainty Quantification**: `python monte_carlo_horizon_uncertainty.py`
5. **Optimization**: `python geometry_optimize_kappa.py`

## Output Locations

- **results/**: JSON data files with numerical results
- **figures/**: PNG plots and visualizations
- Test outputs logged to console

## Dependencies

All scripts require the `analog_hawking` package installed:
```bash
pip install -e .
```

See main repository README for complete installation instructions.
