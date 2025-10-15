This archive accompanies the manuscript:

  Analog Hawking Radiation in Laser--Plasma Flows: Horizon Formation, Parameter Exploration, and Detectability

Author: Hunter Bown
Date: October 15, 2025

Contents
--------
- main.tex                         LaTeX manuscript (arXiv-ready; standard article class)
- figures/                         Final figures used in the paper
- results/                         JSON summaries of sweeps and validations
- code/                            Ancillary Python sources referenced in the manuscript
- requirements.txt                 Python dependency hints for regeneration

Key scripts (code/scripts/)
---------------------------
- run_param_sweep.py               Parameter exploration over I, n_e, T, B
- run_full_pipeline.py             End-to-end run including horizon detection and detection metrics
- generate_paramspace_maps.py      Builds kappa and T_H maps
- generate_detection_time_heatmap.py  Builds detection-time figures (PSD and T_H-surrogate)
- plot_horizon_profiles.py         Plots v(x), c_s(x) near horizons for success cases
- hawking_detection_experiment.py  Spectrum helper (applies graybody, normalization knobs)

Core modules (code/src/analog_hawking/...)
------------------------------------------
- physics_engine/plasma_models/fluid_backend.py       Fluid model with intensity scaling toggle
- physics_engine/plasma_models/quantum_field_theory.py QFT spectrum with graybody and normalization
- detection/radio_snr.py                              Radiometer utilities

Quick regeneration (example)
---------------------------
1) Python environment
   python3 -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt

2) Recreate sweep + figures (ensure working directory is repository root)
   python3 scripts/run_param_sweep.py --nI 5 --nN 5 --nT 3 --nB 3 --grid_points 512
   python3 scripts/generate_paramspace_maps.py
   python3 scripts/plot_horizon_profiles.py
   python3 scripts/generate_detection_time_heatmap.py

Notes
-----
- PSD-based detection time is conservative and sensitive to normalization and coupling. T_H-surrogate offers an upper-bound feasibility view.
- Figures in this package reflect the latest run at the time of submission.
