Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling
=============================================================================================

Overview
--------

This repository investigates analog Hawking radiation in laser–plasma systems with a focus on horizon formation as the limiting physics. We provide:
- A robust horizon finder with uncertainty (|v| = c_s roots, κ ± err)
- A power-conserving multi-beam field superposition simulator with envelope-scale coarse-graining (no naïve multipliers)
- Radio-band radiometer SNR modeling and sweeps from the actual QFT spectrum

We emphasize clear assumptions, scale choices, and limitations. PIC/fluid validation remains future work.

Key Modules
-----------
- `physics_engine/horizon.py`: sound speed; horizon finding; κ estimates
- `physics_engine/multi_beam_superposition.py`: time-averaged field superposition; gradient and κ surrogates
- `detection/radio_snr.py`: radiometer utilities and sweeps

Reproducibility
---------------
- Make key figures: `make figures`
- Validate frequency gating: `make validate`
- Enhancement summary: `make enhancements`

See also: `docs/Methods.md`, `docs/Results.md`, `docs/Limitations.md`, `docs/Validation.md`.
