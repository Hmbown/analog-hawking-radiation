Analog Hawking Radiation: Gradient-Limited Horizon Formation and Radio-Band Detection Modeling
=============================================================================================

Overview
--------

This repository investigates analog Hawking radiation in laser-plasma systems with a focus on horizon formation as the limiting physics. The framework provides:

* A robust horizon finder with uncertainty quantification (|v| = c_s roots, κ ± err)
* A power-conserving multi-beam field superposition simulator with envelope-scale coarse-graining (avoiding naive multiplicative assumptions)
* Radio-band radiometer signal-to-noise ratio modeling and parameter sweeps from the actual quantum field theory spectrum

The implementation emphasizes clear assumptions, scale choices, and limitations. Particle-in-cell (PIC) and fluid validation remains future work.

Key Modules
-----------

* `physics_engine/horizon.py`: Sound speed calculation; horizon finding algorithms; surface gravity estimates
* `physics_engine/multi_beam_superposition.py`: Time-averaged field superposition; gradient and surface gravity surrogate calculations
* `detection/radio_snr.py`: Radiometer equation utilities and detection parameter sweeps

Reproducibility
---------------

To reproduce the key results and figures:

* Generate key figures: `make figures`
* Validate frequency gating: `make validate`
* Generate enhancement summary: `make enhancements`

Related Documentation
---------------------

For additional information, see also:

* `docs/Methods.md` - Detailed methodology and implementation
* `docs/Results.md` - Summary of key findings and numerical results
* `docs/Limitations.md` - Discussion of current limitations and uncertainties
* `docs/Validation.md` - Description of validation protocols and testing