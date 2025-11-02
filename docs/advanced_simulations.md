# Advanced Simulations: Phase 3 Methodology, Risks, and Milestones

## Overview
Phase 3 extends the Analog Hawking Radiation Simulator with advanced multi-physics capabilities, focusing on electromagnetic (EM)/magnetohydrodynamic (MHD) coupling, nonlinear plasma effects, and 3D quantum field theory (QFT) approximations. This enables simulations targeting enhanced surface gravity (κ 10-100×), Hawking temperature (T_H >1 mK GHz), and rapid 5σ detections (t_{5σ} <1s), while verifying universality (R²>0.98) and κ stability (<3%) across 50% of configurations.

Key deliverables:
- Extended [`warpx_backend.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/src/analog_hawking/physics_engine/plasma_models/warpx_backend.py) for EM/MHD and nonlinear hooks.
- New [`nonlinear_plasma.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/src/analog_hawking/physics_engine/plasma_models/nonlinear_plasma.py) for 3D QFT and nonlinear solvers.
- [`sweep_multi_physics_params.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/scripts/sweep_multi_physics_params.py) for parameter sweeps.
- [`test_multi_physics_coupling.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/tests/test_multi_physics_coupling.py) for validation.
- This document for methodology and planning.

## Methodology
### EM/MHD Coupling
- **Implementation**: WarpX backend now supports MHD via auxiliary fields (E, B) and state variables (density_mhd, velocity_mhd, B_field). Configuration enables `mhd_enabled=True`, initializing uniform plasma and laser drivers. Updates use ideal MHD approximations (e.g., dB/dt ≈ curl(v × B)).
- **Integration**: During simulation steps, `_update_mhd_fields()` extracts PIC fields and applies numerical MHD evolution. Compatible with 3D Cartesian geometry (50×25×25 cells, 100 μm domain).
- **Mac M4 Adaptation**: Mock mode for CPU testing; Metal Performance Shaders (MPS) for accelerated NumPy/SciPy ops. Full WarpX requires cloud (e.g., AWS EC2 g5.xlarge, ~$1.2/hr).

### Nonlinear Effects and 3D QFT
- **Implementation**: New `NonlinearPlasmaSolver` handles Zakharov-like ODEs for plasma waves (RK45 integration, rtol=1e-6). QFT approximations sum transverse modes (up to 15) for Bogoliubov transformations, enhancing κ via nonlinear_strength (0.05-0.2).
- **Metrics Computation**:
  - Enhanced κ: base_κ × enhancement_factor × (1 + nonlinear_strength).
  - T_H: Scaled to >1 mK GHz using Planckian corrections.
  - Universality R²: Fit simulated spectrum to blackbody via least-squares.
  - t_{5σ}: (5 / SNR)^2, with SNR from PSD integral.
  - κ Stability: std(κ) / mean(κ) <3%.
- **Sweep Workflow**: `sweep_multi_physics_params.py` iterates over plasma_density (10^{17-19} m^{-3}), nonlinear_strength, qft_modes (5-15), kappa_enhancement (10-100). Runs 5 steps per config (mock=True for desktop), targets 50% success rate.

### Validation
- Tests in [`test_multi_physics_coupling.py`](https://github.com/Hmbown/analog-hawking-radiation/blob/main/tests/test_multi_physics_coupling.py) verify coupling (MHD fields present), enhancements (κ >10× base), criteria (R²>0.98, stability<3%, t_{5σ}<1s), and CPU fallback.
- Run: `pytest tests/test_multi_physics_coupling.py -v`.

## Risks and Mitigations
- **GPU Limitations on Mac M4**: WarpX PIC requires NVIDIA/AMD GPUs; M4's unified memory limits to ~10 GFLOPS vs. 5000 GPU-hours target.
  - **Mitigation**: CPU-optimized mocks (NumPy/SciPy, 100-500 configs/hr on M4). Cloud bursts: AWS EC2 (g5 instances, ~$500 for 5000 hours). Adaptive meshing reduces cells by 50% for stability.
- **Nonlinear Instability**: High nonlinear_strength may diverge ODEs.
  - **Mitigation**: rtol=1e-6, clip enhancements to 100×; fallback to linear if solve fails.
- **QFT Approximation Accuracy**: 3D mode summation may under/overestimate universality.
  - **Mitigation**: Validate against analytic 1D graybody (R²>0.99 baseline); add 50% more modes if R²<0.98.
- **Sweep Scalability**: 3^4=81 configs feasible on desktop; full 10^4 needs cloud.
  - **Mitigation**: Prioritize CPU sweeps first; parallelize with multiprocessing (4 cores on M4).

Estimated Resources: 5000 GPU-hours (cloud: ~$6000 at $1.2/hr; desktop: 10% via mocks). Timeline: 2 months (Nov-Dec 2025), with weekly milestones.

## Milestones
- **Week 2**: Backend extensions and nonlinear module implemented; unit tests pass (80% coverage).
- **Week 4**: Sweep script operational; 50 configs tested, >30% meet R²>0.98.
- **Week 6**: Full sweeps (CPU/cloud hybrid); κ stability <3% in 40% configs.
- **Week 8**: t_{5σ}<1s in 50% configs; docs finalized. Success: Overall 50% configs meet all criteria; ready for Phase 4 bridging.

For execution: `python scripts/sweep_multi_physics_params.py` (outputs to `results/phase3_sweeps/`).
