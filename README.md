# Laser-Plasma Sonic Horizon Simulator (Alpha Research Code)

[![Python Version](https://img.shields.io/badge/python-3.9%E2%80%933.11-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/) [![Version](https://img.shields.io/badge/version-0.3.1--alpha-blue.svg)](https://github.com/hmbown/analog-hawking-radiation/releases)

**⚠️ RESEARCH PROTOTYPE - NOT VALIDATED AGAINST EXPERIMENTAL DATA ⚠️**

This code implements theoretical models that have **NOT** been benchmarked against ELI or other laser facilities. All "validation" claims refer to unit tests only, not experimental verification. Use for method exploration only.

---

## What This Code Does

Implements a **fluid-dynamics approximation** of analog Hawking radiation in laser-plasma systems:

- Detects sonic horizons in plasma flow profiles (1D/2D/3D)
- Estimates surface gravity κ from velocity gradients
- Calculates Hawking temperature T_H = ħκ/(2πk_B) (theoretical)
- Applies graybody transmission models (dimensionless approximation)
- Validates if laser parameters are achievable at ELI facilities

**What it does NOT do**: Simulate actual quantum effects, predict measurable signals, or replace PIC codes.

---

## Quick Start (For Method Exploration)

```bash
# Install (development mode)
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .

# Run basic horizon detection example
python examples/basic_horizon_detection.py
```

**Expected output**: A plot showing where flow velocity crosses sound speed, with order-of-magnitude κ estimate.

---

## Scientific Status

### Implemented
- Sonic horizon detection in fluid approximation
- Surface gravity calculation from velocity gradients
- Dimensionless graybody factor (Ω ↦ r²/(1+r²))
- ELI facility parameter validation (intensity, wavelength, pulse duration)

### Not Implemented
- PIC code coupling (OSIRIS, WarpX interfaces are placeholders)
- Quantum fluctuation models
- Kinetic plasma effects (Vlasov-Fokker-Planck)
- Laser-plasma instabilities (Raman, Brillouin, filamentation)
- Synthetic diagnostic outputs

### Needs Experimental Validation
1. Horizon detection algorithm vs. PIC simulations
2. Surface gravity scaling law κ ∝ (dv/dx)
3. Graybody transmission coefficient
4. Plasma mirror intensity thresholds
5. Uncertainty propagation methods

---

## Citation

**If you use this code, cite appropriately:**

```bibtex
@software{bown2025simulator,
  author = {Bown, Hunter},
  title = {Laser-Plasma Sonic Horizon Simulator (v0.3.1-alpha)},
  url = {https://github.com/Hmbown/analog-hawking-radiation},
  version = {0.3.1-alpha},
  year = {2025},
  note = {Research prototype, not experimentally validated}
}
```

**And cite the original concept:**

Chen, P. & Mourou, G. *Accelerating Plasma Mirrors to Investigate the Black Hole Information Loss Paradox*. Sci. Rep. 7, 1-7 (2017).

---

## Seeking Collaborators

This project needs domain experts to progress:

- **Experimental plasma physicist** with ELI facility access
- **PIC code expert** (OSIRIS, WarpX) for coupling implementation  
- **Theoretical physicist** to audit sonic horizon approximations
- **Peer reviewer** to validate uncertainty propagation

If you are one of these, please open an issue. This code cannot advance without expert validation.

---

## Development Notes

- **Code maturity**: Alpha research prototype (not production software)
- **Test coverage**: ~70% unit tests, zero integration tests with physics codes
- **Performance**: Not HPC-optimized, limited to ~512³ grids on typical workstations
- **Documentation**: API docs complete, physics validation docs are aspirational

---

## License

MIT License - See LICENSE file for details

## Contact

**Lead Developer**: Hunter Bown  
**Status**: Independent researcher seeking academic collaboration  
**Email**: hunterbown@example.com

---

## Version History

- **v0.3.1-alpha** (Nov 2025): Initial public release with basic horizon detection
- **v0.2.0** (Oct 2025): Core physics modules implemented  
- **v0.1.0** (Sep 2025): Project structure established