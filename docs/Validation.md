Validation & Checks
===================

The framework implements comprehensive validation protocols to ensure scientific rigor and computational reliability.

Theoretical Validation
----------------------

Core physics implementations are validated against known analytical solutions:

* **Plasma Frequency Calculation**: Verified against the analytical expression ω_pe = √(n_e * e² / (ε₀ * m_e))
* **Relativistic Parameter a₀**: Validated using a₀ = eE₀/(mₑωc) with proper units
* **Hawking Temperature from κ**: Confirmed against T_H = ħκ/(2πk_B)

Frequency Gating Validation
---------------------------

The frequency gating system ensures appropriate calculations for low Hawking temperatures:

* Automatic band selection ensures radio-band calculations for low T_H
* Frequency range validation prevents unphysical extrapolations

Horizon Finder Validation
-------------------------

The horizon detection algorithms include built-in uncertainty quantification:

* Multi-stencil finite difference methods provide robust error estimates
* Root finding algorithms validated against analytical test cases
* Convergence testing confirms numerical stability

Numerical Validation
--------------------

Comprehensive numerical validation ensures computational reliability:

* **Spatial Convergence**: Second-order convergence verified through grid refinement studies
* **Temporal Convergence**: Time-stepping algorithms demonstrate proper convergence properties
* **Parameter Sensitivity**: Physically reasonable responses across parameter space

Validation Commands
-------------------

To execute the validation suite:

```bash
make validate
```

This command runs all unit tests and validation checks to verify the correct implementation of physics formulas and numerical methods.

Validation Results

CI and Units Checks
-------------------

- Continuous Integration runs the full test suite on Python 3.9–3.11.
- A lightweight units test validates κ→T_H scaling and κ method ratios (see `tests/test_units_lite.py`).
------------------

The validation suite confirms:

1. Perfect agreement between analytical solutions and computational results for core physics formulas
2. Proper convergence behavior for numerical methods
3. Physically reasonable parameter dependencies
4. Correct implementation of uncertainty quantification procedures

These validation results establish confidence in the framework's ability to provide reliable scientific insights into analog Hawking radiation phenomena.
