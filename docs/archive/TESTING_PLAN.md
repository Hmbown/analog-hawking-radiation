# Testing and Validation Plan

## Overview

This document outlines the validation strategy for the analog Hawking radiation computational framework. The goal is to ensure all physics implementations are correct, numerically stable, and reproducible.

## Validation Categories

### 1. Physics Validation

#### 1.1 Hawking Radiation Theory
- Verify Hawking temperature formula: T_H = ħκ/(2πk_B)
- Validate quantum field theory spectrum implementation against analytical solutions
- Check graybody transmission calculations against WKB theory
- Verify surface gravity (κ) derivation and numerical uncertainty estimates

#### 1.2 Plasma Physics
- Confirm sound speed calculation: c_s = √(γkT_e/m_i)
- Validate relativistic parameter a₀ = eE₀/(mₑωc) against analytical formula
- Check plasma frequency calculations against standard expressions
- Verify magnetosonic speed approximations

### 2. Numerical Methods Validation

#### 2.1 Horizon Detection
- Test root finding algorithm (bracketing and bisection) on known profiles
- Validate multi-stencil finite difference uncertainty quantification
- Verify adaptive smoothing and κ-plateau diagnostics
- Test edge cases: no horizons, multiple horizons, weak gradients

#### 2.2 Multi-Beam Superposition
- Confirm power conservation across all beam configurations
- Validate envelope-scale coarse-graining assumptions
- Test geometric configurations: rings, crossings, standing waves
- Verify gradient enhancement calculations

#### 2.3 Detection Modeling
- Test radiometer equation: SNR = (T_sig/T_sys)√(B·t)
- Validate quantum field theory spectrum integration
- Check frequency band gating logic for radio/microwave regime
- Verify graybody transmission integration

### 3. Experimental Parameter Validation

#### 3.1 Physical Realizability
- Verify laser intensities are achievable (10¹⁷-10¹⁸ W/m²)
- Check plasma densities are realistic (10²³-10²⁴ m⁻³)
- Validate temperature ranges (10⁶-10⁷ K)
- Assess detection time estimates for reasonableness

#### 3.2 Detection Requirements
- Validate system temperature assumptions (10-100K range)
- Check bandwidth requirements (MHz-GHz)
- Verify integration time calculations
- Assess 5σ detection feasibility

### 4. Reproducibility Testing

#### 4.1 Figure Generation
- Regenerate all figures: `make figures`
- Verify figure captions match actual content
- Check data consistency between figures and JSON result files
- Validate enhancement statistics match bar chart data

#### 4.2 Data Consistency
- Cross-check `results/enhancement_stats.json` with code outputs
- Verify `results/horizon_summary.json` data integrity
- Check numerical precision and uncertainty bounds
- Validate unit consistency across all results

#### 4.3 Installation Testing
- Test installation in clean Python environment
- Verify all dependencies resolve correctly
- Execute key scripts and validate outputs
- Test on Python 3.8, 3.9, 3.10+

### 5. Integration Testing

#### 5.1 Module Interactions
- Test data flow: Plasma Models → Horizon Detection → QFT → Detection
- Validate parameter passing between modules
- Check error handling for edge cases
- Verify memory usage is reasonable

#### 5.2 Backend Integration
- Test fluid backend calculations against analytical limits
- Validate WarpX backend mock implementation
- Check adaptive sigma integration
- Verify fluctuation injector functionality

## Test Suite Execution

### Automated Tests
```bash
# Run complete test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src/analog_hawking

# Run specific test categories
pytest tests/test_horizon_kappa_analytic.py -v
pytest tests/test_graybody.py -v
pytest tests/integration_test.py -v
```

### Manual Validation
```bash
# Validate all figures and analysis
make all

# Run full pipeline demonstration
python scripts/run_full_pipeline.py --demo

# Execute formation frontier analysis
python scripts/compute_formation_frontier.py
```

## Success Criteria

### Required (Must Pass)
- All unit tests pass (`pytest`)
- All integration tests pass
- Theoretical formulas match analytical solutions within numerical precision
- Parameter ranges are physically achievable
- Code executes without errors on supported Python versions

### Expected (Should Pass)
- Enhancement factors match documented values within ±5%
- Horizon detection uncertainty < 10% for well-resolved profiles
- Detection time estimates within factor of 2 of independent calculation
- Multi-beam power conservation within 1%
- Numerical convergence demonstrated with grid refinement

### Desirable (Nice to Have)
- Performance benchmarks within expected ranges
- Memory usage scales appropriately with problem size
- Documentation matches code behavior exactly
- All cross-references resolve correctly

## Validation Protocols

### Analytical Benchmarks
- Compare numerical horizon finding with known linear profile solutions
- Validate Hawking spectrum against Planck distribution in appropriate limits
- Check Rayleigh-Jeans limit for low frequencies
- Verify graybody transmission limits

### Numerical Convergence
- Grid refinement studies for spatial convergence
- Time step refinement for temporal convergence
- Verify second-order convergence where expected
- Check numerical stability under parameter variations

### Physical Consistency
- Verify dimensional analysis of all quantities
- Check limiting behavior (low T, high T, weak field, strong field)
- Validate parameter sensitivities are physically reasonable
- Ensure energy/power conservation where applicable

## Documentation

All validation results should be documented with:
- Test conditions and parameters
- Expected vs actual results
- Uncertainty estimates
- References to analytical solutions or literature values
- Any discrepancies and their explanations

## Continuous Validation

As the codebase evolves:
- Run full test suite before any commit
- Update tests when adding new features
- Document any changes to physics models
- Maintain backward compatibility or document breaking changes
- Keep validation documentation up to date

This testing plan ensures scientific rigor and reproducibility of all results.
