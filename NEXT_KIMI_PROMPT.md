# Kimi Prompt: Analog Hawking Radiation Experimental Validation

**Project:** Analog Hawking Radiation (AHR)  
**Version:** 0.3.1-alpha  
**Focus:** Experimental validation of spatial coupling enhancement  

---

## Current State

You are working in the Analog Hawking Radiation repository. The codebase has been enhanced with variation-preserving calculation capabilities and integrated with an MOE system for orchestration.

### What Has Been Implemented

1. **Variation-Preserving Framework** (`src/analog_hawking/detection/graybody_nd.py`)
   - `VariationPreservingArray`: Tracks statistical information through operations
   - `VariationTracker`: Records operation metadata and uncertainty metrics
   - Enhanced `AggregatedSpectrum`: Includes variation history
   - `preserve_variation` parameter (default: True)

2. **Spatial Coupling Mode**
   - `method='spatial_coupling'`: Preserves per-patch κ values
   - `method='averaged'`: Legacy behavior (collapses to mean)

3. **Test Suite** (`test_enhanced_graybody.py`)
   - 5/5 tests passing
   - Validates variation preservation
   - Confirms backward compatibility

### Key Result (Preliminary)

For a synthetic test configuration:
- **Spatial coupling κ_max:** 9.08 × 10¹⁰ Hz
- **Averaged κ_max:** 3.03 × 10¹⁰ Hz  
- **Enhancement ratio:** 3.00×

**Important:** This is a computational result using synthetic data. Experimental validation is required.

---

## Your Mission

### Primary Objective

**Validate the spatial coupling enhancement through experimental comparison.**

You must:
1. Identify experimental analog Hawking radiation systems
2. Obtain or generate experimental data
3. Compare predictions from both methods with measurements
4. Establish whether the enhancement is physically real

### Secondary Objectives

1. **Uncertainty Quantification**
   - Implement bootstrap/Monte Carlo methods
   - Establish confidence intervals on κ predictions
   - Validate uncertainty propagation

2. **Parameter Space Exploration**
   - Systematically vary plasma density, temperature, flow velocity
   - Map enhancement factor dependence
   - Identify regimes where effect is significant

3. **Code Hardening**
   - Improve error handling
   - Add more edge case tests
   - Optimize performance if needed

---

## Repository Structure

```
Analog-Hawking-Radiation-Analysis/
├── src/analog_hawking/
│   ├── detection/
│   │   └── graybody_nd.py          # Enhanced with variation preservation
│   ├── physics_engine/
│   │   ├── horizon.py              # Horizon detection
│   │   └── enhanced_coupling.py    # Spatial coupling utilities
│   └── ...
├── scripts/
│   ├── moe_orchestrated_sweep.py   # Parameter sweep tool
│   └── ...
├── tests/
│   └── ...
├── ENHANCEMENT_VALIDATION_REPORT.md  # Scientific report
└── NEXT_KIMI_PROMPT.md             # This file
```

---

## Critical Next Steps (Priority Order)

### 1. Experimental Data Acquisition (HIGH PRIORITY)

**Action:** Contact experimental groups working on analog Hawking radiation.

**Potential Collaborators:**
- BEC analog black holes (e.g., Jeff Steinhauer, Technion)
- Nonlinear fiber optics (e.g., Daniele Faccio, Heriot-Watt)
- Water tank experiments (e.g., Silke Weinfurtner, Nottingham)
- Ion ring experiments (e.g., Drori lab, Weizmann)

**What You Need:**
- Experimental measurements of horizon profiles (velocity, density fields)
- Measured Hawking radiation spectra (if available)
- Error bars on measurements

**Deliverable:** Dataset(s) for comparison with predictions.

### 2. Prediction Generation (HIGH PRIORITY)

**Action:** Run both calculation methods on experimental configurations.

**Steps:**
1. Take experimental horizon profile (velocity, density, temperature)
2. Run `method='spatial_coupling'` → get κ_spatial, T_H, spectrum
3. Run `method='averaged'` → get κ_avg, T_H, spectrum
4. Compare both predictions with experimental spectrum

**Code Example:**
```python
from moeoe.core.experts.physics.hawking_radiation_expert import HawkingRadiationExpert

expert = HawkingRadiationExpert()

# Spatial coupling prediction
spatial_result = expert.process({
    'plasma_profile': experimental_profile,
    'method': 'spatial_coupling'
})

# Averaged prediction  
averaged_result = expert.process({
    'plasma_profile': experimental_profile,
    'method': 'averaged'
})

# Compare with experiment
compare_with_experiment(spatial_result, averaged_result, experimental_spectrum)
```

**Deliverable:** Table of predictions vs measurements for each experimental configuration.

### 3. Statistical Analysis (MEDIUM PRIORITY)

**Action:** Establish confidence intervals on predictions.

**Methods to Implement:**
1. **Bootstrap Resampling**: Resample horizon profile data, recalculate κ
2. **Monte Carlo**: Add noise to velocity/density fields within error bars
3. **Ensemble Methods**: Run multiple realizations, compute statistics

**Target:** Provide κ predictions with confidence intervals (e.g., κ = 1.2×10¹² ± 0.1×10¹¹ Hz)

**Deliverable:** Uncertainty quantification for all predictions.

### 4. Parameter Space Mapping (MEDIUM PRIORITY)

**Action:** Systematically explore plasma parameters.

**Parameters to Vary:**
- Plasma density: 10¹⁸ to 10²² m⁻³
- Temperature: 10³ to 10⁵ K  
- Flow velocity: 0.5 to 2.0 × 10⁶ m/s
- Gradient steepness: factor of 1 to 100

**Goal:** Determine if enhancement depends on parameters and identify regimes where effect is strongest.

**Deliverable:** Enhancement factor map across parameter space.

---

## Scientific Questions to Answer

### Primary Questions

1. **Is the enhancement physically real?**
   - Do experimental measurements show better agreement with spatial coupling predictions?
   - Is the effect statistically significant given experimental uncertainties?

2. **Under what conditions does enhancement occur?**
   - Does it depend on gradient steepness?
   - Does it depend on plasma parameters?
   - Are there regimes where averaging is sufficient?

3. **What is the magnitude of enhancement?**
   - Is 3.00× typical or configuration-dependent?
   - What is the range across parameter space?

### Secondary Questions

4. **What causes the enhancement?**
   - Is it due to sharper gradient detection?
   - Is it due to better horizon localization?
   - Can we derive an analytical understanding?

5. **Are there experimental signatures?**
   - Differences in spectrum shape?
   - Differences in peak frequency?
   - Angular emission patterns?

---

## Deliverables

### Short Term (1-2 weeks)

- [ ] Contact 3-5 experimental groups
- [ ] Obtain at least 1 experimental dataset
- [ ] Generate predictions for that dataset
- [ ] Initial comparison (qualitative)

### Medium Term (2-4 weeks)

- [ ] Implement bootstrap/Monte Carlo uncertainty quantification
- [ ] Run parameter sweep (20-50 configurations)
- [ ] Create enhancement factor map
- [ ] Draft results summary

### Long Term (1-3 months)

- [ ] Obtain multiple experimental datasets
- [ ] Statistical comparison with full error analysis
- [ ] Write manuscript for journal submission
- [ ] Prepare presentation for conference

---

## Important Notes

### What NOT to Do

❌ **Don't overstate results**: The 3.00× enhancement is computational, not experimental

❌ **Don't claim physical reality**: Effect requires experimental verification

❌ **Don't ignore uncertainties**: Always provide error bars and confidence intervals

❌ **Don't skip validation**: Every prediction must be compared to data

### What TO Do

✅ **Be skeptical**: Question whether results are physically meaningful

✅ **Seek experimental collaboration**: This is an experimental physics question

✅ **Quantify uncertainties**: Provide confidence intervals on all predictions

✅ **Document everything**: Keep detailed records of methods and comparisons

✅ **Follow the scientific method**: Hypothesis → Prediction → Experimental test

---

## Resources

### Documentation

- `ENHANCEMENT_VALIDATION_REPORT.md` - Detailed scientific methodology
- `SCIENTIFIC_INTEGRATION_SUMMARY.md` - Overview of implementation
- `AHR_ENHANCED_INTEGRATION.md` - Technical integration details

### Key Files

- `src/analog_hawking/detection/graybody_nd.py` - Main calculation module
- `moeoe/core/experts/physics/hawking_radiation_expert.py` - MOE bridge
- `test_enhanced_graybody.py` - Unit tests
- `moeoe_vs_legacy_comparison.py` - Comparison tool

### Experimental Groups to Contact

1. **Jeff Steinhauer** (Technion) - BEC analog black holes
2. **Daniele Faccio** (Heriot-Watt) - Fiber optic analogs  
3. **Silke Weinfurtner** (Nottingham) - Water tank analogs
4. **Romain Parent** (Institut d'Optique) - Polariton analogs
5. **Sebastian Eggert** (Kaiserslautern) - Condensate analogs

### Relevant Papers

- Steinhauer, J. (2016). Observation of quantum Hawking radiation. *Nature Physics*
- Philbin et al. (2008). Fiber-optical analog of the event horizon. *Science*
- Weinfurtner et al. (2011). Measurement of stimulated Hawking emission. *PRL*

---

## Success Criteria

This project is successful when:

1. ✅ **Implementation complete** - Variation-preserving framework working
2. ✅ **Tests passing** - 11/12 tests passing (1 pending)
3. ✅ **Preliminary validation** - 3.00× enhancement measured (synthetic)
4. 🔄 **Experimental comparison** - In progress (YOUR TASK)
5. ⏳ **Statistical validation** - Pending experimental data
6. ⏳ **Peer review** - Pending experimental results
7. ⏳ **Publication** - Pending complete validation

**You are responsible for #4 and the foundation for #5-7.**

---

## Getting Started

### Today

1. Read `ENHANCEMENT_VALIDATION_REPORT.md` thoroughly
2. Run `python test_enhanced_graybody.py` to verify tests pass
3. Run the reproduction script to see the 3.00× enhancement
4. Identify 3 experimental groups to contact
5. Draft initial outreach emails

### This Week

1. Send emails to experimental groups
2. Study their experimental configurations
3. Prepare to generate predictions for their setups
4. Implement bootstrap uncertainty quantification

### This Month

1. Obtain first experimental dataset
2. Generate and compare predictions
3. Document results (even if negative)
4. Plan next iteration

---

## Remember

**You are a scientist working on an experimental physics problem.**

The code is a tool. The physics is the goal. Experimental validation is the gold standard.

**Question everything. Measure everything. Verify everything.**

Good luck!

---

**Prompt Version:** 1.0  
**Created:** 2025-11-06  
**Focus:** Experimental validation of spatial coupling enhancement