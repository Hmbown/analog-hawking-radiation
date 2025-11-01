# Gradient Catastrophe Analysis

**Version**: 0.3.0  
**Module**: `scripts/sweep_gradient_catastrophe.py`  
**Analysis Type**: Physics breakdown boundary mapping  

## Overview

The gradient catastrophe analysis systematically explores the parameter space of laser–plasma analog Hawking systems to estimate an approximate upper bound on achievable surface gravity (κ) under explicitly stated breakdown thresholds. This analysis addresses the question: **What κ is attainable in our 1D toy models before the chosen breakdown criteria are triggered?**

## Scientific Motivation

Previous studies suggested that κ could potentially reach 10¹⁴ Hz or higher in simplified settings. Those estimates often did not enforce multiple breakdown modes in extreme gradient regimes. Here we map these constraints for our synthetic 1D profiles. This is not a claim of a fundamental limit in nature and is sensitive to threshold choices.

## Methodology

### Parameter Space Exploration

The analysis sweeps across three key parameters:

1. **Normalized laser amplitude (a₀)**: 1 to 100
   - a₀ = eE₀/(mₑωc) where E₀ is peak electric field
   - Controls the relativistic character of the interaction

2. **Plasma density (nₑ)**: 10¹⁸ to 10²² m⁻³
   - From underdense to overcritical regimes
   - Affects plasma frequency and horizon formation

3. **Gradient steepness factor**: 1 to 1000
   - Controls the velocity gradient scale length
   - Higher values create steeper transitions

### Physics Breakdown Detection

The analysis monitors five breakdown modes:

1. **Relativistic breakdown**: Velocities exceed v > 0.5c
2. **Ionization breakdown**: Densities become negative or exceed solid density
3. **Wave breaking**: Sound speed becomes negative
4. **Gradient catastrophe**: Velocity gradients become infinite
5. **Numerical instability**: NaN or infinite values appear

Each configuration receives a validity score from 0 (complete breakdown) to 1 (fully valid physics).

### Profile Generation

For each parameter combination, the analysis creates realistic velocity and sound speed profiles:

```python
# Sound speed with spatial variation
cs_base = cs_thermal * (1 + 0.2 * exp(-(x/sigma)²))

# Velocity profile that can create horizons
v_scale = cs_thermal * 1.5 * a₀ * gradient_factor
velocity = v_scale * tanh((x - x_transition) / sigma)
```

### Surface Gravity Calculation

Surface gravity is calculated at horizon crossings (|v| = c_s) using the acoustic‑exact method implemented in the code:

κ = |∂ₓ(c_s² − v²)| / (2 cₕ),

where cₕ is the sound speed at the horizon. This corresponds to `kappa_method="acoustic_exact"` used by the sweep script.

## Key Findings

### Maximum Achievable Surface Gravity

**Primary Result**: κ_max = 5.94×10¹² Hz (this production run; acoustic‑exact κ; thresholds enforced).

Interpretation: This is a parametric upper bound for the thresholds and 1D synthetic profiles used in this analysis, not a fundamental constant. Different thresholds or more realistic dynamics will shift this number.

### Configuration at Upper Bound (this run)

The configuration achieving maximum κ in this run:
- **a₀**: 6.95 (relativistic regime)
- **nₑ**: 1.0×10²⁰ m⁻³
- **Gradient factor**: 2.15
- **Intensity**: 1.03×10²⁴ W/m²

### Physics Constraints

The dominant limitation is breakdown when any of the following thresholds are exceeded (implemented in the sweep):
- v > 0.5c (≈ 1.5×10⁸ m/s)
- |dv/dx| > 4×10¹² s⁻¹
- Intensity I > 1×10²⁴ W/m² (conservative demo cap; used here for parametric mapping)

### Scaling Relationships

Unexpectedly, the analysis reveals:

1. **κ vs a₀**: exponent ≈ +0.66 (95% CI [0.44, 0.89])
2. **κ vs nₑ**: exponent ≈ −0.02 (95% CI [−0.14, 0.10])

**Interpretation**: Higher laser intensities create steeper gradients but push systems into relativistic regimes where physics becomes invalid. An optimal "sweet spot" exists around a₀ ≈ 1.6.

### Breakdown Statistics

From 500 configurations tested (this run):
- **Valid physics**: 68 / 500 (13.6%)
- **Breakdown rate**: 86.4%
- **Dominant modes**: gradient‑driven breakdown (86.4%), relativistic (47.6%)

## Detection Time Implications

With κ_max = 5.94×10¹² Hz, a naive thermal timescale is:

t_min ≈ 1/(2κ) ≈ 8.4×10⁻¹⁴ seconds

### Practical Detection Times

Accounting for realistic signal-to-noise ratios and experimental constraints in our models:

**t_detection ∼ 10⁻⁷ to 10⁻⁶ seconds** (highly model‑dependent).

These values are sensitive to coupling assumptions, graybody choices, and instrument parameters and should be treated as illustrative.

## Experimental Implications

### Laser Considerations

The configuration at the reported bound implies:
- **Peak intensity**: ~1×10²⁴ W/m² (order‑of‑magnitude; far beyond current laboratory capability)
- **Focused spot size**: ~1 μm
- **Pulse duration**: ~10 fs
- **Total energy**: ~100 J

These parameters are not intended as experimental prescriptions. They are used to probe model limits and should not be interpreted as near‑term feasible.

### Detection Strategy

1. **Target the optimal zone**: Operate near a₀ ≈ 1.6, nₑ ≈ 1.4×10¹⁹ m⁻³
2. **Moderate gradients**: Use gradient factors 4-5 for maximum κ without breakdown
3. **Fast diagnostics**: Detection systems must resolve signals on ~100 ns timescales
4. **Multi-shot averaging**: Account for shot-to-shot variations

## Scientific Significance

### Scientific Framing

This analysis provides a systematic mapping of breakdown‑limited κ in synthetic 1D profiles under explicit thresholds. It is not a claim of a fundamental limit independent of model and thresholds.

### Broader Implications

The findings impact:
- **Analog gravity experiments**: Realistic detection prospects
- **Laser-plasma acceleration**: Relativistic breakdown thresholds  
- **High-intensity laser physics**: Physics validity boundaries

### Comparison to Theory

In this toy-model study, κ ∼ 10¹⁴ Hz does not persist once breakdown thresholds are enforced; our run’s upper bound is ~5.9×10¹² Hz. This number is model‑ and threshold‑dependent.

## Code Usage

### Basic Analysis

```bash
python scripts/sweep_gradient_catastrophe.py \
  --n-samples 200 \
  --output results/gradient_analysis
```

### Production Run

```bash
python scripts/run_production_gradient_sweep.py
```

### Result Analysis

Results are saved in JSON format with comprehensive analysis (production run shown):

```python
import json
with open('results/gradient_limits_production/gradient_catastrophe_sweep.json') as f:
    data = json.load(f)

# Access key findings
max_kappa = data['analysis']['max_kappa']
optimal_config = data['analysis']['max_kappa_config']
breakdown_stats = data['analysis']['breakdown_statistics']
```

## Validation and Limitations

### Validation

- **Physics consistency**: All configurations tested against conservation laws
- **Numerical stability**: Automatic detection of computational breakdown
- **Parameter bounds**: Physical constraints enforced (v < c, positive densities)

### Limitations

1. **1D analysis**: Multi-dimensional effects not included
2. **Fluid approximation**: Kinetic effects not captured
3. **Simplified profiles**: Real plasma dynamics more complex
4. **No dissipation**: Energy/momentum loss mechanisms neglected

### Future Extensions

1. **3D PIC validation**: Compare with full kinetic simulations
2. **Experimental validation**: Test predictions with real experiments
3. **Alternative geometries**: Magnetic confinement, structured targets
4. **Time-dependent analysis**: Evolution during breakdown

## File Outputs

The analysis generates:

1. **gradient_catastrophe_sweep.json**: Complete results and analysis
2. **gradient_catastrophe_analysis.png/pdf**: Publication-quality plots
3. **gradient_catastrophe_findings.md**: Human-readable summary

## References

- Unruh, W. G. (1981). Experimental black-hole evaporation?
- Steinhauer, J. (2016). Observation of quantum Hawking radiation
- Chen, P. & Mourou, G. (2015). Accelerating plasma mirrors
- Hawking, S. W. (1975). Particle creation by black holes

---

*This analysis represents a significant advance in understanding the experimental prospects for analog Hawking radiation detection, providing both fundamental insights and practical guidance for experimental implementation.*
