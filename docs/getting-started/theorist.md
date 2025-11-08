# Getting Started: Theorist & Analyst

**Quick Navigation**: [Back to README](../../README.md) | [Quick Links](../QUICKLINKS.md) | [Full Documentation](../index.md)

This guide is for **theorists and analysts** who want to:
- Test new physical models and assumptions
- Validate computational methods against theory
- Explore parameter spaces and scaling laws
- Quantify uncertainties and systematic effects

---

## 🎯 Your Goals & How We Help

| Your Goal | Our Tool | Time to Result |
|-----------|----------|----------------|
| Validate horizon finding algorithms | `ahr validate` | 1 minute |
| Test graybody models against theory | `ahr pipeline --demo` | 2 minutes |
| Explore parameter scaling laws | `ahr sweep --gradient` | 15 minutes |
| Quantify numerical uncertainties | `ahr validate --dashboard` | 1 minute |

---

## ⚡ Quick Start (5 minutes)

### 1. Install & Setup
```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .[dev]  # Include dev tools
ahr info              # Check your system
```

### 2. Run Validation Suite
```bash
ahr validate --dashboard
```

This runs 42 physics validation tests and shows a visual dashboard.

### 3. Test Core Algorithms
```bash
# Benchmark horizon finder
ahr bench

# Test with different κ methods
ahr quickstart --out results/test_acoustic
ahr quickstart --out results/test_geometric
```

### 4. Explore Physics
```bash
# Run gradient catastrophe analysis
ahr sweep --gradient --n-samples 200 --output results/physics_exploration

# Check scaling laws
grep -r "kappa" results/physics_exploration/ | python scripts/analyze_scaling.py
```

---

## 🔬 Validation & Verification

### Physics Validation Framework

Our validation suite checks:

1. **Analytical Consistency**
   - κ → T_H mapping: T_H = ħκ/(2πk_B)
   - Graybody limits: high-frequency transparency, low-frequency suppression
   - Conservation laws: energy, momentum

2. **Numerical Convergence**
   - Grid resolution tests
   - Time step independence
   - Round-off error bounds

3. **Physical Bounds**
   - Causality: v < c
   - Stability: numerical Courant condition
   - Thermodynamics: T_H > 0, positive entropy

### Run Specific Validations

```bash
# Core physics only
ahr validate --category horizon,graybody

# Show detailed results
ahr validate --dashboard --verbose

# Export for analysis
ahr validate --report validation_results.json
```

---

## 📊 Testing New Models

### Implementing a New κ Calculation

```python
# src/analog_hawking/physics_engine/my_kappa.py
import numpy as np
from .horizon import KappaResult

def compute_my_kappa(x, v, cs, **kwargs):
    """
    Your new κ calculation method.
    
    Parameters
    ----------
    x : array - position grid [m]
    v : array - flow velocity [m/s]  
    cs : array - sound speed [m/s]
    
    Returns
    -------
    KappaResult with positions, kappa values, uncertainties
    """
    # Your implementation here
    # ...
    
    return KappaResult(
        positions=horizon_positions,
        kappa=kappa_values,
        kappa_err=uncertainties,
        method="my_new_method"
    )
```

### Adding to the Framework

```python
# Register your method
from .horizon import register_kappa_method

register_kappa_method("my_method", compute_my_kappa)

# Now use it
ahr quickstart --kappa-method my_method
```

### Validation Checklist for New Models

- [ ] Converges with grid resolution
- [ ] Recovers known analytical limits
- [ ] Preserves physical bounds
- [ ] Uncertainties are quantified
- [ ] Comparable to existing methods
- [ ] Documented with references

---

## 📈 Scaling Laws & Parameter Studies

### Gradient Catastrophe Analysis

```bash
# Map physics breakdown boundaries
ahr sweep --gradient --n-samples 500 --output results/gradient_study

# Analyze scaling laws
python scripts/analyze_gradient_results.py \
  --input results/gradient_study/gradient_catastrophe_sweep.json \
  --plot results/gradient_study/scaling_laws.png
```

### Custom Parameter Sweeps

```python
# scripts/my_parameter_sweep.py
import numpy as np
from analog_hawking import find_horizons_with_uncertainty, sound_speed

# Define parameter ranges
intensities = np.logspace(20, 24, 50)  # 10^20 to 10^24 W/m²
densities = np.logspace(24, 26, 30)    # 10^24 to 10^26 m⁻³

results = []
for I in intensities:
    for n_e in densities:
        # Compute horizon properties
        # ...
        results.append({
            'intensity': I,
            'density': n_e,
            'kappa': kappa,
            'valid': sanity_check_passed
        })

# Save and analyze
import pandas as pd
df = pd.DataFrame(results)
df.to_csv('results/my_sweep.csv')
```

---

## 🔍 Uncertainty Quantification

### Types of Uncertainties

1. **Numerical Uncertainty** (κ_err in outputs)
   - Grid discretization effects
   - Interpolation errors
   - Round-off accumulation

2. **Model Uncertainty**
   - Graybody approximation validity
   - 1D vs 3D effects
   - Missing physics

3. **Experimental Uncertainty**
   - Laser intensity fluctuations
   - Plasma density variations
   - Detector noise

### Quantifying Numerical Uncertainty

```bash
# Run with different grid resolutions
for nx in 500 1000 2000 4000; do
  ahr quickstart --nx $nx --out results/convergence_$nx
done

# Analyze convergence
python scripts/analyze_convergence.py \
  --pattern results/convergence_*/horizons.json
```

### Monte Carlo Uncertainty Propagation

```bash
# Run comprehensive uncertainty analysis
python scripts/comprehensive_monte_carlo_uncertainty.py \
  --n-samples 1000 \
  --output results/uncertainty_analysis
```

---

## 🧪 Testing Against Theory

### Analytical Test Cases

1. **Constant Velocity Flow**
   - Should produce no horizons if |v| < c_s
   - Should produce horizons if |v| > c_s

2. **Linear Velocity Profile**
   - Known κ = |dv/dx|/2 at horizon
   - Tests gradient calculation

3. **Tanh Profile**
   - Smooth, well-behaved test case
   - Tests root-finding algorithms

### Run Theory Tests

```bash
# Test suite includes analytical cases
pytest tests/test_horizon_analytical.py -v

# Compare different κ methods
python scripts/compare_kappa_methods.py \
  --methods acoustic_exact,geometric,characteristic \
  --output results/kappa_comparison
```

---

## 📚 Key Theoretical Concepts

### Surface Gravity (κ)

Multiple definitions implemented:

1. **Acoustic Exact**: κ = |∂_x(c_s² - v²)|/(2c_H)
   - Most accurate for fluid flows
   - Requires smooth profiles

2. **Geometric**: κ = |∂_x(|v| - c_s)|
   - Simpler approximation
   - Good for steep horizons

3. **Characteristic**: κ = |∂_x(c_s - v)|_horizon
   - Based on ray tracing
   - Physics interpretation clear

### Graybody Factors

Transmission coefficient accounting for frequency-dependent scattering:

```python
# Acoustic WKB approximation
gamma_omega = exp(-2πω/κ)  # Low-frequency suppression

# High-frequency limit
gamma_omega → 1  # Perfect transmission
```

### Detection Modeling

Signal temperature in radio band:

```
T_sig = ∫_0^∞ dω γ_ω (ħω/2π) [|c_H/(c_H + v_H)|]²
```

Integration over detector bandwidth gives measurable signal.

---

## 🛠️ Development Workflow

### Adding New Physics

1. **Fork and clone**
```bash
git fork https://github.com/hmbown/analog-hawking-radiation.git
git clone <your-fork>
cd analog-hawking-radiation
ahr dev --setup
```

2. **Create feature branch**
```bash
git checkout -b feature/my-new-model
```

3. **Implement and test**
```bash
# Your implementation here
# Add tests in tests/

# Run validation
ahr validate

# Check coverage
pytest --cov=src/analog_hawking
```

4. **Submit PR**
```bash
git push origin feature/my-new-model
# Create pull request on GitHub
```

### Code Quality Standards

- ✅ All new physics must include validation tests
- ✅ Analytical solutions should be recovered in limits
- ✅ Uncertainties must be quantified
- ✅ Documentation with references required
- ✅ Performance benchmarks for new algorithms

---

## 📖 Theoretical Background

### Analog Gravity Fundamentals

The framework is based on the analogy between:
- **Acoustic geometry** in moving fluids
- **Spacetime geometry** near black holes

Key insight: Sound waves in a transonic flow experience an effective metric with horizon where flow speed exceeds sound speed.

### Hawking Radiation in Analog Systems

Temperature scaling:
```
T_H = ħκ / (2πk_B)
```

For typical plasma parameters:
- κ ~ 10¹² s⁻¹ → T_H ~ 10⁻⁸ K
- Radio detection requires careful noise modeling

### Literature References

**Foundational**:
- Unruh (1981) - Original analog proposal
- Visser (1998) - Acoustic black holes
- Barceló et al. (2005) - Analog gravity review

**Experimental**:
- Weinfurtner et al. (2011) - Water wave experiment
- Steinhauer (2014) - Bose-Einstein condensate
- Chen & Mourou (2017) - Plasma mirror concept

See [REFERENCES.md](../REFERENCES.md) for complete bibliography.

---

## 🎯 Research Projects

### Suitable for Theorists

1. **New κ definitions** for different flow profiles
2. **Multi-dimensional effects** beyond 1D approximation
3. **Viscosity and dissipation** impacts on horizons
4. **Magnetized plasma** generalizations
5. **Beyond-WKB** graybody calculations

### Collaboration Opportunities

- **Analytical validation** of numerical methods
- **Uncertainty quantification** frameworks
- **Multi-physics coupling** models
- **Experimental design** optimization

See [open issues](https://github.com/hmbown/analog-hawking-radiation/issues) for specific projects.

---

## 📞 Getting Help

**Physics questions?**
- [GitHub Discussions](https://github.com/hmbown/analog-hawking-radiation/discussions)
- Tag: "theory-question"

**Code issues?**
- [GitHub Issues](https://github.com/hmbown/analog-hawking-radiation/issues)
- Include minimal working example

**Methodology discussion?**
- Email: hunter@shannonlabs.dev
- Subject: "Analog Hawking - Theory"

---

## 🎓 Learning Path

**Week 1**: Master basics
- Run all tutorials: `ahr tutorial --list`
- Read [Methods & Algorithms](../Methods.md)
- Understand [Validation Framework](../Validation.md)

**Week 2**: Deep dive
- Study [Gradient Catastrophe Analysis](../GradientCatastropheAnalysis.md)
- Read [Enhanced Physics Models](../Enhanced_Physics_Models_Documentation.md)
- Explore [Limitations](../Limitations.md)

**Week 3**: Contribute
- Pick open issue labeled "theory"
- Implement new model or validation
- Submit pull request

---

<div align="center">

**[Back to README](../../README.md)** | **[Quick Links](../QUICKLINKS.md)** | **[Full Documentation](../index.md)**

*Laboratory Black Hole Detection, Quantified*

</div>
