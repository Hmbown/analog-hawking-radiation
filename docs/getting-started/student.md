# Getting Started: Student & Researcher

**Quick Navigation**: [Back to README](../../README.md) | [Quick Links](../QUICKLINKS.md) | [Full Documentation](../index.md)

This guide is for **students and researchers** who want to:
- Learn about analog Hawking radiation and black holes
- Understand the physics behind the code
- Reproduce published results
- Build intuition for complex phenomena

---

## 🎓 Learning Objectives

By the end of this guide, you will understand:
- What analog black holes are and why they matter
- How plasma flows can mimic event horizons
- The connection between surface gravity and Hawking temperature
- How to interpret simulation results
- Where the current research frontiers are

---

## ⚡ Quick Start (5 minutes)

### 1. Install & Setup
```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
```

### 2. Run Your First Simulation
```bash
ahr quickstart
```

### 3. Look at the Results
```bash
open results/quickstart/quickstart_profile.png
```

**What you should see**: A plot showing:
- **Blue line**: Flow velocity |v| (how fast plasma moves)
- **Orange line**: Sound speed c_s (how fast sound travels in plasma)
- **Red dashed lines**: Where |v| = c_s (the "sonic horizon")

### 4. Take a Tutorial
```bash
ahr tutorial 1  # "What is a Sonic Horizon?"
```

---

## 📚 The Physics in Plain English

### What is a Sonic Horizon?

Imagine you're swimming in a river:
- If you swim faster than the current, you can swim upstream
- If the current is faster than you can swim, you can't go upstream anymore
- The point where current speed = your swimming speed is like a "horizon"

In our plasma:
- **"You"** = sound waves (pressure disturbances)
- **"River"** = plasma flow moving at velocity v
- **"Horizon"** = where plasma speed equals sound speed (|v| = c_s)

Sound waves inside the horizon can't escape, just like light can't escape a black hole!

### From Flow to Hawking Radiation

Stephen Hawking discovered that black holes aren't completely black—they emit faint radiation due to quantum effects near the event horizon.

**Amazing fact**: The same mathematics describes:
1. **Real black holes**: Gravity + quantum mechanics
2. **Analog black holes**: Fluid flow + sound waves

This means we can study black hole physics in the lab using plasma!

### Surface Gravity (κ) - The Key Quantity

Surface gravity tells us how "steep" the horizon is:
- **High κ** = sharp horizon = stronger radiation
- **Low κ** = gentle horizon = weaker radiation

The Hawking temperature depends on κ:
```
T_H = ħ × κ / (2π × k_B)
```

Where:
- ħ = Planck's constant (quantum mechanics)
- k_B = Boltzmann constant (thermodynamics)
- κ = surface gravity (from fluid dynamics)

### Why is the Signal So Weak?

Typical values in our simulations:
- κ ≈ 10¹² s⁻¹
- T_H ≈ 10⁻⁸ K (extremely cold!)

For comparison:
- Cosmic microwave background: 2.7 K
- Liquid helium: ~1 K
- Our signal: 0.00000001 K

This is why detection is challenging but possible with sensitive radio telescopes!

---

## 🎮 Interactive Learning

### Tutorial 1: What is a Sonic Horizon?
```bash
ahr tutorial 1
```

**Hands-on exercise**:
```python
# Create different flow profiles
import numpy as np
import matplotlib.pyplot as plt

# Gentle flow (no horizon)
x = np.linspace(0, 100, 1000)
v_gentle = 100 * np.tanh((x - 50) / 20)  # Max speed: 100 m/s
c_s = np.full_like(x, 200)  # Sound speed: 200 m/s

plt.plot(x, v_gentle, label='|v| (gentle)')
plt.plot(x, c_s, '--', label='c_s')
plt.xlabel('Position')
plt.ylabel('Speed (m/s)')
plt.title('No horizon: |v| < c_s everywhere')
plt.legend()
plt.show()

# Fast flow (creates horizon)
v_fast = 300 * np.tanh((x - 50) / 10)  # Max speed: 300 m/s

plt.plot(x, v_fast, label='|v| (fast)')
plt.plot(x, c_s, '--', label='c_s')
plt.axvspan(30, 70, alpha=0.2, label='Horizon region')
plt.xlabel('Position')
plt.ylabel('Speed (m/s)')
plt.title('Horizon forms: |v| > c_s in region')
plt.legend()
plt.show()
```

### Tutorial 2: Surface Gravity and Temperature
```bash
ahr tutorial 2
```

**Key insight**: Steeper velocity gradients → higher κ → higher temperature

```python
# Compare different gradient strengths
gradients = [1e11, 5e11, 1e12, 5e12]  # Different steepness

for grad in gradients:
    # Create profile with specific gradient at horizon
    v = 200 * np.tanh((x - 50) * grad / 200)
    
    # Find horizon and compute kappa
    # (In real code, use find_horizons_with_uncertainty)
    
    kappa = grad  # Approximate for demonstration
    T_H = 1.22e-23 * kappa / (2 * np.pi * 1.38e-23)
    
    print(f"Gradient: {grad:.1e} s⁻¹, κ: {kappa:.1e} s⁻¹, T_H: {T_H:.1e} K")
```

### Tutorial 3: Detection Challenges
```bash
ahr tutorial 3
```

**Why is this so hard?**

```python
# Calculate signal vs noise
import numpy as np

# Typical values
T_H = 1e-8  # Hawking temperature (K)
bandwidth = 1e9  # Detector bandwidth (Hz)
T_sys = 30  # System temperature (K)

# Signal power (very approximate)
P_signal = 1e-23 * bandwidth * T_H  # k_B * Δν * T
P_noise = 1e-23 * bandwidth * T_sys

print(f"Signal power:   {P_signal:.2e} W")
print(f"Noise power:    {P_noise:.2e} W")
print(f"Signal/Noise:   {P_signal/P_noise:.2e}")
print(f"Need to integrate for ~{int(1/(P_signal/P_noise)**2)} seconds")
```

---

## 🔬 Understanding the Code

### Core Algorithm: Finding Horizons

```python
# Simplified horizon finding
import numpy as np

def find_horizons_simple(x, v, cs):
    """
    Find where |v| = c_s (crossing points)
    """
    # Look for sign changes in (|v| - c_s)
    diff = np.abs(v) - cs
    
    horizons = []
    for i in range(len(x) - 1):
        if diff[i] == 0:  # Exact crossing
            horizons.append(x[i])
        elif diff[i] * diff[i+1] < 0:  # Sign change
            # Linear interpolation for better accuracy
            x_cross = x[i] - diff[i] * (x[i+1] - x[i]) / (diff[i+1] - diff[i])
            horizons.append(x_cross)
    
    return np.array(horizons)

# Test it
x = np.linspace(0, 100e-6, 1000)
v = 2e6 * np.tanh((x - 50e-6) / 10e-6)
cs = np.full_like(x, 1e6)

horizons = find_horizons_simple(x, v, cs)
print(f"Found {len(horizons)} horizon(s) at: {horizons}")
```

### Real Code Structure

The actual code (`src/analog_hawking/physics_engine/horizon.py`) is more sophisticated:
- Handles multiple horizons
- Computes uncertainties
- Different κ calculation methods
- Robust root-finding algorithms

But the basic principle is the same: find where |v| = c_s!

---

## 📊 Visualizing Results

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# Load results
import json
with open('results/quickstart/horizons.json', 'r') as f:
    data = json.load(f)

# Create profile
x = np.linspace(0, 100e-6, 1000)
v = 2e6 * np.tanh((x - 50e-6) / 10e-6)
cs = np.full_like(x, 1e6)

# Plot
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(x*1e6, np.abs(v)/1e6, 'b-', linewidth=2, label='|v| (flow speed)')
plt.plot(x*1e6, cs/1e6, 'orange', linewidth=2, label='c_s (sound speed)')
for pos in data['positions_m']:
    plt.axvline(pos*1e6, color='r', linestyle='--', alpha=0.7, label='Horizon' if pos == data['positions_m'][0] else "")
plt.xlabel('Position (μm)')
plt.ylabel('Speed (×10⁶ m/s)')
plt.title('Plasma Flow Profile with Sonic Horizon')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 1, 2)
kappa = data['kappa_s_inv'][0] if data['kappa_s_inv'] else 0
T_H = 1.22e-23 * kappa / (2 * np.pi * 1.38e-23)
plt.text(0.5, 0.5, f'κ = {kappa:.2e} s⁻¹\nT_H = {T_H:.2e} K', 
         ha='center', va='center', fontsize=14,
         bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.axis('off')
plt.title('Horizon Properties')

plt.tight_layout()
plt.savefig('results/quickstart/detailed_analysis.png', dpi=150)
plt.show()
```

---

## 🎓 Building Physical Intuition

### Experiment: Parameter Sensitivity

```bash
# Try different parameters
ahr quickstart --v0 1e6 --out results/slow_flow
ahr quickstart --v0 3e6 --out results/fast_flow
ahr quickstart --L 5e-6 --out results/sharp_horizon
ahr quickstart --L 20e-6 --out results/gentle_horizon
```

**What to look for**:
- Faster flows → more pronounced horizons
- Sharper transitions → higher κ → higher temperature
- Gentler transitions → lower κ → weaker signal

### Experiment: Multiple Horizons

```python
# Create profile with multiple horizon candidates
import numpy as np

x = np.linspace(0, 200e-6, 2000)

# Two tanh profiles combined
v1 = 1.5e6 * np.tanh((x - 50e-6) / 10e-6)
v2 = 1.5e6 * np.tanh((x - 150e-6) / 10e-6)
v = v1 + v2
cs = np.full_like(x, 1e6)

# Save and analyze
np.savez('results/multi_horizon_profile.npz', x=x, v=v, cs=cs)

# Then run analysis
ahr pipeline --demo --profile results/multi_horizon_profile.npz
```

---

## 📖 Understanding the Output

### Key Results Explained

**`horizons.json`**:
```json
{
  "n_horizons": 2,              // Number of horizons found
  "positions_m": [5e-5, 1.5e-4], // Positions in meters
  "kappa_s_inv": [3e12, 2e12],  // Surface gravity [s⁻¹]
  "kappa_err_s_inv": [1e10, 1e10] // Uncertainty [s⁻¹]
}
```

**What it means**:
- **n_horizons**: How many times flow speed crosses sound speed
- **positions_m**: Where the horizons are located
- **kappa_s_inv**: How "steep" each horizon is (higher = better)
- **kappa_err_s_inv**: Numerical uncertainty (should be small)

### Full Pipeline Output

**`full_pipeline_summary.json`**:
```json
{
  "kappa": 3.2e12,           // Surface gravity
  "T_H_K": 4.5e-8,           // Hawking temperature [Kelvin]
  "t5sigma_s": 2.1e-4,       // Detection time [seconds]
  "sanity_violation": false, // Physics bounds OK?
  "hybrid_enhancement": 4.2  // Plasma mirror boost
}
```

**Key metrics**:
- **T_H_K**: Temperature of Hawking radiation (very cold!)
- **t5sigma_s**: How long to detect with 5σ confidence
- **sanity_violation**: Flag if physics bounds exceeded
- **hybrid_enhancement**: Boost from plasma mirror effects

---

## 🚀 Next Steps

### For Students

**Week 1**: Master the basics
- Run all tutorials
- Read [Scientific Narrative](../scientific_narrative.md)
- Understand [Glossary](../Glossary.md)

**Week 2**: Explore parameter space
- Try different laser intensities
- Vary plasma densities
- Plot scaling relationships

**Week 3**: Compare with literature
- Read [AnaBHEL papers](./REFERENCES.md)
- Reproduce published results
- Write summary of findings

### For Researchers

**Reproduce key results**:
```bash
# Gradient catastrophe analysis
make sweep-gradient-full

# Full validation suite
ahr validate --dashboard

# Generate results package
make comprehensive && make results-pack
```

**Extend the framework**:
- Add new κ calculation methods
- Implement different graybody models
- Test with your own plasma profiles

---

## 🧠 Conceptual Questions

### Q: Why can't we just detect this directly?

**A**: The signal is extremely weak—temperatures of 10⁻⁸ K are much colder than the cosmic microwave background (2.7 K). We need:
- Very sensitive radio detectors
- Long integration times
- Careful noise subtraction
- Optimal plasma parameters

### Q: How realistic are these simulations?

**A**: The validated parts (horizon finding, graybody models) are quite robust. However:
- We use 1D approximations (real plasma is 3D)
- We assume idealized flow profiles
- We don't model all plasma instabilities
- Experimental systematics are not included

See [Limitations](../Limitations.md) for details.

### Q: Has anyone detected analog Hawking radiation yet?

**A**: Not definitively. Several experiments have claimed observations:
- Water waves (Weinfurtner et al. 2011)
- Bose-Einstein condensates (Steinhauer 2014, 2016)
- Optical fibers (Drori et al. 2019)

But plasma experiments (our focus) are still in the planning stage.

### Q: What's the connection to real black holes?

**A**: The mathematics is identical, but the physics is different:

| Feature | Real Black Hole | Analog Black Hole |
|---------|----------------|-------------------|
| Horizon | Gravity | Flow velocity |
| Radiation | Quantum fields | Sound waves |
| Temperature | ~10⁻⁸ K for stellar BH | ~10⁻⁸ K for plasma |
| Detection | Impossible | Difficult but possible |

The analog system lets us test quantum field theory in curved spacetime!

---

## 📚 Further Reading

### For Students
- [Scientific Narrative](../scientific_narrative.md) - Story of analog gravity
- [Overview](../Overview.md) - Conceptual introduction
- [Glossary](../Glossary.md) - Key terms
- [FAQ](../FAQ.md) - Common questions

### For Researchers
- [Methods & Algorithms](../Methods.md) - Technical details
- [Validation](../Validation.md) - How we test
- [Gradient Catastrophe](../GradientCatastropheAnalysis.md) - Physics limits
- [References](../REFERENCES.md) - Full bibliography

### Classic Papers
1. **Unruh (1981)** - "Experimental Black-Hole Evaporation?" (original proposal)
2. **Visser (1998)** - "Acoustic Black Holes" (theoretical foundation)
3. **Barceló et al. (2005)** - "Analogue Gravity" (comprehensive review)
4. **Chen & Mourou (2017)** - Plasma mirror concept

---

## 🎯 Project Ideas

### Beginner Projects
1. **Parameter Sensitivity Study**: How does κ change with laser intensity?
2. **Profile Comparison**: Compare tanh vs Gaussian vs linear profiles
3. **Visualization Gallery**: Create beautiful plots for different scenarios
4. **Tutorial Enhancement**: Improve existing tutorials with more examples

### Intermediate Projects
1. **New κ Method**: Implement and test a different κ calculation
2. **Uncertainty Analysis**: Study how numerical errors propagate
3. **Literature Review**: Compare our results with published papers
4. **Educational Material**: Create presentation for your research group

### Advanced Projects
1. **Multi-dimensional Effects**: Extend to 2D or 3D
2. **New Physics**: Add viscosity or magnetic fields
3. **Experimental Collaboration**: Work with laser facility
4. **Publication**: Contribute to research paper

---

## 📞 Getting Help

**Confused about the physics?**
- [Open a Discussion](https://github.com/hmbown/analog-hawking-radiation/discussions)
- Tag: "student-question"

**Code not working?**
- Include error message and what you tried
- Share your Python version and system info (`ahr info`)

**Want to contribute?**
- Start with beginner projects above
- Read [Contributing Guide](../../CONTRIBUTING.md)
- Email: hunter@shannonlabs.dev

---

<div align="center">

**[Back to README](../../README.md)** | **[Quick Links](../QUICKLINKS.md)** | **[Full Documentation](../index.md)**

*Laboratory Black Hole Detection, Quantified*

</div>
