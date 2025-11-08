# Getting Started: Experimentalist

**Quick Navigation**: [Back to README](../../README.md) | [Quick Links](../QUICKLINKS.md) | [Full Documentation](../index.md)

This guide is for **experimental physicists** who want to:
- Design laser-plasma experiments
- Forecast detection feasibility
- Plan beam time allocation
- Compare diagnostic strategies

---

## 🎯 Your Goals & How We Help

| Your Goal | Our Tool | Time to Result |
|-----------|----------|----------------|
| Check if your laser can create horizons | `ahr experiment --eli` | 5 minutes |
| Forecast detection time for your setup | `ahr pipeline --demo` | 2 minutes |
| Explore parameter space | `ahr sweep --gradient` | 15 minutes |
| Validate against your data | `ahr validate` | 1 minute |

---

## ⚡ Quick Start (5 minutes)

### 1. Install & Setup
```bash
git clone https://github.com/hmbown/analog-hawking-radiation.git
cd analog-hawking-radiation
pip install -e .
ahr info  # Check your system
```

### 2. Run ELI Facility Validation
```bash
ahr experiment --eli --output results/my_eli_test
```

This checks if your parameters are within realistic experimental bounds.

### 3. Forecast Detection for Your Parameters
```bash
# Edit configs/laser_profile.yaml with your parameters
ahr pipeline --demo --config configs/laser_profile.yaml --output results/my_forecast
```

### 4. Check Results
```bash
ls results/my_forecast/
cat results/my_forecast/full_pipeline_summary.json
```

---

## 📊 Understanding Your Results

### Key Output Files

**`full_pipeline_summary.json`**:
```json
{
  "kappa": 3.2e12,           // Surface gravity [s⁻¹]
  "T_H_K": 4.5e-8,           // Hawking temperature [K]
  "t5sigma_s": 2.1e-4,       // 5σ detection time [seconds]
  "sanity_violation": false, // Within physics bounds?
  "valid": true              // All checks passed?
}
```

**`horizons.png`**: Shows where |v| = c_s (your sonic horizon)

**`detection_forecast.png`**: Signal strength vs integration time

### What Matters for Experiments

| Parameter | Typical Range | Why It Matters |
|-----------|---------------|----------------|
| **κ (surface gravity)** | 10¹¹ - 10¹³ s⁻¹ | Higher = stronger signal |
| **t5sigma (detection time)** | 10⁻⁷ - 10⁻³ s | Shorter = easier to measure |
| **T_H (Hawking temperature)** | 10⁻⁹ - 10⁻⁶ K | Determines signal frequency |

**Good news**: t5sigma < 10⁻⁶ s is very promising for modern detectors!

---

## 🔧 Configuring Your Experiment

### Laser Parameters (edit `configs/laser_profile.yaml`)

```yaml
laser:
  intensity: 1e22          # W/m² - your laser intensity
  wavelength: 800e-9       # m - typically 800nm
  pulse_duration: 30e-15   # s - femtosecond pulses

plasma:
  density: 1e25            # m⁻³ - electron density
  temperature: 1e6         # K - plasma temperature
  
detection:
  bandwidth: 1e9           # Hz - detector bandwidth
  system_temperature: 30   # K - detector noise temp
```

### Facility-Specific Presets

We provide presets for major facilities:

```bash
# ELI Beamlines
cp configs/facilities/eli_beamlines.yaml configs/my_experiment.yaml

# APOLLON
cp configs/facilities/apollon.yaml configs/my_experiment.yaml

# Edit with your specific parameters
nano configs/my_experiment.yaml
```

---

## 📈 Planning Your Beam Time

### Step 1: Parameter Scan (1 hour)
```bash
# Sweep key parameters
ahr sweep --gradient --n-samples 100 --output results/parameter_scan

# Check which parameters give best detection times
grep -r "t5sigma_s" results/parameter_scan/ | sort -k2 -n
```

### Step 2: Optimize for Your Constraints
```bash
# Find parameters that minimize detection time
# while staying within your laser capabilities
python scripts/optimize_for_detection.py \
  --max-intensity 1e22 \
  --max-density 5e25 \
  --target-time 1e-6
```

### Step 3: Generate Proposal Figures
```bash
# Create publication-ready plots
python scripts/generate_proposal_figures.py \
  --config configs/my_experiment.yaml \
  --output figures/proposal/
```

---

## 🏭 Facility-Specific Guides

### ELI Beamlines
```bash
# Validate against ELI parameters
ahr experiment --eli --output results/eli_validation

# Check ELI-specific constraints
python scripts/validate_eli_constraints.py \
  --intensity 1e22 \
  --rep-rate 10
```

See: [ELI Experimental Planning Guide](../ELI_Experimental_Planning_Guide.md)

### Other Facilities
- [APOLLON Configuration](../configs/facilities/apollon.yaml)
- [GEMINI Configuration](../configs/facilities/gemini.yaml)
- [Custom Facility Template](../configs/facilities/template.yaml)

---

## 🎯 Realistic Expectations

### What This Code **Can** Do
✅ Predict horizon formation for given laser/plasma parameters  
✅ Estimate detection times with realistic detector models  
✅ Identify optimal parameter regions for experiments  
✅ Validate against known physics constraints  

### What This Code **Cannot** Do
❌ Replace full PIC simulations for detailed plasma dynamics  
❌ Account for all experimental systematic uncertainties  
❌ Guarantee detection (gives statistical estimates)  
❌ Model multi-dimensional effects (currently 1D)  

See: [Limitations & Assumptions](../Limitations.md)

---

## 📋 Pre-Experiment Checklist

Before your beam time, run:

```bash
# 1. Validate physics
ahr validate --dashboard

# 2. Check facility constraints  
ahr experiment --eli

# 3. Forecast detection
ahr pipeline --demo --config my_experiment.yaml

# 4. Generate analysis scripts
python scripts/generate_analysis_plan.py \
  --config my_experiment.yaml \
  --output analysis_plan.py

# 5. Package everything
make results-pack
```

---

## 🔍 Interpreting Results for Proposals

### Strong Signal Indicators
- κ > 10¹² s⁻¹
- t5sigma < 10⁻⁵ s
- Clear horizon in velocity/density profile
- No "sanity_violation" flags

### Red Flags
- κ < 10¹¹ s⁻¹ (very weak signal)
- t5sigma > 10⁻³ s (challenging detection)
- "sanity_violation: true" (physics bounds exceeded)
- Flat coupling_strength scaling (profile issues)

### Proposal Language Template
```
"Based on analog Hawking radiation simulations with 
validated horizon detection and graybody modeling, we 
forecast a 5σ detection time of ~10⁻⁴ seconds for our 
proposed parameters (κ ≈ 3×10¹² s⁻¹, T_H ≈ 10⁻⁷ K), 
well within our allocated beam time."
```

---

## 📞 Getting Help

**Questions about your specific experiment?**
- [Open a GitHub Discussion](https://github.com/hmbown/analog-hawking-radiation/discussions)
- Tag it with "experimental-planning"
- Include your target parameters

**Found a bug or physics issue?**
- [Open a GitHub Issue](https://github.com/hmbown/analog-hawking-radiation/issues)
- Include your config file and output

**Need private consultation?**
- Email: hunter@shannonlabs.dev
- Subject: "Analog Hawking - Experiment Planning"

---

## 🎓 Learning More

**Core Physics**:
- [Analog Gravity Overview](../Overview.md)
- [Methods & Algorithms](../Methods.md)
- [Gradient Catastrophe Analysis](../GradientCatastropheAnalysis.md)

**Experimental Context**:
- [ELI Experimental Planning Guide](../ELI_Experimental_Planning_Guide.md)
- [AnaBHEL Comparison](../AnaBHEL_Comparison.md)
- [Facility Integration](../facilities/)

---

<div align="center">

**[Back to README](../../README.md)** | **[Quick Links](../QUICKLINKS.md)** | **[Full Documentation](../index.md)**

*Laboratory Black Hole Detection, Quantified*

</div>
